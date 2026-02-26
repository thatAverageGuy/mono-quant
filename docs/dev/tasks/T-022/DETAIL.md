# T-022: GGUF Binary Format Writer — Header + KV Metadata

## Status
TODO

## Phase
07-01 — Phase 7: GGUF Binary Format Export

## Requirements
- Write valid GGUF v3 binary files
- Sections: header, KV metadata, tensor info, 32-byte-aligned tensor data
- Pure Python stdlib (`struct`); no gguf-py required for writing (gguf-py is for reading/validation only)

## Decisions
- **GGUF version 3** — current standard; all modern llama.cpp releases (1.0+) support v3
- **Little-endian** throughout (all fields)
- **GGUFWriter accumulates** KV + tensor entries in memory; serializes once on `write()`
- **Atomically writes**: write to `.tmp` file, then rename to final path

## GGUF v3 Binary Layout

```
Offset  Size   Field
──────  ─────  ──────────────────────────────────────────────
0       4      magic = b"GGUF"  (0x47 0x47 0x55 0x46)
4       4      version = 3  (uint32 LE)
8       8      n_tensors  (uint64 LE)
16      8      n_kv  (uint64 LE)
24      var    KV metadata section  (n_kv entries, variable length)
var     var    Tensor info section  (n_tensors entries, variable length)
var     -      Padding: zero bytes to reach next 32-byte boundary from file start
var     var    Tensor data section  (n_tensors blobs, each 32-byte aligned)
```

### KV Entry Wire Format
```
uint64    key_len
bytes     key_data  (UTF-8, no null terminator)
uint32    value_type  (GGUF_TYPE_* below)
varies    value
```

### GGUF_TYPE Enum (used in value_type field)
```
UINT8=0  INT8=1  UINT16=2  INT16=3  UINT32=4  INT32=5
FLOAT32=6  BOOL=7  STRING=8  ARRAY=9  UINT64=10  INT64=11  FLOAT64=12
```

### Value Encoding by Type
| Type    | Wire format                                              |
|---------|----------------------------------------------------------|
| UINT32  | 4 bytes LE                                               |
| INT32   | 4 bytes LE (signed)                                      |
| UINT64  | 8 bytes LE                                               |
| FLOAT32 | 4 bytes LE (IEEE 754)                                    |
| BOOL    | 1 byte (0x01 = true, 0x00 = false)                      |
| STRING  | uint64 length + UTF-8 bytes (no null terminator)         |
| ARRAY   | uint32 elem_type + uint64 count + count × elem encoding  |

### Tensor Info Entry Wire Format
```
uint64    name_len
bytes     name_data  (UTF-8, no null terminator)
uint32    n_dims
uint64[]  dims  (n_dims values; INNERMOST FIRST — reversed from PyTorch order)
uint32    ggml_type  (GGML_TYPE_* defined in T-023)
uint64    offset  (byte offset from start of tensor DATA section, not file start)
```

**Dimension ordering**: For a PyTorch tensor with shape `(out, in)`, store dims as `[in, out]`.
This matches GGUF convention (innermost/contiguous dimension first).

### Tensor Data Alignment
- Start of tensor data section: first 32-byte boundary after tensor info section ends
- Each tensor's data is padded with zero bytes to the next 32-byte boundary
- Offsets in tensor info refer to positions within the data section (not from file start)

## API: GGUFWriter

**File**: `src/mono_quant/export/gguf/writer.py`

```python
class GGUFWriter:
    def __init__(self) -> None
    def add_string(self, key: str, value: str) -> None
    def add_uint32(self, key: str, value: int) -> None
    def add_int32(self, key: str, value: int) -> None
    def add_uint64(self, key: str, value: int) -> None
    def add_float32(self, key: str, value: float) -> None
    def add_bool(self, key: str, value: bool) -> None
    def add_array_string(self, key: str, values: List[str]) -> None
    def add_tensor(self, name: str, data: bytes,
                   shape: Tuple[int, ...], ggml_type: int) -> None
    def write(self, path: Union[str, Path]) -> None
    # private
    def _encode_str(self, s: str) -> bytes        # uint64 len + UTF-8
    def _make_kv(self, key, type_id, val_bytes) -> bytes
    def _pad32(self, data: bytes) -> bytes         # pad to 32-byte multiple
```

Internal state:
- `_kvs: List[bytes]` — pre-serialized KV blobs (appended by add_* methods)
- `_tensors: List[Tuple[str, bytes, Tuple[int, ...], int]]` — (name, data, shape, ggml_type)

## State Machine: write() flow

```
      [add_* calls accumulate state]
                   │
                   ▼
           [write(path) called]
                   │
                   ▼
       [serialize header bytes]        magic + version + n_tensors + n_kv
                   │
                   ▼
       [serialize KV section]          join all _kvs blobs
                   │
                   ▼
       [compute tensor offsets]        cumulative sum, each tensor 32-byte aligned
                   │
                   ▼
      [serialize tensor info section]  with computed offsets
                   │
                   ▼
       [pad header+kv+info to 32]      zero bytes to 32-byte boundary
                   │
                   ▼
       [write tensor data section]     each blob padded to 32 bytes
                   │
                   ▼
        [write .tmp → rename final]    atomic write
                   │
                   ▼
                [DONE]
```

## Implementation Guidance

### Step 1: Create package structure
```
src/mono_quant/export/gguf/__init__.py   (empty or minimal re-exports)
src/mono_quant/export/gguf/writer.py     (GGUFWriter)
```

### Step 2: struct format strings (all little-endian)
```python
import struct
pack_u8  = lambda v: struct.pack('<B', v)
pack_u32 = lambda v: struct.pack('<I', v)
pack_i32 = lambda v: struct.pack('<i', v)
pack_u64 = lambda v: struct.pack('<Q', v)
pack_f32 = lambda v: struct.pack('<f', v)
```

### Step 3: _encode_str helper
```python
def _encode_str(self, s: str) -> bytes:
    b = s.encode('utf-8')
    return struct.pack('<Q', len(b)) + b
```

### Step 4: _make_kv helper
```python
def _make_kv(self, key: str, type_id: int, val: bytes) -> bytes:
    return self._encode_str(key) + struct.pack('<I', type_id) + val
```

Then each add_* method calls `self._kvs.append(self._make_kv(key, TYPE_ID, val_bytes))`.

### Step 5: add_tensor
```python
def add_tensor(self, name, data, shape, ggml_type):
    self._tensors.append((name, bytes(data), tuple(shape), ggml_type))
```

### Step 6: write() implementation
```python
def write(self, path):
    # 1. Header
    header = b"GGUF" + struct.pack('<I', 3)        # magic + version
    header += struct.pack('<QQ', len(self._tensors), len(self._kvs))

    # 2. KV section
    kv_section = b''.join(self._kvs)

    # 3. Compute tensor offsets
    offsets = []
    current = 0
    for _, data, _, _ in self._tensors:
        offsets.append(current)
        size = len(data)
        current += size + ((-size) % 32)  # pad to 32

    # 4. Tensor info section
    info_section = b''
    for (name, data, shape, ggml_type), offset in zip(self._tensors, offsets):
        info_section += self._encode_str(name)
        info_section += struct.pack('<I', len(shape))
        info_section += struct.pack(f'<{len(shape)}Q', *reversed(shape))
        info_section += struct.pack('<IQ', ggml_type, offset)

    # 5. Pad entire pre-data section to 32-byte boundary
    pre_data = header + kv_section + info_section
    pad_len = (32 - len(pre_data) % 32) % 32
    pre_data += b'\x00' * pad_len

    # 6. Tensor data section (each padded to 32 bytes)
    data_section = b''
    for _, data, _, _ in self._tensors:
        data_section += data
        pad = (32 - len(data) % 32) % 32
        data_section += b'\x00' * pad

    # 7. Atomic write
    import os
    path = Path(path)
    tmp = path.with_suffix('.gguf.tmp')
    tmp.write_bytes(pre_data + data_section)
    tmp.rename(path)
```

### Step 7: Tests
File: `tests/test_gguf_export.py` — add the T-022 tests described below.

Install gguf-py for validation: `pip install gguf>=0.1` (dev only).

## Success Criteria
- [ ] GGUF file magic bytes are correct: `f.read(4) == b"GGUF"`
- [ ] GGUF version field reads as 3
- [ ] KV entries read back via `gguf.GGUFReader` with correct types and values
- [ ] Tensor data starts on a 32-byte boundary from file start
- [ ] Tensor offsets within data section are correct (each tensor 32-byte aligned)
- [ ] Writer handles n_tensors=0 (metadata-only file) without error

## Testing Requirements
- `test_gguf_writer_magic_and_version` — write empty file, check bytes 0-7
- `test_gguf_writer_kv_string` — add string KV, read with gguf-py, verify value
- `test_gguf_writer_kv_uint32` — add uint32 KV, verify value and type
- `test_gguf_writer_kv_float32` — add float32 KV, verify
- `test_gguf_writer_tensor_alignment` — write 2 tensors, verify data section 32-byte aligned
- `test_gguf_writer_tensor_offsets` — verify each tensor offset matches actual position in file
Coverage target: all branches in write(); all add_* methods called at least once

## Open Questions
None.
