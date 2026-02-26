"""GGUFWriter — writes GGUF v3 binary files.

GGUF (GGML Universal File) is the binary format used by llama.cpp.

File layout:
    Header   (24 bytes): magic + version + n_tensors + n_kv
    KV metadata section: typed key-value pairs
    Tensor info section: name, shape, ggml_type, offset per tensor
    Padding:             zero bytes to the next 32-byte boundary from file start
    Tensor data section: raw bytes, each tensor padded to 32-byte boundary

All integer fields are little-endian.

References:
    https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
    ggml/src/gguf.cpp (authoritative implementation)
"""

import struct
from pathlib import Path
from typing import List, Tuple, Union


# ---------------------------------------------------------------------------
# GGUF metadata value type enum
# ---------------------------------------------------------------------------

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

_GGUF_MAGIC   = b"GGUF"
_GGUF_VERSION = 3
_ALIGNMENT    = 32


class GGUFWriter:
    """Accumulates KV metadata and tensor data, then serializes to a GGUF v3 file.

    Usage::

        writer = GGUFWriter()
        writer.add_string("general.architecture", "llama")
        writer.add_uint32("general.quantization_version", 2)
        writer.add_tensor("token_embd.weight", data_bytes, (4096, 32000), GGML_TYPE_Q4_K)
        writer.write(Path("model.gguf"))
    """

    def __init__(self) -> None:
        self._kvs: List[bytes] = []
        self._tensors: List[Tuple[str, bytes, Tuple[int, ...], int]] = []

    # ------------------------------------------------------------------
    # Public KV add methods
    # ------------------------------------------------------------------

    def add_string(self, key: str, value: str) -> None:
        """Add a STRING-typed KV entry."""
        self._kvs.append(self._make_kv(key, GGUF_TYPE_STRING, self._enc_str(value)))

    def add_uint32(self, key: str, value: int) -> None:
        """Add a UINT32-typed KV entry."""
        self._kvs.append(self._make_kv(key, GGUF_TYPE_UINT32, struct.pack("<I", value)))

    def add_int32(self, key: str, value: int) -> None:
        """Add an INT32-typed KV entry."""
        self._kvs.append(self._make_kv(key, GGUF_TYPE_INT32, struct.pack("<i", value)))

    def add_uint64(self, key: str, value: int) -> None:
        """Add a UINT64-typed KV entry."""
        self._kvs.append(self._make_kv(key, GGUF_TYPE_UINT64, struct.pack("<Q", value)))

    def add_float32(self, key: str, value: float) -> None:
        """Add a FLOAT32-typed KV entry."""
        self._kvs.append(self._make_kv(key, GGUF_TYPE_FLOAT32, struct.pack("<f", value)))

    def add_bool(self, key: str, value: bool) -> None:
        """Add a BOOL-typed KV entry (stored as 1 byte: 0x01 or 0x00)."""
        self._kvs.append(self._make_kv(key, GGUF_TYPE_BOOL, struct.pack("<B", 1 if value else 0)))

    def add_array_string(self, key: str, values: List[str]) -> None:
        """Add an ARRAY of STRING KV entry."""
        elem_type = struct.pack("<I", GGUF_TYPE_STRING)
        count     = struct.pack("<Q", len(values))
        elems     = b"".join(self._enc_str(v) for v in values)
        self._kvs.append(self._make_kv(key, GGUF_TYPE_ARRAY, elem_type + count + elems))

    # ------------------------------------------------------------------
    # Public tensor add method
    # ------------------------------------------------------------------

    def add_tensor(
        self,
        name: str,
        data: bytes,
        shape: Tuple[int, ...],
        ggml_type: int,
    ) -> None:
        """Register a tensor for serialization.

        Args:
            name:      GGUF tensor name (e.g. "blk.0.attn_q.weight").
            data:      Raw tensor bytes in the block format for ggml_type.
            shape:     Tensor shape as a Python tuple in PyTorch order
                       (e.g. (out_features, in_features)).
                       Will be reversed to GGUF innermost-first order on write.
            ggml_type: GGML tensor type code (e.g. GGML_TYPE_Q4_K = 12).
        """
        self._tensors.append((name, bytes(data), tuple(shape), ggml_type))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def write(self, path: Union[str, Path]) -> None:
        """Serialize accumulated state to a GGUF v3 file.

        Writes atomically: serializes to a .tmp file first, then renames.

        Args:
            path: Destination file path (e.g. Path("model.gguf")).
        """
        path = Path(path)

        # 1. Header
        header  = _GGUF_MAGIC
        header += struct.pack("<I", _GGUF_VERSION)
        header += struct.pack("<Q", len(self._tensors))
        header += struct.pack("<Q", len(self._kvs))

        # 2. KV section
        kv_section = b"".join(self._kvs)

        # 3. Compute tensor data offsets (each blob 32-byte aligned)
        offsets: List[int] = []
        cursor = 0
        for _, data, _, _ in self._tensors:
            offsets.append(cursor)
            size    = len(data)
            cursor += size + ((-size) % _ALIGNMENT)

        # 4. Tensor info section
        info_section = b""
        for (name, data, shape, ggml_type), offset in zip(self._tensors, offsets):
            info_section += self._enc_str(name)
            info_section += struct.pack("<I", len(shape))
            # GGUF stores dimensions innermost-first (reversed from PyTorch order)
            info_section += struct.pack(f"<{len(shape)}Q", *reversed(shape))
            info_section += struct.pack("<I", ggml_type)
            info_section += struct.pack("<Q", offset)

        # 5. Pad pre-data portion to 32-byte boundary
        pre_data = header + kv_section + info_section
        pad_len  = ((-len(pre_data)) % _ALIGNMENT)
        pre_data += b"\x00" * pad_len

        # 6. Tensor data section (each blob followed by alignment padding)
        data_section = b""
        for _, data, _, _ in self._tensors:
            data_section += data
            data_section += b"\x00" * ((-len(data)) % _ALIGNMENT)

        # 7. Atomic write
        tmp = path.with_suffix(".gguf.tmp")
        tmp.write_bytes(pre_data + data_section)
        tmp.rename(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _enc_str(self, s: str) -> bytes:
        """Encode a Python string as GGUF string: uint64 length + UTF-8 bytes."""
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def _make_kv(self, key: str, type_id: int, value_bytes: bytes) -> bytes:
        """Assemble one serialized KV entry."""
        return self._enc_str(key) + struct.pack("<I", type_id) + value_bytes
