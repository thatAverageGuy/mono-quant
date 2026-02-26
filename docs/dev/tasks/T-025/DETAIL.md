# T-025: GGUF Validation — gguf-py + llama.cpp Manual Procedure

## Status
TODO

## Phase
07-04 — Phase 7: GGUF Binary Format Export

## Requirements
- Automated validation: `validate_gguf_checkpoint(path)` using gguf-py
- Manual validation procedure documented (llama.cpp cannot run in CI)
- `gguf` added to optional deps: `pip install mono-quant[gguf]`

## Decisions
- **gguf-py only** for automated validation — no llama.cpp binary in CI
- **No `--validate-runtime` CLI flag** — keep CLI surface minimal; llama.cpp procedure
  is in IMPL_LOG.md instead
- **validate_gguf_checkpoint** lives in `src/mono_quant/export/common/validators.py`
  (alongside existing `validate_onnx_model` and `validate_gptq_checkpoint_structure`)

## Validation Levels

### Level 1 — Automated (gguf-py)
```python
import gguf
reader = gguf.GGUFReader(str(path))
# Checks: magic, version, KV types, tensor count, alignment
# Raises on any structural error
```

### Level 2 — Manual (llama.cpp)
```bash
# Build llama.cpp or use a release binary
./llama-cli -m /path/to/model.gguf -p "Hello, world" -n 20 --no-mmap
```
Expected: no errors, output contains recognizable words (not random characters).

Acceptance criteria for manual test:
1. No "invalid quantization format" or "unsupported tensor type" errors
2. Output text is coherent English (or appropriate language for the model)
3. Generation completes without segfault

### Level 3 — Accuracy (optional, not required for T-025 completion)
Perplexity comparison vs FP32 baseline. Q4_K typically within 0.2-0.5 PPL of FP32.

## API: validate_gguf_checkpoint

**File**: `src/mono_quant/export/common/validators.py` — add:

```python
def validate_gguf_checkpoint(path: Union[str, Path]) -> None:
    """Validate a GGUF checkpoint using gguf-py (Level 1 automated check).

    Checks:
    - File exists and has .gguf extension
    - gguf-py can open and parse the file without error
    - File contains at least one tensor
    - general.architecture KV key is present

    Args:
        path: Path to the .gguf file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file fails structural validation.
        ImportError: If gguf is not installed.
    """
```

Implementation sketch:
```python
def validate_gguf_checkpoint(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {path}")

    try:
        import gguf
    except ImportError:
        raise ImportError(
            "GGUF validation requires gguf. Install with: pip install mono-quant[gguf]"
        )

    try:
        reader = gguf.GGUFReader(str(path))
    except Exception as e:
        raise ValueError(f"gguf-py failed to parse {path}: {e}") from e

    if len(reader.tensors) == 0:
        raise ValueError(f"GGUF file contains no tensors: {path}")

    kv_keys = {kv.name for kv in reader.fields.values()}
    if "general.architecture" not in kv_keys:
        raise ValueError(
            f"GGUF file missing required KV key 'general.architecture': {path}"
        )
```

## Optional Dependency

**File**: `pyproject.toml` — add to `[project.optional-dependencies]`:
```toml
gguf = ["gguf>=0.1"]
```

gguf-py is the Python package published by the llama.cpp project.
Install: `pip install gguf`

## State Machine: validate_gguf_checkpoint()

```
  [path provided]
        │
        ▼
  [file exists?]──no──→ [FileNotFoundError]
        │
        ▼
  [import gguf]──fail──→ [ImportError with install hint]
        │
        ▼
  [GGUFReader(path)]──fail──→ [ValueError: parse error]
        │
        ▼
  [n_tensors > 0?]──no──→ [ValueError: no tensors]
        │
        ▼
  [general.architecture present?]──no──→ [ValueError: missing KV]
        │
        ▼
  [return None (valid)]
```

## Manual llama.cpp Test Procedure

Document in IMPL_LOG.md for T-025 as a step-by-step manual test:

```markdown
### Manual llama.cpp Validation Procedure

Prerequisites:
- llama.cpp built or release binary available
- A small model exported to GGUF (e.g., TinyLlama or GPT-2 small)

Steps:
1. Export model: `monoquant export-gguf ./model ./output --config ./model/config.json`
2. Verify file exists: `ls -lh ./output/model.gguf`
3. Validate structure: `python -c "from mono_quant.export.common.validators import validate_gguf_checkpoint; validate_gguf_checkpoint('./output/model.gguf')"`
4. Load in llama.cpp: `./llama-cli -m ./output/model.gguf -p "The sky is" -n 10 --no-mmap`
5. Check: output contains coherent continuation (not random bytes)
6. Check: no error lines containing "invalid" or "unsupported"

Expected output (example):
  The sky is blue and the sun is shining

Failure modes and diagnosis:
- "unsupported tensor type": GGML_TYPE code in tensor info is wrong → check T-023 GGML_TYPE_Q4_K = 12
- "invalid quantization format": scale/block data malformed → recheck scale packing
- Garbage output (random chars): dequantization formula mismatch → verify d/dmin/ls/lm encoding
- Segfault: alignment or offset error → verify 32-byte alignment in T-022 writer
```

## Success Criteria
- [ ] `validate_gguf_checkpoint` passes on a file produced by GGUFExporter
- [ ] `validate_gguf_checkpoint` raises FileNotFoundError on missing file
- [ ] `validate_gguf_checkpoint` raises ImportError with install hint when gguf not installed
- [ ] `validate_gguf_checkpoint` raises ValueError on empty tensor list
- [ ] `gguf` listed in `pyproject.toml` optional deps
- [ ] Manual llama.cpp procedure documented in IMPL_LOG.md

## Dependencies
- T-023 (quantization types complete)
- T-024 (tensor naming + GGUFExporter complete — validator tests need a valid file to open)

## Testing Requirements
- `test_validate_gguf_checkpoint_valid` — runs on a file produced by export_to_gguf
- `test_validate_gguf_checkpoint_missing_file` — FileNotFoundError
- `test_validate_gguf_checkpoint_import_error` — mock missing gguf module → ImportError
  (use `unittest.mock.patch('builtins.__import__')` to simulate missing gguf)
Coverage target: all branches in validate_gguf_checkpoint

## Open Questions
None.
