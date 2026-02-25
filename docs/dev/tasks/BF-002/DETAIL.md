# BF-002: Fix dual QuantizationInfo classes — result.save() always crashes

## Status
TODO

## Audit Reference
C3 (Critical)

## Problem
Two unrelated classes share the name `QuantizationInfo`:

1. `mono_quant.core.quantizers.QuantizationInfo` — the primary class used throughout
   the quantization pipeline. Fields: `selected_layers`, `skipped_layers`,
   `calibration_samples_used`, `dtype`, `symmetric`, `sqnr_db`, `compression_ratio`,
   `warnings`.

2. `mono_quant.io.formats.QuantizationInfo` — a local metadata class. Fields:
   `dtype`, `symmetric`, `per_channel`, `selected_layers`, `scheme`, `group_size`, `bits`.

`QuantizationResult.save()` passes `self.info` (type = `core.quantizers.QuantizationInfo`)
to `save_model(..., quantization_info=self.info)`. Then `_build_metadata` in `io/formats.py`
accesses `quantization_info.per_channel` and `quantization_info.bits` — both absent on
`core.quantizers.QuantizationInfo`. **Every call to `result.save()` raises AttributeError.**

## Requirements
1. Eliminate the name collision.
2. `result.save()` must complete without error.
3. Metadata written to safetensors files must be accurate (no missing fields).
4. No breaking change to the public API (`result.save("path.safetensors")` keeps working).

## Decisions
- **Decision:** Rename `io.formats.QuantizationInfo` to `SaveMetadata` or similar
  (it's only used internally in `_build_metadata`).
  Reason: The `core.quantizers.QuantizationInfo` is the canonical public type; the formats
  one is a private local data bag with no public exposure.
  Alternatives: Merge the classes (risky — different lifecycles); add missing fields to
  core class (adds io concerns to core, wrong layer).

- **Decision:** Update `_build_metadata` to accept `core.quantizers.QuantizationInfo`
  directly and map its fields to metadata strings, deriving `per_channel`, `bits`, `scheme`
  from the existing fields.
  Reason: `per_channel` is not tracked on core's QuantizationInfo at all — add it, or
  derive from dtype/context. `bits` can be derived from `dtype` (qint8 → 8, float16 → 16).
  `scheme` can be derived from `symmetric` bool.

## Success Criteria
- [ ] `result.save("model.safetensors")` completes without error
- [ ] Safetensors metadata contains `scheme`, `per_channel`, `bits`, `selected_layers`,
      `calibration_samples`, `compression_ratio`, `sqnr_db` where available
- [ ] No import of `io.formats.QuantizationInfo` anywhere outside `io/formats.py`
- [ ] All existing tests still pass

## Dependencies
- None (self-contained)

## Implementation Guidance

### Step 1: Rename `io/formats.py` internal class
In `src/mono_quant/io/formats.py`:
- Rename `QuantizationInfo` → `_SaveMetadata` (or similar private name)
- Update all usages within `formats.py` only (it's not imported elsewhere)

### Step 2: Update `_build_metadata` signature
Change `_build_metadata` to accept `core.quantizers.QuantizationInfo` directly:
```python
# io/formats.py
from mono_quant.core.quantizers import QuantizationInfo as CoreQuantizationInfo

def _build_metadata(
    quantization_info: Optional[CoreQuantizationInfo] = None,
    ...
) -> Dict[str, str]:
    if quantization_info is not None:
        # derive bits from dtype
        bits_map = {torch.qint8: 8, torch.float16: 16}
        bits = bits_map.get(quantization_info.dtype, 8)
        metadata["bits"] = str(bits)
        metadata["scheme"] = "symmetric" if quantization_info.symmetric else "asymmetric"
        metadata["per_channel"] = "true"  # always per-channel in current impl
        metadata["selected_layers"] = json.dumps(quantization_info.selected_layers)
        if quantization_info.calibration_samples_used:
            metadata["calibration_samples"] = str(quantization_info.calibration_samples_used)
        if quantization_info.sqnr_db is not None:
            metadata["sqnr_db"] = str(quantization_info.sqnr_db)
        if quantization_info.compression_ratio is not None:
            metadata["compression_ratio"] = str(quantization_info.compression_ratio)
```

### Step 3: Update `save_model` signature type annotation
Change `quantization_info: Optional[QuantizationInfo]` → `Optional[CoreQuantizationInfo]`
in `save_model()`.

### Step 4: Update `io/__init__.py`
Remove export of `QuantizationInfo` from `io/formats.py` (or rename in `__all__`).
The canonical `QuantizationInfo` lives in `core.quantizers` and is already exported
from the top-level `__init__.py` is not needed.

## Testing Requirements
- Unit: Test `_build_metadata` with a `core.quantizers.QuantizationInfo` instance;
  verify all expected keys present in output dict.
- Integration: Full round-trip `quantize → result.save("x.safetensors")` with both
  INT8 dynamic and INT8 static; verify file written and metadata readable.
- Coverage target: 100% of `_build_metadata` branches

## Open Questions
<!-- MUST be empty before implementation begins -->
