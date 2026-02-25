# Implementation Log: BF-002

## Summary
Fixed the dual `QuantizationInfo` class collision that caused `result.save()`
to always crash with `AttributeError`. Renamed the private `io/formats.py`
class to `_SaveMetadata` and updated `_build_metadata` to accept and correctly
handle `core.quantizers.QuantizationInfo`.

## What Was Done
- Renamed `io.formats.QuantizationInfo` → `_SaveMetadata` (private, unused externally)
- Updated `_build_metadata` to accept `Optional[Any]` and handle core's
  `QuantizationInfo` — derives `per_channel`, `bits`, and `scheme` from existing fields
- Updated `save_model` type annotation from `Optional[QuantizationInfo]` → `Optional[Any]`
- Removed `QuantizationInfo` from `io/formats.py` `__all__` and `io/__init__.py` exports
- Updated `save_model` docstring example to reference `core.quantizers.QuantizationInfo`
- Added tests verifying `_build_metadata` works with CoreQuantizationInfo and that
  `result.save("x.safetensors")` completes without error

## How It Was Done
LEVEL 2 local change confined to `io/formats.py` and `io/__init__.py`. A lazy
local import (`from mono_quant.core.quantizers import QuantizationInfo as _CoreInfo`)
inside `_build_metadata` avoids any circular import risk. Fields not present on
`CoreQuantizationInfo` are derived:
- `per_channel`: hardcoded `"true"` (always per-channel in current implementation)
- `bits`: derived from `dtype` via `{qint8: 8, quint8: 8, float16: 16}`
- `scheme`: derived from `symmetric` bool
- `sqnr_db`, `compression_ratio`: read from `CoreQuantizationInfo` if set (via `hasattr`)

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/io/formats.py | Modified | Renamed class, updated `_build_metadata`, updated `save_model` annotation + docstring, updated `__all__` |
| src/mono_quant/io/__init__.py | Modified | Removed `QuantizationInfo` import and export |
| tests/test_bugfixes.py | Modified | `test_build_metadata_with_core_quantization_info`, `test_result_save_completes_without_error` |

## Why These Choices Were Made
- `_SaveMetadata` kept (not deleted) as a private class in case of legacy usage;
  it's not exported so users won't encounter it
- `Optional[Any]` annotation preferred over `Optional["CoreQuantizationInfo"]`
  with `TYPE_CHECKING` to keep the change minimal
- `hasattr` guards on `sqnr_db`/`compression_ratio` make the function flexible for
  both CoreQuantizationInfo and any future type that might be passed

## Testing Results
- Unit: `test_build_metadata_with_core_quantization_info` — PASS
- Integration: `test_result_save_completes_without_error` — PASS (file written, >0 bytes)
- All 15 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
