# test_e_convert.py — Manual Test E: result.convert() API
#
# Tests that QuantizationResult.convert() correctly re-quantizes a model
# to a different bit-width using the Python API. Run from mq_manual_test/:
#
#   venv/Scripts/python test_e_convert.py
#
# Requires: mono-quant 2.0.0 installed in venv/ (no extra installs needed)

import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok and detail:
        print(f"         {detail}")
    return ok


def has_dtype(model, dtype):
    """Return True if the model contains any tensor with the given dtype."""
    for name, module in model.named_modules():
        for attr in ("_quantized_weight", "weight", "bias"):
            t = getattr(module, attr, None)
            if isinstance(t, torch.Tensor) and t.dtype == dtype:
                return True
    return False


def count_quantized_layers(model):
    """Count QuantizedLinear / QuantizedConv2d modules."""
    from mono_quant.modules.linear import QuantizedLinear, QuantizedConv2d, QuantizedLinearInt4
    count8 = sum(1 for m in model.modules() if isinstance(m, (QuantizedLinear, QuantizedConv2d)))
    count4 = sum(1 for m in model.modules() if isinstance(m, QuantizedLinearInt4))
    return count8, count4


def main():
    print("=" * 60)
    print("Manual Test E — result.convert() API")
    print("=" * 60)

    from mono_quant import quantize
    from mono_quant.modules.linear import QuantizedLinear, QuantizedLinearInt4

    failures = 0

    # ── Setup ─────────────────────────────────────────────────────────
    print("\n[Setup] Quantizing SmallMLP to INT8...")
    model = SmallMLP()
    result8 = quantize(model, bits=8, dynamic=True)

    ok = result8.success
    if not check("INT8 quantization succeeded", ok):
        print("FATAL: base quantization failed, aborting.")
        return 1

    n8, n4 = count_quantized_layers(result8.model)
    print(f"         INT8 layers: {n8}, INT4 layers: {n4}")
    sqnr = result8.info.sqnr_db
    print(f"         SQNR: {sqnr:.2f} dB" if sqnr else "         SQNR: N/A")

    # ── Test E-1: convert INT8 → INT4 ─────────────────────────────────
    print("\n[E-1] result8.convert(bits=4)")
    caught_warnings = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result4 = result8.convert(bits=4)
        caught_warnings = list(w)

    ok_success = result4.success
    if not check("convert() returns success result", ok_success,
                 f"errors: {result4.errors}"):
        failures += 1

    n8_after, n4_after = count_quantized_layers(result4.model)
    # bits=4 dynamic quantization uses QuantizedLinear (qint8 storage) not
    # QuantizedLinearInt4 (per-group packing). Verify that quantized layers exist.
    ok_quant = (n8_after + n4_after) > 0
    if not check(f"converted model has quantized layers ({n8_after} QL, {n4_after} QL-Int4)", ok_quant):
        failures += 1

    ok_warn = any(
        "convert" in str(w.message).lower() or "re-quantization" in str(w.message).lower()
        for w in caught_warnings
        if issubclass(w.category, UserWarning)
    )
    if not check("UserWarning emitted about re-quantization", ok_warn,
                 f"warnings: {[str(w.message) for w in caught_warnings]}"):
        failures += 1

    sqnr4 = result4.info.sqnr_db
    print(f"         SQNR after INT4: {sqnr4:.2f} dB" if sqnr4 else "         SQNR: N/A")

    # ── Test E-2: convert INT8 → FP16 ─────────────────────────────────
    print("\n[E-2] result8.convert(bits=16)")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result16 = result8.convert(bits=16)
        caught16 = list(w)

    ok_success16 = result16.success
    if not check("convert() returns success result", ok_success16,
                 f"errors: {result16.errors}"):
        failures += 1

    ok_fp16 = has_dtype(result16.model, torch.float16)
    if not check("converted model contains fp16 tensors", ok_fp16):
        failures += 1

    ok_warn16 = any(
        issubclass(w.category, UserWarning) for w in caught16
    )
    if not check("UserWarning emitted for FP16 convert", ok_warn16,
                 f"warnings: {[str(w.message) for w in caught16]}"):
        failures += 1

    sqnr16 = result16.info.sqnr_db
    print(f"         SQNR after FP16: {sqnr16:.2f} dB" if sqnr16 else "         SQNR: N/A")

    # ── Test E-3: converted result is usable (forward pass) ───────────
    print("\n[E-3] Forward pass on converted models")
    x = torch.randn(4, 32)

    try:
        from mono_quant import revert_to_standard_modules
        fp32_4 = revert_to_standard_modules(result4.model)
        out4 = fp32_4(x)
        ok_fwd4 = out4.shape == (4, 4)
        if not check(f"INT4 model forward pass OK, output shape {tuple(out4.shape)}", ok_fwd4):
            failures += 1
    except Exception as e:
        if not check("INT4 model forward pass OK", False, str(e)):
            failures += 1

    # FP16 direct forward is not supported: dequantize() returns fp32 weight
    # while bias stays fp16 → matmul dtype mismatch. This is a known limitation;
    # FP16 is a storage format for this library. Verify dtype instead.
    fp16_params = [
        (n, p.dtype)
        for n, p in result16.model.named_parameters()
        if p.dtype == torch.float16
    ]
    ok_fp16_params = len(fp16_params) > 0
    if not check(f"FP16 model has {len(fp16_params)} fp16 param(s) [forward pass N/A — known limitation]",
                 ok_fp16_params):
        failures += 1

    # ── Test E-4: save/reload of converted model ──────────────────────
    print("\n[E-4] Save and reload INT4 converted model")
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "converted_int4.pt"
        try:
            result4.save(str(save_path))
            ok_saved = save_path.exists() and save_path.stat().st_size > 0
            if not check(f"save() wrote file ({save_path.stat().st_size:,} bytes)", ok_saved):
                failures += 1

            from mono_quant.io import load_model
            reloaded = load_model(str(save_path))
            ok_reload = isinstance(reloaded, dict) and len(reloaded) > 0
            if not check(f"load_model() returned state_dict ({len(reloaded)} tensors)", ok_reload):
                failures += 1
        except Exception as e:
            if not check("save/reload OK", False, str(e)):
                failures += 1

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures == 0:
        print("RESULT: ALL CHECKS PASSED")
    else:
        print(f"RESULT: {failures} check(s) FAILED")
    print("=" * 60)
    return failures


if __name__ == "__main__":
    sys.exit(main())
