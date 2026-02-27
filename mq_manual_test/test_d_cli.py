# test_d_cli.py — Manual Test D: CLI smoke tests
#
# Tests that all mq CLI commands are reachable and produce correct output
# for a small synthetic model. Run from the mq_manual_test/ directory with:
#
#   venv/Scripts/python test_d_cli.py
#
# Requires: mono-quant 2.0.0 installed in venv/ (no extra installs needed)

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

VENV_MQ = Path(__file__).parent / "venv" / "Scripts" / "mq.exe"


def make_model():
    # Use nn.Sequential so torch.load can reconstruct the model in a
    # subprocess (mq.exe context) without needing a custom class import.
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


def run(args, cwd=None):
    """Run a CLI command, return (returncode, stdout, stderr)."""
    cmd = [str(VENV_MQ)] + args
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # prevent ✓/emoji encoding crash on Windows
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        cwd=cwd,
    )
    return result.returncode, result.stdout, result.stderr


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok and detail:
        print(f"         {detail}")
    return ok


def main():
    print("=" * 60)
    print("Manual Test D — CLI Smoke Tests")
    print("=" * 60)

    failures = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ── Setup: create test files ──────────────────────────────────
        print("\n[Setup] Creating test model files...")

        model = make_model()

        # fp32_full_model.pt — for mq quantize (load_model → torch.load → nn.Module)
        # Must save the full model object so load_model returns an nn.Module that
        # quantize() can accept without an architecture arg.
        fp32_full_path = tmpdir / "fp32_full_model.pt"
        torch.save(model, str(fp32_full_path))

        # fp32_state_dict.pt — for mq compare (compare_cmd uses .values() → needs dict)
        from mono_quant.io import save_model
        fp32_sd_path = tmpdir / "fp32_state_dict.pt"
        save_model(model, str(fp32_sd_path))

        # int8_full_model.pt — for mq convert (uses torch.load → nn.Module)
        from mono_quant import quantize
        result = quantize(model, bits=8, dynamic=True)
        int8_full_path = tmpdir / "int8_full_model.pt"
        torch.save(result.model, str(int8_full_path))

        print(f"  fp32_full_model.pt   ({fp32_full_path.stat().st_size:,} bytes)")
        print(f"  fp32_state_dict.pt   ({fp32_sd_path.stat().st_size:,} bytes)")
        print(f"  int8_full_model.pt   ({int8_full_path.stat().st_size:,} bytes)")

        # ── Test 1: mq export --list-formats ─────────────────────────
        print("\n[D-1] mq export --list-formats")
        rc, out, err = run(["export", "--list-formats"])
        ok = rc == 0 and "onnx" in out.lower()
        if not check("exit code 0, output contains 'onnx'", ok, f"rc={rc}\nstdout={out}\nstderr={err}"):
            failures += 1
        else:
            print(f"         formats listed: {[l.strip().split()[0] for l in out.splitlines() if l.strip() and not l.startswith('S')]}")

        # ── Test 2: mq quantize ───────────────────────────────────────
        print("\n[D-2] mq quantize --model fp32_full_model.pt --bits 8 --dynamic")
        int8_sd_path = tmpdir / "int8_state_dict.pt"
        rc, out, err = run([
            "quantize",
            "--model", str(fp32_full_path),
            "--bits", "8",
            "--dynamic",
            "--output", str(int8_sd_path),
        ])
        ok_rc = rc == 0
        ok_file = int8_sd_path.exists()
        if not check("exit code 0", ok_rc, f"rc={rc}\nstdout={out}\nstderr={err}"):
            failures += 1
        if not check("output file created", ok_file, f"expected: {int8_sd_path}"):
            failures += 1
        if ok_rc and "SQNR" in out:
            print(f"         {[l for l in out.splitlines() if 'SQNR' in l][0].strip()}")

        # ── Test 3: mq validate ───────────────────────────────────────
        print("\n[D-3] mq validate int8_state_dict.pt")
        if int8_sd_path.exists():
            rc, out, err = run(["validate", str(int8_sd_path)])
            ok = rc == 0 and "passed" in out.lower()
            if not check("exit code 0, 'passed' in output", ok, f"rc={rc}\nstdout={out}\nstderr={err}"):
                failures += 1
        else:
            print("  [SKIP] int8_state_dict.pt missing (D-2 failed)")
            failures += 1

        # ── Test 4: mq info ───────────────────────────────────────────
        print("\n[D-4] mq info int8_state_dict.pt")
        if int8_sd_path.exists():
            rc, out, err = run(["info", str(int8_sd_path)])
            ok_rc = rc == 0
            ok_tensors = "tensor" in out.lower() or "total" in out.lower()
            if not check("exit code 0", ok_rc, f"rc={rc}\nstdout={out}\nstderr={err}"):
                failures += 1
            if not check("tensor info in output", ok_tensors, f"stdout={out}"):
                failures += 1

        # ── Test 5: mq compare ────────────────────────────────────────
        # compare_cmd calls load_model() on both paths → needs state_dicts (.values() used)
        print("\n[D-5] mq compare fp32_state_dict.pt int8_state_dict.pt")
        if int8_sd_path.exists():
            rc, out, err = run(["compare", str(fp32_sd_path), str(int8_sd_path)])
            ok_rc = rc == 0
            ok_output = any(kw in out.lower() for kw in ["sqnr", "size", "ratio", "parameter"])
            if not check("exit code 0", ok_rc, f"rc={rc}\nstdout={out}\nstderr={err}"):
                failures += 1
            if not check("comparison output present", ok_output, f"stdout={out}"):
                failures += 1

        # ── Test 6: mq convert ────────────────────────────────────────
        print("\n[D-6] mq convert int8_full_model.pt int4_full_model.pt --bits 4")
        int4_full_path = tmpdir / "int4_full_model.pt"
        rc, out, err = run([
            "convert",
            str(int8_full_path),
            str(int4_full_path),
            "--bits", "4",
        ])
        ok_rc = rc == 0
        ok_file = int4_full_path.exists()
        if not check("exit code 0", ok_rc, f"rc={rc}\nstdout={out}\nstderr={err}"):
            failures += 1
        if not check("output file created", ok_file, f"expected: {int4_full_path}"):
            failures += 1
        if ok_rc and "SQNR" in out:
            print(f"         {[l for l in out.splitlines() if 'SQNR' in l][0].strip()}")

        # ── Test 7: mq info --format json ────────────────────────────
        print("\n[D-7] mq info int8_state_dict.pt --format json")
        if int8_sd_path.exists():
            rc, out, err = run(["info", str(int8_sd_path), "--format", "json"])
            ok_rc = rc == 0
            ok_json = False
            if ok_rc:
                import json
                try:
                    parsed = json.loads(out)
                    ok_json = "tensor_count" in parsed
                except Exception:
                    pass
            if not check("exit code 0, valid JSON with tensor_count", ok_rc and ok_json,
                         f"rc={rc}\nstdout={out[:200]}\nstderr={err}"):
                failures += 1

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    total_checks = 12
    if failures == 0:
        print(f"RESULT: ALL CHECKS PASSED")
    else:
        print(f"RESULT: {failures} check(s) FAILED")
    print("=" * 60)
    return failures


if __name__ == "__main__":
    sys.exit(main())
