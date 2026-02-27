# Manual Test B — GPTQ → vLLM

**Target environment**: Ubuntu (native boot, not Dev Containers)
**GPU required**: Yes — CUDA-capable GPU (RTX 4050 or equivalent)
**Time estimate**: ~30 min including installs

---

## 1. File transfer: Windows → Ubuntu

`opt_fp32.pt` (626 MB) is already saved at:
```
C:\Users\ghost\OneDrive\Desktop\Test\Projects\mono-quant\mq_manual_test\opt_fp32.pt
```

From Ubuntu, mount your Windows NTFS partition and copy the file:

```bash
# Find Windows partition (usually /dev/sda2 or /dev/nvme0n1p3 — check with lsblk)
lsblk

# Mount it (adjust device as needed)
sudo mkdir -p /mnt/win
sudo mount -o ro /dev/nvme0n1p3 /mnt/win

# Copy model file to a working directory
mkdir -p ~/mq_test
cp "/mnt/win/Users/ghost/OneDrive/Desktop/Test/Projects/mono-quant/mq_manual_test/opt_fp32.pt" ~/mq_test/
```

If the NTFS partition is already mounted (Ubuntu detects it on boot), it will be under
`/media/<user>/` — check with `ls /media/$USER/`.

---

## 2. Python environment

```bash
cd ~/mq_test

python3 -m venv venv
source venv/bin/activate

# Install mono-quant first (needed before vLLM to avoid torch version conflict)
pip install mono-quant

# Install vLLM — this pulls in its own torch, CUDA libs, etc. (~5 GB)
pip install vllm
```

> **Note**: vLLM installs its own `torch` build. If you see a torch version conflict
> after, uninstall mono-quant's torch and let vLLM's version win:
> ```bash
> pip install mono-quant --no-deps
> pip install mono-quant[export]  # if ONNX extras needed
> ```

Verify GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 3. Quantize the model to INT4

`mq export` takes a pre-quantized model. Quantize first:

```bash
python - <<'EOF'
import torch
from mono_quant import quantize

print("Loading opt_fp32.pt …")
model = torch.load("opt_fp32.pt", weights_only=False)

print("Quantizing to INT4 …")
result = quantize(model, bits=4, dynamic=True)
print(f"  Layers quantized: {len(result.info.selected_layers)}")

torch.save(result.model, "opt_int4.pt")
print("Saved opt_int4.pt")
EOF
```

---

## 4. Export to GPTQ

```bash
mq export \
  --model opt_int4.pt \
  --output ./opt_gptq/ \
  --format gptq \
  --group-size 128
```

Expected output:
```
Loading model: opt_int4.pt
Exporting to gptq: ./opt_gptq/
Export complete: ./opt_gptq/
```

Verify files exist:
```bash
ls -lh ./opt_gptq/
# Expected: model.safetensors  quantize_config.json
```

---

## 5. Add HuggingFace config (required by vLLM)

vLLM needs `config.json` alongside the GPTQ weights. Save it from HF:

```bash
python - <<'EOF'
from transformers import AutoConfig
AutoConfig.from_pretrained("facebook/opt-125m").save_pretrained("./opt_gptq/")
print("config.json saved to ./opt_gptq/")
EOF
```

---

## 6. Validate GPTQ checkpoint structure (no vLLM needed)

```bash
python - <<'EOF'
from mono_quant.export.common.validators import validate_gptq_checkpoint_structure
validate_gptq_checkpoint_structure("./opt_gptq/")
print("Structure: OK")
EOF
```

---

## 7. Load and run with vLLM

```bash
python - <<'EOF'
from vllm import LLM, SamplingParams

print("Loading model with vLLM …")
llm = LLM(model="./opt_gptq/", quantization="gptq")

params = SamplingParams(max_tokens=20, temperature=0.0)
outputs = llm.generate(["Hello, my name is"], params)
print("Output:", outputs[0].outputs[0].text)
EOF
```

---

## Success Criteria

- [ ] `opt_int4.pt` created without error
- [ ] `./opt_gptq/model.safetensors` and `./opt_gptq/quantize_config.json` exist
- [ ] `validate_gptq_checkpoint_structure` passes
- [ ] vLLM loads the model without `ValueError` or `RuntimeError`
- [ ] vLLM generates at least one token (output is not empty or all `<unk>`)

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `lsblk` shows no Windows partition | Windows fast boot may have locked NTFS — disable fast startup in Windows before rebooting |
| `torch.cuda.is_available()` is False | Check NVIDIA driver: `nvidia-smi`. If missing: `sudo apt install nvidia-driver-535` then reboot |
| `ValueError: in_features not divisible by group_size` | Use `--group-size 64` |
| `RuntimeError: CUDA out of memory` | Add `gpu_memory_utilization=0.8` to `LLM(...)` constructor |
| `KeyError: quantize_config` | Confirm `quantize_config.json` is in `./opt_gptq/` |
| vLLM can't find `config.json` | Re-run Step 5 |
| vLLM output is gibberish | Expected — mono-quant uses dynamic (non-calibrated) INT4, not Hessian-based GPTQ. Coherent tokens are a bonus; the test only requires non-crash loading |

---

## Known Limitations

mono-quant re-quantizes from FP32 using asymmetric INT4 with uniform groups —
no calibration data, no Hessian-based weight ordering. Output quality will be
lower than true GPTQ. That is expected and not a test failure.
