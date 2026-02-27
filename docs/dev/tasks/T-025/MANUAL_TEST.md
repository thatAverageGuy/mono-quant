# Manual Test C — GGUF → llama.cpp

**Target environment**: Ubuntu (native boot, not Dev Containers)
**GPU required**: No — CPU inference is sufficient for format validation
**Time estimate**: ~20 min including llama.cpp build

---

## 1. File transfer: Windows → Ubuntu

Same as Test B. If you already did Test B, `opt_fp32.pt` is in `~/mq_test/`.

If not:
```bash
lsblk  # find Windows partition
sudo mkdir -p /mnt/win
sudo mount -o ro /dev/nvme0n1p3 /mnt/win  # adjust device

mkdir -p ~/mq_test
cp "/mnt/win/Users/ghost/OneDrive/Desktop/Test/Projects/mono-quant/mq_manual_test/opt_fp32.pt" ~/mq_test/
```

---

## 2. Python environment

If you already set up the venv for Test B, reuse it:
```bash
cd ~/mq_test
source venv/bin/activate
```

Otherwise:
```bash
cd ~/mq_test
python3 -m venv venv
source venv/bin/activate
pip install mono-quant
```

---

## 3. Get HuggingFace config (needed for GGUF metadata)

The GGUF exporter reads `config.json` to populate model metadata (architecture,
hidden size, num layers, etc.) in the GGUF header.

```bash
python - <<'EOF'
from transformers import AutoConfig
AutoConfig.from_pretrained("facebook/opt-125m").save_pretrained("./opt_config/")
print("Saved config to ./opt_config/")
EOF
```

---

## 4. Quantize and export to GGUF

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

mq export \
  --model opt_int4.pt \
  --output opt_125m.gguf \
  --format gguf \
  --config ./opt_config/config.json
```

> **Architecture note**: `facebook/opt` is not in mono-quant's GGUF arch map
> (`llama`, `mistral`, `qwen2`, `gpt2`, `deepseek_v2` are supported). OPT will
> auto-detect as `"generic"`, which exports with sequential tensor names
> (`blk.N.weight_M`). The GGUF file is structurally valid and verifiable with
> `validate_gguf_checkpoint`, but llama.cpp **cannot load generic-named tensors**
> — Step 7 is skipped. See "Note on llama.cpp inference" at the bottom.

Verify file:
```bash
ls -lh opt_125m.gguf
# Expected: non-zero file, typically 50–100 MB for OPT-125m INT4
```

---

## 5. Structural validation (automated, no llama.cpp needed)

```bash
pip install gguf  # lightweight gguf-py parser

python - <<'EOF'
from mono_quant.export.common.validators import validate_gguf_checkpoint
validate_gguf_checkpoint("opt_125m.gguf")
print("Structural validation: OK")
EOF
```

This verifies:
- Valid GGUF magic + version header
- `general.architecture` key present
- At least one quantized tensor entry

---

## 6. (Optional) llama.cpp inference test — requires LLaMA model

OPT exports with generic tensor names — llama.cpp can't load them. To test
actual llama.cpp inference, you need a LLaMA-family model (llama, mistral,
qwen2) which maps to named tensors. This is out of scope for v2.0 testing.

If you want to run it anyway with a supported model at a later point:

```bash
# Build llama.cpp (CPU only — no CUDA needed)
sudo apt install -y cmake build-essential
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_NATIVE=OFF
cmake --build build --config Release -j$(nproc)

# Then export a llama/mistral model and run:
~/llama.cpp/build/bin/llama-cli \
  -m <llama_model>.gguf \
  -p "The sky is" \
  -n 10 \
  --no-mmap \
  -t 4
```

---

## Success Criteria

- [ ] `mq export` completes and `opt_125m.gguf` is created (non-zero file)
- [ ] `validate_gguf_checkpoint` passes — header + `general.architecture` key + tensor list all valid
- [ ] (Optional) llama.cpp inference passes if a LLaMA-family model is available

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `validate_gguf_checkpoint` ImportError | `pip install gguf` |
| `validate_gguf_checkpoint` ValueError: missing `general.architecture` | Re-export with `--config ./opt_config/config.json` |
| `mq export` fails with "no quantized layers" | Re-run the INT4 quantize step — `opt_int4.pt` may be from an INT8 quantize |
| `opt_125m.gguf` is 0 bytes | Export error silently failed — re-run with `mq --verbose export ...` |

---

## Note on llama.cpp Inference with OPT

This is a **mono-quant gap**, not a llama.cpp limitation. llama.cpp has an OPT
model loader and can run OPT inference — it just expects specific tensor names
in the GGUF file.

The fix is adding OPT to `src/mono_quant/export/gguf/arch_maps.py`:
- `"opt"` entry in `_HF_TYPE_TO_ARCH`
- OPT tensor name map in `_ARCH_TENSOR_MAPS` (translating PyTorch names like
  `model.decoder.layers.0.self_attn.q_proj.weight` to the names llama.cpp's
  OPT loader expects)

Currently supported architectures: `llama`, `mistral`, `qwen2`, `gpt2`,
`deepseek_v2`. OPT was never added. Post-v2.0 work.
