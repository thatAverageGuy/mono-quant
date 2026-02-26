# T-021: Manual vLLM Compatibility Test

## Environment

- GPU: RTX 4050 (or any CUDA-capable GPU)
- OS: Windows 11 / Linux
- Python: 3.11+

## Prerequisites

```bash
pip install vllm   # latest stable — installs torch, transformers, etc.
pip install mono-quant
```

## Step 1: Obtain and quantize a test model

Use `facebook/opt-125m` — small enough to download quickly, large enough to
exercise GPTQ packing across many layers.

```python
# prepare_model.py
from transformers import OPTForCausalLM
import torch

model = OPTForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float32)
torch.save(model, "opt_fp32.pt")
print("Saved opt_fp32.pt")
```

```bash
python prepare_model.py
```

## Step 2: Export to GPTQ

```bash
monoquant export-gptq \
  --model opt_fp32.pt \
  --output ./opt_gptq/ \
  --group-size 128
```

Expected output:
```
Loading model: opt_fp32.pt
Exporting to GPTQ: ./opt_gptq/
Export complete: ./opt_gptq/
  model.safetensors
  quantize_config.json
```

## Step 3: Validate checkpoint structure (no vLLM required)

```python
from mono_quant.export.common.validators import validate_gptq_checkpoint_structure
validate_gptq_checkpoint_structure("./opt_gptq/")
print("Structure valid.")
```

## Step 4: Load with vLLM

```python
# test_vllm.py
from vllm import LLM, SamplingParams

llm = LLM(model="./opt_gptq/", quantization="gptq")
params = SamplingParams(max_tokens=20)
outputs = llm.generate(["Hello, my name is"], params)
print(outputs[0].outputs[0].text)
```

```bash
python test_vllm.py
```

## Success Criteria

- [ ] `export-gptq` completes without error
- [ ] `./opt_gptq/model.safetensors` and `./opt_gptq/quantize_config.json` exist
- [ ] `validate_gptq_checkpoint_structure` passes
- [ ] vLLM loads the model without `ValueError` or `RuntimeError`
- [ ] vLLM generates at least 1 coherent token (not all `<unk>` or crash)

## Known Limitations

- mono-quant re-quantizes from FP32 using asymmetric INT4 with uniform groups.
  This is not calibration-data-aware GPTQ (no Hessian-based weight ordering).
  Perplexity will be higher than "true" GPTQ outputs. That is expected.

- `desc_act: false` in `quantize_config.json` — vLLM supports this path.
  Setting `desc_act: true` would require the `g_idx` to be sorted by group,
  which mono-quant does not currently implement.

- HuggingFace transformer models saved with `torch.save(model, path)` are
  loadable but the `config.json` is not present in the checkpoint dir. vLLM
  may require it. If so, also copy `config.json` from the downloaded model:

  ```python
  from transformers import AutoConfig
  AutoConfig.from_pretrained("facebook/opt-125m").save_pretrained("./opt_gptq/")
  ```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ValueError: in_features not divisible by group_size` | Use `--group-size 64` or a divisor of the layer's in_features |
| `RuntimeError: CUDA out of memory` | Use a smaller model or reduce `gpu_memory_utilization` in vLLM |
| `KeyError: quantize_config` | Confirm `quantize_config.json` is in the output directory |
| vLLM can't find `config.json` | Copy model config as shown in Known Limitations above |
