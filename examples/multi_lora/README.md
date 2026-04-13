# Multi-LoRA Training Example

Train two LoRA adapters simultaneously on Qwen2.5-0.5B, each specializing
on a different math dataset with its own reward function:

- **gsm8k** — grade school math, uses `rm_type: math`
- **dapo_math** — competition math (DAPO-Math-17k), uses `rm_type: dapo`

Each adapter.yaml specifies the dataset path, column keys, and reward type.

## Directory Structure

```
adapters/
  math/
    adapter.yaml       # rank=16, data=gsm8k, rm_type=math
  dapo_math/
    adapter.yaml       # rank=16, data=dapo_math_17k, rm_type=dapo
```

## Usage

```bash
bash run.sh
```
