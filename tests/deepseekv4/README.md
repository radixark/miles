### This is now runnable but not tested correctness

now supported: run 5 layer model with no TP on 1 GPU. Megatron's weight should be loaded from torch_dist checkpoint, reference code load from converted native checkpoint.

### Usage
Megatron:
```bash
python tests/deepseekv4/test_forward_pass.py megatron \
    --hf-checkpoint /data/weights/hello2026_5layer \
    --ref-load /root/models/DeepSeek-V4-285B-5layer_torch_dist
```

Reference:
```bash
python tests/deepseekv4/test_forward_pass.py reference \
    --ckpt-path /data/weights/hello2026_5layer_native_tp1
```