### 2026.01.26b

NOTE: I have not tested the "math + seq len 1000" case yet. only tested short case.

```shell
rm -rf /tmp/sglang_dump_ref && CUDA_VISIBLE_DEVICES=4 SGLANG_DUMPER_ENABLE=1 SGLANG_DUMPER_PARTIAL_NAME=ref SGLANG_DUMPER_SERVER_PORT=-1 python tests/deepseekv4/test_forward_pass.py reference --ckpt-path /data/weights/hello2026_5layer_native_tp1 --prompt-mode math --input-seq-len 1000

rm -rf /tmp/sglang_dump_sgl && CUDA_VISIBLE_DEVICES=5 SGLANG_DUMPER_ENABLE=1 SGLANG_DUMPER_PARTIAL_NAME=sgl SGLANG_DUMPER_SERVER_PORT=-1 python tests/deepseekv4/test_forward_pass.py megatron --hf-checkpoint /data/weights/hello2026_5layer --ref-load /root/models/DeepSeek-V4-285B-5layer_torch_dist --prompt-mode math --input-seq-len 1000

DUMP_COMPARATOR_UNIFY_MILES=1 python -m sglang.srt.debug_utils.dump_comparator --baseline-path /tmp/sglang_dump_ref --target-path /tmp/sglang_dump_sgl
```

### 2026.01.26a

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