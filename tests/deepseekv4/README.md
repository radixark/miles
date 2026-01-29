### 2026.1.28

```
rm -rf /tmp/sglang_dump_megatron_1rank /tmp/routing_replay_tp1 && CUDA_VISIBLE_DEVICES=4,5,6,7 SGLANG_DUMPER_ENABLE=1 SGLANG_DUMPER_PARTIAL_NAME=megatron_tp1 SGLANG_DUMPER_SERVER_PORT=-1 python tests/deepseekv4/test_forward_pass.py megatron-backward --hf-checkpoint /data/weights/hello2026_5layer --ref-load /root/models/DeepSeek-V4-285B-5layer_torch_dist --prompt-mode math --input-seq-len 1000 --tp-size 1 --routing-replay-dump-path /tmp/routing_replay_tp1

rm -rf /tmp/sglang_dump_megatron_tp2cp2 && CUDA_VISIBLE_DEVICES=4,5,6,7 SGLANG_DUMPER_ENABLE=1 SGLANG_DUMPER_PARTIAL_NAME=megatron_tp4 SGLANG_DUMPER_SERVER_PORT=-1 python tests/deepseekv4/test_forward_pass.py megatron-backward --hf-checkpoint /data/weights/hello2026_5layer --ref-load /root/models/DeepSeek-V4-285B-5layer_torch_dist --prompt-mode math --input-seq-len 1000 --tp-size 2 --cp-size 2 --routing-replay-load-path /tmp/routing_replay_tp1

DUMP_COMPARATOR_UNIFY_MILES=1 python -m sglang.srt.debug_utils.dump_comparator --baseline-path /tmp/sglang_dump_megatron_1rank --target-path /tmp/sglang_dump_megatron_tp2cp2
```


### 2026.1.27

Add 
(1) multi-GPU (TP) 
(2) backward gradient dump
(3) routing replay file sync for cross-TP gradient comparison

Note: when specify SGLANG_DUMPER_DISABLE_FORWARD_DUMP=1 will only dump grad, if not specify will dump both forward activation and backward gradients

Example: compare Megatron's gradient for TP=4 and TP=1 with same routing decisions

```
rm -rf /tmp/sglang_dump_megatron_tp1 /tmp/routing_replay_tp1 && CUDA_VISIBLE_DEVICES=4,5,6,7 SGLANG_DUMPER_ENABLE=1 SGLANG_DUMPER_PARTIAL_NAME=megatron_tp1 SGLANG_DUMPER_SERVER_PORT=-1 SGLANG_DUMPER_DISABLE_FORWARD_DUMP=1 python tests/deepseekv4/test_forward_pass.py megatron-backward --hf-checkpoint /data/weights/hello2026_5layer --ref-load /root/models/DeepSeek-V4-285B-5layer_torch_dist --prompt-mode math --input-seq-len 1000 --tp-size 1 --routing-replay-dump-path /tmp/routing_replay_tp1

rm -rf /tmp/sglang_dump_megatron_tp4 && CUDA_VISIBLE_DEVICES=4,5,6,7 SGLANG_DUMPER_ENABLE=1 SGLANG_DUMPER_PARTIAL_NAME=megatron_tp4 SGLANG_DUMPER_SERVER_PORT=-1 SGLANG_DUMPER_DISABLE_FORWARD_DUMP=1 python tests/deepseekv4/test_forward_pass.py megatron-backward --hf-checkpoint /data/weights/hello2026_5layer --ref-load /root/models/DeepSeek-V4-285B-5layer_torch_dist --prompt-mode math --input-seq-len 1000 --tp-size 4 --routing-replay-load-path /tmp/routing_replay_tp1

DUMP_COMPARATOR_UNIFY_MILES=1 python -m sglang.srt.debug_utils.dump_comparator --baseline-path /tmp/sglang_dump_megatron_tp1 --target-path /tmp/sglang_dump_megatron_tp4
```

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
