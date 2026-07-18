---
title: Thinking Machines
description: Miles recipes for Thinking Machines Lab models — Inkling (975 B), a multimodal MoE with short convolution, relative attention, and a shared-expert sink.
---
Miles ships a native Megatron recipe for **Inkling**, Thinking Machines Lab's 975 B / 41 B-active multimodal mixture-of-experts model: local and global relative attention, the residual ShortConv, the shared-sink router and experts, and the image and audio encoders. The same backend drives both full-parameter and LoRA RL.

## Variants

| Model | Active / Total | Layers | HF ID | Recipe |
|---|---|---|---|---|
| Inkling | 41 B / 975 B | 66 | [thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling) | [inkling](/models/thinkingmachines/inkling) |

## Fastest path to train

Inkling needs 16 nodes of 4× GB300 and the `radixark/miles:inkling` image:

```bash
cd /root/miles
python scripts/run_inkling_975b.py train \
   --model-name Inkling --train-mode full --task dapo_math \
   --num-nodes 16 --num-gpus-per-node 4
```

See the [Inkling](/models/thinkingmachines/inkling) page for the architecture summary, HF → Megatron conversion, validated parallelism layouts, training attention backends, LoRA RL, and multimodal RL.
