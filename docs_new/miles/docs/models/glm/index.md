---
title: GLM
description: Miles recipes for the GLM4, GLM4.5, GLM4.7 Flash, and GLM5 families — dense and MoE.
---
Miles ships RL recipes for every GLM generation currently in production: the dense GLM4 line (9 B, 32 B — Zhipu "Z1" reasoning checkpoints), the GLM4.5 MoE at 106 B-A12B and 355 B-A32B, the compact GLM4.7 Flash with 64 routed experts, and the 744 B-A40B GLM5 flagship.

## Variants

| Family | Class | Sizes | Recipe |
|---|---|---|---|
| GLM4 | Dense | 9 B · 32 B | [glm4](/miles/docs/models/glm/glm4) |
| GLM4.5 | MoE | 12 B / 106 B · 32 B / 355 B | [glm4-5](/miles/docs/models/glm/glm4-5) |
| GLM4.7 Flash | MoE (64 experts, top-4) | Compact | [glm4-7-flash](/miles/docs/models/glm/glm4-7-flash) |
| GLM5 | MoE | 40 B / 744 B | [glm5](/miles/docs/models/glm/glm5) |

## Fastest path to train

GLM4-9B (GLM-Z1-9B-0414) on a single 8× H100 node — the smallest GLM recipe:

```bash
cd /root/miles
hf download zai-org/GLM-Z1-9B-0414 --local-dir /root/GLM-Z1-9B-0414
bash scripts/run-glm4-9B.sh
```

See the [GLM4 Dense](/miles/docs/models/glm/glm4) page for weight conversion and the full walkthrough.

## Which variant do I pick?

- **Single-node GLM first try** → GLM4-9B ([glm4](/miles/docs/models/glm/glm4)).
- **Larger dense** → GLM4-32B ([glm4](/miles/docs/models/glm/glm4)).
- **MoE on a budget** → GLM4.5-106B-A12B ([glm4-5](/miles/docs/models/glm/glm4-5)).
- **Full MoE scale (multi-node)** → GLM4.5-355B-A32B ([glm4-5](/miles/docs/models/glm/glm4-5)).
- **Compact MoE for routing experiments (R3)** → GLM4.7 Flash ([glm4-7-flash](/miles/docs/models/glm/glm4-7-flash)).
- **Frontier scale (744 B)** → GLM5 ([glm5](/miles/docs/models/glm/glm5)).
