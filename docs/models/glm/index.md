---
title: GLM
sidebarTitle: Overview
description: Miles recipes for the GLM4.5, GLM4.7 Flash, GLM5, and GLM5.2 families.
---
Miles ships RL recipes for every GLM generation currently in production: the GLM4.5 MoE at 106 B-A12B and 355 B-A32B, the compact GLM4.7 Flash with 64 routed experts, and the 744 B-A40B GLM5 and GLM5.2 flagships.

## Variants

| Family | Class | Sizes | Recipe |
|---|---|---|---|
| GLM4.5 | MoE | 12 B / 106 B · 32 B / 355 B | [glm4-5](/models/glm/glm4-5) |
| GLM4.7 Flash | MoE (64 experts, top-4) | Compact | [glm4-7-flash](/models/glm/glm4-7-flash) |
| GLM5 | MoE | 40 B / 744 B | [glm5](/models/glm/glm5) |
| GLM5.2 | MoE | 40 B / 744 B | [glm5-2](/models/glm/glm5-2) |

## Fastest path to train

GLM4.7 Flash on a single 8× H100 node — the smallest GLM recipe:

```bash
python scripts/run_glm47_flash.py
```

See the [GLM4.7 Flash](/models/glm/glm4-7-flash) page for weight conversion and the full walkthrough.

## Which variant do I pick?

- **Single-node GLM first try** → GLM4.7 Flash ([glm4-7-flash](/models/glm/glm4-7-flash)).
- **MoE on a budget** → GLM4.5-106B-A12B ([glm4-5](/models/glm/glm4-5)).
- **Full MoE scale (multi-node)** → GLM4.5-355B-A32B ([glm4-5](/models/glm/glm4-5)).
- **Compact MoE for routing experiments (R3)** → GLM4.7 Flash ([glm4-7-flash](/models/glm/glm4-7-flash)).
- **Frontier scale (744 B)** → GLM5.2 ([glm5-2](/models/glm/glm5-2)); GLM5/GLM5.1 ([glm5](/models/glm/glm5)) for the previous generation.
