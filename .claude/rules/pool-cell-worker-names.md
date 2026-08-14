---
paths:
  - "miles/**/*.py"
  - "tests/**/*.py"
  - "charts/**"
---

# Pool, Cell, Worker

| Layer | What it is | Identity |
| --- | --- | --- |
| run | one training run | release |
| pool | the homogeneous cells one role deploys | `pool_id` |
| cell | one reconcile unit | `cell_id` |
| pod | one lifecycle unit | pod name / uid |
| worker | one Megatron rank / SGLang engine / ... | `<cell_id>-<worker_in_cell_index>` |

- A spec *declares* a pool. `spec` is a description, never an identity.
- `component` is the chart-rendered object prefix; say it only where those names
  are matched, `pool_id` everywhere else.
- `fleet` and `group` are banned in our own naming. `LWS` may appear only where
  an upstream literal is quoted.
