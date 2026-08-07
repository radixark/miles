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
| cell | one reconcile unit | `cell_id`, opaque |
| pod | one lifecycle unit | pod name / uid |
| worker | one rank | `<cell_id>-<worker_in_cell_index>` |

- A spec *declares* a pool. `spec` is a description, never an identity.
- `component` is the chart-rendered object prefix; say it only where those names
  are matched, `pool_id` everywhere else.
- Never parse `cell_id` and never assume cells are numbered `0..N-1`. Sorting ids
  gives an order everyone agrees on, not a numeric one.
- `worker_in_cell_index` stays dense because `torch.distributed` needs a rank.
- Numbering a cell is exempt in exactly three places: the Ray worker manager
  slicing a pool's gpus and ports between its own cells; the arithmetic on
  chart-rendered pod names that pairs colocated pods; and the chart's own python,
  which has to know how a StatefulSet or LeaderWorkerSet numbers its pods to
  address them before any of them exists. Everywhere else a cell is its `cell_id`
  and nothing else, and no synonym for that number may be coined.
- `fleet` and `group` are banned in our own naming. `LWS` may appear only where
  an upstream literal is quoted.
