---
title: Moore Threads MUSA GPUs
description: Run the validated Miles Qwen3-30B-A3B recipe on Moore Threads GPUs with Megatron-MUSA-Titan.
---

Miles runs on Moore Threads GPUs through the [Megatron-MUSA-Titan
integration](https://github.com/MooreThreads/Megatron-MUSA-Titan). This page covers
the validated `examples/rl/miles/demo_run-qwen3-30B-A3B.sh` recipe.

<Warning>

We support only the official Miles `v0.1.0` source layout. Miles
`main`, newer releases, and development snapshots are unsupported.

</Warning>

## Images

No MUSA Miles image has been published yet. Once one is available, this page
will be updated with its image name, registry location, tag, and startup
instructions. Until then, use a preconfigured Moore Threads host with the MUSA
driver/toolkit, MUSA-enabled PyTorch (`torch_musa`), the MUSA
[MT-TransformerEngine](https://github.com/MooreThreads/MT-TransformerEngine),
MCCL, MUSA-compatible SGLang, Megatron-LM, Miles `v0.1.0`, and
Megatron-MUSA-Titan installed. Keep the same software, model, and dataset paths
on every Ray node.

## Runtime

For a new Miles checkout:

```bash
git clone https://github.com/radixark/miles.git /path/to/miles
git -C /path/to/miles checkout v0.1.0
python -m pip install -e /path/to/miles --no-deps
```

Create the launcher environment from the Titan example and fill in the local
paths and host settings:

```bash
cd /path/to/megatron-musa-titan
cp examples/rl/miles/set_env.sh.example examples/rl/miles/set_env.sh
${EDITOR:-vi} examples/rl/miles/set_env.sh
```

Set the Miles, Megatron, Titan, model, checkpoint, and dataset paths together
with `MUSA_VISIBLE_DEVICES`, `MCCL_LIB`, and `GLOO_SOCKET_IFNAME`. Keep
`MUSA_TITAN_PATH` first in `RUNTIME_PYTHONPATH`. Set `RAY_HEAD_IP`, or provide
node addresses in the ignored `examples/rl/miles/hostfile`; the default Ray
Dashboard port is `8972`. For multi-node runs, allow Gloo/MCCL traffic and the
update-weight rendezvous ports starting at TCP port `23000`.

## Verify

```bash
source /path/to/megatron-musa-titan/examples/rl/miles/set_env.sh
cd "$MILES_PATH"
python -c 'import miles, torch, torch_musa; assert torch.musa.is_available(); print(torch.musa.device_count())'
```

## Launch training

Use an existing Ray cluster. The validated demo requests 8 actor GPUs and 16
rollout GPUs:

```bash
cd /path/to/megatron-musa-titan
source examples/rl/miles/set_env.sh
bash examples/rl/miles/demo_run-qwen3-30B-A3B.sh
```

The launcher configures the MUSA SGLang layout, `UPDATE_MODE=p2p-broadcast`,
and the startup environment for Ray and SGLang child processes. It writes the
followed log to `examples/rl/miles/logs/miles_demo_qwen3-30B-A3B.log`.

A run is valid only after the log reaches a training marker such as `step 0` or
`step 1`, following MUSA-Titan bootstrap, healthy SGLang engines, and a
completed weight update.

## References

- [Megatron-MUSA-Titan](https://github.com/MooreThreads/Megatron-MUSA-Titan)
- [MUSA Miles example README](https://github.com/MooreThreads/Megatron-MUSA-Titan/blob/main/examples/rl/miles/README.md)
- [Validated Qwen3-30B-A3B launcher](https://github.com/MooreThreads/Megatron-MUSA-Titan/blob/main/examples/rl/miles/demo_run-qwen3-30B-A3B.sh)
