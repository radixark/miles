---
title: "Multi-LoRA Tinker Gateway"
description: "Serve concurrent LoRA fine-tuning clients on one shared base model through the Tinker protocol."
# Generated from examples/multi_lora/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
> **Read the docs:** [Multi-LoRA training](https://miles.radixark.com/docs/advanced/lora#multi-lora-training).

- `run_gateway.py`: prepare Qwen3-30B-A3B and launch the gateway.
- `run_multi_tenant_example.py`: check marker memorization for one client or adapter isolation across concurrent tenants.

## Layout

One 8-GPU node, disaggregated (multi-LoRA forbids `--colocate`):

- 4 training GPUs: TP2 for the dense layers, EP4 for the 128 routed experts.
- 4 sampling GPUs: two SGLang engines of 2 GPUs each, serving adapter versions by name.
- 4 adapter slots (`--multi-lora-n-adapters`), rank up to 32, covering attention
  (`linear_qkv`, `linear_proj`), the per-expert MoE projections (`linear_fc1`, `linear_fc2`),
  and the output layer (`output_layer`) so the cookbook's default `train_unembed=True` is servable.

## Run

The gateway implements the `tinker==0.26.2` wire schema (newer SDKs renamed protobuf fields); install that exact version on the serving node and the client:

```bash
pip install "tinker==0.26.2"
```

Start the gateway:

```bash
python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py prepare   # once per node
python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve     # Tinker API on :10613
```

Checkpoints default to `<output_dir>/checkpoints/<run_id>`; use `--save-dir` to choose another root.

Install `tinker` on the client, then run the marker checks:

```bash
# one client: train, save for sampler, sample back the marker
python examples/multi_lora/run_multi_tenant_example.py --base-model /root/models/Qwen3-30B-A3B --mode single

# four tenants training concurrently on the same prompt with different markers;
# passing means the adapters stayed isolated end to end
python examples/multi_lora/run_multi_tenant_example.py --base-model /root/models/Qwen3-30B-A3B --mode multi --clients 4
```

## Measured slot capacity

`--n-adapters -1` (`--multi-lora-n-adapters auto`) lets the gateway size the slot pool instead
of guessing it. The trainer launches alone with one probe slot, runs a max-size
forward/backward and an optimizer step, measures the CUDA bytes that slot owns and the
head-room left, and is rebuilt at the resolved count before the engines launch. `auto` is the
smallest of three bounds, and the log names the binding one:

1. the trainer's memory, measured on every rank (the worst rank rules; `--train-memory-margin-bytes` is the head-room kept free);
2. the rollout engines' memory with every slot sampling at once: each engine GPU holds every slot's adapter plus the KV cache for `--multi-lora-rollout-seqs-per-slot` sequences of `--multi-lora-rollout-tokens-per-seq` tokens (defaults: 8 sequences, the context length; 0 sequences skips the bound);
3. `torch._grouped_mm`'s group limit, measured on the rank at probe time, over the expert adapters (`max groups // local experts` slots).

```bash
python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve --n-adapters -1 \
  --extra-args "--multi-lora-rollout-seqs-per-slot 16 --multi-lora-rollout-tokens-per-seq 8192"
# the log tells you what it resolved to:
#   multi-LoRA capacity: 63 slots, bound by torch._grouped_mm's 1023-group limit (16 local experts per slot) (gpu=81, host=unchecked, once_fb=63, rollout=74, ...)
```

Unless `--sglang-max-loaded-loras` is set, the engines keep at most `slots + 16` adapter versions
loaded; without a cap every TP-rank process keeps a host copy of every version ever published.

## Load test and profiling

`run_pressure_test.sh` at the repo root serves the gateway with `--n-adapters -1`, reads the
slot count it resolved to, runs one `run_client_recipes.py` tenant per slot at once (`TASK=rl`
for the cookbook's GRPO on GSM8K, `sft`, or `both`; each tenant is its own `TINKER_API_KEY`),
and saves the tables as `report.txt`. Knobs are environment variables (`STEPS`, `MAX_TOKENS`,
`GROUP_SIZE`, `BATCH_SIZE`, `CLIENT_LORA_RANK`, ...); on a multi-node Ray cluster set
`MILES_SCRIPT_EXTERNAL_RAY=1` and `RAY_ADDRESS`.

`miles/utils/multi_lora_profiling.py` wraps the backend from `serve_tinker.py`: every op
(`load_slot`, `forward_backward`, `optim_step`, `export_slot`, `sample`, ...) is timed and every
tenant request is timed from arrival to result by an HTTP middleware. The gateway logs
`multi-LoRA profile: {...}`, `multi-LoRA metrics: {...}` and `multi-LoRA requests: [...]` lines
after every forward/backward unit and optimizer step. The report renders them per LoRA per
step: the server side (operation level, share of the trainer's busy time), the client side
(API level, one row per Tinker API with mean/p50/p90/max and queueing, plus one LoRA's step
split into publish, rollout and train), reward, loss, log_prob and mean_len over the first and
last 10% of steps, and each node's GPU peaks:

```bash
python -m miles.utils.multi_lora_profiling --serve-log <gateway log> --gpu-csv gpu-<ip>.csv \
    --client-log <tenant log>  # --client-log is repeatable
```

## Failure handling

A terminal failure of `forward_backward`, `optim_step`, or `load_state` ends
that model's training stream, including commands already queued behind it.
This includes content validation failures with a valid model and sequence.
Create a new model and restore a saved checkpoint to continue; completed
futures and published checkpoints keep their results.

Known request-local failures of `forward` or sampling leave the training stream
available. Checkpoint load/save execution failures, including filesystem errors,
invalidate the shared trainer cell and stop the server.
Saving sampler weights commits an immutable directory;
it does not call the inference engines. Sampling loads that snapshot from disk
on demand, including after cache eviction. An engine load failure fails the
sampling request; it leaves the snapshot and training state intact. Unknown
trainer execution failures invalidate the shared trainer cell and stop the server.

This gateway provides failure isolation, not automatic training recovery.
Checkpoints persist; futures, deduplication, and unsaved accumulation do not
survive a server restart.

## Sampler snapshots

Training and inference must use the same base checkpoint. Tinker engines load
that frozen base at startup and serve without trainer weight updates; dummy
loading and `update_weights: true` are rejected. Ordinary full-model and
single-LoRA training continue to use the existing weight updater.

`--tinker-checkpoint-root` must be on storage shared by the trainers, gateway,
and every inference engine. A sampler save exports the current adapter weights,
then commits its tensors, adapter config, and `META.json` together by renaming
the completed directory. Existing versions cannot be overwritten. Saving between
`forward_backward` and `optim_step` neither applies nor discards pending gradients.
