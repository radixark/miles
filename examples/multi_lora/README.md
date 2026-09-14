# Multi-LoRA Tinker Gateway

> **Read the docs:** [Multi-LoRA training](https://miles.radixark.com/docs/advanced/lora#multi-lora-training).

- `serve_qwen3_30b_a3b_tinker.py`: prepare Qwen3-30B-A3B and launch the gateway; `--n-adapters -1` sizes the slot pool from measured memory.
- `run_multi_tenant_example.py`: check marker memorization for one client or adapter isolation across concurrent tenants.
- `run_client_recipes.py`: the official tinker-cookbook recipes against the gateway, the wire-contract acceptance bar.

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

`--n-adapters -1` lets the gateway size the slot pool instead of guessing it. The trainer
launches alone with one probe slot, runs a max-size forward/backward and an optimizer step,
measures the CUDA bytes that slot owns and the head-room left, and rebuilds at the resolved
count before the engines launch. `auto` is the smallest of three bounds, and the log names the
binding one:

1. the trainer's memory, measured on every rank (the worst rank rules; `--train-memory-margin-bytes` is the head-room kept free);
2. the rollout engines' memory with every slot sampling at once: each engine GPU holds every slot's adapter buffer plus the KV cache for `--multi-lora-rollout-seqs-per-slot` sequences of `--multi-lora-rollout-tokens-per-seq` tokens (defaults: 8 sequences, the context length);
3. `torch._grouped_mm`'s 1023-group limit on the expert adapters (`1023 // local_experts` slots).

```bash
python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve --n-adapters -1 \
  --extra-args "--multi-lora-rollout-seqs-per-slot 16 --multi-lora-rollout-tokens-per-seq 8192"
# the log tells you what it resolved to:
#   multi-LoRA capacity: 47 slots, bound by the rollout engines' memory with every slot sampling at once [...]
```

Unless `--sglang-max-loaded-loras` is set, the engines keep at most `slots + 16` adapter versions
loaded; without a cap every TP-rank process keeps a host copy of every version ever published.

## Load test and profiling

`run_pressure_test.sh` at the repo root serves the gateway with `--n-adapters -1`, reads the
slot count it resolved to, runs one `run_client_recipes.py` tenant per slot at once (`TASK=rl`
for the cookbook's GRPO on GSM8K, `sft`, or `both`; each tenant is its own `TINKER_API_KEY`),
and saves the tables as `report.txt`. Knobs are environment variables; on a multi-node Ray
cluster set `MILES_SCRIPT_EXTERNAL_RAY=1` and `RAY_ADDRESS`.

The gateway times every backend op (`forward_backward`, `optim_step`, `export_slot`,
`push_slot`, `sample`, ...) with `miles.utils.multi_lora_profiling.OpProfiler` and logs
`multi-LoRA profile: {...}` plus a table after every optimizer step and at exit. Render it,
with the slot count and a node's GPU peaks, from the logs:

```bash
python -m miles.utils.multi_lora_profiling --serve-log <gateway log> --gpu-csv gpu-<ip>.csv
```

## Failure handling

A terminal failure of `forward_backward`, `optim_step`, or `load_state` ends
that model's training stream, including commands already queued behind it.
This includes content validation failures with a valid model and sequence.
Create a new model and restore a saved checkpoint to continue; completed
futures and published checkpoints keep their results.

Known request-local failures of `forward`, saving, or sampling leave the
training stream available. Saving sampler weights commits an immutable directory;
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
