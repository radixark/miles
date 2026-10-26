# Multi-LoRA Tinker Gateway

> **Read the docs:** [Multi-LoRA training](https://miles.radixark.com/docs/advanced/lora#multi-lora-training).

- `serve_qwen3_30b_a3b_tinker.py`: prepare Qwen3-30B-A3B and launch the gateway.
- `run_multi_tenant_example.py`: check marker memorization for one client or adapter isolation across concurrent tenants.
- `tinker_agent/`: client package for agentic training; see [Agentic training](#agentic-training).

## Layout

One 8-GPU node, disaggregated (multi-LoRA forbids `--colocate`):

- 4 training GPUs: TP2 for the dense layers, EP4 for the 128 routed experts.
- 4 sampling GPUs: two SGLang engines of 2 GPUs each, serving adapter versions by name.
- 4 adapter slots (`--multi-lora-n-adapters`), rank up to 32, covering attention
  (`linear_qkv`, `linear_proj`), the per-expert MoE projections (`linear_fc1`, `linear_fc2`),
  and the output layer (`output_layer`) so the cookbook's default `train_unembed=True` is servable.

The example enables attention, MLP, and output-head training. Client SDK flags
must match the server's selected groups. See [LoRA target selection](../../docs/advanced/lora.md#hf-target-source-of-truth)
for `--target-modules attn,mlp,unembed`; use `attn,mlp` to disable output-head
training. Tinker accepts only these group names and rejects `--exclude-modules`.

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

The launcher sets `--log-probs-chunk-size 1024`, which computes log-probs in 1024-token chunks
to bound peak GPU memory on long sequences. To override it, add
`--extra-args "--log-probs-chunk-size <N>"` to the `serve` command.

Install `tinker` on the client, then run the marker checks:

```bash
# one client: train, save for sampler, sample back the marker
python examples/multi_lora/run_multi_tenant_example.py --base-model /root/models/Qwen3-30B-A3B --mode single

# four tenants training concurrently on the same prompt with different markers;
# passing means the adapters stayed isolated end to end
python examples/multi_lora/run_multi_tenant_example.py --base-model /root/models/Qwen3-30B-A3B --mode multi --clients 4
```

## Agentic training

The gateway also supports agentic RL with your own sandbox or harness through the official
Tinker cookbook capture recipe. The [tinker_agent](./tinker_agent/README.md) client package
records each model call's exact tokens and log-probs and converts each episode into training
data.

## Context window

Each sampled sequence, prompt plus completion, must fit a single window, defined as the smaller of:

- the trainer's per-sequence limit: `--max-tokens-per-gpu`, rounded down to a multiple of
  TP × `--data-pad-size-multiplier`, times the context-parallel size;
- the inference limit: `--sglang-context-length` (the model's maximum length if unset) minus 7
  tokens, so that SGLang accepts a full-window sequence as a prompt.

With the launcher defaults, the window is 8192 tokens. For each sampling request, the gateway:

- clamps `max_tokens` to the remaining window, without modifying the client's request;
- rejects a prompt that leaves no generation budget with HTTP 400 `prompt is too long`, before
  dispatch to the engine. Agent harnesses use this error to trigger history compaction;
- accepts a full-window sequence for scoring with `compute_logprobs`.

Training Datums are validated against the trainer limit only; a shorter inference window does
not reduce the trainable sequence length.

## Supported inputs

Training accepts text with 1-D loss inputs. 2-D soft targets, including SDFT,
are not supported. Sampling requires a `/sampler_weights/` path returned by
`save_weights_for_sampler()`; `/weights/` training checkpoints cannot be sampled directly.

## Failure handling

A terminal failure of `forward_backward`, `optim_step`, or `load_state` ends
training for that model, including commands already queued behind it.
This includes content validation failures with a valid model and sequence.
Create a new model and restore a saved checkpoint to continue; completed
futures and published checkpoints keep their results.

Known request-local failures of `forward` or sampling leave model training
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
and every inference engine. All trainer ranks participate in adapter gathering;
rank 0 writes the tensors and config, then `META.json` after all ranks finish.
Existing sampler versions cannot be overwritten. Saving between `forward_backward`
and `optim_step` neither applies nor discards pending gradients.

Training checkpoint saves and loads are serialized within the gateway. Overwriting
deletes the previous checkpoint before writing the new one; a failed overwrite does
not preserve the previous checkpoint. Checkpoints are ordinary directories;
overwriting does not retain hidden versions.
