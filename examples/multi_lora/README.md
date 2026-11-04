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

The gateway validates requests with the `tinker==0.36.0` request models and rejects clients on any other SDK version, patch releases included; install that exact version on the serving node and the client (`tinker-cookbook` only requires `tinker>=0.35`, so pin the SDK explicitly):

```bash
pip install "tinker==0.36.0"
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
and every inference engine. All trainer ranks participate in adapter gathering.
`sampler_weights/<v>` is a link to `.<v>.<commit_id>/`, which holds the adapter
files, `tenant.json` (the tenant digest) and `adapter_parametrization.json`; the
gateway commits the link after all ranks finish writing.
Existing sampler versions cannot be overwritten. Saving between `forward_backward`
and `optim_step` neither applies nor discards pending gradients.

Training checkpoint saves and loads are serialized within the gateway, and one
gateway holds a checkpoint root at a time (an `flock` on the root). That needs a filesystem that
enforces `flock` across hosts: NFS mounted `nolock` and Lustre with `localflock` lock only within one host,
so two gateways on different hosts could both hold the root. A save writes a
new commit and then points the name at it; an overwrite retires the previous commit
only after the new one is committed, so a failed overwrite keeps the previous
checkpoint. A load adopts the alpha the checkpoint was trained with.

### Adapter layout

`--lora-type canonical_lora` gives Q, K, V, gate and up independent factors in each slot,
matching PEFT and the Tinker cookbook; it is the default for `serve_tinker.py`.
`--lora-type lora` keeps one adapter for each fused QKV or gate/up linear.
It can be combined with `--experts-shared-outer-loras`: expert gate and up then each
have their own A matrix shared across experts; their B matrices remain per expert.
Saving and loading such slots in the committed checkpoint format has not been verified on GPU yet.
These settings are fixed for the gateway. A training checkpoint saved under another layout loads its
weights only, converted to this gateway's layout where that is exact: a shared factor splits, but
per-component or per-expert factors become shared only when they are equal. Its optimizer state, and
a checkpoint saved before commits, need the same layout.
Canonical Q/K/V needs every TP rank to hold whole KV heads (`num_query_groups >= tensor_model_parallel_size`),
and canonical gate/up needs separate HF gate and up tensors in dense MLPs; a model or layout outside that fails at
startup, so launch it with `--lora-type lora`.
On Qwen3.5 / 3.6, `canonical_lora` trains the selected GDN `in_proj_q/k/v/z/b/a` components with their own
`lora_A`, as official Tinker adapters do (the `attn` group takes `in_proj_q/k/v/z`); under `--lora-type lora` the
`attn` group adapts the fused `in_proj` whole, `in_proj_b` / `in_proj_a` included.
