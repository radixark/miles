---
title: INT4 Quantization-Aware Training
description: Train MoE policies with fake-quantized expert weights in Megatron and packed W4A16 rollout weights in SGLang.
---

miles INT4 QAT keeps the actor parameters in BF16 and applies symmetric INT4
fake quantization to routed MoE expert weights before each Megatron forward.
SGLang serves the same expert projections from a packed W4A16 checkpoint.

This reduces the precision mismatch between training and rollout while keeping
the rollout model compact. It does not quantize Megatron parameters, optimizer
state, gradients, or activations to INT4, so it is not a trainer-memory feature.

The current path is beta, uses the Megatron backend, and covers routed experts
implemented by Transformer Engine `GroupedLinear`. Attention, embeddings,
routers, shared experts, dense MLPs, and the LM head remain in their configured
precision.

## Component roles

| Component | Role |
|---|---|
| Hugging Face checkpoint | `--hf-checkpoint` initializes SGLang and supplies the quantization config used by every weight update. |
| Megatron | Keeps trainable BF16 weights. Before a grouped-expert forward, it quantizes each weight group to INT4 and dequantizes it back to BF16 for the GEMM. |
| Megatron Bridge or the miles raw-mode weight exporter | Maps updated Megatron weights to Hugging Face names and returns the packed tensors expected by the rollout checkpoint. |
| SGLang | Selects a compressed-tensors WNA16 MoE implementation, loads the packed INT4 weights, and runs rollout with 16-bit activations. |

The `W4` in W4A16 describes the stored SGLang expert weights. On the trainer,
INT4 is simulated in the forward pass; the underlying Megatron parameters stay
in BF16. `A16` means the expert GEMM uses BF16 or FP16 activations rather than
quantized activations.

## Weight lifecycle

1. SGLang loads the compressed Hugging Face checkpoint from
   `--hf-checkpoint`.
2. Megatron initializes the actor in its training dtype. Kimi-K2.5 can load the
   packed checkpoint directly through Megatron Bridge; raw-mode recipes
   initialize from a higher-precision Megatron checkpoint.
3. During training, Megatron fake-quantizes every routed-expert
   `GroupedLinear` weight immediately before the forward GEMM. Backward uses a
   straight-through estimator (STE), so the optimizer updates the BF16 weight.
4. At a weight-update boundary, Megatron Bridge or the miles raw-mode exporter
   exports Hugging Face-named tensors. The Kimi-K2.5 Bridge returns routed
   experts already packed as INT4; the generic miles path packs them from the
   checkpoint quantization config.
5. SGLang opens a weight-update session, restores the checkpoint-facing tensor
   layout, loads the new packed tensors, and rebuilds the layout required by
   its selected WNA16 kernel before rollout resumes.

For a Megatron weight group `w`, fake quantization is:

```text
scale = max(max(abs(w)) / 7, 1e-5)
fake_w = clamp(round(w / scale), -7, 7) * scale
```

## Quantization contract

The trainer, live exporter, and SGLang checkpoint must agree on all of the
following:

- Format: compressed-tensors `pack-quantized` with `num_bits: 4`,
  `strategy: group`, no activation quantization, and `symmetric: true`.
- Group size: the checkpoint's `group_size` must equal
  `OPEN_TRAINING_INT4_GROUP_SIZE`.
- Tensor scope: only the routed expert projections handled by Megatron
  `GroupedLinear` may be packed. All other tensors stay in their checkpoint
  dtype.
- Shape: the final dimension of each packed matrix must be divisible by the
  group size.

Changing the environment variable does not convert a checkpoint. Changing only
the checkpoint config does not change the fake-quantization grid used by
Megatron.

## Kimi-K2.5

[Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) publishes routed
expert weights as symmetric group-size-32 INT4 tensors. It is the clearest
reference for the Bridge path.

With `--megatron-to-hf-mode bridge`, the Kimi Bridge handles both directions:

- On Hugging Face to Megatron load, it unpacks the INT4 expert tensors and
  casts each distributed shard to the Megatron parameter dtype, BF16 in the
  current recipe.
- On Megatron to Hugging Face export, it repacks the updated routed experts to
  group-size-32 INT4 and returns `weight_packed`, `weight_scale`, and
  `weight_shape` tensors for SGLang.

The Bridge can therefore initialize a fresh actor directly from the published
INT4 checkpoint:

```bash
--hf-checkpoint /root/models/Kimi-K2.5
--megatron-to-hf-mode bridge
```

When a fresh run has no usable `--load` checkpoint and no `--ref-load`, miles
falls back to `--hf-checkpoint` for Bridge initialization. The current
`scripts/run_kimi_k25.py` launcher also materializes a BF16 copy and passes it
through `--ref-load`; that is how the launcher is written today, not a
requirement of the Kimi Bridge.

Run the CI-sized two-layer recipe on one 4-GPU H200 node:

```bash
python scripts/run_kimi_k25.py full-train \
  --model-name Kimi-K2.5-2layer \
  --num-nodes 1 \
  --num-gpus-per-node 4
```

For the full 32-node recipe, start Ray on the cluster and then run on the head
node:

```bash
python scripts/run_kimi_k25.py prepare \
  --model-name Kimi-K2.5 \
  --num-nodes 32

MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_kimi_k25.py train \
  --model-name Kimi-K2.5 \
  --num-nodes 32
```

See the [Kimi-K2.5 model guide](/models/kimi/kimi-k2.5) for its parallelism and
RL settings.

## Prepare another MoE checkpoint

The miles direct converter creates a symmetric, group-wise INT4 checkpoint
without a calibration dataset:

```bash
python tools/convert_hf_to_int4_direct.py \
  --model-dir /root/models/MyMoE-BF16 \
  --save-dir /root/models/MyMoE-INT4 \
  --group-size 128
```

Its default ignore rules leave embeddings, norms, attention, routers, shared
experts, dense MLPs, vision modules, and the LM head unquantized. Before using a
new model, compare those rules with its actual Hugging Face tensor names. The
converter requires CUDA and the `fake_int4_quant_cuda` extension installed by
the miles image.

The converter defaults to group size 32. The Qwen3 INT4 configurations set
Megatron QAT to group size 128, so pass `--group-size 128` explicitly when
preparing those checkpoints.

The current `scripts/run_qwen3_30b_a3b.py --rollout-int4` preparation path does
not pass that argument: it converts with the group-size-32 default while setting
Megatron QAT to 128. Until the launcher is aligned, prepare the Qwen3 INT4
checkpoint manually with the command above and run the training stage against
that directory.

In raw mode, use the INT4 checkpoint for SGLang and a higher-precision
`torch_dist` checkpoint to initialize Megatron:

```bash
--hf-checkpoint /root/models/MyMoE-INT4
--ref-load /root/models/MyMoE-BF16_torch_dist
```

Bridge mode can load the packed checkpoint directly only when that model's
Bridge implements the corresponding dequantization path, as Kimi-K2.5 does.

## Enable fake QAT

Set both variables in the Ray runtime environment used by every Megatron
worker:

```bash
RUNTIME_ENV_JSON='{
  "env_vars": {
    "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
    "OPEN_TRAINING_INT4_GROUP_SIZE": "128"
  }
}'

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python train.py ...
```

The Kimi launcher sets the same variables through `U.execute_train`, using
group size 32. If the selected Megatron model does not build its routed experts
with Transformer Engine `GroupedLinear`, the environment variables do not
enable QAT for that model.

## Validate the setup

Inspect the checkpoint config before launch. The expression handles both a
top-level quantization config and Kimi-K2.5's nested text config:

```bash
jq '(.quantization_config // .text_config.quantization_config) | {
  format,
  quant_method,
  ignore,
  weights: .config_groups.group_0.weights
}' /root/models/MyMoE-INT4/config.json
```

Confirm that:

- `format` is `pack-quantized` and `quant_method` is `compressed-tensors`.
- `num_bits` is `4`, `strategy` is `group`, and `symmetric` is `true`.
- `group_size` matches `OPEN_TRAINING_INT4_GROUP_SIZE`.
- The safetensors index has `weight_packed`, `weight_scale`, and
  `weight_shape` entries for routed experts only.

During a smoke test, confirm that SGLang selects a WNA16 MoE scheme and that a
weight update completes after the first optimizer step. For a new model, also
compare reward, KL, gradient norm, and train-versus-rollout log-probability
differences with a BF16 rollout baseline.

| Symptom | Check |
|---|---|
| SGLang fails at load or weight update | Check the packed tensor names, shapes, group size, and checkpoint ignore rules. |
| Training runs but QAT has no effect | Check that the environment reached every Megatron worker and that routed experts use Transformer Engine `GroupedLinear`. |
| Train and rollout log-probabilities diverge | Check the quantized tensor scope and group size first; MoE routing can be a separate source of mismatch. |
| Trainer memory does not decrease | Expected. Parameters, gradients, and optimizer state are not stored in INT4. |
| The converter cannot import `fake_int4_quant_cuda` | Use the miles CUDA image or build the repository's INT4 QAT extension. |

## Hardware and related features

The direct converter and generic miles INT4 packer use the CUDA
`fake_int4_quant_cuda` extension. SGLang's CUDA WNA16 MoE path requires NVIDIA
compute capability 8.0 or newer. SGLang also contains a ROCm WNA16 MoE
implementation, but the repository's reduced end-to-end INT4 QAT coverage is
currently the Kimi-K2.5 test on H200; treat other platforms as unvalidated.

INT4 reduces storage and weight-update traffic for the packed routed experts.
The whole-model reduction is smaller because scales, metadata, and unquantized
tensors remain. Rollout throughput depends on the model, batch shape, expert
parallelism, and the WNA16 backend SGLang selects.

[Rollout Routing Replay (R3)](/advanced/miles-router) addresses a different MoE
mismatch by replaying SGLang's expert choices in Megatron. Truncated Importance
Sampling (TIS) can correct residual train/rollout policy mismatch. Neither
changes the INT4 checkpoint or group-size contract.

## Related guides

- [Low Precision RL](/advanced/low-precision)
- [Kimi-K2.5 model guide](/models/kimi/kimi-k2.5)
- [P2P weight transfer](/advanced/p2p-weight-transfer)
- [slime low-precision training and rollout](https://thudm.github.io/slime/advanced/low-precision.html)
