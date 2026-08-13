---
title: INT4 Quantization-Aware Training
description: Train MoE policies with fake-quantized expert weights in Megatron and packed W4A16 rollout weights in SGLang.
---

miles INT4 QAT keeps actor parameters in Megatron's training dtype (BF16 by
default) and applies symmetric INT4 fake quantization to routed MoE expert
weights during each Megatron forward. SGLang serves the routed expert
projections from a packed W4A16 checkpoint.

The purpose is to expose training forwards to the weight quantization used by
rollout. The hook does not store Megatron parameters, optimizer state,
gradients, or activations in INT4, so it does not reduce their storage.

INT4 QAT uses the Megatron backend and covers routed experts implemented by
Transformer Engine `GroupedLinear`. Attention, embeddings, routers, shared
experts, dense MLPs, and the LM head remain in their configured precision.

## Component roles

| Component | Role |
|---|---|
| Hugging Face checkpoint | `--hf-checkpoint` initializes SGLang and defines the rollout storage format. The generic miles exporter reads its quantization config when packing live updates. |
| Megatron | Keeps trainable weights in the configured training dtype. Before a grouped-expert GEMM, it quantizes each weight group to INT4 and dequantizes it back to the input weight's dtype. |
| Megatron Bridge or the miles raw-mode weight exporter | Maps updated Megatron weights to Hugging Face names. For an INT4 rollout checkpoint, the Kimi Bridge returns packed tensors itself; the generic miles exporter packs eligible `.weight` tensors. |
| SGLang | Loads the packed tensors with a compressed-tensors W4A16 MoE implementation and runs the expert GEMMs with 16-bit activations. |

The `W4` in W4A16 describes the stored SGLang expert weights. On the trainer,
INT4 is simulated in the forward pass; the underlying Megatron parameters stay
in the configured training dtype. `A16` means the expert GEMM uses BF16 or FP16
activations rather than quantized activations.

## Weight lifecycle

1. SGLang loads the compressed Hugging Face checkpoint from
   `--hf-checkpoint`.
2. Megatron initializes the actor in its training dtype. Kimi-K2.5 can load the
   packed checkpoint directly through Megatron Bridge; the maintained Qwen3
   raw-mode recipe initializes from a higher-precision Megatron checkpoint.
3. During training, Megatron fake-quantizes every routed-expert
   `GroupedLinear` weight immediately before the forward GEMM. Backward uses a
   straight-through estimator (STE), so the optimizer updates the original
   trainable weight.
4. At a weight-update boundary, Megatron Bridge or the miles raw-mode exporter
   exports Hugging Face-named tensors. The Kimi-K2.5 Bridge returns routed
   experts already packed as INT4; the generic miles path packs them from the
   checkpoint quantization config.
5. SGLang opens a weight-update session and loads the new packed tensors. On
   the CUDA/Marlin WNA16 path used by the registered H200 tests, the session
   restores the checkpoint-facing tensor shapes before loading and rebuilds
   the Marlin layout before rollout resumes.

For a Megatron weight group `w`, fake quantization is:

```text
scale = max(max(abs(w)) / 7, 1e-5)
fake_w = clamp(round(w / scale), -7, 7) * scale
```

## Quantization settings

The rollout checkpoint and live exporter use the compressed-tensors storage
contract:

- Format: compressed-tensors `pack-quantized` with `num_bits: 4`,
  `strategy: group`, no activation quantization, and `symmetric: true`.
- Tensor scope: to match the current Megatron hook, pack the routed expert
  projections handled by `GroupedLinear` and keep other modules in their
  checkpoint dtype.
- Shape: the generic miles packer and direct converter require the final
  dimension of each packed matrix to be divisible by the group size.

Megatron's fake-QAT hook is configured separately through
`OPEN_TRAINING_INT4_GROUP_SIZE`. Set it to the checkpoint's `group_size` when
the training forward should use the same grouping as rollout. The two paths are
separate implementations, so matching this value does not by itself establish
bitwise-identical train and rollout weights.

Changing the environment variable does not convert a checkpoint. Changing only
the checkpoint config does not change the fake-quantization grid used by
Megatron.

## Kimi-K2.5

[Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5) publishes routed
expert weights as symmetric group-size-32 INT4 tensors. The maintained Kimi
launcher exercises this Bridge path.

With `--megatron-to-hf-mode bridge`, the Kimi Bridge handles both directions:

- On Hugging Face to Megatron load, it unpacks the INT4 expert tensors and
  maps them to the Megatron parameter dtype, BF16 in the maintained recipe.
- On Megatron to Hugging Face export, it repacks the updated routed experts to
  group-size-32 INT4 and returns `weight_packed`, `weight_scale`, and
  `weight_shape` tensors for SGLang.

The Bridge can therefore initialize a fresh actor directly from the published
INT4 checkpoint:

```bash
--hf-checkpoint /root/models/Kimi-K2.5
--megatron-to-hf-mode bridge
--model-name kimi_k25
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

The ignore rules are name-based rather than architecture-aware. By default they
match `lm_head`, `norm`, `embed`, `self_attn`, `shared_experts`,
`mlp.(gate|up|gate_up|down)_proj`, and `mlp.gate`, plus names beginning with
`vision_tower` or `mm_projector`. Compare those patterns with a new model's
actual Hugging Face tensor names before conversion. The converter requires
CUDA and the `fake_int4_quant_cuda` extension installed by the miles image.

The converter defaults to group size 32. The current
`scripts/run_qwen3_30b_a3b.py --rollout-int4` path uses that default for the
rollout checkpoint and sets Megatron fake QAT to group size 128. The path is
therefore an example where the rollout storage group and training fake-QAT
group are configured independently; it does not simulate the same grouping on
both sides.

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
- `group_size` matches `OPEN_TRAINING_INT4_GROUP_SIZE` if train and rollout
  should use the same grouping.
- The safetensors index has `weight_packed`, `weight_scale`, and
  `weight_shape` entries for the intended routed expert projections, and that
  other modules remain unpacked.

During a smoke test, confirm that SGLang selects a compressed-tensors W4A16 MoE
scheme and that a live weight update completes. For a new model, also compare
reward, KL, gradient norm, and train-versus-rollout log-probability differences
with a BF16 rollout baseline.

| Symptom | Check |
|---|---|
| SGLang fails at load or weight update | Check the packed tensor names, shapes, group size, and checkpoint ignore rules. |
| Training runs but QAT has no effect | Check that the environment reached every Megatron worker and that routed experts use Transformer Engine `GroupedLinear`. |
| Train and rollout log-probabilities diverge | Check the quantized tensor scope and group size first; MoE routing can be a separate source of mismatch. |
| INT4 QAT does not reduce trainer parameter or optimizer storage | Expected. Those states are not stored in INT4. |
| The converter cannot import `fake_int4_quant_cuda` | Use the miles CUDA image or build the repository's INT4 QAT extension. |

## Hardware and test coverage

The direct converter and generic miles INT4 packer use the CUDA
`fake_int4_quant_cuda` extension. SGLang's CUDA WNA16 MoE path requires NVIDIA
compute capability 8.0 or newer. SGLang also contains a ROCm WNA16 MoE
implementation. The repository registers two INT4 end-to-end tests, the
Kimi-K2.5 two-layer Bridge recipe and the Qwen3-30B-A3B raw-mode recipe, both on
4-GPU H200. It does not register an INT4 QAT end-to-end test on ROCm.

INT4 reduces storage and weight-update traffic for the packed routed experts.
The whole-model reduction is smaller because scales, metadata, and unquantized
tensors remain. Rollout throughput depends on the model, batch shape, expert
parallelism, and the W4A16 backend SGLang selects.

## Related guides

- [Low Precision RL](/advanced/low-precision)
- [Kimi-K2.5 model guide](/models/kimi/kimi-k2.5)
- [P2P weight transfer](/advanced/p2p-weight-transfer)
- [slime low-precision training and rollout](https://thudm.github.io/slime/advanced/low-precision.html)
