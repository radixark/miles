# Full-parameter Tinker training (experimental)

`serve_tinker.py --tinker-full-training` uses Miles' native Megatron model and
Adam optimizer. It accepts one active training model per trainer. The existing
multi-LoRA mode remains selected by `--multi-lora-n-adapters`.

Use the usual model and parallelism arguments, `--megatron-to-hf-mode bridge`,
`--optimizer adam`, and `--tinker-checkpoint-root`. Trainers and inference engines
must start from the same HF checkpoint. The checkpoint directory must be visible
to every trainer rank and inference replica.

This first implementation requires BF16, CP=1, resident training and inference
GPUs, and synchronous parameter gathers. Use dynamic batching with a token
budget, or `--micro-batch-size 1`. FP16 loss scaling, independent DP, optimizer
offload, and MTP training are rejected at startup.

Create a model with `parameterization: {"type": "full"}` and no `lora_config`.
The `tinker==0.26.2` SDK has no public full-training creation helper; the manual
validation below demonstrates creating the model over HTTP and constructing its
ordinary SDK `TrainingClient`. Subsequent forward, backward, optimizer, and
checkpoint calls use that client.

## Training and checkpoints

`forward_backward` returns per-datum losses and logprobs and accumulates gradients.
`optim_step` applies the request's Adam parameters to all trainable model weights.
The server preserves the normalization supplied in the datum's loss inputs.
There is no server-side learning-rate schedule.

Each backward command runs Miles' Megatron pipeline. After gradient reduction,
the backend retains the optimizer-owned shards until `optim_step`. This adds one
FP32 gradient shard per rank and allows several commands to contribute to an update
without re-reducing previous commands' gradients.

`save_state` writes the full model and native optimizer state with Megatron
distributed checkpointing. Save after `optim_step`; pending gradients are not
checkpointed. Loading with the optimizer restores Adam moments and master weights.
A weights-only load resets Adam history and refreshes its master weights. These
checkpoints do not capture RNG state, so restoring them does not promise identical
future dropout or sampling results.

Releasing a model frees its admission slot. The next model starts from the
original HF checkpoint with fresh Adam history.

## Sampling

`save_weights_for_sampler` exports a complete HF checkpoint. Sampler names are
immutable. Sampling loads the requested snapshot onto every inference replica
and checks their reported versions before submitting generation. Repeated use
of the same snapshot avoids reloading the fleet until a replica changes.

The standalone gateway serializes sampling requests while sharing one inference
fleet. Switching between saved versions can therefore be expensive. A sampling
request can still generate several sequences concurrently through `num_samples`.
Training uses separate GPUs and can continue during sampling.

## Embedding the trainer

An external command backend can use the same `TrainerController` methods as the
multi-LoRA integration. Select `FullTrainingRayActor` by setting the launch option
`tinker_full_training=True`, then construct `MilesBackend(..., full_training=True)`.
The gateway process and its inference controller are optional for this use.

| Command | Full-training behavior |
| --- | --- |
| `load_slot(0, None, None)` | Admit a fresh full model. |
| `forward_backward` / `forward_only` | Execute native Megatron with Tinker losses. |
| `optim_step({0: adam_params})` | Update all trainable weights. |
| `save_slot(0, path)` | Capture model and optimizer into a complete directory. |
| `load_slot(0, None, None, ckpt_path=path, load_optimizer=...)` | Restore training state. |
| `export_slot(0, None, None, path)` | Capture full HF weights for inference. |
| `unload_slot(0)` | Release the model and pending gradients. |

These capture calls finish writing their directories before returning. An external
backend can then persist or publish the completed capture through its own storage
and inference infrastructure.

## Validation

Run the CPU gateway tests with `pytest tests/fast/tinker`. The loss tests in
`tests/fast/backends/training_utils/loss/test_tinker_loss_outputs.py` cover full
training and LoRA with the same client normalization.

Against a running full-training gateway:

```bash
python tests/manual/tinker/validate_full_training.py \
  --base-url http://localhost:10613 \
  --base-model Qwen/Qwen3-0.6B
```

The check saves before training, performs two optimizer steps, restores a model
and optimizer checkpoint, compares split and combined backward commands, and
checks that an older named sampler survives a switch to newer weights.
