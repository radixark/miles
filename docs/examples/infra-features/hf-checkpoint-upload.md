---
title: "Hugging Face model publishing"
description: "Uploads intermediate Megatron HF checkpoints to the Hugging Face Hub."
# Generated from examples/infra_features/hf_checkpoint_upload/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Publish intermediate or final Megatron model exports to the Hugging Face Hub
using native Miles flags.

## Usage

Add these flags to an existing Megatron training recipe, replacing its save flags
if necessary:

```bash
--save /checkpoints/my-run/trainer \
--save-interval 20 \
--save-hf '/checkpoints/my-run/hf/step-{rollout_id}' \
--push-to-hub \
--hub-model-id your-org/my-model \
--hub-private-repo \
--hub-strategy every_save
```

No custom post-save hook is needed. Publishing requires the Megatron backend,
`--save`, `--save-hf`, and a positive `--save-interval`. The normal training loop
also saves at the final iteration, even when it does not match the interval.

| Option | Behavior |
| --- | --- |
| `--push-to-hub` | Enable publication of completed HF exports. Disabled by default. |
| `--hub-model-id` | Required destination model repository, such as `your-org/my-model`. |
| `--hub-private-repo` | Create a private repository. Without this flag a new repository is public. Existing repository visibility is unchanged. |
| `--hub-strategy every_save` | Publish after every save, including the final one. This is the default. |
| `--hub-strategy end` | Publish only at the final training iteration. Local saves still follow the normal interval. |

Both strategies update the model at the repository root. Previous published
versions remain accessible through Hub commit history; Miles logs the commit ID.
Obsolete weight shards are removed in the same commit, while repository
documentation such as a model card is preserved. Use a dedicated repository for
each run to avoid concurrent writers overwriting each other's models.

The Hub receives exported model weights and configuration, plus the adapter
subdirectory for LoRA exports. Optimizer and other training state stay in the
native `--save` checkpoint. `checkpoint` and `all_checkpoints` strategies are not
supported. Interrupted training does not trigger an additional final upload.

## Authentication

Use `hf auth login` in the trainer environment with write access to the destination
repository, or your deployment's existing HF credential provisioning mechanism.
For remote Ray workers, the trainer must have access to the login cache; logging
in only on the submission machine is insufficient. Do not put tokens in launch
arguments or committed configuration files.

The repository is created on the first upload if it does not exist. The
implementation uses the standard `huggingface_hub` client and its authentication.
See the [Hub upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload).

## Transfer behavior

Only the actor's global rank zero uploads, after the export's `.complete` marker
confirms success. Incomplete exports are skipped. Uploads are synchronous: training
waits for the transfer. Choose a save interval and distributed timeout appropriate
for checkpoint size and network bandwidth. Background upload is not implemented.

Remote failures produce a warning and preserve the local export so training can
continue. Check the logs to confirm publication, including the final upload. To
retry manually from an authenticated environment:

```bash
hf upload your-org/my-model /checkpoints/my-run/hf/step-19 . --exclude '.complete'
```

## Loading a published model

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-org/my-model")
# Load a previous published version by passing revision="<commit-id>".
```

Use the base model's tokenizer if it is not included in the export. To resume
Miles training, use the original native trainer checkpoint.
