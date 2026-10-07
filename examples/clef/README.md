# Clef-style calibration training

## Broad decision curriculum

`python -m examples.clef.build_broad_dataset` prepares 65,536 training cases
and a 4,096-case validation candidate pool: 30% rule-based business workflows, 15%
ToolACE tool selection, 15% SNLI evidence inference, 15% CLINC intent routing,
15% medium/hard SuperGPQA, 5% recorded human preferences, and 5% exact probability
problems. Every case has one or more named choice, yes/no, or ordered-score fields.
Targets and reference actions are outside the encoded input. The loader accepts
all three field types and preserves semantic field names during augmentation.

Finalize the training release with `python -m examples.clef.finalize_broad_dataset
--input-dir <candidate-directory> --output-dir <new-release-directory>
--tokenizer-dir <local-tokenizer>`. This keeps all 65,536 training cases byte for
byte and selects only 1,024 validation cases, stratified by source. It rechecks
every input using the native encoder and loader. The resulting `train.jsonl` and
`validation.jsonl` are directly consumable by the supervised decision trainer.
Upload these native records to the data dashboard as tables; they are decision
schemas with separate probability targets, not generated-answer chat traces.

The builder requires the pinned source cache, SuperGPQA source, previous validation
file (to preserve its SuperGPQA holdout), public JevBench directory, MMLU-Pro
parquet, GPQA ZIP, and local Qwen tokenizer; see `--help`. It checks duplicate
states/IDs/groups, target mappings, and actual full-schema tokenization through
the serving encoder for every case, without truncation. The manifest records
source revisions, checksums, field counts, lengths, exclusions and limitations.
GPQA and MMLU-Pro are not direct training sources; exact normalized matches are
excluded from SuperGPQA. JevBench and ForecastBench are not training sources.

Synthetic workflows cover invoices, returns, security triage and agent audits
with explicit policies, time-ordered evidence, and unrelated document distractors.
Some records bundle 4–20 independent cases with explicit per-field case references,
providing long contexts and dozens to hundreds of decision fields. These are
batched audits, not a substitute for organically long, coherent business scenarios.
Their labels are programmatically verifiable but realism is not established.
Validation holds out policy combinations and presentation, not every generator
family. Public-source validation holds out whole source examples/conversations.
Exact text exclusion is not a semantic contamination guarantee across the entire
Decision Index suite; audit the exact evaluation revision before claiming that.

This is a starting curriculum, not a reproduction of Cloudflare's private data.
SNLI supplies short inference rather than long contract inference; CLINC supplies
intent routing rather than passage ranking; ToolACE supplies tool selection
rather than argument generation; recorded preferences supply pairwise helpfulness
rather than aesthetic judgment or consensus probabilities. Monitor these missing
subskills separately. Cases differ in field counts; the trainer averages field
loss within each case before batch averaging.
Validation now includes every decision field, rather than only the first field
of a case. `brier` retains equal case weighting to match training, while
`brier_per_field` reports the pooled field average. Accuracy, ECE and collapse
measure all field distributions; `records` and `questions` distinguish input
cases from decision fields.

## Model training

This experimental Miles example attaches a freshly initialized Cloudflare Clef
joint schema head to original Qwen3.8-27B weights. It first trains the head with
the text backbone frozen, then fine-tunes the full text backbone and head. It
does not use LoRA, autoregressive probability reports, or generated reasoning.
The vision encoder is preserved but frozen and unused by this text-only pilot.

The head and record encoder are adapted from
[Cloudflare/clef](https://huggingface.co/Cloudflare/clef), revision
`2f3de3dd85f379784083b0814d997ab627200f0c`, under Apache-2.0. The architecture
and input encoding are unchanged. Its three scalar parameters are represented
as length-one parameters for FSDP2 and restored to scalars in serving exports.
The original head uses a 1024-wide representation, two evidence-routing layers,
four joint decoder layers, 16 heads, and a 4096-wide feedforward block.

## Data and objective

The data directory must contain `train.jsonl` and `validation.jsonl` in the
calibration format: `prompt` and `metadata` with `id`, `source`, `choices`, and
`target`. The adapter removes the old JSON-report instruction and turns each
question into a state plus a typed choice schema. It validates finite,
nonnegative normalized targets, unique IDs, and disjoint train/validation IDs.
Oversized records are rejected; no prompt is silently truncated.

The loss is differentiable Brier loss on the softmax of the head's option
logits. Synthetic examples retain their exact outcome distributions. Choices
are shuffled and relabeled together with their target probabilities at every
training visit. By default, 25% of records add a yes/no field asking whether a
random candidate is the answer/outcome. Its target is the corresponding
marginal probability, so this supplies consistent multi-field supervision
without leaking labels into the prompt. Fields are averaged within a record;
records receive equal weight.

Keep final benchmark questions out of both training and validation. This
example never loads benchmark data during training. Validation is deterministic;
there are no sampling-temperature or GRPO-group parameters for this supervised
phase. Interaction RL, if needed later, requires a separate action reward.

## Distributed execution

Use a uv environment with CUDA PyTorch and install the dependencies in
`examples/clef/requirements.txt`. The entrypoint is launched directly with
`torchrun`; it reuses Miles FSDP precision and checkpoint utilities and native
dashboard telemetry, rather than its token-level RL actor. Both nodes are
trainers. Logs and traces may use node-local output storage. Checkpoint storage
must be shared by every rank; set --checkpoint-dir for multi-node runs.

Example, issued on each of two nodes with its corresponding node rank:

```text
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=<0-or-1> \
  --master-addr=<rank-zero-address> --master-port=29550 \
  -m examples.clef.train \
  --model-dir /models/Qwen3.8-27B \
  --data-dir /data/calibration \
  --output-dir /scratch/clef/run \
  --checkpoint-dir /shared/checkpoints/clef/run \
  --run-name <dated-run-name> \
  --global-batch-size 64 --micro-batch-size 1 \
  --head-warmup-steps 128 --epochs 2 \
  --backbone-lr 3e-7 --head-lr 1e-5 \
  --save-interval 128 --eval-interval 32 \
  --wandb-project <project> --wandb-entity <entity>
```

With 16,384 examples, the joint phase has 256 updates per epoch, hence 512
updates for two epochs plus 128 head-warmup updates, for 640 updates total.
Warmup visits 8,192 shuffled examples before the two complete joint epochs.
Each of 16 ranks handles one record per microbatch and accumulates four
microbatches. Learning rates are constant within each phase, gradient norm is
clipped at 1, weight decay is zero, and FP32 master weights/Adam states preserve
small updates while FSDP2 computes in BF16 with FP32 gradient reduction.

The default maximum input length is 65,536 tokens; actual memory limits must
be established by preflight before a full run. Every process temporarily
loads the full model before sharding, so host/GPU initialization memory must
also be checked. There is no separate rollout context or generation budget.

## Artifacts and monitoring

Every 128 updates and at the final update, `checkpoints/step_<number>` contains
a native sharded model/optimizer checkpoint and `hf/` containing backbone
safetensors, the processor, `joint_head.safetensors`, configuration and loader.
Only checkpoints containing `COMPLETE.json` are eligible for evaluation or
resumption. `latest.json` is published atomically after all ranks finish saving.
Resume with `--resume <complete-checkpoint-directory>` and the original
configuration; incompatible data/optimizer/training settings are rejected.
Filesystem paths and logging destinations may change when moving a checkpoint
to a different machine. Training and validation file hashes must still match;
all model and optimizer tensors are restored from the native checkpoint.
Use the original head configuration and compatible processor when relocating
their files. The resumed run writes its effective configuration to `config.json`.

Validation saves every probability vector and logs overall/per-source Brier,
single-answer accuracy, ECE, confidently wrong answers, exact probability-1
collapse and near-collapse. ECE and accuracy only use one-hot targets; soft
synthetic targets use Brier. These metrics are also logged on each training
batch, alongside loss, gradient norm, learning rates and step time.

W&B is enabled by passing its project. A Prometheus endpoint exposes
`miles_metric_*` gauges on port 9090; use `--prometheus-port` if needed. Native
Miles dashboard streams are written under the output directory. Serve them with:

```text
python -m miles.dashboard.serve --dump-details <output-directory> --follow --port 7788
```

Per-rank JSONL traces preserve exact input token IDs, targets, predicted
distributions and option mappings. Unlike a generative rollout, these contain
no response tokens or reasoning. The dashboard metrics work with these native
streams; its token-generation rollout viewer is not applicable to this example.

For text serving, `load_release_model` and `systemone` in the exported
`joint_schema_model.py` produce probabilities directly in one prefill pass.
Standard SGLang chat serving does not execute this custom schema head.

The experimental `examples.clef.sglang_models` adapter replaces embedding
pooling with the trained schema head. Set `SGLANG_EXTERNAL_MODEL_PACKAGE` to
that package, `CLEF_MODEL_PATH` to the local serving export, and
`CLEF_METADATA_DIR` to a local directory shared with the SystemOne gateway.
Set `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0` for the backend: its ordinary
health endpoint must not submit a schema-free embedding probe. The gateway's
health endpoint instead executes a real schema request through the trained head.
Launch SGLang with `--is-embedding --tp-size 1 --disable-radix-cache
--chunked-prefill-size -1 --disable-cuda-graph --max-running-requests 1`.
Disable the automatic server warmup, which does not supply schema metadata.
Then run `python -m examples.clef.serve_systemone --model-path <export>
--metadata-dir <same-directory> --engine-url http://127.0.0.1:31000`.
The gateway accepts text-only `/v1/systemone` requests, preserves exact token
spans, and exposes full precision distributions separately from the rounded
SystemOne answer fields. This is deterministic prefill inference; generation
temperature and sampling repetitions do not apply. Validate probability parity
against the native release loader before benchmarking a new SGLang version.

S3 checkpoint storage is selected with --checkpoint-dir s3://bucket/prefix.
Native distributed shards are streamed to S3; the serving export is staged on
rank zero locally, uploaded with a SHA-256 manifest and size checks, and removed
only after publication of COMPLETE.json. Object-store latest.json is published
as a complete replacement object. Resumption accepts a complete S3 checkpoint
URI. This requires s3fs and uses the devbox credential chain.

For an infrastructure-provided object-store mount, pass its filesystem path to
--checkpoint-dir and add --checkpoint-object-store. This uses the mount's
credentials and sequential writes without relying on unsupported filesystem
renames. Native metadata is copied into place and COMPLETE.json remains the
publication boundary; exports are staged locally, verified, then removed.
The distributed timeout defaults to one hour to accommodate checkpoint upload
barriers; use --distributed-timeout-seconds to change it.

Gradient accumulation reduces every microbatch into sharded gradients to avoid retaining full-model FP32 gradient buffers. Loss scaling still preserves the configured global batch. Worker failures print their original traceback and exit without collective cleanup, allowing torchrun to stop peers.

To extend a completed run, resume its native checkpoint with --max-steps set to the desired cumulative update count (for example, 2000). The deterministic epoch ordering continues from the saved update. Resume checks allow extending the horizon while retaining dataset and optimizer settings.
