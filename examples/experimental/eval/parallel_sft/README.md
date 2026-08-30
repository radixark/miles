# Parallel checkpoint eval during SFT

This experiment connects pure SFT (`--debug-train-only`) to Miles' existing
asynchronous snapshot-eval path. Training uses the actor GPUs. A separate eval
fleet loads an HF export of the requested training step, and
`ParallelCommandEvalFn` starts all benchmark drivers in the YAML manifest at the
same time. Training continues while the eval runs.

The mental model is:

1. SFT finishes step 200.
2. Miles collectively exports `step_200` as an HF snapshot.
3. The eval engines load and verify weight version `200`.
4. Terminal Bench and HLE run concurrently against that pinned endpoint.
5. Their JSON summaries are flattened into `eval/<command>/...` metrics at
   training step 200.
6. With overflow policy `skip`, a still-running eval causes the next scheduled
   point to be recorded as busy instead of slowing training.

## Training flags

Add these to an SFT launch using `train_async.py`:

```text
--debug-train-only
--rollout-num-gpus 0
--eval-num-gpus 8
--eval-num-gpus-per-engine 1
--eval-function-path examples.experimental.eval.parallel_sft.parallel_command_eval.ParallelCommandEvalFn
--eval-datasets external-command
--eval-interval 200
--eval-hf-dir /shared/eval-snapshots
--eval-max-in-flight 1
--eval-overflow-policy skip
--eval-keep-snapshots 1
--skip-eval-before-train
--eval-sglang-context-length 262144
--eval-sglang-served-model-name qwen-checkpoint
```

`--eval-datasets external-command` satisfies the generic eval CLI contract; the
command manifest owns the actual benchmark inputs. Use a shared filesystem for
`--eval-hf-dir`, because the training actors export there and eval engines on
another node load from it.

Set the runner configuration before launch:

```text
MILES_PARALLEL_EVAL_CONFIG=/shared/config/parallel_eval.yaml
MILES_PARALLEL_EVAL_OUTPUT_DIR=/shared/eval-results
MILES_PARALLEL_EVAL_MODEL=qwen-checkpoint
```

`MILES_PARALLEL_EVAL_MODEL` and `--eval-sglang-served-model-name` must match.
You may omit both; the runner then uses the full `--hf-checkpoint` path, which
is SGLang's default served-model name.

Copy `parallel_eval.example.yaml` and set the referenced Terminal Bench and HLE
environment variables. The Terminal Bench example matches the established
Terminus-2 decoding settings (temperature 1.0, top-p 0.95, top-k 20, 81,920
output tokens, and 262,144 context). The agent server still owns sandbox setup,
summarization, and its three-hour agent timeout.

## Driver contract

Commands are executed without a shell. The following placeholders are available
in every `argv`, `env`, and `metrics_path` value:

- `{checkpoint_dir}`
- `{litellm_model}` (`openai/` plus the configured model name)
- `{model}`
- `{openai_base_url}`
- `{output_dir}` (unique to the training step)
- `{rollout_id}`
- `{router_url}`
- `{weight_version}`

The same values are exported as `MILES_EVAL_*` environment variables. A command
may write any JSON object to `metrics_path`. Numeric leaves are logged under
`eval/<command>/...`. A top-level `per_task` mapping is retained in the artifact
but omitted from W&B to avoid creating hundreds of metric keys. For per-sample
score reporting, emit this shape:

```json
{
  "metrics": {"accuracy": 0.42, "completed": 300},
  "rewards": [1.0, 0.0]
}
```

Stdout, stderr, and a copy of parsed metrics remain under the step output
directory. A failed driver is recorded with success 0 and its return code; it
does not terminate training or discard metrics from the other driver.
