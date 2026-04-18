---
title: FAQ
description: The questions every new Miles user asks in their first week.
---

# FAQ

??? question "Why do I see garbled text during training?"

    Almost always a checkpoint-loading issue. Megatron requires a directory that contains
    `latest_checkpointed_iteration.txt`. Verify:

    * `--load` (and/or `--ref-load`) point to a directory with that file.
    * If you want a specific iteration, use `--ckpt-step <N>`.

??? question "My job is stuck on the Ray submission page."

    Check whether you're running **colocated** or **disaggregated**:

    * **Colocated** (`--colocate`): total GPUs ≥ `actor_num_nodes × actor_num_gpus_per_node`.
    * **Disaggregated**: total GPUs ≥ `actor_num_nodes × actor_num_gpus_per_node + rollout_num_gpus`.

    A common mistake is forgetting `--colocate` when sharing GPUs.

??? question "I'm OOM during training. What is `max_tokens_per_gpu`?"

    `max-tokens-per-gpu` caps how many tokens a single GPU sees per micro-batch (only when
    `--use-dynamic-batch-size` is on, which it should be).

    Safe starting point:
    ```
    max_tokens_per_gpu = rollout_max_response_len / cp_size
    ```
    Then bump it up until you OOM, then back off ~10%.

    Still OOM with a small value? Your individual samples are too long — enable context
    parallel (`--context-parallel-size N`) to spread one sample across N ranks.

??? question "Multi-node training fails with `transformers cannot find a model`."

    Multiple workers calling `AutoConfig.from_pretrained` on a shared filesystem race each
    other. Set `--model-name <hf-id>` so workers don't re-resolve the path.

??? question "How do I resume training?"

    Set `--load` to whatever directory `--save` was writing to. That's it.

??? question "How is the batch size calculated?"

    For one rollout:

    * `rollout_batch_size` prompts are sampled.
    * Each prompt produces `n_samples_per_prompt` responses.

    So one rollout produces `rollout_batch_size × n_samples_per_prompt` samples.

    `--num-steps-per-rollout` decides how many optimiser steps consume that data. The
    invariant is:

    > `rollout-batch-size × n-samples-per-prompt = global-batch-size × num-steps-per-rollout`

??? question "Does Miles do data packing / varlen?"

    Yes, always. Variable-length samples are packed into the same micro-batch and the loss
    is corrected per-sample (or per-token if you set `--calculate-per-token-loss`). You
    never need to pad manually.

??? question "SGLang gives `Max retries exceeded with url: /get_model_info`."

    Multiple SGLang servers are colliding on the same node. Reduce the number of SGLang
    instances per node — e.g. set `tp_size=8` so there's exactly one server per host.

??? question "Gradient norm is huge and training crashes."

    Check the chat template first — most "exploding gradient" reports come from feeding
    already-templated data into a model that re-applies its template. Then read
    [Debugging](developer/debug.md).

??? question "SGLang takes forever, GPUs at 100%, no output."

    Stop tokens are misconfigured. The model is babbling without ever emitting EOS. Set
    them explicitly with `--rollout-stop` or `--rollout-stop-token-ids`.

??? question "SGLang error: `illegal memory access`."

    Per the [SGLang FAQ](https://docs.sglang.io/references/faq.html), this is usually OOM
    masquerading. Lower `--sglang-mem-fraction-static`.

??? question "`JSONDecodeError` from `torch.compile` / inductor."

    Torch's compiler cache is corrupt. Add `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` to your
    Ray env vars and re-run.

??? question "Gradient is NaN / Inf."

    Add `--no-check-for-nan-in-loss-and-grad` to skip the offending steps temporarily —
    then go investigate the data + model alignment that caused it.

??? question "NCCL error: `Failed to bind NVLink SHARP Multicast memory ... CUDA error 2`."

    Seen on H100 colocate + piece-wise CUDA graph. Piece-wise CUDA graph is now off by
    default in colocate. If you explicitly enabled it via
    `--sglang-enforce-piecewise-cuda-graph`, remove that flag.

??? question "Where do logs live?"

    | What | Path |
    |---|---|
    | Trainer stdout | wherever you redirected `ray job submit` |
    | SGLang | `/tmp/sglang/*.log` (or `--log-dir` if set) |
    | Ray workers | `~/.ray/session_latest/logs/` |
    | wandb | `wandb/` in your run dir, plus the cloud UI |

Still stuck? Drop a thread in the Miles channel of the [SGLang Slack](https://slack.sglang.ai)
or open an issue on [GitHub](https://github.com/radixark/miles/issues).
