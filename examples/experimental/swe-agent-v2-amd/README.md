# Agent V2 — Qwen3 / Qwen3-Coder on ROCm/AMD

ROCm/AMD-oriented variant of [`../../swe-agent-v2`](../../swe-agent-v2). It is **self-contained** —
it carries its own copies of `run.py`, `generate.py`, and `swe_agent_function.py` so it runs
without modifying the shared example. See [`../../swe-agent-v2/README.md`](../../swe-agent-v2/README.md)
for the full pipeline/architecture; this doc covers only the Qwen3 launcher and the ROCm notes.

## Qwen3 / Qwen3-Coder

`run-qwen3-swe.py` runs the same pipeline with Qwen3 instead of GLM-4.7-Flash. It reuses
`run.py` and only swaps the model + SGLang/TITO parsers (`qwen25` / `qwen3` / `tito-model qwen3`).

```bash
# General model
python run-qwen3-swe.py --prompt-data /root/swe_train.jsonl

# Coding-specialised base (recommended for SWE-bench — see "Reward signal" below)
python run-qwen3-swe.py --coder --prompt-data /root/swe_train.jsonl
```

`--coder` uses **Qwen3-Coder-30B-A3B-Instruct**, which is arch-identical to Qwen3-30B-A3B
(same converter and `scripts/models/qwen3-30B-A3B.sh`) but uses `rope_theta=1e7`. The launcher
sets `MODEL_ARGS_ROTARY_BASE=10000000` for the Megatron side automatically; SGLang reads the
value from the HF config.

### Reward signal (why the coder)

GRPO learns only from **within-group reward variance**. A cold general model that solves ~0%
of SWE-bench Verified produces all-zero groups → zero advantage → no learning. Use a base that
lands in the "some pass / some fail" band (the coder), an easier task set (e.g. SWE-Gym), or a
warm-started checkpoint. With `rollout_batch_size=2`, also consider
`--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std`
plus over-sampling so every step contains variance-bearing groups.

### ROCm / AMD prerequisites

- **SGLang build with the `return_meta_info` chat-completion patch.** The TITO session server
  requires `choice.meta_info.output_token_logprobs` from `/v1/chat/completions`; a build
  without it returns HTTP 502 on every agent call. (This is a miles-SGLang-version
  requirement, not AMD-specific.)
- **GPU visibility:** handled automatically — `run.py` sets
  `RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1` only on ROCm.
- **agent-env container:** needs the Docker Compose v2 plugin (`docker compose`) — the Debian
  `docker.io` package does not bundle it — and, for `mini-swe-agent`, a litellm install that
  includes its proxy deps (recent litellm eagerly imports them): install with
  `--with 'litellm[proxy]'` or pin litellm, ideally baked into the task base image so it is not
  reinstalled per trial.
