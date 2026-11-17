# tau2-bench

Agentic RL and evaluation on [tau2-bench](https://github.com/sierra-research/tau2-bench)
(airline / retail / telecom). The sibling `../tau-bench` example targets the original
tau-bench (`tau_bench` package, retail + airline only) and does not run tau2.

> **By default the policy also plays the user simulator and retail's NL-assertion judge.**
> Both then move with every weight sync, so pass rates are not comparable across steps or
> with the leaderboard, and retail's reward is graded by the model being trained. The
> launcher logs a warning when `TAU2_USER_LLM` / `TAU2_JUDGE_LLM` are unset. For
> comparable numbers point them at a fixed external model (the leaderboard uses `gpt-4.1`)
> or a frozen copy of the initial weights (`TAU2_USER_API_BASE` / `TAU2_JUDGE_API_BASE`).

| File | Role |
|---|---|
| `tau2_agent_function.py` | one tau2 simulation per sample (`--custom-agent-function-path`) |
| `tau2_generate.py` | generate wrapper that hands the router URL to the user simulator and aborts the samples of a failed episode, plus the reward hook |
| `make_tau2_data.py` | writes `<domain>_<split>.jsonl` and `all_train.jsonl` (178 train tasks) |
| `run_qwen3_6_35b_a3b_tau2.py` | full-parameter GRPO launcher, Qwen3.6-35B-A3B on 8x H200 |

Each sample is one unmodified tau2 text simulation (`tau2.run.run_single_task`):

- tau2's `llm_agent` is the policy, pointed at miles' session server, so every agent
  turn is recorded token-in/token-out (`--tito-model qwen36`). The session matcher is
  `role_content_only` because tau2 replays assistant turns without `reasoning_content`;
  the stored turn (thinking + tool calls) wins.
- tau2 owns the user simulator, the user-side tools (telecom), the stop conditions and
  the evaluator; the evaluator's reward is the sample reward.
- **The user simulator and retail's NL-assertion judge default to the policy model
  itself**, served untraced by the rollout router with thinking off. They therefore move
  with training, and pass rates are not comparable to the leaderboard, which uses
  `gpt-4.1` as the user. Set `TAU2_USER_LLM` / `TAU2_JUDGE_LLM` (plus the provider key in
  the shell that starts ray) for fixed external models.
- Communicate checks (airline) use tau2's original substring matcher; tau2 2.0 defaults to
  an OpenAI judge, selectable with `TAU2_COMMUNICATE_SCORER=semantic`.

## Setup

```bash
git clone https://github.com/sierra-research/tau2-bench.git /root/tau2-bench
pip install -e /root/tau2-bench --no-deps
pip install deepdiff addict docstring-parser loguru toml tenacity tabulate rich python-dotenv gymnasium
```

tau2 pins `litellm<1.82.7`; the miles image ships a newer litellm, which works for the
text path used here.

## Run

```bash
cd examples/experimental/tau2-bench
# download, convert to torch_dist (--ref-load), write the prompt data
python run_qwen3_6_35b_a3b_tau2.py prepare
# eval before step 0 + rollouts only
python run_qwen3_6_35b_a3b_tau2.py train --mode debug_rollout_only --num-rollout 1
# GRPO, eval every 10 steps
python run_qwen3_6_35b_a3b_tau2.py train --num-rollout 100 --eval-interval 10
```

`eval/airline`, `eval/retail` and `eval/telecom` are the per-domain pass^1 on each
domain's test split (20 / 40 / 40 tasks), averaged over 4 samples per task at temperature
1.0. With one sample per task the run-to-run noise was 5-10 points.

## Measured

100 steps with the launcher defaults on 8x H200 (about 36 h, evals included), with the
policy as user simulator and judge:

| step | airline | retail | telecom |
|---|---|---|---|
| 0 | 0.80 | 0.825 | 0.675 |
| 49 | 0.80 | 0.719 | 0.838 |
| 99 | 0.85 | 0.806 | 0.656 |

Rollout, the user simulator, the judge and the reward work end to end, but **this recipe
does not yet improve the pass rate**: telecom peaks mid-run and falls back while response
length grows (6k -> 8k tokens) and telecom episodes increasingly loop until
`TAU2_EPISODE_TIMEOUT_S` (1800 s). A frozen user simulator is the first thing to try.

An agent turn whose thinking exhausts `--rollout-max-response-len` leaves an empty reply,
which tau2 rejects; the episode scores 0 (see below).

## Episode outcomes

miles' agentic generate collects whatever the session recorded, whether or not the episode
finished, so the example decides the sample status itself: `tau2_agent_function.run` records
in the sample metadata whether the episode is valid and how tau2 ended it, and
`tau2_generate.generate` aborts the samples of an invalid one.

| How the episode ended | Metadata | Status | Effect |
|---|---|---|---|
| tau2 finished it (user stop, agent stop, step cap, protocol error, ...) | `tau2_termination_reason` = tau2's reason, `reward` = the evaluator's | as the session assembled it | trains |
| it ran past `--max-seq-len` (miles does not stop the agent: the session serves every turn, tau2 finishes and scores the episode) | as above | `TRUNCATED` by sample assembly, which keeps the first `--max-seq-len` tokens | trains with the evaluator's reward and counts in `truncated_ratio`; `--rollout-sample-filter-path miles.rollout.filter_hub.sample_filters.mask_truncated_zero_reward` leaves out only those scored 0 |
| the policy's reply was empty (its thinking spent `--rollout-max-response-len`; tau2's `AssistantMessage.validate` rejects it) | `tau2_termination_reason: "empty_reply"`, `reward: 0` | as the session assembled it (`TRUNCATED` when the turn hit the length limit) | trains with reward 0; `mask_truncated_zero_reward` leaves it out of the loss |
| `TAU2_EPISODE_TIMEOUT_S` elapsed | `tau2_termination_reason: "episode_timeout"`, `reward: 0` | as the session assembled it | trains |
| anything else raised (user simulator or judge endpoint, engine 5xx, the agent process, ...) | `tau2_episode_valid: false` | `ABORTED` | the whole group is dropped before training |

## Env vars

| Variable | Default |
|---|---|
| `TAU2_USER_LLM` / `TAU2_USER_API_BASE` | the rollout router (policy model) |
| `TAU2_JUDGE_LLM` / `TAU2_JUDGE_API_BASE` | the rollout router (policy model) |
| `TAU2_COMMUNICATE_SCORER` | `substring` |
| `TAU2_MAX_STEPS` | 100 |
| `TAU2_EPISODE_TIMEOUT_S` | 1800 |
