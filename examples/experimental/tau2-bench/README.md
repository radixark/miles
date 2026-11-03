# tau2-bench

Agentic RL and evaluation on [tau2-bench](https://github.com/sierra-research/tau2-bench)
(airline / retail / telecom). The sibling `../tau-bench` example targets the original
tau-bench (`tau_bench` package, retail + airline only) and does not run tau2.

Each sample is one unmodified tau2 text simulation (`tau2.run.run_single_task`):

- tau2's `llm_agent` is the policy, pointed at miles' session server, so every agent
  turn is recorded token-in/token-out (`--tito-model qwen36`). The session matcher is
  `role_content_only` because tau2 replays assistant turns without `reasoning_content`;
  the stored turn (thinking + tool calls) wins.
- tau2 owns the user simulator, the user-side tools (telecom), the stop conditions and
  the evaluator; the evaluator's reward is the sample reward.
- The user simulator and retail's NL-assertion judge (retail rewards are DB x NL_ASSERTION)
  default to the policy model, served untraced by the rollout router with thinking off.
  Pass rates are therefore not comparable to the leaderboard, which uses `gpt-4.1` as the
  user; set `TAU2_USER_LLM` / `TAU2_JUDGE_LLM` (plus the provider key in the shell that
  starts ray) for external models.
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

## Run (Qwen3.6-35B-A3B, 8x H200, full-parameter)

```bash
cd examples/experimental/tau2-bench
# download, convert to torch_dist (--ref-load), write the prompt data
python run_qwen3_6_35b_a3b_tau2.py prepare
# eval before step 0 + rollouts only
python run_qwen3_6_35b_a3b_tau2.py train --mode debug_rollout_only --num-rollout 1
# GRPO, eval every 10 steps
python run_qwen3_6_35b_a3b_tau2.py train --num-rollout 100 --eval-interval 10
```

`eval/airline`, `eval/retail` and `eval/telecom` are the per-domain pass rates (pass^1)
on each domain's test split (20 / 40 / 40 tasks, so expect several points of run-to-run
noise). An agent turn whose thinking exhausts `--rollout-max-response-len` leaves an empty
reply, which tau2 rejects; the episode scores 0. wandb runs go to `radixarkai/miles-tau2`; the API key is read
from `~/.netrc`.
