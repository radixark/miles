---
title: Multi-Agent Co-Evolution
description: Two specialized agents train together and improve each other.
---

# Multi-Agent Co-Evolution

**What you'll learn:** how to wire up an asynchronous multi-agent system in Miles, where
two (or more) specialized agents take alternating turns and the joint outcome drives a
single shared reward.

This example uses a dual-agent setup that interleaves a "thinker" and a "verifier", but
the same pattern scales to:

* Doctor / patient simulations.
* Multi-step DeepResearch pipelines.
* Adversarial games (proposer / solver).

The supporting framework for the production version of this is
[MrlX](https://github.com/AQ-MedAI/MrlX) — Miles ships the kernel of the same idea so
you can hack on it without pulling in MrlX's full dependency tree.

## Prerequisites

* You've completed the [Qwen3-30B-A3B](../models/qwen/qwen3-moe.md) recipe (the
  example uses that model).
* Familiar with [Customization](../user-guide/customization.md).

## Files

```text
examples/multi_agent/
├── agent_system.py                       # the agent state machine
├── prompts.py                            # role / system prompts
├── rollout_with_multi_agents.py          # custom rollout (calls agent_system)
└── run-qwen3-30B-A3B-multi-agent.sh      # launch script
```

## Quick start

```bash
cd /root/miles
bash examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh
```

## Configuration

```python
MULTI_AGENT_CONFIGS = {
    "custom_multi_agent_function_path":
        "examples.multi_agent.agent_system.run_agent_system",
    "num_parallel": 5,                  # parallel agent runs per prompt
    "incorrect_reward_weight": 0.8,     # weight on agent A's reward when wrong
    "correct_reward_weight": 1.2,       # weight on agent A's reward when right
}
```

Asymmetric reward weighting (0.8 / 1.2) gives a small bias toward upweighting "correct"
trajectories, which empirically stabilizes early training when most attempts fail.

## Launch script highlights

```bash
ROLLOUT_ARGS=(
   --custom-generate-function-path \
       examples.multi_agent.rollout_with_multi_agents.generate_with_multi_agents
   --prompt-data /root/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt --label-key label
   --apply-chat-template --rollout-shuffle
   --rm-type deepscaler

   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8

   --rollout-max-context-len 16384       # entire conversation budget
   --rollout-max-response-len 8192       # per-turn cap

   --global-batch-size 256
   --balance-data
)
```

Two flags matter most:

* `--rollout-max-context-len` — total context budget across all turns. Larger than
  `--rollout-max-response-len` because we accumulate.
* `--global-batch-size 256 = 32 × 8` — matches the rollout invariant.

## Walkthrough — the agent loop

The shipped pipeline is **solver → rewriter → selector**: `num_parallel` solver
attempts run in parallel, each rewriter rewrites the previous solutions, and a
`SelectorAgent` picks one. Sampling params are set on `args` upstream by the rollout
helper, so `run_agent_system` only takes `(args, sample)`.

```python agent_system.py
async def run_agent_system(args, sample):
    args = deepcopy(args)
    args.sample = sample
    args.results_dict = {"solver": [], "rewriter": [], "selector": []}

    problem_statement = sample.prompt

    # 1. Solver: num_parallel attempts in parallel.
    tasks = [solver_worker(args, problem_statement, i)
             for i in range(args.num_parallel)]
    solver_solutions = await asyncio.gather(*tasks, return_exceptions=True)
    rewards = await batched_async_rm(args, args.results_dict["solver"])
    for s, r in zip(args.results_dict["solver"], rewards):
        s.reward = r

    previous = [r for r in solver_solutions if isinstance(r, str)]

    # 2. Rewriter: each worker rewrites the previous solutions.
    tasks = [rewrite_worker(args, previous, problem_statement, i)
             for i in range(args.num_parallel)]
    rewritten = [r for r in await asyncio.gather(*tasks, return_exceptions=True)
                 if isinstance(r, str)]

    # 3. Selector: pick one of the rewritten solutions.
    selector = SelectorAgent()
    response = await selector.select(args, problem_statement, rewritten)

    # ... apply asymmetric reward weighting using
    # args.incorrect_reward_weight / args.correct_reward_weight on the solver
    # and rewriter samples, then return them all together.
    return args.results_dict["solver"] + args.results_dict["rewriter"] + ...
```

Both roles share the same SGLang process — `solver_worker`, `rewrite_worker`, and
`SelectorAgent.select` all post to the same engine, just with different prompts. So
**both agents are the same model** updating in lockstep. For *architecturally distinct*
agents (separate models), see the MrlX repo.

## Walkthrough — rollout integration

`rollout_with_multi_agents.py` exposes `generate_with_multi_agents(args, sample,
sampling_params, evaluation=False)`. Internally it:

1. Sets `args.sampling_params = sampling_params` and `args.tokenizer`, then loads the
   custom multi-agent function from `args.custom_multi_agent_function_path`.
2. Calls `await custom_multi_agent_func(args, sample)` to get the list of samples
   produced by the solver / rewriter / selector pipeline.
3. Returns the shuffled list of `Sample`s for the trainer to pack.

The per-sample tokenization and reward already happen inside `solver_worker` /
`rewrite_worker` / `SelectorAgent.select` (which call `batched_async_rm`), so the
rollout integration itself is a thin wrapper.

## Tuning knobs

| Knob | Effect |
|---|---|
| `MAX_TURNS` | Conversation depth — longer = more context = slower |
| `incorrect_reward_weight` / `correct_reward_weight` | Asymmetric shaping |
| `num_parallel` | Rollouts per prompt running concurrently |
| `--rollout-max-context-len` | Stops the conversation when budget is hit |

## What to watch

```text
multi_agent/avg_turns                       2.5 – 4.0
multi_agent/early_termination_rate          0.4 – 0.6 (reaches <final_answer>)
multi_agent/conversation_token_count        4096 – 12288
loss_mask/role_split                        balanced (~50/50)
reward/avg                                  trending up
```

If `loss_mask/role_split` is heavily skewed, one role is dominating — typically the
verifier becomes verbose. Tighten its system prompt or reduce its `max_tokens`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| OOM mid-rollout | Reduce `MAX_TURNS` or `--rollout-max-context-len` |
| Both agents repeat each other | Verifier prompt is too permissive — make it adversarial |
| Reward never moves | Check that `<final_answer>` extraction matches the verifier output |
| Rollout much slower than baseline | Per-turn SGLang RTT × MAX_TURNS — consider async rollout |

## Variations

### VLM multi-turn

Replace `call_role` with a VLM-aware caller that includes images in messages. Miles
supports VLM multi-turn natively — same pattern, just `multimodal_train_inputs` in the
sample dict (see [Customization #13](../user-guide/customization.md#training)).

### True asymmetric agents

Run two SGLang services — one per agent — and have your rollout function call the
appropriate URL per turn. The trainer can either train both jointly (one optimizer per
model) or train one and freeze the other (PvE).

### Adversarial pairing

Instead of a verifier, the second agent is an adversary that tries to find weaknesses
in the thinker's answer. Reward both: thinker for surviving, adversary for breaking.
This is the seed of self-play RLHF.
