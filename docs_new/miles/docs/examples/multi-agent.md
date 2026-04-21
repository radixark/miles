---
title: Multi-Agent Co-Evolution
description: Two specialised agents train together and improve each other.
---

# Multi-Agent Co-Evolution

**What you'll learn:** how to wire up an asynchronous multi-agent system in Miles, where
two (or more) specialised agents take alternating turns and the joint outcome drives a
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
trajectories, which empirically stabilises early training when most attempts fail.

## Launch script highlights

```bash
ROLLOUT_ARGS=(
   --custom-generate-function-path \
       examples.multi_agent.rollout_with_multi_agents.generate_with_multi_agents
   --prompt-data /data/dapo-math-17k/dapo-math-17k.jsonl
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

```python title="agent_system.py"
async def run_agent_system(args, sample, sampling_params):
    history = [{"role": "system", "content": SYSTEM_PROMPT_THINKER}]
    history.append({"role": "user", "content": sample.prompt})

    for turn in range(MAX_TURNS):
        # 1. Thinker moves
        thinker_out = await call_role(
            history,
            system_prompt=SYSTEM_PROMPT_THINKER,
            sampling_params=sampling_params,
        )
        history.append({"role": "assistant", "content": thinker_out})

        # 2. Verifier responds
        verifier_out = await call_role(
            history,
            system_prompt=SYSTEM_PROMPT_VERIFIER,
            sampling_params=sampling_params,
        )
        history.append({"role": "assistant", "content": verifier_out})

        # 3. Termination check
        if "<final_answer>" in verifier_out:
            break

    return assemble_sample(history, sample)
```

The same SGLang process serves both "roles" — we only swap the system prompt between
turns. This means **both agents are the same model** updating in lockstep. For
*architecturally distinct* agents (separate models), see the MrlX repo.

## Walkthrough — rollout integration

`rollout_with_multi_agents.py` exposes `generate_with_multi_agents(args, sample,
sampling_params)`. Internally it:

1. Calls `run_agent_system(args, sample, sampling_params)` to get a multi-turn
   transcript.
2. Tokenises the transcript and builds `loss_mask` — only the model's outputs (both
   roles) get loss=1; the prompt and any system messages are masked out.
3. Computes a single per-trajectory reward (using the parent `--rm-type`).
4. Returns the `Sample` for the trainer to pack.

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
appropriate URL per turn. The trainer can either train both jointly (one optimiser per
model) or train one and freeze the other (PvE).

### Adversarial pairing

Instead of a verifier, the second agent is an adversary that tries to find weaknesses
in the thinker's answer. Reward both: thinker for surviving, adversary for breaking.
This is the seed of self-play RLHF.
