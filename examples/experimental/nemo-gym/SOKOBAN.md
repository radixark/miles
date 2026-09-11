# Single-turn Sokoban RL

Miles generates a solution using its standard SGLang rollout path, including
routing replay for MoE models. The reward hook posts only the final move sequence to the
actual NeMo Gym Reasoning Gym resource server. That server replays the moves on
the initial board and returns a binary solved reward. Network failures raise an
error instead of becoming unsuccessful puzzle attempts.

Use a separate Python 3.13 environment for the CPU verifier. Install
`requirements-sokoban-server.txt` with `uv pip install`. Check out NeMo Gym at the
revision pinned in that file and put its root on the verifier's `PYTHONPATH`.
Serve `sokoban_server:create_app` using Uvicorn's `--factory` option, with this
example directory as `--app-dir`, listening on port 8210. The factory installs
the upstream resource server's HTTP routes without starting a second Ray
cluster. This is sufficient for the stateless Sokoban verifier.

Each Miles JSONL row should contain:

- `prompt`: a user message containing the original Reasoning Gym question;
- `label`: the reference move sequence (only sent to the verifier);
- `metadata`: the original board metadata, plus `sokoban_question` containing
  the original question and `source_dataset: "sokoban"`.

Keep reference solutions out of the prompt. The current reward hook expects a
reasoning model that generates `</think>` before its final answer; the opening
`<think>` may already be in the prompt. Require exactly one
`<answer>...</answer>` block after that boundary. Answers may contain uppercase
`U`, `D`, `L`, `R` and whitespace, so both `<answer>UDLR</answer>` and
`<answer>U D L R</answer>` are accepted. An optional trailing `<|im_end|>` is
supported. Boxed answers, raw text, nested or multiple answer blocks, and
answers without a reasoning boundary are rejected. A different reasoning
format needs an explicit adapter change; do not fall back to scanning all text.

Preserve the complete generation, including reasoning, for training and traces.
Only the validated final moves are sent to NeMo Gym, wrapped in a fresh answer
block. Sending reasoning as `output_text` lets upstream answer extraction consume
tags mentioned in the reasoning; the permissive Sokoban scorer can then replay
movement letters from that prose.

Truncated completions and invalid final answers receive zero without contacting
the verifier. `sokoban_grading_status` records the rejection reason, and
`sokoban_grader_version` identifies this policy as `final-answer-v1`.
Malformed task metadata, unexpected sample states, HTTP failures, invalid
verifier responses, and disagreement between the submitted and extracted moves
raise errors instead of becoming zero-reward training examples. Valid replies
must have matching numeric binary `score` and `reward` values.

In the Miles worker environment, put this example directory on `PYTHONPATH`
and set `NEMO_GYM_SOKOBAN_URL=http://127.0.0.1:8210`. Add
`--custom-rm-path sokoban_reward.reward_func` to a compatible model recipe, and
override its prompt data, checkpoint, batch size, and response budget as needed.
The localhost URL assumes the verifier and rollout worker share a node.

Before training, send known solutions and broken paths through the HTTP endpoint
and the reward hook, using completed samples with an explicit reasoning boundary
and final answer. Expect rewards 1 and 0, respectively. Include a correct final
answer following an unclosed answer tag in reasoning, and an incorrect final
answer following a correct reasoning-only plan. Run the offline regression suite
with `pytest tests/fast/examples/experimental/nemo_gym/test_sokoban_reward.py`.
During training,
inspect solve-rate variation, nonzero gradients, routing-replay diagnostics, and
sample traces. Training reward on the training puzzles is not a held-out score.

For policy-only Nemotron-H training, leave `MILES_NEMOTRONH_KEEP_MTP` unset or
set it to `0`. A constructed MTP head can add a next-token loss independently
of task rewards. For a fresh HF run, add
`--custom-megatron-before-train-step-hook-path sokoban_training_checks.before_train_step`
to fail before an optimizer update if the actual model contains an MTP head
or the initialization flags allow resumed training state.
