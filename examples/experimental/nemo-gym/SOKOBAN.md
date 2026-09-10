# Single-turn Sokoban RL

Miles generates a solution using its standard SGLang rollout path, including
routing replay for MoE models. The reward hook posts the generated text to the
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

Keep reference solutions out of the prompt. Preserve the generated completion,
including its reasoning, when sending it for grading; NeMo Gym extracts the
answer from `<answer>...</answer>`, a boxed answer, or raw text.

In the Miles worker environment, put this example directory on `PYTHONPATH`
and set `NEMO_GYM_SOKOBAN_URL=http://127.0.0.1:8210`. Add
`--custom-rm-path sokoban_reward.reward_func` to a compatible model recipe, and
override its prompt data, checkpoint, batch size, and response budget as needed.
The localhost URL assumes the verifier and rollout worker share a node.

Before training, send known solutions and broken paths through the HTTP endpoint
and the reward hook. Expect rewards 1 and 0, respectively. During training,
inspect solve-rate variation, nonzero gradients, routing-replay diagnostics, and
sample traces. Training reward on the training puzzles is not a held-out score.
