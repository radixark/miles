---
title: NeMo Gym
description: Train on NVIDIA NeMo Gym environments through the agent-function extension point.
---

[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) is NVIDIA's RL environment
ecosystem: environments are HTTP *resources servers* (code execution, search,
SWE tasks, ...) paired with *agents* that drive an episode end-to-end and
grade it. Task containers run through NeMo Gym's own sandbox provider API
(`nemo_gym.sandbox`) — Docker locally, or Daytona / Apptainer / ECS Fargate /
OpenSandbox — selected by config, no agent changes.

Miles integrates NeMo Gym as an
[agent-function integration](/user-guide/environments): per sample, the agent
function POSTs the task to a NeMo Gym agent server's `/run` endpoint with
`policy_base_url` set to the session's OpenAI-compatible URL. NeMo Gym runs
its agent harness (mini-swe-agent v2 in `mini_swe_agent_2`) against that URL,
so Miles' session server records every turn losslessly (token ids, logprobs,
loss masks — see [Agentic Rollout (TITO)](/user-guide/agentic-rollout)); NeMo Gym
grades the episode itself and the grade enters training through a custom
reward hook reading `sample.metadata["reward"]`.

## Try it

The maintained recipe is **SWE-bench GRPO with mini-swe-agent** in
[`examples/experimental/nemo-gym`](https://github.com/radixark/miles/tree/main/examples/experimental/nemo-gym).
In short:

1. **Environment side** — on any docker-capable host, clone NeMo Gym `main`
   (>= `fcca3a8`) and start the `mini_swe_agent_2` responses-API agent server
   with the docker sandbox provider config.
2. **Data** — convert SWE-bench Verified to Miles prompt data with
   `download_and_process_data.py`; the task instance rides in each sample's
   `metadata`.
3. **Training side** — point `NEMO_GYM_URL` at the agent server and launch
   `run.py` (requires `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`, which the
   launcher sets), wiring the chain:

```bash
--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
--custom-agent-function-path nemogym_agent_function.run
--custom-rm-path nemogym_generate.reward_func
--use-session-server
--tito-model qwen3
```

The recipe is validated end-to-end: golden and API-policy scans on a real
docker host, plus a 4-GPU GRPO training smoke whose episodes ran in real task
containers with the SWE-bench harness grading them. Follow the
[recipe README](https://github.com/radixark/miles/blob/main/examples/experimental/nemo-gym/README.md)
for the NeMo Gym server setup, no-GPU validation (golden scan / API-policy
scan), the launch walkthrough, and known limitations (SWE-Gym eval specs,
Qwen3 template soft-mismatch diagnostics).
