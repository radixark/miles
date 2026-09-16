---
title: "VLM Multi-Turn (geo3k dataset)"
description: "The same dataset over multiple turns, with the model cropping images through an interactive environment."
# Generated from examples/geo3k_vlm/multi_turn/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Training VLM on [geo3k dataset](https://huggingface.co/datasets/hiyouga/geometry3k) with multi-turn reasoning with interactive environment feedback, using GRPO. For the dataset, we used the [processed version](https://huggingface.co/datasets/VeraIsHere/geo3k_imgurl_processed).

Note: Please make sure the cudnn version in the environment is 9.16.0.29 to prevent severe performance regression in conv3d in torch 2.9 mentioned in https://github.com/pytorch/pytorch/issues/168167. Otherwise, you can reinstall cudnn with:
```bash
pip install nvidia-cudnn-cu12==9.16.0.29
```

The multi-turn rollout is implemented through a [custom generate function](https://github.com/radixark/miles/blob/main/examples/geo3k_vlm/multi_turn/rollout.py#L309), overriding the original generate function.

In terms of the environment interaction, this example initializes a [custom interactive environment](https://github.com/radixark/miles/blob/main/examples/geo3k_vlm/multi_turn/env_geo3k.py) with the APIs below.

<Accordion title="Environment API (geo3k)">

- `build_env(sample: Sample | None = None, args: Any | None = None, **_) -> Geo3kEnv`: constructs the env.
- `reset() -> tuple[dict, dict]`: clears internal state.
- `step(response_text: str) -> tuple[dict, bool, dict]`: parses the actor's response text and update the state. Return new observation, a flag that marks whether the task is done, and step_info.
- `format_observation(observation: dict) -> dict`: converts an env observation into a chat message.

</Accordion>

<br />

The reward model is the default math RM. 

![VLM multi-turn geo3k reward](https://raw.githubusercontent.com/radixark/miles/main/examples/geo3k_vlm/multi_turn/geo3k_vlm_multi_turn_reward.png)
![Rollout megatron](https://raw.githubusercontent.com/radixark/miles/main/examples/geo3k_vlm/multi_turn/rollout_experiment_result_megatron.png)

## Reproduce
```bash
# 1) Set environment variable
export WANDB_API_KEY=...
export MILES_SCRIPT_MODEL_NAME=Qwen3-VL-2B-Instruct
export MILES_SCRIPT_NUM_GPUS=4
export MILES_SCRIPT_TRAIN_BACKEND=megatron

# 2) Download the dataset
hf download --repo-type dataset VeraIsHere/geo3k_imgurl_processed --local-dir /root/datasets/geo3k_imgurl_processed

# 3) Run the script:
cd /root/miles
python examples/geo3k_vlm/multi_turn/run_geo3k_vlm_multi_turn.py
```

## What each file does
- `examples/geo3k_vlm/multi_turn/run_geo3k_vlm_multi_turn.py`: downloads model, sets training/rollout args, and launches the run.
- `examples/geo3k_vlm/multi_turn/geo3k_vlm_multi_turn_config.yaml`: specifies `max_turns` and `rollout_interaction_env_path` for the multi-turn rollout.
- `examples/geo3k_vlm/multi_turn/rollout.py`: custom multi-turn rollout that calls SGLang for token generation, builds loss masks/log_probs, enforces max_turns, and early-stops on max_new_tokens.
- `examples/geo3k_vlm/multi_turn/env_geo3k.py`: geo3k tool-calling env that parses &lt;tool_call>\{...\}&lt;/tool_call>, scores math answers, and returns tool feedback per turn.
