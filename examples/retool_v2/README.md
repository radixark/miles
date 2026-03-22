# Retool-v2

This example demonstrates how to use the retool functionality for tool-enabled language model RL training with multi-turn and agentic generation.

## Overview

The retool example provides:
- Safe Python code execution in a sandbox environment
- Tool registry for managing available tools
- Multi-turn and agentic generation modes
- Integration with GRPO/PPO training

## Files

- `run_multi_turn.sh`: RL training with multi-turn tool calling
- `run_agentic.sh`: RL training with agentic tool calling
- `tool_sandbox.py`: Tool execution and safety management

## Usage

### 1. Setup and download datasets

```bash
cd miles
pip install -e . --no-deps

# Download datasets
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
```

### 2. Convert Model to Megatron-LM Format

```bash
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --rotary-base 5000000 \
    --save /root/Qwen3-4B_torch_dist
```

### 3. Run Multi-turn RL Training

```bash
bash examples/retool_v2/run_multi_turn.sh
```

This will train the model using multi-turn tool calling with DAPO-math-17k dataset.

### 4. Run Agentic RL Training

```bash
bash examples/retool_v2/run_agentic.sh
```

This will train the model using agentic tool calling with DAPO-math-17k dataset.

## Tool Format

The system uses the following tool format:

```
You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "code_interpreter", "description": "A tool for executing code.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The code to execute."}}, "required": ["code"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

## Safety Features

- Code execution in isolated sandbox
- Memory and time limits
- Dangerous operation detection
- Allowed module restrictions
