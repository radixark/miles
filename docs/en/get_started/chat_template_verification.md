# Chat Template Verification

## Background

In agentic workflows (multi-turn tool-calling), miles uses sglang's **pretokenized prefix** mechanism to avoid re-tokenizing the entire conversation history on every turn. This requires the chat template to satisfy an **append-only invariant**: rendering the first N messages must produce a string that is an exact prefix of rendering all messages. Some model families (e.g. certain Qwen3 variants) ship templates that use `loop.last` or similar context-dependent Jinja logic, which breaks this property.

miles ships a one-click verification tool and an autofix mechanism to handle this.

## Quick Start

### Verify a HuggingFace model's template

```shell
python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B
```

Example output for a template that **fails**:

```
Template source: HuggingFace: Qwen/Qwen3-0.6B
Thinking cases:  disabled

  [FAIL] single_tool-N3           -- Prefix mismatch!
  [FAIL] multi_turn-N3            -- Prefix mismatch!
  [PASS] no_tool-N3

Results: 1/12 passed, 11 failed

Verdict: FAIL - template is NOT append-only after last user message
```

### Verify with autofix

If miles has a built-in fix for the model, use `--autofix` to test the fixed version:

```shell
python scripts/tools/verify_chat_template.py --model Qwen/Qwen3-0.6B --autofix
```

```
Template source: fixed template: .../miles/utils/chat_template_utils/templates/qwen3_fixed.jinja
Thinking cases:  disabled

  [PASS] single_tool-N3
  [PASS] multi_turn-N3
  ...
  [PASS] no_tool-N3

Results: 12/12 passed, 0 failed

Verdict: PASS - template IS append-only after last user message
```

### Verify a local template file

If you have a custom `.jinja` template, verify it directly:

```shell
python scripts/tools/verify_chat_template.py --template path/to/my_template.jinja
```

### Include thinking-specific cases

For models that support `enable_thinking` (e.g. Qwen3.5, GLM-5), add `--thinking` to also run thinking-specific test cases:

```shell
python scripts/tools/verify_chat_template.py --model Qwen/Qwen3.5-0.8B --thinking
```

This runs 26 cases in total (12 standard + 14 thinking with `enable_thinking=True/False`).

## CLI Reference

```
usage: verify_chat_template.py (--template PATH | --model MODEL_ID)
                               [--autofix] [--thinking]
```

| Argument | Description |
| :--- | :--- |
| `--template PATH` | Path to a local `.jinja` chat template file |
| `--model MODEL_ID` | HuggingFace model ID (e.g. `Qwen/Qwen3-0.6B`) |
| `--autofix` | When using `--model`, apply miles' fixed template if one exists |
| `--thinking` | Also run thinking-specific cases (`enable_thinking=True/False`) |

The script exits with code **0** if all cases pass, or **1** if any case fails.

## How It Works

The verifier simulates the pretokenized incremental tokenization path at the text level:

1. **Prefix render**: Render the first N messages with `add_generation_prompt=False`
2. **Full render**: Render all messages with `add_generation_prompt=True`
3. **Prefix check**: Verify that the full render starts with the prefix render
4. **Equivalence check**: Verify that `prefix + incremental == full`

This is tested across 12 diverse trajectory patterns covering single-turn, multi-turn, parallel tool calls, long chains, and no-tool scenarios.

## Autofix: Built-in Template Fixes

miles includes fixed templates for model families known to break the append-only invariant. When you pass `--chat-template-path autofix` in your training command, miles automatically selects the right fix:

| Model Pattern | Fixed Template |
| :--- | :--- |
| `Qwen3-*` (base, e.g. Qwen3-0.6B, Qwen3-4B) | `qwen3_fixed.jinja` |
| `Qwen3-*B-Thinking-2507` | `qwen3_thinking_2507_and_next_fixed.jinja` |
| `Qwen3-Next-*-Thinking` | `qwen3_thinking_2507_and_next_fixed.jinja` |

Models that are already append-only (e.g. Qwen3.5, GLM-5, Qwen3-Instruct-2507, Qwen3-Coder-Next) do not need a fix.

### Using autofix in training

```shell
python run.py \
    --hf-checkpoint Qwen/Qwen3-4B \
    --chat-template-path autofix \
    ...
```

## Running Tests

The verification logic is covered by comprehensive unit tests:

```shell
# Run all chat template tests (autofix mapping + append-only verification)
python -m pytest tests/fast/utils/chat_template_utils/ -v
```
