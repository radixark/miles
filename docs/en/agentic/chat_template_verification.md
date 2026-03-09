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

  [FAIL] single_tool-N3                -- Prefix mismatch!
  [PASS] single_tool-N3-no_tools
  [FAIL] multi_turn-N4                 -- Prefix mismatch!
  [FAIL] multi_tool_single_turn-N3     -- Prefix mismatch!
  [FAIL] parallel_tools-N3             -- Prefix mismatch!
  [FAIL] long_chain-N4                 -- Prefix mismatch!
  [FAIL] long_chain-N6                 -- Prefix mismatch!
  [FAIL] multi_user_tool_chain-N8      -- Prefix mismatch!
  [PASS] simple_no_tool-N3-no_tools
  [FAIL] retry_system-N3               -- Prefix mismatch!
  [FAIL] retry_system-N5               -- Prefix mismatch!
  [FAIL] intermediate_system-N5        -- Prefix mismatch!
  [FAIL] intermediate_system-N8        -- Prefix mismatch!

Results: 2/13 passed, 11 failed

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
  [PASS] single_tool-N3-no_tools
  [PASS] multi_turn-N4
  ...
  [PASS] intermediate_system-N8

Results: 13/13 passed, 0 failed

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
python scripts/tools/verify_chat_template.py --model Qwen/Qwen3.5-0.8B --autofix --thinking
```

This runs 29 cases in total (13 standard + 16 thinking with `enable_thinking=True/False`).

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

1. **Prefix render**: Render the first N messages with `add_generation_prompt=False` → `prefix_text`
2. **Full render**: Render all messages with `add_generation_prompt=True` → `full_text`
3. **Prefix check**: Verify that `full_text` starts with `prefix_text`
4. **Equivalence check**: Verify that the full render from the pretokenized path equals the standard full render

This is tested across 13 diverse cases covering 9 trajectory patterns (single-turn, multi-turn, parallel tool calls, long chains, multi-user tool chain, no-tool, retry with intermediate system, and multi-step intermediate system scenarios).

When `--thinking` is enabled, an additional 16 cases are added: 6 thinking-specific trajectory patterns × 2 (`enable_thinking=True/False`) + 2 intermediate system thinking patterns × 2.

## Autofix: Built-in Template Fixes

miles includes fixed templates for model families known to break the append-only invariant. When you pass `--chat-template-path autofix` in your training command, miles automatically selects the right fix:

| Model Pattern | Fixed Template |
| :--- | :--- |
| `Qwen3.5-*` (e.g. Qwen3.5-0.8B, Qwen3.5-32B) | `qwen3.5_fixed.jinja` |
| `Qwen3-Next-*-Thinking` | `qwen3_thinking_2507_and_next_fixed.jinja` |
| `Qwen3-*B-Thinking-2507` | `qwen3_thinking_2507_and_next_fixed.jinja` |
| `Qwen3-*` (base, e.g. Qwen3-0.6B, Qwen3-4B) | `qwen3_fixed.jinja` |

Rules are matched in order (first match wins), so more specific patterns take priority over the general `Qwen3-*` rule.

Models that are already append-only (e.g. GLM-5, GLM-4, GLM-4.7-Flash, Qwen3-Instruct-2507, Qwen3-Next-Instruct, Qwen3-Coder-Next) do not need a fix.

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
