"""Unit tests for try_get_fixed_chat_template mapping logic.

hf_checkpoint is often a local path with user-customized naming rather than a
clean HuggingFace repo ID, so we test a wide variety of realistic (and weird)
path formats to make sure glob matching is robust.
"""

import os

import pytest

from miles.utils.chat_template_utils import TEMPLATE_DIR, try_get_fixed_chat_template

_QWEN3_FIXED = str(TEMPLATE_DIR / "qwen3_fixed.jinja")
_THINKING_FIXED = str(TEMPLATE_DIR / "qwen3_thinking_2507_and_next_fixed.jinja")


# ---------------------------------------------------------------------------
# Qwen3 base models → qwen3_fixed.jinja
# ---------------------------------------------------------------------------


class TestQwen3Mapping:
    """Qwen3 LLM models (non-Thinking) should map to qwen3_fixed.jinja."""

    @pytest.mark.parametrize(
        "checkpoint",
        [
            # Standard HF repo IDs
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen3-4B",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-14B",
            "Qwen/Qwen3-32B",
            "Qwen/Qwen3-235B-A22B",
            # Local absolute paths
            "/data/models/Qwen3-4B",
            "/home/user/checkpoints/Qwen3-8B-epoch3",
            "/mnt/nfs/shared_models/Qwen3-32B",
            # Local relative paths
            "models/Qwen3-14B",
            "./downloaded/Qwen3-0.6B-finetuned",
            # User renamed / suffixed
            "Qwen3-4B-sft-v2",
            "Qwen3-8B_merged_lora",
            "/data/Qwen3-235B-A22B-quantized",
            # Lowercase / odd casing
            "qwen/qwen3-4b",
            "QWEN/QWEN3-32B",
            "Qwen/qWeN3-14b",
        ],
    )
    def test_resolves_to_qwen3_fixed(self, checkpoint):
        path = try_get_fixed_chat_template(checkpoint)
        assert path is not None, f"{checkpoint} should match a fix rule"
        assert path == _QWEN3_FIXED
        assert os.path.isfile(path), f"Fixed template must exist: {path}"


# ---------------------------------------------------------------------------
# Qwen3-Thinking-2507 → qwen3_thinking_2507_and_next_fixed.jinja
# ---------------------------------------------------------------------------


class TestQwen3Thinking2507Mapping:
    """Qwen3-Thinking-2507 models should map to the shared thinking fix."""

    @pytest.mark.parametrize(
        "checkpoint",
        [
            # Standard HF repo IDs
            "Qwen/Qwen3-4B-Thinking-2507",
            "Qwen/Qwen3-8B-Thinking-2507",
            "Qwen/Qwen3-32B-Thinking-2507",
            # Local paths
            "/data/models/Qwen3-4B-Thinking-2507",
            "/home/user/Qwen3-32B-Thinking-2507-sft",
            "models/Qwen3-8B-Thinking-2507-merged",
            # Weird casing
            "qwen/qwen3-4b-thinking-2507",
            "QWEN/QWEN3-8B-THINKING-2507",
        ],
    )
    def test_resolves_to_thinking_fixed(self, checkpoint):
        path = try_get_fixed_chat_template(checkpoint)
        assert path is not None, f"{checkpoint} should match a fix rule"
        assert path == _THINKING_FIXED
        assert os.path.isfile(path), f"Fixed template must exist: {path}"


# ---------------------------------------------------------------------------
# Qwen3-Next → Thinking gets fix, Instruct does NOT
# ---------------------------------------------------------------------------


class TestQwen3NextMapping:
    """Qwen3-Next Thinking models reuse the 2507 fix; Instruct needs no fix."""

    @pytest.mark.parametrize(
        "checkpoint",
        [
            "Qwen/Qwen3-Next-80B-A3B-Thinking",
            "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
            # Local path variants
            "/data/Qwen3-Next-80B-A3B-Thinking",
            "models/qwen3-next-80b-a3b-thinking-sft",
            "/mnt/nfs/Qwen3-Next-80B-A3B-Thinking-epoch5",
        ],
    )
    def test_thinking_resolves_to_fix(self, checkpoint):
        path = try_get_fixed_chat_template(checkpoint)
        assert path is not None, f"{checkpoint} should match a fix rule"
        assert path == _THINKING_FIXED

    @pytest.mark.parametrize(
        "checkpoint",
        [
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
            "/data/models/Qwen3-Next-80B-A3B-Instruct",
            "models/qwen3-next-80b-a3b-instruct-lora",
        ],
    )
    def test_instruct_no_fix(self, checkpoint):
        path = try_get_fixed_chat_template(checkpoint)
        assert path is None, f"{checkpoint} should NOT match any fix rule"


# ---------------------------------------------------------------------------
# Models that should never match
# ---------------------------------------------------------------------------


class TestNoMatch:
    """Models outside fix coverage should return None."""

    @pytest.mark.parametrize(
        "checkpoint",
        [
            # Other Qwen family
            "Qwen/Qwen3.5-0.8B",
            "Qwen/Qwen2.5-72B",
            "/data/models/Qwen3.5-32B-sft",
            # Non-Qwen
            "zai-org/GLM-5",
            "meta-llama/Llama-3.1-8B",
            "deepseek-ai/DeepSeek-V3",
            "/home/user/models/llama-3-70b",
            # Edge cases that look similar but shouldn't match
            "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "some-org/not-qwen3-at-all-4B",
        ],
    )
    def test_returns_none(self, checkpoint):
        assert try_get_fixed_chat_template(checkpoint) is None


# ---------------------------------------------------------------------------
# Case insensitivity
# ---------------------------------------------------------------------------


class TestCaseInsensitive:
    """Pattern matching should be case-insensitive."""

    @pytest.mark.parametrize(
        "checkpoint",
        [
            "QWEN/QWEN3-4B",
            "Qwen/qWeN3-4b",
            "qwen/QWEN3-14B",
            "/DATA/MODELS/QWEN3-8B",
            "QWEN/QWEN3-4B-THINKING-2507",
            "qwen/qwen3-next-80b-a3b-thinking",
        ],
    )
    def test_always_matches(self, checkpoint):
        assert try_get_fixed_chat_template(checkpoint) is not None


# ---------------------------------------------------------------------------
# Rule priority: specific patterns must match before general ones
# ---------------------------------------------------------------------------


class TestRulePriority:
    """More specific rules must match before general ones."""

    def test_thinking_2507_before_generic_qwen3(self):
        path = try_get_fixed_chat_template("Qwen/Qwen3-4B-Thinking-2507")
        assert path == _THINKING_FIXED

    def test_next_thinking_before_generic_qwen3(self):
        path = try_get_fixed_chat_template("Qwen/Qwen3-Next-80B-A3B-Thinking")
        assert path == _THINKING_FIXED

    def test_generic_qwen3_gets_qwen3_fix(self):
        path = try_get_fixed_chat_template("Qwen/Qwen3-4B")
        assert path == _QWEN3_FIXED

    def test_local_path_thinking_still_specific(self):
        path = try_get_fixed_chat_template("/data/train/Qwen3-32B-Thinking-2507-sft-v3")
        assert path == _THINKING_FIXED

    def test_local_path_base_still_generic(self):
        path = try_get_fixed_chat_template("/data/train/Qwen3-32B-sft-v3")
        assert path == _QWEN3_FIXED
