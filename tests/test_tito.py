"""Tests for Token-In/Token-Out (TITO) generate behavior.

These tests verify the --token-io-mode flag behavior:
- token_out: require engine token IDs (token in, token out)
- retokenize: legacy behavior (token in, text out, then retokenize)
"""

import pytest
from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

from miles.utils.types import Sample


def make_args(**overrides):
    """Create a minimal args namespace for testing."""
    defaults = {
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_port": 8000,
        "token_io_mode": "retokenize",
        "use_miles_router": False,
        "miles_router_middleware_paths": [],
        "use_rollout_routing_replay": False,
        "sglang_speculative_algorithm": None,
        "partial_rollout": False,
        "apply_chat_template_kwargs": {},
        "hf_checkpoint": "test-model",
        "use_tis": False,
        "use_rollout_logprobs": False,
        "get_mismatch_metrics": False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def make_sample(**overrides):
    """Create a minimal sample for testing."""
    sample = Sample(prompt="test prompt", label="test label", index=0)
    sample.response = ""
    sample.tokens = []
    sample.response_length = 0
    sample.rollout_log_probs = None
    sample.weight_versions = []
    sample.status = Sample.Status.PENDING
    for k, v in overrides.items():
        setattr(sample, k, v)
    return sample


def make_sglang_response(output_ids=None, output_token_logprobs=None, finish_reason="stop", text="generated text"):
    """Create a mock SGLang /generate response."""
    meta_info = {"finish_reason": {"type": finish_reason}}
    if output_token_logprobs is not None:
        meta_info["output_token_logprobs"] = output_token_logprobs
    response = {"text": text, "meta_info": meta_info}
    if output_ids is not None:
        response["output_ids"] = output_ids
    return response


class TestTokenOutMode:
    """Tests for --token-io-mode=token_out."""

    @pytest.mark.asyncio
    async def test_strict_requires_token_ids(self):
        """Strict mode should error when no token IDs are available."""
        from miles.rollout.sglang_rollout import generate

        args = make_args(token_io_mode="token_out")
        sample = make_sample()

        # Response with no output_ids and no output_token_logprobs
        response = make_sglang_response(output_ids=None, output_token_logprobs=None)

        with patch("miles.rollout.sglang_rollout.GenerateState") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.tokenizer = MagicMock()
            mock_state.processor = MagicMock()
            mock_state_cls.return_value = mock_state

            with patch("miles.rollout.sglang_rollout.prepare_model_inputs") as mock_prepare:
                mock_prepare.return_value = ([1, 2, 3], {"images": [], "videos": []})

                with patch("miles.rollout.sglang_rollout.post", new_callable=AsyncMock) as mock_post:
                    mock_post.return_value = response

                    with pytest.raises(RuntimeError, match="token_out mode requires engine token IDs"):
                        await generate(args, sample, {"max_new_tokens": 100})

    @pytest.mark.asyncio
    async def test_token_out_errors_on_mismatch(self):
        """token_out mode should error when output_ids and output_token_logprobs disagree."""
        from miles.rollout.sglang_rollout import generate

        args = make_args(token_io_mode="token_out")
        sample = make_sample()

        # Response with mismatched token IDs
        output_ids = [10, 20, 30]
        output_token_logprobs = [(-1.0, 10), (-2.0, 21), (-3.0, 30)]  # token 21 != 20

        response = make_sglang_response(
            output_ids=output_ids, output_token_logprobs=output_token_logprobs
        )

        with patch("miles.rollout.sglang_rollout.GenerateState") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.tokenizer = MagicMock()
            mock_state.processor = MagicMock()
            mock_state_cls.return_value = mock_state

            with patch("miles.rollout.sglang_rollout.prepare_model_inputs") as mock_prepare:
                mock_prepare.return_value = ([1, 2, 3], {"images": [], "videos": []})

                with patch("miles.rollout.sglang_rollout.post", new_callable=AsyncMock) as mock_post:
                    mock_post.return_value = response

                    with pytest.raises(RuntimeError, match="mismatch between output_ids and output_token_logprobs"):
                        await generate(args, sample, {"max_new_tokens": 100})

    @pytest.mark.asyncio
    async def test_token_out_succeeds_with_matching_ids(self):
        """token_out mode should succeed when output_ids and output_token_logprobs match."""
        from miles.rollout.sglang_rollout import generate

        args = make_args(token_io_mode="token_out")
        sample = make_sample()

        output_ids = [10, 20, 30]
        output_token_logprobs = [(-1.0, 10), (-2.0, 20), (-3.0, 30)]

        response = make_sglang_response(
            output_ids=output_ids, output_token_logprobs=output_token_logprobs
        )

        with patch("miles.rollout.sglang_rollout.GenerateState") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.tokenizer = MagicMock()
            mock_state.processor = MagicMock()
            mock_state_cls.return_value = mock_state

            with patch("miles.rollout.sglang_rollout.prepare_model_inputs") as mock_prepare:
                mock_prepare.return_value = ([1, 2, 3], {"images": [], "videos": []})

                with patch("miles.rollout.sglang_rollout.post", new_callable=AsyncMock) as mock_post:
                    mock_post.return_value = response

                    result = await generate(args, sample, {"max_new_tokens": 100})

                    assert result.tokens == [1, 2, 3, 10, 20, 30]
                    assert result.response_length == 3
                    assert result.rollout_log_probs == [-1.0, -2.0, -3.0]

    @pytest.mark.asyncio
    async def test_token_out_uses_logprobs_when_output_ids_missing(self):
        """token_out mode should use output_token_logprobs when output_ids is missing."""
        from miles.rollout.sglang_rollout import generate

        args = make_args(token_io_mode="token_out")
        sample = make_sample()

        output_token_logprobs = [(-1.0, 10), (-2.0, 20), (-3.0, 30)]

        response = make_sglang_response(
            output_ids=None, output_token_logprobs=output_token_logprobs
        )

        with patch("miles.rollout.sglang_rollout.GenerateState") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.tokenizer = MagicMock()
            mock_state.processor = MagicMock()
            mock_state_cls.return_value = mock_state

            with patch("miles.rollout.sglang_rollout.prepare_model_inputs") as mock_prepare:
                mock_prepare.return_value = ([1, 2, 3], {"images": [], "videos": []})

                with patch("miles.rollout.sglang_rollout.post", new_callable=AsyncMock) as mock_post:
                    mock_post.return_value = response

                    result = await generate(args, sample, {"max_new_tokens": 100})

                    assert result.tokens == [1, 2, 3, 10, 20, 30]
                    assert result.rollout_log_probs == [-1.0, -2.0, -3.0]


class TestRetokenizeMode:
    """Tests for --token-io-mode=retokenize."""

    @pytest.mark.asyncio
    async def test_retokenize_uses_retrieve_from_text(self):
        """retokenize mode should use retrieve_from_text when router is enabled."""
        from miles.rollout.sglang_rollout import generate

        args = make_args(
            token_io_mode="retokenize",
            use_miles_router=True,
            miles_router_middleware_paths=["RadixTreeMiddleware"],
        )
        sample = make_sample()

        # Response with no token IDs
        response = make_sglang_response(output_ids=None, output_token_logprobs=None)
        retrieve_response = {
            "tokens": [1, 2, 3, 10, 20, 30],
            "loss_mask": [0, 0, 0, 1, 1, 1],
            "rollout_logp": [0.0, 0.0, 0.0, -1.0, -2.0, -3.0],
        }

        with patch("miles.rollout.sglang_rollout.GenerateState") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.tokenizer = MagicMock()
            mock_state.processor = MagicMock()
            mock_state_cls.return_value = mock_state

            with patch("miles.rollout.sglang_rollout.prepare_model_inputs") as mock_prepare:
                mock_prepare.return_value = ([1, 2, 3], {"images": [], "videos": []})

                with patch("miles.rollout.sglang_rollout.post", new_callable=AsyncMock) as mock_post:
                    mock_post.side_effect = [response, retrieve_response]

                    with patch("miles.rollout.sglang_rollout.get_response_lengths") as mock_lengths:
                        mock_lengths.return_value = [3]

                        result = await generate(args, sample, {"max_new_tokens": 100})

                        assert result.tokens == [1, 2, 3, 10, 20, 30]
                        assert mock_post.call_count == 2  # generate + retrieve_from_text

    @pytest.mark.asyncio
    async def test_retokenize_uses_logprobs_without_router(self):
        """retokenize mode should use output_token_logprobs when router is not enabled."""
        from miles.rollout.sglang_rollout import generate

        args = make_args(token_io_mode="retokenize", use_miles_router=False)
        sample = make_sample()

        output_token_logprobs = [(-1.0, 10), (-2.0, 20), (-3.0, 30)]

        response = make_sglang_response(
            output_ids=None, output_token_logprobs=output_token_logprobs
        )

        with patch("miles.rollout.sglang_rollout.GenerateState") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.tokenizer = MagicMock()
            mock_state.processor = MagicMock()
            mock_state_cls.return_value = mock_state

            with patch("miles.rollout.sglang_rollout.prepare_model_inputs") as mock_prepare:
                mock_prepare.return_value = ([1, 2, 3], {"images": [], "videos": []})

                with patch("miles.rollout.sglang_rollout.post", new_callable=AsyncMock) as mock_post:
                    mock_post.return_value = response

                    result = await generate(args, sample, {"max_new_tokens": 100})

                    assert result.tokens == [1, 2, 3, 10, 20, 30]
                    assert result.rollout_log_probs == [-1.0, -2.0, -3.0]


class TestListResponseHandling:
    """Tests for handling list responses (n>1)."""

    @pytest.mark.asyncio
    async def test_rejects_multiple_completions(self):
        """Should error when response is a list with multiple completions."""
        from miles.rollout.sglang_rollout import generate

        args = make_args(token_io_mode="retokenize")
        sample = make_sample()

        # Response as a list with multiple items
        response = [
            make_sglang_response(output_ids=[10, 20], output_token_logprobs=[(-1.0, 10), (-2.0, 20)]),
            make_sglang_response(output_ids=[30, 40], output_token_logprobs=[(-1.0, 30), (-2.0, 40)]),
        ]

        with patch("miles.rollout.sglang_rollout.GenerateState") as mock_state_cls:
            mock_state = MagicMock()
            mock_state.tokenizer = MagicMock()
            mock_state.processor = MagicMock()
            mock_state_cls.return_value = mock_state

            with patch("miles.rollout.sglang_rollout.prepare_model_inputs") as mock_prepare:
                mock_prepare.return_value = ([1, 2, 3], {"images": [], "videos": []})

                with patch("miles.rollout.sglang_rollout.post", new_callable=AsyncMock) as mock_post:
                    mock_post.return_value = response

                    with pytest.raises(RuntimeError, match="expects a single completion"):
                        await generate(args, sample, {"max_new_tokens": 100})
