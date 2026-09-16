"""Coverage plan for the token trajectory collector; every test is a skeleton until WP-1 lands."""

import pytest

SKELETON = pytest.mark.skip(reason="skeleton: implementation lands with the collector (WP-1)")


@SKELETON
def test_render_prompt_uses_generation_prompt():
    """render_prompt applies the HF chat template with add_generation_prompt=True and the configured kwargs."""


@SKELETON
def test_sample_payload_matches_tinker_sample():
    """build_sample_payload produces prompt_tokens, num_samples and sampling_params exactly as Tinker sample does."""


@SKELETON
def test_chat_records_turn_with_engine_ids():
    """chat() records input_ids as rendered and output_ids/logprobs as returned by the fake backend, untouched."""


@SKELETON
def test_prefix_ok_true_when_turn_extends_previous():
    """compute_prefix_ok is True when input_ids starts with the previous input_ids + output_ids, False otherwise."""


@SKELETON
def test_auto_register_requires_bearer():
    """A new session id auto-registers with a valid bearer and is rejected without one."""


@SKELETON
def test_prebound_session_accepts_dummy_key():
    """After POST /oai/sessions/{sid}, chat requests without a bearer are served and recorded."""


@SKELETON
def test_bound_session_rejects_other_model():
    """A request naming a different tinker:// path than the bound one gets a UserInputError (400)."""


@SKELETON
def test_get_and_delete_check_ownership():
    """trajectory()/delete() raise OwnershipError for another tenant's key."""


@SKELETON
def test_unknown_sampler_path_is_user_error():
    """A tinker:// path with no META.json is a UserInputError, not a server failure."""


@SKELETON
def test_separate_reasoning_splits_think_block():
    """separate_reasoning=true moves a leading <think>...</think> block into reasoning_content."""


@SKELETON
def test_fake_stream_emits_single_chunk_and_done():
    """stream=true yields one SSE chunk with the whole message followed by data: [DONE]."""


@SKELETON
def test_sweep_expires_idle_sessions():
    """sweep() drops sessions idle longer than session_ttl_s and keeps the others."""
