"""Coverage plan for the token trajectory collector; every test is a skeleton until WP-1 lands.

Reused, not reimplemented: ``tests/fast/tinker/harness.py`` (``make_service`` with its ``FakeBackend.sample``, and the
checkpoint writer that lays down sampler META.json through ``export_slot``); the only new fake is a character tokenizer.
"""

import pytest

SKELETON = pytest.mark.skip(reason="skeleton: implementation lands with the collector (WP-1)")


@SKELETON
def test_render_prompt_uses_generation_prompt():
    """render_prompt applies the HF chat template with add_generation_prompt=True and the configured kwargs."""


@SKELETON
def test_sample_payload_matches_tinker_sample():
    """to_sample_payload produces prompt_tokens, num_samples and sampling_params exactly as decode_sample_request does for Tinker sample; max_tokens is required; stop=[] is dropped."""


@SKELETON
def test_chat_samples_through_tinker_service():
    """chat() goes through TinkerService.submit_sample (FakeBackend.sample sees lora_path=M@V) and records input_ids as rendered, output_ids/logprobs as returned."""


@SKELETON
def test_auto_register_requires_bearer():
    """A new session id auto-registers with a valid bearer and raises UnknownSessionError without one."""


@SKELETON
def test_prebound_session_accepts_dummy_key():
    """After bind(), chat requests without a bearer are served with the owner's tenant and recorded."""


@SKELETON
def test_bound_session_rejects_other_model():
    """A request naming a different tinker:// path than the bound one gets a UserInputError (400)."""


@SKELETON
def test_get_and_delete_check_ownership():
    """trajectory()/delete() raise OwnershipError for another tenant's key and UnknownSessionError afterwards."""


@SKELETON
def test_unknown_sampler_path_is_user_error():
    """bind() with a tinker:// path lacking META.json is a UserInputError (resolve_sampler_checkpoint), another tenant's path an OwnershipError."""


@SKELETON
def test_backend_failure_records_no_turn():
    """A failed sampling future surfaces as an error and leaves the session without a half turn."""


@SKELETON
def test_sweep_expires_idle_sessions():
    """sweep() drops sessions idle longer than session_ttl_s and keeps the others."""


@SKELETON
def test_turns_round_trip_through_cookbook():
    """Two chained turns exported by GET /oai/sessions/{sid} become one Datum through turns_to_trajectory + trajectory_to_data; a broken prefix becomes two."""
