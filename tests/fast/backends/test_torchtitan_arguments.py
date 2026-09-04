"""The torchtitan backend's argument gate.

These rejections are load-bearing rather than cosmetic: each one names a
combination whose failure mode downstream is either a hard crash deep inside
torchtitan or a run whose numbers are quietly wrong (CP-sharded logits fed to
a loss that assumes full sequences). Failing at startup is the difference
between a clear error and either of those.
"""

from argparse import Namespace

import pytest
import torch

from miles.backends.fsdp_utils.arguments import FSDPArgs
from miles.backends.torchtitan_utils.arguments import TorchtitanArgs, validate_torchtitan_args


def _args(**overrides) -> Namespace:
    base = dict(
        titan_attn_backend="flex",
        titan_model_name="qwen3",
        titan_seq_len=8192,
        titan_pipeline_parallel_degree=1,
        titan_context_parallel_degree=1,
        titan_expert_parallel_degree=1,
        rollout_max_context_len=8192,
        rollout_max_response_len=4096,
        ref_update_interval=None,
        save_debug_train_data=None,
    )
    return Namespace(**{**base, **overrides})


def test_the_supported_configuration_passes(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    validate_torchtitan_args(_args())


def test_sdpa_is_rejected_outright(monkeypatch):
    """The pinned torchtitan removed sdpa for language models; offering it here
    would fail deep inside the model registry instead."""
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="sdpa"):
        validate_torchtitan_args(_args(titan_attn_backend="sdpa"))


@pytest.mark.parametrize(
    ("backend", "needed"),
    [("flex", "2.13"), ("flex_flash", "2.13"), ("varlen", "2.12")],
)
def test_attention_backends_carry_their_own_torch_threshold(monkeypatch, backend, needed):
    """flex and varlen do not unblock at the same torch version: varlen_attn(enable_gqa=)
    is public from 2.12, create_block_mask(separate_full_blocks=) only from 2.13. Quoting
    one threshold for both is how a 2.12 bump would drop the gate and break flex."""
    monkeypatch.setattr(torch, "__version__", "2.11.0")
    with pytest.raises(ValueError, match=f"torch>={needed}"):
        validate_torchtitan_args(_args(titan_attn_backend=backend))


def test_varlen_passes_on_212_while_flex_still_needs_213(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.12.0")
    validate_torchtitan_args(_args(titan_attn_backend="varlen"))
    with pytest.raises(ValueError, match="torch>=2.13"):
        validate_torchtitan_args(_args(titan_attn_backend="flex"))


def test_pipeline_parallelism_is_accepted(monkeypatch):
    """The PP schedule lives behind the engine's forward_backward_step; nothing
    in the shared loop conditions on it, so nothing rejects it here. (torchtitan
    itself rejects PP for weight-tied flavors at build time.)"""
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    validate_torchtitan_args(_args(titan_pipeline_parallel_degree=2))


def test_context_and_expert_parallelism_are_accepted(monkeypatch):
    """Both stay inside the trainer: it shards the sequence and gathers the
    logits back before the loss, and the expert axis only changes where a
    parameter lives. The shared loop is told cp=1 either way."""
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    validate_torchtitan_args(_args(titan_context_parallel_degree=2))
    validate_torchtitan_args(_args(titan_expert_parallel_degree=2))


def test_context_parallelism_is_rejected_for_qwen3_5(monkeypatch):
    """Upstream needs the whole sequence for GatedDeltaNet, so torchtitan itself
    refuses -- deep inside the model build rather than here."""
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="context parallelism"):
        validate_torchtitan_args(_args(titan_model_name="qwen3_5", titan_context_parallel_degree=2))


def test_a_sequence_longer_than_the_rotary_tables_is_rejected(monkeypatch):
    """torchtitan sizes its rotary embeddings from --titan-seq-len; a longer
    prompt-plus-response indexes past them and asserts inside the rope kernel,
    pointing at rope rather than at the setting."""
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="titan-seq-len"):
        validate_torchtitan_args(_args(titan_seq_len=8192, rollout_max_context_len=16384))


def test_without_a_context_bound_the_response_must_leave_room_for_a_prompt(monkeypatch):
    """--rollout-max-context-len is optional, and then the prompt length is
    unknown -- but a sequence is at least one prompt token plus the response, so
    a seq_len merely equal to the response length already overflows."""
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="no room for a prompt"):
        validate_torchtitan_args(
            _args(titan_seq_len=8192, rollout_max_context_len=None, rollout_max_response_len=8192)
        )
    validate_torchtitan_args(
        _args(titan_seq_len=16384, rollout_max_context_len=None, rollout_max_response_len=8192)
    )


def test_periodic_reference_refresh_is_rejected_rather_than_ignored(monkeypatch):
    """The reference model is built once from --ref-load; the actor-to-ref copy FSDP
    does on an interval is not wired up, and ignoring the flag would train against a
    reference the user believes is being refreshed."""
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="ref-update-interval"):
        validate_torchtitan_args(_args(ref_update_interval=4))


def test_debug_train_dump_is_rejected_rather_than_ignored(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    with pytest.raises(ValueError, match="save-debug-train-data"):
        validate_torchtitan_args(_args(save_debug_train_data="/tmp/dump"))


def test_args_extend_rather_than_restate_the_common_training_options():
    """FSDPArgs holds the non-Megatron training options (optimizer, LR schedule,
    precision); torchtitan inherits them so the two cannot drift apart."""
    assert issubclass(TorchtitanArgs, FSDPArgs)
    common = {"lr", "adam_beta1", "adam_beta2", "adam_eps", "weight_decay", "fp16"}
    assert common <= set(TorchtitanArgs.__dataclass_fields__)


def test_titan_specific_fields_are_namespaced():
    titan_only = set(TorchtitanArgs.__dataclass_fields__) - set(FSDPArgs.__dataclass_fields__)
    assert titan_only and all(name.startswith("titan_") for name in titan_only)
