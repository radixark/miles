"""Running tinker-cookbook's rl/train.py loop unchanged against the gateway: the requests it sends that plain Tinker examples do not.

Reused, not reimplemented: ``harness.make_service`` / ``await_settled`` and ``oai_fakes.write_sampler`` (the real checkpoint writer).
"""

import pytest
from tests.fast.tinker.harness import await_settled, make_service
from tests.fast.tinker.oai_fakes import TENANT, write_sampler

from miles.tinker.core.future import DONE
from miles.tinker.core.types import UserInputError
from miles.tinker.server.encoding import decode_command, validate_create_model

COOKBOOK_TTL = 604_800  # tinker_cookbook.rl.train.Config.ttl_seconds default; CheckpointManager sends it on every save


def test_ttl_seconds_is_accepted_and_ignored():
    """save_weights_for_sampler / save_state with ttl_seconds (the cookbook's 7-day default) decode without a validation_error."""
    op, decoded = decode_command(
        "save_state", {"model_id": "m", "seq_id": 1, "path": "ck", "ttl_seconds": COOKBOOK_TTL}
    )
    assert (op, decoded) == ("save_state", {"model_id": "m", "seq_id": 1, "name": "ck", "overwrite": False})

    op, decoded = decode_command(
        "save_weights_for_sampler", {"model_id": "m", "seq_id": 2, "path": "s", "ttl_seconds": COOKBOOK_TTL}
    )
    assert (op, decoded) == ("save_weights_for_sampler", {"model_id": "m", "seq_id": 2, "sampler_path": "s"})

    _, decoded = decode_command("save_state", {"model_id": "m", "seq_id": 3, "ttl_seconds": None})
    assert "validation_error" not in decoded


def _cookbook_create_payload(service, session_id: str) -> dict:
    """What tinker 0.26.2's create_lora_training_client(model_name, rank=16, user_metadata=...) puts on the wire (rl/train.py)."""
    return {
        "session_id": session_id,
        "model_seq_id": 0,
        "base_model": service.config.base_model,
        "user_metadata": {"recipe": "harbor-tinker"},
        "lora_config": {"rank": 16, "seed": None, "train_unembed": True, "train_mlp": True, "train_attn": True},
    }


async def test_create_model_accepts_cookbook_defaults(tmp_path):
    """create_lora_training_client(model_name, rank, user_metadata) with seed=None and train_*=True is accepted when the server layout trains attn, mlp and unembed."""
    service = make_service(tmp_path / "unembed", trains_unembed=True)
    payload = _cookbook_create_payload(service, service.create_session(TENANT))
    validate_create_model(payload)
    request_id, model_id = service.create_model(TENANT, payload)
    future = await await_settled(service, TENANT, request_id)
    assert future.state == DONE and service.models[model_id].lora_rank == 16

    # the --target-modules default leaves lm_head frozen; the SDK's train_unembed=True default is then a clear 400,
    # so a cookbook run needs the gateway started with lm_head in --target-modules
    default_layout = make_service(tmp_path / "default")
    with pytest.raises(UserInputError, match="train_unembed"):
        default_layout.create_model(
            TENANT, _cookbook_create_payload(default_layout, default_layout.create_session(TENANT))
        )


def test_weights_info_carries_what_resume_needs(tmp_path):
    """weights_info returns base_model, is_lora, lora_rank and train_* so create_training_client_from_state_with_optimizer can rebuild the client."""
    service = make_service(tmp_path)
    write_sampler(tmp_path, service.config.base_model, kind="weights", name="ck")
    assert service.weights_info(TENANT, "tinker://m1/weights/ck") == {
        "base_model": service.config.base_model,
        "is_lora": True,
        "lora_rank": 8,
        "train_attn": True,
        "train_mlp": True,
        "train_unembed": False,
    }
