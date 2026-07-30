from pathlib import Path

import pytest

from miles.ray.tinker.protocol import TinkerError
from miles.ray.tinker.state import TinkerModelConfig, TinkerState


def _active_model(state: TinkerState):
    session = state.create_session(tags=[], user_metadata=None, sdk_version="test", project_id=None)
    config = TinkerModelConfig(
        rank=8,
        alpha=8,
        save=Path("/tmp/tinker-test-model"),
        seed=1,
        train_unembed=True,
        train_mlp=True,
        train_attn=True,
    )
    model, _future, _ = state.begin_model_create(
        session_id=session.session_id,
        model_seq_id=0,
        base_model="test/model",
        config=config,
        payload={"base_model": "test/model"},
    )
    model.status = "active"
    return model


def test_model_operations_are_ordered_and_idempotent():
    state = TinkerState()
    model = _active_model(state)
    future, operation = state.submit_model_operation(
        model_id=model.model_id,
        seq_id=1,
        kind="forward",
        payload={"value": 1},
    )
    duplicate, duplicate_operation = state.submit_model_operation(
        model_id=model.model_id,
        seq_id=1,
        kind="forward",
        payload={"value": 1},
    )

    assert duplicate.request_id == future.request_id
    assert operation is not None
    assert duplicate_operation is None

    with pytest.raises(TinkerError, match="different request"):
        state.submit_model_operation(
            model_id=model.model_id,
            seq_id=1,
            kind="forward",
            payload={"value": 2},
        )
    with pytest.raises(TinkerError, match="expected seq_id 2"):
        state.submit_model_operation(
            model_id=model.model_id,
            seq_id=3,
            kind="forward",
            payload={"value": 3},
        )


def test_checkpoint_paths_are_sdk_uri_and_megatron_step_dir():
    state = TinkerState()
    model = _active_model(state)

    checkpoint = state.allocate_checkpoint(
        model_id=model.model_id,
        seq_id=1,
        requested_name="checkpoint-one",
        checkpoint_type="training",
        ttl_seconds=None,
        overwrite=False,
    )
    duplicate = state.allocate_checkpoint(
        model_id=model.model_id,
        seq_id=1,
        requested_name="checkpoint-one",
        checkpoint_type="training",
        ttl_seconds=None,
        overwrite=False,
    )

    assert checkpoint.tinker_path == f"tinker://{model.model_id}/weights/checkpoint-one"
    assert checkpoint.local_path == model.config.save / "checkpoints" / "step_1"
    assert duplicate is checkpoint
    with pytest.raises(TinkerError, match="is pending"):
        state.require_checkpoint(checkpoint.tinker_path)

    state.complete_checkpoint(checkpoint.tinker_path)
    assert state.require_checkpoint(checkpoint.tinker_path) is checkpoint


def test_checkpoint_validation_does_not_advance_or_allocate_on_bad_sequence():
    state = TinkerState()
    model = _active_model(state)

    with pytest.raises(TinkerError, match="expected seq_id 1"):
        state.validate_model_operation(
            model_id=model.model_id,
            seq_id=2,
            kind="save_weights",
            payload={"path": "bad"},
        )

    assert model.next_seq_id == 1
    assert state.checkpoints == {}


def test_model_create_can_be_rolled_back_after_registration_failure():
    state = TinkerState()
    session = state.create_session(tags=[], user_metadata=None, sdk_version="test", project_id=None)
    config = TinkerModelConfig(
        rank=8,
        alpha=8,
        save=Path("/tmp/tinker-test-model"),
        seed=1,
        train_unembed=True,
        train_mlp=True,
        train_attn=True,
    )
    model, future, _ = state.begin_model_create(
        session_id=session.session_id,
        model_seq_id=0,
        base_model="test/model",
        config=config,
        payload={"base_model": "test/model"},
    )

    state.rollback_model_create(model.model_id, future.request_id)

    assert model.model_id not in state.models
    assert future.request_id not in state.futures
    assert (session.session_id, 0) not in state.model_create_keys


def test_sampling_sequences_start_at_zero_and_are_idempotent():
    state = TinkerState()
    session = state.create_session(tags=[], user_metadata=None, sdk_version="test", project_id=None)
    sampling, _ = state.create_sampling_session(
        session_id=session.session_id,
        sampling_session_seq_id=0,
        base_model="test/model",
        model_path=None,
        adapter_path=None,
        adapter_name=None,
        payload={"base_model": "test/model"},
    )

    future, operation = state.submit_sample(
        sampling_session_id=sampling.sampling_session_id,
        seq_id=0,
        payload={"prompt": [1]},
    )
    duplicate, duplicate_operation = state.submit_sample(
        sampling_session_id=sampling.sampling_session_id,
        seq_id=0,
        payload={"prompt": [1]},
    )

    assert operation is not None
    assert operation.payload["seq_id"] == 0
    assert duplicate.request_id == future.request_id
    assert duplicate_operation is None
