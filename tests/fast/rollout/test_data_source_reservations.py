import copy
import json
from argparse import Namespace
from pathlib import Path
from typing import cast

import pytest

import miles.rollout.data_source as data_source_module
from miles.rollout.data_source import (
    RolloutDataSource,
    RolloutDataSourceWithBuffer,
    SourceReservation,
    SourceReservationId,
)
from miles.utils.types import Sample


@pytest.fixture(autouse=True)
def patch_processors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_source_module, "load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr(data_source_module, "load_processor", lambda *args, **kwargs: None)


def _source_args(tmp_path: Path, *, rollout_shuffle: bool = False) -> Namespace:
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "\n".join(json.dumps({"prompt": prompt}) for prompt in ("alpha", "bravo", "charlie", "delta")),
        encoding="utf-8",
    )
    return Namespace(
        rollout_global_dataset=True,
        hf_checkpoint="unused",
        chat_template_path=None,
        dump_details=None,
        prompt_data=str(prompt_path),
        rollout_max_prompt_len=None,
        input_key="prompt",
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_seed=100,
        rollout_shuffle=rollout_shuffle,
        n_samples_per_prompt=2,
        save=str(tmp_path),
        load=str(tmp_path),
        save_interval=1,
        save_trigger_sentinel=None,
        buffer_filter_path=None,
    )


def _reservation(reservation_id: int, *, prompt: str) -> SourceReservation:
    first_sample_index = reservation_id * 2
    return SourceReservation(
        reservation_id=SourceReservationId(str(reservation_id)),
        samples=tuple(
            Sample(group_index=reservation_id, index=sample_index, prompt=prompt)
            for sample_index in (first_sample_index, first_sample_index + 1)
        ),
    )


def test_reserve_returns_exact_pristine_groups(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path))

    assert source.reserve_samples(3) == [
        _reservation(0, prompt="alpha"),
        _reservation(1, prompt="bravo"),
        _reservation(2, prompt="charlie"),
    ]


def test_reservation_keeps_every_parent_slot(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path))
    [reservation] = source.reserve_samples(1)

    with pytest.raises(AttributeError, match="'tuple' object has no attribute 'pop'"):
        cast(list[Sample], reservation.samples).pop()

    source.acknowledge_reservations([reservation], rollout_id=0)


@pytest.mark.parametrize("num_groups", [True, -1, 1.5, "1"])
def test_reserve_rejects_invalid_group_count(tmp_path: Path, num_groups: object) -> None:
    source = RolloutDataSource(_source_args(tmp_path))

    with pytest.raises(
        ValueError,
        match=rf"num_groups must be a nonnegative integer, got {num_groups!r}\.",
    ):
        source.reserve_samples(cast(int, num_groups))

    assert source.reserve_samples(1) == [_reservation(0, prompt="alpha")]


def test_duplicate_settlement_is_atomic(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path))
    first, second = source.reserve_samples(2)

    with pytest.raises(
        ValueError,
        match=r"Reservation settlement contains duplicate identities: \['0', '0'\]\.",
    ):
        source.acknowledge_reservations([first, first], rollout_id=0)

    source.acknowledge_reservations([first, second], rollout_id=0)
    assert source.reserve_samples(1) == [_reservation(2, prompt="charlie")]


def test_reissued_reservation_rejects_stale_attempt(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path))
    [first_attempt] = source.reserve_samples(1)
    source.requeue_reservations([first_attempt])

    [second_attempt] = source.reserve_samples(1)
    assert second_attempt == first_attempt
    assert second_attempt is not first_attempt

    with pytest.raises(
        RuntimeError,
        match=r"Source reservations are not the current outstanding attempts: \['0'\]\.",
    ):
        source.acknowledge_reservations([first_attempt], rollout_id=0)

    source.acknowledge_reservations([second_attempt], rollout_id=0)


def test_mixed_stale_settlement_leaves_valid_attempt_outstanding(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path))
    first, old_second = source.reserve_samples(2)
    source.requeue_reservations([old_second])
    [new_second] = source.reserve_samples(1)

    with pytest.raises(
        RuntimeError,
        match=r"Source reservations are not the current outstanding attempts: \['1'\]\.",
    ):
        source.acknowledge_reservations([first, old_second], rollout_id=0)

    source.acknowledge_reservations([first, new_second], rollout_id=0)
    assert source.reserve_samples(1) == [_reservation(2, prompt="charlie")]


def test_mixed_invalid_requeue_leaves_valid_attempt_outstanding(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path))
    [valid] = source.reserve_samples(1)
    forged = _reservation(99, prompt="forged")

    with pytest.raises(
        RuntimeError,
        match=r"Source reservations are not the current outstanding attempts: \['99'\]\.",
    ):
        source.requeue_reservations([valid, forged])

    source.acknowledge_reservations([valid], rollout_id=0)


def test_requeue_reconstructs_pristine_samples_in_process(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path))
    [first_attempt] = source.reserve_samples(1)
    first_attempt.samples[0].prompt = "mutated"
    first_attempt.samples[1].response = "generated"

    source.requeue_reservations([first_attempt])

    assert source.reserve_samples(1) == [_reservation(0, prompt="alpha")]


def test_requeue_replays_holes_in_source_order(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path))
    first, second, third, fourth = source.reserve_samples(4)
    source.acknowledge_reservations([second, fourth], rollout_id=0)
    source.requeue_reservations([third, first])

    assert source.reserve_samples(3) == [
        _reservation(0, prompt="alpha"),
        _reservation(2, prompt="charlie"),
        _reservation(4, prompt="alpha"),
    ]


def test_legacy_reads_are_rejected_after_durable_ownership_starts(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path))
    [reservation] = source.reserve_samples(1)

    with pytest.raises(
        RuntimeError,
        match="Cannot use get_samples after durable source reservations have started.",
    ):
        source.get_samples(1)

    source.requeue_reservations([reservation])
    assert source.reserve_samples(2) == [
        _reservation(0, prompt="alpha"),
        _reservation(1, prompt="bravo"),
    ]


def test_shuffled_multi_epoch_replay_reconstructs_exact_groups(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path, rollout_shuffle=True))
    reservations = source.reserve_samples(10)
    expected = copy.deepcopy(reservations)
    for reservation in reservations:
        reservation.samples[0].prompt = "mutated"
    source.requeue_reservations(reservations)

    assert source.reserve_samples(10) == expected


def test_materialization_failure_does_not_advance_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = RolloutDataSource(_source_args(tmp_path))
    original_deepcopy = copy.deepcopy
    copy_count = 0

    def fail_during_second_group(value: object) -> object:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 3:
            raise RuntimeError("injected materialization failure")
        return original_deepcopy(value)

    monkeypatch.setattr(data_source_module.copy, "deepcopy", fail_during_second_group)
    with pytest.raises(RuntimeError, match="injected materialization failure"):
        source.reserve_samples(2)

    monkeypatch.setattr(data_source_module.copy, "deepcopy", original_deepcopy)
    assert source.reserve_samples(2) == [
        _reservation(0, prompt="alpha"),
        _reservation(1, prompt="bravo"),
    ]


def test_non_persistent_source_rejects_durable_reservations(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    args.rollout_global_dataset = False
    source = RolloutDataSource(args)

    with pytest.raises(
        RuntimeError,
        match="RolloutDataSource does not support durable source reservations when rollout_global_dataset is disabled.",
    ):
        source.reserve_samples(1)


def test_reservation_aware_save_fails_before_checkpoint_support(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path))
    first, second = source.reserve_samples(2)
    source.acknowledge_reservations([first], rollout_id=0)
    source.requeue_reservations([second])

    with pytest.raises(
        RuntimeError,
        match=(
            "Cannot save source state after durable source reservations have started "
            "until reservation checkpointing is available."
        ),
    ):
        source.save(rollout_id=0)

    assert not (tmp_path / "rollout" / "global_dataset_state_dict_0.pt").exists()
    assert source.reserve_samples(1) == [_reservation(1, prompt="bravo")]


def test_buffered_source_rejects_reservations_that_bypass_retry_buffer(tmp_path: Path) -> None:
    source = RolloutDataSourceWithBuffer(_source_args(tmp_path))
    retry_group = list(_reservation(7, prompt="retry").samples)
    source.add_samples([retry_group])

    with pytest.raises(
        RuntimeError,
        match=(
            "RolloutDataSourceWithBuffer does not support durable source reservations because they would bypass its retry buffer."
        ),
    ):
        source.reserve_samples(1)

    assert source.get_samples(1) == [retry_group]
    assert source.get_buffer_length() == 0
