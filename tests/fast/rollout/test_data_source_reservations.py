import copy
import json
import threading
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import numpy
import pytest
import torch
from PIL import Image
from pydantic import ValidationError

import miles.rollout.data_source as data_source_module
from miles.rollout.data_source import (
    RolloutDataSource,
    RolloutDataSourceWithBuffer,
    SourceReservation,
    SourceReservationId,
)
from miles.utils import chat_template_utils
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


@pytest.mark.parametrize("num_groups", [True, 1.5, "1"])
def test_reserve_rejects_non_integer_group_count(tmp_path: Path, num_groups: object) -> None:
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


def test_restart_replays_unsettled_groups_before_advancing(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    first, second, third = source.reserve_samples(3)
    source.acknowledge_reservations([second], rollout_id=7)
    source.requeue_reservations([third])
    first.samples[0].response = "mutated after reservation"
    third.samples[0].prompt = "also mutated"
    source.save(rollout_id=7)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=7)

    replayed = restored.reserve_samples(2)
    assert replayed == [
        _reservation(0, prompt="alpha"),
        _reservation(2, prompt="charlie"),
    ]
    restored.acknowledge_reservations(replayed, rollout_id=8)
    assert restored.reserve_samples(1) == [_reservation(3, prompt="delta")]


def test_load_rejects_source_configuration_mismatch(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=9)

    incompatible_args = _source_args(tmp_path)
    incompatible_args.n_samples_per_prompt = 3
    restored = RolloutDataSource(incompatible_args)

    with pytest.raises(
        ValueError,
        match="Source reservation checkpoint configuration does not match the current data source.",
    ):
        restored.load(rollout_id=9)


def test_load_rejects_changed_dataset_contents_at_same_path(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=9)

    Path(args.prompt_data).write_text(
        "\n".join(json.dumps({"prompt": prompt}) for prompt in ("omega", "bravo", "charlie", "delta")),
        encoding="utf-8",
    )
    restored = RolloutDataSource(args)

    with pytest.raises(
        ValueError,
        match="Source reservation checkpoint configuration does not match the current data source.",
    ):
        restored.load(rollout_id=9)


def test_load_accepts_changed_unused_chat_template_contents(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    template_path = tmp_path / "chat-template.jinja"
    template_path.write_text("first template", encoding="utf-8")
    args.chat_template_path = str(template_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=9)

    template_path.write_text("second template", encoding="utf-8")
    restored = RolloutDataSource(args)
    restored.load(rollout_id=9)

    assert restored.reserve_samples(1) == [_reservation(0, prompt="alpha")]


def test_load_rejects_changed_processed_samples_at_same_source_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _source_args(tmp_path)
    args.apply_chat_template = True
    processed_prompt = "first processed prompt"
    monkeypatch.setattr(
        chat_template_utils,
        "apply_chat_template",
        lambda *args, **kwargs: processed_prompt,
    )
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=9)

    processed_prompt = "second processed prompt"
    restored = RolloutDataSource(args)

    with pytest.raises(
        ValueError,
        match="Source reservation checkpoint configuration does not match the current data source.",
    ):
        restored.load(rollout_id=9)


def test_load_rejects_changed_materialized_multimodal_payload(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    assert source.dataset is not None
    source.dataset.origin_samples[0].multimodal_inputs = {
        "images": [Image.fromarray(numpy.array([[0, 1]], dtype=numpy.uint8))]
    }
    source.reserve_samples(1)
    source.save(rollout_id=9)

    restored = RolloutDataSource(args)
    assert restored.dataset is not None
    restored.dataset.origin_samples[0].multimodal_inputs = {
        "images": [Image.fromarray(numpy.array([[0, 2]], dtype=numpy.uint8))]
    }

    with pytest.raises(
        ValueError,
        match="Source reservation checkpoint configuration does not match the current data source.",
    ):
        restored.load(rollout_id=9)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("hf_checkpoint", "equivalent-tokenizer"),
        ("metadata_key", "unused-metadata"),
        ("tool_key", "unused-tools"),
        ("apply_chat_template_kwargs", {"unused": True}),
        ("rollout_seed", 101),
    ],
)
def test_load_accepts_equivalent_materialized_source(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=9)

    setattr(args, field, value)
    restored = RolloutDataSource(args)
    restored.load(rollout_id=9)

    assert restored.reserve_samples(1) == [_reservation(0, prompt="alpha")]


def test_load_accepts_equivalent_source_at_different_path(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=9)

    equivalent_path = tmp_path / "equivalent.jsonl"
    equivalent_path.write_bytes(Path(args.prompt_data).read_bytes())
    args.prompt_data = str(equivalent_path)
    restored = RolloutDataSource(args)
    restored.load(rollout_id=9)

    assert restored.reserve_samples(1) == [_reservation(0, prompt="alpha")]


def test_load_accepts_changed_ignored_source_fields(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=9)

    Path(args.prompt_data).write_text(
        "\n".join(
            json.dumps({"prompt": prompt, "ignored": "changed"}) for prompt in ("alpha", "bravo", "charlie", "delta")
        ),
        encoding="utf-8",
    )
    restored = RolloutDataSource(args)
    restored.load(rollout_id=9)

    assert restored.reserve_samples(1) == [_reservation(0, prompt="alpha")]


def test_load_rejects_changed_shuffle_order(tmp_path: Path) -> None:
    args = _source_args(tmp_path, rollout_shuffle=True)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=9)

    args.rollout_seed += 1
    restored = RolloutDataSource(args)

    with pytest.raises(
        ValueError,
        match="Source reservation checkpoint configuration does not match the current data source.",
    ):
        restored.load(rollout_id=9)


def test_load_accepts_legacy_checkpoint_without_reservations(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_10.pt"
    checkpoint_path.parent.mkdir()
    torch.save(
        {
            "sample_offset": 2,
            "epoch_id": 0,
            "sample_group_index": 2,
            "sample_index": 4,
            "metadata": {"legacy": True},
        },
        checkpoint_path,
    )

    restored = RolloutDataSource(args)
    restored.load(rollout_id=10)

    assert restored.reserve_samples(1) == [_reservation(2, prompt="charlie")]
    assert restored.metadata == {"legacy": True}


def test_load_minus_one_is_a_noop(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path))

    source.load(rollout_id=-1)

    assert source.get_samples(1) == [list(_reservation(0, prompt="alpha").samples)]


def test_load_rejects_active_durable_ownership(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    [reservation] = source.reserve_samples(1)
    source.save(rollout_id=10)

    with pytest.raises(
        RuntimeError,
        match="Cannot load source state after durable source reservations have started.",
    ):
        source.load(rollout_id=10)

    source.requeue_reservations([reservation])
    assert source.reserve_samples(1) == [_reservation(0, prompt="alpha")]


@pytest.mark.parametrize("missing_field", ["sample_offset", "epoch_id", "sample_group_index", "sample_index"])
def test_load_rejects_missing_versioned_cursor_field(tmp_path: Path, missing_field: str) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=10)
    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_10.pt"
    state = torch.load(checkpoint_path)
    del state[missing_field]
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(
        ValueError,
        match=rf"Checkpoint is missing source cursor fields: \['{missing_field}'\]\.",
    ):
        restored.load(rollout_id=10)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_offset", True),
        ("epoch_id", -1),
        ("sample_group_index", 1.5),
        ("sample_index", "2"),
    ],
)
def test_load_rejects_invalid_versioned_cursor_value(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=10)
    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_10.pt"
    state = torch.load(checkpoint_path)
    state[field] = value
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(
        ValueError,
        match=rf"Checkpoint {field} must be a nonnegative integer, got {value!r}\.",
    ):
        restored.load(rollout_id=10)


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        (
            "sample_index",
            0,
            "Checkpoint sample frontier 0 does not match group frontier 1 with 2 samples per prompt.",
        ),
        (
            "sample_offset",
            5,
            "Checkpoint sample offset 5 exceeds dataset size 4.",
        ),
        (
            "epoch_id",
            1,
            "Checkpoint group frontier 1 does not match dataset cursor at epoch 1 offset 1 for dataset size 4.",
        ),
    ],
)
def test_load_rejects_inconsistent_versioned_cursor(
    tmp_path: Path,
    field: str,
    value: int,
    expected_error: str,
) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=10)
    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_10.pt"
    state = torch.load(checkpoint_path)
    state[field] = value
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(ValueError) as error:
        restored.load(rollout_id=10)

    assert str(error.value) == expected_error


@pytest.mark.parametrize("group_index", [True, "1", 1.0])
def test_load_rejects_coerced_replay_group_index(
    tmp_path: Path,
    group_index: object,
) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(2)
    source.save(rollout_id=10)
    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_10.pt"
    state = torch.load(checkpoint_path)
    state["source_reservations"]["replay"][1]["group_index"] = group_index
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(ValidationError, match="group_index"):
        restored.load(rollout_id=10)


def test_load_rejects_unknown_reservation_schema_version(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=10)
    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_10.pt"
    state = torch.load(checkpoint_path)
    state["source_reservations"]["schema_version"] = 2
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(ValidationError, match="schema_version"):
        restored.load(rollout_id=10)


def test_load_rejects_duplicate_replay_identity(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(2)
    source.save(rollout_id=10)
    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_10.pt"
    state = torch.load(checkpoint_path)
    state["source_reservations"]["replay"][1]["group_index"] = 0
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(
        ValueError,
        match=r"Checkpoint contains duplicate source reservation identities: \['0', '0'\]\.",
    ):
        restored.load(rollout_id=10)


def test_load_rejects_replay_at_saved_frontier(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=10)
    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_10.pt"
    state = torch.load(checkpoint_path)
    state["source_reservations"]["replay"][0]["group_index"] = 1
    torch.save(state, checkpoint_path)

    restored = RolloutDataSource(args)
    with pytest.raises(
        ValueError,
        match="Checkpoint contains a source reservation outside the saved group frontier.",
    ):
        restored.load(rollout_id=10)


def test_checkpoint_replays_acknowledgements_after_its_rollout_frontier(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    first, second, _ = source.reserve_samples(3)
    source.acknowledge_reservations([first], rollout_id=5)
    source.acknowledge_reservations([second], rollout_id=6)
    source.save(rollout_id=5)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=5)

    assert restored.reserve_samples(2) == [
        _reservation(1, prompt="bravo"),
        _reservation(2, prompt="charlie"),
    ]


def test_acknowledge_rejects_published_rollout_id(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    first, second = source.reserve_samples(2)
    source.save(rollout_id=5)

    with pytest.raises(
        ValueError,
        match="Reservation rollout_id 5 must be newer than published checkpoint 5.",
    ):
        source.acknowledge_reservations([second], rollout_id=5)

    source.acknowledge_reservations([first, second], rollout_id=6)
    source.save(rollout_id=6)
    restored = RolloutDataSource(args)
    restored.load(rollout_id=6)
    assert restored.reserve_samples(1) == [_reservation(2, prompt="charlie")]


def test_save_rejects_rollout_id_regression(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=5)

    with pytest.raises(
        ValueError,
        match="Source checkpoint rollout_id must not move backward from 5 to 4.",
    ):
        source.save(rollout_id=4)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=5)
    assert restored.reserve_samples(1) == [_reservation(0, prompt="alpha")]


def test_legacy_save_rejects_acknowledgement_at_published_rollout_id(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.get_samples(1)
    source.save(rollout_id=5)
    reservation = source.reserve_samples(1)[0]

    with pytest.raises(
        ValueError,
        match="Reservation rollout_id 5 must be newer than published checkpoint 5.",
    ):
        source.acknowledge_reservations([reservation], rollout_id=5)

    source.acknowledge_reservations([reservation], rollout_id=6)


def test_legacy_save_rejects_rollout_id_regression(tmp_path: Path) -> None:
    source = RolloutDataSource(_source_args(tmp_path))
    source.get_samples(1)
    source.save(rollout_id=5)

    with pytest.raises(
        ValueError,
        match="Source checkpoint rollout_id must not move backward from 5 to 4.",
    ):
        source.save(rollout_id=4)


def test_legacy_source_save_preserves_checkpoint_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_fingerprint(_source: RolloutDataSource) -> str:
        raise AssertionError("legacy source use must not fingerprint prompt data")

    monkeypatch.setattr(RolloutDataSource, "_processed_samples_sha256", reject_fingerprint)
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    assert source.get_samples(1) == [list(_reservation(0, prompt="alpha").samples)]

    source.save(rollout_id=10)

    checkpoint_path = tmp_path / "rollout" / "global_dataset_state_dict_10.pt"
    assert torch.load(checkpoint_path) == {
        "sample_offset": 1,
        "epoch_id": 0,
        "sample_group_index": 1,
        "sample_index": 2,
        "metadata": {},
    }


@pytest.mark.parametrize(
    ("rollout_shuffle", "first_prompts", "restored_prompts"),
    [
        (False, ("alpha", "bravo", "charlie", "delta", "alpha"), ("bravo", "charlie", "delta")),
        (True, ("alpha", "charlie", "delta", "bravo", "alpha"), ("delta", "charlie", "bravo")),
    ],
)
def test_legacy_source_preserves_multi_epoch_order_across_restart(
    tmp_path: Path,
    rollout_shuffle: bool,
    first_prompts: tuple[str, ...],
    restored_prompts: tuple[str, ...],
) -> None:
    args = _source_args(tmp_path, rollout_shuffle=rollout_shuffle)
    source = RolloutDataSource(args)

    assert source.get_samples(5) == [
        list(_reservation(group_index, prompt=prompt).samples) for group_index, prompt in enumerate(first_prompts)
    ]
    source.save(rollout_id=10)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=10)
    assert restored.get_samples(3) == [
        list(_reservation(group_index, prompt=prompt).samples)
        for group_index, prompt in enumerate(restored_prompts, start=5)
    ]


def test_shuffled_multi_epoch_reservations_reconstruct_exact_groups(tmp_path: Path) -> None:
    args = _source_args(tmp_path, rollout_shuffle=True)
    source = RolloutDataSource(args)
    reservations = source.reserve_samples(10)
    expected = copy.deepcopy(reservations)
    for reservation in reservations:
        reservation.samples[0].prompt = "mutated"
    source.save(rollout_id=11)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=11)

    assert restored.reserve_samples(10) == expected


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


def test_failed_save_preserves_previous_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=12)
    source.reserve_samples(1)

    def fail_after_partial_write(state: object, path: str) -> None:
        Path(path).write_bytes(b"partial checkpoint")
        raise RuntimeError("injected save failure")

    monkeypatch.setattr(data_source_module.torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="injected save failure"):
        source.save(rollout_id=12)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=12)
    assert restored.reserve_samples(2) == [
        _reservation(0, prompt="alpha"),
        _reservation(1, prompt="bravo"),
    ]


def test_failed_checkpoint_replace_preserves_previous_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.reserve_samples(1)
    source.save(rollout_id=12)
    source.reserve_samples(1)

    def fail_replace(_source_path: str, destination_path: str) -> None:
        raise RuntimeError(f"injected replace failure for {destination_path}")

    monkeypatch.setattr(data_source_module.os, "replace", fail_replace)
    with pytest.raises(RuntimeError, match="injected replace failure"):
        source.save(rollout_id=12)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=12)
    assert restored.reserve_samples(2) == [
        _reservation(0, prompt="alpha"),
        _reservation(1, prompt="bravo"),
    ]


def test_save_is_linearized_with_reservation_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    [first] = source.reserve_samples(1)
    save_started = threading.Event()
    allow_save = threading.Event()
    settlement_started = threading.Event()
    settlement_finished = threading.Event()
    original_torch_save = torch.save

    def blocking_save(state: object, path: str) -> None:
        save_started.set()
        assert allow_save.wait(timeout=5)
        original_torch_save(state, path)

    def settle_after_save_starts() -> None:
        settlement_started.set()
        source.acknowledge_reservations([first], rollout_id=1)
        settlement_finished.set()

    monkeypatch.setattr(data_source_module.torch, "save", blocking_save)
    with ThreadPoolExecutor(max_workers=2) as executor:
        save_future = executor.submit(source.save, 0)
        assert save_started.wait(timeout=5)
        settlement_future = executor.submit(settle_after_save_starts)
        assert settlement_started.wait(timeout=5)
        assert not settlement_finished.wait(timeout=0.05)
        allow_save.set()
        save_future.result(timeout=5)
        settlement_future.result(timeout=5)

    restored = RolloutDataSource(args)
    restored.load(rollout_id=0)
    assert restored.reserve_samples(1) == [first]


def test_no_checkpoint_mode_rejects_a_false_durable_save(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    args.save_interval = None
    source = RolloutDataSource(args)
    [reservation] = source.reserve_samples(1)
    source.acknowledge_reservations([reservation], rollout_id=0)

    with pytest.raises(
        RuntimeError,
        match="Cannot save durable source reservations without --save-interval",
    ):
        source.save(rollout_id=0)


def test_sentinel_only_checkpoint_rejects_durable_reservations(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    args.save_interval = None
    args.save_trigger_sentinel = str(tmp_path / "save-trigger")
    source = RolloutDataSource(args)

    with pytest.raises(
        RuntimeError,
        match="Durable source reservations require a periodic save interval when a save trigger is configured.",
    ):
        source.reserve_samples(1)

    assert source.get_samples(1) == [list(_reservation(0, prompt="alpha").samples)]


def test_no_checkpoint_mode_allows_reservations_but_refuses_to_persist_them(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    args.save_interval = None
    args.save_trigger_sentinel = None
    source = RolloutDataSource(args)

    assert source.supports_source_reservations is True
    assert source.reserve_samples(2) == [
        _reservation(0, prompt="alpha"),
        _reservation(1, prompt="bravo"),
    ]

    with pytest.raises(
        RuntimeError,
        match="Cannot save durable source reservations without --save-interval",
    ):
        source.save(rollout_id=0)


def test_resume_without_save_interval_replays_reservations_but_refuses_to_persist_them(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    first, second = source.reserve_samples(2)
    source.acknowledge_reservations([second], rollout_id=4)
    source.requeue_reservations([first])
    source.save(rollout_id=4)

    resumed_args = copy.copy(args)
    resumed_args.save_interval = None
    resumed_args.save_trigger_sentinel = None
    restored = RolloutDataSource(resumed_args)
    restored.load(rollout_id=4)

    assert restored.supports_source_reservations is True
    assert restored.reserve_samples(2) == [
        _reservation(0, prompt="alpha"),
        _reservation(2, prompt="charlie"),
    ]

    with pytest.raises(
        RuntimeError,
        match="Cannot save durable source reservations without --save-interval",
    ):
        restored.save(rollout_id=5)


def test_sentinel_only_resume_refuses_a_checkpoint_holding_reservations(
    tmp_path: Path,
) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    first, second = source.reserve_samples(2)
    source.acknowledge_reservations([second], rollout_id=4)
    source.requeue_reservations([first])
    source.save(rollout_id=4)

    resumed_args = copy.copy(args)
    resumed_args.save_interval = None
    resumed_args.save_trigger_sentinel = str(tmp_path / "save-trigger")
    restored = RolloutDataSource(resumed_args)
    assert restored.supports_source_reservations is False

    with pytest.raises(
        ValueError,
        match=r"^Source reservation checkpoint cannot be loaded: Durable source reservations require a periodic save interval when a save trigger is configured\.$",
    ):
        restored.load(rollout_id=4)

    assert restored.supports_source_reservations is False
    assert (restored.epoch_id, restored.sample_offset) == (0, 0)
    assert (restored.sample_group_index, restored.sample_index) == (0, 0)
    assert restored.metadata == {}
    # The replay list has no public reader on a source the gate has closed.
    assert restored._replay_reservations == []
    assert restored._durable_reservations_started is False
    assert restored.get_samples(1) == [list(_reservation(0, prompt="alpha").samples)]


def test_sentinel_only_resume_accepts_a_checkpoint_without_reservations(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    source.metadata = {"legacy": True}
    source.get_samples(2)
    source.save(rollout_id=4)

    resumed_args = copy.copy(args)
    resumed_args.save_interval = None
    resumed_args.save_trigger_sentinel = str(tmp_path / "save-trigger")
    restored = RolloutDataSource(resumed_args)
    assert restored.supports_source_reservations is False

    restored.load(rollout_id=4)

    assert (restored.epoch_id, restored.sample_offset) == (0, 2)
    assert (restored.sample_group_index, restored.sample_index) == (2, 4)
    assert restored.metadata == {"legacy": True}
    assert restored.get_samples(1) == [list(_reservation(2, prompt="charlie").samples)]


def test_buffered_resume_refuses_a_checkpoint_holding_reservations(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    source = RolloutDataSource(args)
    first, second = source.reserve_samples(2)
    source.acknowledge_reservations([second], rollout_id=4)
    source.requeue_reservations([first])
    source.save(rollout_id=4)

    restored = RolloutDataSourceWithBuffer(copy.copy(args))
    assert restored.supports_source_reservations is False

    with pytest.raises(
        ValueError,
        match=r"^Source reservation checkpoint cannot be loaded: RolloutDataSourceWithBuffer does not support durable source reservations\.$",
    ):
        restored.load(rollout_id=4)

    assert (restored.epoch_id, restored.sample_offset) == (0, 0)
    assert (restored.sample_group_index, restored.sample_index) == (0, 0)
    assert restored.metadata == {}
    # The replay list has no public reader on a source the gate has closed.
    assert restored._replay_reservations == []
    assert restored._durable_reservations_started is False
    assert restored.get_samples(1) == [list(_reservation(0, prompt="alpha").samples)]


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


def test_configured_source_reports_reservation_support(tmp_path: Path) -> None:
    assert RolloutDataSource(_source_args(tmp_path)).supports_source_reservations is True


def test_source_without_global_dataset_reports_no_reservation_support(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    args.rollout_global_dataset = False

    assert RolloutDataSource(args).supports_source_reservations is False


def test_sentinel_only_checkpoint_source_reports_no_reservation_support(tmp_path: Path) -> None:
    args = _source_args(tmp_path)
    args.save_interval = None
    args.save_trigger_sentinel = str(tmp_path / "trigger")

    assert RolloutDataSource(args).supports_source_reservations is False


def test_buffered_source_reports_no_reservation_support(tmp_path: Path) -> None:
    assert RolloutDataSourceWithBuffer(_source_args(tmp_path)).supports_source_reservations is False
