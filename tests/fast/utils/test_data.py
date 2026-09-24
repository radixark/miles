from __future__ import annotations

import json
from typing import Any

import pytest

from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.utils import object_store
from miles.utils.data import Dataset, RolloutDataPack, remove_rollout_data_refs, remove_train_output_refs
from miles.utils.object_store import (
    BaseObjectStore,
    ObjectStoreGetResult,
    StoreObjectRef,
    ValueSpec,
    _MooncakeStoreObjectRef,
)


class _RecordingProcessor:
    def __init__(self) -> None:
        self.prompts = []

    def extract_media(self, prompt):
        self.prompts.append(prompt)
        return {"images": ["image"], "videos": []}


def test_mixed_dataset_uses_processor_only_for_structured_prompts(tmp_path) -> None:
    text_prompt = "Fix the bug"
    multimodal_prompt = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image.png"},
                {"type": "text", "text": "Describe it"},
            ],
        }
    ]
    prompt_path = tmp_path / "prompts.jsonl"
    prompt_path.write_text(
        "\n".join(
            [
                json.dumps({"prompt": text_prompt}),
                json.dumps({"prompt": multimodal_prompt}),
            ]
        )
        + "\n"
    )
    processor = _RecordingProcessor()

    dataset = Dataset(
        str(prompt_path),
        tokenizer=None,
        processor=processor,
        max_length=None,
        prompt_key="prompt",
        apply_chat_template=False,
    )

    assert dataset.samples[0].prompt == text_prompt
    assert dataset.samples[0].multimodal_inputs is None
    assert dataset.samples[1].prompt == multimodal_prompt
    assert dataset.samples[1].multimodal_inputs == {
        "images": ["image"],
        "videos": [],
    }
    assert processor.prompts == [multimodal_prompt]


def _ref(payload: Any) -> StoreObjectRef:
    return _MooncakeStoreObjectRef(payload=payload)


class _RecordingStore(BaseObjectStore):
    def __init__(self) -> None:
        self.removed: list[StoreObjectRef] = []

    def put(self, value: Any, value_spec: dict[str, ValueSpec] | None = None) -> StoreObjectRef:
        return _ref(value)

    def get(self, ref: StoreObjectRef) -> ObjectStoreGetResult:
        raise NotImplementedError

    def remove(self, ref: StoreObjectRef) -> None:
        self.removed.append(ref)


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> _RecordingStore:
    instance = _RecordingStore()
    monkeypatch.setattr(object_store, "_INSTANCE", instance)
    return instance


class TestRemoveTrainOutputRefs:
    def test_every_shipped_ref_is_released(self, store: _RecordingStore):
        """Nothing else frees these objects, so a missed ref leaks for the whole run under mooncake."""
        refs = [_ref("a"), _ref("b")]

        remove_train_output_refs([TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=ref) for ref in refs])

        assert store.removed == refs

    def test_a_worker_that_shipped_nothing_is_skipped(self, store: _RecordingStore):
        """Only pp-last-stage critic workers ship values, so the rest carry None and must not reach the store."""
        ref = _ref("a")

        remove_train_output_refs(
            [
                TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=None),
                TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=ref),
                TrainStepOutput(outcome=TrainStepOutcome.NORMAL, values=None),
            ]
        )

        assert store.removed == [ref]


class TestRemoveRolloutDataRefs:
    def test_the_reference_the_rollout_shipped_is_released(self, store: _RecordingStore):
        """The driver is the only process that frees a rollout, once per step, or the store fills up."""
        ref = _ref("rollout")

        remove_rollout_data_refs(None, rollout_data_pack=RolloutDataPack(sample_indices=[0], data_ref=ref))

        assert store.removed == [ref]

    def test_every_shard_of_a_split_rollout_is_released(self, store: _RecordingStore):
        """Without --delay-split-train-data-by-dp there is one object per dp rank, and each one pins memory."""
        refs = [_ref("a"), _ref("b")]

        remove_rollout_data_refs(None, rollout_data_pack=RolloutDataPack(sample_indices=[0], data_ref=refs))

        assert store.removed == refs

    def test_a_pack_that_carries_no_data_never_reaches_the_store(self, store: _RecordingStore):
        """A pack without a data ref ships no object, and asking the store to free None would raise."""
        remove_rollout_data_refs(None, rollout_data_pack=RolloutDataPack())

        assert store.removed == []
