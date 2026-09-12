from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.ray.train.group import TrainerController
from miles.utils.data import RolloutDataPack
from miles.utils.object_store import StoreObjectRef
from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs


def _round_trip(pack: RolloutDataPack) -> RolloutDataPack:
    serializer = collect_rpc_method_specs(RolloutExecutor)["get"].serializer
    return serializer.decode_result(serializer.encode_result(pack))


class TestWhatARolloutHandsToTheDriver:
    def test_the_data_reference_arrives_as_a_store_reference(self):
        """A pack typed as a plain mapping degrades the reference to a dict, and nothing can free it again."""
        pack = RolloutDataPack(sample_indices=[1, 2], data_ref=StoreObjectRef(payload={"key": "store/7"}))

        restored = _round_trip(pack)

        assert isinstance(restored.data_ref, StoreObjectRef)
        assert restored == pack

    def test_a_sharded_rollout_arrives_as_a_list_of_store_references(self):
        """With --delay-split-train-data-by-dp off the driver receives one reference per dp rank."""
        refs = [StoreObjectRef(payload={"key": f"store/{index}"}) for index in range(3)]
        pack = RolloutDataPack(sample_indices=[0], data_ref=refs)

        restored = _round_trip(pack)

        assert restored.data_ref == refs
        assert all(isinstance(ref, StoreObjectRef) for ref in restored.data_ref)

    def test_an_empty_batch_timeout_arrives_as_a_field(self):
        """The multi-LoRA driver reads this instead of catching a remote exception type."""
        restored = _round_trip(RolloutDataPack(empty_batch_timeout=True))

        assert restored.empty_batch_timeout is True and restored.data_refs == []

    def test_an_unknown_key_is_refused_rather_than_carried(self):
        """The pack is a contract between two processes; a key only one side knows is a silent mismatch."""
        with pytest.raises(ValidationError):
            RolloutDataPack(sample_indices=[0], data_reference=None)


class TestThePackTheTrainerControllerIsGiven:
    def test_the_controller_receives_the_same_pack_the_executor_returned(self):
        """The controller forwards data_ref to its cells, so it must arrive as a reference here too."""
        spec = collect_rpc_method_specs(TrainerController)["train"]
        pack = RolloutDataPack(sample_indices=[4], data_ref=StoreObjectRef(payload={"key": "store/4"}))

        decoded = spec.serializer.decode_query(
            spec.serializer.encode_query(dict(rollout_id=1, rollout_data_pack=pack))
        )

        assert decoded["rollout_data_pack"] == pack
        assert isinstance(decoded["rollout_data_pack"].data_ref, StoreObjectRef)
