from __future__ import annotations

from argparse import Namespace

import pytest
from pydantic import TypeAdapter, ValidationError

from miles.utils import object_store

from miles.utils.object_store import RayObjectStore, StoreObjectRef, _MooncakeStoreObjectRef, _RayStoreObjectRef

FREED_OBJECT_TIMEOUT_SECONDS = 60.0


def _ray_store(*, frees_objects: bool = True) -> RayObjectStore:
    return RayObjectStore(frees_objects=frees_objects)


_REF_ADAPTER = TypeAdapter(StoreObjectRef)


def _after_a_json_round_trip(ref: StoreObjectRef) -> StoreObjectRef:
    return _REF_ADAPTER.validate_json(ref.model_dump_json())


class TestStoreObjectRef:
    def test_a_mooncake_reference_survives_a_json_round_trip(self):
        """The trainer ships it to the driver over rpc, which encodes the model as json."""
        ref = _MooncakeStoreObjectRef(payload={"key": "miles-object-store/7", "size": 12})

        restored = _after_a_json_round_trip(ref)

        assert restored == ref
        assert restored.payload == {"key": "miles-object-store/7", "size": 12}

    def test_a_reference_cannot_be_repointed_after_it_is_handed_over(self):
        """A consumer holding it must read the object the producer put, not one somebody swapped in."""
        ref = _MooncakeStoreObjectRef(payload="k")

        with pytest.raises(ValidationError):
            ref.payload = "other"

    def test_each_store_rebuilds_its_own_kind_of_reference(self, ray_local_mode):
        """Only a ray reference may be freed by hand, so the tag it travels with has to survive the wire."""
        assert isinstance(_after_a_json_round_trip(_ray_store().put("payload")), _RayStoreObjectRef)
        assert isinstance(_after_a_json_round_trip(_MooncakeStoreObjectRef(payload="k")), _MooncakeStoreObjectRef)


class TestTheStoreARunBuilds:
    @pytest.mark.parametrize("wire, frees_objects", [("rpc", True), ("ray", False)])
    def test_only_the_wire_that_pins_objects_frees_them(self, wire: str, frees_objects: bool):
        """The mode is what decides, so a run must not have to remember which references it may free."""
        args = Namespace(object_store_backend="ray", worker_comm_backend=wire)

        store = object_store._create_instance(args, contribute_segment=None)

        assert store._frees_objects is frees_objects


class TestARayReferenceOnTheWire:
    def test_it_survives_a_json_round_trip(self, ray_local_mode):
        """comm-backend=rpc still allows the ray object store, so its reference has to cross the wire."""
        ref = _ray_store().put({"tokens": [1, 2, 3]})

        restored = _after_a_json_round_trip(ref)

        assert _ray_store().get(restored).value == {"tokens": [1, 2, 3]}

    def test_the_encoded_form_is_a_string(self, ray_local_mode):
        """An ObjectRef is not json, so it travels as the cloudpickle bytes ray documents as the last resort."""
        ref = _ray_store().put("payload")

        assert isinstance(ref.model_dump(mode="json")["payload"], str)

    def test_a_reference_that_never_left_this_process_is_unchanged(self, ray_local_mode):
        """Ray communication passes the model as is, and re-encoding it would cost a copy per call."""
        ref = _ray_store().put("payload")

        assert _ray_store().get(ref).value == "payload"

    def test_removing_a_reference_releases_it_by_hand(self, monkeypatch, ray_local_mode):
        """Cloudpickling a reference pins the object, so reference counting can no longer free it."""
        import ray

        freed: list[list] = []
        monkeypatch.setattr(ray._private.internal_api, "free", lambda refs: freed.append(list(refs)))
        restored = _after_a_json_round_trip(_ray_store().put("payload"))

        _ray_store().remove(restored)

        assert freed == [[restored.payload]]

    def test_ray_communication_leaves_the_object_to_reference_counting(self, monkeypatch, ray_local_mode):
        """Nothing pins the object on that wire, and a free would destroy it for every other holder."""
        import ray

        freed: list[list] = []
        monkeypatch.setattr(ray._private.internal_api, "free", lambda refs: freed.append(list(refs)))
        store = _ray_store(frees_objects=False)
        ref = store.put("payload")

        store.remove(ref)

        assert freed == []
        assert store.get(ref).value == "payload"


class TestFreeingARayObjectForReal:
    def test_the_object_is_gone_after_its_reference_is_removed(self, ray_local_mode):
        """The point of the explicit free is that the pinned object really stops occupying the store."""
        import ray

        restored = _after_a_json_round_trip(_ray_store().put({"tokens": [1, 2, 3]}))

        _ray_store().remove(restored)

        with pytest.raises(ray.exceptions.ObjectLostError):
            ray.get(restored.payload, timeout=FREED_OBJECT_TIMEOUT_SECONDS)

    def test_an_object_nobody_removed_is_still_readable(self, ray_local_mode):
        """A free reaching further than the reference it was handed would take the run's data with it."""
        kept = _after_a_json_round_trip(_ray_store().put({"tokens": [4]}))
        removed = _after_a_json_round_trip(_ray_store().put({"tokens": [5]}))

        _ray_store().remove(removed)

        assert _ray_store().get(kept).value == {"tokens": [4]}
