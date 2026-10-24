import copy
import pickle
import signal
from typing import Any, ClassVar

import pytest
import torch
from pydantic import TypeAdapter

from miles.utils.args.enhanced_argparse_namespace import EnhancedArgparseNamespace, SerializableConfigDict


class _Namespace(EnhancedArgparseNamespace):
    _mutable_fields: ClassVar[frozenset[str]] = frozenset({"step"})


class TestNamespaceMutation:
    def test_nested_scopes_restore_immutability_after_an_exception(self) -> None:
        """An exceptional inner scope must preserve the outer lease and finally revoke it."""
        config = _Namespace(step=1, model="fixed")
        with config.mutable():
            with pytest.raises(RuntimeError):
                with config.mutable():
                    config.step = 2
                    raise RuntimeError("abort")
            config.step = 3
            with pytest.raises(AttributeError, match="immutable"):
                config.model = "changed"

        assert config.step == 3
        with pytest.raises(AttributeError, match="immutable"):
            config.step = 4

    def test_deletion_obeys_the_same_scope_and_allowlist_as_assignment(self) -> None:
        """Deletion cannot bypass the configuration mutation boundary."""
        config = _Namespace(step=1, model="fixed")
        with pytest.raises(AttributeError, match="immutable"):
            del config.step
        with config.mutable():
            with pytest.raises(AttributeError, match="immutable"):
                del config.model
            del config.step
            config.step = 5

        assert vars(config) == {"step": 5, "model": "fixed"}

    @pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
    def test_copying_an_open_scope_never_transfers_mutation_authority(self, clone: Any) -> None:
        """Transported configurations start immutable even when copied during a mutable scope."""
        config = _Namespace(step=1)
        with config.mutable():
            restored = clone(config)
            with pytest.raises(AttributeError, match="immutable"):
                restored.step = 2
            config.step = 3

        assert restored.step == 1
        with restored.mutable():
            restored.step = 4
        assert config.step == 3


class TestNamespaceSerialization:
    def test_nested_types_survive_json_without_turning_tuples_into_lists(self) -> None:
        """Backend dtype, enum and tuple values retain their types at every nesting level."""
        adapter = TypeAdapter(SerializableConfigDict)
        values = {"nested": [{"pair": (torch.float32, signal.Signals.SIGTERM)}], "empty": (), "null": None}

        restored = adapter.validate_json(adapter.dump_json(values))

        assert restored == values
        assert isinstance(restored["nested"][0]["pair"], tuple)
        assert restored["nested"][0]["pair"][0] is torch.float32
        assert restored["nested"][0]["pair"][1] is signal.Signals.SIGTERM

    def test_namespace_json_round_trip_preserves_the_subclass_and_immutability(self) -> None:
        """Namespace deserialization retains the backend's mutation policy."""
        adapter = TypeAdapter(_Namespace)
        original = _Namespace(step=7, dtype=torch.bfloat16)

        restored = adapter.validate_json(adapter.dump_json(original))

        assert type(restored) is _Namespace
        assert restored.dtype is torch.bfloat16
        assert restored.step == 7
        with pytest.raises(AttributeError, match="immutable"):
            restored.step = 8

    def test_marker_like_mappings_with_other_keys_remain_plain_mappings(self) -> None:
        """An ordinary dictionary containing a reserved-looking key is not decoded as a tag."""
        adapter = TypeAdapter(SerializableConfigDict)
        values = {"value": {"__tuple__": [1], "metadata": "ordinary"}}

        assert adapter.validate_json(adapter.dump_json(values)) == values

    @pytest.mark.parametrize(
        "payload",
        [
            {"value": {"__enum__": "os:PathLike:missing"}},
            {"value": {"__enum__": "signal:signal:missing"}},
            {"value": {"__torch_dtype__": "Tensor"}},
        ],
    )
    def test_invalid_tagged_values_are_rejected(self, payload: dict[str, Any]) -> None:
        """Malformed backend tags cannot become arbitrary imported objects."""
        with pytest.raises(ValueError):
            TypeAdapter(SerializableConfigDict).validate_python(payload)

    @pytest.mark.parametrize("value", [None, 1, [], "config"])
    def test_non_mapping_namespace_inputs_are_rejected(self, value: Any) -> None:
        """A namespace payload must be an existing namespace or an explicit mapping."""
        with pytest.raises(ValueError, match="namespace or mapping"):
            TypeAdapter(_Namespace).validate_python(value)


class TestForceAllMutable:
    def test_fixed_fields_can_be_added_changed_and_deleted_only_inside_the_scope(self) -> None:
        """Forced mutation covers fields outside the normal allowlist and ends at scope exit."""
        config = _Namespace(model="old")
        other = _Namespace(model="other")
        with config.force_all_mutable():
            config.model = "new"
            config.extra = 3
            del config.model
            with pytest.raises(AttributeError, match="immutable"):
                other.model = "forbidden"

        assert vars(config) == {"extra": 3}
        with pytest.raises(AttributeError, match="immutable"):
            config.model = "forbidden"
        with pytest.raises(AttributeError, match="immutable"):
            del config.extra

    def test_exceptional_nested_force_scope_preserves_outer_force_scope(self) -> None:
        """An inner exception leaves outer forced mutation available until its own exit."""
        config = _Namespace(model="old")
        with config.force_all_mutable():
            with pytest.raises(RuntimeError, match="abort"):
                with config.force_all_mutable():
                    config.model = "inner"
                    raise RuntimeError("abort")
            config.model = "outer"

        assert config.model == "outer"
        with pytest.raises(AttributeError, match="immutable"):
            config.model = "forbidden"

    def test_exception_restores_the_enclosing_allowlist_scope(self) -> None:
        """Leaving forced mutation restores the outer ordinary scope's allowlist."""
        config = _Namespace(step=1, model="old")
        with config.mutable():
            with pytest.raises(RuntimeError, match="abort"):
                with config.force_all_mutable():
                    config.model = "new"
                    raise RuntimeError("abort")
            config.step = 2
            with pytest.raises(AttributeError, match="immutable"):
                config.model = "forbidden"

        assert config.step == 2
        with pytest.raises(AttributeError, match="immutable"):
            config.step = 3

    @pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
    def test_cloned_namespaces_do_not_inherit_forced_mutation(self, clone: Any) -> None:
        """Copying or serializing during forced mutation produces a frozen independent object."""
        config = _Namespace(model="old")
        with config.force_all_mutable():
            restored = clone(config)
            with pytest.raises(AttributeError, match="immutable"):
                restored.model = "forbidden"
            config.model = "new"

        assert vars(restored) == {"model": "old"}
        assert vars(config) == {"model": "new"}
