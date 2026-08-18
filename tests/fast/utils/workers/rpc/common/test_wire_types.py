from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from pydantic.errors import PydanticSchemaGenerationError

from miles.utils.workers.rpc.common.metadata import collect_rpc_method_specs
from miles.utils.workers.rpc.common.wire_types import PICKLED_TAG, Pickled


class _PickledCall(BaseModel):
    args: Pickled


class _MixedCall(BaseModel):
    args: Pickled
    rollout_id: int


class TestThePickledEscapeHatch:
    def test_a_namespace_survives_a_json_round_trip_unchanged(self):
        """The trainer is built from the whole args, and a lossy round trip would change its run."""
        call = _PickledCall(args=Namespace(num_rollout=7, save=None))

        restored = _PickledCall.model_validate_json(call.model_dump_json())

        assert isinstance(restored.args, Namespace)
        assert (restored.args.num_rollout, restored.args.save) == (7, None)

    def test_a_value_no_wire_type_reproduces_survives(self):
        """This is the whole reason for the hatch: args holds objects json cannot describe."""
        call = _PickledCall(args=Namespace(path=Path("/models/qwen"), tags={"a", "b"}))

        restored = _PickledCall.model_validate_json(call.model_dump_json())

        assert restored.args.path == Path("/models/qwen")
        assert restored.args.tags == {"a", "b"}

    def test_the_encoded_form_is_a_tagged_string(self):
        """A reader of the wire can tell the hatch apart from an ordinary payload."""
        encoded = _PickledCall(args=Namespace(a=1)).model_dump(mode="json")["args"]

        assert isinstance(encoded[PICKLED_TAG], str)

    def test_an_in_process_value_is_taken_as_is(self):
        """A caller in the same process passes the object itself, and a copy would drop mutations."""
        args = Namespace(a=1)

        assert _PickledCall(args=args).args is args

    def test_the_hatch_does_not_loosen_its_neighbours(self):
        """The hatch is per parameter: everything beside it stays strictly wire-typed."""
        with pytest.raises(ValidationError):
            _MixedCall(args=Namespace(a=1), rollout_id="seven")


class TestTheStrictnessThePickledHatchDoesNotWeaken:
    def test_an_unannotated_parameter_is_still_refused(self):
        """Without an annotation there is no wire type at all, and the hatch is opt-in per parameter."""

        class _Worker:
            def act(self, value) -> None:
                pass

        with pytest.raises(TypeError, match="must be type-annotated"):
            collect_rpc_method_specs(_Worker)

    def test_a_parameter_without_the_hatch_keeps_its_wire_type(self):
        """A plain object reaches a strictly typed parameter only if the wire type accepts it."""

        class _Worker:
            def act(self, *, args: Namespace) -> None:
                pass

        with pytest.raises(PydanticSchemaGenerationError):
            collect_rpc_method_specs(_Worker)

    def test_the_hatch_makes_that_same_parameter_acceptable(self):
        """The hatch is what lets the argparse Namespace cross while the rest stays typed."""

        class _Worker:
            def act(self, *, args: Pickled, rollout_id: int) -> None:
                pass

        specs = collect_rpc_method_specs(_Worker)

        assert specs["act"].serializer.encode_query(dict(args=Namespace(a=1), rollout_id=3))["rollout_id"] == 3
