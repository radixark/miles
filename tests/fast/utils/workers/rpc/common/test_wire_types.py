from __future__ import annotations

from argparse import Namespace

import pytest
from pydantic import BaseModel, ValidationError

from miles.utils.workers.rpc.common.wire_types import WireNamespace


class _Call(BaseModel):
    args: WireNamespace


class TestWireNamespace:
    def test_a_namespace_survives_a_json_round_trip(self):
        """The driver hands the worker its whole args over rpc, which encodes the model as json."""
        call = _Call(args=Namespace(num_rollout=7, save=None))

        restored = _Call.model_validate_json(call.model_dump_json())

        assert isinstance(restored.args, Namespace)
        assert (restored.args.num_rollout, restored.args.save) == (7, None)

    def test_a_namespace_is_serialized_as_its_own_mapping(self):
        """A worker of another language, and a log of the call, both read the fields by name."""
        assert _Call(args=Namespace(a=1)).model_dump() == {"args": {"a": 1}}

    def test_an_already_built_namespace_is_taken_as_is(self):
        """An in-process caller passes the object itself, and copying it would drop later mutations."""
        args = Namespace(a=1)

        assert _Call(args=args).args is args

    def test_a_payload_that_is_not_a_mapping_is_rejected(self):
        """Anything else silently becomes a namespace with no fields, and the worker reads defaults."""
        with pytest.raises(ValidationError):
            _Call(args=[("a", 1)])
