from __future__ import annotations

import pytest
from pydantic import ValidationError

from miles.ray.rollout.cell_state import (
    CellAddrInfo,
    StateDisposed,
    StateInitializing,
    StatePendingWeights,
    StateServing,
    StateUninitialized,
)

_ADDR_INFO = CellAddrInfo(
    server_url="http://10.0.0.1:30000",
    bootstrap_port=None,
    gate_url="http://10.0.0.1:13000",
)

_STATE_FACTORIES = [
    ("StateUninitialized", StateUninitialized, {}),
    ("StateInitializing", StateInitializing, {"addr_info": _ADDR_INFO}),
    ("StatePendingWeights", StatePendingWeights, {"addr_info": _ADDR_INFO}),
    ("StateServing", StateServing, {"addr_info": _ADDR_INFO}),
    ("StateDisposed", StateDisposed, {}),
]


class TestCellStateModelsRejectUnknownFields:
    """A typo'd payload field must fail loudly instead of being silently dropped."""

    @pytest.mark.parametrize(
        "model, kwargs",
        [(model, kwargs) for _name, model, kwargs in _STATE_FACTORIES],
        ids=[name for name, _model, _kwargs in _STATE_FACTORIES],
    )
    def test_an_unknown_field_is_rejected(self, model, kwargs) -> None:
        """Every state payload forbids extra fields."""
        with pytest.raises(ValidationError, match="stale_addr_info"):
            model(**kwargs, stale_addr_info=_ADDR_INFO)

    def test_cell_addr_info_rejects_an_unknown_field(self) -> None:
        """The address payload forbids extra fields too."""
        with pytest.raises(ValidationError, match="bootstrap_host"):
            CellAddrInfo(
                server_url="http://10.0.0.1:30000",
                bootstrap_port=None,
                gate_url="http://10.0.0.1:13000",
                bootstrap_host="10.0.0.1",
            )


class TestCellStateModelsAreFrozen:
    """The state machine replaces the whole state object; in-place edits must be impossible."""

    @pytest.mark.parametrize(
        "model, kwargs",
        [(model, kwargs) for _name, model, kwargs in _STATE_FACTORIES],
        ids=[name for name, _model, _kwargs in _STATE_FACTORIES],
    )
    def test_every_state_payload_is_frozen(self, model, kwargs) -> None:
        """Every state payload retains the frozen-model contract, including fieldless states."""
        state = model(**kwargs)

        assert state.model_config["frozen"] is True

        if "addr_info" in kwargs:
            with pytest.raises(ValidationError):
                state.addr_info = _ADDR_INFO

    def test_cell_addr_info_cannot_be_reassigned_in_place(self) -> None:
        """A shared address payload must stay safe to hand out by reference."""
        with pytest.raises(ValidationError):
            _ADDR_INFO.server_url = "http://10.0.0.2:30000"
