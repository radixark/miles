from unittest.mock import MagicMock

import miles.rollout.inference_rollout.inference_rollout_common as inference_rollout_common
from miles.rollout.base_types import RolloutFnConstructorInput
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn


def test_inference_rollout_fn_exposes_the_constructor_input(monkeypatch) -> None:
    """The base class promises constructor_input, so the framework's own subclasses must set it too."""
    monkeypatch.setattr(inference_rollout_common, "GenerateState", lambda args: MagicMock())
    constructor_input = RolloutFnConstructorInput(args=MagicMock(), data_source=MagicMock())

    fn = InferenceRolloutFn(constructor_input)

    assert fn.constructor_input is constructor_input
