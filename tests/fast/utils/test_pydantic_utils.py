import math

import pytest
from pydantic import BaseModel

from miles.utils.pydantic_utils import (
    FrozenOpenBaseModel,
    FrozenPartialBaseModel,
    FrozenStrictBaseModel,
    StrictBaseModel,
)


class _StrictFloatModel(StrictBaseModel):
    value: float


class _FrozenStrictFloatModel(FrozenStrictBaseModel):
    value: float


class _FrozenPartialFloatModel(FrozenPartialBaseModel):
    value: float


class _FrozenOpenFloatModel(FrozenOpenBaseModel):
    value: float


class TestNonFiniteFloatJsonRoundTrip:
    @pytest.mark.parametrize(
        "model_class",
        [_StrictFloatModel, _FrozenStrictFloatModel, _FrozenPartialFloatModel, _FrozenOpenFloatModel],
    )
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_every_shared_model_base_preserves_non_finite_floats_across_json(
        self,
        model_class: type[BaseModel],
        value: float,
    ) -> None:
        """Every shared model base preserves non-finite floats across a JSON round trip."""
        restored_value = model_class.model_validate_json(model_class(value=value).model_dump_json()).value

        if math.isnan(value):
            assert math.isnan(restored_value)
        else:
            assert restored_value == value
