from typing import ClassVar

import pytest
from pydantic import ValidationError
from tests.fast.utils.args.leaf_slice_fixtures import _DefaultLeaf

from miles.utils.args.runtime_base import BaseLeafConfig


class _Leaf(BaseLeafConfig):
    _mutable_fields: ClassVar[frozenset[str]] = frozenset({"step"})
    step: int
    model: str


class _ExtendedLeaf(_Leaf):
    _mutable_fields: ClassVar[frozenset[str]] = frozenset({"epoch"})
    epoch: int


class TestLeafMutation:
    def test_inherited_mutable_fields_remain_validated(self) -> None:
        """Subclass mutation permissions combine with the parent's validated permissions."""
        config = _ExtendedLeaf(step=1, epoch=2, model="fixed")
        config.step = 3
        config.epoch = 4

        assert config.step == 3
        assert config.epoch == 4
        with pytest.raises(ValidationError):
            config.step = "invalid"
        assert config.step == 3
        with pytest.raises(TypeError, match="immutable"):
            config.model = "changed"

    def test_unknown_fields_are_rejected_at_construction_and_assignment(self) -> None:
        """Typos cannot create undeclared runtime configuration fields."""
        with pytest.raises(ValidationError, match="extra_forbidden"):
            _Leaf(step=1, model="fixed", typo=3)
        config = _Leaf(step=1, model="fixed")
        with pytest.raises(ValidationError, match="no_such_attribute"):
            config.typo = 3


class TestLeafSlice:
    def test_slicing_keeps_only_leaf_fields_without_changing_the_source(self) -> None:
        """A worker leaf cannot receive unrelated fields from its wider source configuration."""
        source = _ExtendedLeaf(step=3, model="fixed", epoch=5)

        result = _Leaf.slice_from(source)

        assert result.model_dump() == {"step": 3, "model": "fixed"}
        assert source.model_dump() == {"step": 3, "model": "fixed", "epoch": 5}

    def test_missing_source_fields_are_rejected_even_when_the_leaf_has_defaults(self) -> None:
        """Leaf defaults cannot conceal an incomplete source schema."""
        with pytest.raises(AssertionError, match="it lacks.*count"):
            _DefaultLeaf.slice_from(BaseLeafConfig())

    def test_slicing_validates_source_values_instead_of_bypassing_the_leaf_schema(self) -> None:
        """Unchecked source values cannot cross the worker configuration boundary."""
        source = _Leaf.model_construct(step="invalid", model="fixed")

        with pytest.raises(ValidationError, match="step"):
            _Leaf.slice_from(source)
