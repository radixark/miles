import argparse
from typing import Any, ClassVar

from pydantic import ConfigDict

from miles.utils.args.schema import BaseConfig


class BaseLeafConfig(BaseConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    _mutable_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        for trait in cls.__bases__:
            if trait is not BaseLeafConfig:
                trait.add_arguments(parser=parser)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        cls._mutable_fields = frozenset(name for base in cls.__mro__ for name in vars(base).get("_mutable_fields", ()))

    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self).model_fields and name not in self._mutable_fields:
            raise TypeError(f"{type(self).__name__}.{name} is immutable")
        super().__setattr__(name, value)
