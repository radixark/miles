from typing import Any, ClassVar

from pydantic import ConfigDict

from miles.utils.pydantic_utils import StrictBaseModel


class BaseLeafConfig(StrictBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    _mutable_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        cls._mutable_fields = frozenset(name for base in cls.__mro__ for name in vars(base).get("_mutable_fields", ()))

    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self).model_fields and name not in self._mutable_fields:
            raise TypeError(f"{type(self).__name__}.{name} is immutable")
        super().__setattr__(name, value)
