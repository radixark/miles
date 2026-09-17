from typing import Any

from pydantic import ConfigDict

from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.runtime import TrainerConfig
from miles.utils.args.runtime_base import BaseLeafConfig
from miles.utils.args.schema import BaseConfig


class ImmutableNamespace(BaseConfig):
    model_config = ConfigDict(extra="allow", frozen=True)


def compute_custom_function_config(
    args: BaseLeafConfig,
    function: CustomFunctionConfig,
    *runtime_sources: dict[str, Any],
) -> ImmutableNamespace:
    sources = [dict(args)]
    if isinstance(args, TrainerConfig):
        sources.append(vars(args.backend))
    if (config := function.config) is not None:
        sources.append(dict(config))
    return ImmutableNamespace.model_validate(_merge_dicts(*sources, *runtime_sources))


def _merge_dicts(*sources: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for source in sources:
        for name, value in source.items():
            if name in values and values[name] != value:
                raise ValueError(
                    f"Custom configuration field {name!r} has conflicting values {values[name]!r} and {value!r}"
                )
            values[name] = value
    return values
