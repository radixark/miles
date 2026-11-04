from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.schema import BaseConfig


class _HookConfig(BaseConfig):
    hook_marker: int = 41


def _hook(path: str = "test86.hook") -> CustomFunctionConfig:
    return CustomFunctionConfig(path=path, config=_HookConfig())
