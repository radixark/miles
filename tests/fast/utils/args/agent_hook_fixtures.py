from typing import Annotated

from miles.utils.args.schema import Arg, BaseConfig


class _AgentHookConfig(BaseConfig):
    agent_probe_threshold: Annotated[float, Arg()] = 0.5
