from typing import Any

from pydantic import BaseModel


def config_values(config: object) -> dict[str, Any]:
    return dict(config) if isinstance(config, BaseModel) else vars(config).copy()
