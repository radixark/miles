from __future__ import annotations

import time
from enum import auto
from pathlib import Path

from pydantic import Field, ValidationError, model_validator

try:
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

from miles.utils.file_utils import atomic_write_text
from miles.utils.pydantic_utils import FrozenStrictBaseModel

STATE_FILE_FLAG = "--state-file"


class OrchestratorStatus(StrEnum):
    STARTED = auto()
    EXITED = auto()


class OrchestratorState(FrozenStrictBaseModel):
    status: OrchestratorStatus
    exit_code: int | None = None
    timestamp: float = Field(default_factory=time.time)

    @property
    def is_terminal(self) -> bool:
        return self.status is OrchestratorStatus.EXITED

    @model_validator(mode="after")
    def _check_exit_code(self) -> OrchestratorState:
        assert self.status is not OrchestratorStatus.EXITED or self.exit_code is not None, (
            f"A terminal state is the verdict of the whole run, so it has to carry the exit code the launcher "
            f"passes back to its caller, but {self} names none"
        )
        return self

    @classmethod
    def read(cls, path: str | Path) -> OrchestratorState | None:
        try:
            payload = Path(path).read_text()
        except FileNotFoundError:
            return None

        try:
            return cls.model_validate_json(payload)
        except ValidationError:
            return None

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, self.model_dump_json())
