from __future__ import annotations

import abc
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated
from uuid import uuid4

from pydantic import Discriminator, Field
from tests.utils.soak.deploy.types import (
    DeploymentObservationDetails,
    DeploymentTarget,
    HotRestartDetails,
    HotRestartTakeOverEvidence,
)
from tests.utils.soak.ft.types import CellFaultDetails, CellFaultEvidence, CellTarget

from miles.utils.audit_utils.event_logger.models import Event
from miles.utils.pydantic_utils import FrozenStrictBaseModel

if TYPE_CHECKING:
    from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
    from tests.utils.soak.core.views import SoakActionRecord

SoakTarget = Annotated[CellTarget | DeploymentTarget, Discriminator("kind")]

SoakActionDetails = Annotated[CellFaultDetails | HotRestartDetails, Discriminator("form")]

SoakActionEvidence = Annotated[CellFaultEvidence | HotRestartTakeOverEvidence, Discriminator("kind")]

SoakObservationDetails = DeploymentObservationDetails


class SoakActionRequest(FrozenStrictBaseModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    target: SoakTarget
    form_name: str
    details: SoakActionDetails


# ================================ action forms ================================

SoakForms = dict[str, list["BaseSoakActionForm"]]


class BaseSoakActionForm(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def harms_target(self) -> bool: ...

    @abc.abstractmethod
    def maybe_create_request(
        self,
        *,
        target: SoakTarget,
        observation: SoakObservationEvent,
        events: list[SoakEvent],
        rng: random.Random,
    ) -> SoakActionRequest | None: ...

    @abc.abstractmethod
    async def execute(
        self, request: SoakActionRequest, *, report_applied: Callable[[SoakActionEvidence], None]
    ) -> None: ...

    @abc.abstractmethod
    def is_recovered(self, *, action: SoakActionRecord, events: list[SoakEvent]) -> bool: ...

    @property
    def sut_event_file_patterns(self) -> tuple[str, ...]:
        return ()

    @property
    def sut_event_types(self) -> tuple[type[Event], ...]:
        return ()


def find_form(forms: SoakForms, *, kind: str, name: str) -> BaseSoakActionForm:
    matched = [form for form in forms[kind] if form.name == name]
    assert len(matched) == 1, f"Expected exactly one action form named {name!r} for {kind!r}, got {len(matched)}"
    return matched[0]


# ================================= observers ==================================


class SoakObserver(abc.ABC):
    @abc.abstractmethod
    async def observe(self) -> SoakObservationEvent: ...
