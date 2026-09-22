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
from tests.utils.soak.ft.types import CellTarget, InjectFaultDetails, ObservedCellFault, PodDetails
from tests.utils.soak.k8s_utils.pod_manipulation import PodDeletedEvidence
from tests.utils.soak.k8s_utils.process_target import ProcessSignalReceipt

from miles.utils.pydantic_utils import FrozenStrictBaseModel

if TYPE_CHECKING:
    from tests.utils.soak.core.events import SoakEvent, SoakObservationEvent
    from tests.utils.soak.core.views import SoakActionRecord

SoakTarget = Annotated[CellTarget | DeploymentTarget, Discriminator("kind")]

SoakActionDetails = Annotated[InjectFaultDetails | PodDetails | HotRestartDetails, Discriminator("form")]

SoakActionEvidence = Annotated[
    ObservedCellFault | PodDeletedEvidence | ProcessSignalReceipt | HotRestartTakeOverEvidence,
    Discriminator("kind"),
]

SoakObservationDetails = DeploymentObservationDetails


class SoakActionRequest(FrozenStrictBaseModel):
    request_id: str = Field(default_factory=lambda: uuid4().hex)
    target: SoakTarget
    form_name: str
    next_due_at: float | None = None
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


def find_form(forms: SoakForms, *, kind: str, name: str) -> BaseSoakActionForm:
    matched = [form for form in forms[kind] if form.name == name]
    assert len(matched) == 1, f"Expected exactly one action form named {name!r} for {kind!r}, got {len(matched)}"
    return matched[0]


# ================================= observers ==================================


class SoakObserver(abc.ABC):
    @abc.abstractmethod
    async def observe(self) -> SoakObservationEvent: ...
