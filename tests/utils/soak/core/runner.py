from dataclasses import dataclass

from tests.utils.soak.core.config import SoakRunnerConfig
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.types import SoakForms, SoakObserver


@dataclass(kw_only=True)
class SoakRunner:
    observer: SoakObserver
    forms: SoakForms
    event_log: EventLog
    config: SoakRunnerConfig
