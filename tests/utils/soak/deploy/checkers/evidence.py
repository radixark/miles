from tests.utils.deploy.hot_restart.evidence import HotRestartEvidence
from tests.utils.soak.core.events import SoakEvent


def project_hot_restart_evidence(events: list[SoakEvent], *, release: str, namespace: str) -> HotRestartEvidence:
    raise NotImplementedError
