from __future__ import annotations

from miles.utils.workers.worker_provider.kubernetes.helm.naming import CHART_NAME

ORCHESTRATOR_COMPONENT = "orchestrator"


class RunNames:
    @staticmethod
    def release(*, run_id: str) -> str:
        return f"{CHART_NAME}-{run_id}"
