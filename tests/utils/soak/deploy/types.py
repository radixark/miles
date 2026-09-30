from pathlib import Path
from typing import Literal

from tests.utils.deploy.hot_restart.cluster_observer import ClusterSnapshot

from miles.utils.pydantic_utils import FrozenStrictBaseModel

DEPLOYMENT_TARGET_KIND: str = "deployment"


class DeploymentTarget(FrozenStrictBaseModel):
    kind: Literal["deployment"] = "deployment"
    identity: str
    incarnation: str
    alive: bool
    ready: bool
    namespace: str
    release: str
    workload_stamps: dict[str, str | None]
    workload_uids: dict[str, str]
    saved_iteration: int | None
    finished_rollout_id: int | None
    state_file: Path | None
    uninstall_job_uid: str | None


class HotRestartDetails(FrozenStrictBaseModel):
    form: Literal["hot_restart"] = "hot_restart"


class HotRestartTakeOverEvidence(FrozenStrictBaseModel):
    kind: Literal["hot_restart_take_over"] = "hot_restart_take_over"
    after: DeploymentTarget


class DeploymentObservationDetails(FrozenStrictBaseModel):
    kind: Literal["deployment"] = "deployment"
    cluster: ClusterSnapshot
