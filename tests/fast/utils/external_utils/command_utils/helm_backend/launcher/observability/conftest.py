import time

from miles.utils.workers.k8s_types import (
    ContainerState,
    ContainerStateTerminated,
    ContainerStatus,
    Event,
    ObjectReference,
    Pod,
    PodCondition,
    PodMetadata,
    PodStatus,
)


def make_container(name="app", container_id="", previous_container_id="", running=False, restarts=0):
    return ContainerStatus(
        name=name,
        container_id=container_id,
        restart_count=restarts,
        state=ContainerState(running={} if running else None),
        last_state=ContainerState(
            terminated=ContainerStateTerminated(container_id=previous_container_id) if previous_container_id else None
        ),
    )


def make_pod(name="p", uid="u", phase="Running", ready=True, restarts=0, scheduling_gated=False, containers=()) -> Pod:
    conditions = [PodCondition(type="Ready", status="True" if ready else "False")]
    if scheduling_gated:
        conditions.append(PodCondition(type="PodScheduled", status="False", reason="SchedulingGated"))
    containers = containers or ((make_container(restarts=restarts),) if restarts else ())
    return Pod(
        metadata=PodMetadata(name=name, uid=uid),
        status=PodStatus(phase=phase, conditions=conditions, container_statuses=list(containers)),
    )


def make_event(pod_name="p", reason="FailedScheduling", message="no node", count=1, event_type="Warning"):
    return Event(
        involved_object=ObjectReference(name=pod_name, kind="Pod"),
        reason=reason,
        message=message,
        count=count,
        type=event_type,
    )


def wait_for(condition, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("the condition the test waited for never happened")
