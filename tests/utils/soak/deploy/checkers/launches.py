from tests.utils.soak.core.events import SoakEvent, SoakLaunchFinishedEvent
from tests.utils.soak.core.views import project_actions
from tests.utils.soak.deploy.types import DEPLOYMENT_TARGET_KIND


def assert_hot_restart_launches_finished(events: list[SoakEvent]) -> None:
    actions = {
        request_id: action
        for request_id, action in project_actions(events).items()
        if action.requested.request.target.kind == DEPLOYMENT_TARGET_KIND
    }
    order: list[str | None] = [None, *actions]
    for finished in events:
        if not isinstance(finished, SoakLaunchFinishedEvent):
            continue
        assert finished.outcome != "failed", f"Launcher failed: {finished}"
        if finished.outcome != "replaced":
            continue
        index = order.index(finished.request_id)
        assert index + 1 < len(order), f"The final launcher was replaced with no successor: {finished}"
        successor = actions[order[index + 1]]
        assert (
            successor.applied is not None and successor.requested.timestamp <= finished.timestamp
        ), f"No applied successor explains the replacement: {finished}"
