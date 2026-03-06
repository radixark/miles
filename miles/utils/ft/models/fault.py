from enum import Enum

from pydantic import Field

from miles.utils.ft.models.base import FtBaseModel


class ActionType(str, Enum):
    NONE = "none"
    MARK_BAD_AND_RESTART = "mark_bad_and_restart"
    ENTER_RECOVERY = "enter_recovery"
    NOTIFY_HUMAN = "notify_human"


class TriggerType(str, Enum):
    NONE = ""
    HANG = "hang"
    NAN_LOSS = "nan_loss"
    CRASH = "crash"


class NodeFault(FtBaseModel):
    node_id: str
    reason: str
    ephemeral: bool = False


class Decision(FtBaseModel):
    action: ActionType
    bad_node_ids: list[str] = Field(default_factory=list)
    reason: str
    trigger: TriggerType = TriggerType.NONE

    @classmethod
    def no_fault(cls, reason: str) -> "Decision":
        return cls(action=ActionType.NONE, reason=reason)

    @classmethod
    def from_node_faults(
        cls,
        faults: "list[NodeFault]",
        *,
        fallback_reason: str,
    ) -> "Decision":
        if not faults:
            return cls(action=ActionType.NONE, reason=fallback_reason)

        return cls(
            action=ActionType.MARK_BAD_AND_RESTART,
            bad_node_ids=sorted(unique_node_ids(faults)),
            reason="; ".join(f.reason for f in faults),
        )


def unique_node_ids(faults: list["NodeFault"]) -> list[str]:
    """Return deduplicated node IDs from faults, preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for fault in faults:
        if fault.node_id not in seen:
            seen.add(fault.node_id)
            result.append(fault.node_id)
    return result
