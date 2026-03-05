from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AgentResult:
    """Standardized result returned by an agent function to the generate layer."""
    reward: float = 0.0
    status: Literal["completed", "truncated", "aborted"] = "completed"
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
