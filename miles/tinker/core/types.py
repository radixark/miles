"""Shared types of the Tinker gateway.

Core speaks only the gateway's internal language (commands, rows, results).
server/ translates the SDK wire (JSON and proto); runtime.py translates
miles (trainer batches). Each foreign language lives only at its boundary.
"""

from dataclasses import dataclass, field


class UserInputError(Exception):
    """Rejected request content; fails the promise with category User."""


class OwnershipError(Exception):
    """model/checkpoint does not belong to the caller's tenant."""


@dataclass
class GatewayConfig:
    base_model: str
    n_slots: int
    checkpoint_root: str
    max_datums_per_request: int = 1024
    max_tokens_per_datum: int = 32768
    max_tokens_per_request: int = 4_000_000
    lora_alpha: float | None = None  # None: 2 * rank
    lease_timeout_s: float = 300.0  # sessions stale beyond this lose their sampling, models, and slots
    unit_token_budget: int = 262_144  # packing bound per work unit


@dataclass
class Command:
    model_id: str
    seq_id: int
    kind: str
    payload: dict
    request_id: str
    arrival: int  # global submit order, the planner's FCFS key


@dataclass
class ModelRecord:
    model_id: str
    tenant: str
    slot: int
    base_model: str
    lora_rank: int
    lora_alpha: float
    session_id: str
    sampler_version: int = 0
    user_metadata: dict = field(default_factory=dict)
