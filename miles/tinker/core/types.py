"""Shared types of the Tinker gateway.

Core speaks only the gateway's internal language (commands, datums, results).
server/ translates the SDK wire (JSON and proto); runtime.py translates
miles (trainer batches). Each foreign language lives only at its boundary.
"""

from dataclasses import dataclass, field
from enum import Enum


class CommandOp(str, Enum):
    FORWARD_BACKWARD = "forward_backward"
    FORWARD_ONLY = "forward_only"
    OPTIM_STEP = "optim_step"
    SAVE_STATE = "save_state"
    LOAD_STATE = "load_state"
    SAVE_WEIGHTS_FOR_SAMPLER = "save_weights_for_sampler"

    def is_batch(self) -> bool:
        """Batch ops pack into BatchUnits; every other op is a barrier."""
        return self in (CommandOp.FORWARD_BACKWARD, CommandOp.FORWARD_ONLY)


# wire loss_fn_inputs key -> internal datum key
LOSS_INPUT_KEYS = {"weights": "weights", "advantages": "advantages", "logprobs": "sampling_logprobs"}

# the wire inputs each loss_fn reads from every datum; admission rejects what
# execution would trip over, before the datum can poison a shared batch
LOSS_FN_INPUTS = {
    "cross_entropy": ("weights",),
    "importance_sampling": ("logprobs", "advantages"),
    "ppo": ("logprobs", "advantages"),
    "cispo": ("logprobs", "advantages"),
    "dro": ("logprobs", "advantages"),
}


class UserInputError(Exception):
    """Rejected request content; fails the future with category User."""


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
    max_samples_per_request: int = 64
    lora_alpha: float | None = None  # None: 2 * rank
    lease_timeout_s: float = 300.0  # sessions stale beyond this lose their sampling, models, and slots
    batch_token_budget: int = 262_144  # packing bound per BatchUnit


@dataclass
class Command:
    model_id: str
    seq_id: int
    op: CommandOp
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
    # The mint counter: a failed publication burns its number, never reuses it,
    # so published_sampler_versions can have gaps.
    next_sampler_version: int = 1
    published_sampler_versions: set[int] = field(default_factory=set)
    user_metadata: dict = field(default_factory=dict)
