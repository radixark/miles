"""Shared commands, datums, and configuration for the Tinker gateway.

An internal datum has tokens = model_input + target_tokens[-1:] and explicit target_tokens.
Its target_len counts every label position, including prompt positions; loss inputs align to it.
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

    def changes_training_state(self) -> bool:
        return self in (CommandOp.FORWARD_BACKWARD, CommandOp.OPTIM_STEP, CommandOp.LOAD_STATE)


# wire loss_fn_inputs key -> internal datum key
LOSS_INPUT_KEYS = {"weights": "weights", "advantages": "advantages", "logprobs": "sampling_logprobs"}

# reject missing loss inputs before they enter a shared batch
LOSS_FN_INPUTS = {
    "cross_entropy": ("weights",),
    "importance_sampling": ("logprobs", "advantages"),
    "ppo": ("logprobs", "advantages"),
    "cispo": ("logprobs", "advantages"),
    "dro": ("logprobs", "advantages"),
}


class UserInputError(Exception):
    """Rejected request content; fails the future with category user."""


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
    max_lora_rank: int = 32  # the slot capacity the server was built with (--lora-rank)
    lora_alpha: float | None = None  # None: 2 * rank
    # what the server-wide adapter layout trains; create_model rejects deviations
    trains_attn: bool = True
    trains_mlp: bool = True
    trains_unembed: bool = False
    lease_timeout_s: float = 300.0  # sessions stale beyond this lose their sampling, models, and slots
    batch_token_budget: int = 262_144  # packing bound per BatchUnit


@dataclass
class Command:
    model_id: str
    seq_id: int
    op: CommandOp
    payload: dict
    request_id: str
    arrival: int  # global submit order for selecting the planner's seed
    validation_error: str | None = None


@dataclass
class ModelRecord:
    model_id: str
    tenant: str
    slot: int
    base_model: str
    lora_rank: int
    lora_alpha: float
    # failed publications burn their version number, leaving gaps
    next_sampler_version: int = 1
    published_sampler_versions: set[str] = field(default_factory=set)
