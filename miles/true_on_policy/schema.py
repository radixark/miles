from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TrueOnPolicyContractName = Literal["true_on_policy_v1"]


@dataclass(frozen=True)
class TrueOnPolicyContractSchema:
    """Cross-repo identity of a true-on-policy program version.

    Names a version of the numerical program, not a model. Model-specific facts (attention
    backend, which roles bind) live where they are consumed.
    """

    name: TrueOnPolicyContractName
    disable_megatron_sequence_parallel: bool


TRUE_ON_POLICY_V1_SCHEMA = TrueOnPolicyContractSchema(
    name="true_on_policy_v1",
    disable_megatron_sequence_parallel=True,
)
