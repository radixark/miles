from __future__ import annotations

from .schema import TRUE_ON_POLICY_V1_SCHEMA, TrueOnPolicyContractSchema

TRUE_ON_POLICY_V1 = TRUE_ON_POLICY_V1_SCHEMA

_CONTRACT_BY_NAME = {
    TRUE_ON_POLICY_V1.name: TRUE_ON_POLICY_V1,
}


def get_true_on_policy_contract(name: str) -> TrueOnPolicyContractSchema:
    try:
        return _CONTRACT_BY_NAME[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_CONTRACT_BY_NAME))
        raise ValueError(f"Unsupported true-on-policy contract {name!r}. Supported contracts: {supported}") from exc
