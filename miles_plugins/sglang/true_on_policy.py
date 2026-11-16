"""SGLang hooks for Miles' true-on-policy contracts.

SGLang loads this module through the ``sglang.srt.plugins`` entry point in every engine process. SGLang
ships only ``qwen3_dense_true_on_policy_v1`` (Qwen3 dense on Hopper); this adds every Miles contract to
its contract table under Miles' name (the Hopper one as ``qwen3_dense_hopper_bf16_true_on_policy_v1``).
Without it the engines reject the contract name at startup.
"""


def _register_contracts() -> None:
    """SGLang's runtime policy does not depend on the contract, so each Miles contract gets the same
    switches as SGLang's own; the attention backend reaches SGLang as a server flag."""
    from dataclasses import fields

    from sglang.srt.true_on_policy import contracts
    from sglang.srt.true_on_policy.schema import TrueOnPolicyContractSchema

    from miles.true_on_policy.contracts import all_true_on_policy_contracts

    for contract in all_true_on_policy_contracts():
        schema = TrueOnPolicyContractSchema(
            **{f.name: getattr(contract.schema, f.name) for f in fields(TrueOnPolicyContractSchema)}
        )
        contracts._CONTRACT_BY_NAME.setdefault(schema.name, contracts.SGLangTrueOnPolicyContract(schema=schema))


def register() -> None:
    _register_contracts()
