TOP_SPEC = ("miles_plugins.top.spec", "get_top_spec")


def _contract_v1() -> str:
    # Deferred: SGLang's true-on-policy package is required only by runs that select a TOP contract.
    from sglang.srt.true_on_policy.contracts import TRUE_ON_POLICY_V1

    return TRUE_ON_POLICY_V1


def select_true_on_policy(args) -> None:
    """``--true-on-policy-mode`` is the one switch: it selects the only contract and the TOP block spec.

    ``--true-on-policy-contract`` also turns TOP on. Both may still be spelled out, but only with these values, so
    the trainer spec, the trainer mode and the engines' contract are on or off together.
    """
    spec = args.spec
    contract = args.true_on_policy_contract
    if not (args.true_on_policy_mode or contract):
        if spec is not None and tuple(spec) == TOP_SPEC:
            raise ValueError("--spec miles_plugins.top.spec get_top_spec needs --true-on-policy-mode")
        return
    v1 = _contract_v1()
    if contract not in (None, v1):
        raise ValueError(f"unknown --true-on-policy-contract {contract!r}; the only contract is {v1}")
    if spec is not None and tuple(spec) != TOP_SPEC:
        raise ValueError(f"--true-on-policy-mode builds the TOP block spec; remove --spec {' '.join(spec)}")
    args.true_on_policy_mode = True
    args.true_on_policy_contract = v1
    args.spec = list(TOP_SPEC)


def uses_true_on_policy_v1(args) -> bool:
    """Whether the run selected the v1 contract with ``--true-on-policy-contract``."""
    contract = getattr(
        args, "true_on_policy_contract", None
    )  # config-access-exempt: weight converters and tools pass partial namespaces that carry no TOP contract
    return contract is not None and contract == _contract_v1()
