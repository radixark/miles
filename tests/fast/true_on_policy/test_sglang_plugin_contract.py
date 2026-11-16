from dataclasses import replace
from types import SimpleNamespace

import pytest


def test_plugin_registers_every_miles_contract_with_sglang_runtime_policy():
    pytest.importorskip("sglang")
    from sglang.srt.true_on_policy import contracts

    from miles.true_on_policy.contracts import all_true_on_policy_contracts
    from miles_plugins.sglang.true_on_policy import _register_contracts

    _register_contracts()
    server_args = SimpleNamespace(tp_size=1)
    sglang_own = contracts.get_true_on_policy_contract("qwen3_dense_true_on_policy_v1").policy_for(server_args)
    for contract in all_true_on_policy_contracts():
        registered = contracts.get_true_on_policy_contract(contract.name).policy_for(server_args)
        assert registered == replace(sglang_own, contract_name=contract.name)
