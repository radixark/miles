from tests.fast.import_isolation_utils import run_with_blocked_imports


def test_runs_without_a_contract_do_not_need_the_top_package():
    """Every Megatron forward and the raw HF load ask whether the run is a TOP run; without a contract, the answer
    must not require SGLang's true-on-policy package."""
    run_with_blocked_imports(
        script="""
from types import SimpleNamespace

from miles.true_on_policy import uses_true_on_policy_v1
from miles.true_on_policy.contracts import select_true_on_policy
from miles.true_on_policy.gdn import is_top_gdn, is_top_glm53

args = SimpleNamespace(true_on_policy_mode=False, true_on_policy_contract=None, spec=None, hf_checkpoint="/none")
select_true_on_policy(args)
assert not uses_true_on_policy_v1(args)
assert not uses_true_on_policy_v1(SimpleNamespace())
assert not is_top_gdn(args)
assert not is_top_glm53(args)
""",
        blocked_prefixes=("sglang.srt.true_on_policy",),
    )
