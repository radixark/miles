"""A disabled CI test is a promise to come back; its reason says where that promise is tracked.

``disabled=`` keeps a registration in the plan as a reported skip instead of
deleting it, which only pays off if someone returns to it. "Disabled due to
bugs." names neither the bug nor who owns the fix, and 41 registrations had
accumulated with reasons like that and nothing to follow. A reason therefore
cites the issue or PR that tracks re-enabling the test, as ``#123`` or a GitHub
issue / pull URL.

``_UNLINKED_BEFORE_THE_RULE`` spares the registrations disabled before this rule,
and it only shrinks: the second test fails once an entry is re-enabled or gains
a link, so a stale exception cannot linger.
"""

import re
from pathlib import Path

from tests.ci.ci_register import collect_tests, discover_ci_files, register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

REPO_ROOT = Path(__file__).resolve().parents[3]

_TRACKING_REF = re.compile(r"#\d+|https://github\.com/[\w.-]+/[\w.-]+/(?:issues|pull)/\d+")

# (file, suite) registrations whose reason predates the rule. Delete an entry when
# the test is re-enabled, moves suite, or its reason gains a link; never add one.
_UNLINKED_BEFORE_THE_RULE: frozenset[tuple[str, str]] = frozenset(
    {
        ("tests/e2e/agentic/test_harbor_rollout.py", "stage-c-2-gpu-h200"),
        ("tests/e2e/ckpt/test_glm47_flash_ckpt.py", "stage-c-8-gpu-h100"),
        ("tests/e2e/deploy/test_hot_restart_checkpointed.py", "stage-c-8-gpu-h200"),
        ("tests/e2e/deploy/test_hot_restart_no_checkpoint.py", "stage-c-8-gpu-h200"),
        ("tests/e2e/deploy/test_hot_restart_realistic_gsm8k.py", "stage-c-8-gpu-h200"),
        ("tests/e2e/deploy/test_split_deterministic.py", "stage-c-8-gpu-h200"),
        ("tests/e2e/deploy/test_split_multi_policy.py", "stage-c-4-gpu-h200"),
        ("tests/e2e/ft/test_random_crash__kill_rollout__dp4__colocate.py", "stage-c-8-gpu-h200"),
        ("tests/e2e/ft/test_random_crash__kill_train__dp2_cp2__moe_5layer.py", "stage-c-8-gpu-h200"),
        (
            "tests/e2e/ft/test_random_crash__kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer.py",
            "stage-c-8-gpu-h200",
        ),
        ("tests/e2e/ft/test_random_crash__kill_train_rollout__dp2_cp2.py", "stage-c-8-gpu-h200"),
        ("tests/e2e/ft/test_random_crash_fully_async__kill_train_rollout__dp2_cp2.py", "stage-c-8-gpu-h200"),
        ("tests/e2e/ft/test_realistic_gsm8k__kill_train_rollout.py", "stage-c-8-gpu-h200"),
        ("tests/e2e/ft/test_realistic_gsm8k_fully_async__kill_train_rollout.py", "stage-c-8-gpu-h200"),
        ("tests/e2e/ft/test_rollout_deterministic__kill_rollout__dp4__colocate.py", "stage-c-8-gpu-h200"),
        ("tests/e2e/ft/test_trainer_deterministic__kill_train__dp2_cp2__moe_5layer.py", "stage-c-8-gpu-h200"),
        (
            "tests/e2e/ft/test_trainer_deterministic__kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer.py",
            "stage-c-8-gpu-h200",
        ),
        (
            "tests/e2e/ft/test_trainer_deterministic__kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer.py",
            "stage-c-8-gpu-h200",
        ),
        (
            "tests/e2e/ft/test_trainer_deterministic__kill_train__dp4_cp2__fake_rollout__moe_5layer.py",
            "stage-c-8-gpu-h200",
        ),
        ("tests/e2e/ft/test_trainer_with_failure__kill_train__dp2_cp2.py", "stage-c-8-gpu-h200"),
        (
            "tests/e2e/ft/test_trainer_with_failure__kill_train__dp2_cp2_pp2__fake_rollout__moe_5layer.py",
            "stage-c-8-gpu-h200",
        ),
        (
            "tests/e2e/ft/test_trainer_with_failure__kill_train__dp2_cp2_tp2_ep2__fake_rollout__moe_5layer.py",
            "stage-c-8-gpu-h200",
        ),
        ("tests/e2e/megatron/model_scripts/test_deepseek_v32_5layer_mxfp8.py", "stage-c-8-gpu-b200"),
        ("tests/e2e/megatron/model_scripts/test_glm5_1_744b_a40b_6layer_lora_ci.py", "stage-c-8-gpu-h200"),
        ("tests/e2e/megatron/model_scripts/test_glm5_2_744b_a40b_5layer_lora_ci.py", "stage-c-4-gpu-h200"),
        ("tests/e2e/megatron/model_scripts/test_glm5_3_flash_4layer_ci.py", "stage-c-8-gpu-h200"),
        (
            "tests/e2e/megatron/test_glm47_flash/test_amd_spec_mtptrain_r3_bf16_sgl_tp4_meg_tp2pp2.py",
            "stage-c-4-gpu-mi350",
        ),
        ("tests/e2e/megatron/test_qwen3_30B_A3B/test_amd_moriep_fp8_bridge.py", "stage-c-4-gpu-mi350"),
        ("tests/e2e/megatron/test_qwen3_30B_A3B/test_baseline.py", "nightly-stage-c-4-gpu-mi350"),
        ("tests/e2e/megatron/test_qwen3_30B_A3B/test_baseline.py", "stage-c-4-gpu-h200"),
        ("tests/e2e/megatron/test_qwen3_30B_A3B/test_fully_async.py", "nightly-stage-c-4-gpu-mi350"),
        ("tests/e2e/megatron/test_qwen3_30B_A3B/test_fully_async.py", "stage-c-4-gpu-h200"),
        ("tests/e2e/megatron/test_qwen3_30B_A3B/test_r3_baseline.py", "nightly-stage-c-4-gpu-mi350"),
        ("tests/e2e/megatron/test_qwen3_30B_A3B/test_r3_baseline.py", "stage-c-4-gpu-h200"),
        ("tests/e2e/megatron/test_qwen3_30B_A3B/test_r3_deepep_fp8.py", "stage-c-4-gpu-h200"),
        ("tests/e2e/megatron/test_qwen3_5_35B_A3B_cp.py", "stage-c-8-gpu-h100"),
        ("tests/e2e/precision/test_qwen3_0.6B_parallel_check.py", "stage-c-8-gpu-h100"),
        ("tests/e2e/sglang/test_r3_router_equivalence.py", "stage-c-4-gpu-h200"),
        ("tests/e2e/sglang/test_session_server_multi_role/test_minimax_m27.py", "stage-c-4-gpu-h200"),
        ("tests/fast-gpu/test_mxfp8_quantizer.py", "stage-b-2-gpu-h200"),
        ("tests/fast-gpu/test_semaphore.py", "stage-b-2-gpu-h200"),
    }
)


def _unlinked_disabled(monkeypatch) -> list[tuple[str, str]]:
    """(file, suite) of each disabled registration whose reason cites no issue."""
    # discover_ci_files() globs repo-relative paths, so it reads whatever cwd the
    # runner was started from; pin it to the checkout.
    monkeypatch.chdir(REPO_ROOT)
    return [
        (r.filename, r.suite)
        for r in collect_tests(discover_ci_files())
        if r.disabled is not None and not _TRACKING_REF.search(r.disabled)
    ]


def test_every_disabled_registration_cites_its_tracking_issue(monkeypatch):
    unlinked = sorted(key for key in _unlinked_disabled(monkeypatch) if key not in _UNLINKED_BEFORE_THE_RULE)
    assert unlinked == [], (
        "disabled= must cite the issue or PR that tracks re-enabling the test, "
        f'e.g. disabled="flaky step-3 KL on H200 (#1234)"; missing on: {unlinked}'
    )


def test_no_spared_entry_outlives_its_reason(monkeypatch):
    stale = sorted(_UNLINKED_BEFORE_THE_RULE - set(_unlinked_disabled(monkeypatch)))
    assert stale == [], f"these exceptions are re-enabled or now cite an issue; delete them: {stale}"
