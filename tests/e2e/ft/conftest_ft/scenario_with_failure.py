# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import json
from pathlib import Path

from tests.e2e.ft.conftest_ft.app import create_comparison_app_and_run_ci
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_ft_args, get_train_env_vars_arg
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.test_utils.comparisons.dumps import (
    INPUT_TENSORS_ALLOW_FAILED_PATTERN,
    INPUT_TENSORS_SKIP_PATTERN,
    compare_dumps,
)
from miles.utils.test_utils.comparisons.metrics import compare_metrics
from miles.utils.test_utils.reconfigure_assertions import ReconfigureInfo, assert_reconfigure_events

NUM_PHASE_A_STEPS: int = 1
NUM_PHASE_B_STEPS: int = 4

# Per-tensor pass predicates. A few specific near-zero grads diverge under the
# crash-recovery (solo / degraded-quorum) collective's reduction order while their
# weights stay bit-identical:
#   - starved MoE expert grads (grad__...mlp.experts.*), max_abs ~1e-5..4e-4;
#   - attention q/k_layernorm grads (grad__...q_layernorm.* / ...k_layernorm.*, the two
#     QK-norm weights), measured rel ~1.1-1.5%, max_abs_diff ~6e-5..2.3e-4, failing
#     layer varies per rollout -> FP noise.
# Both are cancellation-dominated near-zero grads, so they also tolerate
# max_abs <= 1e-3 (well below real grads ~1e-2): a real diff there still fails, and
# everything else stays strict via the catch-all (an unmatched tensor is fail-closed).
_DIFF_THRESHOLDS: list[tuple[str, str]] = [
    (r"grad__.*\.mlp\.experts\..*", "rel <= 0.0085 or max_abs <= 1e-3"),
    (r"grad__.*\.[qk]_layernorm\..*", "rel <= 0.0085 or max_abs <= 1e-3"),
    (".*", "rel <= 0.0085"),
]

# Post-fault (injected) rollouts in the real_rollout mode: training data is injected to be
# bitwise-identical, but the target's weights carry the fault-inherent ulp drift of the
# degraded-quorum commit. On the converged dense model that drift lands in the
# cancellation-dominated near-zero grads of the decoder-layer norms and attention/MLP
# matrices as absolute noise measured <= 2.8e-3 (40 tensors, 2026-06-12; q_layernorm up to
# rel 20% at max_abs 2.6e-3) while real grads sit at ~1e-2 — only those measured families
# get a 3e-3 floor. Everything else (embeddings, output layer, final norm, all
# activations/values) stays strict, and all passed at rel <= 0.85% in the same run.
_POST_FAULT_DIFF_THRESHOLDS: list[tuple[str, str]] = [
    (r"grad__.*\.[qk]_layernorm\..*", "rel <= 0.0085 or max_abs <= 3e-3"),
    (r"grad__.*\.layer_norm_weight", "rel <= 0.0085 or max_abs <= 3e-3"),
    (r"grad__.*\.self_attention\.linear_qkv\.weight", "rel <= 0.0085 or max_abs <= 3e-3"),
    (r"grad__.*\.self_attention\.linear_proj\.weight", "rel <= 0.0085 or max_abs <= 3e-3"),
    (r"grad__.*\.mlp\.linear_fc[12]\.weight", "rel <= 0.0085 or max_abs <= 3e-3"),
    (".*", "rel <= 0.0085"),
]

# rollout_id in phase_b starts from NUM_PHASE_A_STEPS (ckpt resume offset)
_WITH_FAILURE_ACTIONS: list[dict] = [
    {
        "at_rollout": NUM_PHASE_A_STEPS + 1,
        "action": "crash_before_allreduce",
        "cell_index": -1,
        "rank": 0,
        "attempt": 0,
    },
    {"at_rollout": NUM_PHASE_A_STEPS + 1, "action": "stop_cell_at_end", "cell_index": -1},
    {"at_rollout": NUM_PHASE_A_STEPS + 1, "action": "start_cell_at_end", "cell_index": -1},
]


def _expected_reconfigures(*, is_target: bool, phase: str, num_cells: int) -> list[ReconfigureInfo]:
    if not (is_target and phase == "phase_b"):
        return []
    return [
        ReconfigureInfo(
            rollout_id=NUM_PHASE_A_STEPS + 1,
            src_cell_index=None,
            healed_cell_indices=[],
            alive_cell_indices_after=list(range(num_cells - 1)),
        ),
        ReconfigureInfo(
            rollout_id=NUM_PHASE_A_STEPS + 2,
            src_cell_index=0,
            healed_cell_indices=[num_cells - 1],
            alive_cell_indices_after=list(range(num_cells)),
        ),
    ]


def _build_phase_args(mode: FTTestMode, dump_dir: str, *, is_target: bool, enable_dumper: bool = True) -> str:
    is_phase_a: bool = dump_dir.endswith("phase_a")
    base = get_common_train_args(mode, dump_dir=dump_dir, num_steps=NUM_PHASE_B_STEPS, enable_dumper=enable_dumper)
    base += get_train_env_vars_arg(mode, deterministic=False)

    if is_target:
        base += get_ft_args(mode)

    if is_phase_a:
        base += f"--save {dump_dir}/ckpt --save-interval 1 "
        base += f"--debug-exit-after-rollout {NUM_PHASE_A_STEPS} "
    else:
        phase_a_dir = dump_dir.replace("/phase_b", "/phase_a")
        base += f"--load {phase_a_dir}/ckpt "
        if is_target:
            base += f"--ci-ft-test-actions '{json.dumps(_WITH_FAILURE_ACTIONS)}' "
            if mode.has_real_rollout:
                # Post-fault rollouts inject the baseline's recorded data (see README).
                baseline_dump_dir = dump_dir.replace("/target/", "/baseline/")
                base += (
                    f"--ci-inject-rollout-data-path {baseline_dump_dir}/rollout_data/{{rollout_id}}.pt "
                    f"--ci-inject-rollout-data-start-rollout-id {NUM_PHASE_A_STEPS + 2} "
                    "--ci-inject-rollout-data-min-match-ratio 0.5 "
                )

    return base


def _build_baseline_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return _build_phase_args(mode, dump_dir, is_target=False, enable_dumper=enable_dumper)


def _build_target_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return _build_phase_args(mode, dump_dir, is_target=True, enable_dumper=enable_dumper)


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    for side in ["baseline", "target"]:
        for phase in PHASES:
            assert_reconfigure_events(
                Path(f"{dump_dir}/{side}/{phase}/events"),
                expected=_expected_reconfigures(is_target=side == "target", phase=phase, num_cells=mode.num_cells),
            )

    compare_metrics(
        baseline_dir=f"{dump_dir}/baseline/phase_b",
        target_dir=f"{dump_dir}/target/phase_b",
        rtol=5e-2,
        atol=1e-7,
        key_prefixes=["train/"],
        exclude_keys=[],
    )

    phase_b_rollout_ids = range(NUM_PHASE_A_STEPS, NUM_PHASE_B_STEPS)
    expected_leaves = {f"fwd_bwd/rollout_{rollout_id}" for rollout_id in phase_b_rollout_ids}
    actual_leaves = {
        str(p.parent.relative_to(Path(f"{dump_dir}/baseline/phase_b/dumps")))
        for p in Path(f"{dump_dir}/baseline/phase_b/dumps").rglob("*.pt")
    }
    assert actual_leaves == expected_leaves, (
        f"Dump leaves {actual_leaves} do not match the per-rollout comparison loop "
        f"{expected_leaves}; a new leaf would silently skip comparison"
    )

    first_injected_rollout_id = NUM_PHASE_A_STEPS + 2
    for rollout_id in phase_b_rollout_ids:
        is_post_fault = mode.has_real_rollout and rollout_id >= first_injected_rollout_id
        compare_dumps(
            baseline_dir=f"{dump_dir}/baseline/phase_b",
            target_dir=f"{dump_dir}/target/phase_b",
            diff_thresholds=_POST_FAULT_DIFF_THRESHOLDS if is_post_fault else _DIFF_THRESHOLDS,
            allow_skipped_pattern=INPUT_TENSORS_SKIP_PATTERN,
            allow_failed_pattern=INPUT_TENSORS_ALLOW_FAILED_PATTERN,
            phase_subdir=f"fwd_bwd/rollout_{rollout_id}",
        )
    print("With-failure comparison test PASSED")


TEST_NAME: str = "trainer_ft_with_failure"
PHASES: list[str] = ["phase_a", "phase_b"]


app, run_ci = create_comparison_app_and_run_ci(
    test_name=TEST_NAME,
    build_baseline_args=_build_baseline_args,
    build_target_args=_build_target_args,
    compare_fn=_compare,
    phases=PHASES,
)

if __name__ == "__main__":
    app()
