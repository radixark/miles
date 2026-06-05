# NOTE: You MUST read tests/e2e/ft/README.md as source-of-truth and documentations
# WARNING: Do NOT relax any assert logic in this file. All assertions must remain strict.

import json
import sys
from pathlib import Path

_MILES_ROOT: Path = Path(__file__).resolve().parents[3]
_miles_root_str = str(_MILES_ROOT)
if _miles_root_str in sys.path:
    sys.path.remove(_miles_root_str)
sys.path.insert(0, _miles_root_str)

from tests.e2e.ft.conftest_ft.app import create_comparison_app
from tests.e2e.ft.conftest_ft.execution import get_common_train_args, get_ft_args
from tests.e2e.ft.conftest_ft.modes import FTTestMode

from miles.utils.test_utils.comparisons import compare_dumps, compare_metrics

NUM_PHASE_A_STEPS: int = 1
NUM_PHASE_B_STEPS: int = 3

_DETERMINISTIC_ENV_VARS: str = (
    '--train-env-vars \'{"NCCL_ALGO": "Ring", '
    '"NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0", '
    '"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}\' '
)

# rollout_id in phase_b starts from NUM_PHASE_A_STEPS (ckpt resume offset)
_DETERMINISTIC_ACTIONS: list[dict] = [
    {"at_rollout": NUM_PHASE_A_STEPS + 1, "action": "stop_cell_at_end", "cell_index": -1},
    {"at_rollout": NUM_PHASE_A_STEPS + 1, "action": "start_cell_at_end", "cell_index": -1},
]

def _build_phase_args(mode: FTTestMode, dump_dir: str, *, is_target: bool, enable_dumper: bool = True) -> str:
    is_phase_a: bool = dump_dir.endswith("phase_a")
    base = get_common_train_args(mode, dump_dir=dump_dir, num_steps=NUM_PHASE_B_STEPS, enable_dumper=enable_dumper)
    base += "--deterministic-mode " + _DETERMINISTIC_ENV_VARS

    if is_target:
        base += get_ft_args(mode)

    if is_phase_a:
        base += f"--save {dump_dir}/ckpt --save-interval 1 "
        base += f"--debug-exit-after-rollout {NUM_PHASE_A_STEPS} "
    else:
        phase_a_dir = dump_dir.replace("/phase_b", "/phase_a")
        base += f"--load {phase_a_dir}/ckpt "
        if is_target:
            base += f"--ci-ft-test-actions '{json.dumps(_DETERMINISTIC_ACTIONS)}' "

    return base


def _build_baseline_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return _build_phase_args(mode, dump_dir, is_target=False, enable_dumper=enable_dumper)


def _build_target_args(mode: FTTestMode, dump_dir: str, enable_dumper: bool = True) -> str:
    return _build_phase_args(mode, dump_dir, is_target=True, enable_dumper=enable_dumper)


def _compare(dump_dir: str, mode: FTTestMode) -> None:
    rtol: float = 3e-2 if mode.has_real_rollout else 1e-2
    atol: float = 2e-8 if mode.has_real_rollout else 1e-8
    compare_metrics(
        baseline_dir=f"{dump_dir}/baseline/phase_b",
        target_dir=f"{dump_dir}/target/phase_b",
        rtol=rtol,
        atol=atol,
        key_prefixes=["train/"],
    )
    compare_dumps(
        baseline_dir=f"{dump_dir}/baseline/phase_b",
        target_dir=f"{dump_dir}/target/phase_b",
    )
    print("Deterministic healing comparison test PASSED")


app = create_comparison_app(
    test_name=Path(__file__).stem,
    build_baseline_args=_build_baseline_args,
    build_target_args=_build_target_args,
    compare_fn=_compare,
    phases=["phase_a", "phase_b"],
)

if __name__ == "__main__":
    app()
