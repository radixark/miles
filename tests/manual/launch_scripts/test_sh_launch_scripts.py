from dataclasses import dataclass, field

import pytest
from tests.fast.launch_scripts.sh_harness import (
    REPO_ROOT,
    assert_matches_snapshot,
    format_invocations,
    iter_launch_scripts,
    run_launch_script,
)

_SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "launch_scripts" / "sh"


@dataclass(frozen=True)
class LaunchScriptCase:
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


_HEAD_NODE_IP = "10.0.0.1"

_SCRIPTS_REFUSING_TO_RUN_WITHOUT_EXPLICIT_INPUTS: dict[str, LaunchScriptCase] = {
    "examples/lora/run-qwen2.5-3B-megatron-lora-disaggregated-multi-node.sh": LaunchScriptCase(args=("p2p", "0")),
    "examples/on_policy_distillation/qwen3_5_35b_selfdistill/phase1_rlvr_teacher.sh": LaunchScriptCase(
        env={"OUTPUT_DIR": "{workdir}"}
    ),
    "examples/on_policy_distillation/qwen3_5_35b_selfdistill/phase2_gb200.sh": LaunchScriptCase(
        env={"OUTPUT_DIR": "{workdir}"}
    ),
    "examples/on_policy_distillation/qwen3_5_35b_selfdistill/phase2_opd_selfdistill.sh": LaunchScriptCase(
        env={"OUTPUT_DIR": "{workdir}"}
    ),
    "examples/infra_features/p2p_weight_transfer/run-glm4.5-air-8node-profile.sh": LaunchScriptCase(
        args=("p2p", "0", _HEAD_NODE_IP), env={"MILES_LOG_DIR": "{workdir}"}
    ),
    "examples/infra_features/p2p_weight_transfer/run-glm4.7-flash-2node-profile.sh": LaunchScriptCase(
        args=("p2p", "0", _HEAD_NODE_IP)
    ),
    "examples/infra_features/p2p_weight_transfer/run-glm5-disagg-profile.sh": LaunchScriptCase(
        args=("GLM-5", "p2p", "0", _HEAD_NODE_IP), env={"MILES_LOG_DIR": "{workdir}"}
    ),
    "examples/infra_features/p2p_weight_transfer/run-kimi-k2-64node-profile.sh": LaunchScriptCase(
        args=("p2p", "0", _HEAD_NODE_IP)
    ),
    "examples/infra_features/p2p_weight_transfer/run-qwen3-235B-A22B-16node-profile.sh": LaunchScriptCase(
        args=("p2p", "0", _HEAD_NODE_IP)
    ),
    "examples/infra_features/p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh": LaunchScriptCase(
        args=("p2p", "0", _HEAD_NODE_IP)
    ),
}

_SCRIPTS = [script.relative_to(REPO_ROOT).as_posix() for script in iter_launch_scripts()]


@pytest.fixture(params=_SCRIPTS, scope="module")
def recorded(request, tmp_path_factory):
    rel = request.param
    case = _SCRIPTS_REFUSING_TO_RUN_WITHOUT_EXPLICIT_INPUTS.get(rel, LaunchScriptCase())
    tmp_path = tmp_path_factory.mktemp("launch_script")
    workdir = tmp_path / "workdir"
    run = run_launch_script(
        REPO_ROOT / rel,
        sandbox=tmp_path,
        args=case.args,
        extra_env={key: value.format(workdir=workdir) for key, value in case.env.items()},
    )
    return rel, run


class TestEveryLaunchScript:
    def test_invocations_match_snapshot(self, recorded):
        """Every launch script must issue exactly the recorded sequence of external commands."""
        rel, run = recorded
        snapshot = _SNAPSHOT_DIR / f"{rel}.txt"
        actual = f"# returncode: {run.returncode}\n\n{format_invocations(run.invocations)}"

        assert_matches_snapshot(snapshot, actual, rel)

    def test_submits_exactly_one_ray_job(self, recorded):
        """A launch script that no longer reaches `ray job submit` is broken, whatever else it does."""
        _, run = recorded
        assert run.returncode == 0
        assert len(run.ray_job_submit_argv()) > 10


class TestDiscovery:
    def test_every_discovered_script_has_a_snapshot_and_vice_versa(self):
        """A script that stops matching the discovery filter would otherwise vanish silently."""
        discovered = {f"{rel}.txt" for rel in _SCRIPTS}
        recorded = {path.relative_to(_SNAPSHOT_DIR).as_posix() for path in _SNAPSHOT_DIR.rglob("*.txt")}

        assert discovered == recorded
        assert len(discovered) > 30

    def test_a_shell_script_outside_the_snapshots_only_delegates(self):
        """Shell launchers stay Ray only, so the snapshots are their whole protection against drift."""
        recorded = {path.stem for path in _SNAPSHOT_DIR.rglob("*.txt")}
        undiscovered = [
            path
            for path in (REPO_ROOT / "scripts").rglob("*.sh")
            if path.name not in recorded and "ray job submit" not in path.read_text(errors="replace")
        ]

        for path in undiscovered:
            text = path.read_text(errors="replace")
            delegates_to = [line for line in text.splitlines() if ".py" in line and "python" in line]
            assert delegates_to, f"{path.name} launches training without a snapshot covering it"
