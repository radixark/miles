from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHARTS_DIR = REPO_ROOT / "charts"

BASE_VALUES: dict[str, list[str]] = {
    "miles-workbench": ["--set", "objectName=lint-miles-workbench"],
}

SHARED_INFRA_VARIANTS: list[list[str]] = [
    ["--set", "infra.sharedStorage.type=pvc", "--set", "infra.sharedStorage.pvcClaimName=shared"],
    ["--set", "infra.sharedStorage.type=none"],
    ["--set", "infra.paths.repos.miles=alice/miles", "--set", "infra.paths.repos.megatron=alice/Megatron-LM"],
]

VARIANTS: dict[str, list[list[str]]] = {
    "miles-workbench": [
        *SHARED_INFRA_VARIANTS,
        ["--set", "rbac.create=false", "--set", "serviceAccount.name=preexisting"],
        ["--set", "rbac.leaderWorkerSets=false"],
    ],
    "miles-run": [
        *SHARED_INFRA_VARIANTS,
        ["--set-json", 'run.orchestrator.command=["python","train.py"]'],
        [
            "--set-json",
            'run.staticWorkers=[{"name":"router","objectName":"lint-router",'
            '"command":["python","-m","router"],"ports":[{"name":"http","port":30000}]}]',
        ],
        [
            "--set",
            "commandJob.enabled=true",
            "--set",
            "commandJob.name=convert",
            "--set",
            "commandJob.objectName=lint-convert",
            "--set-json",
            'commandJob.command=["bash"]',
        ],
        [
            "--set-json",
            'run.inferenceEngines=[{"name":"engine","objectName":"lint-engine","replicas":2,"size":4,'
            '"command":["python","-m","sglang.launch_server"],"resources":{"limits":{"nvidia.com/gpu":8}}}]',
        ],
        [
            "--set-json",
            'run.inferenceEngines=[{"name":"prefill","objectName":"lint-prefill","replicas":1,"size":4,"command":["python"]},'
            '{"name":"decode","objectName":"lint-decode","replicas":8,"command":["python"]}]',
        ],
        [
            "--set-json",
            'run.trainerEngines=[{"name":"trainer-engine-actor","objectName":"lint-trainer-engine-actor","replicas":2,"size":2,'
            '"command":["python","-m","supervisor"]},'
            '{"name":"trainer-engine-critic","objectName":"lint-trainer-engine-critic","command":["python","-m","supervisor"]}]',
        ],
        [
            "--set-json",
            'run.colocate={"namespace":"lint","release":"lint","trainer_pool_id":"lint-t",'
            '"inference_pools":[{"pool_id":"lint-e","layout":{"num_inference_cells":2,"num_trainer_cells":2,'
            '"num_pods_per_inference_cell":1,"num_pods_per_trainer_cell":2,"num_gpus_per_node":8,"gpu_offset":0}},'
            '{"pool_id":"lint-f","layout":{"num_inference_cells":2,"num_trainer_cells":2,'
            '"num_pods_per_inference_cell":1,"num_pods_per_trainer_cell":2,"num_gpus_per_node":8,'
            '"gpu_offset":16}}]}',
            "--set-json",
            'run.inferenceEngines=[{"name":"e","objectName":"lint-e","replicas":2,"command":["python"]},'
            '{"name":"f","objectName":"lint-f","replicas":2,"command":["python"]}]',
            "--set-json",
            'run.trainerEngines=[{"name":"t","objectName":"lint-t","replicas":2,"size":2,"command":["python"]}]',
        ],
    ],
}


REJECTED_VARIANTS: dict[str, list[list[str]]] = {
    "miles-workbench": [["--set", "infra.env.PYTHONPATH=/somewhere"]],
    "miles-run": [["--set", "infra.env.PYTHONPATH=/somewhere"]],
}


def run(command: list[str]) -> subprocess.CompletedProcess:
    print("+ " + " ".join(command), file=sys.stderr)
    return subprocess.run(command, capture_output=True, text=True)


def all_charts() -> list[Path]:
    return sorted(chart_yaml.parent for chart_yaml in CHARTS_DIR.glob("*/Chart.yaml"))


def lint_chart(chart: Path) -> bool:
    if (chart / "Chart.lock").exists():
        built = run(["helm", "dependency", "build", str(chart)])
        if built.returncode != 0:
            print(built.stdout + built.stderr, file=sys.stderr)
            return False

    ok = True
    base = BASE_VALUES.get(chart.name, [])
    for extra in [[], *VARIANTS.get(chart.name, [])]:
        result = run(["helm", "lint", str(chart), *base, *extra])
        if result.returncode != 0:
            print(result.stdout + result.stderr, file=sys.stderr)
            ok = False
    for extra in REJECTED_VARIANTS.get(chart.name, []):
        result = run(["helm", "lint", str(chart), *extra])
        if result.returncode == 0:
            print(f"{chart.name} accepted values it must refuse: {extra}", file=sys.stderr)
            ok = False
    return ok


def main(argv: Sequence[str] | None = None) -> int:
    argparse.ArgumentParser(description="helm lint every chart under charts/").parse_args(argv)

    if shutil.which("helm") is None:
        message = "helm is not installed"
        if os.environ.get("CI"):
            print(f"{message}; CI must provide it", file=sys.stderr)
            return 1
        print(f"{message}; skipping chart lint", file=sys.stderr)
        return 0

    return 0 if all([lint_chart(chart) for chart in all_charts()]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
