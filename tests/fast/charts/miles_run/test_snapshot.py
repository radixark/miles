import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from megatron.training import arguments as megatron_arguments
from tests.fast.charts.utils import NAMESPACE, RUN_CHART_DIR, RUN_ID, RUN_RELEASE_NAME, requires_helm
from tests.fast.launch_scripts.sh_harness import REPO_ROOT, SANDBOX_PLACEHOLDER, assert_matches_snapshot

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.arguments import parse_args
from miles.utils.external_utils.command_utils.common import rsync_cmd
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.external_utils.model_args_utils import load_model_args
from miles.utils.workers.serving.utils import override_argv

SNAPSHOT_DIR = REPO_ROOT / "tests" / "snapshots" / "charts" / "miles-run"
FIXTURE_DIR = Path(__file__).resolve().parent
INFRA_VALUES = FIXTURE_DIR / "typical-infra.yaml"
SGLANG_CONFIG = FIXTURE_DIR / "typical-sglang.yaml"
HF_CHECKPOINT = FIXTURE_DIR / "typical-model"

PYTHON_PLACEHOLDER = "<PYTHON>"
FIXTURE_DIR_PLACEHOLDER = "<FIXTURE_DIR>"
RANDOM_SEED_PLACEHOLDER = "<RANDOM_SEED>"
RANDOM_SEED_FLAG = "--random-seed"

SCENARIOS = ("typical-values", "typical")

MODEL_TYPE = "qwen3-4B"
ROTARY_BASE = "1000000"

PREFILL_GPUS = 16
DECODE_GPUS = 16
GPUS_PER_NODE = 8
TRAINER_NODES = 4

ORCHESTRATOR_COMMAND = ["python", "scripts/run_qwen3_4b.py", "train", "--cluster-backend", "kubernetes"]
WORKER_ARGV = ["--cluster-backend", "kubernetes", "--rollout-num-gpus", str(PREFILL_GPUS + DECODE_GPUS)]
PREPARE_CMD = {"trainer": rsync_cmd("/cluster-storage/models/Qwen3-4B", "/scratch/Qwen3-4B")}
PARSER_ENV = {"CUDA_DEVICE_MAX_CONNECTIONS": "1"}

SCENARIO_ARGV = [
    *shlex.split(load_model_args(MODEL_TYPE, rotary_base=ROTARY_BASE)),
    # named rather than left to sglang's own probe, which reads the launcher's accelerator and
    # has none to read on the cpu lane that renders this snapshot
    "--sglang-device",
    "cuda",
    "--hf-checkpoint",
    str(HF_CHECKPOINT),
    "--load",
    "/cluster-storage/models/Qwen3-4B_torch_dist",
    "--save",
    "/cluster-storage/myteam/miles_data/miles-runs/myrun/checkpoints",
    "--save-interval",
    "20",
    "--prompt-data",
    "/cluster-storage/datasets/dapo-math-17k/dapo-math-17k.jsonl",
    "--input-key",
    "prompt",
    "--label-key",
    "label",
    "--apply-chat-template",
    "--rollout-shuffle",
    "--rm-type",
    "math",
    "--num-rollout",
    "16",
    "--rollout-batch-size",
    "32",
    "--n-samples-per-prompt",
    "8",
    "--rollout-max-response-len",
    "8192",
    "--rollout-temperature",
    "1",
    "--global-batch-size",
    "256",
    "--balance-data",
    "--optimizer",
    "adam",
    "--lr",
    "1e-6",
    "--lr-decay-style",
    "constant",
    "--weight-decay",
    "0.1",
    "--adam-beta1",
    "0.9",
    "--adam-beta2",
    "0.98",
    "--advantage-estimator",
    "grpo",
    "--eps-clip",
    "0.2",
    "--eps-clip-high",
    "0.28",
    "--use-dynamic-batch-size",
    "--max-tokens-per-gpu",
    "9216",
    "--sglang-config",
    str(SGLANG_CONFIG),
    "--rollout-num-gpus",
    str(PREFILL_GPUS + DECODE_GPUS),
    "--sglang-chunked-prefill-size",
    "4096",
    "--sglang-mem-fraction-static",
    "0.7",
    "--tensor-model-parallel-size",
    "2",
    "--sequence-parallel",
    "--pipeline-model-parallel-size",
    "1",
    "--context-parallel-size",
    "4",
    "--cp-comm-type",
    "a2a",
    "--expert-model-parallel-size",
    "1",
    "--expert-tensor-parallel-size",
    "1",
    "--recompute-granularity",
    "full",
    "--recompute-method",
    "uniform",
    "--recompute-num-layers",
    "1",
    "--attention-dropout",
    "0.0",
    "--hidden-dropout",
    "0.0",
    "--accumulate-allreduce-grads-in-fp32",
    "--attention-softmax-in-fp32",
    "--attention-backend",
    "flash",
    "--actor-num-nodes",
    str(TRAINER_NODES),
    "--actor-num-gpus-per-node",
    str(GPUS_PER_NODE),
    "--num-gpus-per-node",
    str(GPUS_PER_NODE),
    "--colocate",
    "--use-session-server",
    "--cluster-backend",
    "kubernetes",
    "--run-uuid",
    "0123456789abcdef",
]


@pytest.fixture(autouse=True)
def parser_process_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PARSER_ENV.items():
        monkeypatch.setenv(name, value)
    # megatron validates the arguments against the local accelerator, which the cpu lane that
    # runs this snapshot has none of; the rendered values do not depend on the answer
    monkeypatch.setattr(megatron_arguments, "get_device_arch_version", lambda: _DEVICE_ARCH_VERSION)


_DEVICE_ARCH_VERSION = 9


def _dump_values(values: dict[str, Any]) -> str:
    # unwrapped: yaml folds long lines against the real interpreter path, which the snapshot
    # only replaces afterwards, so a machine whose path is a different length folds elsewhere
    return yaml.safe_dump(values, default_flow_style=False, sort_keys=True, width=_NO_WRAP)


_NO_WRAP = 1 << 30


def synthetic_specs() -> list[Any]:
    with override_argv(SCENARIO_ARGV):
        return compute_specs(parse_args())


def synthetic_run_values() -> dict[str, Any]:
    return build_values(
        synthetic_specs(),
        LaunchPlan(
            run_id=RUN_ID,
            release=RUN_RELEASE_NAME,
            namespace="rl",
            state_file=f"/cluster-storage/myteam/miles_data/miles-runs/{RUN_ID}/state/orchestrator-260101-000000-000001.state",
            orchestrator_command=ORCHESTRATOR_COMMAND,
            worker_argv=WORKER_ARGV,
            env={"PYTHONUNBUFFERED": "1", **PARSER_ENV},
            colocate=True,
            prepare_cmd=PREPARE_CMD,
        ),
    ).as_values()


def render_from(values_file: Path) -> str:
    result = subprocess.run(
        [
            "helm",
            "template",
            RUN_RELEASE_NAME,
            str(RUN_CHART_DIR),
            "-n",
            NAMESPACE,
            "-f",
            str(INFRA_VALUES),
            "-f",
            str(values_file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def freeze(text: str, sandbox: Path) -> str:
    # only this test's own fixtures really move from machine to machine. the chart mounts the repo at
    # a fixed container path, so masking the whole checkout instead would mask that constant away on
    # any machine that happens to be checked out there, and record a snapshot only it can reproduce
    masked = text.replace(str(sandbox), SANDBOX_PLACEHOLDER).replace(str(FIXTURE_DIR), FIXTURE_DIR_PLACEHOLDER)
    return mask_random_seeds(masked.replace(sys.executable, PYTHON_PLACEHOLDER))


def mask_random_seeds(text: str) -> str:
    lines = text.split("\n")
    masked = [
        (
            re.sub(r"\d+", RANDOM_SEED_PLACEHOLDER, line)
            if index and _yaml_scalar(lines[index - 1]) == RANDOM_SEED_FLAG
            else line
        )
        for index, line in enumerate(lines)
    ]
    return "\n".join(masked)


def _yaml_scalar(line: str) -> str:
    return line.strip().removeprefix("- ").strip("'\"")


@requires_helm
class TestGeneratedValuesSnapshot:
    def test_the_launcher_turns_the_specs_into_exactly_the_recorded_values(self, tmp_path):
        """The spec to values transform decides a run's whole shape, so it is pinned end to end."""
        values = _dump_values(synthetic_run_values())

        assert_matches_snapshot(
            SNAPSHOT_DIR / "typical-values.yaml", freeze(values, sandbox=tmp_path), "miles-run generated values"
        )

    def test_those_values_render_exactly_the_recorded_manifests(self, tmp_path):
        """Rendering the file the launcher really writes is what pins the two halves to each other."""
        values_file = tmp_path / "run-values.yaml"
        values_file.write_text(_dump_values(synthetic_run_values()))

        assert_matches_snapshot(
            SNAPSHOT_DIR / "typical.yaml", freeze(render_from(values_file), sandbox=tmp_path), "miles-run manifests"
        )


class TestRandomSeedMasking:
    def test_a_seed_the_engine_drew_becomes_a_placeholder_in_the_generated_values(self):
        """sglang draws a fresh seed per render, so the values a run generates cannot record the number."""
        values = "    - --tp-size\n    - '8'\n    - --random-seed\n    - '379064976'\n    - --enable-metrics\n"

        assert mask_random_seeds(values) == (
            "    - --tp-size\n    - '8'\n    - --random-seed\n    - '<RANDOM_SEED>'\n    - --enable-metrics\n"
        )

    def test_a_seed_the_engine_drew_becomes_a_placeholder_in_the_rendered_manifests(self):
        """The manifests quote their argv differently from the values, and must be masked all the same."""
        manifests = '              - "--random-seed"\n              - "723999131"\n'

        assert mask_random_seeds(manifests) == ('              - "--random-seed"\n              - "<RANDOM_SEED>"\n')

    def test_a_number_that_no_seed_flag_introduces_is_left_alone(self):
        """Masking every number would hide the real argv, so only the seed's own value is replaced."""
        argv = "    - --tp-size\n    - '8'\n"

        assert mask_random_seeds(argv) == argv


class TestSnapshotFiles:
    def test_the_recorded_files_are_exactly_the_declared_ones(self):
        """A renamed or deleted scenario must not leave an orphan golden, nor a new one go unrecorded."""
        recorded = {path.stem for path in SNAPSHOT_DIR.glob("*.yaml")}

        assert recorded == set(SCENARIOS)
