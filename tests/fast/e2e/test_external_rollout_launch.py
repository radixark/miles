import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.fast.charts.utils import RUN_CHART_DIR, documents_of, requires_helm
from tests.fast.e2e.external_rollout_script import load_external_rollout_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

from miles.utils.arguments import parse_args
from miles.utils.external_utils.command_utils.base_backend import ExecuteTrainConfig, ExecuteTrainRequest
from miles.utils.external_utils.command_utils.helm_backend import naming
from miles.utils.external_utils.command_utils.helm_backend.launcher import command_wrapper, entrypoint
from miles.utils.external_utils.command_utils.helm_backend.launcher.command_wrapper import Helm
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import LaunchPlan
from miles.utils.external_utils.model_args_utils import shell_safe_model_args
from miles.utils.workers.serving.utils import override_argv
from miles.utils.workers.types import ClusterBackend

script = load_external_rollout_script()

NAMESPACE = "rl"
RUN_ID = "260101-000000-000"
LAUNCH_TOKEN = "260101-000000-000001"
MODEL_CONFIG_JSON = """\
{
  "architectures": ["Qwen2ForCausalLM"],
  "model_type": "qwen2",
  "hidden_size": 896,
  "intermediate_size": 4864,
  "num_hidden_layers": 24,
  "num_attention_heads": 14,
  "num_key_value_heads": 2,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-06,
  "rope_theta": 1000000.0,
  "tie_word_embeddings": true,
  "vocab_size": 151936,
  "max_position_embeddings": 32768,
  "torch_dtype": "bfloat16"
}
"""

EXTERNAL_ROLLOUT_FLAG = "--rollout-external-engine-addrs"
CONTROLLER_POOL = "inference-controller"
TRAINER_POOL = "trainer-engine-actor"
ENGINE_POOL_PREFIX = "inference-engine"
STATIC_ENGINE_PROVIDER = "miles.ray.rollout.external_engine_provider.static_inference_engine_provider"
POOL_SECTIONS = ("staticWorkers", "inferenceEngines", "trainerEngines")


@dataclass(frozen=True)
class _Launch:
    plan: LaunchPlan
    values: dict[str, Any]
    rendered: str


@pytest.fixture(autouse=True)
def host_without_a_partition_or_a_wandb_key(monkeypatch):
    for name in ("CUDA_VISIBLE_DEVICES", "WANDB_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")


def infra_file(sandbox: Path) -> Path:
    path = sandbox / "infra.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "infra": {
                    "image": {"repository": "myregistry.example/miles", "tag": "v1"},
                    "volumes": [
                        {
                            "name": "cluster-storage",
                            "hostPath": {"path": f"{sandbox}/cluster-storage"},
                            "mounts": [{"mountPath": f"{sandbox}/cluster-storage"}],
                        }
                    ],
                    "paths": {"runsRoot": f"{sandbox}/cluster-storage/miles_data"},
                }
            }
        )
    )
    return path


def values_file(sandbox: Path) -> Path:
    return (
        sandbox / "cluster-storage" / "miles_data" / "miles-runs" / RUN_ID / "values" / f"values-{LAUNCH_TOKEN}.yaml"
    )


def model_dir(sandbox: Path) -> Path:
    path = sandbox / script.MODEL_NAME
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(MODEL_CONFIG_JSON)
    return path


def ref_load_dir(sandbox: Path) -> Path:
    path = sandbox / f"{script.MODEL_NAME}_torch_dist"
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_args_of(config: ExecuteTrainConfig, addrs: list[str], sandbox: Path) -> str:
    written = f"{shell_safe_model_args(script.MODEL_TYPE)} " + script._train_args(
        addrs, object_store_args=script._object_store_args(config)
    )
    written = written.replace(f"/root/models/{script.MODEL_NAME}/", f"{model_dir(sandbox)}/")
    return written.replace(f"/root/{script.MODEL_NAME}_torch_dist/", f"{ref_load_dir(sandbox)}/")


def launch(monkeypatch, sandbox: Path) -> _Launch:
    config = ExecuteTrainConfig(
        cluster_backend=ClusterBackend.KUBERNETES,
        namespace=NAMESPACE,
        run_id=RUN_ID,
        helm_values=(str(infra_file(sandbox)),),
    )
    engines = script._external_engines(config)
    request = ExecuteTrainRequest(
        train_args=train_args_of(config, engines.addrs, sandbox),
        num_gpus_per_node=script.NUM_TRAIN_GPUS,
        megatron_model_type=script.MODEL_TYPE,
        train_script="tests/e2e/short/test_qwen2.5_0.5B_external_rollout.py",
        train_backend_fsdp=False,
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
        megatron_path="/root/Megatron-LM",
        before_ray_job_submit=None,
        prepare_cmd=engines.prepare_cmd,
        extra_manifests=engines.extra_manifests,
    )

    planned: list[LaunchPlan] = []
    build_values = entrypoint.build_values

    def record_plan(specs, plan):
        planned.append(plan)
        return build_values(specs, plan)

    monkeypatch.setattr(entrypoint, "build_values", record_plan)

    def helm_answered_nothing(command, **kwargs) -> subprocess.CompletedProcess:
        # the launcher renders the upgrade it is about to install and reads the manifest out of the
        # json, so a helm that answers with nothing at all has to answer with an empty manifest
        rendered = '{"manifest": ""}' if "json" in command else ""
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=rendered, stderr="")

    monkeypatch.setattr(command_wrapper, "run_process", helm_answered_nothing)
    monkeypatch.setattr(Helm, "get_manifest", staticmethod(lambda release, namespace: None))
    monkeypatch.setattr(entrypoint, "repo_base_dir", str(REPO_ROOT))
    monkeypatch.setattr(naming, "_new_launch_token", lambda: LAUNCH_TOKEN)
    monkeypatch.setattr(entrypoint, "_follow_until_finished", lambda **kwargs: None)

    entrypoint.execute_train(request=request, config=config)

    return _Launch(
        plan=planned[-1],
        values=yaml.safe_load(values_file(sandbox).read_text()),
        rendered=render(infra_file(sandbox), values_file(sandbox)),
    )


def render(*values_files: Path) -> str:
    arguments = [argument for path in values_files for argument in ("-f", str(path))]
    result = subprocess.run(
        ["helm", "template", "myrun", str(RUN_CHART_DIR), "-n", NAMESPACE, *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def pool_entries(values: dict[str, Any]) -> list[dict[str, Any]]:
    return [entry for section in POOL_SECTIONS for entry in values["run"].get(section, [])]


def named_pool_entry(values: dict[str, Any], name: str) -> dict[str, Any]:
    matched = [entry for entry in pool_entries(values) if entry["name"] == name]
    assert (
        len(matched) == 1
    ), f"expected one pool named {name}, got {[entry['name'] for entry in pool_entries(values)]}"
    return matched[0]


class TestTheScriptOwnArgvSelectsTheExternalPath:
    def test_the_addrs_are_what_the_run_builds_its_engine_provider_from(self, tmp_path):
        """Reaching the pod is not enough: the addrs matter only if they select the static provider."""
        config = ExecuteTrainConfig(
            cluster_backend=ClusterBackend.KUBERNETES, helm_values=(str(infra_file(tmp_path)),)
        )

        argv = shlex.split(train_args_of(config, script._external_engines(config).addrs, tmp_path))

        with override_argv(argv):
            args = parse_args()

        assert args.rollout_external
        assert args.custom_inference_engine_provider_path == STATIC_ENGINE_PROVIDER


@requires_helm
class TestTheScriptOwnArgvSurvivesTheWholeLauncher:
    def test_the_run_is_given_the_object_store_the_backend_can_serve(self, monkeypatch, tmp_path):
        """Kubernetes forces mooncake, and a launch that named none aborts here, long before helm installs."""
        plan = launch(monkeypatch, tmp_path).plan

        assert plan.mooncake_plan is not None
        assert plan.mooncake_plan.port > 0

    def test_the_master_the_pods_dial_is_the_one_this_release_installs(self, monkeypatch, tmp_path):
        """The script names a loopback address, which is nothing at all from another pod."""
        launched = launch(monkeypatch, tmp_path)

        assert launched.values["run"]["mooncake"]["enabled"] is True
        assert "127.0.0.1" not in " ".join(launched.plan.worker_argv)

    def test_the_run_declares_no_inference_engine_pool_of_its_own(self, monkeypatch, tmp_path):
        """External rollout means miles provisions none, and one rendered anyway would take gpus and idle."""
        values = launch(monkeypatch, tmp_path).values

        assert [entry["name"] for entry in pool_entries(values) if entry["name"].startswith(ENGINE_POOL_PREFIX)] == []

    def test_the_run_still_deploys_everything_an_engine_is_not(self, monkeypatch, tmp_path):
        """A launch that rendered nothing at all would satisfy every other assertion about engines."""
        names = {entry["name"] for entry in pool_entries(launch(monkeypatch, tmp_path).values)}

        assert {CONTROLLER_POOL, TRAINER_POOL} <= names

    def test_the_addrs_the_script_computed_reach_the_controller_that_dials_them(self, monkeypatch, tmp_path):
        """These addrs are written beside the manifest that serves them, and only argv carries them across."""
        launched = launch(monkeypatch, tmp_path)
        command = named_pool_entry(launched.values, CONTROLLER_POOL)["command"]
        addrs = script._external_engines(
            ExecuteTrainConfig(cluster_backend=ClusterBackend.KUBERNETES, helm_values=(str(infra_file(tmp_path)),))
        ).addrs

        start = command.index(EXTERNAL_ROLLOUT_FLAG) + 1
        assert command[start : start + len(addrs)] == addrs

    def test_the_engines_the_script_wrote_are_installed_with_the_run(self, monkeypatch, tmp_path):
        """The whole point of the kubernetes half: the engines ride along in the release that trains against them."""
        objects = documents_of(launch(monkeypatch, tmp_path).rendered)
        engine_objects = {
            obj["kind"] for obj in objects if obj["metadata"]["name"].startswith(script.ENGINE_OBJECT_NAME)
        }

        assert engine_objects == {"Service", "StatefulSet"}

    def test_no_rendered_pod_of_the_run_launches_an_engine_itself(self, monkeypatch, tmp_path):
        """Only the manifest above may start an engine; a second one would fight it for the same gpus."""
        objects = documents_of(launch(monkeypatch, tmp_path).rendered)
        own = [obj for obj in objects if not obj["metadata"]["name"].startswith(script.ENGINE_OBJECT_NAME)]

        assert "sglang.launch_server" not in yaml.safe_dump(own)
