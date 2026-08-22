import json
import os
import shlex
import textwrap
from dataclasses import dataclass
from typing import Any

import yaml
from tests.ci.ci_register import register_cuda_ci

from miles.utils.external_utils import command_utils
from miles.utils.external_utils.command_utils.common import chart_dir, repo_base_dir
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.helm_values_types import (
    InfraValues,
    VolumeEntry,
    VolumeMount,
)
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import InfraInfo
from miles.utils.workers.types import ClusterBackend

register_cuda_ci(est_time=700, suite="stage-c-4-gpu-h200", labels=["short"])

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_TRAIN_GPUS = 2
NUM_ENGINES = 2
GPUS_PER_ENGINE = 1

RAY_ENGINE_HOST = "127.0.0.1"
RAY_ENGINE_PORTS = [32001, 32002]

ENGINE_OBJECT_NAME = "external-sglang"
ENGINE_PORT = 30000
ENGINE_CONTAINER_NAME = "sglang"
ENGINE_HEALTH_PATH = "/health_generate"


# ===== Entry points =====


def compute_train_and_engine_devices(visible_devices: str | None) -> tuple[list[str], list[str]]:
    devices = (visible_devices or "").split(",") if visible_devices else []
    if not devices:
        devices = [str(i) for i in range(NUM_TRAIN_GPUS + NUM_ENGINES)]
    assert len(devices) == NUM_TRAIN_GPUS + NUM_ENGINES, (
        f"this test trains on {NUM_TRAIN_GPUS} gpus and pins one engine to each of {NUM_ENGINES} more, "
        f"but the runner offered {devices}"
    )
    return devices[:NUM_TRAIN_GPUS], devices[NUM_TRAIN_GPUS:]


def prepare():
    U = command_utils.default_config().create_backend()
    U.exec_command_cpu("mkdir -p /root/models /root/datasets")
    U.exec_command_cpu(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_TRAIN_GPUS)


def execute():
    U = command_utils.default_config().create_backend()
    engines = _external_engines(U.config)

    U.execute_train(
        train_args=_train_args(engines.addrs, object_store_args=_object_store_args(U.config)),
        num_gpus_per_node=NUM_TRAIN_GPUS,
        megatron_model_type=MODEL_TYPE,
        prepare_cmd=engines.prepare_cmd,
        extra_manifests=engines.extra_manifests,
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
    )


@dataclass(frozen=True)
class _ExternalEngines:
    addrs: list[str]
    prepare_cmd: dict[str, str]
    extra_manifests: list[str]


def _external_engines(config: command_utils.ExecuteTrainConfig) -> _ExternalEngines:
    match config.cluster_backend:
        case ClusterBackend.RAY:
            return _engines_started_beside_the_trainer()
        case ClusterBackend.KUBERNETES:
            return _engines_installed_beside_the_run(config.helm_values)


def _object_store_args(config: command_utils.ExecuteTrainConfig) -> str:
    match config.cluster_backend:
        case ClusterBackend.RAY:
            return ""
        case ClusterBackend.KUBERNETES:
            return command_utils.get_mooncake_object_store_args()


# ===== Training arguments =====


def _train_args(engine_addrs: list[str], *, object_store_args: str) -> str:
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/{MODEL_NAME}_torch_dist/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 16 "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 32 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 9216 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    external_args = (
        "--rollout-external-engine-addrs "
        + " ".join(engine_addrs)
        + f" --rollout-num-gpus {NUM_ENGINES * GPUS_PER_ENGINE} "
        + f"--rollout-num-gpus-per-engine {GPUS_PER_ENGINE} "
    )

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_TRAIN_GPUS} "
        "--megatron-to-hf-mode bridge "
    )

    return (
        f"{ckpt_args} "
        f"{object_store_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{command_utils.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{external_args} "
        f"{ci_args} "
        f"{misc_args} "
    )


# ===== Engines the run talks to but does not launch =====


def _engines_started_beside_the_trainer() -> _ExternalEngines:
    _train_devices, engine_devices = compute_train_and_engine_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    return _ExternalEngines(
        addrs=[f"{RAY_ENGINE_HOST}:{port}" for port in RAY_ENGINE_PORTS],
        prepare_cmd={"trainer": _ray_engines_launch_cmd(engine_devices)},
        extra_manifests=[],
    )


def _engines_installed_beside_the_run(helm_values: tuple[str, ...]) -> _ExternalEngines:
    return _ExternalEngines(
        addrs=[f"{ENGINE_OBJECT_NAME}-{index}.{ENGINE_OBJECT_NAME}:{ENGINE_PORT}" for index in range(NUM_ENGINES)],
        prepare_cmd={},
        extra_manifests=[_engine_manifests(_infra_values(helm_values))],
    )


def _ray_engines_launch_cmd(engine_devices: list[str]) -> str:
    # a prepare command runs inside a ray task, and the engines have to outlive it: nohup only
    # answers the hangup, so without a session of their own they are reaped with the task's group
    launches = " ".join(
        f"(CUDA_VISIBLE_DEVICES={device} setsid {shlex.join(_engine_argv(port))} "
        f"> /tmp/miles_external_engine_{port}.log 2>&1 &);"
        for device, port in zip(engine_devices, RAY_ENGINE_PORTS, strict=True)
    )
    probes = " && ".join(
        f"curl -sf http://{RAY_ENGINE_HOST}:{port}/server_info >/dev/null" for port in RAY_ENGINE_PORTS
    )
    wait = (
        f"ok=0; for _ in $(seq 1 120); do {probes} && ok=1 && break; sleep 5; done; "
        f"[ \"$ok\" -eq 1 ] || {{ echo 'external sglang engines failed to start' >&2; "
        f"tail -n 100 /tmp/miles_external_engine_*.log >&2; exit 1; }}"
    )
    return f"{launches} {wait}"


def _engine_argv(port: int) -> list[str]:
    return [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        f"/root/models/{MODEL_NAME}",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tp",
        str(GPUS_PER_ENGINE),
        "--mem-fraction-static",
        "0.7",
        "--trust-remote-code",
    ]


# ===== Kubernetes manifests for those engines =====


def _engine_manifests(infra: InfraValues) -> str:
    scheduling = infra.scheduling

    pod_fields: dict[str, Any] = {
        "imagePullSecrets": [dict(name=secret) for secret in infra.image.pull_secrets or []],
        "nodeSelector": scheduling.node_selector if scheduling is not None else None,
        "tolerations": scheduling.tolerations if scheduling is not None else None,
        "affinity": scheduling.affinity if scheduling is not None else None,
        "volumes": [_volume(volume) for volume in infra.volumes],
    }
    container_fields: dict[str, Any] = {
        "imagePullPolicy": infra.image.pull_policy,
        "env": [dict(name=name, value=value) for name, value in sorted((infra.env or {}).items())],
        "volumeMounts": [_volume_mount(volume, mount) for volume in infra.volumes for mount in volume.mounts],
    }

    return f"""\
apiVersion: v1
kind: Service
metadata:
  name: {ENGINE_OBJECT_NAME}
spec:
  clusterIP: "None"
  selector:
    app: {ENGINE_OBJECT_NAME}
  ports:
    - name: http
      port: {ENGINE_PORT}
      targetPort: {ENGINE_PORT}
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {ENGINE_OBJECT_NAME}
spec:
  replicas: {NUM_ENGINES}
  serviceName: {ENGINE_OBJECT_NAME}
  podManagementPolicy: Parallel
  selector:
    matchLabels:
      app: {ENGINE_OBJECT_NAME}
  template:
    metadata:
      labels:
        app: {ENGINE_OBJECT_NAME}
    spec:
{_yaml_block(pod_fields, indent=6)}\
      containers:
        - name: {ENGINE_CONTAINER_NAME}
          image: {infra.image.repository}:{infra.image.tag}
          command: {json.dumps(_engine_argv(ENGINE_PORT))}
          ports:
            - name: http
              containerPort: {ENGINE_PORT}
          resources:
            limits:
              nvidia.com/gpu: {GPUS_PER_ENGINE}
          readinessProbe:
            httpGet:
              path: {ENGINE_HEALTH_PATH}
              port: {ENGINE_PORT}
            periodSeconds: 10
            failureThreshold: 60
{_yaml_block(container_fields, indent=10)}"""


def _yaml_block(fields: dict[str, Any], *, indent: int) -> str:
    present = {key: value for key, value in fields.items() if value}
    if not present:
        return ""

    return textwrap.indent(yaml.safe_dump(present, sort_keys=False), " " * indent)


def _volume(volume: VolumeEntry) -> dict[str, Any]:
    if (host_path := volume.host_path) is not None:
        return dict(name=volume.name, hostPath=dict(path=host_path.path, type=host_path.type or "Directory"))
    if (claim := volume.persistent_volume_claim) is not None:
        return dict(name=volume.name, persistentVolumeClaim=dict(claimName=claim.claim_name))

    assert volume.empty_dir is not None
    return dict(name=volume.name, emptyDir=volume.empty_dir.as_values())


def _volume_mount(volume: VolumeEntry, mount: VolumeMount) -> dict[str, Any]:
    fields = dict(name=volume.name, mountPath=mount.mount_path)
    if mount.sub_path is not None:
        fields["subPath"] = mount.sub_path
    if mount.read_only:
        fields["readOnly"] = mount.read_only
    return fields


def _infra_values(helm_values: tuple[str, ...]) -> InfraValues:
    return InfraInfo.load(chart_dir(repo_base_dir=repo_base_dir), list(helm_values))


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
