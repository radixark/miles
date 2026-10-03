from pathlib import Path

import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config

from miles.utils.env_report.launcher_report import LAUNCHER_REPORT_ENV_VAR
from miles.utils.external_utils.command_utils.helm_backend.launcher.entrypoint import _compute_orchestrator_command
from miles.utils.orchestration_utils import PayloadOrchestratorStartupInfo
from miles.utils.workers.connection_config import StaticConnConfig, StaticPoolConnInfo
from miles.utils.workers.serving.utils import parse_orchestrator_argv
from miles.utils.workers.worker_spec import DEFAULT_RPC_PORT_INFO


class TestOrchestratorPayload:
    def test_a_launcher_payload_starts_without_scaling_or_rollout_only_fields_and_reads_the_pod_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The real launcher payload must start the orchestrator with its own config and the pod's report."""
        args = parse_fsdp_test_config(
            "--cluster-backend",
            "kubernetes",
            "--rollout-num-gpus",
            "4",
            "--save-debug-rollout-data",
            str(tmp_path / "rollout"),
        )
        connections = StaticConnConfig(
            static_conn_infos={
                "trainer-controller-actor": StaticPoolConnInfo(
                    name="trainer-controller-actor",
                    port_infos=[DEFAULT_RPC_PORT_INFO],
                    worker_class="miles.ray.train.group.TrainerController",
                    num_cells=1,
                    num_workers_per_cell=1,
                    pods_per_cell=1,
                )
            }
        )
        command = _compute_orchestrator_command("train.py", args=args, static_connections=connections)
        payload = parse_orchestrator_argv(command)
        pod_report = str(tmp_path / "pod-report.json")
        monkeypatch.setenv(LAUNCHER_REPORT_ENV_VAR, pod_report)

        startup = PayloadOrchestratorStartupInfo.create(payload)

        assert args.rollout_num_gpus == 4
        assert args.save_debug_rollout_data == str(tmp_path / "rollout")
        assert args.env_report == ""
        assert {"rollout_num_gpus", "actor_num_gpus_per_node", "sglang_scaling"}.isdisjoint(payload.args)
        assert "save_debug_rollout_data" not in payload.args
        assert "env_report" not in payload.args
        assert startup.args.env_report == pod_report
        assert startup.args.cluster_backend == "kubernetes"
        assert startup.args.num_rollout == args.num_rollout
        assert startup.static_connections == connections
