from __future__ import annotations

import shlex
import sys

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.backends.sglang_utils.router_args_utils import parse_router_args_argv
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig
from miles.ray.specs.inference import (
    _compute_spec_router,
    compute_router_pool_id,
    spec_session_server,
    specs_inference_engine,
)
from miles.rollout.session.config import SessionServerConfig
from miles.router.config import MilesRouterConfig
from miles.utils.workers.argv_utils import parse_config_argv
from miles.utils.workers.worker_spec import HostAndPort, LaunchCommandContext


def _make_model_cfg(*worker_types: str) -> ModelConfig:
    groups = [
        ServerGroupConfig(
            worker_type=worker_type,
            num_gpus=4,
            num_gpus_per_engine=4,
            gpu_offset=group_index * 4,
            needs_offload=False,
        )
        for group_index, worker_type in enumerate(worker_types)
    ]
    return ModelConfig(name="default", model_path=None, server_groups=groups, update_weights=True)


def _make_router_ctx(*, port: int = 20000, prometheus_port: int = 4001) -> LaunchCommandContext:
    return LaunchCommandContext(
        cell_index=0,
        worker_in_cell_index=0,
        self_addrs=dict(
            primary=HostAndPort(host="127.0.0.1", port=port),
            prometheus=HostAndPort(host="127.0.0.1", port=prometheus_port),
        ),
        spec_addrs={},
        gpu_ids=[],
    )


class TestComputeSpecRouterLaunchCommand:
    def test_pd_disagg_with_miles_router_asserts(self):
        """Rendering a miles-router launch command for a PD-disaggregated model must fail fast."""
        args = make_args(use_miles_router=True, sglang_router_ip=None, sglang_router_port=None)
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("prefill", "decode"))
        with pytest.raises(AssertionError, match="miles router does not support PD"):
            spec.launch_command(_make_router_ctx())

    def test_sgl_router_launches_the_native_cli(self):
        """The sgl router runs as the upstream CLI with the addresses from the launch context."""
        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=None)
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))
        argv = shlex.split(spec.launch_command(_make_router_ctx()))
        assert argv[0] == sys.executable
        assert argv[1:3] == ["-m", "sglang_router.launch_router"]
        assert argv[argv.index("--port") + 1] == "20000"
        assert argv[argv.index("--prometheus-port") + 1] == "4001"

    def test_sgl_router_launch_preserves_prefixed_raw_inputs(self):
        """Raw --router-* aliases and collections survive the full launch-command path."""
        args = make_args(
            use_miles_router=False,
            sglang_router_ip=None,
            sglang_router_port=None,
            router_tls_cert_path="/certs/server.pem",
            router_prefill=[["http://prefill.invalid", "9000"]],
            router_selector=["app=sglang", "role=prefill"],
        )
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("prefill", "decode"))
        argv = shlex.split(spec.launch_command(_make_router_ctx()))
        parsed = parse_router_args_argv(argv[3:])

        assert parsed.server_cert_path == "/certs/server.pem"
        assert parsed.prefill_urls == [("http://prefill.invalid", 9000)]
        assert parsed.selector == {"app": "sglang", "role": "prefill"}
        assert parsed.pd_disaggregation is True

    def test_miles_router_launches_with_a_parseable_config(self):
        """The miles router command's config payload parses back losslessly."""
        args = make_args(
            use_miles_router=True,
            sglang_router_ip=None,
            sglang_router_port=None,
            miles_router_max_connections=100,
            miles_router_timeout=None,
            miles_router_health_check_failure_threshold=3,
            rollout_health_check_interval=10.0,
        )
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))
        argv = shlex.split(spec.launch_command(_make_router_ctx()))
        assert argv[:3] == [sys.executable, "-m", "miles.router.router"]
        config = parse_config_argv(MilesRouterConfig, argv[3:])
        assert config.host == "127.0.0.1"
        assert config.port == 20000
        assert config.max_connections == 100


class TestComputeSpecSessionServer:
    def test_launch_command_wires_the_router_backend_and_roundtrips(self):
        """The session server command targets the router addr from spec_addrs and its config parses back losslessly."""
        args = make_args(
            use_session_server="v1",
            hf_checkpoint="/fake/model",
            num_session_servers=2,
            sglang_router_ip=None,
            sglang_router_port=None,
            miles_router_timeout=None,
            chat_template_path=None,
            tito_model="default",
            apply_chat_template_kwargs=None,
            use_rollout_indexer_replay=False,
            sglang_speculative_algorithm=None,
            num_layers=None,
            moe_router_topk=None,
            save_debug_trajectory_data=None,
            lora_rank=0,
            lora_adapter_path=None,
        )
        spec = spec_session_server(args)
        assert spec.scheduling.num_cells == 2

        ctx = LaunchCommandContext(
            cell_index=1,
            worker_in_cell_index=0,
            self_addrs=dict(primary=HostAndPort(host="127.0.0.1", port=5006)),
            spec_addrs={compute_router_pool_id(0): [dict(primary=HostAndPort(host="127.0.0.1", port=3000))]},
            gpu_ids=[],
        )
        argv = shlex.split(spec.launch_command(ctx))

        assert argv[:3] == [sys.executable, "-m", "miles.rollout.session.server"]
        config = parse_config_argv(SessionServerConfig, argv[3:])
        assert config.backend_url == "http://127.0.0.1:3000"
        assert config.host == "127.0.0.1"
        assert config.port == 5006
        assert config.instance_id == f"{args.run_uuid}-1"

    def test_disabled_schedules_zero_cells(self):
        """Disabling the session server removes its cells instead of launching idle servers."""
        args = make_args(use_session_server=False)
        assert spec_session_server(args).scheduling.num_cells == 0


class TestSpecsInferenceEngine:
    def test_pg_slot_offsets_accumulate_and_placeholder_groups_keep_their_slots(self, tmp_path):
        """Group offsets follow the config order and a skipped placeholder group still occupies its gpu span."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2},
                    {"worker_type": "placeholder", "num_gpus": 4, "num_gpus_per_engine": 4},
                    {"worker_type": "decode", "num_gpus": 8, "num_gpus_per_engine": 4},
                ]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=16)

        specs = specs_inference_engine(args)

        assert [spec.name for spec in specs] == ["inference-engine-0-0", "inference-engine-0-2"]
        assert [spec.scheduling.pg_slot_offset for spec in specs] == [0, 8]
        assert [spec.scheduling.num_gpu_slots_per_worker for spec in specs] == [2, 4]
        assert all(spec.scheduling.pg_name == "rollout" for spec in specs)

    def test_debug_train_only_produces_no_engine_spec(self, tmp_path):
        """In --debug-train-only the rollout placement group is the trainer's own gpus, so no engine may be specced."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
            )
        )
        args = make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=8,
            colocate=True,
            debug_train_only=True,
        )

        assert specs_inference_engine(args) == []


class TestInferenceSpecPinToHead:
    @pytest.mark.parametrize("pinned", [False, True])
    def test_router_and_session_specs_follow_the_rollout_manager_flag(self, pinned: bool):
        """Both driver-adjacent specs are pinned exactly when the rollout manager is."""
        from miles.ray.specs.inference import _compute_spec_router, spec_session_server

        args = make_args(
            pin_rollout_manager_to_head=pinned,
            use_miles_router=False,
            use_session_server=True,
            hf_checkpoint="/fake/model",
            num_session_servers=1,
            chat_template_path=None,
            tito_model="default",
            apply_chat_template_kwargs=None,
            use_rollout_indexer_replay=False,
            sglang_speculative_algorithm=None,
            num_layers=None,
            moe_router_topk=None,
            save_debug_trajectory_data=None,
            lora_rank=0,
            lora_adapter_path=None,
            miles_router_timeout=None,
        )

        router = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))
        session = spec_session_server(args)

        assert router.scheduling.pin_to_head is pinned
        assert session.scheduling.pin_to_head is pinned


class TestInferenceEnginePortSchema:
    def test_the_master_port_reserves_a_block_for_every_dp_rank(self, tmp_path):
        """sglang needs a contiguous block behind dist_init, so the reservation must grow with dp size."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4, sglang_dp_size=3)

        ports = {info.name: info for info in specs_inference_engine(args)[0].port_infos}

        assert ports["dist_init"].mode == "master"
        assert ports["dist_init"].allow_dynamic is True
        assert ports["dist_init"].num_consecutive == 33
        assert [info.mode for name, info in ports.items() if name != "dist_init"] == ["per_worker"] * (len(ports) - 1)

    def test_only_prefill_engines_get_a_disaggregation_bootstrap_port(self, tmp_path):
        """The bootstrap port belongs to the prefill side alone."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "prefill", "num_gpus": 2, "num_gpus_per_engine": 2},
                    {"worker_type": "decode", "num_gpus": 2, "num_gpus_per_engine": 2},
                ]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4)

        prefill, decode = specs_inference_engine(args)

        assert [info.name for info in prefill.port_infos] == [
            "primary",
            "dist_init",
            "nccl",
            "disaggregation_bootstrap",
            "engine_info_bootstrap",
        ]
        assert [info.name for info in decode.port_infos] == [
            "primary",
            "dist_init",
            "nccl",
            "engine_info_bootstrap",
        ]
