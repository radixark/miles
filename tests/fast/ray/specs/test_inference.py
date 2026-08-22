from __future__ import annotations

import asyncio
import shlex
import sys
from argparse import Namespace

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.backends.sglang_utils.router_args_utils import parse_router_args_argv
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig
from miles.ray.rollout import external_engine_provider as external_engine_provider_module
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.specs import inference as inference_specs
from miles.ray.specs.inference import (
    INFERENCE_CONTROLLER_POOL_ID,
    INFERENCE_CONTROLLER_WORKER_CLASS,
    _compute_router_primary_port_info,
    _compute_session_server_primary_port_info,
    _compute_spec_router,
    compute_engine_pool_ids,
    compute_inference_controller_provider,
    compute_inference_engine_env_vars,
    compute_router_pool_id,
    inference_controller_worker_name,
    spec_inference_controller,
    spec_session_server,
    specs_inference_engine,
    specs_inference_registration_reporter,
)
from miles.rollout.session.config import SessionServerConfig
from miles.router.config import MilesRouterConfig
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.builder import build_values
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.misc import SECTION_OF_CATEGORY, LaunchPlan
from miles.utils.function_registry import load_function
from miles.utils.workers.argv_utils import parse_config_argv
from miles.utils.workers.registration.hub import RegistrationHub
from miles.utils.workers.worker_provider.static import StaticWorkerProvider
from miles.utils.workers.worker_spec import (
    RPC_PORT_NAME,
    HostAndPort,
    LaunchCommandContext,
    WorkerCtorContext,
    WorkerMetaContext,
)


def _controller_layout() -> LaunchPlan:
    return LaunchPlan(
        run_id="260101-000000-000",
        state_file="/cluster-storage/miles_data/miles-runs/run/state/orchestrator-260101-000000-000001.state",
        release="miles-run-260101",
        namespace="rl",
        orchestrator_command=["python", "/repo/train.py"],
        worker_argv=["--rollout-num-gpus", "8"],
    )


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


class TestRouterPortPinning:
    def test_an_unpinned_router_may_move_off_its_preferred_port(self):
        """Nothing outside the job needs to name it, so a busy 8000 must not fail the launch."""
        port_info = _compute_router_primary_port_info(make_args(sglang_router_port=None), model_idx=0)

        assert (port_info.static_port, port_info.allow_dynamic) == (8000, True)

    def test_a_pinned_router_stays_on_the_port_it_was_given(self):
        """Launchers pin it so a firewall rule or a dial-back host can name the port in advance;
        drifting off it would leave those pointing at nothing."""
        port_info = _compute_router_primary_port_info(make_args(sglang_router_port=31000), model_idx=0)

        assert (port_info.static_port, port_info.allow_dynamic) == (31000, False)

    def test_each_models_router_is_pinned_a_port_apart(self):
        """Two models pinned to one port would race for the same socket."""
        ports = [
            _compute_router_primary_port_info(make_args(sglang_router_port=31000), model_idx=i).static_port
            for i in range(2)
        ]

        assert ports == [31000, 31001]


class TestComputeSpecRouterLaunchCommand:
    def test_pd_disagg_with_miles_router_asserts(self):
        """Rendering a miles-router launch command for a PD-disaggregated model must fail fast."""
        args = make_args(use_miles_router=True, sglang_router_ip=None, sglang_router_port=None)
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("prefill", "decode"))
        with pytest.raises(AssertionError, match="miles router does not support PD"):
            spec.launch_command(_make_router_ctx())

    def test_the_external_pd_flag_launches_a_pd_router_in_front_of_regular_groups(self):
        """External PD is discovered after the router is already up, so this flag is the only thing
        that can put the router in PD mode; a router built without it misroutes every request while
        discovery still reports a healthy prefill/decode fleet."""
        args = make_args(
            use_miles_router=False,
            sglang_router_ip=None,
            sglang_router_port=None,
            rollout_external=True,
            rollout_external_router_pd=True,
        )
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        argv = shlex.split(spec.launch_command(_make_router_ctx()))

        assert parse_router_args_argv(argv[3:]).pd_disaggregation is True

    def test_without_the_external_pd_flag_a_regular_model_keeps_a_regular_router(self):
        """Every internal run takes this path, and the new flag defaults to off."""
        args = make_args(use_miles_router=False, sglang_router_ip=None, sglang_router_port=None)
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        argv = shlex.split(spec.launch_command(_make_router_ctx()))

        assert parse_router_args_argv(argv[3:]).pd_disaggregation is False

    def test_the_external_pd_flag_is_rejected_by_the_miles_router(self):
        """The miles router cannot serve PD at all, so the external flag must hit the same guard."""
        args = make_args(
            use_miles_router=True,
            sglang_router_ip=None,
            sglang_router_port=None,
            rollout_external=True,
            rollout_external_router_pd=True,
        )
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

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


def _make_session_server_args(**overrides) -> Namespace:
    return make_args(
        use_session_server="v1",
        num_session_servers=2,
        miles_router_timeout=None,
        chat_template_path=None,
        tito_model="default",
        apply_chat_template_kwargs=None,
        lora_adapter_path=None,
        **overrides,
    )


class TestSessionServerAddressPinning:
    def test_an_unpinned_session_server_may_move_off_its_preferred_port(self):
        """Nothing outside the job names it, so a busy 8000 must not fail the launch nor be shifted per cell."""
        port_info = _compute_session_server_primary_port_info(make_args(session_server_port=None))

        assert (port_info.static_port, port_info.allow_dynamic, port_info.offset_by_cell) == (8000, True, False)

    def test_pinned_session_servers_take_consecutive_ports_from_the_configured_one(self):
        """A pinned port must stay put and be shifted per cell, otherwise every session server races for one socket."""
        port_info = _compute_session_server_primary_port_info(make_args(session_server_port=5100))

        assert (port_info.static_port, port_info.allow_dynamic, port_info.offset_by_cell) == (5100, False, True)

    def test_a_configured_session_server_ip_overrides_the_allocated_host(self):
        """Operators pin the advertised ip so clients can reach it; binding the ray node ip instead ignores them."""
        args = _make_session_server_args(session_server_ip="10.20.30.40")
        spec = spec_session_server(args)
        ctx = LaunchCommandContext(
            cell_index=0,
            worker_in_cell_index=0,
            self_addrs=dict(primary=HostAndPort(host="127.0.0.1", port=5006)),
            spec_addrs={compute_router_pool_id(0): [dict(primary=HostAndPort(host="127.0.0.1", port=3000))]},
            gpu_ids=[],
        )

        config = parse_config_argv(SessionServerConfig, shlex.split(spec.launch_command(ctx))[3:])

        assert config.host == "10.20.30.40"
        assert config.port == 5006


class TestInferenceEngineEnvVars:
    def test_a_process_level_override_wins_over_the_built_in_default(self, monkeypatch):
        """The launcher's environment is how operators retune sglang per cluster, so defaults must not overwrite it."""
        monkeypatch.setenv("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "true")
        monkeypatch.setenv("SGLANG_MEMORY_SAVER_CUDA_GRAPH", "false")

        envs = compute_inference_engine_env_vars(make_args())

        assert envs["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "true"
        assert envs["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] == "false"

    def test_the_built_in_defaults_apply_without_a_process_override(self, monkeypatch):
        """Without an override the engine must still get miles' own safety values rather than sglang's."""
        monkeypatch.delenv("SGLANG_JIT_DEEPGEMM_PRECOMPILE", raising=False)
        monkeypatch.delenv("SGLANG_MEMORY_SAVER_CUDA_GRAPH", raising=False)

        envs = compute_inference_engine_env_vars(make_args())

        assert envs["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] == "false"
        assert envs["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] == "true"

    def test_custom_all_reduce_v2_is_disabled_only_for_colocated_multi_gpu_engines(self, monkeypatch):
        """Only a colocated engine spanning several gpus hits the v2 all-reduce conflict; disabling it elsewhere
        silently gives up throughput."""
        monkeypatch.delenv("SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2", raising=False)

        def _value_for(*, colocate: bool, num_gpus_per_engine: int) -> str:
            args = make_args(colocate=colocate, rollout_num_gpus_per_engine=num_gpus_per_engine)
            return compute_inference_engine_env_vars(args)["SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2"]

        assert _value_for(colocate=True, num_gpus_per_engine=2) == "0"
        assert _value_for(colocate=True, num_gpus_per_engine=1) == "1"
        assert _value_for(colocate=False, num_gpus_per_engine=2) == "1"


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

        assert [spec.name for spec in specs] == ["inference-engine-all-0-0", "inference-engine-all-0-2"]
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

    def test_external_rollout_produces_no_engine_spec(self, tmp_path):
        """Externally launched engines are the operator's to run, so miles must not spec its own."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
            )
        )
        args = make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=8,
            rollout_external=True,
            rollout_external_engine_addrs=["host1:8000"],
        )

        assert specs_inference_engine(args) == []


class TestComputeEnginePools:
    def test_only_engine_specs_are_named(self, tmp_path):
        """These names are what the controller watches, so a router in the list would be reconciled as an engine."""
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

        assert compute_engine_pool_ids(args) == ["inference-engine-all-0-0", "inference-engine-all-0-2"]


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
        assert {name for name, info in ports.items() if info.mode == "per_worker"} == {
            "primary",
            "nccl",
            "engine_info_bootstrap",
        }

    def test_the_gate_port_is_allocated_once_per_cell(self, tmp_path):
        """The out-of-band launch gate lives on the cell's rank-0 engine, like dist_init."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4)

        ports = {info.name: info for info in specs_inference_engine(args)[0].port_infos}

        assert ports["gate"].mode == "master"
        assert ports["gate"].allow_dynamic is True
        assert ports["gate"].num_consecutive == 1

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
            "gate",
        ]
        assert [info.name for info in decode.port_infos] == [
            "primary",
            "dist_init",
            "nccl",
            "engine_info_bootstrap",
            "gate",
        ]


class TestInferenceEngineGatedLaunch:
    def test_the_launch_command_is_told_the_cells_own_gate_port(self, tmp_path, monkeypatch):
        """An engine launched without its gate port would start ungated and ignore the release."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4)
        recorded: dict = {}

        def _record(**kwargs) -> str:
            recorded.update(kwargs)
            return "launch-cmd"

        monkeypatch.setattr(inference_specs, "compute_engine_launch_cmd", _record)
        (spec,) = specs_inference_engine(args)
        spec.launch_command(_make_engine_ctx())

        assert recorded["gated_launch_port"] == 13007

    def test_each_node_of_a_multi_node_engine_is_numbered_within_its_own_cell(self, tmp_path, monkeypatch):
        """node_rank is what tells sglang which member of its own two-node group a process is.
        Numbering it globally would launch the second engine as ranks 2 and 3 of a two-node
        group, and both engines would hang in dist_init with no error."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 16, "num_gpus_per_engine": 8}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=16, num_gpus_per_node=4)
        recorded: list[int] = []

        def _record(**kwargs) -> str:
            recorded.append(kwargs["node_rank"])
            return "launch-cmd"

        monkeypatch.setattr(inference_specs, "compute_engine_launch_cmd", _record)
        (spec,) = specs_inference_engine(args)
        for cell_index in range(2):
            for worker_in_cell_index in range(2):
                spec.launch_command(_make_engine_ctx(cell_index=cell_index, worker_in_cell_index=worker_in_cell_index))

        assert recorded == [0, 1, 0, 1]


def _make_engine_ctx(*, cell_index: int = 0, worker_in_cell_index: int = 0) -> LaunchCommandContext:
    return LaunchCommandContext(
        cell_index=cell_index,
        worker_in_cell_index=worker_in_cell_index,
        self_addrs=dict(
            primary=HostAndPort(host="10.0.0.1", port=30000),
            dist_init=HostAndPort(host="10.0.0.1", port=9000),
            nccl=HostAndPort(host="10.0.0.1", port=10000),
            engine_info_bootstrap=HostAndPort(host="10.0.0.1", port=12000),
            gate=HostAndPort(host="10.0.0.1", port=13007),
        ),
        spec_addrs={},
        gpu_ids=[0, 1],
    )


class TestEngineMetaApiKey:
    def _meta_for(self, tmp_path, *, overrides_yaml: str = "", **args_overrides):
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            "sglang:\n"
            "  - name: default\n"
            "    server_groups:\n"
            "      - worker_type: regular\n"
            "        num_gpus: 8\n"
            "        num_gpus_per_engine: 1\n" + overrides_yaml
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=8, **args_overrides)
        (spec,) = specs_inference_engine(args)
        return spec.meta(WorkerMetaContext(cell_index=0))

    def test_a_group_api_key_override_wins_over_the_args_key(self, tmp_path):
        """The ServerArgs-named api_key override reaches the cell meta ahead of the global args key."""
        meta = self._meta_for(
            tmp_path,
            overrides_yaml="        overrides:\n          api_key: from-override\n",
            sglang_api_key="from-args",
        )
        assert meta["sglang_api_key"] == "from-override"

    def test_the_args_key_is_used_without_an_override(self, tmp_path):
        """Without a group override the engine api key falls back to args.sglang_api_key."""
        meta = self._meta_for(tmp_path, sglang_api_key="from-args")
        assert meta["sglang_api_key"] == "from-args"

    def test_an_explicit_empty_override_is_kept_verbatim(self, tmp_path):
        """An override disabling the key must win over the args key instead of silently falling back."""
        meta = self._meta_for(
            tmp_path,
            overrides_yaml='        overrides:\n          api_key: ""\n',
            sglang_api_key="from-args",
        )
        assert meta["sglang_api_key"] == ""


class TestTrailingPartialEngineRejection:
    def _specs_for(self, tmp_path, *, num_gpus: int, num_gpus_per_engine: int):
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[
                    {"worker_type": "regular", "num_gpus": num_gpus, "num_gpus_per_engine": num_gpus_per_engine}
                ]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=num_gpus, num_gpus_per_node=8)
        return specs_inference_engine(args)

    def test_a_trailing_partial_multi_node_engine_is_rejected(self, tmp_path):
        """24 GPUs cannot host 16-GPU engines on 8-GPU nodes and must fail fast instead of silently flooring."""
        with pytest.raises(AssertionError, match="whole number of"):
            self._specs_for(tmp_path, num_gpus=24, num_gpus_per_engine=16)

    def test_a_whole_number_of_multi_node_engines_passes(self, tmp_path):
        """32 GPUs host exactly two 16-GPU engines and resolve into two cells."""
        (spec,) = self._specs_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert spec.scheduling.num_cells == 2


class TestEngineCellChunking:
    def _spec_for(self, tmp_path, *, num_gpus: int, num_gpus_per_engine: int, gpu_offset: int = 0):
        config_path = tmp_path / "sglang.yaml"
        groups = []
        if gpu_offset:
            groups.append(
                {"worker_type": "placeholder", "num_gpus": gpu_offset, "num_gpus_per_engine": num_gpus_per_engine}
            )
        groups.append({"worker_type": "regular", "num_gpus": num_gpus, "num_gpus_per_engine": num_gpus_per_engine})
        config_path.write_text(make_sglang_config_yaml(server_groups=groups))
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=num_gpus + gpu_offset, num_gpus_per_node=8)
        return specs_inference_engine(args)[-1]

    def test_a_single_gpu_engine_becomes_its_own_cell(self, tmp_path):
        """With one gpu per engine on 8-gpu nodes, the group resolves into eight one-worker cells."""
        spec = self._spec_for(tmp_path, num_gpus=8, num_gpus_per_engine=1)
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (8, 1)

    def test_a_multi_node_engine_chunks_its_node_ranks_into_one_cell(self, tmp_path):
        """A 16-gpu engine on 8-gpu nodes spans two workers, so 32 gpus collapse into two cells."""
        spec = self._spec_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (2, 2)

    def test_a_multi_node_engine_gets_one_pod_per_node(self, tmp_path):
        """The chart turns this into the pods of a leaderworkerset, so a wrong count mis-sizes every group."""
        spec = self._spec_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)

        assert spec.scheduling.num_gpus_per_node == 8
        assert (spec.scheduling.pods_per_cell(), spec.scheduling.workers_per_pod()) == (2, 1)

    def test_an_engine_inside_one_node_stays_in_one_pod(self, tmp_path):
        """A cell that fits a node must not be split, or its ranks would talk over the network for nothing."""
        spec = self._spec_for(tmp_path, num_gpus=8, num_gpus_per_engine=4)

        assert (spec.scheduling.pods_per_cell(), spec.scheduling.workers_per_pod()) == (1, 1)

    def test_single_gpu_cells_carry_contiguous_gpu_offsets(self, tmp_path):
        """Every cell must claim its own gpu span, otherwise two engines share the same devices."""
        spec = self._spec_for(tmp_path, num_gpus=8, num_gpus_per_engine=1)
        offsets = [spec.meta(WorkerMetaContext(cell_index=index))["gpu_offset"] for index in range(8)]
        assert offsets == list(range(8))

    def test_multi_node_cells_advance_by_a_whole_engine(self, tmp_path):
        """The per-cell stride is workers x slots, so a 16-gpu engine advances the offset by 16, not by 1."""
        spec = self._spec_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        offsets = [spec.meta(WorkerMetaContext(cell_index=index))["gpu_offset"] for index in range(2)]
        assert offsets == [0, 16]

    def test_the_group_gpu_offset_shifts_every_cell(self, tmp_path):
        """A group placed after another starts counting from that group's end, per cell as well as overall."""
        spec = self._spec_for(tmp_path, num_gpus=16, num_gpus_per_engine=1, gpu_offset=8)
        offsets = [spec.meta(WorkerMetaContext(cell_index=index))["gpu_offset"] for index in range(16)]
        assert offsets == list(range(8, 24))


class TestSpecInferenceController:
    def _args(self, tmp_path, **overrides) -> Namespace:
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 4}]
            )
        )
        return make_args(sglang_config=str(config_path), rollout_num_gpus=8, **overrides)

    def _ctor_context(self, capability: FakeBackendCapability) -> WorkerCtorContext:
        return WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

    def test_every_run_gets_exactly_one_gpuless_controller(self, tmp_path):
        """It is a control-plane worker on both backends; a gpu request would reserve a whole node for it."""
        spec = spec_inference_controller(self._args(tmp_path))

        assert spec.name == INFERENCE_CONTROLLER_POOL_ID
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (1, 1)
        assert spec.scheduling.num_gpus_per_worker == 0
        assert spec.scheduling.num_gpu_slots_per_worker == 0

    def test_the_worker_class_is_the_controller_itself(self, tmp_path):
        """The spec names the class a pod or actor constructs, so it must be the real implementation."""
        spec = spec_inference_controller(self._args(tmp_path))

        assert load_function(spec.worker_class) is InferenceController

    def test_it_declares_only_the_platform_reads_its_provider_performs(self, tmp_path):
        """Watching engine pods needs reads, while deleting them remains outside this worker's capability."""
        spec = spec_inference_controller(self._args(tmp_path))

        assert spec.needs_platform_read_permission is True
        assert spec.needs_platform_delete_permission is False

    def test_the_registration_reporter_declares_only_platform_reads(self, tmp_path):
        """Reporting engine pods must not grant the standalone reporter either deletion or uninstall rights."""
        (spec,) = specs_inference_registration_reporter(self._args(tmp_path, deploy_component="inference"))

        assert spec.needs_platform_read_permission is True
        assert spec.needs_platform_delete_permission is False

    def test_the_worker_name_is_stable(self):
        """The driver looks the controller up by name, so this name is part of the release's contract."""
        assert inference_controller_worker_name() == "inference-controller-00000-00000"

    def test_it_renders_into_static_workers_with_its_rpc_port(self, tmp_path):
        """The release has to contain the controller pod, or the address book would point at nothing."""
        spec = spec_inference_controller(self._args(tmp_path))

        values = build_values([spec], _controller_layout()).as_values()

        (entry,) = values["run"]["staticWorkers"]
        assert SECTION_OF_CATEGORY[spec.category] == "staticWorkers"
        assert entry["name"] == INFERENCE_CONTROLLER_POOL_ID
        assert entry["ports"] == [{"name": "rpc", "port": 8000}]
        assert entry["command"][entry["command"].index("--pool-id") + 1] == INFERENCE_CONTROLLER_POOL_ID
        assert spec.worker_class == INFERENCE_CONTROLLER_WORKER_CLASS
        assert "resources" not in entry

    def test_it_asks_for_a_provider_over_the_engine_pools_it_will_observe(self, tmp_path):
        """The controller never learns which backend reports those cells, only which pools it wants reported."""
        args = self._args(tmp_path)
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        kwargs = spec_inference_controller(args).ctor_kwargs(self._ctor_context(capability))

        assert capability.requested_pool_ids == [compute_engine_pool_ids(args)]
        assert kwargs["engine_provider"] is capability.cells_provider

    def test_it_asks_for_one_router_provider_per_model(self, tmp_path):
        """Every model is served by its own router pool, so one provider cannot answer for all of them."""
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        kwargs = spec_inference_controller(self._args(tmp_path)).ctor_kwargs(self._ctor_context(capability))

        assert capability.requested_static_pool_ids == [compute_router_pool_id(0)]
        assert kwargs["router_providers"] == [capability.static_provider]

    def test_a_train_only_run_builds_a_controller_over_an_empty_pool(self, tmp_path):
        """--debug-train-only deploys no engines, so the controller observes no pools at all."""
        args = self._args(tmp_path, debug_train_only=True)
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        spec_inference_controller(args).ctor_kwargs(self._ctor_context(capability))

        assert capability.requested_pool_ids == [[]]

    def test_the_static_discovery_path_never_asks_the_backend(self, tmp_path, monkeypatch):
        """External engines belong to no backend, so the capability must never be asked for them."""
        args = self._args(
            tmp_path,
            rollout_external=True,
            rollout_external_engine_addrs=["host1:8000"],
            custom_inference_engine_provider_path=(
                "miles.ray.rollout.external_engine_provider.static_inference_engine_provider"
            ),
        )
        capability = FakeBackendCapability(cells_provider=None, static_provider=object())
        monkeypatch.setattr(
            external_engine_provider_module, "StaticInferenceEngineWorkerProvider", _RecordingStaticProvider
        )

        kwargs = spec_inference_controller(args).ctor_kwargs(self._ctor_context(capability))

        assert isinstance(kwargs["engine_provider"], _RecordingStaticProvider)
        assert kwargs["engine_provider"].args is args
        assert capability.requested_pool_ids == []

    def test_the_provider_factory_path_is_loaded_unconditionally(self, tmp_path):
        """Provider selection lives in arg validation, so the spec must run whatever path args carry."""
        args = self._args(
            tmp_path,
            rollout_external=True,
            rollout_external_engine_addrs=["host1:8000"],
            custom_inference_engine_provider_path=f"{__name__}._fake_engine_provider_factory",
        )
        capability = FakeBackendCapability(cells_provider=None, static_provider=object())

        kwargs = spec_inference_controller(args).ctor_kwargs(self._ctor_context(capability))

        assert kwargs["engine_provider"] == ("custom-provider", args, capability)


class _RecordingStaticProvider:
    def __init__(self, *, args) -> None:
        self.args = args


def _fake_engine_provider_factory(args, *, capability):
    return ("custom-provider", args, capability)


class TestTheEngineEnvironment:
    def test_every_engine_is_told_to_report_its_own_env_vars(self) -> None:
        """Without the gate the engine answers /server_info with no env_vars, and the audit is empty."""
        assert compute_inference_engine_env_vars(make_args())["SGLANG_EXPOSE_OWN_ENV_VARS"] == "1"


class TestRegistrationWiring:
    @staticmethod
    def _args(tmp_path, **overrides) -> Namespace:
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 4}]
            )
        )
        return make_args(sglang_config=str(config_path), rollout_num_gpus=8, **overrides)

    @staticmethod
    def _ctor_context(capability: FakeBackendCapability) -> WorkerCtorContext:
        return WorkerCtorContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[], capability=capability)

    def test_a_run_serving_its_own_engines_keeps_the_engine_provider_it_always_had(self, tmp_path):
        """Every unsplit run must reach its own engines exactly as it did before registration existed."""
        args = self._args(tmp_path)
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        kwargs = spec_inference_controller(args).ctor_kwargs(self._ctor_context(capability))

        assert not isinstance(kwargs["engine_provider"], RegistrationHub)

    def test_a_run_that_deploys_no_engines_of_its_own_serves_from_the_registered_ones(self, tmp_path):
        """The rest of the run must not know which deployment launched an engine it generates from."""
        args = self._args(tmp_path, deploy_component="primary")
        capability = FakeBackendCapability(cells_provider=object(), static_provider=object())

        kwargs = spec_inference_controller(args).ctor_kwargs(self._ctor_context(capability))

        assert isinstance(kwargs["engine_provider"], RegistrationHub)

    def test_an_engine_deployment_reports_into_the_controller_it_was_given(self, tmp_path):
        """It derives no name of another release, so this address is the only way it finds the run."""
        args = self._args(
            tmp_path,
            deploy_component="inference",
            inference_controller_addr="controller:9000",
        )
        capability = FakeBackendCapability(cells_provider=object())

        provider = compute_inference_controller_provider(args, capability=capability)

        assert isinstance(provider, StaticWorkerProvider)
        addrs = asyncio.run(provider.get_addrs(f"{INFERENCE_CONTROLLER_POOL_ID}-00000-00000"))
        assert addrs[RPC_PORT_NAME] == HostAndPort(host="controller", port=9000)
        assert capability.requested_static_pool_ids == []

    def test_a_run_that_holds_its_controller_addresses_it_by_its_own_release(self, tmp_path):
        """Naming another release's pods from here is exactly what a split run may not do."""
        args = self._args(tmp_path)
        capability = FakeBackendCapability(static_provider=object())

        compute_inference_controller_provider(args, capability=capability)

        assert capability.requested_static_pool_ids == [INFERENCE_CONTROLLER_POOL_ID]

    def test_an_engine_deployment_names_its_pools_after_the_instance_it_deploys(self, tmp_path):
        """Two engine groups of one run install the same pools, and a shared name would collide in the run."""
        args = self._args(tmp_path, deploy_component="inference", deploy_instance_id="west")

        assert compute_engine_pool_ids(args) == ["inference-engine-west-0-0"]

    def test_a_run_deploying_its_own_engines_names_its_pools_after_the_component(self, tmp_path):
        """Every pool id carries a segment, so the unsplit run falls back to the component it deploys."""
        assert compute_engine_pool_ids(self._args(tmp_path)) == ["inference-engine-all-0-0"]
