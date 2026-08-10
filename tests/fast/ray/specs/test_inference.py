from __future__ import annotations

import shlex
import sys
from argparse import Namespace

import pytest
from tests.fast.fixtures.capability_fixtures import FakeBackendCapability
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.backends.sglang_utils.router_args_utils import parse_router_args_argv
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig, resolve_sglang_config
from miles.ray.rollout.inference_controller import InferenceController
from miles.ray.specs import inference as inference_specs
from miles.ray.specs.inference import (
    INFERENCE_CONTROLLER_POOL_ID,
    _compute_router_primary_port_info,
    _compute_session_server_primary_port_info,
    _compute_spec_router,
    compute_engine_pool_id,
    compute_engine_pool_ids,
    compute_inference_engine_env_vars,
    compute_router_pool_id,
    inference_controller_worker_name,
    spec_inference_controller,
    spec_session_server,
    specs_inference_engine,
    specs_router,
)
from miles.rollout.session.config import SessionServerConfig
from miles.router.config import MilesRouterConfig
from miles.utils.function_registry import load_function
from miles.utils.workers.argv_utils import parse_config_argv
from miles.utils.workers.worker_spec import HostAndPort, LaunchCommandContext, WorkerCtorContext, WorkerMetaContext


def _make_model_cfg(*worker_types: str) -> ModelConfig:
    groups = [
        ServerGroupConfig(
            worker_type=worker_type,
            num_gpus=4,
            num_gpus_per_engine=4,
            gpu_offset=group_index * 4,
            engine_offset=group_index,
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
        pool_addrs={},
        gpu_ids=[],
        local_gpu_ids=[],
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
        """The session server command targets the router addr from pool_addrs and its config parses back losslessly."""
        args = make_args(
            use_session_server="v1",
            hf_checkpoint="/fake/model",
            session_server_workers=2,
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
            pool_addrs={compute_router_pool_id(0): [dict(primary=HostAndPort(host="127.0.0.1", port=3000))]},
            gpu_ids=[],
            local_gpu_ids=[],
        )
        argv = shlex.split(spec.launch_command(ctx))

        assert argv[:3] == [sys.executable, "-m", "miles.rollout.session.server"]
        config = parse_config_argv(SessionServerConfig, argv[3:])
        assert config.backend_url == "http://127.0.0.1:3000"
        assert config.host == "127.0.0.1"
        assert config.port == 5006
        assert config.instance_id == f"{args.run_uuid}-1"

    def test_it_reserves_no_cpu_on_the_head_node(self):
        """Pinned to the head unconditionally, a CPU reservation would leave it pending forever on a head started with --num-cpus=0."""
        spec = spec_session_server(_make_session_server_args())

        assert spec.scheduling.pin_to_head is True
        assert spec.scheduling.num_cpus_per_worker == 0

    def test_disabled_schedules_zero_cells(self):
        """Disabling the session server removes its cells instead of launching idle servers."""
        args = make_args(use_session_server=False)
        assert spec_session_server(args).scheduling.num_cells == 0

    def test_debug_train_only_schedules_zero_cells(self):
        """Its launch command reads the router address, and --debug-train-only leaves the router unlaunched."""
        args = _make_session_server_args(debug_train_only=True)

        assert spec_session_server(args).scheduling.num_cells == 0

    def test_only_the_debug_train_only_flag_empties_an_enabled_session_server(self):
        """An enabled session server keeps every requested cell until --debug-train-only takes the router away."""
        cells = {
            debug_train_only: spec_session_server(
                _make_session_server_args(debug_train_only=debug_train_only)
            ).scheduling.num_cells
            for debug_train_only in (False, True)
        }

        assert cells == {False: 2, True: 0}


def _make_session_server_args(**overrides) -> Namespace:
    return make_args(
        use_session_server="v1",
        session_server_workers=2,
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
            pool_addrs={compute_router_pool_id(0): [dict(primary=HostAndPort(host="127.0.0.1", port=3000))]},
            gpu_ids=[],
            local_gpu_ids=[],
        )

        config = parse_config_argv(SessionServerConfig, shlex.split(spec.launch_command(ctx))[3:])

        assert config.host == "10.20.30.40"
        assert config.port == 5006


class TestSessionServerInterpreterFlags:
    def test_the_launch_command_carries_the_parent_interpreter_flags(self, monkeypatch: pytest.MonkeyPatch):
        """A session server launched by a bare interpreter runs with different semantics than its own job."""
        monkeypatch.setattr(
            sys,
            "orig_argv",
            [sys.executable, "-O", "-X", "faulthandler", "-m", "miles.train", "--config", "x.yaml"],
        )
        spec = spec_session_server(_make_session_server_args())
        ctx = LaunchCommandContext(
            cell_index=0,
            worker_in_cell_index=0,
            self_addrs=dict(primary=HostAndPort(host="127.0.0.1", port=5006)),
            pool_addrs={compute_router_pool_id(0): [dict(primary=HostAndPort(host="127.0.0.1", port=3000))]},
            gpu_ids=[],
            local_gpu_ids=[],
        )

        argv = shlex.split(spec.launch_command(ctx))

        assert argv[:6] == [sys.executable, "-O", "-X", "faulthandler", "-m", "miles.rollout.session.server"]
        assert parse_config_argv(SessionServerConfig, argv[6:]).port == 5006

    def test_the_flags_are_captured_when_the_spec_is_built(self, monkeypatch: pytest.MonkeyPatch):
        """The command is rendered inside the worker manager actor, whose own argv is Ray's worker script."""
        monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-O", "-m", "miles.train"])
        spec = spec_session_server(_make_session_server_args())
        monkeypatch.setattr(sys, "orig_argv", [sys.executable, "/ray/workers/setup_worker.py"])
        ctx = LaunchCommandContext(
            cell_index=0,
            worker_in_cell_index=0,
            self_addrs=dict(primary=HostAndPort(host="127.0.0.1", port=5006)),
            pool_addrs={compute_router_pool_id(0): [dict(primary=HostAndPort(host="127.0.0.1", port=3000))]},
            gpu_ids=[],
            local_gpu_ids=[],
        )

        argv = shlex.split(spec.launch_command(ctx))

        assert argv[:4] == [sys.executable, "-O", "-m", "miles.rollout.session.server"]


class TestSessionServerRouterPoolLookup:
    def _make_ctx(self, pool_addrs: dict) -> LaunchCommandContext:
        return LaunchCommandContext(
            cell_index=0,
            worker_in_cell_index=0,
            self_addrs=dict(primary=HostAndPort(host="127.0.0.1", port=5006)),
            pool_addrs=pool_addrs,
            gpu_ids=[],
            local_gpu_ids=[],
        )

    def _make_args(self) -> Namespace:
        return _make_session_server_args(sglang_router_ip=None, sglang_router_port=None)

    def test_the_backend_is_read_under_the_router_specs_own_pool_id(self):
        """The key the session server looks up is exactly the name the router pool is registered under."""
        args = self._make_args()
        router_spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))
        ctx = self._make_ctx({router_spec.name: [dict(primary=HostAndPort(host="10.0.0.2", port=3210))]})

        config = parse_config_argv(SessionServerConfig, shlex.split(spec_session_server(args).launch_command(ctx))[3:])

        assert config.backend_url == "http://10.0.0.2:3210"

    def test_an_address_map_keyed_by_worker_names_fails_loudly(self):
        """The map is keyed per pool, not per worker, so a worker-keyed map must raise instead of launching unwired."""
        args = self._make_args()
        ctx = self._make_ctx(
            {f"{compute_router_pool_id(0)}-0-0": [dict(primary=HostAndPort(host="10.0.0.2", port=3210))]}
        )

        with pytest.raises(KeyError):
            spec_session_server(args).launch_command(ctx)

    def test_another_models_router_pool_is_not_mistaken_for_the_first(self):
        """With several router pools present the session server must still target model 0's router."""
        args = self._make_args()
        ctx = self._make_ctx(
            {
                compute_router_pool_id(1): [dict(primary=HostAndPort(host="10.0.0.3", port=3211))],
                compute_router_pool_id(0): [dict(primary=HostAndPort(host="10.0.0.2", port=3210))],
            }
        )

        config = parse_config_argv(SessionServerConfig, shlex.split(spec_session_server(args).launch_command(ctx))[3:])

        assert config.backend_url == "http://10.0.0.2:3210"


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


class TestSpecsRouter:
    def test_one_router_spec_per_model_is_specced_by_default(self, tmp_path):
        """Rollout runs still get their router, one per model in the sglang config."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=8)

        assert [spec.name for spec in specs_router(args)] == ["inference-router-0"]

    def test_debug_train_only_produces_no_router_spec(self, tmp_path):
        """--debug-train-only starts no engines, so a router worker would only wait for workers that never come."""
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

        assert specs_router(args) == []

    def test_flipping_only_the_debug_train_only_flag_removes_the_router(self, tmp_path):
        """One identical run specs a router with the flag off and none with it on, so nothing else decides it."""
        names = {
            debug_train_only: [
                spec.name for spec in specs_router(_make_router_args(tmp_path, debug_train_only=debug_train_only))
            ]
            for debug_train_only in (False, True)
        }

        assert names == {False: ["inference-router-0"], True: []}

    def test_debug_train_only_skips_the_router_however_the_router_itself_is_configured(self, tmp_path):
        """Pinning a port, picking the miles router or asking for session servers must not resurrect it."""
        args = _make_router_args(
            tmp_path,
            debug_train_only=True,
            use_miles_router=True,
            sglang_router_port=31000,
            use_session_server="v1",
        )

        assert specs_router(args) == []

    def test_debug_train_only_skips_the_router_in_a_disaggregated_run(self, tmp_path):
        """The skip is keyed on the flag alone, so a prefill/decode config gets no router either."""
        args = _make_router_args(
            tmp_path,
            server_groups=[
                {"worker_type": "prefill", "num_gpus": 4, "num_gpus_per_engine": 1},
                {"worker_type": "decode", "num_gpus": 4, "num_gpus_per_engine": 1},
            ],
            debug_train_only=True,
        )

        assert specs_router(args) == []


def _make_router_args(tmp_path, *, server_groups: list[dict] | None = None, **overrides) -> Namespace:
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        make_sglang_config_yaml(
            server_groups=server_groups or [{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
        )
    )
    return make_args(sglang_config=str(config_path), rollout_num_gpus=8, **overrides)


class TestComputeEngineSpecNames:
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

        assert compute_engine_pool_ids(args) == ["inference-engine-0-0", "inference-engine-0-2"]


class TestInferenceSpecPinToHead:
    @pytest.mark.parametrize("pinned", [False, True])
    def test_the_router_spec_follows_the_rollout_manager_flag(self, pinned: bool):
        """The router is pinned to the head node exactly when the rollout manager is."""
        from miles.ray.specs.inference import _compute_spec_router

        args = _make_pin_args(pinned=pinned)

        router = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        assert router.scheduling.pin_to_head is pinned

    @pytest.mark.parametrize("pinned", [False, True])
    def test_the_session_servers_are_always_pinned_to_the_head_node(self, pinned: bool):
        """Session servers live on the driver host whatever the rollout manager flag says, as on main."""
        from miles.ray.specs.inference import spec_session_server

        args = _make_pin_args(pinned=pinned)

        session = spec_session_server(args)

        assert session.scheduling.pin_to_head is True


def _make_pin_args(*, pinned: bool):
    return make_args(
        pin_rollout_manager_to_head=pinned,
        use_miles_router=False,
        use_session_server=True,
        hf_checkpoint="/fake/model",
        session_server_workers=1,
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


class TestInferenceEngineRandomSeed:
    _CONFIG_YAML = (
        "sglang:\n"
        "  - name: default\n"
        "    server_groups:\n"
        "      - worker_type: regular\n"
        "        num_gpus: 8\n"
        "        num_gpus_per_engine: 2\n"
        "      - worker_type: placeholder\n"
        "        num_gpus: 4\n"
        "        num_gpus_per_engine: 4\n"
        "  - name: reference\n"
        "    update_weights: false\n"
        "    server_groups:\n"
        "      - worker_type: regular\n"
        "        num_gpus: 16\n"
        "        num_gpus_per_engine: 8\n"
    )

    def _seeds_by_pool(self, args, monkeypatch) -> dict[str, list[int]]:
        recorded: dict[str, list[int]] = {}
        for spec in specs_inference_engine(args):
            seeds = recorded.setdefault(spec.name, [])

            def _record(*, into: list[int] = seeds, **kwargs) -> str:
                into.append(kwargs["random_seed"])
                return "launch-cmd"

            monkeypatch.setattr(inference_specs, "compute_engine_launch_cmd", _record)
            for cell_index in range(spec.scheduling.num_cells):
                for worker_in_cell_index in range(spec.scheduling.num_workers_per_cell):
                    spec.launch_command(
                        _make_engine_ctx(cell_index=cell_index, worker_in_cell_index=worker_in_cell_index)
                    )
        return recorded

    def _oracle_seeds_by_pool(self, args) -> dict[str, list[int]]:
        seeds: dict[str, list[int]] = {}
        global_rank = 0
        for model_idx, model_cfg in enumerate(resolve_sglang_config(args).models):
            for group_index, group_cfg in enumerate(model_cfg.server_groups):
                num_actors = group_cfg.num_gpus // min(group_cfg.num_gpus_per_engine, args.num_gpus_per_node)
                if group_cfg.worker_type != "placeholder":
                    pool_id = compute_engine_pool_id(model_idx=model_idx, group_index=group_index)
                    seeds[pool_id] = [args.seed + global_rank + i for i in range(num_actors)]
                global_rank += num_actors
        return seeds

    @pytest.fixture
    def args(self, tmp_path):
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(self._CONFIG_YAML)
        return make_args(sglang_config=str(config_path), rollout_num_gpus=28, num_gpus_per_node=4, seed=1000)

    def test_every_actor_keeps_the_seed_the_pre_refactor_rank_gave_it(self, args, monkeypatch):
        """Reproducing a speculative-decoding run needs each engine actor's seed to stay put."""
        assert self._seeds_by_pool(args, monkeypatch) == self._oracle_seeds_by_pool(args)

    def test_a_skipped_placeholder_group_still_advances_the_numbering(self, args, monkeypatch):
        """A placeholder group consumed ranks before the refactor, so ignoring it would shift every later seed."""
        seeds = self._seeds_by_pool(args, monkeypatch)

        assert seeds[compute_engine_pool_id(model_idx=1, group_index=0)] == [1005, 1006, 1007, 1008]

    def test_no_two_engine_actors_in_the_cluster_share_a_seed(self, args, monkeypatch):
        """Numbering every pool from the same base would hand two live engines the same RNG stream."""
        seeds = [seed for pool_seeds in self._seeds_by_pool(args, monkeypatch).values() for seed in pool_seeds]

        assert sorted(set(seeds)) == sorted(seeds)

    def test_an_actor_relaunched_onto_other_gpus_keeps_its_seed(self, args, monkeypatch):
        """A restarted engine that draws a fresh seed replays a different RNG stream than the run it resumes."""
        recorded: list[int] = []

        def _record(**kwargs) -> str:
            recorded.append(kwargs["random_seed"])
            return "launch-cmd"

        monkeypatch.setattr(inference_specs, "compute_engine_launch_cmd", _record)
        spec = specs_inference_engine(args)[-1]
        spec.launch_command(_make_engine_ctx(cell_index=1, worker_in_cell_index=1))
        spec.launch_command(
            _make_engine_ctx(cell_index=1, worker_in_cell_index=1, gpu_ids=[26, 27], local_gpu_ids=[2, 3])
        )

        assert recorded == [1008, 1008]

    def test_shifting_the_run_seed_shifts_every_engine_seed_by_the_same_amount(self, args, monkeypatch):
        """The per-engine numbers are an offset on top of --seed, so a rerun with another seed must move as a block."""
        base = self._seeds_by_pool(args, monkeypatch)
        shifted = self._seeds_by_pool(
            make_args(
                sglang_config=args.sglang_config,
                rollout_num_gpus=28,
                num_gpus_per_node=4,
                seed=args.seed + 7,
            ),
            monkeypatch,
        )

        assert shifted == {pool_id: [seed + 7 for seed in seeds] for pool_id, seeds in base.items()}


def _make_engine_ctx(
    *,
    cell_index: int = 0,
    worker_in_cell_index: int = 0,
    gpu_ids: list[int] | None = None,
    local_gpu_ids: list[int] | None = None,
) -> LaunchCommandContext:
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
        pool_addrs={},
        gpu_ids=[0, 1] if gpu_ids is None else gpu_ids,
        local_gpu_ids=[0, 1] if local_gpu_ids is None else local_gpu_ids,
    )


class TestEngineBaseGpuId:
    def test_the_launch_command_uses_the_gpu_ids_the_worker_itself_resolved(self, tmp_path, monkeypatch):
        """The manager runs without a visibility mask, so passing its physical ids would point sglang at
        a device the engine process cannot see."""
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
        spec.launch_command(_make_engine_ctx(gpu_ids=[6, 7], local_gpu_ids=[2, 3]))

        assert recorded["base_gpu_id"] == 2


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


class TestCrossNodeEngineWidth:
    def _specs_for(self, tmp_path, *, num_gpus: int, num_gpus_per_engine: int, num_gpus_per_node: int = 8):
        return self._specs_for_groups(
            tmp_path,
            server_groups=[
                {"worker_type": "regular", "num_gpus": num_gpus, "num_gpus_per_engine": num_gpus_per_engine}
            ],
            num_gpus_per_node=num_gpus_per_node,
        )

    def _specs_for_groups(self, tmp_path, *, server_groups: list[dict], num_gpus_per_node: int = 8):
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(make_sglang_config_yaml(server_groups=server_groups))
        args = make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=sum(group["num_gpus"] for group in server_groups),
            num_gpus_per_node=num_gpus_per_node,
        )
        return specs_inference_engine(args)

    def test_a_cross_node_engine_that_does_not_tile_the_node_is_rejected(self, tmp_path):
        """A 12-gpu engine on 8-gpu nodes silently specs one node of 12 ranks and then waits for ever."""
        with pytest.raises(AssertionError, match="nor tiles whole nodes"):
            self._specs_for(tmp_path, num_gpus=24, num_gpus_per_engine=12)

    def test_an_engine_that_fits_in_one_node_is_accepted(self, tmp_path):
        """Anything up to a node wide needs no tiling at all."""
        (spec,) = self._specs_for(tmp_path, num_gpus=8, num_gpus_per_engine=4)
        assert spec.scheduling.num_workers_per_cell == 1

    def test_an_engine_spanning_whole_nodes_is_accepted(self, tmp_path):
        """A 16-gpu engine covers exactly two 8-gpu nodes, so every rank has a node to run on."""
        (spec,) = self._specs_for(tmp_path, num_gpus=32, num_gpus_per_engine=16)
        assert spec.scheduling.num_workers_per_cell == 2

    def test_a_lone_cross_node_engine_is_rejected_even_with_no_leftover_gpus(self, tmp_path):
        """One 12-gpu engine on 12 gpus divides evenly, so only the tiling rule can catch its four unlaunched ranks."""
        with pytest.raises(AssertionError, match="nor tiles whole nodes"):
            self._specs_for(tmp_path, num_gpus=12, num_gpus_per_engine=12)

    def test_an_engine_narrower_than_a_node_need_not_divide_the_node(self, tmp_path):
        """A 6-gpu engine lives inside one 8-gpu node, so the node size need not be a multiple of it."""
        (spec,) = self._specs_for(tmp_path, num_gpus=12, num_gpus_per_engine=6)
        assert (spec.scheduling.num_workers_per_cell, spec.scheduling.num_gpu_slots_per_worker) == (1, 6)

    def test_the_node_width_in_the_check_comes_from_args_not_a_fixed_eight(self, tmp_path):
        """A 6-gpu engine is fine on 8-gpu nodes but straddles 4-gpu nodes, so the args value must drive the check."""
        with pytest.raises(AssertionError, match="nor tiles whole nodes"):
            self._specs_for(tmp_path, num_gpus=12, num_gpus_per_engine=6, num_gpus_per_node=4)

    def test_an_engine_tiling_a_smaller_node_is_accepted(self, tmp_path):
        """An 8-gpu engine covers exactly two 4-gpu nodes, which a check hardcoded to 8-gpu nodes would misread."""
        (spec,) = self._specs_for(tmp_path, num_gpus=16, num_gpus_per_engine=8, num_gpus_per_node=4)
        assert (spec.scheduling.num_cells, spec.scheduling.num_workers_per_cell) == (2, 2)

    def test_a_later_group_is_checked_and_named_in_the_rejection(self, tmp_path):
        """Every group carries its own width, so a good first group must not excuse a bad second one."""
        with pytest.raises(AssertionError, match="group 'decode'.*num_gpus_per_engine=12"):
            self._specs_for_groups(
                tmp_path,
                server_groups=[
                    {"worker_type": "prefill", "num_gpus": 8, "num_gpus_per_engine": 8},
                    {"worker_type": "decode", "num_gpus": 24, "num_gpus_per_engine": 12},
                ],
            )

    def test_a_placeholder_group_width_is_not_checked(self, tmp_path):
        """A placeholder group only reserves gpus and launches no ranks, so its width cannot strand any."""
        specs = self._specs_for_groups(
            tmp_path,
            server_groups=[
                {"worker_type": "placeholder", "num_gpus": 12, "num_gpus_per_engine": 12},
                {"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 8},
            ],
        )
        assert [spec.scheduling.num_cells for spec in specs] == [1]


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


class TestRouterInterpreterFlags:
    @pytest.mark.parametrize(
        ("use_miles_router", "module"),
        [(True, "miles.router.router"), (False, "sglang_router.launch_router")],
        ids=["miles_router", "sglang_router"],
    )
    def test_the_router_launch_command_carries_the_parent_interpreter_flags(
        self, monkeypatch: pytest.MonkeyPatch, use_miles_router: bool, module: str
    ):
        """A router launched by a bare interpreter runs with different semantics than the job that spawned it."""
        monkeypatch.setattr(
            sys,
            "orig_argv",
            [sys.executable, "-O", "-X", "faulthandler", "-m", "miles.train", "--config", "x.yaml"],
        )
        args = make_args(use_miles_router=use_miles_router, sglang_router_ip=None, sglang_router_port=None)
        spec = _compute_spec_router(args, model_idx=0, model_cfg=_make_model_cfg("regular"))

        argv = shlex.split(spec.launch_command(_make_router_ctx()))

        assert argv[:6] == [sys.executable, "-O", "-X", "faulthandler", "-m", module]


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

    def test_the_worker_name_is_stable(self):
        """The driver looks the controller up by name, so this name is part of the release's contract."""
        assert inference_controller_worker_name() == "inference-controller-0-0"

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
