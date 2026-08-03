from __future__ import annotations

import shlex
import sys

import pytest
from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.backends.sglang_utils.router_args_utils import parse_router_args_argv
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig
from miles.ray.specs.inference import (
    _compute_spec_router,
    compute_engine_pool_ids,
    compute_router_pool_id,
    spec_session_server,
    specs_inference_engine,
)
from miles.rollout.session.config import SessionServerConfig
from miles.router.config import MilesRouterConfig
from miles.utils.workers.argv_utils import parse_config_argv
from miles.utils.workers.worker_spec import HostAndPort, LaunchCommandContext, WorkerMetaContext


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
        pool_addrs={},
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

    def test_it_reserves_no_cpu_on_the_head_node(self):
        """Pinned to the head unconditionally, a CPU reservation would leave it pending forever on a head started with --num-cpus=0."""
        spec = spec_session_server(_make_session_server_args())

        assert spec.scheduling.pin_to_head is True
        assert spec.scheduling.num_cpus_per_worker == 0

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
