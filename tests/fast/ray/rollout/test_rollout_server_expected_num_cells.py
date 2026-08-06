from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
from tests.fast.ray.rollout.conftest import make_args

from miles.ray.rollout.rollout_server import RolloutServer, create_rollout_servers
from miles.ray.specs.inference import compute_engine_pool_id, specs_inference_engine
from miles.utils.context_lock import ContextLock
from miles.utils.ft_utils.health_checker import ActiveAndEpoch
from miles.utils.workers.worker_spec import HostAndPort

_CONFIG_SINGLE_GROUP: list[dict] = [
    dict(
        name="actor",
        server_groups=[dict(worker_type="regular", num_gpus=8, num_gpus_per_engine=1)],
    ),
]

_CONFIG_WITH_PLACEHOLDER: list[dict] = [
    dict(
        name="actor",
        server_groups=[
            dict(worker_type="regular", num_gpus=8, num_gpus_per_engine=4),
            dict(worker_type="placeholder", num_gpus=8, num_gpus_per_engine=4),
        ],
    ),
]

_CONFIG_PD_DISAGGREGATED: list[dict] = [
    dict(
        name="actor",
        server_groups=[
            dict(worker_type="prefill", num_gpus=4, num_gpus_per_engine=2),
            dict(worker_type="decode", num_gpus=8, num_gpus_per_engine=4),
        ],
    ),
]

_CONFIG_MULTI_MODEL: list[dict] = [
    dict(
        name="actor",
        server_groups=[
            dict(worker_type="regular", num_gpus=8, num_gpus_per_engine=2),
            dict(worker_type="placeholder", num_gpus=2, num_gpus_per_engine=2),
        ],
    ),
    dict(
        name="ref",
        model_path="/fake/ref-model",
        update_weights=False,
        server_groups=[dict(worker_type="regular", num_gpus=4, num_gpus_per_engine=4)],
    ),
]

_CONFIG_MULTI_NODE_ENGINE: list[dict] = [
    dict(
        name="actor",
        server_groups=[dict(worker_type="regular", num_gpus=16, num_gpus_per_engine=16)],
    ),
]


def _render_config_yaml(models: list[dict]) -> str:
    lines: list[str] = ["sglang:"]
    for model in models:
        lines.append(f"  - name: {model['name']}")
        if model.get("model_path") is not None:
            lines.append(f"    model_path: {model['model_path']}")
        if model.get("update_weights") is not None:
            lines.append(f"    update_weights: {str(model['update_weights']).lower()}")
        lines.append("    server_groups:")
        for group in model["server_groups"]:
            lines.append(f"      - worker_type: {group['worker_type']}")
            lines.append(f"        num_gpus: {group['num_gpus']}")
            lines.append(f"        num_gpus_per_engine: {group['num_gpus_per_engine']}")
    return "\n".join(lines) + "\n"


def _make_args_with_config(models: list[dict], tmp_path: Path) -> Namespace:
    config_path = tmp_path / "sglang_config.yaml"
    config_path.write_text(_render_config_yaml(models))
    total_num_gpus = sum(group["num_gpus"] for model in models for group in model["server_groups"])
    return make_args(
        sglang_config=str(config_path),
        rollout_num_gpus=total_num_gpus,
        num_gpus_per_node=8,
        debug_rollout_only=True,
    )


def _expected_num_cells_from_specs(args: Namespace) -> dict[int, int]:
    specs_by_name = {spec.name: spec for spec in specs_inference_engine(args)}
    counts: dict[int, int] = {}
    for name, spec in specs_by_name.items():
        model_idx = int(name.split("-")[-2])
        counts[model_idx] = counts.get(model_idx, 0) + spec.scheduling.num_cells
    return counts


def _make_router_addrs(models: list[dict]) -> dict[str, HostAndPort]:
    return {
        model["name"]: HostAndPort(host="127.0.0.1", port=20000 + model_idx) for model_idx, model in enumerate(models)
    }


async def _create_servers(args: Namespace, models: list[dict]) -> dict[str, RolloutServer]:
    return await create_rollout_servers(
        args,
        context_lock=ContextLock("InferenceController"),
        global_health_checker_activeness=lambda: ActiveAndEpoch(active=True, epoch=0),
        router_addrs=_make_router_addrs(models),
    )


class TestExpectedNumCellsMatchesTheEngineSpecs:
    @pytest.mark.parametrize(
        "models",
        [
            _CONFIG_SINGLE_GROUP,
            _CONFIG_WITH_PLACEHOLDER,
            _CONFIG_PD_DISAGGREGATED,
            _CONFIG_MULTI_MODEL,
            _CONFIG_MULTI_NODE_ENGINE,
        ],
        ids=["single_group", "with_placeholder", "pd_disaggregated", "multi_model", "multi_node_engine"],
    )
    async def test_the_startup_barrier_expects_exactly_the_cells_the_specs_launch(
        self, models: list[dict], tmp_path: Path
    ) -> None:
        """The barrier target must equal the engine cells RayWorkerManager actually starts, or startup hangs until timeout."""
        args = _make_args_with_config(models=models, tmp_path=tmp_path)
        expected_per_model_idx = _expected_num_cells_from_specs(args)

        servers = await _create_servers(args, models)

        actual_per_model_idx = {
            model_idx: servers[model["name"]].expected_num_cells for model_idx, model in enumerate(models)
        }
        assert actual_per_model_idx == expected_per_model_idx

    async def test_a_placeholder_group_contributes_no_cell_to_the_barrier(self, tmp_path: Path) -> None:
        """Placeholder groups only reserve GPU slots, so counting them would make the barrier unreachable."""
        args = _make_args_with_config(models=_CONFIG_WITH_PLACEHOLDER, tmp_path=tmp_path)

        servers = await _create_servers(args, _CONFIG_WITH_PLACEHOLDER)

        assert servers["actor"].expected_num_cells == 2

    async def test_every_model_gets_its_own_barrier_target(self, tmp_path: Path) -> None:
        """Sharing one pool size across models would block the small model behind the big one."""
        args = _make_args_with_config(models=_CONFIG_MULTI_MODEL, tmp_path=tmp_path)

        servers = await _create_servers(args, _CONFIG_MULTI_MODEL)

        assert servers["actor"].expected_num_cells == 4
        assert servers["ref"].expected_num_cells == 1

    async def test_every_model_talks_to_its_own_router(self, tmp_path: Path) -> None:
        """A server paired with another model's router would advertise its cells to the wrong fleet."""
        args = _make_args_with_config(models=_CONFIG_MULTI_MODEL, tmp_path=tmp_path)

        servers = await _create_servers(args, _CONFIG_MULTI_MODEL)

        actual = {
            name: HostAndPort(host=server.router_ip, port=server.router_port) for name, server in servers.items()
        }
        assert actual == _make_router_addrs(_CONFIG_MULTI_MODEL)


class TestEngineSpecNamingUsedByTheCrossCheck:
    def test_pool_names_carry_the_model_index_the_cross_check_parses(self) -> None:
        """The cross-check maps specs back to models by name, so that encoding must stay stable."""
        assert compute_engine_pool_id(model_idx=3, group_index=7) == "inference-engine-3-7"


class TestRouterFlagsAtStartup:
    async def test_a_pinned_router_port_is_accepted(self, tmp_path: Path) -> None:
        """Three examples pass --sglang-router-port so a firewall rule can name the port in
        advance, and the spec layer pins it; rejecting it here fails those launches outright."""
        args = _make_args_with_config(models=_CONFIG_SINGLE_GROUP, tmp_path=tmp_path)
        args.sglang_router_port = 31000

        assert await _create_servers(args)

    async def test_an_external_router_ip_is_still_rejected(self, tmp_path: Path) -> None:
        """Attaching to a router miles did not start is not supported yet, and silently starting
        one anyway would put two routers in front of the same engines."""
        args = _make_args_with_config(models=_CONFIG_SINGLE_GROUP, tmp_path=tmp_path)
        args.sglang_router_ip = "10.0.0.9"

        with pytest.raises(AssertionError, match="external router mode was removed"):
            await _create_servers(args)
