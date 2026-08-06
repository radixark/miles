from __future__ import annotations

from tests.fast.ray.rollout.conftest import make_args, make_sglang_config_yaml

from miles.ray.specs.entrypoint import compute_specs
from miles.utils.workers.worker_spec import BaseWorkerSpec


class TestComputeSpecs:
    def test_launches_the_controller_then_routers_then_the_session_server_then_every_engine(self, tmp_path):
        """The manager's whole inventory comes from here, so every component must be listed exactly once."""
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
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=16, use_session_server=False)

        specs = compute_specs(args)

        assert [spec.name for spec in specs] == [
            "inference-controller",
            "inference-router-0",
            "session-server",
            "inference-engine-0-0",
            "inference-engine-0-2",
            "trainer-actor",
        ]

    def test_a_disabled_session_server_is_listed_with_no_cells(self, tmp_path):
        """Disabling the session server must not remove it from the inventory, only empty it."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 4, "num_gpus_per_engine": 2}]
            )
        )
        args = make_args(sglang_config=str(config_path), rollout_num_gpus=4, use_session_server=False)

        specs = {spec.name: spec for spec in compute_specs(args)}

        assert specs["session-server"].scheduling.num_cells == 0
        assert specs["inference-engine-0-0"].scheduling.num_cells == 2

    def test_debug_train_only_lists_no_inference_engine(self, tmp_path):
        """--debug-train-only must instantiate no sglang engine, since its bundles are the trainer's own gpus."""
        config_path = tmp_path / "sglang.yaml"
        config_path.write_text(
            make_sglang_config_yaml(
                server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
            )
        )
        args = make_args(
            sglang_config=str(config_path),
            rollout_num_gpus=8,
            use_session_server=False,
            colocate=True,
            debug_train_only=True,
        )

        specs = compute_specs(args)

        assert [spec.name for spec in specs if spec.name.startswith("inference-engine")] == []

    def test_debug_train_only_lists_no_router(self, tmp_path):
        """A router has no engine to route to here, so paying a worker for it is pure waste."""
        specs = _debug_train_only_specs(tmp_path)

        assert [spec.name for spec in specs if spec.name.startswith("inference-router")] == []

    def test_debug_train_only_still_lists_the_trainer_and_an_empty_session_server(self, tmp_path):
        """Dropping the router must not drop the training side, and the session server survives with no cells."""
        specs = {spec.name: spec for spec in _debug_train_only_specs(tmp_path)}

        assert list(specs) == ["session-server", "trainer-actor"]
        assert specs["session-server"].scheduling.num_cells == 0


def _debug_train_only_specs(tmp_path) -> list[BaseWorkerSpec]:
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(
        make_sglang_config_yaml(server_groups=[{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}])
    )
    args = make_args(
        sglang_config=str(config_path),
        rollout_num_gpus=8,
        use_session_server="v1",
        session_server_workers=2,
        colocate=True,
        debug_train_only=True,
    )
    return compute_specs(args)
