from tests.fast.utils.args.mutation_permission_fixtures import _orchestrator_config


class TestRuntimeTraitMutationPermissions:
    def test_dataset_progress_fields_remain_mutable_on_the_real_orchestrator(self) -> None:
        """Dataset startup may update rollout progress on the actual component configuration."""
        config = _orchestrator_config()

        config.start_rollout_id = 3
        config.num_rollout = 7

        assert config.start_rollout_id == 3
        assert config.num_rollout == 7

    def test_router_discovery_can_publish_its_resolved_endpoints(self) -> None:
        """Router discovery can publish both the model map and the primary endpoint."""
        config = _orchestrator_config()

        config.sglang_model_routers = {"default": ("router", 8000)}
        config.sglang_router_ip = "router"
        config.sglang_router_port = 8000

        assert config.sglang_model_routers == {"default": ("router", 8000)}
        assert config.sglang_router_ip == "router"
        assert config.sglang_router_port == 8000

    def test_session_discovery_can_record_an_empty_instance_roster(self) -> None:
        """Session discovery distinguishes an observed empty roster from an uninitialized one."""
        config = _orchestrator_config()
        assert config.session_server_instances is None

        config.session_server_instances = []

        assert config.session_server_instances == []

    def test_mlflow_can_publish_its_assigned_run_identifier(self) -> None:
        """Tracking initialization may record an assigned MLflow run on the component."""
        config = _orchestrator_config()

        config.mlflow_run_id = "assigned-mlflow-run"

        assert config.mlflow_run_id == "assigned-mlflow-run"

    def test_wandb_can_publish_its_assigned_run_identifier(self) -> None:
        """Tracking initialization may record an assigned WandB run on the component."""
        config = _orchestrator_config()

        config.wandb_run_id = "assigned-wandb-run"

        assert config.wandb_run_id == "assigned-wandb-run"
