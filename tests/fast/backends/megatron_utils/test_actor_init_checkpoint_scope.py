import inspect
from pathlib import Path
from types import ModuleType

import pytest
from tests.fast.backends.megatron_utils.init_checkpoint_fixtures import _init_worker
from tests.fast.backends.megatron_utils.test_actor import actor_module

from miles.backends.megatron_utils.checkpoint_request import CHECKPOINT_LOAD_FIELDS, MegatronCkptLoadConfig
from miles.utils.ft_utils.indep_dp import IndepDPInfo

__all__ = ["actor_module"]


class TestActorInitCheckpointScope:
    @pytest.mark.parametrize(
        "role,mode",
        [
            ("actor", "fresh"),
            ("critic", "fresh"),
            ("actor", "resume"),
            ("critic", "resume"),
            ("actor", "peer"),
            ("actor", "explicit"),
            ("critic", "explicit"),
        ],
    )
    @pytest.mark.parametrize("failure", [None, "build", "load"])
    def test_initialization_applies_only_the_current_request_and_cleans_every_exit(
        self,
        actor_module: ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        role: str,
        mode: str,
        failure: str | None,
    ) -> None:
        """Actor and critic initialization scope runtime checkpoint inputs through real model build and state loading."""
        iteration = 0 if mode == "fresh" else 7
        path = tmp_path / "checkpoint"
        worker, sdk = _init_worker(
            actor_module=actor_module,
            monkeypatch=monkeypatch,
            path=path,
            role=role,
            iteration=iteration,
            failure=failure,
            peer=mode == "peer",
        )
        if mode == "explicit":
            (path / "latest_checkpointed_iteration.txt").unlink()
            (path / "iter_0000007").mkdir()
        request = MegatronCkptLoadConfig(
            load=str(path),
            resume_from_ckpt=mode in {"resume", "peer"},
            finetune=mode == "fresh",
            no_load_optim=mode == "fresh",
            no_load_rng=mode == "fresh",
            ckpt_step=7 if mode == "explicit" else (0 if mode == "fresh" else None),
        )
        expected = MegatronCkptLoadConfig.for_peer_recovery() if mode == "peer" else request
        expected_fields = {name: vars(expected)[name] for name in CHECKPOINT_LOAD_FIELDS}
        info = (
            IndepDPInfo(cell_index=1, num_cells=2, alive_rank=1, alive_size=2, quorum_id=1, alive_cell_indices=[0, 1])
            if mode == "peer"
            else IndepDPInfo.create_trivial()
        )
        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(sdk.args.backend))
        original_pretrained_checkpoint = sdk.args.backend.pretrained_checkpoint

        def initialize() -> int:
            return inspect.unwrap(actor_module.MegatronTrainRayActor.init)(
                worker,
                args=sdk.args,
                role=role,
                ckpt_load_config=None if mode == "peer" else request,
                recv_ckpt_src_rank=0 if mode == "peer" else None,
                indep_dp_info=info,
                indep_dp_store_addr="127.0.0.1:1234" if mode == "peer" else None,
            )

        if failure is None:
            assert initialize() == (0 if mode == "fresh" else 8)
            assert worker.model[0].role == role
            assert worker.model[0].lora_attached is False
            assert worker.optimizer is worker.opt_param_scheduler is None
        else:
            with pytest.raises(RuntimeError, match="SDK failed"):
                initialize()

        assert sdk.args.backend.pretrained_checkpoint == original_pretrained_checkpoint
        assert sdk.observations["build"] == expected_fields
        if failure != "build":
            assert sdk.observations["load"] == expected_fields
            assert sdk.args.backend.consumed_train_samples == 64
            if mode == "peer":
                from miles.backends.megatron_utils.ft.in_memory_checkpoint import InMemoryCheckpointManager

                assert sdk.checkpointing_context is not None
                manager = sdk.checkpointing_context["local_checkpoint_manager"]
                assert isinstance(manager, InMemoryCheckpointManager)
                assert manager.find_latest() == iteration
            else:
                assert sdk.checkpointing_context is None
        else:
            assert "load" not in sdk.observations
        assert sdk.peer_sources == ([0] if mode == "peer" else [])
        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(sdk.args.backend))
        assert "requested_load" not in vars(sdk.args.backend)
        with pytest.raises(AttributeError, match="immutable"):
            sdk.args.backend.load = "/leaked"
