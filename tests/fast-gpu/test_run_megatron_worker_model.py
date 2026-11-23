from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=30, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"], num_gpus=1
)

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from miles.backends.megatron_utils import checkpoint
from miles.backends.megatron_utils.checkpoint_request import MegatronCkptLoadConfig
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.runtime import TrainerConfig
from miles.utils.debug_utils.run_megatron.worker import main as worker


class TestStandaloneModelConstruction:
    @pytest.mark.parametrize("fails", [False, True])
    def test_model_build_restores_backend_immutability_after_sdk_success_or_failure(
        self, monkeypatch: pytest.MonkeyPatch, fails: bool
    ) -> None:
        """Megatron model construction temporarily permits derived fields without leaking mutability."""
        backend = MegatronArgsNamespace(load=None, model_type="before")
        args = TrainerConfig.model_construct(
            backend=backend, custom_model_provider_path=CustomFunctionConfig(path="test.unused_provider")
        )
        model = torch.nn.Linear(2, 2).eval()
        calls: list[bool] = []

        def build(provider: object, model_type: object, *, wrap_with_ddp: bool) -> list[torch.nn.Module]:
            calls.append(wrap_with_ddp)
            backend.model_type = "derived"
            if fails:
                raise RuntimeError("SDK model construction failed")
            return [model]

        monkeypatch.setattr(worker, "get_model", build)
        script = SimpleNamespace(role="actor", run_backward=False)

        if fails:
            with pytest.raises(RuntimeError, match="SDK model construction failed"):
                worker._build_and_load_model(args=args, script=script)
        else:
            assert worker._build_and_load_model(args=args, script=script) == [model]
            assert model.training

        assert calls == [False]
        assert backend.model_type == "derived"
        with pytest.raises(AttributeError, match="immutable"):
            backend.model_type = "leaked"

    def test_checkpoint_loading_receives_the_typed_trainer_and_current_request(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Standalone loading reaches the real checkpoint wrapper with its typed trainer and active request."""
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("7")
        args = TrainerConfig.model_construct(
            backend=MegatronArgsNamespace(spec=None),
            custom_model_provider_path=CustomFunctionConfig(path="test.unused_provider"),
            lora_rank=0,
        )
        request = MegatronCkptLoadConfig.for_peer_recovery().model_copy(update={"load": str(tmp_path)})
        model = torch.nn.Linear(2, 2).eval()
        loader = Mock(return_value=(7, 0))
        builder = Mock(return_value=[model])
        monkeypatch.setattr(worker, "get_model", builder)
        monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", loader)

        with request.apply(args.backend):
            result = worker._build_and_load_model(args=args, script=SimpleNamespace(role="actor", run_backward=True))

        assert result == [model]
        assert model.training
        builder.assert_called_once()
        assert builder.call_args.kwargs["wrap_with_ddp"] is True
        loader.assert_called_once()
        assert loader.call_args.kwargs["ddp_model"] == [model]
        assert "load" not in vars(args.backend)


class TestSharedStandaloneConfiguration:
    @pytest.mark.parametrize("role", ["actor", "critic"])
    @pytest.mark.parametrize("offload", [False, True])
    def test_real_shared_parser_builds_the_requested_typed_trainer(
        self, monkeypatch: pytest.MonkeyPatch, role: str, offload: bool
    ) -> None:
        """Standalone parsing accepts the requested role only with supported training options."""
        monkeypatch.setenv("WORLD_SIZE", "1")
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setenv("MILES_SCRIPT_ENV_REPORT", "")
        argv = [
            "worker",
            "--script-hf-checkpoint",
            "/model",
            "--script-token-ids-file",
            "/tokens",
            "--script-role",
            role,
            "--num-layers",
            "2",
            "--hidden-size",
            "128",
            "--num-attention-heads",
            "2",
            "--micro-batch-size",
            "1",
        ]
        if offload:
            argv.append("--offload-train")
        monkeypatch.setattr(sys, "argv", argv)

        if offload:
            with pytest.raises(AssertionError):
                worker._parse_args()
            return

        inputs = worker._parse_args()

        assert isinstance(inputs.trainer, TrainerConfig)
        assert isinstance(inputs.trainer.backend, MegatronArgsNamespace)
        assert inputs.trainer.trainer_role == role
        assert inputs.trainer.backend.world_size == 1
        assert inputs.trainer.backend.micro_batch_size == 1
        assert not inputs.trainer.offload_train
        assert inputs.script.role == role
        assert inputs.ckpt_load_config.load == "/model"
        assert not inputs.ckpt_load_config.resume_from_ckpt
        assert sys.argv is argv
