from argparse import Namespace
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed.checkpoint as dist_cp
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.megatron_utils import checkpoint
from miles.backends.training_utils.model_companion import ModelCompanion
from miles_plugins.models.deepseek_v4.arguments import DSV4_SPEC_MODULE, DSV41_SPEC_MODULE


class TestCheckpointTrainingProvenance:
    @pytest.mark.parametrize(
        ("source", "finetune", "ckpt_step", "local_iteration", "expected"),
        [
            ("0", False, None, None, True),
            ("release", False, None, None, True),
            ("0", True, None, None, False),
            ("iter_0000000", False, None, None, True),
            ("release", False, 0, None, True),
            ("release", False, None, -1, True),
            ("release", False, None, 0, True),
        ],
    )
    def test_native_load_uses_finetune_for_rollout_numbering(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        source: str,
        finetune: bool,
        ckpt_step: int | None,
        local_iteration: int | None,
        expected: bool,
    ) -> None:
        """Native loading preserves the existing finetune-based rollout numbering."""
        load_path = tmp_path / (source if source.startswith("iter_") else "checkpoint")
        load_path.mkdir()
        if source.startswith("iter_"):
            (load_path / "metadata.json").write_text("{}")
        else:
            (load_path / "latest_checkpointed_iteration.txt").write_text(source)
        args = make_trainer_args(
            load=str(load_path), finetune=finetune, ckpt_step=ckpt_step, lora_rank=0, lora_adapter_path=None, spec=None
        )
        monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **_kwargs: (0, 123))
        checkpointing_context = (
            None
            if local_iteration is None
            else {"local_checkpoint_manager": SimpleNamespace(find_latest=lambda: local_iteration)}
        )

        result = checkpoint.load_checkpoint(
            ddp_model=[],
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=False,
            args=args,
        )

        assert result == (0, expected)

    @pytest.mark.parametrize("finetune", [False, True])
    def test_loading_hf_weights_never_claims_restored_training_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finetune: bool
    ) -> None:
        """HF weights initialize a model without restoring any completed rollout."""
        (tmp_path / "config.json").write_text("{}")
        args = make_trainer_args(
            load=str(tmp_path), finetune=finetune, ckpt_step=None, lora_rank=0, lora_adapter_path=None
        )
        monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", lambda **_kwargs: (0, 0))

        result = checkpoint.load_checkpoint(
            ddp_model=[],
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=None,
            skip_load_to_model_and_opt=False,
            args=args,
        )

        assert result.iteration == 0
        assert result.restored_trained_iteration is False


@pytest.mark.parametrize(
    ("ckpt_step", "iteration", "finetune", "fails"),
    [(0, 0, False, False), (0, 7, False, True), (7, 0, True, False)],
)
def test_explicit_checkpoint_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ckpt_step: int, iteration: int, finetune: bool, fails: bool
) -> None:
    """An explicit step selects Megatron even without a tracker and cannot silently load another step."""
    (tmp_path / f"iter_{ckpt_step:07d}").mkdir()
    args = make_trainer_args(load=str(tmp_path), ckpt_step=ckpt_step, finetune=finetune, lora_rank=0, spec=None)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **_kwargs: (iteration, 0))

    with pytest.raises(ValueError, match="Requested checkpoint step") if fails else nullcontext():
        result = checkpoint.load_checkpoint([], None, None, None, False, args=args)
        assert result == (iteration, not finetune)


@pytest.mark.parametrize("layout", ["dsv4", "linear_attention"])
def test_layout_validation_uses_explicit_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str) -> None:
    """Layout validation inspects the explicitly selected checkpoint step."""
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("100")
    args = make_trainer_args(load=str(tmp_path), ckpt_step=7, finetune=False, lora_rank=0)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", lambda **_kwargs: (7, 0))
    monkeypatch.setattr(checkpoint, "is_dsv4_model", lambda _args: layout == "dsv4")
    monkeypatch.setattr(checkpoint, "_has_linear_attention", lambda _model: layout == "linear_attention")
    validate = Mock()
    monkeypatch.setattr(checkpoint, "assert_checkpoint_is_current", validate)
    monkeypatch.setattr(checkpoint, "assert_linear_attn_checkpoint_is_current", validate)

    checkpoint.load_checkpoint([], None, None, None, False, args=args)

    validate.assert_called_once_with(str(tmp_path / "iter_0000007"))


@pytest.mark.parametrize("hf_checkpoint", [False, True])
def test_exit_on_missing_checkpoint_does_not_fall_back_to_hf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hf_checkpoint: bool
) -> None:
    """Explicit missing-checkpoint handling preserves the selected native or HF source."""
    if hf_checkpoint:
        (tmp_path / "config.json").write_text("{}")
    else:
        (tmp_path / "iter_0000007").mkdir()
    args = make_trainer_args(load=str(tmp_path), finetune=False, exit_on_missing_checkpoint=True)
    load_hf = Mock(return_value=(0, 0))
    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", load_hf)

    with nullcontext() if hf_checkpoint else pytest.raises(FileNotFoundError, match="checkpoint"):
        checkpoint.load_checkpoint([], None, None, None, False, args=args)

    assert load_hf.call_count == int(hf_checkpoint)


class TestRawMbridgeCompanionIsolation:
    @pytest.mark.parametrize("fail", [False, True])
    def test_hf_loading_excludes_and_restores_companion_state(
        self, raw_mbridge_loader: Callable[[bool], list[set[str]]], fail: bool
    ) -> None:
        """HF loading must not map training metadata or discard it when loading fails."""
        observed = raw_mbridge_loader(fail)
        models = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
        companions = [ModelCompanion(pipeline_rank=0, chunk_index=i, replica_id=(0, 0, 0)) for i in range(2)]
        for model, companion in zip(models, companions, strict=True):
            companion.weight_version.fill_(7)
            companion.sample_consumptions.data = torch.tensor([[1, 2, 3, 4, 0]])
            model.add_module("model_companion", companion)

        with pytest.raises(RuntimeError, match="HF loading failed") if fail else nullcontext():
            checkpoint._load_hf_weights_with_mbridge(
                ddp_model=models, args=Namespace(offload_train=False), load_path="/unused/hf"
            )

        assert observed == [{"weight", "bias"}, {"weight", "bias"}]
        for model, companion in zip(models, companions, strict=True):
            assert model.model_companion is companion
            assert companion.weight_version.item() == 7
            assert companion.sample_consumptions.tolist() == [[1, 2, 3, 4, 0]]
            torch.testing.assert_close(model.weight, torch.full_like(model.weight, 5))


class TestDeepSeekCheckpointCompatibility:
    @pytest.mark.parametrize("spec_module", [DSV4_SPEC_MODULE, DSV41_SPEC_MODULE])
    def test_old_attention_keys_are_rejected_before_loading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spec_module: str
    ) -> None:
        """The backend model spec must reject legacy DeepSeek weights before loading partial state."""
        dist_cp.save(
            {"decoder.layers.0.self_attention.wq_a.weight": torch.zeros(2)},
            checkpoint_id=str(tmp_path / "release"),
        )
        (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
        args = make_trainer_args(load=str(tmp_path), spec=[spec_module, "get_spec"])
        monkeypatch.setattr(
            checkpoint, "_load_checkpoint_megatron", lambda **kwargs: pytest.fail("Loaded legacy weights")
        )

        with pytest.raises(ValueError, match="Re-convert it from the HuggingFace checkpoint"):
            checkpoint.load_checkpoint(
                ddp_model=[],
                optimizer=None,
                opt_param_scheduler=None,
                checkpointing_context=None,
                skip_load_to_model_and_opt=False,
                args=args,
            )
