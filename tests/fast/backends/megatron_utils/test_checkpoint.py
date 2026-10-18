import sys
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from miles.backends.megatron_utils import checkpoint
from miles.backends.training_utils.model_companion import ModelCompanion


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
        args = Namespace(load=str(load_path), finetune=finetune, ckpt_step=ckpt_step, lora_rank=0)
        monkeypatch.setattr(checkpoint, "get_args", lambda: args)
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
        )

        assert result == (0, expected)

    def test_loading_hf_weights_preserves_non_finetune_rollout_numbering(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HF loading preserves the existing non-finetune rollout numbering."""
        (tmp_path / "config.json").write_text("{}")
        args = Namespace(load=str(tmp_path), finetune=False, ckpt_step=None, lora_rank=0)
        monkeypatch.setattr(checkpoint, "get_args", lambda: args)
        monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", lambda **_kwargs: (0, 0))

        result = checkpoint.load_checkpoint(
            ddp_model=[],
            optimizer=None,
            opt_param_scheduler=None,
            checkpointing_context=None,
            skip_load_to_model_and_opt=False,
        )

        assert result == (0, True)


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


class TestRawMbridgeNativeLoRAIsolation:
    @pytest.mark.parametrize("fail", [False, True])
    def test_base_load_preserves_nested_adapter_state(self, raw_mbridge_loader, fail):
        from miles.backends.megatron_utils.lora.native.modules.linear import NativeLoRAAdapter
        from miles.backends.megatron_utils.lora.native.spec.base import ProjectionSpec, ShardLayout

        observed = raw_mbridge_loader(fail)
        model = torch.nn.Linear(2, 2)
        model.self_attention = torch.nn.Module()
        adapter = NativeLoRAAdapter(
            "model.layers.0.self_attn", (ProjectionSpec("q_proj", "q", ShardLayout.COLUMN),), tp_rank=0
        )
        adapter.register_parameter("q_A", torch.nn.Parameter(torch.full((2, 2), 7.0)))
        adapter.register_parameter("q_B", torch.nn.Parameter(torch.zeros(2, 2)))
        adapter.register_buffer("persistent", torch.tensor(3.0))
        adapter.register_buffer("scratch", torch.tensor(4.0), persistent=False)
        model.self_attention.lora_qkv_adapter = adapter
        before = {name: value.clone() for name, value in adapter.state_dict().items()}

        with pytest.raises(RuntimeError, match="HF loading failed") if fail else nullcontext():
            checkpoint._load_hf_weights_with_mbridge([model], Namespace(offload_train=False), "/unused/hf")

        assert observed == [{"weight", "bias"}]
        assert model.self_attention.lora_qkv_adapter is adapter
        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 5))
        for name, value in before.items():
            torch.testing.assert_close(adapter.state_dict()[name], value)
            torch.testing.assert_close(model.state_dict()[f"self_attention.lora_qkv_adapter.{name}"], value)


def test_raw_hf_load_normalizes_state_before_disabling_memory_saver(raw_mbridge_loader, monkeypatch):
    """State hooks can replace persistent storage; staging-copy bypass must start afterward."""
    from megatron.core.transformer.moe.router import TopKRouter

    raw_mbridge_loader(False)
    router = TopKRouter.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.register_buffer("expert_bias", torch.zeros(4, dtype=torch.bfloat16))
    model = torch.nn.Linear(2, 2)
    model.add_module("router", router)

    @contextmanager
    def disable():
        assert router.expert_bias.dtype == torch.float32
        storage = router.expert_bias.data_ptr()
        yield
        assert router.expert_bias.data_ptr() == storage

    # the CPU runner has no torch_memory_saver; the loader imports it at call time
    monkeypatch.setitem(
        sys.modules, "torch_memory_saver", SimpleNamespace(torch_memory_saver=SimpleNamespace(disable=disable))
    )
    checkpoint._load_hf_weights_with_mbridge([model], Namespace(offload_train=True), "/unused/hf")
    torch.testing.assert_close(router.expert_bias, torch.zeros(4))
    torch.testing.assert_close(model.weight, torch.full_like(model.weight, 5))
