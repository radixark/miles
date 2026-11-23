from types import SimpleNamespace

import pytest
import torch
import torch.distributed.fsdp as fsdp

from miles.backends.fsdp_utils.actor import apply_fsdp2


class TestExplicitShardingConfig:
    @pytest.mark.parametrize("fp16,expected", [(False, torch.bfloat16), (True, torch.float16)])
    def test_supplied_config_controls_the_policy_received_by_the_sharding_sdk(
        self, monkeypatch: pytest.MonkeyPatch, fp16: bool, expected: torch.dtype
    ) -> None:
        """The supplied backend precision reaches FSDP for the root and every selected layer."""
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        model._no_split_modules = ["Linear"]
        model.config = SimpleNamespace(tie_word_embeddings=True)
        calls: list[tuple[torch.nn.Module, dict]] = []
        monkeypatch.setattr(fsdp, "fully_shard", lambda module, **kwargs: calls.append((module, kwargs)))
        args = SimpleNamespace(backend=SimpleNamespace(fp16=fp16))

        result = apply_fsdp2(model, args=args)

        assert result is model
        assert [module for module, _ in calls] == [model[0], model]
        assert all(kwargs["mp_policy"].param_dtype == expected for _, kwargs in calls)
        assert all(kwargs["mp_policy"].reduce_dtype == torch.float32 for _, kwargs in calls)

    def test_omitted_config_rejects_before_any_sharding_side_effect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing trainer config cannot partially shard a model."""
        calls: list[object] = []
        monkeypatch.setattr(fsdp, "fully_shard", lambda *args, **kwargs: calls.append(args))

        with pytest.raises(TypeError, match="args"):
            apply_fsdp2(torch.nn.Linear(2, 2))

        assert calls == []
