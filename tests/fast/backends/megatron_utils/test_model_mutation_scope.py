import pytest
from tests.fast.backends.megatron_utils.backend_scope_fixtures import _model_loader, _parsed_trainer

from miles.backends.megatron_utils import model


class TestModelConstructionMutationScope:
    @pytest.mark.parametrize("fails", [False, True])
    def test_model_sdk_can_publish_model_type_without_leaving_backend_mutable(
        self, monkeypatch: pytest.MonkeyPatch, fails: bool
    ) -> None:
        """Model construction permits its SDK metadata update and revokes permission on every exit."""
        args = _parsed_trainer("--megatron-to-hf-mode", "raw", "--debug-disable-optimizer")
        monkeypatch.setattr(model, "get_model", _model_loader(backend=args.backend, fails=fails))
        monkeypatch.setattr(model, "is_first_replica_megatron_main_rank", lambda: False)

        if fails:
            with pytest.raises(RuntimeError, match="model construction failed"):
                model.setup_model_and_optimizer(args=args)
        else:
            chunks, optimizer, scheduler = model.setup_model_and_optimizer(args=args)
            assert len(chunks) == 1
            assert optimizer is scheduler is None
            assert chunks[0].lora_attached is False

        assert args.backend.model_type is model.ModelType.encoder_or_decoder
        with pytest.raises(AttributeError, match="immutable"):
            args.backend.model_type = None
