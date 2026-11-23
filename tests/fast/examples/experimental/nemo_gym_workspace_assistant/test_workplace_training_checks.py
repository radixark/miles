import logging
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.checkpoint_request import CHECKPOINT_LOAD_FIELDS, MegatronCkptLoadConfig
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.component_trainer import TrainerOnlyConfig
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.custom_view import compute_custom_function_config

_MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[5]
        / "examples/experimental/nemo-gym-workspace-assistant/workplace_training_checks.py"
    )
)


class TestWorkplaceTrainingChecks:
    @pytest.mark.parametrize("field", ["finetune", "no_load_optim", "no_load_rng", "load"])
    def test_invalid_fresh_request_fails_without_leaking_load_fields(self, field: str) -> None:
        """Each fresh-start invariant rejects bad requests without changing the trainer backend."""
        backend = MegatronArgsNamespace()
        args = TrainerOnlyConfig(backend=backend, trainer_role="actor", trainer_actor_index=0)
        request = MegatronCkptLoadConfig(
            load="hf", resume_from_ckpt=False, finetune=True, no_load_optim=True, no_load_rng=True, ckpt_step=None
        ).model_copy(update={field: "other" if field == "load" else False})
        hook = CustomFunctionConfig(path="workplace_training_checks.initialize_training_state")
        view = compute_custom_function_config(
            args, hook, request.model_dump(), {"hf_checkpoint": "hf", "start_rollout_id": None}
        )

        with pytest.raises(AssertionError, match="Expected"):
            _MODULE["initialize_training_state"](view)

        assert CHECKPOINT_LOAD_FIELDS.isdisjoint(vars(backend))
        assert "resume_from_ckpt" not in vars(backend)

    @pytest.mark.parametrize("start_rollout_id", [None, 1])
    def test_recovery_does_not_claim_a_fresh_start(self, start_rollout_id: int | None) -> None:
        """Recovery skips fresh-only checks unless the caller explicitly restarts at rollout zero."""
        args = SimpleNamespace(
            start_rollout_id=start_rollout_id, **MegatronCkptLoadConfig.for_peer_recovery().model_dump()
        )
        _MODULE["initialize_training_state"](args)

    def test_explicit_rollout_zero_rejects_recovery_state(self) -> None:
        """Resetting the rollout cursor cannot conceal restored optimizer or RNG state."""
        args = SimpleNamespace(start_rollout_id=0, **MegatronCkptLoadConfig.for_peer_recovery().model_dump())
        with pytest.raises(AssertionError, match="fresh training state"):
            _MODULE["initialize_training_state"](args)

    def test_fresh_request_and_model_validation_emit_separate_evidence(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The first update validates MTP without reading expired checkpoint request fields."""
        monkeypatch.setattr(_MODULE["dist"], "get_rank", lambda: 0)
        args = SimpleNamespace(
            start_rollout_id=None,
            hf_checkpoint="hf",
            resume_from_ckpt=False,
            finetune=True,
            no_load_optim=True,
            no_load_rng=True,
            load="hf",
        )
        with caplog.at_level(logging.INFO):
            _MODULE["initialize_training_state"](args)
            assert not any("WORKPLACE_MTP_DISABLED" in record.message for record in caplog.records)
            _MODULE["before_train_step"](
                SimpleNamespace(start_rollout_id=None, enable_mtp_training=False), 0, 0, [], None, None
            )
        assert any("WORKPLACE_FRESH_HF_REQUEST" in record.message for record in caplog.records)
        assert any("WORKPLACE_MTP_DISABLED" in record.message for record in caplog.records)
