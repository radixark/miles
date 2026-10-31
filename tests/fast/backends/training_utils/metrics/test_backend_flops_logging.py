from argparse import Namespace

import pytest
from tests.fast.backends.training_utils.backend_owner_fixtures import native_backend_owner_state

from miles.backends.training_utils.metrics import log_utils
from miles.utils.timer import Timer

__all__ = ["native_backend_owner_state"]


class TestBackendFlopsLogging:
    def test_performance_log_turns_the_actor_fwd_flops_into_tflops_and_mfu(
        self, native_backend_owner_state: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real performance path derives train TFLOPs, MFU and throughput from the actor's forward FLOPs."""
        timer = Timer()
        monkeypatch.setattr(timer, "timers", {"actor_train": 2.0})
        monkeypatch.setattr(timer, "seq_lens", [4], raising=False)
        logs: list[dict] = []
        monkeypatch.setattr(log_utils.tracking, "log", lambda args, metrics, step_key: logs.append(metrics))
        args = Namespace(mfu_peak_tflops=100.0, trainer_model_id=None, wandb_always_use_train_step=False)
        seen: list[list[int]] = []

        def fwd_tflops(seq_lens: list[int]) -> float:
            seen.append(list(seq_lens))
            return 1e-3 * sum(seq_lens)

        log_utils.log_perf_data(rollout_id=1, args=args, compute_total_fwd_flops=fwd_tflops)

        [metrics] = logs
        assert seen == [[4]]
        assert metrics["perf/actor_train_tflops"] == pytest.approx(6e-3, rel=1e-12, abs=0)
        assert metrics["perf/actor_train_mfu"] == pytest.approx(6e-5, rel=1e-12, abs=0)
        assert metrics["perf/actor_train_tok_per_s"] == 2.0
