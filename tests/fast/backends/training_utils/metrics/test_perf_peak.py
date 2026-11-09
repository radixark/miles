from argparse import Namespace

import pytest

from miles.backends.training_utils.metrics import perf
from miles.utils.timer import Timer


class TestPerformancePeak:
    def test_quantization_time_uses_existing_perf_namespace_and_is_drained(self, monkeypatch):
        instance = Timer()
        monkeypatch.setattr(instance, "timers", {"update_weights_quantization": 0.75})
        calls = []
        monkeypatch.setattr(perf.tracking, "log", lambda args, payload, step_key: calls.append((payload, step_key)))
        args = Namespace(trainer_model_id="actor", wandb_always_use_train_step=False)

        perf.log_perf_data_raw(3, args, is_primary_rank=True, compute_total_fwd_flops=None)

        [(metrics, step_key)] = calls
        assert metrics == {"actor/perf/update_weights_quantization_time": 0.75, "actor/rollout/step": 3}
        assert step_key == "actor/rollout/step"
        assert instance.log_dict() == {}

    @pytest.mark.parametrize("configured_peak,detected_peak", [(200.0, 100.0), (None, 100.0), (None, None)])
    def test_mfu_uses_the_configured_peak_or_available_device_peak(
        self,
        monkeypatch: pytest.MonkeyPatch,
        configured_peak: float | None,
        detected_peak: float | None,
    ) -> None:
        """MFU uses an explicit peak before device detection and omits unavailable peaks."""
        timer = Timer()
        monkeypatch.setattr(timer, "timers", {"actor_train": 2.0})
        monkeypatch.setattr(timer, "seq_lens", [4, 6], raising=False)
        calls: list[dict[str, float]] = []
        monkeypatch.setattr(perf, "local_peak_bf16_tflops", lambda: detected_peak)
        monkeypatch.setattr(perf.tracking, "log", lambda args, payload, step_key: calls.append(payload))
        args = Namespace(mfu_peak_tflops=configured_peak, trainer_model_id=None, wandb_always_use_train_step=False)

        perf.log_perf_data_raw(
            rollout_id=3,
            args=args,
            is_primary_rank=True,
            compute_total_fwd_flops=lambda seq_lens: float(sum(seq_lens) * 10),
        )

        [metrics] = calls
        assert metrics["perf/actor_train_tflops"] == 150.0
        assert metrics["perf/actor_train_tok_per_s"] == 5.0
        if configured_peak is not None:
            assert metrics["perf/mfu_peak_tflops"] == 200.0
            assert metrics["perf/actor_train_mfu"] == 0.75
        elif detected_peak is not None:
            assert metrics["perf/mfu_peak_tflops"] == 100.0
            assert metrics["perf/actor_train_mfu"] == 1.5
        else:
            assert "perf/mfu_peak_tflops" not in metrics
            assert "perf/actor_train_mfu" not in metrics
