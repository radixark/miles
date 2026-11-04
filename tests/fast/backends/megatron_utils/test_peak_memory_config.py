import pytest
from tests.fast.fixtures.args_fixtures import make_trainer_args

from miles.backends.megatron_utils import ci_utils


class TestPeakMemoryConfig:
    @pytest.mark.parametrize(
        "ci,low_memory,checkpoint,peak_gb,checked,rejected",
        [
            (False, True, "Qwen3-4B", 100.0, False, False),
            (True, False, "Qwen3-4B", 100.0, False, False),
            (True, True, None, 100.0, False, False),
            (True, True, "another-model", 100.0, False, False),
            (True, True, "Qwen3-4B", 19.99, True, False),
            (True, True, "Qwen3-4B", 20.0, True, True),
        ],
    )
    def test_only_requested_qwen_memory_checks_touch_cuda_and_enforce_the_limit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ci: bool,
        low_memory: bool,
        checkpoint: str | None,
        peak_gb: float,
        checked: bool,
        rejected: bool,
    ) -> None:
        """CI memory checks honor both switches and the model identity before touching CUDA."""
        calls: list[bool] = []

        def allocated() -> float:
            calls.append(True)
            return peak_gb * 1024**3

        monkeypatch.setattr(ci_utils.torch.cuda, "max_memory_allocated", allocated)
        monkeypatch.setattr(ci_utils.torch.distributed, "is_initialized", lambda: False)
        args = make_trainer_args(ci_test=ci, low_memory_resume=low_memory, hf_checkpoint=checkpoint)

        if rejected:
            with pytest.raises(AssertionError, match="exceeds threshold"):
                ci_utils.check_peak_gpu_memory_after_load(args)
        else:
            ci_utils.check_peak_gpu_memory_after_load(args)

        assert len(calls) == int(checked)
