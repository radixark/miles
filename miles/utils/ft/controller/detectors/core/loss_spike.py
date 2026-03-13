from pydantic import ConfigDict, Field

from miles.utils.ft.controller.detectors.base import BaseFaultDetector, DetectorContext
from miles.utils.ft.controller.detectors.checks.metrics import SpikeResult, check_metric_spike
from miles.utils.ft.controller.types import ActionType, Decision, TriggerType
from miles.utils.ft.utils.base_model import FtBaseModel


class LossSpikeDetectorConfig(FtBaseModel):
    """Paper §4.1: 5x increase in loss/gradient norms signals abnormal metrics.

    Detected spikes trigger ENTER_RECOVERY (same path as NaN loss) to
    pause training and run stop-time diagnostics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    spike_threshold: float = Field(default=5.0, gt=1.0)
    recent_steps: int = Field(default=5, ge=1)
    baseline_steps: int = Field(default=50, ge=1)
    metric_names: tuple[str, ...] = ("loss", "grad_norm")


class LossSpikeDetector(BaseFaultDetector):
    def __init__(self, config: LossSpikeDetectorConfig | None = None) -> None:
        self._config = config or LossSpikeDetectorConfig()

    def _evaluate_raw(self, ctx: DetectorContext) -> Decision:
        spikes: list[SpikeResult] = []

        for metric_name in self._config.metric_names:
            result = check_metric_spike(
                ctx.metric_store.mini_wandb,
                metric_name=metric_name,
                recent_steps=self._config.recent_steps,
                baseline_steps=self._config.baseline_steps,
                spike_threshold=self._config.spike_threshold,
            )
            if result is not None:
                spikes.append(result)

        if not spikes:
            return Decision.no_fault(reason="no metric spike detected")

        details = "; ".join(
            f"{s.metric_name} {s.ratio:.1f}x ({s.current_avg:.4f} vs baseline {s.baseline_avg:.4f})"
            for s in spikes
        )
        return Decision(
            action=ActionType.ENTER_RECOVERY,
            reason=f"metric spike: {details}",
            trigger=TriggerType.NAN_LOSS,
        )
