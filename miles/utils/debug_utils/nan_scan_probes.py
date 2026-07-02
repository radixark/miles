"""Domain-specific NaN scan probes.

Each probe is a one-line call site in main code; the knowledge of what to scan
at that boundary (key names, lazy construction, step bookkeeping) lives here.
All probes are no-ops unless MILES_NAN_SCAN=1 (see nan_scan.py).

For ad-hoc extra scan points during an investigation, call `nan_scanner.scan`
directly or inject calls via the dumper source patcher instead of adding
probes here; a probe is for a boundary worth watching permanently.
"""

import torch

from miles.utils.debug_utils.nan_scan import nan_scanner


def scan_loss_inputs(batch, logits: torch.Tensor) -> None:
    """Probe at loss_function entry: one microbatch step + all loss inputs."""
    nan_scanner.step()
    nan_scanner.scan("loss_in/logits", logits)
    for key in ("log_probs", "rollout_log_probs", "ref_log_probs", "advantages"):
        nan_scanner.scan(f"loss_in/{key}", batch.get(key))
    if batch.get("log_probs") is not None and batch.get("rollout_log_probs") is not None:
        nan_scanner.scan(
            "loss_in/old_minus_rollout_logprob",
            lambda: [
                a.float() - b.float() for a, b in zip(batch["log_probs"], batch["rollout_log_probs"], strict=False)
            ],
        )


def scan_loss_outputs(loss: torch.Tensor, log: dict) -> None:
    """Probe after the loss function: loss value + per-microbatch metrics."""
    nan_scanner.scan("loss_out/loss", loss)
    if nan_scanner.enabled():
        metrics = {k: float(v) for k, v in log.items()}
        print(f"[NAN_SCAN] step={nan_scanner.current_step} loss_out/metrics {metrics}", flush=True)


def scan_policy_loss_pre_sanitize(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    active_tokens: torch.Tensor,
    log_probs,
    logits: torch.Tensor,
) -> None:
    """Probe in policy_loss_function just before nan_to_num sanitization.

    This is the last point where non-finite ppo_kl/advantages are still visible
    (nan_to_num masks them in the loss while their NaNs still poison the CE
    backward), plus grad hooks on the two tensors backward enters through.
    """
    nan_scanner.scan("pre_sanitize/ppo_kl_active", lambda: ppo_kl.detach()[active_tokens])
    nan_scanner.scan("pre_sanitize/advantages_active", lambda: advantages.detach()[active_tokens])
    nan_scanner.scan("pre_sanitize/train_log_probs", log_probs)
    nan_scanner.scan_grad("dL_dlogits", logits)
    nan_scanner.scan_grad("dL_dlogprob", log_probs)
