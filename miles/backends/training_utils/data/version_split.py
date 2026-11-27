"""Per-token sampling versions under fully async rollout.

Each response token was sampled at some weight version. Tokens sampled at the version of the proximal forward
("fresh") and earlier ones ("stale") are reported separately: the stale split carries the policy drift that
staleness adds, and the fresh split what remains (kernel and routing mismatch between trainer and engine).
"""

import torch

# Response tokens with no numeric sampling version: observations, tool output, or engines without version spans.
UNVERSIONED = -1.0


def token_versions_from_weight_versions(calls: list[dict], *, total_length: int, response_length: int) -> list[float]:
    """Sampling version of every response token, from the per-call spans in ``Sample.weight_versions``.

    Spans carry absolute positions in the sample's tokens; the response starts at total_length - response_length.
    """
    response_start = total_length - response_length
    versions = [UNVERSIONED] * response_length
    for call in calls:
        for span in call["spans"]:
            if not str(span["version"]).isdigit():
                continue
            start = max(span["abs_start"], response_start) - response_start
            end = min(span["abs_end"], total_length) - response_start
            versions[start:end] = [float(span["version"])] * max(0, end - start)
    return versions


def add_token_versions(rollout_data: dict) -> bool:
    """Adds ``rollout_token_versions`` (unsliced, one float list per sample); False if no sample carries spans."""
    calls = rollout_data.get("weight_versions")
    if not calls or not any(calls):
        return False
    rollout_data["rollout_token_versions"] = [
        token_versions_from_weight_versions(c or [], total_length=t, response_length=r)
        for c, t, r in zip(calls, rollout_data["total_lengths"], rollout_data["response_lengths"], strict=True)
    ]
    return True


def add_fresh_stale_masks(rollout_data: dict, *, proximal_version: int) -> None:
    """Float masks over each sample's (CP-local) response tokens: sampled at ``proximal_version`` or earlier."""
    fresh, stale = [], []
    for v in rollout_data["rollout_token_versions"]:
        fresh.append((v == proximal_version).float())
        stale.append(((v >= 0) & (v != proximal_version)).float())
    rollout_data["rollout_token_fresh"] = fresh
    rollout_data["rollout_token_stale"] = stale


def fresh_stale_metrics(
    *,
    abs_diff: torch.Tensor,
    fresh: torch.Tensor,
    stale: torch.Tensor,
    active_tokens: torch.Tensor,
    sum_of_sample_mean,
) -> dict[str, torch.Tensor]:
    """Fresh/stale split of the train/rollout log-prob diff; ``abs_diff`` is |trainer proximal - rollout| over the
    micro-batch's response tokens."""
    fresh = fresh * active_tokens
    stale = stale * active_tokens
    return {
        "train_rollout_logprob_abs_diff_fresh": sum_of_sample_mean(abs_diff * fresh).detach(),
        "train_rollout_logprob_abs_diff_stale": sum_of_sample_mean(abs_diff * stale).detach(),
        "fresh_token_frac": sum_of_sample_mean(fresh).detach(),
        "stale_token_frac": sum_of_sample_mean(stale).detach(),
    }
