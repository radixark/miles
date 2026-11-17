"""Trainer-side R3 of an inline run against an output store run of the same deterministic rollouts.

Each run dumps train data to ``{dump_dir}/{mode}/{rollout_id}_{rank}.pt`` (``--save-debug-train-data``), which keeps
rollout_routed_experts as routing replay consumed it on that rank: each sample's rows at the rank's CP positions,
padded, before packing. The output store run reads only those rows, so the two runs are compared per CP rank.
"""

from collections import Counter
from pathlib import Path

import torch


def save_debug_train_data_arg(dump_dir: Path, mode: str) -> str:
    return f"--save-debug-train-data {dump_dir}/{mode}/{{rollout_id}}_{{rank}}.pt "


def trainer_r3(dump_dir: Path, mode: str, *, cp_size: int) -> dict[tuple[int, tuple[int, ...], int], torch.Tensor]:
    """(rollout id, tokens, CP rank) -> R3 over every rank's dumps of one run."""
    r3 = {}
    for path in sorted((dump_dir / mode).glob("*.pt")):
        dump = torch.load(path, map_location="cpu", weights_only=False)
        rollout_data = dump["rollout_data"]
        for tokens, routed in zip(rollout_data["tokens"], rollout_data["rollout_routed_experts"], strict=True):
            key = (dump["rollout_id"], tuple(tokens.tolist()), dump["cp_rank"])
            if key in r3:
                assert torch.equal(
                    r3[key], routed
                ), f"{mode}: ranks of CP rank {key[2]} hold different R3 for a rollout {key[0]} sample"
            else:
                r3[key] = routed
    ranks = Counter((rollout_id, tokens) for rollout_id, tokens, _ in r3)
    assert set(ranks.values()) == {cp_size}, f"{mode}: expected each sample on {cp_size} CP ranks, got {ranks}"
    return r3


def assert_same_trainer_r3(dump_dir: Path, *, cp_size: int, num_samples: int) -> None:
    inline, store = trainer_r3(dump_dir, "inline", cp_size=cp_size), trainer_r3(dump_dir, "store", cp_size=cp_size)
    samples = {(rollout_id, tokens) for rollout_id, tokens, _ in inline}
    assert len(samples) == num_samples, f"expected {num_samples} distinct samples, got {len(samples)}"
    assert inline.keys() == store.keys(), "the two runs generated different tokens; R3 cannot be compared"
    for (rollout_id, tokens, cp_rank), expected in inline.items():
        actual = store[(rollout_id, tokens, cp_rank)]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, (actual.dtype, actual.shape)
        assert torch.equal(
            actual, expected
        ), f"R3 differs on CP rank {cp_rank} for a {len(tokens)}-token sample of rollout {rollout_id}"
