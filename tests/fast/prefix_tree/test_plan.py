import pytest
import torch

from miles.backends.training_utils.prefix_tree.plan import build_prefix_tree
from miles.backends.training_utils.prefix_tree.sft import SFTLossLayout


def test_uneven_forks_duplicates_and_internal_endpoints():
    sequences = [[1, 2, 3, 4], [1, 2, 9], [1, 2], [1, 2, 3, 4], [8]]
    plan = build_prefix_tree(sequences)
    assert len(plan.tokens) == 6
    assert plan.expanded_tokens == 14
    for sequence, nodes in zip(sequences, plan.sample_nodes, strict=True):
        assert [plan.tokens[n] for n in nodes] == sequence
        assert [plan.positions[n] for n in nodes] == list(range(len(sequence)))
    for segment in range(len(plan.segment_parents)):
        q_start, q_end = plan.segment_offsets[segment : segment + 2]
        start, end = plan.kv_offsets[segment : segment + 2]
        path = plan.kv_indices[start:end]
        assert path[-(q_end - q_start) :] == tuple(range(q_start, q_end))
        assert [plan.positions[n] for n in path] == list(range(len(path)))


def test_adapter_and_replay_identity_prevent_unsafe_sharing():
    sequences = [[1, 2, 3]] * 3
    plan = build_prefix_tree(sequences, sharing_keys=[("adapter", 1), ("adapter", 2), ("adapter", 1)])
    assert len(plan.tokens) == 6
    plan = build_prefix_tree(sequences[:2], computation_keys=[[0, 0, 0], [0, 1, 0]])
    assert len(plan.tokens) == 5  # Matching tokens after the replay fork must not remerge.
    assert len(build_prefix_tree(sequences, share_prefixes=False).tokens) == 9


def test_targets_and_weights_remain_per_occurrence():
    plan = build_prefix_tree([[1, 2], [1, 2]])
    layout = SFTLossLayout.from_plan(plan, [[2, 3], [4, -100]], weights=[[2, 1], [3, 0]], device="cpu")
    logits = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
    expected = (
        2 * torch.nn.functional.cross_entropy(logits[:1], torch.tensor([2]))
        + torch.nn.functional.cross_entropy(logits[1:], torch.tensor([3]))
        + 3 * torch.nn.functional.cross_entropy(logits[:1], torch.tensor([4]))
    ) / 6
    torch.testing.assert_close(layout.loss(logits), expected, rtol=0, atol=0)


@pytest.mark.parametrize("sequences", [[], [[]], [[1], []]])
def test_empty_inputs_are_rejected(sequences):
    with pytest.raises(ValueError, match="nonempty"):
        build_prefix_tree(sequences)
