"""Full-response Tinker contracts survive zigzag partitioning and loss recomputation."""

from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.cp_utils import slice_log_prob_with_cp
from miles.backends.training_utils.loss import loss_function
from miles.backends.training_utils.loss_hub.tinker_losses import TINKER_LOSS_FUNCTIONS
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state


def _zigzag_rows(values, rank, size):
    # Independent construction of the model's two chunks, including per-sequence padding.
    width = (len(values) + 2 * size - 1) // (2 * size)
    padded = F.pad(values, (0, 0, 0, 2 * size * width - len(values)))
    return torch.cat(
        [padded[rank * width : (rank + 1) * width], padded[-(rank + 1) * width : len(padded) - rank * width]]
    )


def _exercise_partition(rank, size, port, *, recompute):
    init_gloo(rank, size, port=port)
    tp_group = [dist.new_group([member]) for member in range(size)][rank]
    generator = torch.Generator().manual_seed(73)
    totals, responses = [13, 8, 3, 32], [11, 3, 1, 5]
    masks = [torch.ones(length) for length in responses]
    masks[0][1::3] = 0
    masks[1].zero_()  # A DP-padding datum must return scores but contribute no loss or gradient.
    batch = {
        "unconcat_tokens": [torch.randint(0, 64, (length,), generator=generator) for length in totals],
        "target_tokens": [torch.randint(0, 64, (length,), generator=generator).tolist() for length in responses],
        "loss_weights": [torch.randn(length, generator=generator).tolist() for length in responses],
        "advantages": [torch.randn(length, generator=generator).tolist() for length in responses],
        "rollout_log_probs": [torch.linspace(-5.5, -3.5, length) for length in responses],
        "loss_masks": masks,
        "sample_indices": [7, 2, 19, 4],
        "total_lengths": totals,
        "response_lengths": responses,
    }
    args = make_args(true_on_policy_mode=False, multi_lora=True, recompute_loss_function=recompute)
    values = torch.randn(sum(totals), 64, generator=generator)
    try:
        for name in TINKER_LOSS_FUNCTIONS:
            batch["loss_fn"] = name
            single = GroupInfo(rank=0, size=1, group=None)
            state = ParallelState(
                intra_dp=single,
                intra_dp_cp=single,
                cp=single,
                tp=GroupInfo(rank=0, size=1, group=tp_group),
                pp=single,
                ep=single,
                etp=single,
                indep_dp=single,
            )
            set_parallel_state(state)
            reference = values.unsqueeze(0).clone().requires_grad_()
            expected_loss, _, expected_report = loss_function(
                args, batch, 1, reference, apply_megatron_loss_scaling=True
            )
            expected_loss.backward()

            state.cp = GroupInfo(rank=rank, size=size, group=dist.group.WORLD)
            state.intra_dp_cp = state.cp
            local_batch = {
                **batch,
                "rollout_log_probs": [
                    slice_log_prob_with_cp(prob, total, response)
                    for prob, total, response in zip(batch["rollout_log_probs"], totals, responses, strict=True)
                ],
            }
            local = torch.cat([_zigzag_rows(part, rank, size) for part in values.split(totals)]).unsqueeze(0)
            local.requires_grad_()
            actual_loss, _, actual_report = loss_function(
                args, local_batch, 1, local, apply_megatron_loss_scaling=True
            )
            actual_loss.backward()
            # The dispatcher scales by CP size to compensate Megatron's gradient averaging.
            full_loss = actual_loss.detach() / size
            dist.all_reduce(full_loss)
            torch.testing.assert_close(full_loss, expected_loss.detach(), rtol=2e-6, atol=2e-6)
            expected_grad = torch.cat(
                [_zigzag_rows(part, rank, size) for part in reference.grad.squeeze(0).split(totals)]
            ).unsqueeze(0)
            torch.testing.assert_close(local.grad / size, expected_grad, rtol=2e-6, atol=2e-6)
            for actual, expected in zip(actual_report["per_datum"], expected_report["per_datum"], strict=True):
                assert actual["sample_index"] == expected["sample_index"]
                torch.testing.assert_close(actual["logprobs"], expected["logprobs"], rtol=0, atol=0)
                torch.testing.assert_close(actual["loss"], expected["loss"], rtol=2e-6, atol=2e-6)
                assert not actual["logprobs"].requires_grad and not actual["loss"].requires_grad
            # Reusing a microbatch must not slice its shared fields a second time.
            assert [len(mask) for mask in local_batch["loss_masks"]] == responses
            assert [len(weight) for weight in local_batch["loss_weights"]] == responses
            assert [len(advantage) for advantage in local_batch["advantages"]] == responses
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("recompute", [False, True])
def test_tinker_responses_and_gradients_match_unsharded(cp_size, recompute):
    run_multiprocess(partial(_exercise_partition, recompute=recompute), world_size=cp_size)
