"""``--log-probs-backend fused`` on the torchtitan CP-local path (ContextParallelLogits) must match the
fused and torch backends on the full logits."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="stage-b-1-gpu-h200", labels=["torchtitan"], hardware=["hopper", "blackwell"])

import pytest
import torch
from tests.fast.backends.training_utils.loss.loss_test_utils import make_args, make_parallel_state

from miles.backends.training_utils.loss.hub.context_parallel_logits import ContextParallelLogits
from miles.backends.training_utils.loss.hub.logit_processors import get_log_probs_and_entropy

_VOCAB = 4096
_LENGTHS = [40, 25, 31]
_RESPONSES = [12, 9, 20]
_SEQ_LEN = 104  # packed thd batch padded past sum(_LENGTHS)


def _loss_inputs(logits, tokens, args):
    res = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=tokens,
        total_lengths=_LENGTHS,
        response_lengths=_RESPONSES,
        with_entropy=True,
    )
    log_probs, entropy = torch.cat(res["log_probs"]), torch.cat(res["entropy"])
    weights = torch.linspace(0.5, 1.5, log_probs.numel(), device=log_probs.device)
    return (weights * log_probs).sum() + 0.1 * entropy.sum(), log_probs.detach(), entropy.detach()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("recompute_loss_function", [False, True])
def test_cp_local_fused_matches_the_full_logits(nccl_world, dtype, recompute_loss_function):
    """One rank holds every row in head-tail order (cp 2 layout, cp_size 1 gather), so the CP-local path
    reorders, scores and gathers exactly the rows the full-logit path slices."""
    make_parallel_state()
    gen = torch.Generator(device="cuda").manual_seed(3)
    full = (torch.randn(1, _SEQ_LEN, _VOCAB, device="cuda", generator=gen) * 4).to(dtype)
    tokens = [torch.randint(0, _VOCAB, (n,), device="cuda", generator=gen) for n in _LENGTHS]
    chunks = torch.arange(_SEQ_LEN, device="cuda").chunk(4)
    order = torch.cat([chunks[0], chunks[3], chunks[1], chunks[2]])
    restore = order.argsort()
    targets = torch.zeros(_SEQ_LEN, dtype=torch.long, device="cuda")
    targets[: sum(_LENGTHS) - 1] = torch.cat(tokens)[1:]

    results = {}
    for name, backend in (("torch_full", "torch"), ("fused_full", "fused"), ("fused_cp_local", "fused")):
        args = make_args(
            true_on_policy_mode=False,
            log_probs_backend=backend,
            rollout_temperature=0.8,
            recompute_loss_function=recompute_loss_function,
        )
        if name == "fused_cp_local":
            leaf = full[:, order].clone().requires_grad_(True)
            logits = ContextParallelLogits(
                local=leaf * 1, targets=targets[order], gather_rows=lambda rows: rows.index_select(0, restore)
            )
        else:
            leaf = full.clone().requires_grad_(True)
            logits = leaf * 1
        loss, log_probs, entropy = _loss_inputs(logits, tokens, args)
        loss.backward()
        grad = leaf.grad[:, restore] if name == "fused_cp_local" else leaf.grad
        results[name] = (log_probs, entropy, grad.float())

    fused_lp, fused_ent, fused_grad = results["fused_full"]
    local_lp, local_ent, local_grad = results["fused_cp_local"]
    torch.testing.assert_close(local_lp, fused_lp, rtol=0, atol=0)
    torch.testing.assert_close(local_ent, fused_ent, rtol=0, atol=0)
    torch.testing.assert_close(local_grad, fused_grad, rtol=0, atol=0)

    torch_lp, torch_ent, _ = results["torch_full"]
    torch.testing.assert_close(local_lp, torch_lp, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(local_ent, torch_ent, rtol=1e-5, atol=5e-5)
