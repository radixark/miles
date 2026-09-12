"""Exercise Miles masks through the real, optional Megatron MTP loss consumer."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils import cp_utils
from miles.backends.training_utils import data as data_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState

mtp = pytest.importorskip("megatron.core.transformer.multi_token_prediction")
packed_seq = pytest.importorskip("megatron.core.packed_seq_params")


@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("labels_provided", [False, True])
@pytest.mark.parametrize("all_masked", [False, True])
@pytest.mark.parametrize("per_token_loss", [False, True])
@pytest.mark.parametrize("qkv_format", ["thd", "bshd"])
def test_actual_mtp_loss_selects_intended_targets(
    monkeypatch: pytest.MonkeyPatch,
    num_layers: int,
    labels_provided: bool,
    all_masked: bool,
    per_token_loss: bool,
    qkv_format: str,
) -> None:
    group = GroupInfo(rank=0, size=1, group=None)
    state = ParallelState(
        intra_dp=group, intra_dp_cp=group, cp=group, tp=group, pp=group, ep=group, etp=group, indep_dp=group
    )
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(cp_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(mtp.parallel_state, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.tensor(1.0))

    token_ids = [torch.arange(1, 13), torch.arange(101, 111)]
    mask_values = [[0, 0, 1, 1, 1, 1, 1, 0, 0], [1, 0, 1, 1, 1, 0, 1]]
    masks = [torch.tensor(values) * int(not all_masked) for values in mask_values]
    rollout = {
        "tokens": token_ids,
        "loss_masks": masks,
        "total_lengths": [len(t) for t in token_ids],
        "response_lengths": [len(mask) for mask in masks],
        "max_seq_lens": [24, 24],
    }
    batch = data_utils.get_batch(
        data_utils.DataIterator(rollout, micro_batch_size=2),
        list(rollout),
        pad_multiplier=8,
        qkv_format=qkv_format,
        get_input_loss_masks=True,
    )
    targets = []
    losses = []

    def output_layer(hidden: torch.Tensor, *, labels: torch.Tensor, **kwargs: object) -> torch.Tensor:
        # Only the expensive model projection/CE is replaced. Real Megatron
        # rolling, masking, normalization and autograd injection run unchanged.
        targets.append(labels)
        loss = torch.ones_like(labels, dtype=torch.float32, requires_grad=True)
        losses.append(loss)
        return loss

    next_tokens = {0: 0}
    expected_targets = []
    for tokens, mask in zip(token_ids, masks, strict=True):
        ids = tokens.tolist()
        next_tokens.update({token_id: ids[i + 1] if i + 1 < len(ids) else 0 for i, token_id in enumerate(ids)})
        expected_targets.extend(
            token_id for token_id, selected in zip(ids[3:], mask.tolist(), strict=True) if selected
        )
    labels = torch.tensor([[next_tokens[token_id] for token_id in row] for row in batch["tokens"].tolist()])
    config = SimpleNamespace(
        mtp_num_layers=num_layers,
        mtp_detach_heads=False,
        calculate_per_token_loss=per_token_loss,
        cross_entropy_loss_fusion=True,
        cross_entropy_fusion_impl="linear",
        mtp_loss_scaling_factor=0.2,
    )
    mask = batch["full_loss_masks"] if labels_provided else batch["input_loss_masks"]
    original_mask = mask.clone()
    batch_size, seq_len = batch["tokens"].shape
    hidden = torch.zeros((num_layers + 1) * seq_len, batch_size, 1, requires_grad=True)
    packed_params = None
    if qkv_format == "thd":
        packed_params = packed_seq.PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=batch["cu_seqlens"],
            cu_seqlens_kv=batch["cu_seqlens"],
            max_seqlen_q=batch["max_seqlen"],
            max_seqlen_kv=batch["max_seqlen"],
        )
    output = mtp.process_mtp_loss(
        hidden_states=hidden,
        labels=labels if labels_provided else None,
        loss_mask=mask,
        output_layer=output_layer,
        output_weight=None,
        runtime_gather_output=False,
        is_training=False,
        compute_language_model_loss=None,
        config=config,
        cp_group=None,
        tp_group=None,
        packed_seq_params=packed_params,
        input_ids=batch["tokens"],
    )
    output.sum().backward()
    for target, loss in zip(targets, losses, strict=True):
        assert loss.grad is not None
        assert torch.isfinite(loss.grad).all()
        assert target[loss.grad != 0].tolist() == expected_targets
    torch.testing.assert_close(mask, original_mask)
