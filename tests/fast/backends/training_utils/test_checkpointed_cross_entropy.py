import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import tensor_parallel

from miles.backends.training_utils.loss_hub import logit_processors
from miles.backends.training_utils.loss_hub.checkpointed_cross_entropy import (
    SFTCheckpointedOutputContext,
    checkpointed_sft_output_processor,
    checkpointed_vocab_parallel_cross_entropy,
    install_checkpointed_linear_cross_entropy,
)


class _OutputLayer(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden_size, dtype=torch.float64))
        self.sequence_parallel = False
        self.tp_group = None
        self.seen_sequence_lengths: list[int] = []

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        assert runtime_gather_output is False
        self.seen_sequence_lengths.append(input_.size(0))
        return F.linear(input_, self.weight if weight is None else weight), None


class _TensorParallelOutputLayer(_OutputLayer):
    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        assert runtime_gather_output is False
        assert self.sequence_parallel is False
        self.seen_sequence_lengths.append(input_.size(0))
        input_parallel = tensor_parallel.copy_to_tensor_model_parallel_region(input_, group=self.tp_group)
        return F.linear(input_parallel, self.weight if weight is None else weight), None


def test_checkpointed_cross_entropy_matches_dense_forward_and_backward() -> None:
    torch.manual_seed(7)
    layer = _OutputLayer(hidden_size=5, vocab_size=11)
    hidden = torch.randn(7, 2, 5, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4, -100, 6, 7], [7, 6, 5, 4, 3, 2, 1]])

    actual = checkpointed_vocab_parallel_cross_entropy(
        hidden,
        labels,
        output_layer=layer,
        output_weight=None,
        chunk_size=3,
        sequence_parallel_input=False,
    )
    actual.sum().backward()
    actual_hidden_grad = hidden.grad.detach().clone()
    actual_weight_grad = layer.weight.grad.detach().clone()

    dense_hidden = hidden.detach().clone().requires_grad_(True)
    dense_weight = layer.weight.detach().clone().requires_grad_(True)
    dense_logits = F.linear(dense_hidden, dense_weight).float()
    safe_labels = labels.transpose(0, 1).masked_fill(labels.transpose(0, 1) == -100, 0)
    expected = F.cross_entropy(
        dense_logits.flatten(0, 1),
        safe_labels.flatten(),
        reduction="none",
    ).view_as(safe_labels)
    expected = expected.masked_fill(labels.transpose(0, 1) == -100, 0).transpose(0, 1)
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_hidden_grad, dense_hidden.grad)
    torch.testing.assert_close(actual_weight_grad, dense_weight.grad)
    assert max(layer.seen_sequence_lengths) <= 3
    assert len(layer.seen_sequence_lengths) == 6


def test_sft_output_processor_selects_shifted_response_positions(monkeypatch) -> None:
    monkeypatch.setattr(
        logit_processors,
        "get_parallel_state",
        lambda: SimpleNamespace(cp=SimpleNamespace(size=1)),
    )
    torch.manual_seed(11)
    layer = _OutputLayer(hidden_size=4, vocab_size=19)
    hidden = torch.randn(6, 1, 4, dtype=torch.float64, requires_grad=True)
    tokens = torch.tensor([10, 11, 12, 13, 14, 15])
    args = SimpleNamespace(qkv_format="thd", allgather_cp=False, rollout_temperature=1.0)
    context = SFTCheckpointedOutputContext(
        args=args,
        batch={
            "unconcat_tokens": [tokens],
            "total_lengths": [6],
            "response_lengths": [3],
            "max_seq_lens": None,
        },
        chunk_size=2,
    )

    actual = checkpointed_sft_output_processor(
        hidden_states=hidden,
        output_layer=layer,
        output_weight=None,
        context=context,
        scale_logits=lambda logits: logits,
        runtime_gather_output=None,
    )
    expected = F.log_softmax(F.linear(hidden[2:5, 0], layer.weight).float(), dim=-1).gather(
        -1,
        tokens[3:6, None],
    )
    torch.testing.assert_close(actual, expected.squeeze(-1))
    assert layer.seen_sequence_lengths == [2, 1]


def test_install_routes_megatron_linear_cross_entropy_method() -> None:
    layer = _OutputLayer(hidden_size=3, vocab_size=5)
    layer._compute_linear_and_cross_entropy_loss = lambda: None
    model = SimpleNamespace(
        output_layer=layer,
        config=SimpleNamespace(cross_entropy_loss_fusion=False, cross_entropy_fusion_impl="native"),
        fuse_linear_cross_entropy=False,
    )

    install_checkpointed_linear_cross_entropy(model, chunk_size=4)

    assert layer._miles_cross_entropy_chunk_size == 4
    assert model.config.cross_entropy_loss_fusion is True
    assert model.config.cross_entropy_fusion_impl == "linear"
    assert model.fuse_linear_cross_entropy is True


@pytest.mark.skipif(int(os.environ.get("WORLD_SIZE", "1")) != 2, reason="requires torchrun with two GPUs")
def test_sequence_and_tensor_parallel_checkpointed_cross_entropy() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)

    torch.manual_seed(19)
    global_hidden = torch.randn(8, 1, 6, device=device, dtype=torch.bfloat16)
    global_weight = torch.randn(14, 6, device=device, dtype=torch.bfloat16)
    labels = torch.tensor([[1, 5, 8, 2, 11, 7, 3, 12]], device=device)

    local_hidden = global_hidden.chunk(2, dim=0)[rank].detach().clone().requires_grad_(True)
    local_weight = global_weight.chunk(2, dim=0)[rank].detach().clone()
    layer = _TensorParallelOutputLayer(hidden_size=6, vocab_size=7).to(device=device, dtype=torch.bfloat16)
    layer.weight = torch.nn.Parameter(local_weight)
    layer.sequence_parallel = True
    layer.tp_group = dist.group.WORLD

    actual = checkpointed_vocab_parallel_cross_entropy(
        local_hidden,
        labels,
        output_layer=layer,
        output_weight=None,
        chunk_size=3,
        sequence_parallel_input=True,
    )
    actual.sum().backward()

    dense_hidden = global_hidden.detach().clone().requires_grad_(True)
    dense_weight = global_weight.detach().clone().requires_grad_(True)
    dense_logits = F.linear(dense_hidden, dense_weight).float()
    expected = (
        F.cross_entropy(
            dense_logits.flatten(0, 1),
            labels.transpose(0, 1).flatten(),
            reduction="none",
        )
        .view_as(labels.transpose(0, 1))
        .transpose(0, 1)
    )
    expected.sum().backward()

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(local_hidden.grad, dense_hidden.grad.chunk(2, dim=0)[rank], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(layer.weight.grad, dense_weight.grad.chunk(2, dim=0)[rank], atol=2e-2, rtol=2e-2)
    assert max(layer.seen_sequence_lengths) <= 3

    if initialized_here:
        dist.destroy_process_group()
