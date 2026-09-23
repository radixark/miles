from types import SimpleNamespace

import torch
import torch.nn.functional as F

from miles.backends.megatron_utils.compact_logits import compact_logits_output_processor


class _TensorParallelGroup:
    @staticmethod
    def size() -> int:
        return 1


class _RecordingOutputLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(21, dtype=torch.float32).reshape(7, 3))
        self.sequence_parallel = False
        self.tp_group = _TensorParallelGroup()
        self.projected_hidden_states = None

    def forward(self, input_, weight=None, runtime_gather_output=None):
        self.projected_hidden_states = input_.detach().clone()
        projection_weight = self.weight if weight is None else weight
        return F.linear(input_, projection_weight), None


def test_compact_logits_projects_only_selected_hidden_states():
    hidden_states = torch.arange(15, dtype=torch.float32).reshape(5, 1, 3)
    loss_mask = torch.tensor([[0, 1, 0, 1, 1]])
    output_layer = _RecordingOutputLayer()

    logits = compact_logits_output_processor(
        hidden_states=hidden_states,
        output_layer=output_layer,
        output_weight=None,
        labels=None,
        loss_mask=loss_mask,
        inference_context=None,
        runtime_gather_output=False,
        scale_logits=lambda value: value,
        config=SimpleNamespace(mtp_num_layers=None),
    )

    assert torch.equal(output_layer.projected_hidden_states, hidden_states[[1, 3, 4]])
    assert logits.shape == (1, 3, 7)
