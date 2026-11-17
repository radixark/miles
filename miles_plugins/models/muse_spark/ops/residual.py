import torch
from torch import nn


class MuseSpark1xResidualConnection(nn.Module):
    def __init__(
        self,
        dim,
        temperature=0.3,
        channel_wise=True,
        use_stream_gate=True,
        eps=1e-3,
        gated=True,
        dtype=None,
        sequence_parallel=False,
    ):
        super().__init__()
        self.temperature = temperature
        self.use_stream_gate = use_stream_gate
        self.eps = eps
        self.gate = nn.Parameter(torch.zeros(dim if channel_wise else 1, dtype=dtype)) if gated else None
        if self.gate is not None:
            self.gate.sequence_parallel = sequence_parallel

    def forward(self, stream, branch):
        if stream.numel() == 0:
            return stream
        branch = branch.to(stream.dtype)
        if self.gate is None:
            return stream + branch
        scaled_gate = self.gate.float() / self.temperature
        beta = scaled_gate.sigmoid()
        alpha = (torch.sigmoid(-scaled_gate) * (1 + beta)).clamp_min(self.eps).sqrt() if self.use_stream_gate else 1.0
        return (alpha * stream + beta * branch).to(stream.dtype)


class MuseSpark1xBiasDropoutAdd(MuseSpark1xResidualConnection):
    """Adapt the trainable residual to TransformerLayer's BDA factory contract."""

    def forward(self, training, fused):
        def apply(output_with_bias, residual, probability):
            if probability != 0:
                raise ValueError("Spark serial validation requires hidden_dropout=0")
            output, bias = output_with_bias
            if bias is not None:
                output = output + bias
            return super(MuseSpark1xBiasDropoutAdd, self).forward(residual, output)

        return apply
