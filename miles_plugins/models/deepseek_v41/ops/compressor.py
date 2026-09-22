import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.transformer.module import mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.deepseek_v41.ops.norm import RMSNorm


class DeepSeekV41Compressor(nn.Module):
    def __init__(self, config: TransformerConfig, head_dim: int, compress_ratio: int):
        super().__init__()
        assert compress_ratio in (1, 2)
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        dim = config.hidden_size
        self.linear_wkv = nn.Linear(dim, head_dim, bias=False, dtype=torch.bfloat16)
        if compress_ratio > 1:
            self.linear_wgate = nn.Linear(dim, head_dim, bias=False, dtype=torch.bfloat16)
        self.norm = RMSNorm(head_dim, config.layernorm_epsilon)
        mark_keep_in_fp32(self.norm.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seqlen, bsz, _ = x.shape
        ratio = self.compress_ratio
        if ratio == 1:
            return self.norm(F.linear(x, self.linear_wkv.weight))
        groups = seqlen // ratio
        x = x[: groups * ratio].float()
        kv = F.linear(x, self.linear_wkv.weight.float())
        score = F.linear(x, self.linear_wgate.weight.float())
        kv = kv.view(groups, ratio, bsz, self.head_dim)
        score = score.view(groups, ratio, bsz, self.head_dim)
        kv = (kv * score.softmax(dim=1)).sum(dim=1)
        return self.norm(kv.to(torch.bfloat16))
