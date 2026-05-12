"""
MiniMax Self-Attention with Global QK LayerNorm

核心特性：
1. QK LayerNorm 在 reshape 之前执行（全局归一化）
2. 在 TP 下使用 Gather-Norm-Scatter 策略保证计算等价性
3. 完全兼容 Megatron 的并行策略（TP/SP/CP/PP）
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType

from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import build_module
from megatron.core import mpu
from megatron.core.tensor_parallel import (
    gather_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
)

class MinimaxSelfAttention(SelfAttention):
    """
    关键差异：
    - QK LayerNorm 对整个投影向量（6144 维）做归一化，而不是每个 head（128 维）
    - 在 TP 下使用 Gather-Norm-Scatter 确保与单 GPU 计算完全等价
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        # ========== 保存原始的 QK LayerNorm spec ==========
        self._original_q_norm = submodules.q_layernorm
        self._original_k_norm = submodules.k_layernorm
        
        # ========== 临时移除，让父类正常初始化 ==========
        # 因为父类会用 hidden_size_per_attention_head 作为维度
        # 我们需要用 full hidden_size
        submodules.q_layernorm = None
        submodules.k_layernorm = None
        
        # ========== 调用父类初始化 ==========
        # 这会初始化 linear_qkv, core_attention, linear_proj 等
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
        
        if self._original_q_norm is not None:
            self.q_layernorm = build_module(
                self._original_q_norm,
                hidden_size=self.config.num_attention_heads*self.hidden_size_per_attention_head,  # ← 6144（完整维度）
                config=config,
                eps=config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None
        
        if self._original_k_norm is not None:
            # 对于 GQA，K/V 的维度可能不同
            # 但 MiniMax 的 QK Norm 应该都是在完整投影上做
            # 这里我们也用 full hidden_size
            self.k_layernorm = build_module(
                self._original_k_norm,
                hidden_size=self.config.num_query_groups*self.hidden_size_per_attention_head,  # ← 6144（完整维度）
                config=config,
                eps=config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None
        
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
    
    def get_query_key_value_tensors(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        output_gate: bool = False,
        split_qkv: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, List[int]]]:
        """
        重写：在 reshape 成多头之前应用全局 QK LayerNorm
        
        Args:
            hidden_states: [seq_len, batch, hidden_size] (TP 下可能是分片的)
        
        Returns:
            query: [seq_len, batch, num_heads_per_partition, head_dim]
            key:   [seq_len, batch, num_query_groups_per_partition, head_dim]
            value: [seq_len, batch, num_query_groups_per_partition, head_dim]
        """
        
        if output_gate:
            raise ValueError("MinimaxSelfAttention does not support output_gate.")

        # linear_qkv 是父类初始化的，已经是 Column Parallel Linear
        mixed_qkv, _ = self.linear_qkv(hidden_states)
        
        # 在 TP 模式下，mixed_qkv 的最后一维已经被切分
        # 形状：[seq_len, batch, (q_size + kv_size + kv_size) / tp_size]
        
        # 参考父类的实现
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (self.num_attention_heads_per_partition // self.num_query_groups_per_partition)
                * self.hidden_size_per_attention_head
                + 2 * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        
        # ========== 3. 拆分 Q, K, V ==========
        split_sizes = [
            (self.num_attention_heads_per_partition // self.num_query_groups_per_partition)
            * self.hidden_size_per_attention_head,  # Query
            self.hidden_size_per_attention_head,     # Key
            self.hidden_size_per_attention_head,     # Value
        ]

        if not split_qkv:
            return mixed_qkv, split_sizes

        (query, key, value) = torch.split(mixed_qkv, split_sizes, dim=-1)
        
        # 此时的形状：
        # query: [seq_len, batch, num_query_groups_per_partition, q_heads_per_group * head_dim]
        # key:   [seq_len, batch, num_query_groups_per_partition, head_dim]
        # value: [seq_len, batch, num_query_groups_per_partition, head_dim]
        
        if self.q_layernorm is not None:
            query = self._apply_global_layernorm(query, self.q_layernorm)
        
        if self.k_layernorm is not None:
            key = self._apply_global_layernorm(key, self.k_layernorm)
        
        query = query.reshape(
            query.size(0),
            query.size(1),
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
        )        
        return query, key, value
    
    def _apply_global_layernorm(
        self, 
        x: torch.Tensor, 
        layernorm: nn.Module
    ) -> torch.Tensor:
        """
        在 TP 下应用全局 LayerNorm
        
        策略：Gather → LayerNorm → Scatter
        
        Args:
            x: [seq_len, batch, num_groups, group_dim]
               在 TP 下，num_groups 和 group_dim 的乘积是分片后的
        
        Returns:
            output: 相同形状
        """
        
        original_shape = x.shape
        seq_len, batch = x.size(0), x.size(1)
        
        x_flat = x.reshape(seq_len, batch, -1)
        
        # ========== TP > 1 时：Gather（收集所有分片）==========
        if self.tp_size > 1:
            # gather_from_tensor_model_parallel_region 会在最后一维拼接
            # 输入:  [seq_len, batch, local_hidden_size]  (每个 GPU)
            # 输出:  [seq_len, batch, full_hidden_size]   (每个 GPU 都有完整数据)
            x_full = gather_from_tensor_model_parallel_region(x_flat)
        else:
            x_full = x_flat
        
        # 此时所有 GPU 的 x_full 都是完全相同的
        # x_full: [seq_len, batch, 6144]
        
        # ========== LayerNorm（全局归一化）==========
        # 因为所有 GPU 的输入相同，所以输出也相同
        x_normed_full = layernorm(x_full)
        
        # ========== TP > 1 时：Scatter（重新分配）==========
        if self.tp_size > 1:
            # scatter_to_tensor_model_parallel_region 会切分最后一维
            # 输入:  [seq_len, batch, full_hidden_size]   (每个 GPU 都有)
            # 输出:  [seq_len, batch, local_hidden_size]  (每个 GPU 只保留自己的部分)
            x_normed_local = scatter_to_tensor_model_parallel_region(x_normed_full)
        else:
            x_normed_local = x_normed_full
        
        # ========== Reshape 回原始形状 ==========
        x_normed = x_normed_local.reshape(*original_shape)
        
        return x_normed
