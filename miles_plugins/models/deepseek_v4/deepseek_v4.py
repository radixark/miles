import copy
import os

import einops
import torch
import torch.nn as nn

from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TELinear, TENorm, TERowParallelLinear
from megatron.core.models.gpt import experimental_attention_variant_module_specs as _eav_specs
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer, DSAIndexerSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from miles_plugins.models.deepseek_v4.ops.compressor import DeepSeekV4Compressor
from miles_plugins.models.deepseek_v4.ops.cp_utils import (
    all_gather_cp,
    get_compress_topk_idxs_cp,
    get_freqs_cis_for_cp,
    get_q_positions_for_cp,
    get_window_topk_idxs_cp,
)
from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla import sparse_attn_tilelang
from miles_plugins.models.deepseek_v4.ops.qat import fp8_simulate_qat
from miles_plugins.models.deepseek_v4.ops.rope import apply_rotary_emb, wrapped_precompute_freqs_cis
from miles_plugins.models.deepseek_v4.ops.v4_indexer import V4Indexer


def _enable_deepseek_v4_tf32():
    # Match TileKernels MHC TF32 GEMM precision for DSV4 fp32 matmuls.
    torch.backends.cuda.matmul.allow_tf32 = True


class DeepSeekV4Attention(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        submodules=None,
        layer_number: int = 1,
        attn_mask_type=None,
        attention_type: str = None,
        cp_comm_type: str = None,
        pg_collection=None,
    ):
        _enable_deepseek_v4_tf32()
        super().__init__(config=config)

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
        else:
            assert hasattr(pg_collection, "tp")
        self.pg_collection = pg_collection
        self.tp_group = self.pg_collection.tp
        self.cp_group = pg_collection.cp if hasattr(pg_collection, "cp") else None
        self.cp_size = self.cp_group.size() if self.cp_group else 1

        layer_id = layer_number - 1
        del layer_number

        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // config.tensor_model_parallel_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.dsv4_o_lora_rank
        self.head_dim = config.kv_lora_rank
        self.rope_head_dim = config.qk_pos_emb_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.n_groups = config.dsv4_o_groups
        self.n_local_groups = self.n_groups // config.tensor_model_parallel_size
        self.window_size = config.dsv4_window_size
        self.compress_ratio = config.dsv4_compress_ratios[layer_id] if config.dsv4_compress_ratios else 0
        self.eps = config.layernorm_epsilon
        self.use_fp8_qat = config.fp8 is not None

        assert self.o_lora_rank == 1024
        assert self.head_dim == 512
        assert self.rope_head_dim == 64
        assert self.nope_head_dim == 448
        assert self.window_size == 128

        config_no_sp = copy.copy(config)
        config_no_sp.sequence_parallel = False

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.attn_sink._keep_fp32 = True
        self.attn_sink.tensor_model_parallel = True
        self.attn_sink.partition_dim = 0
        self.attn_sink.partition_stride = 1

        self.wq_a = TELinear(
            self.dim,
            self.q_lora_rank,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.q_norm = TENorm(config_no_sp, self.q_lora_rank, eps=self.eps)
        self.wq_b = TEColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            config=config_no_sp,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )
        self.wkv = TELinear(
            self.dim,
            self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.kv_norm = TENorm(config_no_sp, self.head_dim, eps=self.eps)

        for p in list(self.wq_a.parameters()) + list(self.wkv.parameters()):
            p.sequence_parallel = False

        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            config=config_no_sp,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
        )
        assert self.wo_a.weight.dtype == torch.bfloat16
        self.wo_b = TERowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.dim,
            config=config_no_sp,
            init_method=config.init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
            tp_group=self.tp_group,
        )
        self.softmax_scale = self.head_dim**-0.5
        self.sequence_parallel = config.sequence_parallel

        if self.compress_ratio:
            self.compressor = DeepSeekV4Compressor(
                config=config,
                head_dim=self.head_dim,
                compress_ratio=self.compress_ratio,
                rotate=False,
                cp_group=self.cp_group,
            )
            if self.compress_ratio == 4:
                if os.environ.get("V4_INDEXER_IMPL", "tilelang") == "tilelang":
                    self.indexer = V4Indexer(config=config, pg_collection=pg_collection)
                else:
                    indexer_submodules = DSAIndexerSubmodules(
                        linear_wq_b=TELinear,
                        linear_wk=TELinear,
                        k_norm=TENorm,
                        linear_weights_proj=TELinear,
                    )
                    self.indexer = DSAIndexer(config=config, submodules=indexer_submodules)
            else:
                self.indexer = None

        rope_base = config.dsv4_compress_rope_theta if self.compress_ratio else config.rotary_base
        yarn_disabled = not self.compress_ratio
        freqs_cis = wrapped_precompute_freqs_cis(
            config, rope_head_dim=self.rope_head_dim, base=rope_base, yarn_disabled=yarn_disabled
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ) -> ShardedStateDict:
        ans = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        ans.update(
            make_sharded_tensors_for_checkpoint(
                state_dict={"attn_sink": self.attn_sink},
                prefix=prefix,
                tensor_parallel_layers_axis_map={"attn_sink": 0},
                sharded_offsets=sharded_offsets,
                tp_group=self.tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        )
        return ans

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
    ) -> torch.Tensor:
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
            )

        x = einops.rearrange(hidden_states, "s b d -> b s d")

        bsz, seqlen_local, _ = x.size()
        freqs_cis = get_freqs_cis_for_cp(self.freqs_cis, seqlen_local, self.cp_size, self.cp_group)
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        q_after_wq_a = self.wq_a(x)[0]
        qr = q = self.q_norm(q_after_wq_a)
        q_after_wq_b = self.wq_b(q)[0]
        q = q_after_wq_b.unflatten(-1, (self.n_local_heads, self.head_dim))
        q_fp32 = q.float()
        q = (q_fp32 * torch.rsqrt(q_fp32.square().mean(-1, keepdim=True) + self.eps)).to(q.dtype)
        q = q.clone()
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv_after_wkv = self.wkv(x)[0]
        kv_vanilla = self.kv_norm(kv_after_wkv)
        kv_vanilla = kv_vanilla.clone()
        apply_rotary_emb(kv_vanilla[..., -rd:], freqs_cis)
        if self.use_fp8_qat:
            kv_vanilla = kv_vanilla.clone()
            kv_vanilla[..., : self.nope_head_dim] = fp8_simulate_qat(kv_vanilla[..., : self.nope_head_dim], 64)

        seqlen_global = seqlen_local * self.cp_size
        q_positions = get_q_positions_for_cp(
            seqlen_local, cp_size=self.cp_size, cp_group=self.cp_group, device=x.device
        )

        topk_idxs = get_window_topk_idxs_cp(q_positions, window_size=win, cp_size=self.cp_size, bsz=bsz)

        if self.compress_ratio:
            kv_compress_offset = seqlen_global
            if self.indexer is not None:
                x_sbd = einops.rearrange(x, "b s d -> s b d")
                qr_sbd = einops.rearrange(qr, "b s d -> s b d")
                if self.sequence_parallel:
                    x_sbd = scatter_to_sequence_parallel_region(x_sbd, group=self.tp_group)
                    qr_sbd = scatter_to_sequence_parallel_region(qr_sbd, group=self.tp_group)
                if isinstance(self.indexer, V4Indexer):
                    compress_topk_idxs = self.indexer(x_sbd, qr_sbd)
                else:
                    indexer_mask = self._compute_indexer_mask(q_positions=q_positions, seqlen_global=seqlen_global)
                    compress_topk_idxs = self.indexer(x_sbd, qr_sbd, mask=indexer_mask, packed_seq_params=None)
                q_first_invalid_group = (q_positions + 1).unsqueeze(1) // ratio
                topk_idx_mask = (compress_topk_idxs >= q_first_invalid_group) | (compress_topk_idxs < 0)
                compress_topk_idxs = torch.where(topk_idx_mask, -1, compress_topk_idxs + kv_compress_offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs_cp(q_positions, ratio=ratio, cp_size=self.cp_size, bsz=bsz)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        kv_compress = None
        if self.compress_ratio:
            x_sbd = einops.rearrange(x, "b s d -> s b d")
            kv_compress_sbd = self.compressor(x_sbd)
            if kv_compress_sbd is not None:
                kv_compress = einops.rearrange(kv_compress_sbd, "s b d -> b s d")

        assert self.attn_sink.dtype == torch.float32

        if self.cp_size > 1:
            kv_vanilla = all_gather_cp(kv_vanilla, dim=1, cp_group=self.cp_group)
            if kv_compress is not None:
                kv_compress = all_gather_cp(kv_compress, dim=1, cp_group=self.cp_group)

        if kv_compress is not None:
            kv = torch.cat([kv_vanilla, kv_compress], dim=1)
            assert kv_compress_offset == kv_vanilla.size(1)
        else:
            kv = kv_vanilla

        kv = copy_to_tensor_model_parallel_region(kv, group=self.tp_group, all_reduce_grad_fp32=True)

        o = sparse_attn_tilelang(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)

        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        o = o.view(bsz, seqlen_local, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x, _ = self.wo_b(o.flatten(2))

        output = einops.rearrange(x, "b s d -> s b d")

        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)

        return output

    def _compute_indexer_mask(self, *, q_positions: torch.Tensor, seqlen_global: int) -> torch.Tensor:
        """Dense causal mask for legacy DSAIndexer path."""
        ratio = 4
        device = q_positions.device
        k_group_idx = torch.arange(seqlen_global // ratio, device=device).unsqueeze(0)
        q_first_invalid_group = (q_positions.unsqueeze(1) + 1) // ratio
        invalid_mask = k_group_idx >= q_first_invalid_group
        return torch.where(invalid_mask, float("-inf"), 0.0)


def _dsv4_attention_module_spec(config, backend=None):
    return ModuleSpec(
        module=DeepSeekV4Attention,
        submodules=None,
        metainfo={"fuse_input_layernorm": False},
    )


def get_dsv4_spec(args, config, vp_stage):
    """
    Usage: --spec miles_plugins.models.deepseek_v4.deepseek_v4 get_dsv4_spec
    """
    _orig_get_spec = _eav_specs.get_experimental_attention_variant_module_spec

    def _patched_get_spec(config, backend=None):
        if config.experimental_attention_variant == "dsv4":
            return _dsv4_attention_module_spec(config, backend)
        return _orig_get_spec(config, backend)

    _eav_specs.get_experimental_attention_variant_module_spec = _patched_get_spec
    try:
        return get_transformer_block_with_experimental_attention_variant_spec(config, vp_stage=vp_stage)
    finally:
        _eav_specs.get_experimental_attention_variant_module_spec = _orig_get_spec
