"""SGLang's DeepSeek-V4 attention runtime inside the trainer (forward only).

The trainer builds SGLang's own DeepSeekV4TokenToKVPool and DeepseekV4AttnBackend
so the sparse-attention indices, compressor plans and indexer metadata of a packed
micro-batch come from the serving code, as a multi-request prefill. Request ``r``
occupies page-aligned slots right after request ``r - 1``'s last page (page 0 first): DS4
compressed slots derive from ``page * 256 + position``, and page 0 is padding.
"""

from types import SimpleNamespace

import torch

PAGE = 256


def dsv4_hf_config():
    """The text config of the model the trainer pinned SGLang to."""
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.runtime_context import get_server_args

    return ModelConfig.from_server_args(get_server_args()).hf_text_config


def freqs_cis_for(cfg, ratio, device):
    """Serving's RoPE table: built at model init on the CUDA default device with BF16
    default dtype; a CPU build differs by an ulp in sin/cos."""
    from sglang.kernels.ops.attention.dsv4.rope import precompute_freqs_cis
    from sglang.srt.utils.hf_transformers_utils import get_rope_config

    rope_theta, rope_scaling = get_rope_config(cfg)
    scaling = dict(rope_scaling) if rope_scaling else {}
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device(device):
            return precompute_freqs_cis(
                dim=cfg.qk_rope_head_dim,
                seqlen=cfg.max_position_embeddings,
                original_seq_len=scaling["original_max_position_embeddings"] if ratio else 0,
                base=cfg.compress_rope_theta if ratio else rope_theta,
                factor=scaling.get("factor", 1.0),
                beta_fast=scaling.get("beta_fast", 32),
                beta_slow=scaling.get("beta_slow", 1),
            )
    finally:
        torch.set_default_dtype(previous)


class DSV4AttentionRuntime:
    def __init__(self, cfg, *, max_requests, max_tokens_per_request, max_tokens, device):
        from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.pages_per_request = (max_tokens_per_request + PAGE - 1) // PAGE
        self.max_requests = max_requests
        # Requests pack page-aligned, so each can waste under one page.
        self.capacity_pages = 1 + (max_tokens + PAGE - 1) // PAGE + max_requests
        capacity = self.capacity_pages * PAGE
        self.pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=max_requests,
            num_req_slots=max_requests + 1,
            swa_size=capacity,
            c4_size=capacity // 4,
            c128_size=capacity // 128,
            c4_state_pool_size=capacity // PAGE * 8,
            c128_state_pool_size=(max_requests + 1) * 128,
            page_size=PAGE,
            swa_page_size=PAGE,
            sliding_window=cfg.sliding_window,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=512 - cfg.qk_rope_head_dim,  # DS4 MQA head is 512 wide
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            indexer_head_dim=cfg.index_head_dim,
            layer_num=cfg.num_hidden_layers,
            device=device,
            enable_memory_saver=False,
            compression_ratios=list(cfg.compress_ratios[: cfg.num_hidden_layers]),
        )
        self.pool.register_mapping(torch.arange(capacity + 1, device=device, dtype=torch.int64))
        self.req_to_token = torch.zeros(
            (max_requests + 1, self.pages_per_request * PAGE), dtype=torch.int32, device=device
        )
        runner = SimpleNamespace(
            device=device,
            model_config=SimpleNamespace(
                context_len=self.pages_per_request * PAGE, head_dim=512, v_head_dim=512, hf_text_config=cfg
            ),
            page_size=PAGE,
            req_to_token_pool=SimpleNamespace(req_to_token=self.req_to_token),
            token_to_kv_pool=self.pool,
            hisparse_coordinator=None,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            is_draft_worker=False,
        )
        self.backend = DeepseekV4AttnBackend(runner)
        self.device = device
        self._key = None

    def prepare(self, lengths, own=None):
        """Prefill metadata for requests of ``lengths`` tokens, packed in order.

        ``own`` = (start, stop) narrows the per-query metadata to those packed rows (all-gather CP), the way
        serving's CP prefill does; cache-write locations and compressor metadata stay those of the whole stream."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        req_lens = tuple(int(n) for n in lengths)
        if (req_lens, own) == self._key:
            return self.forward_batch
        pages = [(n + PAGE - 1) // PAGE for n in req_lens]
        if (
            len(req_lens) > self.max_requests
            or max(req_lens) > self.pages_per_request * PAGE
            or 1 + sum(pages) > self.capacity_pages
        ):
            raise NotImplementedError(
                f"[top] DeepSeek-V4 trainer runtime capacity exceeded: requests of {req_lens} tokens"
            )
        dev = self.device
        self.req_to_token.zero_()
        slots, positions = [], []
        page = 1
        for r, n in enumerate(req_lens):
            base = page * PAGE
            page += pages[r]
            row = torch.arange(base, base + n, dtype=torch.int32, device=dev)
            self.req_to_token[r, :n] = row
            slots.append(row)
            positions.append(torch.arange(n, dtype=torch.int64, device=dev))
        out_cache_loc = torch.cat(slots).long()
        lens = torch.tensor(req_lens, dtype=torch.int32, device=dev)
        starts = torch.cumsum(lens, 0) - lens
        self.backend.forward_metadata = self.backend.init_forward_metadata_prefill(
            max_seq_len=max(req_lens),
            req_pool_indices=torch.arange(len(req_lens), dtype=torch.int64, device=dev),
            seq_lens=lens,
            seq_lens_cpu=list(req_lens),
            out_cache_loc=out_cache_loc,
            num_tokens=sum(req_lens),
            extend_seq_lens=lens,
            extend_seq_lens_cpu=list(req_lens),
            extend_start_loc=starts.to(torch.int32),
        )
        self.positions = torch.cat(positions)
        self.query_positions = self.positions
        self.forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            _original_forward_mode=None,
            tbo_parent_token_range=None,
            batch_size=len(req_lens),
            out_cache_loc=out_cache_loc,
            attn_cp_metadata=None,
            positions=self.positions,
        )
        if own is not None:
            self._narrow_queries(*own)
        self._key = (req_lens, own)
        return self.forward_batch

    def _narrow_queries(self, start, stop):
        """Serving's apply_cp_reindex with an explicit row range, without its global CP state."""
        meta = self.backend.forward_metadata
        core = meta.core_attn_metadata
        for name in core._CP_REINDEX_FIELDS + core._CP_REINDEX_OPTIONAL_FIELDS:
            value = getattr(
                core, name, None
            )  # config-access-exempt: attribute selected at runtime from a table of names
            if value is not None:
                setattr(core, name, value[start:stop].contiguous())
        core.init_flashmla_related(is_prefill=True)
        if meta.indexer_metadata is not None:
            meta.indexer_metadata = self.backend.init_forward_metadata_indexer(core)
        self.query_positions = self.positions[start:stop]


_RUNTIME = None
# Sizes round up to these steps so a run regrows the runtime only a few times (arbitrary).
_TOKEN_STEP = 4096
_MIN_REQUESTS = 16


def _fits(rt, lengths):
    pages = sum((n + PAGE - 1) // PAGE for n in lengths)
    return (
        len(lengths) <= rt.max_requests
        and max(lengths) <= rt.pages_per_request * PAGE
        and 1 + pages <= rt.capacity_pages
    )


def get_runtime(cfg, *, lengths, device):
    """The process-wide runtime, sized to the micro-batches seen so far and regrown when one does not fit."""
    global _RUNTIME
    if _RUNTIME is not None and _fits(_RUNTIME, lengths):
        return _RUNTIME
    step = lambda n: (n + _TOKEN_STEP - 1) // _TOKEN_STEP * _TOKEN_STEP  # noqa: E731
    old = _RUNTIME
    requests = max(_MIN_REQUESTS, 1 << (len(lengths) - 1).bit_length())
    request_tokens = step(max(lengths))
    tokens = step(sum(lengths))
    if old is not None:
        requests = max(requests, old.max_requests)
        request_tokens = max(request_tokens, old.pages_per_request * PAGE)
        tokens = max(tokens, (old.capacity_pages - 1 - old.max_requests) * PAGE)
    _RUNTIME = None  # release the old pool before building the larger one
    del old
    _RUNTIME = DSV4AttentionRuntime(
        cfg, max_requests=requests, max_tokens_per_request=request_tokens, max_tokens=tokens, device=device
    )
    return _RUNTIME
