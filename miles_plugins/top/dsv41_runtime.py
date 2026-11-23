"""SGLang's DeepSeek-V4.1 attention runtime inside the trainer (forward only), and the block's per-forward state.

As for DeepSeek-V4 (dsv4_runtime), the trainer builds SGLang's own DeepSeekV4TokenToKVPool and DeepseekV4AttnBackend,
here with V4.1's KV-source layers and ratio-1/2 caches, so a packed micro-batch runs serving's attention program as one
multi-request prefill: compressed latents written by a KV-source layer are read by the later layers of its ratio, an
index layer's selection is read by the layers after it, and the candidate source's mask by the later index layers,
exactly as in a serving forward. The pool layouts are the ones serving selects on this GPU (select_dsv4_kv_layout).
"""

from types import SimpleNamespace

import torch

from miles_plugins.top.dsv4_runtime import _MIN_REQUESTS, _TOKEN_STEP, PAGE, DSV4AttentionRuntime


class DSV41AttentionRuntime(DSV4AttentionRuntime):
    def __init__(self, cfg, *, max_requests, max_tokens_per_request, max_tokens, device):
        from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool, select_dsv4_kv_layout
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        self.pages_per_request = (max_tokens_per_request + PAGE - 1) // PAGE
        self.max_requests = max_requests
        self.capacity_pages = 1 + (max_tokens + PAGE - 1) // PAGE + max_requests
        capacity = self.capacity_pages * PAGE
        kv_layout, compressed_kv_layout = select_dsv4_kv_layout()
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
            qk_nope_head_dim=512 - cfg.qk_rope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            indexer_head_dim=cfg.index_head_dim,
            layer_num=cfg.num_hidden_layers,
            device=device,
            enable_memory_saver=False,
            compression_ratios=list(cfg.compress_ratios[: cfg.num_hidden_layers]),
            kv_source_layers=list(cfg.kv_source_layer_ids),
            full_size=capacity,
            kv_layout=kv_layout,
            compressed_kv_layout=compressed_kv_layout,
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
        if own is not None:
            raise NotImplementedError("[top] DeepSeek-V4.1 attention has no all-gather CP path")
        fb = super().prepare(lengths)
        req_lens = [int(n) for n in lengths]
        lens = torch.tensor(req_lens, dtype=torch.int32, device=self.device)
        # What the V4.1 low-ratio path reads besides DS4's fields (token_req_indices, the indexer's width).
        fb.req_pool_indices = torch.arange(len(req_lens), dtype=torch.int64, device=self.device)
        fb.extend_seq_lens = lens
        fb.extend_seq_lens_cpu = req_lens
        fb.seq_lens_cpu = req_lens
        fb.encoder_swa_replay = False
        return fb


_RUNTIME = None


def get_runtime(cfg, *, lengths, device):
    """The process-wide V4.1 attention runtime, regrown when a micro-batch does not fit (as dsv4_runtime's)."""
    from miles_plugins.top.dsv4_runtime import _fits

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
    _RUNTIME = None
    del old
    _RUNTIME = DSV41AttentionRuntime(
        cfg, max_requests=requests, max_tokens_per_request=request_tokens, max_tokens=tokens, device=device
    )
    return _RUNTIME


class BlockState:
    """What V4.1's layers hand down one forward besides the streams: the previous hyper-connection's ``pre``, and the
    packed token ids and request lengths the engram layers hash. Reset by the model's forward pre-hook."""

    def __init__(self):
        self.reset(None, None)

    def reset(self, input_ids, lengths):
        self.pre = None
        self.input_ids = input_ids
        self.lengths = lengths


BLOCK_STATE = BlockState()
