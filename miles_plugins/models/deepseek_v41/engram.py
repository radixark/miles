import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig

_TOKEN_MAP_CACHE: dict[str, tuple[list[int], int]] = {}
_LAYOUT_CACHE: dict[tuple, "EngramLayout"] = {}


def find_next_prime(start: int, seen_primes: set[int]) -> int:
    from sympy import isprime

    candidate = start + 1
    while not isprime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def get_compressed_token_map(hf_checkpoint: str) -> tuple[list[int], int]:
    if hf_checkpoint not in _TOKEN_MAP_CACHE:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
        _TOKEN_MAP_CACHE[hf_checkpoint] = build_compressed_token_map(tokenizer)
    return _TOKEN_MAP_CACHE[hf_checkpoint]


def compute_hash_multipliers(layer_ids, max_ngram_size: int, vocab_size: int) -> torch.Tensor:
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append(torch.tensor(values * 2 + 1))
    return torch.stack(rows)


def _mmap_rows(path: str, name: str, row_start: int, row_end: int, row_bytes: int) -> torch.Tensor:
    """Rows [row_start, row_end) of a 2-D uint8 safetensors tensor as a read-only memory map."""
    import struct

    import numpy as np

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    begin, _ = header[name]["data_offsets"]
    n = max(0, row_end - row_start)
    if n == 0:
        return torch.zeros(0, row_bytes, dtype=torch.uint8)
    mm = np.memmap(
        path, dtype=np.uint8, mode="r", offset=8 + header_len + begin + row_start * row_bytes, shape=(n, row_bytes)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.from_numpy(mm)


@dataclass(frozen=True)
class EngramLayout:
    max_ngram_size: int
    layer_ids: tuple
    num_embeddings: tuple
    primes: tuple
    n_heads: int
    head_dim: int


def build_engram_layout(config: TransformerConfig) -> EngramLayout | None:
    layer_ids = tuple(config.v41_engram_layer_ids)
    if not layer_ids:
        return None
    key = (
        layer_ids,
        tuple(config.v41_engram_num_embeddings),
        config.v41_engram_max_ngram_size,
        config.v41_engram_n_heads,
        config.v41_engram_head_dim,
        config.v41_engram_vocab_size,
    )
    if key not in _LAYOUT_CACHE:
        primes, seen = [], set()
        for _ in layer_ids:
            per_ngram = []
            for _ in range(config.v41_engram_max_ngram_size - 1):
                sizes, current = [], config.v41_engram_vocab_size - 1
                for _ in range(config.v41_engram_n_heads):
                    current = find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(tuple(per_ngram))
        _LAYOUT_CACHE[key] = EngramLayout(
            max_ngram_size=config.v41_engram_max_ngram_size,
            layer_ids=layer_ids,
            num_embeddings=tuple(config.v41_engram_num_embeddings),
            primes=tuple(primes),
            n_heads=config.v41_engram_n_heads,
            head_dim=config.v41_engram_head_dim,
        )
    return _LAYOUT_CACHE[key]


# the gate runs in fp32 on the full gathered sequence and autograd keeps several
# [seqlen, hc_mult, dim] fp32 temporaries; checkpointing it in sequence chunks
# bounds that to one chunk (elementwise per token, so the result is unchanged)
ENGRAM_GATE_CHUNK = int(os.environ.get("MILES_DSV41_ENGRAM_GATE_CHUNK", "4096"))


def engram_gate(x, kv, q_weight, k_weight, eps, clamp_value, chunk: int | None = None):
    chunk = ENGRAM_GATE_CHUNK if chunk is None else chunk
    seqlen = x.shape[0]
    if not chunk or seqlen <= chunk:
        return _engram_gate(x, kv, q_weight, k_weight, eps, clamp_value)
    if not torch.is_grad_enabled():
        # log_probs runs the gate over the whole gathered sequence, where each fp32 temporary is
        # 5.4 GB at 64k; the gate is elementwise per token, so chunking writes the same bytes
        out = torch.empty_like(x)
        for s in range(0, seqlen, chunk):
            out[s : s + chunk] = _engram_gate(
                x[s : s + chunk], kv[s : s + chunk], q_weight, k_weight, eps, clamp_value
            )
        return out
    # one fp32 view of each weight for all chunks, so their gradients accumulate in fp32
    q32, k32 = q_weight.float(), k_weight.float()
    outs = [
        torch.utils.checkpoint.checkpoint(
            _engram_gate, x[s : s + chunk], kv[s : s + chunk], q32, k32, eps, clamp_value, use_reentrant=False
        )
        for s in range(0, seqlen, chunk)
    ]
    return torch.cat(outs, dim=0)


def _engram_gate(x, kv, q_weight, k_weight, eps, clamp_value):
    hc_mult, dim = x.shape[-2:]
    key, value = kv.split([hc_mult * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc_mult, dim))
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class DeepSeekV41Engram(MegatronModule):
    def __init__(self, config: TransformerConfig, layer_id: int, tp_group, cp_group=None):
        super().__init__(config=config)
        self.cp_group = cp_group
        self.cp_size = cp_group.size() if cp_group is not None else 1
        layout = build_engram_layout(config)
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.hc_mult = config.num_residual_streams
        self.dim = config.hidden_size
        self.head_dim = layout.head_dim
        self.max_ngram_size = layout.max_ngram_size
        self.eps = config.layernorm_epsilon
        self.clamp_value = 1e-6
        self.tp_group = tp_group
        self.tp_size = tp_group.size() if tp_group is not None else 1
        tp_rank = tp_group.rank() if tp_group is not None else 0
        rows = layout.num_embeddings[self.layer_hash_index]
        self.rows = rows
        self.rows_local = -(-rows // self.tp_size)
        self.row_start = tp_rank * self.rows_local
        self.rows_avail = max(0, min(self.row_start + self.rows_local, rows) - self.row_start)
        self.table = None
        self.table_scale = None
        self._table_loaded = False

        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.linear_wkv = TELinear(
            n_hash_cols * layout.head_dim,
            self.dim * (self.hc_mult + 1),
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim, dtype=torch.float32))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim, dtype=torch.float32))
        mark_keep_in_fp32(self.q_weight)
        mark_keep_in_fp32(self.k_weight)

        self.sequence_parallel = config.sequence_parallel
        for p in self.parameters():
            p.sequence_parallel = False
        token_map, vocab_size = get_compressed_token_map(config.v41_hf_checkpoint)
        assert vocab_size == config.v41_engram_compressed_vocab_size, (
            vocab_size,
            config.v41_engram_compressed_vocab_size,
        )
        self.pad_id = token_map[config.v41_engram_pad_token_id]
        flat = [p for per_ngram in layout.primes[self.layer_hash_index] for p in per_ngram]
        offsets = np.cumsum([0, *flat[:-1]])
        multipliers = compute_hash_multipliers(layout.layer_ids, layout.max_ngram_size, vocab_size)
        self.register_buffer("token_map", torch.tensor(token_map, dtype=torch.int64), persistent=False)
        self.register_buffer("multipliers", multipliers[self.layer_hash_index].clone(), persistent=False)
        self.register_buffer(
            "primes", torch.tensor(layout.primes[self.layer_hash_index], dtype=torch.int64), persistent=False
        )
        self.register_buffer("offsets", torch.tensor(np.array(offsets), dtype=torch.int64), persistent=False)
        self._load_table()

    def _load_table(self):
        ckpt = self.config.v41_hf_checkpoint
        index = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
        end = self.row_start + self.rows_avail
        self.table = _mmap_rows(
            os.path.join(ckpt, index[f"layers.{self.layer_id}.engram.embed.weight"]),
            f"layers.{self.layer_id}.engram.embed.weight",
            self.row_start,
            end,
            self.head_dim,
        )
        self.table_scale = _mmap_rows(
            os.path.join(ckpt, index[f"layers.{self.layer_id}.engram.embed.scale"]),
            f"layers.{self.layer_id}.engram.embed.scale",
            self.row_start,
            end,
            self.head_dim // 32,
        )
        self._table_loaded = True

    def hash_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seqlen = input_ids.shape
        compressed = self.token_map[input_ids]
        positions = torch.arange(seqlen, device=input_ids.device).expand(batch, seqlen)
        tokens, blocked = [], torch.zeros_like(positions, dtype=torch.bool)
        for shift in range(self.max_ngram_size):
            source = compressed.gather(1, (positions - shift).clamp_min(0))
            blocked = blocked | (positions < shift)
            tokens.append(torch.where(blocked, self.pad_id, source))
        tokens = torch.stack(tokens, dim=-1)
        products = tokens * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        if not self._table_loaded:
            self._load_table()
        local = ids - self.row_start
        owned = (local >= 0) & (local < self.rows_avail)
        local = local.masked_fill(~owned, 0)
        local_cpu = local.to("cpu")
        rows = self.table[local_cpu].pin_memory().to(ids.device, non_blocking=True)
        scales = self.table_scale[local_cpu].pin_memory().to(ids.device, non_blocking=True)
        rows = rows.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, 32))
        scales = scales.view(torch.float8_e8m0fnu).float().unsqueeze(-1)
        values = (rows * scales).flatten(-2).to(torch.bfloat16)
        values = values.masked_fill(~owned.unsqueeze(-1), 0)
        if self.tp_size > 1:
            torch.distributed.all_reduce(values, group=self.tp_group)
        return values

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, tensor_parallel_output_grad=False, group=self.tp_group
            )
        seqlen, bsz, _ = hidden_states.shape
        assert input_ids is not None and input_ids.shape[0] == bsz and input_ids.shape[1] == seqlen
        with torch.no_grad():
            if self.cp_size > 1:
                chunks = [torch.empty_like(input_ids) for _ in range(self.cp_size)]
                torch.distributed.all_gather(chunks, input_ids.contiguous(), group=self.cp_group)
                start = self.cp_group.rank() * seqlen
                ids = self.hash_ids(torch.cat(chunks, dim=1))[:, start : start + seqlen].transpose(0, 1)
            else:
                ids = self.hash_ids(input_ids).transpose(0, 1)
            rows = self.lookup(ids)
        kv, _ = self.linear_wkv(rows.flatten(-2))
        x = hidden_states.view(seqlen, bsz, self.hc_mult, self.dim)
        out = engram_gate(x, kv, self.q_weight, self.k_weight, self.eps, self.clamp_value).view(
            seqlen, bsz, self.hc_mult * self.dim
        )
        if self.sequence_parallel:
            out = scatter_to_sequence_parallel_region(out, group=self.tp_group)
        return out
