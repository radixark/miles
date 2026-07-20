"""M1 gate, GPU: real build + streaming HF load + forward-logprob parity vs HF
transformers on a packed, padded batch. Exercises the real FSDP2-sharded path
(run under torchrun with as many ranks as available).

Usage: torchrun --nproc_per_node=<N> -m \
    miles.backends.experimental.torchtitan_utils.tests.test_m1_build_load_forward \
    <hf_ckpt_dir>
"""

import os
import sys

import torch
import torch.distributed as dist

from miles.backends.experimental.torchtitan_utils import compat  # noqa: F401  (shim first)
from miles.backends.experimental.torchtitan_utils import models
from miles.backends.experimental.torchtitan_utils.model import _load_hf_checkpoint, build_and_load_model


class _Args:
    tt_tensor_parallel_size = 1
    tt_expert_parallel_size = 1
    tt_dp_replicate = 1
    tt_attn_backend = "flex"
    tt_ac_mode = "none"
    tt_compile = False


def _make_packed_batch(tokenizer, seq_len: int, device: str):
    """4 short documents packed into one row, right-padded to seq_len."""
    docs = [
        "The capital of France is",
        "In mathematics, the sum of 2 and 2 equals",
        "A group of lions is called a",
        "Photosynthesis converts sunlight into",
    ]
    ids_per_doc = [tokenizer(d, return_tensors="pt")["input_ids"][0] for d in docs]
    tokens, positions = [], []
    for ids in ids_per_doc:
        tokens.append(ids)
        positions.append(torch.arange(len(ids)))
    tokens_flat = torch.cat(tokens)
    positions_flat = torch.cat(positions)
    pad = seq_len - tokens_flat.numel()
    assert pad >= 0
    tokens_padded = torch.cat([tokens_flat, torch.zeros(pad, dtype=tokens_flat.dtype)])
    positions_padded = torch.cat([positions_flat, torch.zeros(pad, dtype=positions_flat.dtype)])
    return tokens_padded.unsqueeze(0).to(device), positions_padded.unsqueeze(0).to(device), ids_per_doc


def run(hf_dir: str) -> int:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    from torchtitan.distributed import ParallelDims

    parallel_dims = ParallelDims(
        dp_replicate=1, dp_shard=-1, cp=1, tp=1, pp=1, ep=1, world_size=world_size
    )
    parallel_dims.build_mesh()

    spec, hf = models.spec_from_hf(hf_dir)
    args = _Args()
    seq_len = 256
    model, adapter = build_and_load_model(
        spec, hf_dir, parallel_dims=parallel_dims, seq_len=seq_len, args=args, device=device
    )
    _load_hf_checkpoint(model, adapter, hf_dir)
    model.eval()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    tokens, positions, ids_per_doc = _make_packed_batch(tokenizer, seq_len, device)

    with torch.no_grad():
        logits = model(
            tokens=tokens, positions=positions, attention_masks=model.get_attention_masks(positions)
        ).float()

    if rank == 0:
        from transformers import AutoModelForCausalLM

        ref_model = AutoModelForCausalLM.from_pretrained(
            hf_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(device)
        ref_model.eval()

        # Compare per-document, unpacked (HF has no packed-doc forward): this is the
        # true parity check — same tokens, same positions-within-doc, titan packed
        # vs HF separately-forwarded.
        max_abs_diff = 0.0
        offset = 0
        for doc_ids in ids_per_doc:
            n = doc_ids.numel()
            titan_logprobs = torch.log_softmax(logits[0, offset : offset + n].float(), dim=-1)
            with torch.no_grad():
                ref_logits = ref_model(input_ids=doc_ids.unsqueeze(0).to(device)).logits[0].float()
            ref_logprobs = torch.log_softmax(ref_logits, dim=-1)
            # score against the actual next-token to reduce this to a 1D comparison
            next_tokens = torch.cat([doc_ids[1:], doc_ids[:1]]).to(device)
            titan_scored = titan_logprobs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)[: n - 1]
            ref_scored = ref_logprobs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)[: n - 1]
            diff = (titan_scored - ref_scored).abs().max().item()
            max_abs_diff = max(max_abs_diff, diff)
            offset += n

        print(f"M1 FORWARD PARITY: max_abs_diff={max_abs_diff:.6f} (bar: <1e-2 bf16)")
        print("M1 FORWARD PARITY:", "PASS" if max_abs_diff < 1e-2 else "FAIL")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
