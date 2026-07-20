"""M6 gate, GPU: the real GDN packed-document correctness test. Forwards the same 4
documents twice through the SAME model instance — once packed into one row (positions
reset to 0 at each document start, exactly like miles' real RL packed batches), once
separately (each document alone, unpacked) — and compares per-document log-probs. If
qwen3_5_packing.apply()'s cu_seqlens patch is working, the two should match closely
(bf16 kernel noise only); if it's missing or broken, the GDN layers (24 of 32) leak
context across document boundaries and the packed run diverges from the separate run,
worse for documents later in the packed row (more upstream context to leak from).

This is titan-vs-titan (packed vs separate), not titan-vs-HF — the single-doc test
(test_m6_qwen35_single_doc.py) already covers architecture correctness against HF;
this test isolates the packing/boundary-masking correctness specifically.

Usage: torchrun --nproc_per_node=<N> -m \
    miles.backends.experimental.torchtitan_utils.tests.test_m6_qwen35_packed_parity <hf_ckpt_dir>
"""

import sys

import torch
import torch.distributed as dist

from miles.backends.experimental.torchtitan_utils import compat  # noqa: F401
from miles.backends.experimental.torchtitan_utils import models
from miles.backends.experimental.torchtitan_utils.model import _load_hf_checkpoint, build_and_load_model


class _Args:
    tt_tensor_parallel_size = 1
    tt_expert_parallel_size = 1
    tt_dp_replicate = 1
    tt_attn_backend = "flex"
    tt_ac_mode = "none"
    tt_compile = False


DOCS = [
    "The capital of France is",
    "In mathematics, the sum of 2 and 2 equals",
    "A group of lions is called a",
    "Photosynthesis converts sunlight into",
]


def run(hf_dir: str) -> int:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    from torchtitan.distributed import ParallelDims

    parallel_dims = ParallelDims(dp_replicate=1, dp_shard=-1, cp=1, tp=1, pp=1, ep=1, world_size=world_size)
    parallel_dims.build_mesh()

    spec, hf = models.spec_from_hf(hf_dir)
    args = _Args()
    seq_len = 256
    model, adapter = build_and_load_model(spec, hf_dir, parallel_dims=parallel_dims, seq_len=seq_len, args=args, device=device)
    _load_hf_checkpoint(model, adapter, hf_dir)
    model.eval()
    special_tokens = {"image_id": hf.get("image_token_id", -1), "video_id": hf.get("video_token_id", -1)}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    ids_per_doc = [tokenizer(d, return_tensors="pt")["input_ids"][0] for d in DOCS]

    # Packed: concatenate tokens, positions reset to 0 at each doc start. Deliberately NOT
    # padded to seq_len: padding with position=0 would make every pad token look like a new
    # 1-token "document" to the position-reset boundary detector, corrupting the GDN cu_seqlens
    # derivation for reasons unrelated to the actual packing correctness under test.
    tokens_flat = torch.cat(ids_per_doc)
    positions_flat = torch.cat([torch.arange(len(ids)) for ids in ids_per_doc])
    assert tokens_flat.numel() <= seq_len
    tokens_packed = tokens_flat.unsqueeze(0).to(device)
    positions_packed = positions_flat.unsqueeze(0).to(device)

    with torch.no_grad():
        packed_logits = model(
            tokens=tokens_packed,
            positions=positions_packed,
            attention_masks=model.get_attention_masks(positions_packed),
            special_tokens=special_tokens,
        ).float()

    # model(...) runs the FSDP2-sharded forward, a collective op every rank must join —
    # so every "separate" forward below runs on ALL ranks (not gated on rank==0, which
    # would make rank 0 enqueue collectives the other 7 ranks never join -> NCCL hang).
    # Only the diffing/printing is rank-0-only.
    max_abs_diff_per_doc = []
    offset = 0
    for doc_ids in ids_per_doc:
        n = doc_ids.numel()
        separate_tokens = doc_ids.unsqueeze(0).to(device)
        separate_positions = torch.arange(n).unsqueeze(0).to(device)
        with torch.no_grad():
            separate_logits = model(
                tokens=separate_tokens,
                positions=separate_positions,
                attention_masks=model.get_attention_masks(separate_positions),
                special_tokens=special_tokens,
            ).float()

        if rank == 0:
            packed_lp = torch.log_softmax(packed_logits[0, offset : offset + n], dim=-1)
            separate_lp = torch.log_softmax(separate_logits[0], dim=-1)
            diff = (packed_lp - separate_lp).abs()
            per_pos_max = diff.max(dim=-1).values
            print(f"doc[offset={offset}, len={n}]: max_abs_diff={per_pos_max.max().item():.6f}  per-position={per_pos_max.tolist()}")
            max_abs_diff_per_doc.append(per_pos_max.max().item())
        offset += n

    if rank == 0:
        overall = max(max_abs_diff_per_doc)
        print(f"M6 PACKED-VS-SEPARATE overall max_abs_diff={overall:.6f} (bar: <1e-2 bf16, same bar as test_m1_build_load_forward)")
        print("M6 GDN PACKING PARITY:", "PASS" if overall < 1e-2 else "FAIL")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
