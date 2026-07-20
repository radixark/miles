"""M6 gate, GPU: fp32-forced version of test_m6_qwen35_packed_parity.py. The bf16
version showed ~0.15-0.25 max_abs_diff for EVERY document including doc[0] (which has
no preceding context, so GDN cross-boundary leakage can't be the cause there) — a
bisection (reverting the GDN patch to upstream) proved docs 1-3 blow up to ~15-22
without the patch (confirming the patch is essential and working) while doc[0] stays
at ~0.15 either way. This isolates whether that residual ~0.15 is pure bf16
kernel-config noise (different total sequence lengths - 29 packed vs 5-12 separate -
can select different autotuned flex-attention kernels, matching the exact
flex-vs-sdpa-style noise already characterized and cleared for qwen3 via
test_m1_fp32_check.py) or a real remaining bug, by forcing fp32 end-to-end.

Usage: torchrun --nproc_per_node=<N> -m \
    miles.backends.experimental.torchtitan_utils.tests.test_m6_qwen35_packed_parity_fp32 <hf_ckpt_dir>
"""

import sys

import torch
import torch.distributed as dist

from miles.backends.experimental.torchtitan_utils import compat  # noqa: F401
from miles.backends.experimental.torchtitan_utils import models

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
    seq_len = 64

    from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.tools.utils import set_default_dtype
    import miles.backends.experimental.torchtitan_utils.model as model_mod

    parallelism = ParallelismConfig(data_parallel_shard_degree=parallel_dims.dp_shard, tensor_parallel_degree=1)
    training = TrainingConfig(seq_len=seq_len, dtype="float32", mixed_precision_param="float32", mixed_precision_reduce="float32")
    model_mod._apply_update_from_config(spec.model, parallelism=parallelism, seq_len=seq_len)
    if getattr(spec.model, "vision_encoder", None) is not None:
        spec.model.vision_encoder = None
    with torch.device("meta"):
        with set_default_dtype(torch.float32):
            model = spec.model.build()
    model = spec.parallelize_fn(
        model, parallel_dims=parallel_dims, training=training, parallelism=parallelism,
        compile_config=CompileConfig(enable=False), ac_config=None, dump_folder="/tmp/titan_dump",
    )
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_weights(buffer_device=None)
    adapter = spec.state_dict_adapter(spec.model, hf_dir)
    model_mod._load_hf_checkpoint(model, adapter, hf_dir)
    model.eval()
    special_tokens = {"image_id": hf.get("image_token_id", -1), "video_id": hf.get("video_token_id", -1)}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    ids_per_doc = [tokenizer(d, return_tensors="pt")["input_ids"][0] for d in DOCS]

    tokens_flat = torch.cat(ids_per_doc)
    positions_flat = torch.cat([torch.arange(len(ids)) for ids in ids_per_doc])
    tokens_packed = tokens_flat.unsqueeze(0).to(device)
    positions_packed = positions_flat.unsqueeze(0).to(device)

    with torch.no_grad():
        packed_logits = model(
            tokens=tokens_packed,
            positions=positions_packed,
            attention_masks=model.get_attention_masks(positions_packed),
            special_tokens=special_tokens,
        ).float()

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
            print(f"[FP32] doc[offset={offset}, len={n}]: max_abs_diff={per_pos_max.max().item():.8f}")
            max_abs_diff_per_doc.append(per_pos_max.max().item())
        offset += n

    if rank == 0:
        overall = max(max_abs_diff_per_doc)
        print(f"M6 FP32 PACKED-VS-SEPARATE overall max_abs_diff={overall:.8f} (bar: <1e-4, same bar as test_m1_fp32_check)")
        print("M6 GDN PACKING FP32 CHECK:", "PASS" if overall < 1e-4 else "FAIL")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
