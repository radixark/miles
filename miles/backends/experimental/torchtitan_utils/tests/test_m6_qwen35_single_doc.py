"""M6 gate, GPU: single unpacked document, titan vs HF transformers. Mirrors
test_m1_single_doc.py's bisection role for qwen3 — if this fails, the bug is in
build/load/dtype or the GDN/full-attention forward itself, independent of any packing
concern (single doc has no document boundary to get wrong).

Usage: torchrun --nproc_per_node=<N> -m \
    miles.backends.experimental.torchtitan_utils.tests.test_m6_qwen35_single_doc <hf_ckpt_dir>
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
    seq_len = 64
    model, adapter = build_and_load_model(spec, hf_dir, parallel_dims=parallel_dims, seq_len=seq_len, args=args, device=device)
    _load_hf_checkpoint(model, adapter, hf_dir)
    model.eval()
    # Qwen35Model.forward dereferences special_tokens["image_id"/"video_id"] unconditionally.
    special_tokens = {"image_id": hf.get("image_token_id", -1), "video_id": hf.get("video_token_id", -1)}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"][0]
    n = ids.numel()
    tokens = ids.unsqueeze(0).to(device)
    positions = torch.arange(n).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(
            tokens=tokens, positions=positions, attention_masks=model.get_attention_masks(positions), special_tokens=special_tokens
        ).float()

    if rank == 0:
        from transformers import AutoModelForCausalLM

        ref_model = AutoModelForCausalLM.from_pretrained(
            hf_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
        ).to(device)
        ref_model.eval()
        with torch.no_grad():
            ref_logits = ref_model(input_ids=tokens).logits[0].float()

        titan_lp = torch.log_softmax(logits[0].float(), dim=-1)
        ref_lp = torch.log_softmax(ref_logits, dim=-1)
        diff = (titan_lp - ref_lp).abs()
        print(f"per-position max_abs_diff: {diff.max(dim=-1).values.tolist()}")
        print(f"overall max_abs_diff={diff.max().item():.6f}")
        pos, vocab = divmod(diff.argmax().item(), diff.shape[-1])
        print(f"argmax at position={pos} vocab_id={vocab} titan={titan_lp[pos, vocab].item():.4f} ref={ref_lp[pos, vocab].item():.4f}")
        print(f"titan top1 per position: {logits[0].argmax(dim=-1).tolist()}")
        print(f"ref   top1 per position: {ref_logits.argmax(dim=-1).tolist()}")
        argmax_match = (logits[0].argmax(dim=-1) == ref_logits.argmax(dim=-1)).float().mean().item()
        print(f"argmax match rate: {argmax_match:.4f}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
