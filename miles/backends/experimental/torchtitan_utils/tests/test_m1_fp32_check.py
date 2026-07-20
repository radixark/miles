"""fp32-vs-fp32 sanity check: if bf16 titan-vs-HF shows a large gap but fp32 titan-vs-HF
is tight, the gap is bf16 kernel-numerics (flex vs sdpa), not a correctness bug.

Usage: torchrun --nproc_per_node=<N> -m \
    miles.backends.experimental.torchtitan_utils.tests.test_m1_fp32_check <hf_ckpt_dir>
"""

import sys

import torch
import torch.distributed as dist

from miles.backends.experimental.torchtitan_utils import compat  # noqa: F401
from miles.backends.experimental.torchtitan_utils import models
from miles.backends.experimental.torchtitan_utils.model import build_and_load_model


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

    # Force fp32 end-to-end (patch build_and_load_model's hardcoded bf16 mixed-precision
    # by monkeypatching TrainingConfig construction via a fresh call — simplest: reimplement
    # the same build sequence here with dtype="float32", mixed_precision_param="float32".
    import torchtitan.models.qwen3 as _q3  # noqa: F401  (ensure module loaded before patching)
    from torchtitan.config import TORCH_DTYPE_MAP, CompileConfig, ParallelismConfig, TrainingConfig
    from torchtitan.tools.utils import set_default_dtype
    import miles.backends.experimental.torchtitan_utils.model as model_mod

    parallelism = ParallelismConfig(data_parallel_shard_degree=parallel_dims.dp_shard, tensor_parallel_degree=1)
    training = TrainingConfig(seq_len=seq_len, dtype="float32", mixed_precision_param="float32", mixed_precision_reduce="float32")
    model_mod._apply_update_from_config(spec.model, parallelism=parallelism, seq_len=seq_len)
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

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_dir)
    ids = tokenizer("The capital of France is", return_tensors="pt")["input_ids"][0]
    tokens = ids.unsqueeze(0).to(device)
    positions = torch.arange(ids.numel()).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tokens=tokens, positions=positions, attention_masks=model.get_attention_masks(positions)).float()

    if rank == 0:
        from transformers import AutoModelForCausalLM

        ref_model = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=torch.float32, attn_implementation="sdpa").to(device)
        ref_model.eval()
        with torch.no_grad():
            ref_logits = ref_model(input_ids=tokens).logits[0].float()

        diff = (torch.log_softmax(logits[0], dim=-1) - torch.log_softmax(ref_logits, dim=-1)).abs()
        print(f"FP32 per-position max_abs_diff: {diff.max(dim=-1).values.tolist()}")
        print(f"FP32 overall max_abs_diff={diff.max().item():.8f}  (bar: <1e-4)")
        print("FP32 CHECK:", "PASS" if diff.max().item() < 1e-4 else "FAIL")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
