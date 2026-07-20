"""M1 gate, GPU: verify checkpoint.py's CheckpointManager-based initial HF load produces
the same weights as the hand-rolled _load_hf_checkpoint recipe (test_m1_fp32_check.py's
proven-correct path), by running the same fp32 forward-parity check through the new
checkpoint.build()/.load() call instead.

Usage: torchrun --nproc_per_node=<N> -m \
    miles.backends.experimental.torchtitan_utils.tests.test_m1_checkpoint_manager <hf_ckpt_dir>
"""

import sys

import torch
import torch.distributed as dist

from miles.backends.experimental.torchtitan_utils import checkpoint, compat  # noqa: F401
from miles.backends.experimental.torchtitan_utils import models
from miles.backends.experimental.torchtitan_utils.model import build_optimizer_and_lr_scheduler


class _Args:
    tt_tensor_parallel_size = 1
    tt_expert_parallel_size = 1
    tt_dp_replicate = 1
    tt_attn_backend = "flex"
    tt_ac_mode = "none"
    tt_compile = False
    lr = 5e-6
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    adam_eps = 1e-8
    weight_decay = 0.1
    lr_warmup_iters = 0
    save = None
    wandb_group = "test-m1-checkpoint-manager"
    start_rollout_id = 0


class _FakeActor:
    """Just enough of TorchTitanTrainRayActor's shape for checkpoint.build()/.load()."""

    def __init__(self, args, model, adapter, optimizer, lr_scheduler):
        self.args = args
        self.model = model
        self.adapter = adapter
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.global_step = 0
        self.micro_step = 0


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

    # Force fp32 end-to-end (same as test_m1_fp32_check.py) so this test isolates
    # loading-correctness from the already-characterized bf16 flex-vs-sdpa numerics gap.
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

    optimizer, lr_scheduler = build_optimizer_and_lr_scheduler(model, spec, args, parallel_dims, training_steps=100)

    actor = _FakeActor(args, model, adapter, optimizer, lr_scheduler)
    actor.checkpointer = checkpoint.build(actor)
    checkpoint.load(actor)
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
        print(f"CKPT-MANAGER FP32 max_abs_diff={diff.max().item():.8f} (bar: <1e-4)")
        print("CKPT-MANAGER CHECK:", "PASS" if diff.max().item() < 1e-4 else "FAIL")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
