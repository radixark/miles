"""Dump the FSDP training forward's per-module dtype inventory for one HF checkpoint.

Drives exactly the actor's model-construction path — class patches, config-time packing patches,
fp32 master, ``apply_fsdp2`` with the resolved ``PrecisionPolicy`` (spec included), and the forward
under ``precision_forward_context`` — but feeds it synthetic token ids instead of rollout data, since
dtypes do not depend on the token values. Combined with the non-intrusive Dumper it yields the
module_structure.json + console log that the ``dumper-module-report`` skill turns into the dtype
report and DAG.

    torchrun --standalone --nproc-per-node=2 tools/dump_fsdp_module_dtypes.py \\
        --hf-checkpoint /path/to/Qwen3.5-4B --seq-len 256
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

from miles.backends.experimental.fsdp_utils.actor import apply_fsdp2
from miles.backends.experimental.fsdp_utils.adaptations import (
    apply_class_patches,
    apply_fp32_master,
    apply_packing,
    apply_post_load_fixups,
    precision_forward_context,
    resolve_dtype,
    resolve_precision_policy,
)
from miles.backends.experimental.fsdp_utils.debug_dump import maybe_dumper_step, maybe_register_module_dumper

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("dump_fsdp_module_dtypes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint", required=True)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1, help="forward passes to dump")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--disable-fp32-master", dest="keep_fp32_master", action="store_false", default=True)
    parser.add_argument("--fsdp-precision-rules", default=None)
    # Explore a policy the arch has no spec for yet; a finding here becomes a specs/<arch>.py hook.
    parser.add_argument("--gather-dtype", choices=["fp32", "bf16", "fp16"], default=None)
    parser.add_argument("--autocast-dtype", choices=["fp32", "bf16", "fp16", "none"], default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--backward", action="store_true", help="also run a backward, dumping grad dtypes")
    # Knobs the shared construction path reads but this harness does not vary.
    parser.set_defaults(true_on_policy_mode=False, sglang_true_on_policy_contract=None, dp_replicate_size=1)
    return parser.parse_args()


def build_model(args, hf_config, mesh):
    """The actor's construction order, verbatim."""
    apply_class_patches(hf_config, args)
    apply_packing(None, hf_config, "config")

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_checkpoint,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    policy = resolve_precision_policy(hf_config, args)
    if args.gather_dtype:
        policy.param_dtype = resolve_dtype(args.gather_dtype)
    if args.autocast_dtype:
        policy.autocast_dtype = None if args.autocast_dtype == "none" else resolve_dtype(args.autocast_dtype)
    if policy.keep_fp32_master:
        model = apply_fp32_master(model, policy.sync_dtype_resolver)
    apply_post_load_fixups(model, hf_config, args.hf_checkpoint)
    apply_packing(model, hf_config, "post_load")
    model.train()

    model = apply_fsdp2(
        model.cuda(),
        mesh=mesh,
        cpu_offload=False,
        args=args,
        param_dtype=policy.param_dtype,
        reduce_dtype=policy.reduce_dtype,
        precision_spec=policy.precision_spec,
        cast_forward_inputs=policy.autocast_dtype is None,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model, policy


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    mesh = torch.distributed.device_mesh.init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))

    model, policy = build_model(args, hf_config, mesh)
    logger.info(
        f"[rank {rank}] policy: param(gather)={policy.param_dtype}, reduce={policy.reduce_dtype}, "
        f"autocast={policy.autocast_dtype}, fp32_master={policy.keep_fp32_master}, "
        f"spec_rules={len(policy.precision_spec.rules)}"
    )

    vocab = hf_config.get_text_config().vocab_size
    input_ids = torch.randint(0, vocab, (1, args.seq_len), device="cuda")
    position_ids = torch.arange(args.seq_len, device="cuda").unsqueeze(0)

    for step in range(args.steps):
        maybe_register_module_dumper(model)
        with precision_forward_context(policy):
            logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None).logits
        logger.info(f"[rank {rank}] step {step}: logits {tuple(logits.shape)} {logits.dtype}")
        if args.backward:
            logits.float().mean().backward()
        maybe_dumper_step()

    dist.barrier()
    if rank == 0:
        logger.info(f"done; dumper dir = {os.environ.get('DUMPER_DIR')}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
