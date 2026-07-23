"""Offline HF -> Megatron torch_dist conversion via NVIDIA ``megatron.bridge``.

The existing ``tools/convert_hf_to_torch_dist.py`` uses the community ``mbridge``
backend, which does not support the Nemotron-H (hybrid Mamba+Attention+MoE)
architecture. nemotron_h is bridged through ``miles_plugins.megatron_bridge``
(NVIDIA ``megatron.bridge``), the same path the live training run uses
(``--megatron-to-hf-mode bridge``, see
``miles/backends/megatron_utils/checkpoint.py::_load_checkpoint_hf``).

This tool mirrors that load path so a nemotron_h HF checkpoint can be converted
once to a Megatron-native dist checkpoint. Subsequent training runs then
``--load`` that directory and skip the (~10-15 min) per-run bridge conversion.

Usage (single node, N GPUs, N <= num_layers):
    torchrun --nproc-per-node 4 tools/convert_hf_to_torch_dist_bridge.py \
        <MODEL_ARGS...>                 # same structural args as the run script
        --hf-checkpoint /path/to/hf_ckpt \
        --save /path/to/megatron_dist_out

Then in the run script swap:
    --hf-checkpoint <hf>  -->  --load /path/to/megatron_dist_out
(keep --ref-load pointing at the HF dir, or convert a ref copy too).
"""

import gc
import os
import shutil

import torch
import torch.distributed as dist
from megatron.bridge import AutoBridge
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model

import miles_plugins.megatron_bridge  # noqa: F401  registers MilesNemotronHBridge
from miles.backends.megatron_utils.arguments import set_default_megatron_args
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.utils import megatron_bridge_utils
from miles.utils.logging_utils import configure_logger
from miles.utils.memory_utils import print_memory


def add_convertion_args(parser):
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="bridge",
        help="Must be 'bridge' for nemotron_h (megatron.bridge path).",
    )
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass
    return parser


def get_args():
    args = parse_args(add_convertion_args)
    args = set_default_megatron_args(args)

    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = world_size

    # If explicit parallelism (TP/PP/EP > 1) is requested, respect it as-is so the
    # saved checkpoint MATCHES the training run's layout and avoids dist-checkpoint
    # resharding (which can hit a BytesIO/extra_state bug for MoE+MTP models). Only
    # auto-spread PP across all GPUs when no parallelism was specified.
    explicit_parallel = (
        args.tensor_model_parallel_size > 1
        or args.pipeline_model_parallel_size > 1
        or getattr(args, "expert_model_parallel_size", 1) > 1
    )

    def ceildiv(a, b):
        return -(a // -b)

    # Spread layers across PP so each rank holds >=1 layer (mirror upstream tool).
    if not explicit_parallel and args.pipeline_model_parallel_size == 1 and world_size > 1:
        assert world_size <= args.num_layers, (
            f"World size {world_size} must be <= number of layers {args.num_layers}. "
            "Use fewer GPUs (--nproc-per-node) for this conversion."
        )
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)
            if args.decoder_last_pipeline_num_layers > 0:
                break
            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(f"Cannot find a valid PP size for {args.num_layers} layers and {world_size} GPUs.")
    print(
        f"Using PP size: {args.pipeline_model_parallel_size}, "
        f"decoder_last_pipeline_num_layers: {args.decoder_last_pipeline_num_layers}"
    )

    validate_args(args)
    return args


def main():
    configure_logger()

    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )

    args = get_args()
    assert args.megatron_to_hf_mode == "bridge", "nemotron_h requires --megatron-to-hf-mode bridge"
    init(args)

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Load HF weights through megatron.bridge (same path as the live run's
    # checkpoint.py::_load_checkpoint_hf), so nemotron_h MoE/MTP mappings apply.
    hf_model_path = args.hf_checkpoint
    with megatron_bridge_utils.patch_megatron_model(model):
        bridge = AutoBridge.from_hf_pretrained(hf_model_path, trust_remote_code=True)
        bridge.load_hf_weights(model)
    print(f"Model loaded via megatron.bridge: {hf_model_path}")

    print_memory("after loading model")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    save_checkpoint(1, model, None, None, 0)

    if dist.get_rank() == 0:
        source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        shutil.move(source_dir, target_dir)

    dist.barrier()

    # Must be the last step (after the barrier): downstream scripts treat the
    # tracker file as the success signal.
    if dist.get_rank() == 0:
        with open(get_checkpoint_tracker_filename(args.save), "w") as f:
            f.write("release")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
