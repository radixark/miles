#!/usr/bin/env python3
"""
Worker script for running Megatron model forward pass.
This script is called by test_forward_pass.py via torchrun with MODEL_ARGS from shell script.

Usage:
    source scripts/models/deepseek-v4-285B-5layer.sh && \
    PYTHONPATH=$MEGATRON_PATH torchrun --nproc-per-node 1 run_megatron.py \
        "${MODEL_ARGS[@]}" \
        --prompt-file /tmp/prompt.txt \
        --hf-checkpoint /path/to/hf_model \
        --ref-load /path/to/megatron_ckpt \
        --seq-length 128 \
        --micro-batch-size 1
"""

import logging
import os

import torch
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_distributed():
    """Initialize torch distributed."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
        )

    return rank, world_size


def add_extra_args(parser):
    """Add extra arguments for this script."""
    group = parser.add_argument_group(title="miles_model")
    group.add_argument("--hf-checkpoint", type=str, default=None,
                       help="Path to HuggingFace checkpoint")
    group.add_argument("--megatron-to-hf-mode", choices=["raw", "bridge"], default="raw")
    group.add_argument("--custom-model-provider-path", type=str, default=None)
    group.add_argument("--ref-load", type=str, default=None,
                       help="Path to Megatron checkpoint")
    
    # Test-specific args
    test_group = parser.add_argument_group(title="test_args")
    test_group.add_argument("--prompt-file", type=str, default=None,
                            help="Path to file containing prompt text")
    test_group.add_argument("--apply-chat-template", action="store_true",
                            help="Apply chat template to prompt")
    test_group.add_argument("--top-k", type=int, default=5,
                            help="Top-k predictions to show")
    test_group.add_argument("--tp-size", type=int, default=1,
                            help="Tensor parallel size")
    test_group.add_argument("--run-backward", action="store_true",
                            help="Run backward pass with dummy loss after forward")
    return parser


def parse_args():
    """Parse arguments using Megatron's parser + extra args.
    
    MODEL_ARGS from shell script are passed directly on command line.
    """
    from megatron.training.arguments import parse_args as megatron_parse_args, validate_args
    from miles.backends.megatron_utils.arguments import set_default_megatron_args

    args = megatron_parse_args(extra_args_provider=add_extra_args)
    args = set_default_megatron_args(args)

    args.rank = int(os.environ.get("LOCAL_RANK", 0))
    args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    tp_size = args.tp_size
    args.tensor_model_parallel_size = tp_size
    args.pipeline_model_parallel_size = 1
    args.context_parallel_size = 1
    args.expert_model_parallel_size = tp_size
    args.expert_tensor_parallel_size = 1
    
    if tp_size > 1:
        args.sequence_parallel = True

    if not args.global_batch_size:
        args.global_batch_size = args.micro_batch_size

    validate_args(args)

    return args


def initialize_megatron(args):
    """Initialize Megatron distributed environment."""
    from megatron.core import mpu, tensor_parallel
    from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
    from megatron.training.global_vars import _build_tokenizer, set_args

    set_args(args)

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=args.context_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        expert_tensor_parallel_size=args.expert_tensor_parallel_size,
    )

    tensor_parallel.model_parallel_cuda_manual_seed(args.seed)

    if args.hf_checkpoint:
        _build_tokenizer(args)

    global_batch_size = getattr(args, "global_batch_size", None) or args.micro_batch_size
    init_num_microbatches_calculator(
        args.rank,
        getattr(args, "rampup_batch_size", None),
        global_batch_size,
        args.micro_batch_size,
        getattr(args, "data_parallel_size", 1) or 1,
        getattr(args, "decrease_batch_size_if_needed", False),
    )

    logger.info(f"Initialized Megatron with TP={args.tensor_model_parallel_size}, PP={args.pipeline_model_parallel_size}")


def create_model_and_load_checkpoint(args):
    """Create the Megatron model and load checkpoint weights."""
    from megatron.core.enums import ModelType
    from megatron.training.training import get_model
    from miles.backends.megatron_utils.checkpoint import load_checkpoint
    from miles.backends.megatron_utils.model_provider import get_model_provider_func
    from miles.utils.transformers_patch import with_transformers_patch

    logger.info("Creating model...")

    with with_transformers_patch():
        model_provider = get_model_provider_func(args, role="actor")

    # Don't use DDP to save memory - we'll add a dummy finish_grad_sync for finalize_model_grads
    model = get_model(model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False)

    if isinstance(model, list):
        num_params = sum(p.numel() for p in model[0].parameters())
    else:
        num_params = sum(p.numel() for p in model.parameters())

    # Add dummy finish_grad_sync to allow finalize_model_grads to work without DDP
    # (DP=1 means this is a no-op anyway)
    def _dummy_finish_grad_sync(self):
        pass
    
    if isinstance(model, list):
        for m in model:
            m.finish_grad_sync = _dummy_finish_grad_sync.__get__(m, type(m))
    else:
        model.finish_grad_sync = _dummy_finish_grad_sync.__get__(model, type(model))
    logger.info(f"Model created with {num_params:,} parameters")

    load_path = getattr(args, "load", None) or getattr(args, "ref_load", None)
    if load_path:
        logger.info(f"Loading checkpoint from: {load_path}")
        
        original_load = args.load
        args.load = load_path
        args.no_load_optim = True
        args.no_load_rng = True
        args.finetune = True
        
        try:
            iteration, _ = load_checkpoint(
                ddp_model=model if isinstance(model, list) else [model],
                optimizer=None,
                opt_param_scheduler=None,
                checkpointing_context={},
                skip_load_to_model_and_opt=False,
            )
            logger.info(f"Checkpoint loaded successfully (iteration: {iteration})")
        finally:
            args.load = original_load
    else:
        logger.warning("No checkpoint path specified. Model weights will be randomly initialized!")

    if isinstance(model, list):
        model = model[0]

    model.eval()
    return model


def prepare_inputs(tokenizer, prompt: str, seq_length: int, batch_size: int, apply_chat_template: bool):
    """Prepare input tensors for forward pass."""
    device = torch.cuda.current_device()

    if apply_chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt

    input_ids = tokenizer.encode(text)
    original_len = len(input_ids)
    
    # Pad or truncate
    if len(input_ids) < seq_length:
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        input_ids = input_ids + [pad_id] * (seq_length - len(input_ids))
    else:
        input_ids = input_ids[:seq_length]
        original_len = seq_length
    
    # Set env var for dump comparison (last actual token position)
    import os
    os.environ["MEGATRON_HACK_DUMP_LOGITS_POS"] = str(original_len - 1)

    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    if batch_size > 1:
        input_ids = input_ids.repeat(batch_size, 1)

    actual_seq_length = input_ids.shape[1]
    position_ids = torch.arange(actual_seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    attention_mask = torch.ones(
        (batch_size, 1, actual_seq_length, actual_seq_length), dtype=torch.bool, device=device
    )
    attention_mask = torch.tril(attention_mask)

    logger.info(f"Input text: {text[:100]}...")
    logger.info(f"Input shape: {input_ids.shape}")

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }


def run_forward_pass(model, inputs: dict, enable_grad: bool = False) -> torch.Tensor:
    """Run forward pass and return logits."""
    logger.info(f"Running forward pass... (enable_grad={enable_grad})")

    if enable_grad:
        output = model(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            attention_mask=inputs["attention_mask"],
            runtime_gather_output=True,
        )
    else:
        with torch.no_grad():
            output = model(
                input_ids=inputs["input_ids"],
                position_ids=inputs["position_ids"],
                attention_mask=inputs["attention_mask"],
                runtime_gather_output=True,
            )

    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    logger.info(f"Forward pass complete. Output logits shape: {logits.shape}")
    return logits


def run_backward_pass(logits: torch.Tensor, input_ids: torch.Tensor, model: torch.nn.Module = None) -> None:
    """Run backward pass with cross-entropy loss (next-token prediction).
    
    Args:
        logits: Model output logits of shape [batch_size, seq_length, vocab_size]
        input_ids: Input token IDs of shape [batch_size, seq_length], used as labels
        model: Optional model to dump parameter gradients
    """
    from sglang.srt.debug_utils.dumper import dumper

    logger.info("Running backward pass...")

    # Use cross-entropy loss
    shift_logits = logits[:, :-1, :].contiguous()  # [batch, seq-1, vocab]
    shift_labels = input_ids[:, 1:].contiguous()   # [batch, seq-1]
    
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),  # [batch*(seq-1), vocab]
        shift_labels.view(-1),                          # [batch*(seq-1)]
        reduction='mean'
    )

    print(f"Cross-entropy loss value: {loss.item():.6f}")

    loss.backward()

    # Finalize gradients (all-reduce for sequence_parallel params, etc.)
    if model is not None:
        from megatron.core.distributed import finalize_model_grads
        finalize_model_grads([model])
        dumper.dump_param_grads(model, name_prefix="model")

    logger.info("Backward pass complete.")


def print_top_predictions(logits: torch.Tensor, input_ids: torch.Tensor, top_k: int, tokenizer, pad_token_id: int = None):
    """Print top-k predictions for each position (excluding padding)."""
    batch_size, seq_length, vocab_size = logits.shape

    # Find the last non-padding position
    if pad_token_id is not None:
        # Find positions where input_ids != pad_token_id
        non_pad_mask = input_ids[0] != pad_token_id
        non_pad_indices = torch.where(non_pad_mask)[0]
        if len(non_pad_indices) > 0:
            last_token_pos = non_pad_indices[-1].item()
        else:
            last_token_pos = seq_length - 1
    else:
        last_token_pos = seq_length - 1

    print("\n" + "=" * 80)
    print(f"Top-{top_k} Predictions (batch 0, last 5 non-padding positions)")
    print(f"Sequence length: {seq_length}, Last token position: {last_token_pos}")
    print("=" * 80)

    # Show last 5 positions before (and including) last_token_pos
    start_pos = max(0, last_token_pos - 4)
    end_pos = last_token_pos + 1

    for pos in range(start_pos, end_pos):
        input_token = input_ids[0, pos].item()
        probs = torch.softmax(logits[0, pos], dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)

        input_token_str = tokenizer.decode([input_token]) if tokenizer else f"token_{input_token}"
        print(f"\nPosition {pos} (input: {input_token_str!r}):")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token_str = tokenizer.decode([idx.item()]) if tokenizer else f"token_{idx.item()}"
            print(f"  {i + 1}. {token_str!r} (id={idx.item()}): {prob.item():.4f}")


def main():
    # Initialize distributed
    rank, world_size = initialize_distributed()
    logger.info(f"Rank {rank}/{world_size} initialized")

    # Parse args (MODEL_ARGS from shell script + extra args)
    args = parse_args()

    # Initialize Megatron
    initialize_megatron(args)

    # Create model
    model = create_model_and_load_checkpoint(args)

    # Get tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    # Read prompt from file
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompt = f.read()
        logger.info(f"Read prompt from {args.prompt_file} ({len(prompt)} chars)")
    else:
        raise ValueError("--prompt-file is required")

    # Prepare inputs
    inputs = prepare_inputs(
        tokenizer=tokenizer,
        prompt=prompt,
        seq_length=args.seq_length,
        batch_size=args.micro_batch_size,
        apply_chat_template=getattr(args, "apply_chat_template", False),
    )

    # Run forward pass
    run_backward = getattr(args, "run_backward", False)
    logits = run_forward_pass(model, inputs, enable_grad=run_backward)

    # Run backward pass if requested
    if run_backward:
        run_backward_pass(logits, input_ids=inputs["input_ids"], model=model)

    # Print results
    top_k = getattr(args, "top_k", 5)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if rank == 0:
        print_top_predictions(logits, inputs["input_ids"], top_k, tokenizer, pad_token_id)

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Model type: {type(model).__name__}")
        print(f"Input shape: {inputs['input_ids'].shape}")
        print(f"Output shape: {logits.shape}")
        print(f"Output dtype: {logits.dtype}")
        print("=" * 80)

    if dist.is_initialized():
        dist.barrier()

    logger.info("Forward pass completed successfully!")


if __name__ == "__main__":
    main()

