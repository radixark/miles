import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_model
from transformers import AutoTokenizer

def get_reference_impl_dir() -> Path:
    import sglang
    sglang_root = Path(sglang.__file__).parent.parent.parent
    return sglang_root / "sunrise" / "reference_implementation_updated"
 
REF_IMPL_DIR = get_reference_impl_dir()
sys.path.insert(0, str(REF_IMPL_DIR))

from generate import generate
from model import ModelArgs, Transformer


def print_top_predictions_for_rank(logits: torch.Tensor, input_ids: torch.Tensor, top_k: int, tokenizer, rank: int):
    """Print top-k predictions for this rank, compact format (one line per position)."""
    seq_length = logits.shape[0]
    print(f"\n--- Rank {rank} (seq_len={seq_length}) ---")
    for pos in range(seq_length):
        input_token = input_ids[pos]
        probs = torch.softmax(logits[pos], dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        input_str = tokenizer.decode([input_token]) if tokenizer else f"t{input_token}"
        preds = ", ".join(
            f"{tokenizer.decode([idx.item()]) if tokenizer else f't{idx.item()}'}({prob.item():.3f})"
            for prob, idx in zip(top_probs, top_indices)
        )
        print(f"pos[{pos:3d}] {input_str!r:12s} -> {preds}")


def print_top_predictions_all_ranks(logits: torch.Tensor, input_ids: torch.Tensor, top_k: int, tokenizer):
    """Print top-k predictions from all ranks sequentially (rank 0 first, then rank 1, etc.)."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if rank == 0:
        print("\n" + "=" * 80)
        print(f"Top-{top_k} Predictions (all ranks)")
        print(f"World size: {world_size}")
        print("=" * 80)

    for r in range(world_size):
        if dist.is_initialized():
            dist.barrier()
        if rank == r:
            print_top_predictions_for_rank(logits, input_ids, top_k, tokenizer, rank)
            sys.stdout.flush()

    if dist.is_initialized():
        dist.barrier()


def reset_kv_caches(model: Transformer):
    """Reset all KV caches in the model to zeros."""
    for layer in model.layers:
        # Reset attention KV cache
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'kv_cache'):
            layer.attn.kv_cache.zero_()
        # Reset compressor KV cache if exists
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'compressor'):
            if layer.attn.compressor is not None and hasattr(layer.attn.compressor, 'kv_cache'):
                if layer.attn.compressor.kv_cache is not None:
                    layer.attn.compressor.kv_cache.zero_()
        # Reset indexer KV cache if exists
        if hasattr(layer, 'attn') and hasattr(layer.attn, 'indexer'):
            if layer.attn.indexer is not None and hasattr(layer.attn.indexer, 'kv_cache'):
                if layer.attn.indexer.kv_cache is not None:
                    layer.attn.indexer.kv_cache.zero_()


@torch.inference_mode()
def forward_pass(model: Transformer, input_ids: list) -> torch.Tensor:
    """Run forward pass on input tokens and return logits for all positions (incremental mode)."""
    seq_len = len(input_ids)
    tokens = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    
    all_logits = []
    for i in range(1, seq_len + 1):
        reset_kv_caches(model)
        partial_tokens = tokens[:, :i]
        logits = model.forward(partial_tokens, 0)
        all_logits.append(logits[0].clone())  # [vocab_size]
    
    return torch.stack(all_logits, dim=0)  # [seq_len, vocab_size]


@torch.inference_mode()
def forward_pass_prefill(model: Transformer, input_ids: list) -> torch.Tensor:
    """Run forward pass on all input tokens at once (prefill mode, for comparison with Megatron)."""
    tokens = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    reset_kv_caches(model)
    logits = model.forward(tokens, 0)
    return logits


def main():
    parser = ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5, help="Top-k predictions to show")
    parser.add_argument("--forward-only", action="store_true", 
                        help="Only do forward pass (no generation), print logprobs")
    parser.add_argument("--prefill-mode", action="store_true",
                        help="Use prefill mode (all tokens at once) instead of incremental mode")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size (used by torchrun, this arg is for documentation)")
    args = parser.parse_args()

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)

    with open(args.config_path) as f:
        model_args = ModelArgs(**json.load(f))
    model_args.max_batch_size = 1

    with torch.device("cuda"):
        model = Transformer(model_args)

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    load_model(
        model,
        os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"),
        strict=False,
    )
    torch.set_default_device("cuda")

    prompt_tokens = tokenizer.encode(args.prompt, add_special_tokens=True)
    print(f"Prompt: {args.prompt[:100]}...")
    print(f"Prompt tokens: {len(prompt_tokens)}")

    if args.forward_only:
        # Forward pass only - print logprobs like run_megatron.py
        if args.prefill_mode:
            print("\nRunning forward pass (prefill mode - all tokens at once)...")
            logits = forward_pass_prefill(model, prompt_tokens)
        else:
            print("\nRunning forward pass (incremental mode - token by token)...")
            logits = forward_pass(model, prompt_tokens)
        print(f"Output logits shape: {logits.shape}")
        
        print_top_predictions_all_ranks(logits, prompt_tokens, args.top_k, tokenizer)
            
        if rank == 0:
            print("\n" + "=" * 80)
            print("Summary")
            print("=" * 80)
            print(f"Model type: {type(model).__name__}")
            print(f"Input length: {len(prompt_tokens)}")
            print(f"Output shape: {logits.shape}")
            print(f"Output dtype: {logits.dtype}")
            print("=" * 80)
    else:
        # Generation mode
        completion_tokens = generate(
            model,
            [prompt_tokens],
            args.max_new_tokens,
            tokenizer.eos_token_id,
            args.temperature,
        )

        if rank == 0:
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(f"Completion tokens: {completion_tokens[0]}")
            print(f"Completion: {completion}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
