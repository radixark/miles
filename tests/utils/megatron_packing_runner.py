import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.training.training import get_model
from transformers import AutoTokenizer

import miles_plugins.mbridge  # noqa: F401
from mbridge import AutoBridge
from miles.backends.megatron_utils.arguments import parse_args, set_default_megatron_args, validate_args
from miles.backends.megatron_utils.data import DataIterator, get_batch
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func


_CHAT_MESSAGES = [
    {
        "role": "system",
        "content": (
            "\nYou are a highly capable and helpful math problem-solving assistant. Your goal is to carefully analyze "
            "and solve the user's mathematics question by providing a detailed, step-by-step explanation that is easy "
            "to follow and logically sound.\n\n## Instructions\n\n1. **Carefully read and fully understand the user's "
            "question.** Identify what is being asked and any relevant information or constraints.\n\n2. **Generate a "
            "detailed problem-solving process**, breaking down the solution into clear, logical steps. Explain each "
            "step thoroughly, including all intermediate calculations, reasoning, definitions, and justifications—even "
            "if they seem simple or obvious. This helps ensure clarity and aids understanding.\n\n3. **Structure your "
            "reasoning by enclosing all these problem-solving steps within `<think>` tags.**\n\n4. **After completing "
            "the detailed reasoning, provide the final answer clearly and unambiguously, enclosed within `<answer>` "
            "tags.** Ensure the answer directly addresses the question.\n\n5. If the problem involves multiple parts or "
            "sub-questions, address each part sequentially, separating your reasoning and answers clearly.\n\n6. Use "
            "correct mathematical notation and terminology throughout.\n\n7. Avoid skipping steps or assuming knowledge; "
            "assume the reader needs a comprehensive explanation.\n\n## Output Format\n<think>\n<!-- Detailed, step-by-"
            "step problem-solving explanation goes here -->\n</think>\n\n<answer>\n<!-- Final answer to the problem goes "
            "here -->\n</answer>\n"
        ),
    },
    {
        "role": "user",
        "content": (
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
            "How many clips did Natalia sell altogether in April and May?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Okay, let's see. Natalia sold clips to 48 friends in April. Then she sold half as many in May. So first, "
            "I need to find out how many clips she sold in May.\n\nHmm, half as many as April's total. April has 48 "
            "friends, so the number sold in May would be half of 48. Let me calculate that. 48 divided by 2 is 24. "
            "So she sold 24 clips in May.\n\nNow, to find the total number sold in both months, I just add the April "
            "sales and May sales. April was 48, May was 24. Adding them together gives me 48 + 24. Let me check that "
            "again. 48 plus 24 is 72. So Natalia sold 72 clips altogether in April and May.\n\nWait, let me confirm. "
            "Yes, 48 in April and half that in May equals 24, so total 72. I think that's right. No steps skipped here, "
            "just basic arithmetic."
        ),
    },
]


def _add_packing_args(parser):
    parser.add_argument("--hf-checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--pad-multiplier", type=int, default=128)
    parser.add_argument("--backward", action="store_true")
    return parser


def _get_dist_env() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    return world_size, local_rank, rank


def _build_rollout_data(tokenizer: AutoTokenizer, device: torch.device, num_samples: int) -> dict:
    full_token_ids = tokenizer.apply_chat_template(_CHAT_MESSAGES, tokenize=True)
    prompt_token_ids = tokenizer.apply_chat_template(
        _CHAT_MESSAGES[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )
    response_length = len(full_token_ids) - len(prompt_token_ids)
    loss_mask = torch.ones(response_length, device=device, dtype=torch.float32)

    tokens = [torch.tensor(full_token_ids, device=device, dtype=torch.long) for _ in range(num_samples)]
    loss_masks = [loss_mask for _ in range(num_samples)]
    response_lengths = [response_length for _ in range(num_samples)]
    total_lengths = [len(full_token_ids) for _ in range(num_samples)]
    return {
        "tokens": tokens,
        "loss_masks": loss_masks,
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
    }


def _build_padded_batch(rollout_data: dict, pad_size: int, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    total_lengths = rollout_data["total_lengths"]
    max_total_len = max(total_lengths)
    if pad_size > 1:
        max_total_len = (max_total_len + pad_size - 1) // pad_size * pad_size

    padded_tokens = []
    padded_loss_masks = []
    for tokens, loss_mask, total_len, response_len in zip(
        rollout_data["tokens"],
        rollout_data["loss_masks"],
        rollout_data["total_lengths"],
        rollout_data["response_lengths"],
        strict=True,
    ):
        prompt_len = total_len - response_len
        full_loss_mask = F.pad(loss_mask, (prompt_len - 1, 1), value=0)
        pad_len = max_total_len - total_len
        padded_tokens.append(F.pad(tokens, (0, pad_len), value=pad_token_id))
        padded_loss_masks.append(F.pad(full_loss_mask, (0, pad_len), value=0))

    return torch.stack(padded_tokens), torch.stack(padded_loss_masks)


def _compute_masked_loss(logits: torch.Tensor, tokens: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    labels = tokens[:, 1:].contiguous()
    logits = logits[:, :-1, :]
    loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)
    aligned_mask = loss_mask[:, :-1].float()
    masked_loss = loss * aligned_mask
    return masked_loss.sum() / torch.clamp_min(aligned_mask.sum(), 1)


def _load_megatron_model(args):
    if args.context_parallel_size != 1:
        raise ValueError("context_parallel_size must be 1 for this packing check.")

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    if len(model) != 1:
        raise ValueError("pipeline_model_parallel_size must be 1 for this packing check.")

    bridge = AutoBridge.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    bridge.load_weights(model, args.hf_checkpoint, memory_efficient=True)
    return model[0]


def main() -> None:
    args = parse_args(_add_packing_args)
    args = set_default_megatron_args(args)
    args.micro_batch_size = 1
    args.global_batch_size = 1
    args.save_interval = 1

    world_size, local_rank, rank = _get_dist_env()
    args.world_size = world_size
    args.rank = rank
    args.local_rank = local_rank

    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    device = torch.device("cuda", local_rank)
    rollout_data = _build_rollout_data(tokenizer, device, args.num_samples)
    max_seq_len = max(rollout_data["total_lengths"])
    args.seq_length = max_seq_len
    args.max_position_embeddings = max(args.max_position_embeddings, args.seq_length)

    validate_args(args)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    init(args)

    model = _load_megatron_model(args)
    model.eval()

    data_iterator = DataIterator(rollout_data, micro_batch_size=len(rollout_data["tokens"]))
    packed_batch = get_batch(
        data_iterator,
        ["tokens", "loss_masks", "total_lengths", "response_lengths"],
        pad_multiplier=args.pad_multiplier,
        qkv_format="thd",
    )

    padded_tokens, padded_loss_masks = _build_padded_batch(
        rollout_data,
        pad_size=args.tensor_model_parallel_size * args.pad_multiplier,
        pad_token_id=0,
    )

    packed_logits = model(
        input_ids=packed_batch["tokens"],
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params=packed_batch["packed_seq_params"],
        loss_mask=packed_batch["full_loss_masks"],
    )
    padded_logits = model(
        input_ids=padded_tokens,
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params=None,
        loss_mask=padded_loss_masks,
    )

    packed_loss = _compute_masked_loss(packed_logits, packed_batch["tokens"], packed_batch["full_loss_masks"])
    padded_loss = _compute_masked_loss(padded_logits, padded_tokens, padded_loss_masks)

    if args.backward:
        model.zero_grad(set_to_none=True)
        packed_loss.backward()
        model.zero_grad(set_to_none=True)
        padded_loss.backward()

    torch.testing.assert_close(packed_loss, padded_loss, rtol=1e-4, atol=1e-5)

    if dist.get_rank() == 0:
        print(
            f"packed_loss={packed_loss.item():.6f} padded_loss={padded_loss.item():.6f} "
            f"diff={abs(packed_loss.item() - padded_loss.item()):.6e}",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
