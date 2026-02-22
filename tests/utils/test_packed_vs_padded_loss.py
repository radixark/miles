from collections.abc import Callable
from functools import lru_cache

import pytest
import torch
import torch.nn.functional as F
from megatron.core import mpu
from transformers import AutoTokenizer

from miles.backends.megatron_utils.data import DataIterator, get_batch


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


@lru_cache
def _get_tokenizer() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")


def _build_message_rollout_data(device: torch.device) -> dict:
    tokenizer = _get_tokenizer()
    full_token_ids = tokenizer.apply_chat_template(_CHAT_MESSAGES, tokenize=True)
    prompt_token_ids = tokenizer.apply_chat_template(
        _CHAT_MESSAGES[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )
    response_length = len(full_token_ids) - len(prompt_token_ids)
    loss_mask = torch.ones(response_length, device=device, dtype=torch.float32)

    num_samples = 20
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


def _generate_random_tokens(
    num_samples: int,
    min_total_len: int,
    max_total_len: int,
    vocab_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    lengths = torch.randint(min_total_len, max_total_len + 1, (num_samples,), device=device)
    return [torch.randint(1, vocab_size, (int(length.item()),), device=device, dtype=torch.long) for length in lengths]


def _generate_random_loss_masks(
    total_lengths: list[int],
    response_lengths: list[int],
    device: torch.device,
) -> list[torch.Tensor]:
    loss_masks = []
    for _total_len, response_len in zip(total_lengths, response_lengths, strict=True):
        mask = torch.randint(0, 2, (response_len,), device=device, dtype=torch.float32)
        if mask.sum() == 0:
            mask[0] = 1
        loss_masks.append(mask)
    return loss_masks


def _build_random_rollout_data(
    num_samples: int,
    min_total_len: int,
    max_total_len: int,
    min_response_len: int,
    max_response_len: int,
    vocab_size: int,
    device: torch.device,
) -> dict:
    tokens = _generate_random_tokens(num_samples, min_total_len, max_total_len, vocab_size, device)
    total_lengths = [t.size(0) for t in tokens]
    response_lengths = [
        int(torch.randint(min_response_len, min(max_response_len, total_len) + 1, (1,), device=device).item())
        for total_len in total_lengths
    ]
    loss_masks = _generate_random_loss_masks(total_lengths, response_lengths, device)
    return {
        "tokens": tokens,
        "loss_masks": loss_masks,
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
    }


def _build_random_long_rollout_data(device: torch.device) -> dict:
    return _build_random_rollout_data(
        num_samples=20,
        min_total_len=2048,
        max_total_len=4096,
        min_response_len=512,
        max_response_len=2048,
        vocab_size=256,
        device=device,
    )


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


def _compute_masked_loss(
    tokens: torch.Tensor,
    loss_mask: torch.Tensor,
    embed: torch.nn.Embedding,
    proj: torch.nn.Linear,
) -> torch.Tensor:
    logits = proj(embed(tokens))
    labels = tokens[:, 1:]
    logits = logits[:, :-1, :]
    vocab_size = logits.size(-1)

    loss = F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), reduction="none")
    loss = loss.view_as(labels)

    aligned_mask = loss_mask[:, :-1].float()
    masked_loss = loss * aligned_mask
    return masked_loss.sum() / torch.clamp_min(aligned_mask.sum(), 1)


def _compute_masked_token_sum(tokens: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    masked_tokens = tokens.float() * loss_mask.float()
    return masked_tokens.sum(), loss_mask.float().sum()


def _get_vocab_size(rollout_data: dict) -> int:
    max_token = max(int(t.max().item()) for t in rollout_data["tokens"])
    return max(max_token + 1, 2)


@pytest.mark.unit
@pytest.mark.parametrize("tp_size", [1, 8])
@pytest.mark.parametrize("rollout_builder", [_build_message_rollout_data, _build_random_long_rollout_data])
def test_packed_vs_padded_loss_matches(
    tp_size: int, rollout_builder: Callable[[torch.device], dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for get_batch (uses .cuda()).")

    device = torch.device("cuda")
    torch.manual_seed(0)

    pad_multiplier = 8
    pad_token_id = 0

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: tp_size)

    rollout_data = rollout_builder(device)
    data_iterator = DataIterator(rollout_data, micro_batch_size=len(rollout_data["tokens"]))

    packed_batch = get_batch(
        data_iterator,
        ["tokens", "loss_masks", "total_lengths", "response_lengths"],
        pad_multiplier=pad_multiplier,
        qkv_format="thd",
    )

    padded_tokens, padded_loss_masks = _build_padded_batch(
        rollout_data,
        pad_size=tp_size * pad_multiplier,
        pad_token_id=pad_token_id,
    )

    vocab_size = _get_vocab_size(rollout_data)
    embed = torch.nn.Embedding(vocab_size, 16, device=device)
    proj = torch.nn.Linear(16, vocab_size, device=device)
    embed.eval()
    proj.eval()

    packed_loss = _compute_masked_loss(packed_batch["tokens"], packed_batch["full_loss_masks"], embed, proj)
    padded_loss = _compute_masked_loss(padded_tokens, padded_loss_masks, embed, proj)

    torch.testing.assert_close(packed_loss, padded_loss, rtol=1e-5, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("tp_size", [1, 8])
@pytest.mark.parametrize("rollout_builder", [_build_message_rollout_data, _build_random_long_rollout_data])
def test_packed_vs_padded_loss_matches_with_cp(
    tp_size: int, rollout_builder: Callable[[torch.device], dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for get_batch (uses .cuda()).")

    device = torch.device("cuda")
    torch.manual_seed(0)

    cp_size = 2
    pad_multiplier = 8
    pad_token_id = 0

    rollout_data = rollout_builder(device)
    padded_tokens, padded_loss_masks = _build_padded_batch(
        rollout_data,
        pad_size=tp_size * pad_multiplier,
        pad_token_id=pad_token_id,
    )
    baseline_sum, baseline_weight = _compute_masked_token_sum(padded_tokens, padded_loss_masks)

    total_sum = torch.tensor(0.0, device=device)
    total_weight = torch.tensor(0.0, device=device)
    for cp_rank in range(cp_size):
        monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: cp_size)
        monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda cp_rank=cp_rank: cp_rank)
        monkeypatch.setattr(mpu, "get_tensor_model_parallel_world_size", lambda: tp_size)

        data_iterator = DataIterator(rollout_data, micro_batch_size=len(rollout_data["tokens"]))
        packed_batch = get_batch(
            data_iterator,
            ["tokens", "loss_masks", "total_lengths", "response_lengths"],
            pad_multiplier=pad_multiplier,
            qkv_format="thd",
        )

        loss_sum, weight = _compute_masked_token_sum(packed_batch["tokens"], packed_batch["full_loss_masks"])
        total_sum += loss_sum
        total_weight += weight

    torch.testing.assert_close(total_sum, baseline_sum, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(total_weight, baseline_weight, rtol=1e-5, atol=1e-6)
