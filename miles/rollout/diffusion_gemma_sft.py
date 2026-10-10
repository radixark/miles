"""Offline text conversations adapted to Miles' existing fixed-data SFT contract."""

from uuid import uuid4

from miles.utils.processing_utils import load_tokenizer


def tokenize_final_response(tokenizer, *, messages: list[dict], template_kwargs: dict) -> tuple[list[int], int]:
    """Keep checkpoint chat rendering; locate the final answer without Qwen masks.

    A sentinel rendered through the same template identifies the assistant body
    boundary without assuming Gemma headers or relying on generation tags being
    present. Token offsets reject merges across that boundary rather than silently
    supervising part of the assistant header.
    """
    if not isinstance(messages, list) or len(messages) < 2 or messages[-1].get("role") != "assistant":
        raise ValueError("offline SFT requires a conversation ending in an assistant answer")
    for message in messages:
        if (
            message.get("role") not in ("system", "user", "assistant")
            or not isinstance(message.get("content"), str)
            or message.get("tool_calls")
        ):
            raise ValueError("DiffusionGemma offline SFT currently supports plain text conversations without tools")
    if not messages[-1]["content"].strip() or messages[-1].get("step_loss_mask", 1) != 1:
        raise ValueError("the final assistant answer must be nonempty and supervised")
    if template_kwargs.get("add_generation_prompt") or template_kwargs.get("continue_final_message"):
        raise ValueError("SFT renders completed turns; generation/continuation template flags are unsupported")
    sentinel = f"MILES_SFT_BODY_{uuid4().hex}"
    probe_messages = [*messages[:-1], {**messages[-1], "content": sentinel}]
    probe = tokenizer.apply_chat_template(probe_messages, tokenize=False, **template_kwargs)
    if probe.count(sentinel) != 1:
        raise ValueError("chat template must render the assistant content exactly once")
    prefix, suffix = probe.split(sentinel)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
    if not rendered.startswith(prefix) or not rendered.endswith(suffix):
        raise ValueError("chat template changes the assistant header based on its content; provide a stable template")
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    boundary = len(prefix)
    if any(start < boundary < end for start, end in offsets):
        raise ValueError("token crosses assistant body boundary; provide a template with a token-aligned header")
    answer_start = next((i for i, (start, end) in enumerate(offsets) if start >= boundary and end > start), None)
    if answer_start is None or answer_start == 0:
        raise ValueError("chat template must produce a nonempty tokenized prompt and final response")
    return encoded["input_ids"], len(encoded["input_ids"]) - answer_start


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Populate fixed labels; this adapter performs no model generation or scoring."""
    if evaluation or not args.rollout_global_dataset or not args.debug_train_only:
        raise ValueError("DiffusionGemma SFT adapter requires offline global-dataset training without rollout eval")
    tokenizer = load_tokenizer(args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True)
    groups = data_buffer.get_samples(args.rollout_batch_size)
    for group in groups:
        if len(group) != 1:
            raise ValueError("offline SFT requires one labeled sample per prompt")
        sample = group[0]
        multimodal_values = (
            value
            for inputs in (sample.multimodal_inputs, sample.multimodal_train_inputs)
            for value in (inputs or {}).values()
        )
        if any(value is not None for value in multimodal_values) or sample.metadata.get("tools"):
            raise ValueError("DiffusionGemma offline SFT currently supports text-only samples without tools")
        tokens, response_length = tokenize_final_response(
            tokenizer,
            messages=sample.prompt,
            template_kwargs=args.apply_chat_template_kwargs or {},
        )
        sample.tokens = tokens
        sample.response_length = response_length
        sample.loss_mask = [1] * response_length
        sample.reward = 0
    return groups
