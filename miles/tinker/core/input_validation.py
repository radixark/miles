import math
import re

from miles.tinker.core.types import (
    LOSS_FN_INPUTS,
    LOSS_INPUT_KEYS,
    CommandOp,
    GatewayConfig,
    ModelRecord,
    UserInputError,
)


def validate_model_config(lora_config: dict, config: GatewayConfig) -> None:
    """Reject per-model settings that conflict with the fixed server adapter layout."""
    if lora_config.get("seed") is not None:
        raise UserInputError("lora_config.seed is not supported: adapter initialization is not per-model seedable")
    layout = {
        "train_attn": config.trains_attn,
        "train_mlp": config.trains_mlp,
        "train_unembed": config.trains_unembed,
    }
    for field, layout_trains in layout.items():
        requested = lora_config.get(field)
        if requested is not None and requested != layout_trains:
            raise UserInputError(
                f"lora_config.{field}={requested} conflicts with this gateway's adapter layout "
                f"({field}={layout_trains}); the layout is fixed by --tinker-train-attn/mlp/unembed at server start"
            )
    rank = lora_config.get("rank", 32)
    if type(rank) is not int or not 1 <= rank <= config.max_lora_rank:
        raise UserInputError(f"lora_config.rank must be an integer in [1, {config.max_lora_rank}], got {rank!r}")


def validate_batch_payload(op: CommandOp, payload: dict, config: GatewayConfig) -> None:
    if not op.is_batch():
        return
    datums = payload["datums"]
    if not datums:
        raise UserInputError("forward_backward with no data")
    if len(datums) > config.max_datums_per_request:
        raise UserInputError(f"{len(datums)} datums exceeds max_datums_per_request={config.max_datums_per_request}")
    required_inputs = LOSS_FN_INPUTS.get(payload["loss_fn"])
    if required_inputs is None:
        raise UserInputError(f"unknown loss_fn {payload['loss_fn']!r}; known: {sorted(LOSS_FN_INPUTS)}")
    loss_fn_config = payload.get("loss_fn_config")
    if loss_fn_config is not None:
        if not isinstance(loss_fn_config, dict):
            raise UserInputError("loss_fn_config must be an object")
        config_keys = {
            "ppo": ("clip_low_threshold", "clip_high_threshold"),
            "cispo": ("clip_low_threshold", "clip_high_threshold"),
            "dro": ("beta",),
        }.get(payload["loss_fn"], ())
        for key in config_keys:
            if key in loss_fn_config:
                value = loss_fn_config[key]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise UserInputError(f"loss_fn_config[{key!r}] must be a finite number")
    total_tokens = 0
    for index, datum in enumerate(datums):
        validate_token_ids(datum["tokens"], config.vocab_size, f"datum {index}: model_input")
        validate_token_ids(datum["target_tokens"], config.vocab_size, f"datum {index}: target_tokens")
        if len(datum["tokens"]) > config.max_tokens_per_datum:
            raise UserInputError(f"datum {index}: {len(datum['tokens'])} tokens exceeds {config.max_tokens_per_datum}")
        total_tokens += len(datum["tokens"])
        for wire_key in required_inputs:
            values = datum.get(LOSS_INPUT_KEYS[wire_key])
            if values is None:
                raise UserInputError(
                    f"datum {index}: loss_fn {payload['loss_fn']!r} needs loss_fn_inputs[{wire_key!r}]"
                )
            if len(values) != datum["target_len"]:
                raise UserInputError(
                    f"datum {index}: loss_fn_inputs[{wire_key!r}] has {len(values)} values "
                    f"for {datum['target_len']} target tokens"
                )
        unread = [
            wire_key
            for wire_key, datum_key in LOSS_INPUT_KEYS.items()
            if wire_key not in required_inputs and datum_key in datum
        ]
        if unread:
            raise UserInputError(
                f"datum {index}: loss_fn {payload['loss_fn']!r} does not read loss_fn_inputs {unread}"
            )
    if total_tokens > config.max_tokens_per_request:
        raise UserInputError(f"{total_tokens} tokens exceeds max_tokens_per_request={config.max_tokens_per_request}")


def validate_sample_payload(payload: dict, config: GatewayConfig) -> None:
    validate_token_ids(payload["prompt_tokens"], config.vocab_size, "prompt")
    num_samples = payload.get("num_samples", 1)
    if type(num_samples) is not int or not 1 <= num_samples <= config.max_samples_per_request:
        raise UserInputError(f"num_samples must be an integer in [1, {config.max_samples_per_request}]")
    topk = payload.get("topk_prompt_logprobs", 0)
    if type(topk) is not int or topk < 0:
        raise UserInputError("topk_prompt_logprobs must be a nonnegative integer")


def validate_seq_id(value, name: str, minimum: int = 1) -> int:
    # request seq_ids are 1-based (last_enqueued_seq_id starts at 0); idempotency keys are 0-based
    if not isinstance(value, int) or value < minimum:
        raise UserInputError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def validate_checkpoint_segment(segment: str) -> None:
    """Reject client path segments that could escape the checkpoint root."""
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", segment) is None:
        raise UserInputError(f"invalid checkpoint path segment {segment!r}")


def validate_checkpoint_compatibility(meta: dict, record: ModelRecord, config: GatewayConfig, shown_path: str) -> None:
    """Reject settings that would change the saved tensors' meaning."""
    expected = {
        "base_model": record.base_model,
        "lora_rank": record.lora_rank,
        "lora_alpha": record.lora_alpha,
        "train_attn": config.trains_attn,
        "train_mlp": config.trains_mlp,
        "train_unembed": config.trains_unembed,
    }
    for key, value in expected.items():
        if meta[key] != value:
            raise UserInputError(
                f"checkpoint {shown_path!r} was saved with {key}={meta[key]!r}; this model expects {key}={value!r}"
            )


def validate_save_options(payload: dict) -> None:
    if payload.get("ttl_seconds") is not None:
        raise UserInputError("ttl_seconds is not supported: checkpoints on this gateway do not expire")
    if payload.get("user_metadata") is not None:
        raise UserInputError("user_metadata is not supported by this gateway")


def validate_token_ids(tokens: list, vocab_size: int, field: str) -> None:
    if any(type(token) is not int or not 0 <= token < vocab_size for token in tokens):
        raise UserInputError(f"{field} must contain integer token IDs in [0, {vocab_size})")


def validate_checkpoint_metadata(meta, shown_path: str) -> None:
    fields = {
        "tenant_digest": (str,),
        "base_model": (str,),
        "lora_rank": (int,),
        "lora_alpha": (int, float),
        "train_attn": (bool,),
        "train_mlp": (bool,),
        "train_unembed": (bool,),
    }
    if not isinstance(meta, dict) or any(type(meta.get(key)) not in types for key, types in fields.items()):
        raise UserInputError(f"checkpoint {shown_path!r} has invalid or unsupported metadata")
    if meta["lora_rank"] <= 0 or not math.isfinite(meta["lora_alpha"]):
        raise UserInputError(f"checkpoint {shown_path!r} has invalid LoRA metadata")
