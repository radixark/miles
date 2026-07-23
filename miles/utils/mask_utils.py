"""Loss-mask strategies for SFT training.

Follows the same ABC + registry + dispatcher pattern as
``miles/utils/tracking_utils/base.py`` (TrackingBackend / BACKEND_REGISTRY /
TrackingManager).  Strategies may hold per-run cached state (e.g.
system_message_length) computed once at construction and reused across samples.
"""

from abc import ABC, abstractmethod
from typing import Any

from transformers import AutoTokenizer

from miles.utils.misc import load_function


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    # return the lengths starting from the first occurrence of 1 to the end of each loss mask
    return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]


LOSS_MASK_REGISTRY: dict[str, type["LossMaskStrategy"]] = {}


def register_loss_mask(name: str):
    """Register a loss mask strategy under one or more names.

    Examples:
        @register_loss_mask("my_custom")
        class MyCustomStrategy(LossMaskStrategy):
            ...
    """
    if not isinstance(name, str):
        raise TypeError(f"Loss mask strategy name must be a string, got {name}")

    def decorator(cls: Any) -> type["LossMaskStrategy"]:
        if not isinstance(cls, type) or not issubclass(cls, LossMaskStrategy):
            raise TypeError(f"Only LossMaskStrategy subclasses can be registered, got {cls}")
        # protect from cases of module reload
        if name in LOSS_MASK_REGISTRY and LOSS_MASK_REGISTRY[name] is not cls:
            raise ValueError(f"Loss mask strategy {name!r} is already registered by a different class")
        LOSS_MASK_REGISTRY[name] = cls
        return cls

    return decorator


class LossMaskStrategy(ABC):
    """Contract for generating a loss mask for a chat-formatted conversation."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def get_loss_mask(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], list[int]]:
        """Return (token_ids, loss_mask) for the given messages.

        The loss mask should have the same length as token_ids and contain 0/1
        values indicating which tokens participate in loss calculation.
        """
        ...


class BasePerMessageLossMaskStrategy(LossMaskStrategy):
    """Shared machinery for strategies that build masks message-by-message."""

    def __init__(self, tokenizer: AutoTokenizer):
        super().__init__(tokenizer)
        self.system_message_length, self.gen_token_length = self._get_system_message_length()

    @staticmethod
    def _find_all_sublist_indices(main_list, sublist):
        sublist_len = len(sublist)
        indices = []
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        return indices

    def _get_system_message_length(self) -> tuple[int, int]:
        test_string = "FOR TESTING ONLY"
        test_messages = [
            {"role": "user", "content": test_string},
            {"role": "user", "content": test_string},
        ]
        raw_token_ids = self.tokenizer(test_string, add_special_tokens=False)["input_ids"]
        chat_template_token = self.tokenizer.apply_chat_template(
            test_messages, add_special_tokens=False, tokenize=False
        )
        chat_template_token_ids = self.tokenizer(chat_template_token, add_special_tokens=False)["input_ids"]
        indices = self._find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        if len(indices) != 2:
            raise ValueError(
                f"Expected to find raw token IDs exactly twice in the chat template, "
                f"but found {len(indices)} occurrences. This can happen if the chat template "
                f"or tokenizer behavior is incompatible with the system message length detection."
            )
        idx_1, idx_2 = indices[0], indices[1]
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages, add_special_tokens=False, tokenize=True, return_dict=False, add_generation_prompt=True
            )
        ) - len(chat_template_token_ids)

        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        return system_message_length, gen_token_length

    def _apply_step_loss_mask(self, message: dict, loss_mask: list[int]) -> list[int]:
        if message.get("step_loss_mask", 1) != 1:
            return [0] * len(loss_mask)
        return loss_mask


@register_loss_mask("qwen")
class QwenLossMaskStrategy(BasePerMessageLossMaskStrategy):
    def get_loss_mask(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        for i, message in enumerate(messages):
            if i == 0:
                message_ids = self.tokenizer.apply_chat_template(
                    [message], tokenize=True, return_dict=False, tools=tools
                )
            else:
                message_ids = self.tokenizer.apply_chat_template([message], tokenize=True, return_dict=False)

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            loss_mask = self._apply_step_loss_mask(message, loss_mask)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks


@register_loss_mask("qwen3")
class Qwen3LossMaskStrategy(BasePerMessageLossMaskStrategy):
    def get_loss_mask(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], list[int]]:
        all_loss_masks = []
        all_token_ids = []

        prefix_message = {"role": "user", "content": "FOR CALCULATING LOSS MASK ONLY"}
        prefix_token_ids = self.tokenizer.apply_chat_template([prefix_message], tokenize=True, return_dict=False)

        for i, message in enumerate(messages):
            if i == 0:
                tailed_message_ids = self.tokenizer.apply_chat_template(
                    [message, prefix_message], tokenize=True, return_dict=False, tools=tools
                )
                message_ids = tailed_message_ids[: -len(prefix_token_ids)]
            else:
                prefixed_message_ids = self.tokenizer.apply_chat_template(
                    [prefix_message, message], tokenize=True, return_dict=False
                )
                message_ids = prefixed_message_ids[len(prefix_token_ids) :]

            if message["role"] != "system" and i > 0:
                message_ids = message_ids[self.system_message_length :]

            if message["role"] == "assistant":
                loss_mask = [0] * self.gen_token_length + [1] * (len(message_ids) - self.gen_token_length)
            else:
                loss_mask = [0] * len(message_ids)

            loss_mask = self._apply_step_loss_mask(message, loss_mask)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks


@register_loss_mask("distill_qwen")
class DistillQwenLossMaskStrategy(LossMaskStrategy):
    def get_loss_mask(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], list[int]]:
        prompt = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True, tools=tools
        )
        response = messages[-1]["content"]
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_tokens = self.tokenizer(response, add_special_tokens=False)["input_ids"]

        response_length = len(response_tokens)
        token_ids = prompt_tokens + response_tokens
        loss_mask = [0] * len(prompt_tokens) + [1] * response_length

        if messages[-1].get("step_loss_mask", 1) != 1:
            loss_mask = [0] * len(token_ids)
        return token_ids, loss_mask


class MultiTurnLossMaskGenerator:
    """Dispatcher that selects and delegates to a LossMaskStrategy."""

    def __init__(self, tokenizer: AutoTokenizer, tokenizer_type: str = "qwen"):
        self.tokenizer = tokenizer
        self.loss_mask_type = tokenizer_type
        self.tokenizer_type = tokenizer_type  # historical alias
        self.strategy = self._resolve_strategy(tokenizer, tokenizer_type)

    def _resolve_strategy(self, tokenizer: AutoTokenizer, strategy_name: str) -> LossMaskStrategy:
        # Auto-detect distillation variants for the qwen family.
        if strategy_name == "qwen" and "<｜Assistant｜>" in tokenizer.get_added_vocab():
            strategy_name = "distill_qwen"

        if strategy_name in LOSS_MASK_REGISTRY:
            return LOSS_MASK_REGISTRY[strategy_name](tokenizer)

        # Allow fully-qualified class paths for custom strategies without registration.
        if "." in strategy_name:
            try:
                cls = load_function(strategy_name)
                if not isinstance(cls, type) or not issubclass(cls, LossMaskStrategy):
                    raise TypeError(f"{strategy_name} is not a LossMaskStrategy subclass")
                return cls(tokenizer)
            except (ImportError, AttributeError, TypeError) as exc:
                raise ValueError(
                    f"Unable to load loss mask strategy {strategy_name!r}. "
                    f"Ensure it is a registered name or a fully-qualified LossMaskStrategy subclass."
                ) from exc

        raise ValueError(
            f"Unsupported loss mask type: {strategy_name!r}. " f"Registered types: {sorted(LOSS_MASK_REGISTRY)}"
        )

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        return get_response_lengths(loss_masks)

    def get_loss_mask(self, messages: list[dict], tools: list[dict] | None = None) -> tuple[list[int], list[int]]:
        return self.strategy.get_loss_mask(messages, tools=tools)

    def get_loss_mask_with_multimodal_alignment(
        self, messages: list[dict], input_ids: list[int], tools: list[dict] | None = None
    ) -> tuple[list[int], list[int]]:
        text = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                text_parts = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                text.append({"role": msg["role"], "content": " ".join(text_parts)})
            else:
                text.append(msg)

        _, loss_mask_text = self.get_loss_mask(text, tools=tools)

        diff = len(input_ids) - len(loss_mask_text)
        assert diff >= 0, (
            f"input_ids (length={len(input_ids)}) is shorter than text loss_mask (length={len(loss_mask_text)}) "
            f"Please check if processor and tokenizer tokenization are consistent."
        )
        loss_mask = [0] * diff + loss_mask_text

        return input_ids, loss_mask

    def get_text_from_loss_mask(self, token_ids: list[int], loss_masks: list[int]) -> list[str]:
        selected_texts = []
        current_tokens = []

        for idx, mask in enumerate(loss_masks):
            if mask == 1:
                current_tokens.append(token_ids[idx])
            elif current_tokens:
                selected_texts.append(self.tokenizer.decode(current_tokens))
                current_tokens = []

        if current_tokens:
            selected_texts.append(self.tokenizer.decode(current_tokens))

        return selected_texts
