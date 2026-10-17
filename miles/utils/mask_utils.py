from transformers import AutoTokenizer


def get_response_lengths(loss_masks: list[list[int]]) -> list[int]:
    # return the lengths starting from the first occurrence of 1 to the end of each loss mask
    return [len(mask[mask.index(1) :]) if 1 in mask else 0 for mask in loss_masks]


class MultiTurnLossMaskGenerator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        tokenizer_type: str = "qwen",
        chat_template_kwargs: dict | None = None,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_type = tokenizer_type
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        if tokenizer_type == "glm5":
            role_tokens = {
                "<|system|>": "system",
                "<|user|>": "user",
                "<|assistant|>": "assistant",
                "<|observation|>": "tool",
            }
            vocab = tokenizer.get_added_vocab()
            if missing := role_tokens.keys() - vocab.keys():
                raise ValueError(f"GLM loss masks require role tokens in the tokenizer: {sorted(missing)}")
            self._glm_role_token_ids = {vocab[token]: role for token, role in role_tokens.items()}
        else:
            self.system_message_length, self.gen_token_length = self.get_system_message_length()

    def get_response_lengths(self, loss_masks: list[list[int]]) -> list[int]:
        return get_response_lengths(loss_masks)

    def find_all_sublist_indices(self, main_list, sublist):
        sublist_len = len(sublist)
        indices = []
        for i in range(len(main_list) - sublist_len + 1):
            if main_list[i : i + sublist_len] == sublist:
                indices.append(i)
        return indices

    def get_system_message_length(self) -> tuple[int, int]:
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
        idx_1, idx_2 = self.find_all_sublist_indices(chat_template_token_ids, raw_token_ids)
        end_interval = len(chat_template_token_ids) - len(raw_token_ids) - idx_2
        gen_token_length = len(
            self.tokenizer.apply_chat_template(
                test_messages, add_special_tokens=False, tokenize=True, return_dict=False, add_generation_prompt=True
            )
        ) - len(chat_template_token_ids)

        system_message_length = idx_1 - ((idx_2 - idx_1) - end_interval - len(raw_token_ids))
        return system_message_length, gen_token_length

    def gen_multi_turn_loss_mask_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
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

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_qwen3(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
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

            if message.get("step_loss_mask", 1) != 1:
                loss_mask = [0] * len(message_ids)

            all_loss_masks.extend(loss_mask)
            all_token_ids.extend(message_ids)

        return all_token_ids, all_loss_masks

    def gen_multi_turn_loss_mask_distill_qwen(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
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

    def gen_multi_turn_loss_mask_glm5(
        self, messages: list[dict], tools: list[dict] = None
    ) -> tuple[list[int], list[int]]:
        # Defer SGLang template dependencies for tokenizer-only loss mask users.
        from miles.utils.chat_template_utils import apply_chat_template

        # GLM templates render reasoning and tool turns using the full conversation.
        token_ids = apply_chat_template(
            messages,
            tokenizer=self.tokenizer,
            tools=tools,
            tokenize=True,
            add_generation_prompt=False,
            **self.chat_template_kwargs,
        )
        boundaries = [
            (index, self._glm_role_token_ids[token])
            for index, token in enumerate(token_ids)
            if token in self._glm_role_token_ids
        ]
        expected_roles = []
        for message in messages:
            role = message["role"]
            if role not in self._glm_role_token_ids.values():
                raise ValueError(f"Unsupported GLM SFT message role: {role}")
            if role != "tool" or not expected_roles or expected_roles[-1] != "tool":
                expected_roles.append(role)
        observed_roles = [role for _, role in boundaries]
        prefix_length = len(observed_roles) - len(expected_roles)
        if (
            prefix_length < 0
            or any(role != "system" for role in observed_roles[:prefix_length])
            or observed_roles[prefix_length:] != expected_roles
        ):
            raise ValueError(
                "GLM SFT role markers do not match conversation messages; "
                f"expected={expected_roles}, observed={observed_roles}"
            )

        assistant_messages = [message for message in messages if message["role"] == "assistant"]
        assistant_boundaries = [index for index, role in boundaries if role == "assistant"]
        if len(assistant_boundaries) != len(assistant_messages):
            raise ValueError("GLM SFT assistant marker count does not match assistant messages")

        loss_mask = [0] * len(token_ids)
        assistant_index = 0
        for boundary_index, (start, role) in enumerate(boundaries):
            end = boundaries[boundary_index + 1][0] if boundary_index + 1 < len(boundaries) else len(token_ids)
            if role == "assistant":
                if assistant_messages[assistant_index].get("step_loss_mask", 1) == 1:
                    loss_mask[start + 1 : end] = [1] * (end - start - 1)
                assistant_index += 1
        return token_ids, loss_mask

    def get_loss_mask(self, messages: list[dict], tools: list[dict] = None) -> tuple[list[int], list[int]]:
        if self.tokenizer_type == "qwen":
            if "<｜Assistant｜>" in self.tokenizer.get_added_vocab():
                return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)

            return self.gen_multi_turn_loss_mask_qwen(messages, tools)
        elif self.tokenizer_type == "qwen3":
            return self.gen_multi_turn_loss_mask_qwen3(messages, tools)
        elif self.tokenizer_type == "distill_qwen":
            return self.gen_multi_turn_loss_mask_distill_qwen(messages, tools)
        elif self.tokenizer_type == "glm5":
            return self.gen_multi_turn_loss_mask_glm5(messages, tools)
        else:
            raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")

    def get_loss_mask_with_multimodal_alignment(
        self,
        messages: list[dict],
        input_ids: list[int],
        tools: list[dict] = None,
        multimodal_token_ids: set[int] | None = None,
    ) -> tuple[list[int], list[int]]:
        text_token_ids, text_loss_mask = self.get_loss_mask(messages, tools=tools)
        multimodal_token_ids = multimodal_token_ids or set()
        if text_token_ids == input_ids:
            return input_ids, [
                0 if token in multimodal_token_ids else mask
                for token, mask in zip(input_ids, text_loss_mask, strict=True)
            ]

        aligned_mask = []
        text_idx = input_idx = 0
        while text_idx < len(text_token_ids) and input_idx < len(input_ids):
            text_token = text_token_ids[text_idx]
            input_token = input_ids[input_idx]
            if text_token != input_token:
                raise ValueError(
                    "Processor tokenization differs from tokenizer tokenization outside a multimodal token run: "
                    f"text_index={text_idx}, input_index={input_idx}, "
                    f"text_token={text_token}, input_token={input_token}"
                )

            if text_token not in multimodal_token_ids:
                aligned_mask.append(text_loss_mask[text_idx])
                text_idx += 1
                input_idx += 1
                continue

            text_run_end = text_idx
            while text_run_end < len(text_token_ids) and text_token_ids[text_run_end] == text_token:
                text_run_end += 1
            input_run_end = input_idx
            while input_run_end < len(input_ids) and input_ids[input_run_end] == input_token:
                input_run_end += 1
            aligned_mask.extend([0] * (input_run_end - input_idx))
            text_idx = text_run_end
            input_idx = input_run_end

        if text_idx != len(text_token_ids) or input_idx != len(input_ids):
            raise ValueError(
                "Processor and tokenizer produced incompatible sequence lengths after multimodal alignment: "
                f"text={len(text_token_ids)}, processor={len(input_ids)}"
            )

        return input_ids, aligned_mask

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
