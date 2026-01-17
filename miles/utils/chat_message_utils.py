# These are helper functions for think token lookup.
THINK_TOKEN_START = {
    "qwen3": ("<think>", 151667),
}
THINK_TOKEN_END = {
    "qwen3": ("</think>", 151668),
}


def get_think_token_start(model_name: str) -> tuple[str, int]:
    return THINK_TOKEN_START[model_name]


def get_think_token_end(model_name: str) -> tuple[str, int]:
    return THINK_TOKEN_END[model_name]


def trim_think_tokens(tokens: list[int], model_name: str) -> list[int]:
    start_index = None
    end_index = None
    for i in range(len(tokens)):
        if tokens[i] == get_think_token_start(model_name)[1]:
            if start_index is None:
                start_index = i
            else:
                raise ValueError("Multiple think token start found for trajectory.")
        if tokens[i] == get_think_token_end(model_name)[1]:
            if end_index is None:
                end_index = i + 1
            else:
                raise ValueError("Multiple think token end found for trajectory.")
    if start_index is None:
        if end_index is None:
            # No think tokens found, no strip.
            return tokens
        else:
            # This should not happen for OpenAI API format.
            raise ValueError(f"No think token start found for trajectory, but end token found for model {model_name}.")
    else:
        # Think part being truncated, so the end index is the last think token index.
        end_index = len(tokens)

    before = tokens[:start_index]
    after = tokens[end_index:]
    return before + after
