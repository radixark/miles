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


def calc_last_think_part_index(tokens: list[int], model_name: str) -> tuple[int | None, int | None]:
    start_index = None
    end_index = None
    for i in range(len(tokens)):
        if tokens[i] == get_think_token_start(model_name)[1]:
            start_index = i

    if start_index is None:
        # No think tokens found, no strip.
        return None, None

    for i in range(start_index + 1, len(tokens)):
        if tokens[i] == get_think_token_end(model_name)[1]:
            end_index = i

    # If think part being truncated, end_index would be None.
    return start_index, end_index


def check_is_truncated_message(tokens: list[int], model_name: str) -> bool:
    # TODO: handle this later.
    pass
