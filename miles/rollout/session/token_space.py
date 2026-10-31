"""Media on the TITO session path with ``--multimodal-token-space-processing``.

The chat template renders one placeholder per media item. SGLang's token-space processor expands each
placeholder to the item's tokens and returns the expanded prompt (``return_prompt_token_ids``). So a
session checkpoint has two forms:

* ``token_ids`` (TITO's form): one placeholder per media item. TITO renders, merges, checks prefixes
  and rolls back in this form, exactly as for text.
* the engine's form: the expanded prompt the engine returned for the checkpoint's turn, plus the
  completion. It is read from the turn's record (``engine_token_ids``), so it needs no state of its own.
  Requests, samples and R3 rows use this form.

A request sends the checkpoint in the engine's form and only the new part in TITO's form, with
``mm_token_expansion_start_len`` at the boundary (``engine_input_ids``), so SGLang expands only the
placeholders of the new media. The messages carry every media item of the conversation as a shared
path (``miles.utils.media_refs``); the trainer rebuilds the processor inputs from those paths.
"""

from typing import Any

from miles.rollout.session.errors import MessageValidationError, TokenizationError, UpstreamResponseError
from miles.rollout.session.types import SessionRecord
from miles.utils.media_refs import has_message_media, materialize_message_media

# Request fields only the session server sets.
MM_TOKEN_EXPANSION_START_LEN = "mm_token_expansion_start_len"


def store_message_media(client_args: dict[str, Any], *, token_space: bool, media_dir: str | None) -> dict[str, Any]:
    """The request with every media source of its messages replaced by a shared path.

    Without ``--multimodal-token-space-processing`` the session server carries no media: a request with
    media is a 400. Text requests pass through unchanged.
    """
    messages = client_args.get("messages")
    if not isinstance(messages, list) or not has_message_media(messages):
        return client_args
    if not token_space:
        raise MessageValidationError(
            "the session server carries images, audio or video only with --multimodal-token-space-processing"
        )
    try:
        return {**client_args, "messages": materialize_message_media(messages, media_dir)}
    except (TypeError, ValueError, OSError) as e:
        raise MessageValidationError(f"cannot store the request's media: {e}") from e


def engine_prompt_ids(record: SessionRecord) -> list[int]:
    """The prompt the engine ran for this turn: the expanded IDs it returned, else the IDs it was sent."""
    if MM_TOKEN_EXPANSION_START_LEN in record.request:
        return record.response["choices"][0]["prompt_token_ids"]
    return record.request["input_ids"]


def engine_token_ids(token_ids: list[int], record: SessionRecord | None) -> list[int]:
    """A checkpoint in the engine's form: ``token_ids`` (TITO's form) as committed by ``record``.

    A turn sent in token space gives its expanded prompt plus its completion; any other turn ran its IDs
    as sent, so both forms are ``token_ids``.
    """
    if record is None or MM_TOKEN_EXPANSION_START_LEN not in record.request:
        return token_ids
    output_ids = [item[1] for item in record.response["choices"][0]["meta_info"]["output_token_logprobs"]]
    return engine_prompt_ids(record) + output_ids


def engine_input_ids(
    prompt_token_ids: list[int],
    checkpoint_token_ids: list[int],
    checkpoint_engine_ids: list[int],
    checkpoint_completion_len: int,
) -> tuple[list[int], int]:
    """The request IDs in the engine's form, and the length of their expanded prefix.

    ``prompt_token_ids`` (TITO's form) extends the checkpoint, up to the few trailing tokens TITO may
    re-render. The part it shares with the checkpoint is replaced by the checkpoint's expanded IDs; the
    rest, with one placeholder per new media item, is left for SGLang to expand. The two forms of a
    checkpoint differ only in its prompt, so this holds when the shared part covers the prompt: the
    unshared tokens are then completion tokens, the same in both forms. A request that keeps less
    raises.
    """
    shared = lcp_len(checkpoint_token_ids, prompt_token_ids)
    unshared = len(checkpoint_token_ids) - shared
    if unshared > checkpoint_completion_len:
        raise TokenizationError(
            f"the new prompt reuses {shared} of the checkpoint's {len(checkpoint_token_ids)} tokens, which "
            f"cuts into its prompt; only its {checkpoint_completion_len} completion tokens may be re-rendered"
        )
    start = len(checkpoint_engine_ids) - unshared
    return checkpoint_engine_ids[:start] + prompt_token_ids[shared:], start


def prepare_engine_request(
    request_body: dict[str, Any],
    *,
    checkpoint_token_ids: list[int],
    checkpoint_record: SessionRecord | None,
    token_space: bool,
) -> None:
    """Rewrite a request with media into the engine's form, in place.

    ``request_body["input_ids"]`` holds TITO's form on entry. The checkpoint (``checkpoint_token_ids``,
    committed by ``checkpoint_record``; empty and None for a new trajectory) is sent expanded, and SGLang
    expands the placeholders after it and returns the expanded prompt. A text request is left as is.
    """
    if not (token_space and has_message_media(request_body.get("messages") or [])):
        return
    completion_len = (
        len(checkpoint_record.response["choices"][0]["meta_info"]["output_token_logprobs"])
        if checkpoint_record is not None
        else 0
    )
    input_ids, start_len = engine_input_ids(
        request_body["input_ids"],
        checkpoint_token_ids,
        engine_token_ids(checkpoint_token_ids, checkpoint_record),
        completion_len,
    )
    request_body["input_ids"] = input_ids
    request_body[MM_TOKEN_EXPANSION_START_LEN] = start_len
    request_body["return_prompt_token_ids"] = True


def check_engine_prompt(request_body: dict[str, Any], choice: dict[str, Any]) -> None:
    """The engine must return its prompt and keep the expanded prefix it was sent."""
    if MM_TOKEN_EXPANSION_START_LEN not in request_body:
        return
    returned = choice.get("prompt_token_ids")
    start = request_body[MM_TOKEN_EXPANSION_START_LEN]
    if not isinstance(returned, list) or returned[:start] != request_body["input_ids"][:start]:
        raise UpstreamResponseError(
            "the engine did not return the expanded prompt it ran, starting with the "
            f"{start} expanded tokens it was sent; it needs --enable-token-space-processor"
        )


def lcp_len(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two token-ID lists."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n
