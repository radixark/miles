"""
Utils to integrate SGLang's `/generate` endpoint with RL things like Sample.
"""

from copy import deepcopy
from typing import Any

import numpy as np
import pybase64

from miles.rollout.generate_utils.rollout_topk_logprobs import (
    append_rollout_topk_logprobs,
    configure_rollout_topk_logprobs_request,
)
from miles.rollout.generate_utils.sampling_mask import (
    append_sampling_metadata,
    read_sampled_logprobs,
    should_return_sampling_mask,
)
from miles.router.backend import make_router_backend
from miles.utils.lora.utils import LORA_ADAPTER_NAME, lora_rollout_enabled
from miles.utils.output_store.plan import (
    FIELDS,
    OUTPUT_STORE_REF_KEY,
    OUTPUT_STORE_REQUEST_KEY,
    StorePlan,
    StorePlanError,
    store_ref_from_dict,
)
from miles.utils.processing_utils import encode_image_for_rollout_engine, extract_multimodal_train_inputs
from miles.utils.types import Sample


def image_pad_id_for(state) -> int | None:
    """The id of the image placeholder the engine expands, or None when no
    placeholder applies. The TITO family's image-token table decides first
    (``TITOTokenizer.image_tokens``: a family whose placeholder the engine
    does not expand collapses nothing); without a TITO family, the HF vision
    processor names the token (``processor.image_token``); a text model has
    neither."""
    tito = getattr(state, "tito_tokenizer", None)  # config-access-exempt: only a session-path state has one
    tokens = tito.image_tokens if tito is not None else None
    if tokens is not None:
        if not tokens.expands:
            return None
        return int(tito.image_token_ids()["placeholder"])
    processor = state.processor
    token = (
        getattr(processor, "image_token", None) if processor is not None else None
    )  # config-access-exempt: only a vision processor names an image token
    if not token:
        return None
    pad = state.tokenizer.convert_tokens_to_ids(token)
    unk = state.tokenizer.unk_token_id
    if not isinstance(pad, int) or pad == unk:
        return None
    return int(pad)


def collapse_image_pads(input_ids: list[int], image_pad_id: int) -> list[int]:
    """``input_ids`` with every run of image placeholders reduced to one.

    An HF vision processor writes the placeholder once per image patch group
    (Qwen3-VL: 336 for a 1024x576 screenshot, the model's view). SGLang's
    exact-token path (sgl-project/sglang#26555) expects one placeholder per
    image in the request and expands it to the same count itself; a request
    that already carries the expansion falls back to decoding and
    re-tokenizing the prompt, so the engine runs ids the training record does
    not hold (60 to 85 percent of screenshot turns differed by 1 to 2 ids on
    Qwen3.8-27B-VL). The training record keeps the expanded ids; only the
    wire form collapses."""
    out: list[int] = []
    previous_pad = False
    for token in input_ids:
        is_pad = token == image_pad_id
        if is_pad and previous_pad:
            continue
        out.append(token)
        previous_pad = is_pad
    return out


# Make this an isolated function because users may want to compute their own
def compute_prompt_ids_from_sample(state, sample, tools=None):
    prompt = sample.prompt

    if state.processor and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        processor_output = state.processor(text=prompt, **sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]

        # TODO shall we move it to other places? then can make this function immutable
        sample.multimodal_train_inputs = extract_multimodal_train_inputs(processor_output)

        return prompt_ids
    else:
        if not isinstance(prompt, str):
            prompt = state.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, tools=tools
            )

        return state.tokenizer.encode(prompt, add_special_tokens=False)


def router_routes_by_key(args) -> bool:
    return make_router_backend(args).routes_by_key(args.sglang.models)


def compute_routing_headers(args, sample: Sample) -> dict[str, str]:
    return make_router_backend(args).routing_headers(sample.routing_key, args.sglang.models)


def compute_request_payload(
    args,
    input_ids: list[int],
    sampling_params: dict,
    multimodal_inputs: dict | None = None,
    *,
    evaluation: bool = False,
    kv_cache_namespace: str | None = None,
    image_pad_id: int | None = None,
) -> tuple[dict[str, Any] | None, Sample.Status | None]:
    """The ``/generate`` request for ``input_ids`` (the model's view: the
    image placeholders already expanded by the processor). With images the
    request carries one placeholder per image (``collapse_image_pads``,
    ``image_pad_id`` from ``image_pad_id_for``); images without a known
    placeholder are refused rather than sent expanded, which the engine would
    re-tokenize. The context arithmetic uses the expanded length, which is
    what the engine runs."""
    sampling_params = deepcopy(sampling_params)
    max_new_tokens = sampling_params.pop("max_new_tokens", args.rollout_max_response_len)
    if x := args.rollout_max_context_len:
        max_new_tokens = min(max_new_tokens, x - len(input_ids))
    if max_new_tokens <= 0:
        return None, Sample.Status.TRUNCATED

    return_sampling_mask = should_return_sampling_mask(args, sampling_params, evaluation=evaluation)

    payload = {
        "input_ids": input_ids,
        "sampling_params": {**sampling_params, "max_new_tokens": max_new_tokens},
        "return_logprob": True,
        "return_routed_experts": args.use_rollout_routing_replay,
        "return_indexer_topk": args.use_rollout_indexer_replay,
    }
    if return_sampling_mask:
        payload["return_sampling_mask"] = True
    if lora_rollout_enabled(args):
        payload["lora_path"] = LORA_ADAPTER_NAME
    if image_data := (multimodal_inputs or {}).get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]
        if image_pad_id is None:
            raise ValueError(
                "the sample carries images but no image placeholder is known for this model: the TITO "
                "family has no entry in the image-token table (ImageTokens, "
                "miles.utils.chat_template_utils.tito_tokenizer) and the processor names no image_token; "
                "the engine would re-tokenize the prompt"
            )
        payload["input_ids"] = collapse_image_pads(input_ids, image_pad_id)
    if kv_cache_namespace is not None:
        payload["extra_key"] = kv_cache_namespace

    if not evaluation:
        configure_rollout_topk_logprobs_request(args, payload)
    if args.use_output_store and not evaluation and any(payload.get(field.request_key) for field in FIELDS):
        if "image_data" in payload:
            # A media prompt's routed experts have one row per expanded media token, not per prompt token.
            raise StorePlanError("--use-output-store does not serve multimodal prompts")
        # TODO: sweep the run's unreferenced objects periodically; a retried request orphans the one an earlier
        # attempt stored.
        payload[OUTPUT_STORE_REQUEST_KEY] = True
    return payload, None


async def update_sample_from_response(
    args,
    sample: Sample,
    payload: dict,
    output: dict,
    update_loss_mask: bool = False,
    prompt_ids: list[int] | None = None,
):
    """``prompt_ids`` is the model's view of the prompt (image placeholders
    expanded); it defaults to the request's ``input_ids``, which differ only
    when the request collapsed the placeholders (``compute_request_payload``)."""
    # Initialize sample.tokens for the first turn
    if (len(sample.response) == 0) and not sample.tokens:
        sample.tokens = list(prompt_ids) if prompt_ids is not None else payload["input_ids"]

    if x := output["meta_info"].get("output_token_logprobs"):
        new_response_tokens = [item[1] for item in x]
        new_response_log_probs = [item[0] for item in x]
    else:
        new_response_tokens, new_response_log_probs = [], []

    via_store = payload.get(OUTPUT_STORE_REQUEST_KEY, False)
    sampling_logprobs_mode = payload.get("sampling_logprobs_mode", "selected")
    if payload.get("return_sampling_mask", False):
        if via_store:
            # Only the mask rows are stored; the sampled tokens' log-probs, one per token, stay in meta_info.
            new_response_log_probs = read_sampled_logprobs(
                output["meta_info"],
                new_response_tokens,
                sampling_logprobs_mode=sampling_logprobs_mode,
            )
        else:
            new_response_log_probs = append_sampling_metadata(
                sample,
                new_response_tokens,
                output["meta_info"],
                sampling_logprobs_mode=sampling_logprobs_mode,
            )

    # Update sample with tokens directly - avoiding re-tokenization
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs
    # With the output store, the trainer builds support-mode candidates from the stored mask.
    if payload.get("top_logprobs_num") or (sampling_logprobs_mode == "support" and not via_store):
        append_rollout_topk_logprobs(
            sample,
            output["meta_info"],
            args.rollout_top_logprobs_num,
            sampling_logprobs_mode=sampling_logprobs_mode,
        )

    if update_loss_mask:
        if sample.loss_mask is None:
            sample.loss_mask = []
        sample.loss_mask += [1] * len(new_response_tokens)

    # TODO handle multi-turn cases (may need concat instead of assignment)
    sample.rollout_routed_experts = get_routed_experts_from_response(args, output, len(sample.tokens) - 1)
    sample.rollout_indexer_topk = get_indexer_topk_from_response(args, output, sample)
    if via_store:
        # This response's routed experts, indexer top-k and mask are in the output store: record its rows.
        sample.rollout_store_plan = _append_store_response(
            sample.rollout_store_plan,
            payload,
            output["meta_info"],
            num_tokens=len(sample.tokens),
            response_length=len(new_response_tokens),
        )

    # TODO may unify (currently there are both methods inside Sample and separate functions)
    sample.update_from_meta_info(args, output["meta_info"])


def _append_store_response(
    plan: StorePlan | None, payload: dict, meta_info: dict, *, num_tokens: int, response_length: int
) -> StorePlan:
    """The sample's plan after a response whose replay arrays SGLang put in the output store."""
    no_output_abort = not response_length and (meta_info.get("finish_reason") or {}).get("type") == "abort"
    new = StorePlan.from_response(
        store_ref_from_dict(meta_info.get(OUTPUT_STORE_REF_KEY)),
        request=payload,
        num_tokens=num_tokens,
        response_length=response_length,
        no_output_abort=no_output_abort,
    )
    return new if plan is None else plan.append_response(new)


def _decode_topk_buffer(info: str, num_tokens: int, num_layers: int, topk: int) -> np.ndarray:
    # validate=True: SGLang sends plain base64; pybase64 skips its slow non-alphabet filtering pass.
    x = np.frombuffer(pybase64.b64decode(info, validate=True), dtype=np.int32)
    if num_tokens <= 0:
        return np.empty((0, num_layers, max(0, topk)), dtype=np.int32)
    if topk == -1:  # indexer: topk dim recovered from buffer length
        topk = len(x) // (num_tokens * num_layers)
    return x.reshape(num_tokens, num_layers, topk)


def get_routed_experts_from_response(args, output, num_tokens: int):
    info = output["meta_info"].get("routed_experts")
    if info is None:
        return None
    routed_experts = _decode_topk_buffer(info, num_tokens, args.num_layers, -1)
    assert routed_experts.size == 0 or routed_experts.any(), (
        "routed_experts payload is all zeros: the sglang engine did not capture routed experts "
        "(topk-bypassing --moe-runner-backend such as flashinfer_trtllm?)."
    )
    return routed_experts


def get_indexer_topk_from_response(args, output, sample):
    info = output["meta_info"].get("indexer_topk")
    if info is None:
        return None
    num_layers = output["meta_info"].get("indexer_topk_num_layers")
    assert num_layers is not None, (
        "Server returned indexer_topk without indexer_topk_num_layers; "
        "sglang-miles must include the layer count in meta_info."
    )
    expected_num_streams = args.rollout_indexer_topk_num_streams
    assert expected_num_streams is None or num_layers == expected_num_streams, (
        f"Server returned indexer_topk with {num_layers} streams but the model has "
        f"{expected_num_streams} indexer layers; replaying it would map streams to the wrong layers."
    )
    return _decode_topk_buffer(info, len(sample.tokens) - 1, num_layers, -1)
