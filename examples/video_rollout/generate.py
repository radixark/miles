from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.multi_turn import generate as default_multi_turn
from miles.rollout.generate_hub.single_turn import generate as default_single_turn
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import (
    create_tool_call_parser,
    execute_tool_calls,
    update_sample_with_tool_responses,
)
from miles.utils.http_utils import post
from miles.utils.misc import load_function
from miles.utils.processing_utils import call_processor
from miles.utils.types import Sample


def render_prompt(tokenizer, prompt, tools=None) -> str:
    if not isinstance(prompt, str):
        prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
        )
    return prompt


def _prepare_prompt(input: GenerateFnInput, sample: Sample, tools=None) -> tuple[list[int], list[int]]:
    prompt = render_prompt(input.state.tokenizer, sample.prompt, tools=tools)
    rollout_prompt_ids = input.state.tokenizer.encode(prompt, add_special_tokens=False)
    processor_output = call_processor(input.state.processor, prompt, sample.multimodal_inputs)
    prompt_ids = processor_output["input_ids"][0]
    prompt_ids = prompt_ids if isinstance(prompt_ids, list) else prompt_ids.tolist()
    sample.multimodal_train_inputs = {
        key: value for key, value in processor_output.items() if key not in ("input_ids", "attention_mask")
    } or None
    return prompt_ids, rollout_prompt_ids


def _set_video_payload(
    payload: dict,
    sample: Sample,
    prompt_ids: list[int],
    rollout_prompt_ids: list[int],
) -> None:
    payload["input_ids"] = rollout_prompt_ids + sample.tokens[len(prompt_ids) :]
    payload["video_data"] = sample.rollout_video_sources


def _check_prefill_configuration(input: GenerateFnInput) -> None:
    if input.args.recompute_logprobs_via_prefill:
        raise NotImplementedError(
            "Video prefill recomputation requires "
            "--rollout-function-path examples.video_rollout.rollout.VideoInferenceRolloutFn"
        )


async def single_turn(input: GenerateFnInput) -> GenerateFnOutput:
    sample = input.sample
    if not sample.rollout_video_sources:
        return await default_single_turn(input)

    _check_prefill_configuration(input)
    args = input.args
    sampling_params = input.sampling_params
    assert sample.status in {Sample.Status.PENDING, Sample.Status.ABORTED}, f"{sample.status=}"

    prompt_ids, rollout_prompt_ids = _prepare_prompt(input, sample)
    if sample.response_length:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)
        assert sampling_params["max_new_tokens"] >= 0
        if sampling_params["max_new_tokens"] == 0:
            sample.status = Sample.Status.TRUNCATED
            return GenerateFnOutput(samples=sample)
    else:
        sample.tokens = prompt_ids.copy()

    payload, halt_status = compute_request_payload(
        args,
        input_ids=sample.tokens,
        sampling_params=sampling_params,
        multimodal_inputs=sample.multimodal_inputs,
    )
    if payload is None:
        sample.status = halt_status
        return GenerateFnOutput(samples=sample)

    _set_video_payload(payload, sample, prompt_ids, rollout_prompt_ids)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    output = await post(url, payload)
    await update_sample_from_response(
        args,
        sample,
        payload=payload,
        output=output,
        update_loss_mask=sample.loss_mask is not None,
    )
    return GenerateFnOutput(samples=sample)


async def multi_turn(input: GenerateFnInput) -> GenerateFnOutput:
    if not input.sample.rollout_video_sources:
        return await default_multi_turn(input)

    _check_prefill_configuration(input)
    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer
    assert not args.partial_rollout, "Partial rollout is not supported"

    execute_tool_function = load_function(args.generate_execute_tool_function_path)
    tool_specs = load_function(args.generate_tool_specs_path)
    tool_call_parser = create_tool_call_parser(tool_specs, args.generate_tool_call_parser)
    prompt_ids, rollout_prompt_ids = _prepare_prompt(input, sample, tools=tool_specs)
    sample.tokens = prompt_ids.copy()
    multi_samples = []
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    for _turn in range(args.generate_max_turns):
        payload, halt_status = compute_request_payload(
            args,
            input_ids=sample.tokens,
            sampling_params=input.sampling_params,
            multimodal_inputs=sample.multimodal_inputs,
        )
        if payload is None:
            sample.status = halt_status
            if args.generate_multi_samples and multi_samples:
                multi_samples[-1].status = halt_status
            break

        _set_video_payload(payload, sample, prompt_ids, rollout_prompt_ids)
        if args.generate_multi_samples:
            context_tokens = sample.tokens
            multimodal_train_inputs = sample.multimodal_train_inputs
            sample = deepcopy(input.sample)
            sample.tokens = context_tokens.copy()
            sample.multimodal_train_inputs = multimodal_train_inputs

        output = await post(url, payload)
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)

        if args.generate_multi_samples:
            multi_samples.append(deepcopy(sample))

        if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
            break

        _, tool_calls = tool_call_parser.parse_non_stream(output["text"])
        if not tool_calls:
            break

        tool_messages = await execute_tool_calls(tool_calls, execute_tool_function)
        update_sample_with_tool_responses(sample, tool_messages, tokenizer=tokenizer)

    return GenerateFnOutput(samples=multi_samples if args.generate_multi_samples else sample)


multi_turn.add_arguments = default_multi_turn.add_arguments
