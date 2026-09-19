import asyncio
import copy
import logging
import uuid
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from miles.ray.rollout.debug_data import save_rollout_candidate_evidence, summarize_rollout_candidates
from miles.rollout.base_types import GenerateFnInput, RolloutFnEvalOutput, RolloutFnTrainOutput
from miles.rollout.filter_hub.base_types import MetricGatherer
from miles.rollout.filter_hub.common_filters import apply_preput_filters
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.utils import dumper_utils
from miles.utils.async_utils import run
from miles.utils.data import Dataset
from miles.utils.eval_config import EvalDatasetConfig
from miles.utils.function_registry import load_function
from miles.utils.http_utils import get, post, router_worker_base_urls
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.lora import LORA_ADAPTER_NAME, lora_rollout_enabled
from miles.utils.misc import SingletonMeta, call_agent_abort_hook
from miles.utils.processing_utils import (
    call_processor,
    encode_image_for_rollout_engine,
    extract_multimodal_train_inputs,
    load_processor,
    load_tokenizer,
)
from miles.utils.types import Sample

from .generate_utils.generate_endpoint_utils import (
    compute_routing_headers,
    get_indexer_topk_from_response,
    policy_uses_routing_key,
)
from .generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from .generate_utils.sample_utils import reward_log_summary, sample_text_preview
from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)

_ROLLOUT_ABORT_TIMEOUT_SECONDS = 5.0
_ROLLOUT_CANCEL_TIMEOUT_SECONDS = 5.0


def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    """Return the router URL for a named model.

    Use this in custom rollout functions to route requests to a specific
    model when multiple models are deployed via ``--sglang-config``::

        url = get_model_url(args, "ref", "/generate")
        resp = await post(url, json=payload)

    Falls back to the default router if *model_name* is not found or
    ``sglang_model_routers`` is not set.
    """
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.submitted_candidate_groups = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
            self.submitted_candidate_groups += 1
            self.remaining_batch_size += 1


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    if state.processor and (
        isinstance(sample.prompt, (list, tuple))
        or (sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()))
    ):
        processor_output = call_processor(state.processor, sample.prompt, sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        prompt_ids = prompt_ids.tolist() if hasattr(prompt_ids, "tolist") else list(prompt_ids)
        sample.multimodal_train_inputs = extract_multimodal_train_inputs(processor_output)
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    opd_top_k = getattr(args, "opd_log_prob_top_k", 0) or 0
    opd_top_k_strategy = getattr(args, "opd_top_k_strategy", "only-student")
    if getattr(args, "use_opd", False) and opd_top_k > 0 and opd_top_k_strategy != "only-teacher":
        payload["top_logprobs_num"] = opd_top_k

    if lora_rollout_enabled(args):
        payload["lora_path"] = LORA_ADAPTER_NAME

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True
    if getattr(args, "use_rollout_indexer_replay", False):
        payload["return_indexer_topk"] = True

    if sample.multimodal_inputs and sample.multimodal_inputs["images"]:
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    if sample.multimodal_inputs and sample.multimodal_inputs.get("audios"):
        import base64 as _b64

        payload["audio_data"] = [
            f"data:audio;base64,{_b64.b64encode(a).decode('ascii')}" for a in sample.multimodal_inputs["audios"]
        ]

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids

    headers = compute_routing_headers(args, sample)

    output = await post(url, payload, headers=headers)
    if getattr(args, "use_opd", False) and opd_top_k > 0 and opd_top_k_strategy != "only-teacher":
        output_top_logprobs = output.get("meta_info", {}).get("output_top_logprobs")
        if output_top_logprobs is not None:
            sample.metadata.setdefault("opd_student_top_logprobs", [])
            sample.metadata["opd_student_top_logprobs"].extend(output_top_logprobs)

    if "output_token_logprobs" in output["meta_info"]:
        new_response_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_response_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_response_tokens, new_response_log_probs = [], []

    # Update sample with tokens directly - avoiding re-tokenization
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]

    # When partial rollout and masking off policy is enabled, update the loss mask
    if sample.loss_mask is not None:
        assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
        sample.loss_mask += [1] * len(new_response_tokens)

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs

    if "routed_experts" in output["meta_info"]:
        _re = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        )
        _ntok = int(output["meta_info"]["prompt_tokens"]) + len(new_response_tokens) - 1
        _topk = _re.size // max(1, _ntok * args.num_layers)
        if _re.size == (_ntok + 1) * args.num_layers * max(1, _topk):
            # stop-edge: sglang also forwarded the final token; its routing rows
            # feed no training position - drop the tail position.
            _re = _re[: _ntok * args.num_layers * _topk]
        assert _re.size == _ntok * args.num_layers * _topk, (
            f"routed_experts buffer {_re.size} != ntok({_ntok}) x layers({args.num_layers}) x topk({_topk}); "
            f"prompt_tokens={output['meta_info'].get('prompt_tokens')} response={len(new_response_tokens)} "
            f"unexpanded_tokens={len(sample.tokens)}"
        )
        sample.rollout_routed_experts = _re.reshape(_ntok, args.num_layers, _topk)
    if "indexer_topk" in output["meta_info"]:
        sample.rollout_indexer_topk = get_indexer_topk_from_response(args, output, sample)

    sample.update_from_meta_info(args, output["meta_info"])

    return sample


async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # dashboard lifecycle probe (design §18.3): the semaphore wait IS the
    # queue; attempt_end fires once generation is over, before reward
    sink = None if evaluation else TrajectoryLifecycle().sink
    if sink is not None:
        sink.attempt_start(sample)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            if sink is not None:
                sink.attempt_end(sample)
            return sample
        if sink is not None:
            sink.gen_start(sample)

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            generate_fn = load_generate_function(custom_func_path) if custom_func_path else None
            if generate_fn is not None:
                output = await generate_fn(
                    GenerateFnInput(state=state, sample=sample, sampling_params=sampling_params, evaluation=evaluation)
                )
                sample = output.samples
            else:
                sample = await generate(args, sample, sampling_params)

    if sink is not None:
        sink.attempt_end(sample)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    # Generate a unique routing_key for each sample in the group (routing-key policies only)
    if policy_uses_routing_key(args):
        for sample in group:
            if sample.routing_key is None:
                sample.routing_key = str(uuid.uuid4())

    tasks = []
    try:
        for idx, sample in enumerate(group):
            current_sampling_params = sampling_params.copy()
            if getattr(args, "sglang_enable_deterministic_inference", False):
                seed = state.group_sampling_seeds[idx]
                current_sampling_params["sampling_seed"] = seed
            tasks.append(
                asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
            )
        group = await asyncio.gather(*tasks)
    except BaseException:
        # gather propagates a child's failure without cancelling its siblings.
        await _cancel_rollout_tasks(tasks)
        raise

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


def _consume_task_exception(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()


async def _cancel_rollout_tasks(tasks) -> None:
    """Cancel only owned tasks, and never wait indefinitely for a misbehaving plugin."""
    tasks = set(tasks)
    if not tasks:
        return
    for task in tasks:
        if not task.done():
            task.cancel()
    done, pending = await asyncio.wait(tasks, timeout=_ROLLOUT_CANCEL_TIMEOUT_SECONDS)
    for task in done:
        _consume_task_exception(task)
    for task in pending:
        task.add_done_callback(_consume_task_exception)
    if pending:
        logger.error("%d rollout tasks ignored cancellation; cleanup grace period expired", len(pending))


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    """Best-effort endpoint abort plus bounded local task cleanup."""
    state = GenerateState(args)
    state.aborted = True
    aborted_samples = []
    abort_task = asyncio.create_task(_abort_and_collect(args, rollout_id, state, aborted_samples))
    try:
        done, _ = await asyncio.wait({abort_task}, timeout=_ROLLOUT_ABORT_TIMEOUT_SECONDS)
        if done:
            abort_task.result()
        else:
            logger.warning("Rollout %s abort timed out; cancelling local generation tasks", rollout_id)
    finally:
        await _cancel_rollout_tasks({abort_task} | state.pendings)
        state.pendings.clear()
    return aborted_samples


async def _abort_and_collect(args, rollout_id, state, aborted_samples) -> None:
    # Discover only this rollout router's workers, never other model/eval endpoints.
    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_miles_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]
    urls = router_worker_base_urls(urls)

    logger.info(f"Abort request for {urls}")
    abort_tasks = [post(f"{url}/abort_request", {"abort_all": True}) for url in urls]
    abort_results = await asyncio.gather(*abort_tasks, return_exceptions=True)
    for url, result in zip(urls, abort_results, strict=False):
        if isinstance(result, Exception):
            logger.warning(f"Failed to abort worker at {url}: {result}")

    # Let the agent integration tear down its in-flight trials so they stop hitting
    # SGLang, instead of running on until their own max_seq_len / timeout.
    await call_agent_abort_hook(args)

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, _ = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            # Keep unconsumed done tasks owned by state if a sibling raises.
            state.pendings.remove(task)
            group = task.result()
            if not args.partial_rollout:
                continue
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")


# A shortfall means this attempt's candidates did not fill the batch. An exhausted
# data source is not in this set: re-sampling it cannot produce different groups.
RETRYABLE_SHORTFALLS = frozenset({"sampling deadline exceeded", "candidate group budget exhausted"})


class InsufficientRolloutBatch(RuntimeError):
    """One rollout could not assemble enough valid groups.

    ``retryable`` separates "this attempt fell short" from "sampling is not
    advancing at all", so a retry budget can never mask a dead engine, a
    stalled queue or an exhausted data source.

    The bar is deliberately throughput-shaped rather than "anything finished":
    an attempt that completed at least ``rollout_batch_size`` candidate groups
    proved the fleet can produce a batch's worth of samples and merely lost
    them to the filters, which is what another draw fixes. An attempt that
    could not even complete that many candidates is throughput-bound, and
    re-drawing only burns another full deadline on the same bottleneck.
    """

    def __init__(
        self, message: str, *, reason: str, valid_groups: int, completed_groups: int, required_groups: int
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.valid_groups = valid_groups
        self.completed_groups = completed_groups
        self.required_groups = required_groups
        self.retryable = reason in RETRYABLE_SHORTFALLS and completed_groups >= required_groups


def _insufficient_rollout_batch(args, rollout_id, state, valid_groups, reason, completed_groups=0) -> RuntimeError:
    return InsufficientRolloutBatch(
        f"Rollout {rollout_id}: insufficient valid groups ({valid_groups}/{args.rollout_batch_size}); {reason}. "
        f"submitted_candidate_groups={state.submitted_candidate_groups}, "
        f"pending_groups={sum(not task.done() for task in state.pendings)}, "
        f"completed_candidate_groups={completed_groups}, "
        f"rollout_max_candidate_groups={getattr(args, 'rollout_max_candidate_groups', None)}, "
        f"rollout_timeout_seconds={getattr(args, 'rollout_timeout_seconds', None)}",
        reason=reason,
        valid_groups=valid_groups,
        completed_groups=completed_groups,
        required_groups=args.rollout_batch_size,
    )


def _report_rollout_candidates(args, rollout_id, state, data, all_data, filter_outputs, *, error, attempt=1):
    """Best-effort diagnostics must not replace the original sampling failure."""
    try:
        summary = summarize_rollout_candidates(args, all_data, filter_outputs)
        summary.update(
            valid_groups=len(data),
            target_groups=args.rollout_batch_size,
            submitted_candidate_groups=state.submitted_candidate_groups,
            pending_groups=sum(not task.done() for task in state.pendings),
            attempt=attempt,
        )
        log = logger.error if error is not None else logger.info
        # Emit the summary before optional filesystem I/O (e.g. a slow shared mount).
        log("Rollout %s candidate summary=%s; error=%s", rollout_id, summary, error)
        path = None
        try:
            path = save_rollout_candidate_evidence(
                args, rollout_id, all_data, filter_outputs, data, summary, error=error, attempt=attempt
            )
        except Exception:
            logger.warning("Rollout %s candidate evidence write failed", rollout_id, exc_info=True)
        log("Rollout %s candidate evidence_file=%s", rollout_id, path)
    except Exception:
        logger.warning("Rollout %s candidate diagnostics failed", rollout_id, exc_info=True)


async def _collect_rollout_samples(args, rollout_id, state, data_source, attempt=1):
    data, all_data, filter_outputs = [], [], []
    metrics = MetricGatherer()
    timeout = getattr(args, "rollout_timeout_seconds", None)
    deadline = asyncio.get_running_loop().time() + timeout if timeout is not None else None
    collector = asyncio.create_task(
        _fill_rollout_batch(args, rollout_id, state, data_source, data, all_data, filter_outputs, metrics, deadline)
    )
    error = None
    try:
        # Unlike wait_for, wait does not wait indefinitely for cancellation handlers.
        done, _ = await asyncio.wait({collector}, timeout=timeout)
        if not done:
            raise _insufficient_rollout_batch(
                args, rollout_id, state, len(data), "sampling deadline exceeded", len(all_data)
            )
        collector.result()
        return data, all_data, metrics
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        await _cancel_rollout_tasks({collector})
        # A deadline or sibling exception may leave completed results unconsumed.
        # Record them without filtering again or admitting them to the train batch.
        evidence_groups = list(all_data)
        for task in state.pendings:
            if task.done() and not task.cancelled() and task.exception() is None:
                evidence_groups.append(task.result())
                filter_outputs.append(None)
        _report_rollout_candidates(
            args, rollout_id, state, data, evidence_groups, filter_outputs, error=error, attempt=attempt
        )
        evidence_groups.clear()
        filter_outputs.clear()
        if error is not None:
            # Do not pin candidate lists in a retained exception traceback.
            data.clear()
            all_data.clear()


async def _collect_rollout_samples_with_retries(args, rollout_id, state, data_source):
    """Re-sample a rollout whose candidates fell short, within a fixed attempt budget.

    Only a shortfall is retried, and only when the failed attempt completed at
    least ``rollout_batch_size`` candidate groups. An exhausted data source, or
    a throughput-bound attempt, still fails immediately: the retry budget exists
    to absorb an unlucky accept rate, not to keep a stalled fleet spinning.

    Returns the collector's ``(data, all_data, metrics)`` plus the partial groups
    that the inter-attempt aborts drained, which the caller must hand back to the
    data source exactly like the ones its own cleanup abort produces.

    Note the sampling deadline and the candidate-group budget are per attempt, so
    the worst case for one rollout is ``rollout_max_attempts`` times each.
    """
    attempts = max(1, int(getattr(args, "rollout_max_attempts", 1) or 1))
    retry_aborted_samples: list[list[Sample]] = []
    previous_completed = None
    if attempts > 1:
        timeout = getattr(args, "rollout_timeout_seconds", None)
        logger.info(
            "Rollout %s may use up to %s sampling attempts; the deadline and candidate budget are per "
            "attempt, so the worst case is %s x %ss and %s x %s candidate groups.",
            rollout_id,
            attempts,
            attempts,
            timeout,
            attempts,
            getattr(args, "rollout_max_candidate_groups", None),
        )
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            # In-flight requests from the failed attempt must not bleed into this one.
            # A cleanup failure here is incidental: it must neither replace the shortfall
            # we diagnosed nor consume the remaining attempts.
            try:
                retry_aborted_samples.extend(await abort(args, rollout_id))
            except BaseException:
                logger.warning(
                    "Rollout %s attempt %s abort failed; continuing with a reset state",
                    rollout_id,
                    attempt,
                    exc_info=True,
                )
            state.reset()
        try:
            data, all_data, metrics = await _collect_rollout_samples(
                args, rollout_id, state, data_source, attempt=attempt
            )
            return data, all_data, metrics, retry_aborted_samples
        except InsufficientRolloutBatch as exc:
            if attempt >= attempts or not exc.retryable:
                raise
            # Two attempts that do not improve mean the bottleneck is not luck.
            if previous_completed is not None and exc.completed_groups < previous_completed:
                logger.error(
                    "Rollout %s attempt %s completed fewer candidate groups than attempt %s (%s < %s); "
                    "the fleet is degrading, not unlucky. Not retrying.",
                    rollout_id,
                    attempt,
                    attempt - 1,
                    exc.completed_groups,
                    previous_completed,
                )
                raise
            previous_completed = exc.completed_groups
            logger.warning(
                "Rollout %s attempt %s/%s fell short (%s/%s valid groups, %s completed candidates); "
                "retrying: %s",
                rollout_id,
                attempt,
                attempts,
                exc.valid_groups,
                args.rollout_batch_size,
                exc.completed_groups,
                exc.reason,
            )
    raise AssertionError("rollout retry loop must return or raise")


async def _fill_rollout_batch(args, rollout_id, state, data_source, data, all_data, filter_outputs, metrics, deadline):
    def check_deadline():
        if deadline is not None and asyncio.get_running_loop().time() >= deadline:
            raise _insufficient_rollout_batch(
                args, rollout_id, state, len(data), "sampling deadline exceeded", len(all_data)
            )

    await dumper_utils.configure_sglang(args)
    dynamic_filter = load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path else None
    max_groups = getattr(args, "rollout_max_candidate_groups", None)
    target = args.rollout_batch_size
    do_print = True
    pbar = tqdm(total=target * args.n_samples_per_prompt, desc="Rollout generation")
    try:
        while len(data) < target:
            check_deadline()
            while state.remaining_batch_size < target:
                check_deadline()
                request_size = args.over_sampling_batch_size
                if max_groups is not None:
                    request_size = min(request_size, max_groups - state.submitted_candidate_groups)
                    if request_size <= 0:
                        break
                samples = data_source(request_size)
                check_deadline()
                if max_groups is not None:
                    samples = samples[:request_size]
                if not samples:
                    raise _insufficient_rollout_batch(
                        args, rollout_id, state, len(data), "data source returned no groups", len(all_data)
                    )
                state.submit_generate_tasks(samples)

            # Exhausting the submission budget does not discard in-flight groups.
            if not state.pendings:
                raise _insufficient_rollout_batch(
                    args, rollout_id, state, len(data), "candidate group budget exhausted", len(all_data)
                )
            done, _ = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
            check_deadline()
            for task in done:
                # Keep unconsumed completed siblings tracked for exception cleanup.
                state.pendings.remove(task)
                group = task.result()
                if do_print:
                    sample = group[0][0] if isinstance(group[0], list) else group[0]
                    logger.info(
                        "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
                        sample_text_preview(sample),
                        str(sample.label)[:100],
                        reward_log_summary(sample.reward),
                    )
                    do_print = False
                assert len(group) == args.n_samples_per_prompt
                all_data.append(group)
                filter_outputs.append(None)
                filter_output = apply_preput_filters(args, dynamic_filter, group)
                filter_outputs[-1] = filter_output
                if not filter_output.keep:
                    metrics.on_dynamic_filter_drop(reason=filter_output.reason)
                check_deadline()
                if not filter_output.keep:
                    state.remaining_batch_size -= 1
                    continue
                if len(data) < target:
                    data.append(group)
                    pbar.update(args.n_samples_per_prompt)
    finally:
        pbar.close()


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset
    state = GenerateState(args)
    # Budgets and task ownership belong to one outer rollout, not to the filter.
    state.reset()
    abort_started = False
    try:
        data, all_data, metric_gatherer, retry_aborted_samples = await _collect_rollout_samples_with_retries(
            args, rollout_id, state, data_source
        )
        sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
        logger.info(
            "Finish rollout: valid_groups=%s, submitted_candidate_groups=%s, "
            "text_preview=%s, label=%s, reward_summary=%s",
            len(data),
            state.submitted_candidate_groups,
            sample_text_preview(sample),
            str(sample.label)[:100],
            reward_log_summary(sample.reward),
        )
        abort_started = True
        # Groups drained by an inter-attempt abort are owed to the data source just
        # like these; dropping them would silently burn their prompts and tokens.
        aborted_samples = retry_aborted_samples + await abort(args, rollout_id)
    except BaseException:
        if not abort_started:
            try:
                await abort(args, rollout_id)
            except BaseException:
                logger.warning("Rollout %s cleanup failed; preserving original exception", rollout_id, exc_info=True)
        raise
    finally:
        # Also reset on failure so a later rollout/eval cannot inherit the budget.
        state.reset()

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )
    if (x := args.rollout_sample_filter_path) is not None:
        filter_func = load_function(x)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if (x := args.rollout_all_samples_process_path) is not None:
        process_func = load_function(x)
        process_func(args, all_samples, data_source)

    await recompute_samples_rollout_logprobs_via_prefill(
        args,
        [sample for group in data for sample in group],
        url=get_model_url(args, "default"),
        sampling_params=state.sampling_params,
    )

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template, args.chat_template_path)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            if policy_uses_routing_key(args):
                sample.routing_key = str(uuid.uuid4())
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logged_sample = sample[0] if isinstance(sample, list) else sample
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(logged_sample.prompt) + logged_sample.response]} "
                f"reward={logged_sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    return output
