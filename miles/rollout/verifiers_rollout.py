from __future__ import annotations

import asyncio
import logging
import random
import sys
import uuid
from argparse import Namespace
from collections import OrderedDict
from collections.abc import Iterable
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
from packaging.version import InvalidVersion, Version

from miles.rollout.base_types import (
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from miles.utils.lora import LORA_ADAPTER_NAME, is_lora_enabled
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_MIN_VERIFIERS_VERSION = Version("0.2.0")
_MAX_VERIFIERS_VERSION = Version("0.2.1")
_MIN_RENDERERS_VERSION = Version("0.1.8")
_UNSUPPORTED_ERROR_PREFIX = "Miles' Verifiers adapter does not support"


def _load_config_data(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if config_path.suffix.lower() != ".toml":
        raise ValueError("--verifiers-config must point to a Verifiers TOML config.")
    if sys.version_info < (3, 11):
        raise _optional_dependency_error()

    import tomllib

    data = tomllib.loads(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the root.")
    return data


def _optional_dependency_error() -> RuntimeError:
    return RuntimeError(
        "Verifiers rollouts require Python 3.11+ and the optional dependencies. "
        "Install Miles with `pip install -e '.[verifiers]'`."
    )


def _installed_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError as error:
        raise _optional_dependency_error() from error


def _check_version(package: str, raw_version: str, minimum: Version, maximum: Version | None = None) -> None:
    try:
        installed = Version(raw_version)
    except InvalidVersion as error:
        raise RuntimeError(f"Could not parse installed {package} version {raw_version!r}.") from error
    if installed < minimum:
        raise RuntimeError(f"Verifiers rollouts require {package}>={minimum}; found {installed}.")
    if maximum is not None and installed >= maximum:
        raise RuntimeError(
            f"Verifiers rollouts require {package}>={minimum},<{maximum}; found {installed}. "
            "Verifiers 0.2.1 requires OpenAI>=2.9, while SGLang 0.5.15 pins OpenAI==2.6.1."
        )


def _import_verifiers():
    if sys.version_info < (3, 11):
        raise _optional_dependency_error()
    _check_version(
        "verifiers",
        _installed_version("verifiers"),
        _MIN_VERIFIERS_VERSION,
        _MAX_VERIFIERS_VERSION,
    )
    _check_version("renderers", _installed_version("renderers"), _MIN_RENDERERS_VERSION)
    try:
        from verifiers.v1 import EnvConfig, Environment, ModelContext, SamplingConfig
        from verifiers.v1.clients.train import TrainClient
        from verifiers.v1.decorators import discover_decorated
        from verifiers.v1.errors import OverlongPromptError, ProviderError
    except ImportError as error:
        raise _optional_dependency_error() from error
    return SimpleNamespace(
        EnvConfig=EnvConfig,
        Environment=Environment,
        discover_decorated=discover_decorated,
        ModelContext=ModelContext,
        OverlongPromptError=OverlongPromptError,
        ProviderError=ProviderError,
        SamplingConfig=SamplingConfig,
        TrainClient=TrainClient,
    )


def _renderer_identity(checkpoint: str) -> str | None:
    from renderers.base import MODEL_RENDERER_MAP

    if checkpoint in MODEL_RENDERER_MAP:
        return checkpoint

    path = Path(checkpoint)
    candidates = []
    for part in path.parts:
        if part.startswith("models--"):
            candidates.append(part.removeprefix("models--").replace("--", "/"))
    candidates.extend(model_id for model_id in MODEL_RENDERER_MAP if model_id.rsplit("/", 1)[-1] == path.name)
    matches = sorted(set(candidates) & MODEL_RENDERER_MAP.keys())
    if not matches:
        return None
    renderer_names = {MODEL_RENDERER_MAP[model_id] for model_id in matches}
    return matches[0] if len(renderer_names) == 1 else None


def _train_client(
    runtime,
    args: Namespace,
    model: str,
    pool_size: int,
    *,
    router_args: Namespace | None = None,
):
    tokenizer_source = getattr(args, "sglang_tokenizer_path", None) or model
    identity = _renderer_identity(model) or _renderer_identity(tokenizer_source)

    # TrainClient uses one path for both tokenizer loading and renderer lookup.
    # Miles checkpoints are commonly local snapshots, so keep local tokenizer
    # files while restoring the canonical identity used by Renderers' registry.
    class TrainClient(runtime.TrainClient):
        @staticmethod
        def _unsupported_request(kind: str):
            return runtime.ProviderError(
                f"{_UNSUPPORTED_ERROR_PREFIX} {kind}.",
                status_code=400,
            )

        async def get_response(self, *args, **kwargs):
            try:
                return await super().get_response(*args, **kwargs)
            except NotImplementedError as error:
                raise runtime.ProviderError(
                    f"{_UNSUPPORTED_ERROR_PREFIX} this request: {error}",
                    status_code=400,
                ) from error
            except ValueError as error:
                if "does not support tools" not in str(error):
                    raise
                raise runtime.ProviderError(
                    f"{_UNSUPPORTED_ERROR_PREFIX} tools with this renderer: {error} "
                    "Use a Renderers-registered model identity in "
                    "--hf-checkpoint or --sglang-tokenizer-path.",
                    status_code=400,
                ) from error

        async def relay(self, *args, **kwargs):
            raise self._unsupported_request("streaming requests")

        async def relay_aux(self, *args, **kwargs):
            raise self._unsupported_request("auxiliary dialect routes")

        def _renderer_pool(self, requested_model, *, chat_template_kwargs=None):
            if identity is None:
                return super()._renderer_pool(
                    requested_model,
                    chat_template_kwargs=chat_template_kwargs,
                )
            if self._pool is None:
                from renderers import RendererPool, create_renderer
                from renderers.base import load_tokenizer

                source = self.renderer_model_name or requested_model

                def factory():
                    tokenizer = load_tokenizer(source)
                    tokenizer.name_or_path = identity
                    return create_renderer(
                        tokenizer,
                        self.config,
                        chat_template_kwargs=chat_template_kwargs,
                    )

                self._pool = RendererPool(factory, size=self.pool_size)
            return self._pool

    return TrainClient(
        MilesSGLangTransport(args, router_args=router_args),
        pool_size=pool_size,
        renderer_model_name=tokenizer_source,
    )


def _generate_url(args: Namespace, endpoint: str = "/generate") -> str:
    routers = getattr(args, "sglang_model_routers", None)
    if routers and "default" in routers:
        ip, port = routers["default"]
    else:
        ip, port = args.sglang_router_ip, args.sglang_router_port
    return f"http://{ip}:{port}{endpoint}"


async def _sglang_worker_urls(args: Namespace) -> list[str]:
    from miles.utils.http_utils import get

    router_url = _generate_url(args).removesuffix("/generate")
    if not getattr(args, "use_miles_router", False):
        try:
            response = await get(f"{router_url}/workers")
            return [worker["url"] for worker in response["workers"]]
        except Exception:
            logger.debug("SGLang /workers lookup failed; trying Miles /list_workers.", exc_info=True)
    response = await get(f"{router_url}/list_workers")
    return list(response["urls"])


def _finish_reason(output: dict[str, Any]) -> str:
    finish_reason = (output.get("meta_info") or {}).get("finish_reason")
    if isinstance(finish_reason, dict):
        finish_reason = finish_reason.get("type")
    if finish_reason == "abort":
        raise RuntimeError("SGLang aborted the Verifiers generation request.")
    return finish_reason if finish_reason in {"stop", "length", "content_filter"} else "stop"


class MilesSGLangTransport:
    """Translate Renderers' vLLM generate wire format to Miles' SGLang endpoint."""

    def __init__(self, args: Namespace, *, router_args: Namespace | None = None):
        self.args = args
        self.router_args = router_args or args
        self._seen_sessions: OrderedDict[str, None] = OrderedDict()
        self._session_cache_size = 10_000

    @property
    def base_url(self) -> str:
        # The refactored RolloutManager constructs rollout functions before it
        # starts SGLang and fills in the router address.
        return f"{_generate_url(self.router_args, '').rstrip('/')}/v1"

    async def get(self, _path: str, **_kwargs) -> dict[str, list[Any]]:
        # Renderers can discover a vLLM context cap here. Miles owns separate
        # prompt and response limits, so the transport enforces them at POST time.
        return {"data": []}

    def _sampling_params(self, raw: dict[str, Any], prompt_len: int) -> dict[str, Any]:
        values = dict(raw)
        values.pop("logprobs", None)
        for source, target in (
            ("max_tokens", "max_new_tokens"),
            ("min_tokens", "min_new_tokens"),
            ("seed", "sampling_seed"),
        ):
            if source in values:
                values[target] = values.pop(source)

        if self.args.rollout_stop is not None:
            values.setdefault("stop", self.args.rollout_stop)
        if self.args.rollout_stop_token_ids is not None:
            renderer_stops = list(values.get("stop_token_ids") or [])
            values["stop_token_ids"] = list(dict.fromkeys([*renderer_stops, *self.args.rollout_stop_token_ids]))
        values["skip_special_tokens"] = self.args.rollout_skip_special_tokens
        values["no_stop_trim"] = True
        values["spaces_between_special_tokens"] = False
        values["n"] = 1

        context_limit = self.args.rollout_max_context_len
        response_limit = self.args.rollout_max_response_len
        requested = int(values.get("max_new_tokens", response_limit))
        if context_limit is not None:
            requested = min(requested, context_limit - prompt_len)
        values["max_new_tokens"] = min(requested, response_limit)
        if values["max_new_tokens"] <= 0:
            runtime = _import_verifiers()
            raise runtime.OverlongPromptError(
                f"prompt has {prompt_len} tokens, rollout_max_context_len={context_limit}"
            )
        return values

    async def post(self, endpoint: str, *, body: dict[str, Any], options=None, **_kwargs) -> httpx.Response:
        if body.get("features") is not None:
            raise NotImplementedError("Miles Verifiers rollouts do not yet support multimodal renderer features.")

        prompt_ids = list(body["token_ids"])
        headers = dict((options or {}).get("headers") or {})
        session_id = headers.get("X-Session-ID")
        if session_id:
            if session_id in self._seen_sessions:
                self._seen_sessions.move_to_end(session_id)
            else:
                max_prompt_len = getattr(self.args, "rollout_max_prompt_len", None)
                if max_prompt_len is not None and len(prompt_ids) > max_prompt_len:
                    runtime = _import_verifiers()
                    raise runtime.OverlongPromptError(
                        f"initial prompt has {len(prompt_ids)} tokens, rollout_max_prompt_len={max_prompt_len}"
                    )
                self._seen_sessions[session_id] = None
                if len(self._seen_sessions) > self._session_cache_size:
                    self._seen_sessions.popitem(last=False)

        payload: dict[str, Any] = {
            "input_ids": prompt_ids,
            "sampling_params": self._sampling_params(body.get("sampling_params") or {}, len(prompt_ids)),
            "return_logprob": True,
        }
        if is_lora_enabled(self.args):
            payload["lora_path"] = LORA_ADAPTER_NAME
        if body.get("priority") is not None:
            payload["priority"] = body["priority"]
        if body.get("cache_salt") is not None:
            payload["extra_key"] = body["cache_salt"]

        request_headers = None
        if getattr(self.args, "sglang_router_policy", None) in ("consistent_hashing", "manual") and session_id:
            request_headers = {"X-SMG-Routing-Key": session_id}

        from miles.utils.http_utils import post

        output = await post(_generate_url(self.router_args), payload, headers=request_headers)
        meta_info = dict(output.get("meta_info") or {})
        token_logprobs = list(meta_info.get("output_token_logprobs") or [])
        completion_ids = [int(item[1]) for item in token_logprobs]
        completion_logprobs = [float(item[0]) for item in token_logprobs]
        expected = int(meta_info.get("completion_tokens", len(completion_ids)))
        if len(completion_ids) != expected:
            raise RuntimeError(
                "SGLang generate response has mismatched completion token metadata: "
                f"{len(completion_ids)} != {expected}"
            )

        response_body = {
            "request_id": output.get("request_id") or f"vf-{uuid.uuid4().hex}",
            "choices": [
                {
                    "token_ids": completion_ids,
                    "logprobs": {"content": [{"logprob": value} for value in completion_logprobs]},
                    "finish_reason": _finish_reason(output),
                }
            ],
        }
        request = httpx.Request("POST", endpoint)
        return httpx.Response(200, json=response_body, request=request)

    async def close(self) -> None:
        return None


def _sample_status(trace) -> Sample.Status:
    if trace.has_error:
        return Sample.Status.FAILED
    if trace.is_truncated:
        return Sample.Status.TRUNCATED
    return Sample.Status.COMPLETED


def _serialize_prompt(prompt):
    if isinstance(prompt, list):
        return [
            message.model_dump(mode="json", exclude_none=True) if hasattr(message, "model_dump") else message
            for message in prompt
        ]
    return prompt or ""


def _validate_group_reward_sample_counts(args: Namespace, tasks, discover_decorated) -> None:
    if not any(discover_decorated(task, "group_reward") for task in tasks):
        return
    if getattr(args, "num_rollout", None) != 0 and args.n_samples_per_prompt < 2:
        raise ValueError("Verifiers tasks with @group_reward require --n-samples-per-prompt >= 2.")
    if getattr(args, "eval_interval", None) is not None and args.n_samples_per_eval_prompt < 2:
        raise ValueError("Verifiers tasks with @group_reward require --n-samples-per-eval-prompt >= 2.")


def _raise_for_unsupported_trace_errors(traces) -> None:
    for trace in traces:
        if trace.error is not None and trace.error.message.startswith(_UNSUPPORTED_ERROR_PREFIX):
            raise RuntimeError(trace.error.message)


def _branch_to_sample(args: Namespace, trace, branch, *, group_index: int, index: int) -> Sample:
    tokens = list(branch.token_ids)
    sampled_mask = list(branch.sampled_mask)
    logprobs = list(branch.logprobs)
    if len(tokens) != len(sampled_mask) or len(tokens) != len(logprobs):
        raise ValueError(
            f"Trace {trace.id} token metadata mismatch: "
            f"tokens={len(tokens)}, mask={len(sampled_mask)}, logprobs={len(logprobs)}"
        )
    first_sampled = sampled_mask.index(True) if True in sampled_mask else len(tokens)
    response_length = len(tokens) - first_sampled
    reward = trace.reward if args.reward_key is None else {**trace.rewards, "reward": trace.reward}
    task_data = trace.task.data
    label = getattr(task_data, "label", None)
    if label is None:
        label = getattr(task_data, "answer", None)

    metadata = {
        "verifiers": {
            "branch_index": branch.index,
            "task_index": getattr(task_data, "idx", None),
            "rewards": dict(trace.rewards),
            "metrics": dict(trace.metrics),
            "stop_condition": trace.stop_condition,
        }
    }
    if trace.error is not None:
        metadata["verifiers"]["error"] = trace.error.model_dump(mode="json", exclude_none=True)

    sample = Sample(
        group_index=group_index,
        index=index,
        prompt=_serialize_prompt(getattr(task_data, "prompt", "")),
        tokens=tokens,
        response=trace.last_reply,
        response_length=response_length,
        label=label,
        reward=reward,
        loss_mask=[int(value) for value in sampled_mask[first_sampled:]],
        rollout_log_probs=logprobs[first_sampled:],
        status=_sample_status(trace),
        metadata=metadata,
        routing_key=trace.id,
    )
    sample.validate()
    return sample


def trace_to_samples(args: Namespace, trace, *, group_index: int, index_start: int) -> list[Sample]:
    if not trace.branches:
        error = trace.error.model_dump(mode="json", exclude_none=True) if trace.error is not None else None
        logger.warning(
            "Verifiers trace %s has no graph branches; omitting it from training (error=%s).",
            trace.id,
            error,
        )
        return []
    if len(trace.branches) != 1:
        raise NotImplementedError(
            "Miles cannot yet preserve Verifiers trace groups when a rollout produces "
            f"multiple graph branches (trace {trace.id} produced {len(trace.branches)})."
        )
    return [
        _branch_to_sample(
            args,
            trace,
            trace.branches[0],
            group_index=group_index,
            index=index_start,
        )
    ]


def trace_to_sample(args: Namespace, trace, *, group_index: int, index: int) -> Sample:
    samples = trace_to_samples(args, trace, group_index=group_index, index_start=index)
    if len(samples) != 1:
        raise ValueError(f"Verifiers trace {trace.id} produced {len(samples)} branches, expected one.")
    return samples[0]


def _trace_metrics(traces) -> dict[str, float]:
    if not traces:
        return {}
    return {
        "verifiers/reward_mean": sum(trace.reward for trace in traces) / len(traces),
        "verifiers/error_rate": sum(trace.has_error for trace in traces) / len(traces),
        "verifiers/truncated_rate": sum(trace.is_truncated for trace in traces) / len(traces),
        "verifiers/num_turns_mean": sum(trace.num_turns for trace in traces) / len(traces),
    }


def _trace_eval_reward(trace, reward_key: str | None):
    if reward_key is None:
        return trace.reward
    rewards = {**trace.rewards, "reward": trace.reward}
    if trace.has_error:
        return rewards.get(reward_key)
    return rewards[reward_key]


def _flatten_samples(values: Iterable[Any]) -> list[Sample]:
    flattened = []
    for value in values:
        if isinstance(value, list):
            flattened.extend(_flatten_samples(value))
        else:
            flattened.append(value)
    return flattened


def _make_eval_args(args: Namespace) -> Namespace:
    eval_args = Namespace(**vars(args))
    for eval_name, rollout_name in (
        ("eval_temperature", "rollout_temperature"),
        ("eval_top_p", "rollout_top_p"),
        ("eval_top_k", "rollout_top_k"),
        ("eval_max_response_len", "rollout_max_response_len"),
        ("eval_max_context_len", "rollout_max_context_len"),
    ):
        if (value := getattr(args, eval_name, None)) is not None:
            setattr(eval_args, rollout_name, value)
    eval_args.rollout_max_prompt_len = args.eval_max_prompt_len
    eval_args.rollout_min_new_tokens = args.eval_min_new_tokens
    eval_args.reward_key = args.eval_reward_key or args.reward_key
    return eval_args


class VerifiersRolloutFn:
    def __init__(self, input: RolloutFnConstructorInput):
        runtime = _import_verifiers()
        self.args = input.args
        self.data_source = input.data_source
        self.config = runtime.EnvConfig.model_validate(_load_config_data(self.args.verifiers_config))
        if self.config.is_legacy:
            raise ValueError("Miles' Verifiers integration supports V1 environment configs only.")
        if self.config.harness.id == "codex":
            raise ValueError(
                "Miles' Verifiers adapter does not support the Codex harness because it uses the Responses dialect."
            )

        self.env = runtime.Environment(self.config)
        self.model = self.args.hf_checkpoint
        self.sampling = self._sampling_config(runtime.SamplingConfig, self.args)
        self.eval_args = _make_eval_args(self.args)
        self.eval_sampling = self._sampling_config(runtime.SamplingConfig, self.eval_args)

        engine_count = self.args.rollout_num_gpus // self.args.rollout_num_gpus_per_engine
        self.max_concurrent = self.args.sglang_server_concurrency * engine_count
        pool_size = max(1, min(self.max_concurrent, 16))
        self.client = _train_client(runtime, self.args, self.model, pool_size)
        self.eval_client = _train_client(
            runtime,
            self.eval_args,
            self.model,
            pool_size,
            router_args=self.args,
        )
        self.ctx = runtime.ModelContext(client=self.client, model=self.model, sampling=self.sampling)
        self.eval_ctx = runtime.ModelContext(client=self.eval_client, model=self.model, sampling=self.eval_sampling)

        from miles.utils.misc import load_function

        self.dynamic_filter = load_function(self.args.dynamic_sampling_filter_path)
        self._tasks = list(self.env.taskset.load())
        if self.args.rollout_shuffle:
            random.Random(self.args.rollout_seed).shuffle(self._tasks)
        if not self._tasks:
            raise ValueError("Verifiers taskset selected zero tasks.")
        _validate_group_reward_sample_counts(self.args, self._tasks, runtime.discover_decorated)
        self._next_train_task_idx = None
        self._next_group_index = 0
        self._next_sample_index = 0

    @staticmethod
    def _sampling_config(SamplingConfig, args: Namespace):
        data: dict[str, Any] = {
            "temperature": args.rollout_temperature,
            "top_p": args.rollout_top_p,
            "max_tokens": args.rollout_max_response_len,
        }
        if args.rollout_top_k is not None:
            data["top_k"] = args.rollout_top_k
        if (min_tokens := getattr(args, "rollout_min_new_tokens", None)) is not None:
            data["min_tokens"] = min_tokens
        if args.apply_chat_template_kwargs:
            data["extra_body"] = {"chat_template_kwargs": args.apply_chat_template_kwargs}
        return SamplingConfig.model_validate(data)

    def _task(self, index: int):
        return self._tasks[index % len(self._tasks)]

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        return await (self._call_eval(input) if input.evaluation else self._call_train(input))

    async def _run_task_group(self, task, n: int, semaphore: asyncio.Semaphore, seed_base: int, ctx=None):
        runtime = _import_verifiers()
        ctx = ctx or self.ctx
        episode = self.env.episode(task, ctx, n=n)
        if getattr(self.args, "sglang_enable_deterministic_inference", False):
            for offset, rollout in enumerate(episode.rollouts):
                sampling = ctx.sampling.model_copy(update={"sampling_seed": seed_base + offset})
                rollout.ctx = runtime.ModelContext(client=ctx.client, model=self.model, sampling=sampling)
        return await episode.run(semaphore)

    def _convert_group(self, traces, *, group_index: int, preserve_empty: bool = False):
        group = []
        complete = True
        for trace in traces:
            converted = trace_to_samples(
                self.args,
                trace,
                group_index=group_index,
                index_start=self._next_sample_index,
            )
            self._next_sample_index += len(converted)
            if not converted:
                complete = False
                if preserve_empty:
                    group.append(None)
                continue
            group.append(converted[0])
        return group if complete or preserve_empty else []

    async def _apply_miles_rewards(self, group) -> None:
        from miles.rollout.rm_hub import async_rm, batched_async_rm

        samples = _flatten_samples(group)
        if self.args.group_rm:
            rewards = await batched_async_rm(self.args, samples)
        elif self.args.custom_rm_path is not None or self.args.rm_type:
            rewards = await asyncio.gather(*(async_rm(self.args, sample) for sample in samples))
        else:
            return
        if rewards is None or len(rewards) != len(samples):
            raise ValueError("Miles reward model returned an unexpected number of rewards.")
        for sample, reward in zip(samples, rewards, strict=True):
            sample.reward = reward

    async def _postprocess_train_samples(self, data, all_data) -> None:
        from miles.utils.misc import load_function

        if function := load_function(self.args.rollout_sample_filter_path):
            function(self.args, data)
        if function := load_function(self.args.rollout_all_samples_process_path):
            function(self.args, all_data, self.data_source)
        await recompute_samples_rollout_logprobs_via_prefill(
            self.args,
            _flatten_samples(data),
            url=_generate_url(self.args),
            sampling_params={
                "temperature": self.args.rollout_temperature,
                "top_p": self.args.rollout_top_p,
                "top_k": self.args.rollout_top_k,
                "max_new_tokens": self.args.rollout_max_response_len,
            },
        )

    async def _cancel_pending(self, futures: Iterable[asyncio.Task]) -> None:
        pending = [future for future in futures if not future.done()]
        for future in pending:
            future.cancel()
        if pending:
            try:
                from miles.utils.http_utils import post

                urls = await _sglang_worker_urls(self.args)
                await asyncio.gather(
                    *(post(f"{url}/abort_request", {"abort_all": True}) for url in urls),
                    return_exceptions=True,
                )
            except Exception:
                logger.exception("Failed to abort pending Verifiers requests.")
        await asyncio.gather(*futures, return_exceptions=True)

    async def _call_train(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        from miles.utils import dumper_utils

        await dumper_utils.configure_sglang(self.args)
        target = self.args.rollout_batch_size
        if self._next_train_task_idx is None:
            self._next_train_task_idx = input.rollout_id * target

        groups = []
        all_groups = []
        all_traces = []
        metrics = MetricGatherer()
        semaphore = asyncio.Semaphore(self.max_concurrent)
        pending: set[asyncio.Task] = set()
        async with self.env.serving():
            try:
                while len(groups) < target:
                    while len(groups) + len(pending) < target:
                        for _ in range(self.args.over_sampling_batch_size):
                            task_index = self._next_train_task_idx
                            self._next_train_task_idx += 1
                            seed = self.args.rollout_seed + task_index * self.args.n_samples_per_prompt
                            pending.add(
                                asyncio.create_task(
                                    self._run_task_group(
                                        self._task(task_index),
                                        self.args.n_samples_per_prompt,
                                        semaphore,
                                        seed,
                                    )
                                )
                            )
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for future in done:
                        try:
                            traces = future.result()
                        except Exception:
                            logger.exception("Verifiers episode failed; resampling.")
                            metrics.on_dynamic_filter_drop(reason="episode_error")
                            continue
                        all_traces.extend(traces)
                        _raise_for_unsupported_trace_errors(traces)
                        if any(trace.has_error for trace in traces):
                            metrics.on_dynamic_filter_drop(reason="trace_error")
                            continue
                        group = self._convert_group(traces, group_index=self._next_group_index)
                        self._next_group_index += 1
                        if len(group) != self.args.n_samples_per_prompt:
                            metrics.on_dynamic_filter_drop(reason="empty_trace")
                            continue
                        await self._apply_miles_rewards(group)
                        all_groups.append(group)
                        result = call_dynamic_filter(
                            self.dynamic_filter,
                            self.args,
                            _flatten_samples(group),
                        )
                        if not result.keep:
                            metrics.on_dynamic_filter_drop(reason=result.reason)
                            continue
                        if len(groups) < target:
                            groups.append(group)
            finally:
                await self._cancel_pending(pending)

        groups.sort(key=lambda group: _flatten_samples(group)[0].index)
        all_groups.sort(key=lambda group: _flatten_samples(group)[0].index)
        await self._postprocess_train_samples(groups, all_groups)
        output_metrics = metrics.collect()
        output_metrics.update(_trace_metrics(all_traces))
        return RolloutFnTrainOutput(samples=groups, metrics=output_metrics)

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnEvalOutput:
        assert not self.args.group_rm, "Group RM is not supported for eval rollout"

        from miles.utils import dumper_utils

        await dumper_utils.configure_sglang(self.args)
        semaphore = asyncio.Semaphore(self.max_concurrent)
        async with self.env.serving():
            futures = [
                asyncio.create_task(
                    self._run_task_group(
                        task,
                        self.args.n_samples_per_eval_prompt,
                        semaphore,
                        self.args.rollout_seed + index * self.args.n_samples_per_eval_prompt,
                        self.eval_ctx,
                    )
                )
                for index, task in enumerate(self._tasks)
            ]
            try:
                trace_groups = await asyncio.gather(*futures)
            finally:
                await self._cancel_pending(futures)

        samples = []
        rewards = []
        truncated = []
        all_traces = []
        reward_key = self.args.eval_reward_key or self.args.reward_key
        use_miles_rewards = bool(self.args.custom_rm_path is not None or self.args.rm_type)
        for group_index, traces in enumerate(trace_groups):
            all_traces.extend(traces)
            _raise_for_unsupported_trace_errors(traces)
            group = self._convert_group(traces, group_index=group_index, preserve_empty=True)
            trainable = [value for value in group if value is not None]
            await self._apply_miles_rewards(trainable)
            samples.extend(_flatten_samples(trainable))
            for trace, value in zip(traces, group, strict=True):
                if use_miles_rewards and value is not None:
                    sample_rewards = [sample.get_reward_value(self.eval_args) for sample in _flatten_samples([value])]
                    reward = sum(sample_rewards) / len(sample_rewards)
                else:
                    reward = trace.reward if use_miles_rewards else _trace_eval_reward(trace, reward_key)
                rewards.append(reward)
                truncated.append(trace.is_truncated)
        return RolloutFnEvalOutput(
            data={
                self.config.env_id
                or "verifiers": {
                    "rewards": rewards,
                    "truncated": truncated,
                    "samples": samples,
                }
            },
            metrics=_trace_metrics(all_traces),
        )


_LEGACY_INSTANCES: dict[tuple[int, int, bool], VerifiersRolloutFn] = {}


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """Legacy Miles entrypoint backed by one persistent rollout adapter."""
    from miles.utils.async_utils import run

    # The refactored rollout manager constructs separate train and eval adapters.
    # Preserve that lifecycle under the legacy function interface as well.
    key = (id(args), id(data_source), evaluation)
    adapter = _LEGACY_INSTANCES.get(key)
    if adapter is None:
        adapter = VerifiersRolloutFn(RolloutFnConstructorInput(args=args, data_source=data_source))
        _LEGACY_INSTANCES[key] = adapter
    input = RolloutFnEvalInput(rollout_id) if evaluation else RolloutFnTrainInput(rollout_id)
    return run(adapter(input))
