"""Tinker protocol backend built on Miles' multi-LoRA registry."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from pathlib import Path
from typing import Any

from miles.ray.multi_lora.backend import MultiLoRABackend
from miles.ray.tinker.protocol import (
    CreateModelRequest,
    CreateSamplingSessionRequest,
    CreateSessionRequest,
    ForwardBackwardRequest,
    ForwardRequest,
    LoadWeightsRequest,
    OptimStepRequest,
    SampleRequest,
    SaveWeightsForSamplerRequest,
    SaveWeightsRequest,
    TinkerError,
    UntypedAPIFuture,
    encoded_tokens,
    forward_payload,
)
from miles.ray.tinker.state import Operation, TinkerModelConfig, TinkerState

logger = logging.getLogger(__name__)


class TinkerBackend(MultiLoRABackend):
    """Official Tinker SDK-compatible control plane.

    Training operations are serialized through ``operation_queue`` and run by
    the Miles driver on the Megatron actor group. Sampling requests run
    concurrently against immutable SGLang LoRA snapshots.
    """

    def __init__(self, args: Any, router_url: str) -> None:
        super().__init__(args, router_url)
        self.state = TinkerState()
        self.operation_queue: asyncio.Queue[Operation] = asyncio.Queue()
        self.sample_tasks: set[asyncio.Task] = set()
        self.sample_semaphore = asyncio.Semaphore(getattr(args, "tinker_max_concurrent_samples", 256))
        self.loaded_sampler_adapters: set[str] = set()
        self.ready = False
        root = getattr(args, "tinker_checkpoint_dir", None)
        if root is None:
            root = Path(getattr(args, "save", None) or "/tmp/miles") / "tinker"
        self.checkpoint_root = Path(root).expanduser().resolve()
        self.model_name = getattr(args, "tinker_model_name", None) or str(args.hf_checkpoint).rstrip("/")
        self.tokenizer_id = getattr(args, "tinker_tokenizer_id", None) or self.model_name

    async def close(self) -> None:
        for task in self.sample_tasks:
            task.cancel()
        if self.sample_tasks:
            await asyncio.gather(*self.sample_tasks, return_exceptions=True)
        self.sample_tasks.clear()
        await super().close()

    def mark_ready(self) -> None:
        self.ready = True

    def client_config(self) -> dict[str, Any]:
        return {
            "pjwt_auth_enabled": False,
            "credential_default_source": "api_key",
            "parallel_fwdbwd_chunks": False,
            "proto_write_fwdbwd": False,
            "proto_compress_fwdbwd": False,
            "fwd_via_fwdbwd": False,
            "sample_no_retries": False,
            "sample_enable_stuck_detection": True,
            "sample_max_concurrent_requests": getattr(self.args, "tinker_max_concurrent_samples", 256),
            "use_pyqwest_transport": False,
        }

    def create_session(self, request: CreateSessionRequest) -> dict[str, Any]:
        session = self.state.create_session(
            tags=request.tags,
            user_metadata=request.user_metadata,
            sdk_version=request.sdk_version,
            project_id=request.project_id,
        )
        return {"type": "create_session", "session_id": session.session_id}

    def session_heartbeat(self, session_id: str) -> dict[str, str]:
        self.state.heartbeat(session_id)
        return {"type": "session_heartbeat"}

    async def create_model(self, request: CreateModelRequest) -> UntypedAPIFuture:
        if request.lora_config is None:
            raise TinkerError(
                "Miles Tinker service requires lora_config; full-parameter model creation is not supported",
                category="user",
            )
        self._validate_base_model(request.base_model)
        lora = request.lora_config
        if not 0 < lora.rank <= self.args.lora_rank:
            raise TinkerError(
                f"requested LoRA rank {lora.rank} must be in [1, {self.args.lora_rank}]",
                category="user",
            )
        if not (lora.train_unembed or lora.train_mlp or lora.train_attn):
            raise TinkerError("at least one LoRA module group must be trainable", category="user")
        self._validate_target_modules(lora.train_unembed, lora.train_mlp, lora.train_attn)

        payload = request.model_dump(mode="json", exclude_none=True)
        prospective_id = self.state.model_create_keys.get((request.session_id, request.model_seq_id))
        if prospective_id is None:
            save = self.checkpoint_root / "_models" / "pending"
        else:
            save = self.checkpoint_root / "_models" / prospective_id[1]
        config = TinkerModelConfig(
            rank=lora.rank,
            alpha=lora.rank,
            save=save,
            seed=lora.seed,
            train_unembed=lora.train_unembed,
            train_mlp=lora.train_mlp,
            train_attn=lora.train_attn,
            user_metadata=request.user_metadata or {},
        )
        model, future, is_new = self.state.begin_model_create(
            session_id=request.session_id,
            model_seq_id=request.model_seq_id,
            base_model=request.base_model,
            config=config,
            payload=payload,
        )
        if is_new:
            config = TinkerModelConfig(
                rank=config.rank,
                alpha=config.alpha,
                save=self.checkpoint_root / "_models" / model.model_id,
                seed=config.seed,
                train_unembed=config.train_unembed,
                train_mlp=config.train_mlp,
                train_attn=config.train_attn,
                user_metadata=config.user_metadata,
            )
            model.config = config
            try:
                await self.register(model.model_id, config)
            except Exception as exc:
                self.state.rollback_model_create(model.model_id, future.request_id)
                if isinstance(exc, RuntimeError) and "No free adapter slots" in str(exc):
                    raise TinkerError(
                        "no free Tinker training-client slots; increase --multi-lora-n-adapters",
                        category="user",
                    ) from None
                raise
            await self.operation_queue.put(
                Operation(
                    request_id=future.request_id,
                    kind="create_model",
                    model_id=model.model_id,
                    payload={"model_id": model.model_id},
                )
            )
        return UntypedAPIFuture(request_id=future.request_id, model_id=model.model_id)

    def get_info(self, model_id: str) -> dict[str, Any]:
        model = self.state.require_model(model_id, active=False)
        return {
            "type": "get_info",
            "model_data": {
                "arch": getattr(self.args, "model_arch", None),
                "model_name": model.base_model,
                "tokenizer_id": self.tokenizer_id,
            },
            "model_id": model.model_id,
            "is_lora": True,
            "lora_rank": model.config.rank,
            "model_name": model.base_model,
        }

    async def unload_model(self, model_id: str) -> UntypedAPIFuture:
        future, operation = self.state.begin_unload(model_id)
        if operation is not None:
            await self.deregister(model_id)
            await self.operation_queue.put(operation)
        return UntypedAPIFuture(request_id=future.request_id, model_id=model_id)

    async def forward(self, request: ForwardRequest) -> UntypedAPIFuture:
        payload = forward_payload(request.forward_input)
        return await self._submit_model_operation(request.model_id, request.seq_id, "forward", payload)

    async def forward_backward(self, request: ForwardBackwardRequest) -> UntypedAPIFuture:
        payload = forward_payload(request.forward_backward_input)
        return await self._submit_model_operation(request.model_id, request.seq_id, "forward_backward", payload)

    async def optim_step(self, request: OptimStepRequest) -> UntypedAPIFuture:
        params = request.adam_params
        values = (
            params.learning_rate,
            params.beta1,
            params.beta2,
            params.eps,
            params.weight_decay,
            params.grad_clip_norm,
        )
        if not all(math.isfinite(value) for value in values):
            raise TinkerError("Adam parameters must be finite", category="user")
        if params.learning_rate < 0:
            raise TinkerError("learning_rate must be non-negative", category="user")
        if not (0 <= params.beta1 < 1 and 0 <= params.beta2 < 1):
            raise TinkerError("Adam betas must be in [0, 1)", category="user")
        if params.eps < 0 or params.weight_decay < 0 or params.grad_clip_norm < 0:
            raise TinkerError("eps, weight_decay and grad_clip_norm must be non-negative", category="user")
        return await self._submit_model_operation(
            request.model_id,
            request.seq_id,
            "optim_step",
            {"adam_params": params.model_dump(mode="json")},
        )

    async def save_weights(self, request: SaveWeightsRequest) -> UntypedAPIFuture:
        idempotency_payload = request.model_dump(mode="json", exclude={"seq_id"})
        seq_id, duplicate = self.state.validate_model_operation(
            model_id=request.model_id,
            seq_id=request.seq_id,
            kind="save_weights",
            payload=idempotency_payload,
        )
        if duplicate is not None:
            return UntypedAPIFuture(request_id=duplicate.request_id, model_id=request.model_id)
        checkpoint = self.state.allocate_checkpoint(
            model_id=request.model_id,
            seq_id=seq_id,
            requested_name=request.path,
            checkpoint_type="training",
            ttl_seconds=request.ttl_seconds,
            overwrite=request.overwrite,
        )
        return await self._submit_model_operation(
            request.model_id,
            request.seq_id,
            "save_weights",
            {
                "tinker_path": checkpoint.tinker_path,
                "local_path": str(checkpoint.local_path),
                "checkpoint_step": checkpoint.checkpoint_step,
                "include_optimizer": True,
                "overwrite": request.overwrite,
            },
            idempotency_payload=idempotency_payload,
        )

    async def load_weights(self, request: LoadWeightsRequest) -> UntypedAPIFuture:
        checkpoint = self.state.require_checkpoint(request.path)
        model = self.state.require_model(request.model_id)
        source = self.state.require_model(checkpoint.model_id, active=False)
        if source.base_model != model.base_model:
            raise TinkerError(
                f"checkpoint base model {source.base_model!r} does not match {model.base_model!r}",
                category="user",
            )
        if source.config.rank != model.config.rank:
            raise TinkerError(
                f"checkpoint LoRA rank {source.config.rank} does not match {model.config.rank}",
                category="user",
            )
        source_groups = (
            source.config.train_unembed,
            source.config.train_mlp,
            source.config.train_attn,
        )
        model_groups = (
            model.config.train_unembed,
            model.config.train_mlp,
            model.config.train_attn,
        )
        if source_groups != model_groups:
            raise TinkerError(
                "checkpoint train_unembed/train_mlp/train_attn configuration does not match the target model",
                category="user",
            )
        if request.optimizer and not checkpoint.include_optimizer:
            raise TinkerError(f"checkpoint {request.path!r} has no optimizer state", category="user")
        return await self._submit_model_operation(
            request.model_id,
            request.seq_id,
            "load_weights",
            {
                "tinker_path": checkpoint.tinker_path,
                "local_path": str(checkpoint.local_path),
                "optimizer": request.optimizer,
            },
        )

    async def save_weights_for_sampler(self, request: SaveWeightsForSamplerRequest) -> UntypedAPIFuture:
        idempotency_payload = request.model_dump(mode="json", exclude={"seq_id"})
        seq_id, duplicate = self.state.validate_model_operation(
            model_id=request.model_id,
            seq_id=request.seq_id,
            kind="save_weights_for_sampler",
            payload=idempotency_payload,
        )
        if duplicate is not None:
            return UntypedAPIFuture(request_id=duplicate.request_id, model_id=request.model_id)
        checkpoint = self.state.allocate_checkpoint(
            model_id=request.model_id,
            seq_id=seq_id,
            requested_name=request.path,
            checkpoint_type="sampler",
            ttl_seconds=request.ttl_seconds,
            overwrite=False,
        )
        return await self._submit_model_operation(
            request.model_id,
            request.seq_id,
            "save_weights_for_sampler",
            {
                "tinker_path": checkpoint.tinker_path,
                "local_path": str(checkpoint.local_path),
                "checkpoint_step": checkpoint.checkpoint_step,
                "include_optimizer": False,
                "sampling_session_seq_id": request.sampling_session_seq_id,
            },
            idempotency_payload=idempotency_payload,
        )

    async def create_sampling_session(self, request: CreateSamplingSessionRequest) -> dict[str, str]:
        self.state.require_session(request.session_id)
        if request.model_path is None:
            assert request.base_model is not None
            self._validate_base_model(request.base_model)
            base_model = request.base_model
            adapter_path = None
            adapter_name = None
        else:
            checkpoint = self.state.require_checkpoint(request.model_path)
            source = self.state.require_model(checkpoint.model_id, active=False)
            base_model = source.base_model
            adapter_path = checkpoint.local_path
            adapter_name = await self._ensure_sampler_loaded(checkpoint.tinker_path, adapter_path)

        session, _ = self.state.create_sampling_session(
            session_id=request.session_id,
            sampling_session_seq_id=request.sampling_session_seq_id,
            base_model=base_model,
            model_path=request.model_path,
            adapter_path=adapter_path,
            adapter_name=adapter_name,
            payload=request.model_dump(mode="json", exclude_none=True),
        )
        return {
            "type": "create_sampling_session",
            "sampling_session_id": session.sampling_session_id,
        }

    def get_sampler(self, sampling_session_id: str) -> dict[str, Any]:
        try:
            session = self.state.sampling_sessions[sampling_session_id]
        except KeyError:
            raise TinkerError(
                f"sampling session {sampling_session_id!r} does not exist",
                category="user",
            ) from None
        return {
            "sampler_id": session.sampling_session_id,
            "base_model": session.base_model,
            "model_path": session.model_path,
        }

    async def sample(self, request: SampleRequest) -> UntypedAPIFuture:
        payload = {
            "num_samples": request.num_samples,
            "prompt": encoded_tokens(request.prompt),
            "sampling_params": request.sampling_params.model_dump(mode="json", exclude_none=True),
            "prompt_logprobs": bool(request.prompt_logprobs),
            "topk_prompt_logprobs": request.topk_prompt_logprobs,
        }
        if request.num_samples <= 0:
            raise TinkerError("num_samples must be positive", category="user")
        if request.topk_prompt_logprobs < 0:
            raise TinkerError("topk_prompt_logprobs must be non-negative", category="user")
        sampling = request.sampling_params
        if sampling.max_tokens is not None and sampling.max_tokens < 0:
            raise TinkerError("max_tokens must be non-negative", category="user")
        if not math.isfinite(sampling.temperature) or sampling.temperature < 0:
            raise TinkerError("temperature must be finite and non-negative", category="user")
        if not math.isfinite(sampling.top_p) or not 0 < sampling.top_p <= 1:
            raise TinkerError("top_p must be finite and in (0, 1]", category="user")
        if sampling.top_k == 0 or sampling.top_k < -1:
            raise TinkerError("top_k must be -1 or a positive integer", category="user")
        prompt_length = len(payload["prompt"])
        if prompt_length > self.args.seq_length:
            raise TinkerError(
                f"prompt has {prompt_length} tokens, exceeding the configured sequence length {self.args.seq_length}",
                category="user",
            )
        if sampling.max_tokens is not None and prompt_length + sampling.max_tokens > self.args.seq_length:
            raise TinkerError(
                f"prompt plus max_tokens exceeds the configured sequence length {self.args.seq_length}",
                category="user",
            )

        if request.sampling_session_id is not None:
            if request.base_model is not None or request.model_path is not None:
                raise TinkerError(
                    "sampling_session_id cannot be combined with base_model or model_path",
                    category="user",
                )
            future, operation = self.state.submit_sample(
                sampling_session_id=request.sampling_session_id,
                seq_id=request.seq_id,
                payload=payload,
            )
        else:
            if (request.base_model is None) == (request.model_path is None):
                raise TinkerError(
                    "exactly one of sampling_session_id, base_model or model_path is required",
                    category="user",
                )
            adapter_path = None
            adapter_name = None
            if request.model_path is not None:
                checkpoint = self.state.require_checkpoint(request.model_path)
                adapter_path = checkpoint.local_path
                adapter_name = await self._ensure_sampler_loaded(checkpoint.tinker_path, adapter_path)
            else:
                assert request.base_model is not None
                self._validate_base_model(request.base_model)
            future = self.state.new_future(model_id=None)
            operation = Operation(
                request_id=future.request_id,
                kind="sample",
                model_id=None,
                payload={
                    **payload,
                    "adapter_path": str(adapter_path) if adapter_path is not None else None,
                    "adapter_name": adapter_name,
                },
            )

        if operation is not None:
            task = asyncio.create_task(self._run_sample(operation))
            self.sample_tasks.add(task)
            task.add_done_callback(self.sample_tasks.discard)
        return UntypedAPIFuture(request_id=future.request_id)

    async def next_operation(self, timeout_s: float | None = None) -> dict[str, Any] | None:
        if timeout_s is None:
            operation = await self.operation_queue.get()
        else:
            try:
                operation = await asyncio.wait_for(self.operation_queue.get(), timeout_s)
            except TimeoutError:
                return None
        return operation.asdict()

    async def complete_operation(self, request_id: str, result: dict[str, Any] | None) -> None:
        future = self.state.require_future(request_id)
        operation_model = self.state.models.get(future.model_id) if future.model_id is not None else None
        result = result or {}
        kind = result.pop("_operation_kind", None)
        if kind == "create_model":
            assert operation_model is not None
            operation_model.status = "active"
            response = {"type": "create_model", "model_id": operation_model.model_id}
        elif kind == "unload_model":
            assert operation_model is not None
            operation_model.status = "unloaded"
            response = {"type": "unload_model", "model_id": operation_model.model_id}
        elif kind == "save_weights":
            self.state.complete_checkpoint(result["tinker_path"])
            response = {"type": "save_weights", "path": result["tinker_path"]}
        elif kind == "load_weights":
            response = {"type": "load_weights", "path": result["tinker_path"]}
        elif kind == "save_weights_for_sampler":
            tinker_path = result["tinker_path"]
            checkpoint = self.state.complete_checkpoint(tinker_path)
            sampling_seq_id = result.get("sampling_session_seq_id")
            if sampling_seq_id is None:
                response = {"type": "save_weights_for_sampler", "path": tinker_path}
            else:
                adapter_name = await self._ensure_sampler_loaded(tinker_path, checkpoint.local_path)
                assert operation_model is not None
                session, _ = self.state.create_sampling_session(
                    session_id=operation_model.session_id,
                    sampling_session_seq_id=sampling_seq_id,
                    base_model=operation_model.base_model,
                    model_path=tinker_path,
                    adapter_path=checkpoint.local_path,
                    adapter_name=adapter_name,
                    payload={"model_path": tinker_path, "ephemeral": True},
                )
                response = {
                    "type": "save_weights_for_sampler",
                    "sampling_session_id": session.sampling_session_id,
                }
        else:
            response = result
        self.state.complete_future(request_id, response)

    async def fail_operation(self, request_id: str, error: str, category: str = "server") -> None:
        future = self.state.require_future(request_id)
        model = self.state.models.get(future.model_id) if future.model_id is not None else None
        was_loading = model is not None and model.status == "loading"
        was_unloading = model is not None and model.status == "unloading"
        valid_category = category if category in {"unknown", "server", "user"} else "server"
        self.state.fail_future(request_id, error, valid_category)
        self.state.fail_checkpoint_for_request(request_id)
        if was_loading:
            await self.deregister(model.model_id)
        elif was_unloading:
            model.status = "failed"

    def retrieve_future(self, request_id: str) -> dict[str, Any]:
        return self.state.retrieve_future(request_id)

    def weights_info(self, tinker_path: str) -> dict[str, Any]:
        checkpoint = self.state.require_checkpoint(tinker_path)
        model = self.state.require_model(checkpoint.model_id, active=False)
        return {
            "base_model": model.base_model,
            "is_lora": True,
            "lora_rank": model.config.rank,
            "train_unembed": model.config.train_unembed,
            "train_mlp": model.config.train_mlp,
            "train_attn": model.config.train_attn,
        }

    async def _submit_model_operation(
        self,
        model_id: str,
        seq_id: int | None,
        kind: str,
        payload: dict[str, Any],
        *,
        idempotency_payload: dict[str, Any] | None = None,
    ) -> UntypedAPIFuture:
        self.state.require_model(model_id)
        registry_record = self.registry.find(model_id)
        if registry_record is None:
            raise TinkerError(f"model {model_id!r} has no live adapter slot", category="server")
        worker_payload = {**payload, "slot": registry_record.slot, "model_id": model_id}
        future, operation = self.state.submit_model_operation(
            model_id=model_id,
            seq_id=seq_id,
            kind=kind,
            payload=worker_payload,
            idempotency_payload=idempotency_payload,
        )
        if operation is not None:
            await self.operation_queue.put(operation)
        return UntypedAPIFuture(request_id=future.request_id, model_id=model_id)

    def _validate_base_model(self, base_model: str) -> None:
        accepted = {
            self.model_name.rstrip("/"),
            str(self.args.hf_checkpoint).rstrip("/"),
            Path(str(self.args.hf_checkpoint).rstrip("/")).name,
        }
        if base_model.rstrip("/") not in accepted:
            raise TinkerError(
                f"this Miles instance serves {self.model_name!r}, not {base_model!r}",
                category="user",
            )

    def _validate_target_modules(self, train_unembed: bool, train_mlp: bool, train_attn: bool) -> None:
        raw_targets = getattr(self.args, "target_modules", []) or []
        if isinstance(raw_targets, str):
            raw_targets = [part.strip() for part in raw_targets.split(",")]
        leaves = {str(value).rsplit(".", 1)[-1] for value in raw_targets}
        all_linear = bool(leaves & {"all", "all-linear", "all_linear"})
        if train_unembed and not leaves & {"output_layer", "lm_head", "unembed_tokens"}:
            raise TinkerError(
                "train_unembed=True requires output_layer, lm_head, or unembed_tokens in --target-modules",
                category="user",
            )
        if train_attn and not (
            all_linear
            or leaves
            & {
                "linear_qkv",
                "linear_q",
                "linear_k",
                "linear_v",
                "linear_proj",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            }
        ):
            raise TinkerError("train_attn=True requires attention LoRA target modules", category="user")
        if train_mlp and not (
            all_linear
            or leaves
            & {
                "linear_fc1",
                "linear_fc1_up",
                "linear_fc1_gate",
                "linear_fc2",
                "gate_proj",
                "up_proj",
                "down_proj",
            }
        ):
            raise TinkerError("train_mlp=True requires MLP LoRA target modules", category="user")

    async def _ensure_sampler_loaded(self, tinker_path: str, adapter_path: Path) -> str:
        # Include the immutable local snapshot path in the name. A named
        # tinker:// URI may be overwritten by a later checkpoint, and reusing
        # the URI-only name would leave SGLang serving the old adapter.
        identity = f"{tinker_path}\0{adapter_path.resolve()}"
        adapter_name = f"tinker-{hashlib.sha256(identity.encode()).hexdigest()[:24]}"
        await self._load_sampler_adapter(adapter_name, adapter_path)
        return adapter_name

    async def _load_sampler_adapter(self, adapter_name: str, adapter_path: Path) -> None:
        """Load or refresh one immutable adapter snapshot on every worker."""
        urls = await self.worker_urls()
        if not urls:
            raise TinkerError("no SGLang workers are available for sampling", category="server")
        assert self.client is not None
        responses = await asyncio.gather(
            *(
                self.client.post(
                    f"{url}/load_lora_adapter",
                    json={
                        "lora_name": adapter_name,
                        "lora_path": str(adapter_path),
                        "pinned": False,
                    },
                )
                for url in urls
            ),
            return_exceptions=True,
        )
        failures = []
        for response in responses:
            if isinstance(response, Exception):
                failures.append(str(response))
            elif response.status_code >= 400:
                body = response.text.lower()
                if "already" not in body:
                    failures.append(response.text)
        if failures:
            raise TinkerError(
                f"failed to load sampler snapshot on {len(failures)}/{len(urls)} workers: {failures[0]}",
                category="server",
            )
        # Deliberately issue load_lora_adapter on every use. Unpinned SGLang
        # adapters may be evicted independently of this API actor.
        self.loaded_sampler_adapters.add(adapter_name)

    async def _run_sample(self, operation: Operation) -> None:
        try:
            async with self.sample_semaphore:
                adapter_name = operation.payload.get("adapter_name")
                adapter_path = operation.payload.get("adapter_path")
                if adapter_name is not None and adapter_path is not None:
                    await self._load_sampler_adapter(adapter_name, Path(adapter_path))
                response = await self._sample_payload(operation.payload)
            self.state.complete_future(operation.request_id, response)
        except TinkerError as exc:
            self.state.fail_future(operation.request_id, str(exc), exc.category)
        except Exception as exc:
            logger.exception("Tinker sampling request failed")
            self.state.fail_future(operation.request_id, str(exc), "server")

    async def _sample_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.client is not None
        params = dict(payload["sampling_params"])
        max_tokens = params.pop("max_tokens", None)
        sglang_params: dict[str, Any] = {
            "temperature": params.pop("temperature", 1.0),
            "top_k": params.pop("top_k", -1),
            "top_p": params.pop("top_p", 1.0),
        }
        if max_tokens is not None:
            sglang_params["max_new_tokens"] = max_tokens
        seed = params.pop("seed", None)
        stop = params.pop("stop", None)
        if isinstance(stop, list) and (not stop or isinstance(stop[0], int)):
            sglang_params["stop_token_ids"] = stop
        elif stop is not None:
            sglang_params["stop"] = stop
        if params:
            raise TinkerError(f"unsupported sampling parameters: {sorted(params)}", category="user")

        prompt_logprobs = payload["prompt_logprobs"]
        topk_prompt_logprobs = payload["topk_prompt_logprobs"]
        requests = []
        for index in range(payload["num_samples"]):
            sample_params = dict(sglang_params)
            if seed is not None:
                sample_params["sampling_seed"] = seed + index
            body = {
                "input_ids": payload["prompt"],
                "sampling_params": sample_params,
                "return_logprob": True,
                "logprob_start_len": 0 if prompt_logprobs or topk_prompt_logprobs else -1,
                "top_logprobs_num": topk_prompt_logprobs,
            }
            if payload.get("adapter_name") is not None:
                body["lora_path"] = payload["adapter_name"]
            requests.append(self.client.post(f"{self.router_url}/generate", json=body))
        responses = await asyncio.gather(*requests)
        bodies = []
        for response in responses:
            if response.status_code >= 400:
                raise TinkerError(f"SGLang sampling failed: {response.text}", category="server")
            bodies.append(response.json())

        sequences = []
        for body in bodies:
            meta = body.get("meta_info", {})
            output_logprobs = meta.get("output_token_logprobs") or []
            finish = meta.get("finish_reason") or {}
            finish_type = finish.get("type") if isinstance(finish, dict) else finish
            output_ids = body.get("output_ids")
            if output_ids is None:
                output_ids = [int(item[1]) for item in output_logprobs]
            sequences.append(
                {
                    "stop_reason": "stop" if finish_type == "stop" else "length",
                    "tokens": output_ids,
                    "logprobs": [float(item[0]) for item in output_logprobs],
                }
            )

        first_meta = bodies[0].get("meta_info", {}) if bodies else {}
        response_body: dict[str, Any] = {
            "type": "sample",
            "sequences": sequences,
            "prompt_cache_hit_tokens": int(first_meta.get("cached_tokens", first_meta.get("cache_hit_tokens", 0)) or 0),
        }
        if prompt_logprobs:
            raw = first_meta.get("input_token_logprobs") or []
            response_body["prompt_logprobs"] = [None if item[0] is None else float(item[0]) for item in raw]
        if topk_prompt_logprobs:
            raw_topk = first_meta.get("input_top_logprobs") or []
            response_body["topk_prompt_logprobs"] = [None if entries is None else [(int(item[1]), float(item[0])) for item in entries] for entries in raw_topk]
        return response_body
