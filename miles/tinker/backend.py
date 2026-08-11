"""Request-driven Tinker facade over a live Miles multi-LoRA actor group."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import httpx

from miles.ray.rollout.train_data_conversion import split_train_data_by_dp
from miles.utils.multi_lora import slot_lora_name


def _tensor_data(value: dict[str, Any] | None, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return list(default or [])
    return list(value.get("data") or default or [])


def _tokens(model_input: dict[str, Any]) -> list[int]:
    chunks = model_input.get("chunks", [])
    if any(chunk.get("type") != "encoded_text" for chunk in chunks):
        raise ValueError("Miles Tinker phase 1 supports encoded-text chunks only")
    return [token for chunk in chunks for token in chunk.get("tokens", [])]


def _float_tensor(values: list[Any]) -> dict[str, Any]:
    data = [float(value) for value in values]
    return {"data": data, "dtype": "float32", "shape": [len(data)]}


class MilesTinkerBackend:
    """Translate Tinker requests into collective Miles actor calls.

    This object lives in the Miles driver process. API state remains in the
    FastAPI app; this class owns only live model-to-slot execution state.
    """

    def __init__(self, args, actor_group, controller, router_url: str):
        self.args = args
        self.actor_group = actor_group
        self.controller = controller
        self.router_url = router_url.rstrip("/")
        self.models: dict[str, dict[str, Any]] = {}
        self.accumulated_samples: defaultdict[str, int] = defaultdict(int)
        self.request_seq = 0

    async def create_model(self, model_id: str, lora_config: dict[str, Any], model_role: str) -> dict[str, Any]:
        if model_role != "policy":
            raise ValueError("Miles Tinker phase 1 supports policy LoRAs only")
        rank = int(lora_config["rank"])
        result = await self.controller.register_external_adapter.remote(model_id, rank, rank)
        self.models[model_id] = {"slot": result["slot"], "rank": rank}
        await self.actor_group.reconcile_adapters()
        await self.actor_group.update_weights()
        return {"model_id": model_id}

    def _training_data(self, model_id: str, request: dict[str, Any]) -> dict[str, Any]:
        slot = self.models[model_id]["slot"]
        rows = request["data"]
        loss_fn = request["loss_fn"]
        data: dict[str, Any] = {
            "dynamic_global_batch_size": len(rows),
            "tokens": [],
            "response_lengths": [],
            "loss_masks": [],
            "rewards": [],
            "truncated": [],
            "sample_indices": [],
            "adapter_slots": [],
        }
        advantages = []
        rollout_log_probs = []
        for index, row in enumerate(rows):
            input_tokens = _tokens(row["model_input"])
            inputs = row.get("loss_fn_inputs", {})
            targets = _tensor_data(inputs.get("target_tokens"))
            if len(targets) != len(input_tokens) or not targets:
                raise ValueError("target_tokens must align one-for-one with model_input tokens")
            # Miles batches carry the full sequence and shift internally, while
            # Tinker carries input tokens plus a same-length next-token target.
            tokens = [*input_tokens, int(targets[-1])]
            if loss_fn in {"importance_sampling", "ppo"}:
                token_weights = [float(x) for x in _tensor_data(inputs.get("advantages"))]
            else:
                if targets[:-1] != input_tokens[1:]:
                    raise ValueError("cross_entropy requires standard next-token target_tokens")
                token_weights = [
                    float(x) for x in _tensor_data(inputs.get("weights"), [1.0] * len(targets))
                ]
            if len(token_weights) != len(targets):
                raise ValueError("per-token weights must align with target_tokens")
            first_train = next(
                (i for i, weight in enumerate(token_weights) if weight != 0), len(token_weights) - 1
            )
            response_length = max(1, len(token_weights) - first_train)
            response_weights = token_weights[-response_length:]
            data["tokens"].append(tokens)
            data["response_lengths"].append(response_length)
            data["loss_masks"].append([int(weight != 0) for weight in response_weights])
            data["rewards"].append(0.0)
            data["truncated"].append(0)
            data["sample_indices"].append(index)
            data["adapter_slots"].append(slot)
            if loss_fn in {"importance_sampling", "ppo"}:
                advantages.append(response_weights)
                rollout_log_probs.append(
                    [float(x) for x in _tensor_data(inputs.get("logprobs"))][-response_length:]
                )
        if loss_fn in {"importance_sampling", "ppo"}:
            data["advantages"] = advantages
            data["rollout_log_probs"] = rollout_log_probs
        return data

    async def forward_backward(self, model_id: str, batch: dict[str, Any]) -> dict[str, Any]:
        if model_id not in self.models:
            raise ValueError(f"unknown model_id {model_id}")
        if batch["loss_fn"] not in {"cross_entropy", "importance_sampling", "ppo"}:
            raise ValueError(f"unsupported Tinker loss_fn: {batch['loss_fn']}")
        data = self._training_data(model_id, batch)
        refs = split_train_data_by_dp(self.args, data, self.args.multi_lora_dp_size)
        self.request_seq += 1
        # Tinker returns the new-policy token logprobs from the same logical
        # forward/backward request. Miles' training primitive does not expose
        # those tensors yet, so score immediately before accumulating grads.
        forward_results = await self.actor_group.external_forward(self.request_seq, refs)
        self.request_seq += 1
        rank_results = await self.actor_group.external_forward_backward(self.request_seq, refs, batch["loss_fn"])
        metrics: dict[str, float] = {}
        for result in rank_results:
            for key, value in result.get("metrics", {}).items():
                if hasattr(value, "item"):
                    value = value.item()
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)
        self.accumulated_samples[model_id] += len(batch["data"])
        ordered: list[Any | None] = [None] * len(batch["data"])
        for result in forward_results:
            for index, log_probs in zip(
                result.get("sample_indices", []), result.get("log_probs", []), strict=True
            ):
                if hasattr(log_probs, "detach"):
                    log_probs = log_probs.detach().cpu().tolist()
                ordered[int(index)] = log_probs
        if any(value is None for value in ordered):
            raise RuntimeError("Miles forward_backward did not return logprobs for every datum")
        return {
            "loss_fn_output_type": batch["loss_fn"],
            "loss_fn_outputs": [{"logprobs": _float_tensor(value)} for value in ordered],
            "metrics": metrics,
        }

    async def forward(self, model_id: str, batch: dict[str, Any]) -> dict[str, Any]:
        if model_id not in self.models:
            raise ValueError(f"unknown model_id {model_id}")
        if batch["loss_fn"] != "cross_entropy":
            raise ValueError("Miles Tinker forward currently supports cross_entropy only")

        slot = self.models[model_id]["slot"]
        data: dict[str, Any] = {
            "dynamic_global_batch_size": len(batch["data"]),
            "tokens": [],
            "response_lengths": [],
            "loss_masks": [],
            "rewards": [],
            "truncated": [],
            "sample_indices": [],
            "adapter_slots": [],
        }
        for index, row in enumerate(batch["data"]):
            input_tokens = _tokens(row["model_input"])
            targets = _tensor_data(row.get("loss_fn_inputs", {}).get("target_tokens"))
            if len(targets) != len(input_tokens) or not targets:
                raise ValueError("target_tokens must align one-for-one with model_input tokens")
            data["tokens"].append([*input_tokens, int(targets[-1])])
            data["response_lengths"].append(len(targets))
            data["loss_masks"].append([1] * len(targets))
            data["rewards"].append(0.0)
            data["truncated"].append(0)
            data["sample_indices"].append(index)
            data["adapter_slots"].append(slot)

        refs = split_train_data_by_dp(self.args, data, self.args.multi_lora_dp_size)
        self.request_seq += 1
        rank_results = await self.actor_group.external_forward(self.request_seq, refs)
        ordered: list[Any | None] = [None] * len(batch["data"])
        for result in rank_results:
            for index, log_probs in zip(
                result.get("sample_indices", []), result.get("log_probs", []), strict=True
            ):
                if hasattr(log_probs, "detach"):
                    log_probs = log_probs.detach().cpu().tolist()
                ordered[int(index)] = log_probs
        if any(value is None for value in ordered):
            raise RuntimeError("Miles forward did not return logprobs for every datum")
        return {
            "loss_fn_output_type": batch["loss_fn"],
            "loss_fn_outputs": [{"logprobs": _float_tensor(value)} for value in ordered],
            "metrics": {},
        }

    async def optim_step(self, model_id: str, adam_params: dict[str, float]) -> dict[str, Any]:
        state = self.models[model_id]
        batch_size = self.accumulated_samples.pop(model_id, 0)
        if batch_size <= 0:
            raise ValueError("optim_step requires a preceding forward_backward")
        results = await self.actor_group.external_optim_step(
            model_id, state["slot"], batch_size, adam_params
        )
        await self.controller.advance_external_step.remote(model_id)
        await self.actor_group.update_weights()
        return {"metrics": next((result for result in results if result), {})}

    async def save_sampler(self, model_id: str, checkpoint_id: str) -> dict[str, Any]:
        await self.actor_group.update_weights()
        return {"path": f"tinker://{model_id}/sampler_weights/{checkpoint_id}"}

    async def save_checkpoint(self, model_id: str, checkpoint_id: str) -> dict[str, Any]:
        await self.actor_group.save_model(self.request_seq, force_sync=True)
        return {"checkpoint_id": checkpoint_id}

    async def load_checkpoint(self, model_id: str, path: str) -> dict[str, Any]:
        raise ValueError("arbitrary Tinker checkpoint loading is not implemented in Miles phase 1")

    async def sample(self, model_id: str | None, request: dict[str, Any]) -> dict[str, Any]:
        if model_id is None:
            raise ValueError("sampling requires a model-backed sampling session")
        tokens = _tokens(request["prompt"])
        params = dict(request.get("sampling_params") or {})
        # Tinker follows OpenAI naming while SGLang's native /generate
        # endpoint uses its internal sampling names.
        if "max_tokens" in params:
            params["max_new_tokens"] = params.pop("max_tokens")
        if "seed" in params:
            params["sampling_seed"] = params.pop("seed")
        num_samples = int(request.get("num_samples", 1))
        payload = {
            "input_ids": tokens,
            "sampling_params": params,
            "lora_path": slot_lora_name(self.models[model_id]["slot"]),
            "n": num_samples,
            "return_logprob": True,
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(f"{self.router_url}/generate", json=payload)
            response.raise_for_status()
            generated = response.json()
        outputs = generated if isinstance(generated, list) else [generated]
        sequences = []
        for output in outputs:
            meta = output.get("meta_info", {})
            token_logprobs = meta.get("output_token_logprobs", [])
            output_tokens = output.get("output_ids") or [item[1] for item in token_logprobs]
            output_logprobs = [item[0] if isinstance(item, (list, tuple)) else item for item in token_logprobs]
            sequences.append(
                {
                    "tokens": output_tokens,
                    "logprobs": output_logprobs,
                    "stop_reason": meta.get("finish_reason", {}).get("type", "length"),
                }
            )
        return {"sequences": sequences}

    async def delete_model(self, model_id: str) -> None:
        await self.controller.deregister_adapter.remote(model_id)
        await self.actor_group.reconcile_adapters()
        self.models.pop(model_id, None)
        self.accumulated_samples.pop(model_id, None)
