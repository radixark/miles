"""Translate gateway datums to trainer batches and sampling requests to SGLang."""

import asyncio
import uuid

from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_VALUE_SPEC
from miles.tinker.core.types import UserInputError
from miles.utils import object_store
from miles.utils.http_utils import post

# internal datum key -> trainer batch key
DATUM_TO_BATCH_KEYS = {"weights": "loss_weights", "advantages": "advantages", "sampling_logprobs": "rollout_log_probs"}


def _pad_to_dp_multiple(slot_datums: list, dp_size: int) -> list:
    """Equalize DP shares with zero-mask datums; their outputs are dropped after the loss pass."""
    remainder = len(slot_datums) % dp_size
    if remainder == 0:
        return slot_datums
    slot, datum = slot_datums[-1]
    filler = dict(datum) | {"padding": True}
    return slot_datums + [(slot, filler)] * (dp_size - remainder)


def _build_train_data(slot_datums: list) -> dict:
    """Miles response_lengths select label positions here, including prompt targets."""
    datums = [datum for _, datum in slot_datums]
    train_data = {
        "tokens": [datum["tokens"] for datum in datums],
        "target_tokens": [datum["target_tokens"] for datum in datums],
        "loss_masks": [[0 if datum.get("padding") else 1] * datum["target_len"] for datum in datums],
        "response_lengths": [datum["target_len"] for datum in datums],
        "total_lengths": [len(datum["tokens"]) for datum in datums],
        "sample_indices": list(range(len(datums))),
        "adapter_slots": [slot for slot, _ in slot_datums],
        "dynamic_global_batch_size": len(datums),
    }
    for datum_key, batch_key in DATUM_TO_BATCH_KEYS.items():
        if datum_key in datums[0]:
            train_data[batch_key] = [datum[datum_key] for datum in datums]
    return train_data


class MilesBackend:
    def __init__(self, trainer, router_url: str, dp_size: int = 1) -> None:
        self.trainer = trainer
        self.router_url = router_url
        self.dp_size = dp_size

    def trainer_dead(self) -> bool:
        return self.trainer.has_errored_cell()

    async def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> None:
        _raise_slot_errors(
            await self.trainer.load_slot(slot, rank, alpha, ckpt_path=ckpt_path, load_optimizer=load_optimizer)
        )

    async def unload_slot(self, slot: int) -> None:
        _raise_slot_errors(await self.trainer.unload_slot(slot))

    async def forward_backward(
        self, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict
    ) -> list[dict]:
        return await self._run_loss_pass("forward_backward", batch_id, slot_datums, loss_fn, loss_fn_config)

    async def forward_only(self, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict) -> list[dict]:
        return await self._run_loss_pass("forward_only", batch_id, slot_datums, loss_fn, loss_fn_config)

    async def _run_loss_pass(
        self, method: str, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict
    ) -> list[dict]:
        train_data = _build_train_data(_pad_to_dp_multiple(slot_datums, self.dp_size))
        train_data["loss_fn"] = loss_fn
        train_data["loss_fn_config"] = loss_fn_config
        worker_results = await self._run_batch(method, batch_id, train_data)
        by_index: dict[int, dict] = {}
        for worker_result in worker_results:
            for datum_output in worker_result["per_datum"]:
                index = int(datum_output["sample_index"])
                if index not in by_index:
                    by_index[index] = {
                        "loss": float(datum_output["loss"]),
                        "logprobs": datum_output["logprobs"].tolist(),
                    }
        return [by_index[index] for index in range(len(slot_datums))]

    async def _run_batch(self, method: str, batch_id: int, train_data: dict) -> list:
        store = object_store.get_instance()
        data_ref = store.put(value=train_data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        try:
            return await getattr(self.trainer, method)(batch_id=batch_id, data_ref=data_ref)
        finally:
            store.remove(data_ref)

    async def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        worker_results = await self.trainer.optim_step(adam_params_by_slot=adam_params_by_slot)
        return worker_results[0]

    async def zero_grads(self, slot: int) -> None:
        await self.trainer.zero_grads(slot=slot)

    async def save_slot(self, slot: int, path: str) -> None:
        _raise_slot_errors(await self.trainer.save_slot(slot=slot, path=path))

    async def export_slot(self, slot: int, rank: int, alpha: float, path: str) -> None:
        _raise_slot_errors(await self.trainer.export_slot(slot=slot, rank=rank, alpha=alpha, path=path))

    async def push_slot(
        self, slot: int, lora_name: str, rank: int, alpha: float, lora_path: str | None = None
    ) -> None:
        await self.trainer.push_slot(slot=slot, lora_name=lora_name, rank=rank, alpha=alpha, lora_path=lora_path)

    # -------- sampling --------

    async def sample(self, payload: dict, lora_name: str | None, lora_path: str | None = None) -> dict:
        request = self._generate_request(payload, lora_name, lora_path)
        responses = await asyncio.gather(
            *[
                post(f"{self.router_url}/generate", _with_sample_seed(request, index))
                for index in range(payload["num_samples"])
            ]
        )
        result = {"sequences": [_to_sequence(response) for response in responses]}
        if payload["prompt_logprobs"]:
            result["prompt_logprobs"] = _prompt_logprobs(responses[0])
        if payload["topk_prompt_logprobs"]:
            result["topk_prompt_logprobs"] = _topk_prompt_logprobs(responses[0], payload["topk_prompt_logprobs"])
        return result

    def _generate_request(self, payload: dict, lora_name: str | None, lora_path: str | None = None) -> dict:
        params = payload["sampling_params"]
        max_tokens = params.get("max_tokens")
        if max_tokens is None:
            raise UserInputError("sampling_params.max_tokens is required")
        sampling_params = {
            "max_new_tokens": max_tokens,
            "temperature": params.get("temperature", 1.0),
            "top_p": params.get("top_p", 1.0),
            "top_k": params.get("top_k", -1),
        }
        if params.get("seed") is not None:
            sampling_params["sampling_seed"] = params["seed"]
        stop = params.get("stop")
        if stop is not None:
            if not stop:
                # tinker defines stop=[] as disabling every stop token, EOS included
                sampling_params["ignore_eos"] = True
            elif isinstance(stop, list) and isinstance(stop[0], int):
                sampling_params["stop_token_ids"] = stop
            else:
                sampling_params["stop"] = stop
        request = {"input_ids": payload["prompt_tokens"], "sampling_params": sampling_params, "return_logprob": True}
        if payload["prompt_logprobs"] or payload["topk_prompt_logprobs"]:
            request["logprob_start_len"] = 0
        if payload["topk_prompt_logprobs"]:
            request["top_logprobs_num"] = payload["topk_prompt_logprobs"]
        if lora_name is not None:
            request["lora_path"] = lora_name
            if lora_path is not None:
                # request-carried backfill source: the engine refills an evicted version itself
                request["lora_backfill_paths"] = {lora_name: lora_path}
        return request


def _raise_slot_errors(worker_results: list) -> None:
    """A non-None result is an actor's {"error": ...} verdict; surface the failure to the caller."""
    errors = [result["error"] for result in worker_results if result is not None]
    if errors:
        raise RuntimeError(errors[0])


def _with_sample_seed(request: dict, index: int) -> dict:
    """Give each sample a distinct seed when the caller pins the request seed."""
    request = dict(request)
    params = dict(request["sampling_params"])
    if (seed := params.get("sampling_seed")) is not None:
        params["sampling_seed"] = seed + index
    request["sampling_params"] = params
    return request


def _prompt_logprobs(response: dict) -> list[float]:
    entries = response["meta_info"]["input_token_logprobs"]
    return [float("nan") if entry[0] is None else float(entry[0]) for entry in entries]


def _topk_prompt_logprobs(response: dict, k: int) -> dict:
    token_ids, logprobs = [], []
    for position in response["meta_info"]["input_top_logprobs"]:
        candidates = position or []
        candidate_token_ids = [entry[1] for entry in candidates][:k]
        candidate_logprobs = [float(entry[0]) for entry in candidates][:k]
        token_ids.append(candidate_token_ids + [0] * (k - len(candidate_token_ids)))
        logprobs.append(candidate_logprobs + [float("nan")] * (k - len(candidate_logprobs)))
    return {"token_ids": token_ids, "logprobs": logprobs}


def _to_sequence(response: dict) -> dict:
    output_token_logprobs = response["meta_info"]["output_token_logprobs"]
    finish = response["meta_info"]["finish_reason"]["type"]
    return {
        "sequence_id": f"seq-{uuid.uuid4().hex}",
        "tokens": [entry[1] for entry in output_token_logprobs],
        "logprobs": [entry[0] for entry in output_token_logprobs],
        "stop_reason": "length" if finish == "length" else "stop",
    }
