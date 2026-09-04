"""The miles-bound half of the gateway.

Implements ExecutorBackend over TrainerController (training commands) and the
sglang router (sampling). This is where the gateway's internal language turns
into miles vocabulary: rows become RolloutBatch keys, neutral sampling params
become /generate payloads. Together with serve_tinker.py this is the only
place allowed to import miles.ray / miles.backends.
"""

import asyncio
import uuid
from argparse import Namespace

from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_VALUE_SPEC
from miles.tinker.core.service import ExecutorBackend
from miles.tinker.core.types import UserInputError
from miles.utils import object_store
from miles.utils.http_utils import post

# internal row key -> trainer batch key
ROW_TO_BATCH_KEYS = {"weights": "loss_weights", "advantages": "advantages", "sampling_logprobs": "rollout_log_probs"}


class MilesBackend(ExecutorBackend):
    def __init__(self, args: Namespace, trainer, router_url: str) -> None:
        self.args = args
        self.trainer = trainer
        self.router_url = router_url

    async def load_slot(
        self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True
    ) -> None:
        await self.trainer.load_slot(slot, rank, alpha, ckpt_path=ckpt_path, load_optimizer=load_optimizer)

    async def unload_slot(self, slot: int) -> None:
        await self.trainer.unload_slot(slot)

    async def forward_backward(self, unit_id: int, slot_rows: list, loss_fn: str, loss_fn_config: dict) -> list[dict]:
        train_data = _build_train_data(slot_rows)
        train_data["loss_fn"] = loss_fn
        train_data["loss_fn_config"] = loss_fn_config
        worker_results = await self._run_unit("forward_backward", unit_id, train_data)
        by_index: dict[int, dict] = {}
        for result in worker_results:
            for item in result["per_datum"]:
                index = int(item["sample_index"])
                if index not in by_index:
                    by_index[index] = {"loss": float(item["loss"]), "logprobs": item["logprobs"].tolist()}
        return [by_index[index] for index in range(len(slot_rows))]

    async def forward_only(self, unit_id: int, slot_rows: list, loss_fn: str, loss_fn_config: dict) -> list[dict]:
        worker_results = await self._run_unit("forward_only_logprobs", unit_id, _build_train_data(slot_rows))
        by_index: dict[int, dict] = {}
        for box in worker_results:
            if box is None:
                continue
            value = _box_get(box)
            for index, logprobs in zip(value["sample_indices"], value["logprobs"], strict=True):
                if index not in by_index:
                    by_index[index] = {"loss": 0.0, "logprobs": logprobs.tolist()}
        return [by_index[index] for index in range(len(slot_rows))]

    async def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, float]:
        worker_results = await self.trainer.optim_step(adam_params_by_slot=adam_params_by_slot)
        return worker_results[0]

    async def save_slot(self, slot: int, path: str) -> None:
        await self.trainer.save_slot(slot=slot, path=path)

    async def push_slot(self, slot: int, lora_name: str, rank: int, alpha: float) -> None:
        await self.trainer.push_slot(slot=slot, lora_name=lora_name, rank=rank, alpha=alpha)

    async def _run_unit(self, method: str, unit_id: int, train_data: dict) -> list:
        store = object_store.get_instance()
        data_ref = store.put(value=train_data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        try:
            return await getattr(self.trainer, method)(unit_id=unit_id, data_ref=data_ref)
        finally:
            store.remove(data_ref)

    # -------- sampling --------

    async def sample(self, payload: dict, lora_name: str | None) -> dict:
        request = self._generate_request(payload, lora_name)
        responses = await asyncio.gather(
            *[post(f"{self.router_url}/generate", dict(request)) for _ in range(payload["num_samples"])]
        )
        result = {"sequences": [_to_sequence(response) for response in responses]}
        if payload["prompt_logprobs"]:
            result["prompt_logprobs"] = _prompt_logprobs(responses[0])
        if payload["topk_prompt_logprobs"]:
            result["topk_prompt_logprobs"] = _topk_prompt_logprobs(responses[0], payload["topk_prompt_logprobs"])
        return result

    def _generate_request(self, payload: dict, lora_name: str | None) -> dict:
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
            if isinstance(stop, list) and stop and isinstance(stop[0], int):
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
        return request


def _build_train_data(slot_rows: list) -> dict:
    rows = [row for _, row in slot_rows]
    train_data = {
        "tokens": [row["tokens"] for row in rows],
        "loss_masks": [[1] * row["target_len"] for row in rows],
        "response_lengths": [row["target_len"] for row in rows],
        "total_lengths": [len(row["tokens"]) for row in rows],
        "sample_indices": list(range(len(rows))),
        "adapter_slots": [slot for slot, _ in slot_rows],
        "dynamic_global_batch_size": len(rows),
    }
    for row_key, batch_key in ROW_TO_BATCH_KEYS.items():
        if row_key in rows[0]:
            train_data[batch_key] = [row[row_key] for row in rows]
    return train_data


def _box_get(box):
    import ray

    return ray.get(box.inner)


def _prompt_logprobs(response: dict) -> list[float]:
    entries = response["meta_info"]["input_token_logprobs"]
    return [float("nan") if entry[0] is None else float(entry[0]) for entry in entries]


def _topk_prompt_logprobs(response: dict, k: int) -> dict:
    token_ids, logprobs = [], []
    for position in response["meta_info"]["input_top_logprobs"]:
        candidates = position or []
        row_tokens = [entry[1] for entry in candidates][:k]
        row_logprobs = [float(entry[0]) for entry in candidates][:k]
        token_ids.append(row_tokens + [0] * (k - len(row_tokens)))
        logprobs.append(row_logprobs + [float("nan")] * (k - len(row_logprobs)))
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
