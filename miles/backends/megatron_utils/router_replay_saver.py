"""Saver protocol for routing-replay artifacts (main payload +
PR²'s predictive metrics + metric tensors).

Used by ``actor.py`` / ``model.py`` to persist per-step artifacts on a
background thread so the train step does not block. File naming +
loading live in ``router_replay_artifacts.py``.
"""
import logging
import os
import threading
import json
from typing import Any

import torch
from megatron.core import parallel_state as mpu
from torch import no_grad

from .router_replay_artifacts import get_router_replay_artifact_paths

logger = logging.getLogger(__name__)


def _empty_logits_cache() -> dict[str, list | dict]:
    return {
        "compute_log_prob": [],
        "training": [],
        "router_weights": {},
        "global_token_ids": [],
        "predictive_bias": [],
    }


def _empty_predictive_metric_tensor_cache() -> dict[str, list]:
    return {
        "old_inputs": [],
        "current_inputs": [],
        "old_logits": [],
        "current_logits": [],
        "predicted_delta_logits": [],
    }


def _truncate_tensor_on_token_dim(tensor: torch.Tensor, max_tokens: int | None) -> torch.Tensor:
    if max_tokens is None or tensor.size(0) <= max_tokens:
        return tensor
    return tensor[:max_tokens]


def _truncate_router_replay_save_dict_tokens(save_dict: dict[str, Any], max_tokens: int | None) -> dict[str, Any]:
    if max_tokens is None:
        return save_dict

    for phase in ("compute_log_prob", "training", "predictive_bias"):
        phase_payload = save_dict.get(phase)
        if not isinstance(phase_payload, dict):
            continue
        for layer_idx, tensor in list(phase_payload.items()):
            phase_payload[layer_idx] = _truncate_tensor_on_token_dim(tensor, max_tokens)

    global_token_ids = save_dict.get("global_token_ids")
    if isinstance(global_token_ids, torch.Tensor):
        save_dict["global_token_ids"] = _truncate_tensor_on_token_dim(global_token_ids, max_tokens)

    return save_dict


def _truncate_predictive_metric_tensor_payload_tokens(payload: dict[str, Any], max_tokens: int | None) -> dict[str, Any]:
    if max_tokens is None:
        return payload

    layers = payload.get("layers")
    if not isinstance(layers, dict):
        return payload

    for layer_payload in layers.values():
        if not isinstance(layer_payload, dict):
            continue
        for tensor_name, tensor in list(layer_payload.items()):
            if isinstance(tensor, torch.Tensor):
                layer_payload[tensor_name] = _truncate_tensor_on_token_dim(tensor, max_tokens)

    return payload


class RouterReplayLogitsSaver:
    def __init__(self, save_dir: str, predictive_loss_type: str | None = None, max_tokens: int | None = None):
        self.save_dir = save_dir
        self.predictive_loss_type = predictive_loss_type
        self.max_tokens = max_tokens
        self.save_threads: list[threading.Thread] = []

        if (
            mpu.get_data_parallel_rank() == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
        ):
            os.makedirs(save_dir, exist_ok=True)
            logger.info("Router replay logits will be saved to: %s", save_dir)

    def _save_logits_sync(self, logits_data: dict[str, list | dict], step: str | int) -> None:
        try:
            tp_rank = mpu.get_tensor_model_parallel_rank()
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            artifact_paths = get_router_replay_artifact_paths(
                save_dir=self.save_dir,
                step=step,
                tp_rank=tp_rank,
                pp_rank=pp_rank,
            )
            os.makedirs(artifact_paths["step_dir"], exist_ok=True)

            save_dict: dict[str, Any] = {
                "step": step,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "dp_world_size": mpu.get_data_parallel_world_size(),
                "max_tokens": self.max_tokens,
                "compute_log_prob": {},
                "training": {},
                "router_weights": {},
                "global_token_ids": [],
                "predictive_bias": {},
            }

            for layer_idx, logits in logits_data.get("compute_log_prob", []):
                save_dict["compute_log_prob"].setdefault(layer_idx, []).append(logits)

            for layer_idx, logits in logits_data.get("training", []):
                save_dict["training"].setdefault(layer_idx, []).append(logits)

            for layer_idx, weight in logits_data.get("router_weights", {}).items():
                if layer_idx not in save_dict["router_weights"]:
                    save_dict["router_weights"][layer_idx] = weight

            predictive_bias = logits_data.get("predictive_bias", [])
            for layer_idx, bias in predictive_bias:
                save_dict["predictive_bias"].setdefault(layer_idx, []).append(bias)

            for token_ids in logits_data.get("global_token_ids", []):
                save_dict["global_token_ids"].append(token_ids)

            for phase in ("compute_log_prob", "training", "predictive_bias"):
                if phase not in save_dict:
                    continue
                for layer_idx in list(save_dict[phase].keys()):
                    tensors = save_dict[phase][layer_idx]
                    save_dict[phase][layer_idx] = torch.cat(tensors, dim=0)

            if save_dict["global_token_ids"]:
                save_dict["global_token_ids"] = torch.cat(save_dict["global_token_ids"], dim=0)

            save_dict = _truncate_router_replay_save_dict_tokens(save_dict, self.max_tokens)

            filepath = artifact_paths["main"]
            torch.save(save_dict, filepath)
            logger.info("Saved router replay logits to %s", filepath)
        except Exception:
            logger.exception("Failed to save router replay logits for step %s", step)

    def save_logits_async(self, logits_data: dict[str, list | dict], step: str | int) -> None:
        logits_data_copy = {
            "compute_log_prob": [(idx, tensor.clone()) for idx, tensor in logits_data.get("compute_log_prob", [])],
            "training": [(idx, tensor.clone()) for idx, tensor in logits_data.get("training", [])],
            "router_weights": {idx: tensor.clone() for idx, tensor in logits_data.get("router_weights", {}).items()},
            "global_token_ids": [tensor.clone() for tensor in logits_data.get("global_token_ids", [])],
            "predictive_bias": [(idx, tensor.clone()) for idx, tensor in logits_data.get("predictive_bias", [])],
        }

        save_thread = threading.Thread(
            target=self._save_logits_sync,
            args=(logits_data_copy, step),
            daemon=True,
        )
        save_thread.start()
        self.save_threads.append(save_thread)
        self.save_threads = [thread for thread in self.save_threads if thread.is_alive()]

    def _save_predictive_metrics_sync(
        self,
        metrics_data: dict[str, dict[str, float]],
        step: str | int,
    ) -> None:
        try:
            if not metrics_data:
                return

            tp_rank = mpu.get_tensor_model_parallel_rank()
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            artifact_paths = get_router_replay_artifact_paths(
                save_dir=self.save_dir,
                step=step,
                tp_rank=tp_rank,
                pp_rank=pp_rank,
            )
            os.makedirs(artifact_paths["step_dir"], exist_ok=True)

            aggregates = {}
            for metric_name, per_layer in metrics_data.items():
                if per_layer:
                    aggregates[metric_name] = sum(per_layer.values()) / len(per_layer)

            payload = {
                "step": step,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "predictive_loss_type": self.predictive_loss_type,
                "aggregates": aggregates,
                "per_layer": metrics_data,
            }

            filepath = artifact_paths["predictive_metrics"]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            logger.info("Saved predictive metrics sidecar to %s", filepath)
        except Exception:
            logger.exception("Failed to save predictive metrics sidecar for step %s", step)

    def save_predictive_metrics_async(
        self,
        metrics_data: dict[str, dict[str, float]],
        step: str | int,
    ) -> None:
        metrics_copy = {
            metric_name: {str(layer_idx): float(value) for layer_idx, value in per_layer.items()}
            for metric_name, per_layer in metrics_data.items()
            if per_layer
        }
        if not metrics_copy:
            return

        save_thread = threading.Thread(
            target=self._save_predictive_metrics_sync,
            args=(metrics_copy, step),
            daemon=True,
        )
        save_thread.start()
        self.save_threads.append(save_thread)
        self.save_threads = [thread for thread in self.save_threads if thread.is_alive()]

    def _save_predictive_metric_tensors_sync(
        self,
        tensor_data: dict[str, list[tuple[int, torch.Tensor]]],
        step: str | int,
        topk: int | None = None,
    ) -> None:
        try:
            if not any(tensor_data.values()):
                return

            tp_rank = mpu.get_tensor_model_parallel_rank()
            pp_rank = mpu.get_pipeline_model_parallel_rank()
            artifact_paths = get_router_replay_artifact_paths(
                save_dir=self.save_dir,
                step=step,
                tp_rank=tp_rank,
                pp_rank=pp_rank,
            )
            os.makedirs(artifact_paths["step_dir"], exist_ok=True)

            payload: dict[str, Any] = {
                "step": step,
                "tp_rank": tp_rank,
                "pp_rank": pp_rank,
                "topk": topk,
                "max_tokens": self.max_tokens,
                "predictive_loss_type": self.predictive_loss_type,
                "layers": {},
            }
            for tensor_name, entries in tensor_data.items():
                for layer_idx, tensor in entries:
                    layer_payload = payload["layers"].setdefault(str(layer_idx), {})
                    layer_payload.setdefault(tensor_name, []).append(tensor)

            for layer_payload in payload["layers"].values():
                for tensor_name, tensors in list(layer_payload.items()):
                    layer_payload[tensor_name] = torch.cat(tensors, dim=0)

            payload = _truncate_predictive_metric_tensor_payload_tokens(payload, self.max_tokens)

            filepath = artifact_paths["predictive_metric_tensors"]
            torch.save(payload, filepath)
            logger.info("Saved predictive metric tensors to %s", filepath)
        except Exception:
            logger.exception("Failed to save predictive metric tensors for step %s", step)

    def save_predictive_metric_tensors_async(
        self,
        tensor_data: dict[str, list[tuple[int, torch.Tensor]]],
        step: str | int,
        topk: int | None = None,
    ) -> None:
        tensor_data_copy = {
            tensor_name: [(layer_idx, tensor.clone()) for layer_idx, tensor in entries]
            for tensor_name, entries in tensor_data.items()
            if entries
        }
        if not tensor_data_copy:
            return

        save_thread = threading.Thread(
            target=self._save_predictive_metric_tensors_sync,
            args=(tensor_data_copy, step, topk),
            daemon=True,
        )
        save_thread.start()
        self.save_threads.append(save_thread)
        self.save_threads = [thread for thread in self.save_threads if thread.is_alive()]

    def wait_all_saves(self) -> None:
        for thread in self.save_threads:
            thread.join()
        self.save_threads = []
        logger.info("All router replay logits saves completed")

    @staticmethod
    @no_grad()
    def gather_logits_from_tp_group(logits_data: dict[str, list | dict]) -> dict[str, list | dict]:
        tp_world_size = mpu.get_tensor_model_parallel_world_size()
        if tp_world_size == 1:
            return logits_data
        return logits_data if mpu.get_tensor_model_parallel_rank() == 0 else _empty_logits_cache()

    @staticmethod
    @no_grad()
    def gather_logits_from_dp_group(
        logits_data: dict[str, list | dict],
        max_tokens: int | None = None,
    ) -> dict[str, list | dict]:
        dp_world_size = mpu.get_data_parallel_world_size()
        if dp_world_size == 1:
            return logits_data

        tokens_per_rank = None
        if max_tokens is not None:
            tokens_per_rank = max((max_tokens + dp_world_size - 1) // dp_world_size, 1)

        dp_rank = mpu.get_data_parallel_rank()
        if hasattr(mpu, "get_data_parallel_group_gloo"):
            dp_group = mpu.get_data_parallel_group_gloo(with_context_parallel=False)
        else:
            dp_group = mpu.get_data_parallel_group()
        gathered_data = _empty_logits_cache()

        for phase in ("compute_log_prob", "training", "router_weights", "global_token_ids", "predictive_bias"):
            phase_data = logits_data.get(phase, {} if phase == "router_weights" else [])
            layer_dict: dict[Any, list[torch.Tensor]] = {}
            token_ids_list: list[torch.Tensor] = []

            if phase == "global_token_ids":
                token_ids_list.extend(phase_data)
            elif phase == "router_weights":
                for layer_idx, tensor in phase_data.items():
                    layer_dict.setdefault(layer_idx, []).append(tensor.cpu().contiguous() if tensor.is_cuda else tensor.contiguous())
            else:
                for layer_idx, tensor in phase_data:
                    layer_dict.setdefault(layer_idx, []).append(tensor.cpu().contiguous() if tensor.is_cuda else tensor.contiguous())

            local_layer_data: list[tuple[Any, torch.Tensor]] = []
            if phase == "global_token_ids":
                if token_ids_list:
                    token_ids = torch.cat(token_ids_list, dim=0)
                    if tokens_per_rank is not None and token_ids.size(0) > tokens_per_rank:
                        token_ids = token_ids[:tokens_per_rank]
                    local_layer_data.append(("global_token_ids", token_ids))
            else:
                for layer_idx in sorted(layer_dict.keys()):
                    local_tensor = layer_dict[layer_idx][0] if phase == "router_weights" else torch.cat(layer_dict[layer_idx], dim=0)
                    if tokens_per_rank is not None and phase != "router_weights" and local_tensor.size(0) > tokens_per_rank:
                        local_tensor = local_tensor[:tokens_per_rank]
                    local_layer_data.append((layer_idx, local_tensor))

            if dp_rank == 0:
                gather_list = [None] * dp_world_size
                torch.distributed.gather_object(local_layer_data, gather_list, dst=0, group=dp_group)
                layer_combined: dict[Any, list[torch.Tensor]] = {}
                for rank_data in gather_list:
                    if rank_data is None:
                        continue
                    for layer_idx, tensor in rank_data:
                        layer_combined.setdefault(layer_idx, []).append(tensor)

                if phase == "global_token_ids":
                    if "global_token_ids" in layer_combined:
                        gathered_data[phase].append(torch.cat(layer_combined["global_token_ids"], dim=0))
                elif phase == "router_weights":
                    for layer_idx in sorted(layer_combined.keys()):
                        gathered_data[phase][layer_idx] = layer_combined[layer_idx][0]
                else:
                    for layer_idx in sorted(layer_combined.keys()):
                        gathered_data[phase].append((layer_idx, torch.cat(layer_combined[layer_idx], dim=0)))
            else:
                torch.distributed.gather_object(local_layer_data, None, dst=0, group=dp_group)

        return gathered_data if dp_rank == 0 else _empty_logits_cache()

    @staticmethod
    @no_grad()
    def gather_predictive_metric_tensors_from_dp_group(
        tensor_data: dict[str, list[tuple[int, torch.Tensor]]],
        max_tokens: int | None = None,
    ) -> dict[str, list[tuple[int, torch.Tensor]]]:
        dp_world_size = mpu.get_data_parallel_world_size()
        if dp_world_size == 1:
            return tensor_data

        tokens_per_rank = None
        if max_tokens is not None:
            tokens_per_rank = max((max_tokens + dp_world_size - 1) // dp_world_size, 1)

        dp_rank = mpu.get_data_parallel_rank()
        if hasattr(mpu, "get_data_parallel_group_gloo"):
            dp_group = mpu.get_data_parallel_group_gloo(with_context_parallel=False)
        else:
            dp_group = mpu.get_data_parallel_group()

        gathered_data = _empty_predictive_metric_tensor_cache()
        for tensor_name, entries in tensor_data.items():
            layer_dict: dict[Any, list[torch.Tensor]] = {}
            for layer_idx, tensor in entries:
                layer_dict.setdefault(layer_idx, []).append(tensor.cpu().contiguous() if tensor.is_cuda else tensor.contiguous())

            local_layer_data: list[tuple[Any, torch.Tensor]] = []
            for layer_idx in sorted(layer_dict.keys()):
                local_tensor = torch.cat(layer_dict[layer_idx], dim=0)
                if tokens_per_rank is not None and local_tensor.size(0) > tokens_per_rank:
                    local_tensor = local_tensor[:tokens_per_rank]
                local_layer_data.append((layer_idx, local_tensor))

            if dp_rank == 0:
                gather_list = [None] * dp_world_size
                torch.distributed.gather_object(local_layer_data, gather_list, dst=0, group=dp_group)
                merged: dict[Any, list[torch.Tensor]] = {}
                for rank_data in gather_list:
                    if rank_data is None:
                        continue
                    for layer_idx, tensor in rank_data:
                        merged.setdefault(layer_idx, []).append(tensor)
                for layer_idx in sorted(merged.keys()):
                    gathered_data[tensor_name].append((layer_idx, torch.cat(merged[layer_idx], dim=0)))
            else:
                torch.distributed.gather_object(local_layer_data, None, dst=0, group=dp_group)

        return gathered_data if dp_rank == 0 else _empty_predictive_metric_tensor_cache()
