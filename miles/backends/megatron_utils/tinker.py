"""Megatron execution path for the Tinker-compatible API."""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config

from miles.backends.megatron_utils.multi_lora_optimizer import (
    _slot_children,
    named_adapter_slot_parameters,
    reset_grad_metadata_keep_grads,
    step_adapter_slots,
    zero_adapter_slot_grads,
)
from miles.backends.megatron_utils.tinker_loss import (
    compute_tinker_loss,
    tensor_data,
    tensor_from_payload,
    validate_and_get_targets,
)
from miles.backends.training_utils.loss_hub.math_utils import (
    _gather_true_on_policy_full_logits,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.ray.multi_lora.controller import get_multi_lora_controller
from miles.ray.tinker.protocol import TinkerError
from miles.utils.distributed_utils import get_gloo_group


def execute_tinker_operation(
    args: Namespace,
    model,
    optimizer,
    loaded_adapters: dict[str, object],
    pending_push: set[str],
    operation: dict[str, Any],
) -> dict[str, Any] | None:
    """Execute one serialized Tinker model operation on every Megatron rank."""
    kind = operation["kind"]
    payload = operation["payload"]
    model_id = operation["model_id"]

    if kind in {"forward", "forward_backward"}:
        return _execute_forward(
            args,
            model,
            payload,
            backward=kind == "forward_backward",
        )
    if kind == "optim_step":
        result = _execute_optim_step(
            model,
            optimizer,
            slot=payload["slot"],
            adam_params=payload["adam_params"],
        )
        pending_push.add(model_id)
        if _is_main_rank():
            ray.get(get_multi_lora_controller().advance_adapter_step.remote(model_id))
            return {"_operation_kind": kind, **result}
        return None
    if kind in {"save_weights", "save_weights_for_sampler"}:
        adapter = _require_loaded_adapter(loaded_adapters, model_id)
        _save_checkpoint(
            args,
            model,
            optimizer,
            adapter=adapter,
            model_id=model_id,
            slot=payload["slot"],
            checkpoint_step=payload["checkpoint_step"],
            local_path=Path(payload["local_path"]),
            include_optimizer=payload["include_optimizer"],
        )
        if _is_main_rank():
            return {
                "_operation_kind": kind,
                "tinker_path": payload["tinker_path"],
                "sampling_session_seq_id": payload.get("sampling_session_seq_id"),
            }
        return None
    if kind == "load_weights":
        adapter = _require_loaded_adapter(loaded_adapters, model_id)
        _load_checkpoint(
            model,
            optimizer,
            adapter=adapter,
            slot=payload["slot"],
            local_path=Path(payload["local_path"]),
            load_optimizer=payload["optimizer"],
        )
        pending_push.add(model_id)
        if _is_main_rank():
            return {
                "_operation_kind": kind,
                "tinker_path": payload["tinker_path"],
            }
        return None
    raise TinkerError(f"unsupported Tinker operation {kind!r}", category="user")


def _execute_forward(
    args: Namespace,
    model,
    payload: dict[str, Any],
    *,
    backward: bool,
) -> dict[str, Any] | None:
    parallel_state = get_parallel_state()
    if parallel_state.pp.size != 1 or parallel_state.cp.size != 1:
        raise TinkerError(
            "the Tinker executor currently requires pipeline and context parallel size 1",
            category="user",
        )

    pad_multiple = max(1, parallel_state.tp.size * args.data_pad_size_multiplier)
    configured_max_tokens = getattr(args, "max_tokens_per_gpu", None)
    max_tokens = args.seq_length if configured_max_tokens is None else configured_max_tokens
    if max_tokens <= 0:
        raise TinkerError("max_tokens_per_gpu must be positive", category="server")
    validated = _validate_forward_payload(
        payload,
        vocab_size=getattr(args, "vocab_size", None),
        max_sequence_length=args.seq_length,
    )
    for datum in validated:
        padded_tokens = _round_up(len(datum["tokens"]), pad_multiple)
        if padded_tokens > max_tokens:
            raise TinkerError(
                f"datum {datum['index']} requires {padded_tokens} padded tokens, exceeding max_tokens_per_gpu={max_tokens}",
                category="user",
            )

    local_datums = validated[parallel_state.effective_dp.rank :: parallel_state.effective_dp.size]
    local_batches = _build_microbatches(
        local_datums,
        slot=payload["slot"],
        n_adapters=args.multi_lora_n_adapters,
        pad_multiple=pad_multiple,
        max_tokens=max_tokens,
    )
    num_microbatches = len(local_batches)
    config = get_model_config(model[0])
    config.timers = None
    local_records: list[dict[str, Any]] = []

    def forward_step(data_iterator, model_chunk, return_schedule_plan: bool = False):
        if return_schedule_plan:
            raise TinkerError("combined 1f1b is not supported by the Tinker executor", category="user")

        from megatron.bridge.peft.multi_lora_layers import set_tokens_per_adapter_slot

        batch = next(data_iterator)
        batch_datums = batch["datums"]
        set_tokens_per_adapter_slot(model_chunk, batch["adapter_token_counts"])
        logits = model_chunk(
            input_ids=batch["tokens"],
            position_ids=None,
            attention_mask=None,
            labels=None,
            packed_seq_params=batch["packed_seq_params"],
            loss_mask=batch["loss_mask"],
        )

        if backward:

            def loss_func(output_tensor):
                local_loss, records = _loss_and_records(
                    output_tensor,
                    batch_datums,
                    payload["loss_fn"],
                    payload["loss_fn_config"],
                    vocab_size=args.vocab_size,
                )
                local_records.extend(records)
                # Megatron averages loss gradients over microbatches and DDP
                # averages over replicas; undo the latter for Tinker's sum loss.
                scaled_loss = local_loss * num_microbatches * parallel_state.intra_dp.size
                return (
                    scaled_loss,
                    torch.tensor(1, dtype=torch.int64, device=scaled_loss.device),
                    {
                        "keys": ["loss"],
                        "values": torch.stack(
                            [
                                torch.ones((), device=scaled_loss.device),
                                local_loss.detach().to(dtype=torch.float32),
                            ]
                        ),
                    },
                )

            return logits, loss_func

        def collect(output_tensor, non_loss_data: bool = False):
            assert non_loss_data
            _local_loss, records = _loss_and_records(
                output_tensor,
                batch_datums,
                payload["loss_fn"],
                payload["loss_fn_config"],
                vocab_size=args.vocab_size,
            )
            local_records.extend(records)
            return {}

        return logits, collect

    if backward:
        for model_chunk in model:
            model_chunk.train()
        reset_grad_metadata_keep_grads(model)
        get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=iter(local_batches),
            model=model,
            num_microbatches=num_microbatches,
            seq_length=args.seq_length,
            micro_batch_size=1,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
        )
    else:
        for model_chunk in model:
            model_chunk.eval()
        try:
            with torch.no_grad():
                get_forward_backward_func()(
                    forward_step_func=forward_step,
                    data_iterator=iter(local_batches),
                    model=model,
                    num_microbatches=num_microbatches,
                    seq_length=args.seq_length,
                    micro_batch_size=1,
                    decoder_seq_length=args.decoder_seq_length,
                    forward_only=True,
                    collect_non_loss_data=True,
                )
        finally:
            for model_chunk in model:
                model_chunk.train()

    response = _gather_forward_response(local_records, len(validated))
    if response is not None:
        response["_operation_kind"] = "forward_backward" if backward else "forward"
    return response


def _validate_forward_payload(
    payload: dict[str, Any],
    *,
    vocab_size: int | None,
    max_sequence_length: int,
) -> list[dict[str, Any]]:
    """Run identical validation on every rank before entering collectives."""
    loss_fn = payload["loss_fn"]
    loss_fn_config = payload["loss_fn_config"]
    validated = []
    for index, datum in enumerate(payload["data"]):
        tokens = datum["tokens"]
        if not tokens:
            raise TinkerError("model_input must contain at least one token", category="user")
        if len(tokens) > max_sequence_length:
            raise TinkerError(
                f"datum {index} has {len(tokens)} tokens, exceeding the configured sequence length {max_sequence_length}",
                category="user",
            )
        if vocab_size is not None and any(token < 0 or token >= vocab_size for token in tokens):
            raise TinkerError(f"datum {index} has a model-input token outside the vocabulary", category="user")
        inputs = {name: tensor_from_payload(value, device="cpu") for name, value in datum["loss_fn_inputs"].items()}
        targets = validate_and_get_targets(
            inputs,
            model_input_length=len(tokens),
            loss_fn=loss_fn,
        )
        if vocab_size is not None and (bool((targets < 0).any()) or bool((targets >= vocab_size).any())):
            raise TinkerError(f"datum {index} has a target token outside the vocabulary", category="user")

        # Validate shapes/config on every rank before any rank can block in a
        # TP/DP collective. The zero logprobs preserve the requested shape.
        compute_tinker_loss(
            torch.zeros_like(targets, dtype=torch.float32),
            inputs,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
        validated.append(
            {
                "index": index,
                "tokens": tokens,
                "inputs": inputs,
            }
        )
    return validated


def _build_microbatches(
    local_datums: list[dict[str, Any]],
    *,
    slot: int,
    n_adapters: int,
    pad_multiple: int,
    max_tokens: int,
) -> list[dict[str, Any]]:
    """Greedily pack local datums without exceeding the per-GPU token cap."""
    if max_tokens <= 0:
        raise TinkerError("max_tokens_per_gpu must be positive", category="server")

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for datum in local_datums:
        datum_tokens = len(datum["tokens"])
        padded_datum_tokens = _round_up(datum_tokens, pad_multiple)
        if padded_datum_tokens > max_tokens:
            raise TinkerError(
                f"datum {datum['index']} requires {padded_datum_tokens} padded tokens, exceeding max_tokens_per_gpu={max_tokens}",
                category="user",
            )
        candidate_tokens = current_tokens + datum_tokens
        if current and _round_up(candidate_tokens, pad_multiple) > max_tokens:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(datum)
        current_tokens += datum_tokens
    if current:
        groups.append(current)
    if not groups:
        groups.append([])

    return [
        _pack_batch(
            group,
            slot=slot,
            n_adapters=n_adapters,
            pad_multiple=pad_multiple,
        )
        for group in groups
    ]


def _pack_batch(
    local_datums: list[dict[str, Any]],
    *,
    slot: int,
    n_adapters: int,
    pad_multiple: int,
) -> dict[str, Any]:
    from megatron.core.packed_seq_params import PackedSeqParams

    device = torch.cuda.current_device()
    token_tensors = [torch.tensor(datum["tokens"], dtype=torch.int64, device=device) for datum in local_datums]
    if not token_tensors:
        token_tensors = [torch.zeros(1, dtype=torch.int64, device=device)]

    cu_seqlens = [0]
    for tokens in token_tensors:
        cu_seqlens.append(cu_seqlens[-1] + tokens.numel())
    packed_tokens = torch.cat(token_tensors)
    padding = (-packed_tokens.numel()) % pad_multiple
    if padding:
        packed_tokens = torch.nn.functional.pad(packed_tokens, (0, padding), value=0)
        cu_seqlens.append(cu_seqlens[-1] + padding)

    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
    max_seqlen = int((cu_seqlens_tensor[1:] - cu_seqlens_tensor[:-1]).max().item())
    counts = torch.zeros(n_adapters, dtype=torch.int32, device=device)
    counts[slot] = packed_tokens.numel()
    return {
        "datums": local_datums,
        "tokens": packed_tokens.unsqueeze(0),
        "loss_mask": torch.zeros_like(packed_tokens).unsqueeze(0),
        "adapter_token_counts": counts,
        "packed_seq_params": PackedSeqParams(
            cu_seqlens_q=cu_seqlens_tensor,
            cu_seqlens_kv=cu_seqlens_tensor,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
        ),
    }


def _loss_and_records(
    logits: torch.Tensor,
    local_datums: list[dict[str, Any]],
    loss_fn: str,
    loss_fn_config: dict[str, float],
    *,
    vocab_size: int,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    parallel_state = get_parallel_state()
    logits = logits.squeeze(0)
    if not local_datums:
        return logits.sum() * 0.0, []

    offset = 0
    losses = []
    records = []
    for datum in local_datums:
        inputs = {name: value.to(device=logits.device, non_blocking=True) for name, value in datum["inputs"].items()}
        targets = inputs["target_tokens"]
        length = len(datum["tokens"])
        datum_logits = logits[offset : offset + length]
        offset += length

        # Tinker logprobs must match serving over the real vocabulary, not
        # Megatron's padded vocabulary. The replicated-loss gather has a custom
        # backward that avoids an extra TP-size gradient multiplier.
        full_logits = _gather_true_on_policy_full_logits(
            datum_logits,
            parallel_state.tp.group,
            vocab_size=vocab_size,
        )
        gather_index = targets.unsqueeze(-1) if targets.ndim == 1 else targets
        target_logprobs = torch.log_softmax(full_logits.to(dtype=torch.float32), dim=-1).gather(
            dim=-1,
            index=gather_index,
        )
        if targets.ndim == 1:
            target_logprobs = target_logprobs.squeeze(-1)

        loss = compute_tinker_loss(
            target_logprobs,
            inputs,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
        losses.append(loss)
        records.append(
            {
                "index": datum["index"],
                "loss": float(loss.detach().to(dtype=torch.float32).cpu()),
                "output": {"logprobs": tensor_data(target_logprobs)},
            }
        )
    return torch.stack(losses).sum(), records


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _gather_forward_response(
    local_records: list[dict[str, Any]],
    num_datums: int,
) -> dict[str, Any] | None:
    if dist.is_initialized():
        group = get_gloo_group()
        gathered: list[object] = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(gathered, local_records, group=group)
        if not _is_main_rank():
            return None
        records = [record for rank_records in gathered for record in rank_records]
    else:
        records = local_records

    # TP/EP ranks produce identical records. Indexing by original datum both
    # de-duplicates those copies and restores the SDK request order.
    by_index = {record["index"]: record for record in records}
    if set(by_index) != set(range(num_datums)):
        missing = sorted(set(range(num_datums)) - set(by_index))
        raise RuntimeError(f"Tinker forward result is missing datum indices {missing}")
    ordered = [by_index[index] for index in range(num_datums)]
    return {
        "loss_fn_output_type": "ArrayRecord",
        "loss_fn_outputs": [record["output"] for record in ordered],
        "metrics": {"loss:sum": float(sum(record["loss"] for record in ordered))},
    }


def _execute_optim_step(
    model,
    optimizer,
    *,
    slot: int,
    adam_params: dict[str, float],
) -> dict[str, Any]:
    for child in _slot_children(optimizer, slot):
        for group in child.param_groups:
            group["lr"] = adam_params["learning_rate"]
            group["betas"] = (adam_params["beta1"], adam_params["beta2"])
            group["eps"] = adam_params["eps"]
            group["weight_decay"] = adam_params["weight_decay"]

    norms = step_adapter_slots(
        optimizer,
        model,
        {slot: 1},
        clip_grad=adam_params["grad_clip_norm"],
        normalize_by_batch_size=False,
    )
    return {"metrics": {"grad_norm": norms.get(slot, 0.0)}}


def _save_checkpoint(
    args: Namespace,
    model,
    optimizer,
    *,
    adapter,
    model_id: str,
    slot: int,
    checkpoint_step: int,
    local_path: Path,
    include_optimizer: bool,
) -> None:
    from miles.backends.megatron_utils.multi_lora_utils import save_multi_lora_checkpoints

    expected = adapter.config.save / "checkpoints" / f"step_{checkpoint_step}"
    if expected != local_path:
        raise RuntimeError(f"Tinker checkpoint path mismatch: expected {expected}, got {local_path}")
    save_multi_lora_checkpoints(
        args,
        model,
        {model_id: checkpoint_step},
        {model_id: adapter},
    )
    if not include_optimizer:
        return

    named_params = named_adapter_slot_parameters(model, slot)
    key_by_param = {id(param): name for name, param in named_params}
    optimizer_state: dict[str, Any] = {}
    optimizer_groups: list[list[dict[str, Any]]] = []
    for child in _slot_children(optimizer, slot):
        groups = []
        for group in child.optimizer.param_groups:
            groups.append({key: _to_cpu_tree(value) for key, value in group.items() if key not in {"params", "miles_multi_lora_slot"}})
        optimizer_groups.append(groups)
        for model_param, main_param in _owned_model_main_parameters(child):
            try:
                name = key_by_param[id(model_param)]
            except KeyError:
                raise RuntimeError("optimizer owns a parameter outside the requested LoRA slot") from None
            if name in optimizer_state:
                raise RuntimeError(f"optimizer owns duplicate LoRA parameter {name!r}")
            optimizer_state[name] = {
                "main_param": main_param.detach().cpu(),
                "state": _to_cpu_tree(child.optimizer.state.get(main_param, {})),
            }

    parallel_state = get_parallel_state()
    state = {
        "format_version": 2,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "topology": _model_parallel_coordinate(parallel_state),
        "optimizer_state": optimizer_state,
        "optimizer_groups": optimizer_groups,
        "retained_grads": {name: None if getattr(param, "main_grad", None) is None else param.main_grad.detach().cpu() for name, param in named_params if name in optimizer_state},
    }
    sidecar = local_path / f"tinker_optimizer_rank{dist.get_rank() if dist.is_initialized() else 0}.pt"
    temporary = sidecar.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, sidecar)
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())


def _load_checkpoint(
    model,
    optimizer,
    *,
    adapter,
    slot: int,
    local_path: Path,
    load_optimizer: bool,
) -> None:
    from megatron.bridge.peft.multi_lora_layers import init_adapter_slot, load_adapter

    from miles.backends.megatron_utils.multi_lora_utils import (
        _apply_tinker_module_groups,
        megatron_shard_name,
        zero_optimizer_state_for_adapter,
    )

    parallel_state = get_parallel_state()
    native_path = local_path / megatron_shard_name(
        parallel_state.tp.rank,
        parallel_state.pp.rank,
        parallel_state.ep.rank,
        parallel_state.ep.size,
    )
    if not native_path.exists():
        raise TinkerError(f"checkpoint shard {native_path} does not exist", category="user")
    state_dict = torch.load(native_path, map_location="cpu", weights_only=True)
    loaded = load_adapter(model, slot, state_dict)
    if loaded <= 0:
        raise TinkerError(
            f"checkpoint shard {native_path} did not contain any compatible LoRA tensors",
            category="user",
        )
    init_adapter_slot(model, slot, rank=adapter.config.rank, alpha=adapter.config.alpha)
    _apply_tinker_module_groups(adapter.config, model, slot)
    optimizer.reload_model_params()

    if load_optimizer:
        saved_by_rank = _load_optimizer_sidecars(local_path, parallel_state)
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        saved = saved_by_rank[local_rank]
        children = _slot_children(optimizer, slot)
        if len(saved["optimizer_groups"]) != len(children):
            raise TinkerError(
                "checkpoint optimizer topology does not match this trainer",
                category="user",
            )
        optimizer_state = _merge_optimizer_state(saved_by_rank.values())
        retained_grads = _merge_retained_grads(saved_by_rank.values())
        key_by_param = {id(param): name for name, param in named_adapter_slot_parameters(model, slot)}
        owned_names = set()
        for child, saved_groups in zip(children, saved["optimizer_groups"], strict=True):
            if len(saved_groups) != len(child.optimizer.param_groups):
                raise TinkerError(
                    "checkpoint optimizer parameter-group topology does not match this trainer",
                    category="user",
                )
            for group, saved_group in zip(child.optimizer.param_groups, saved_groups, strict=True):
                group.update(_to_device_tree(saved_group, _optimizer_group_device(group)))
            for model_param, main_param in _owned_model_main_parameters(child):
                try:
                    name = key_by_param[id(model_param)]
                    parameter_state = optimizer_state[name]
                except KeyError:
                    raise TinkerError(
                        "checkpoint optimizer parameters do not match this trainer",
                        category="user",
                    ) from None
                owned_names.add(name)
                main_param.data.copy_(parameter_state["main_param"].to(device=main_param.device, dtype=main_param.dtype))
                child.optimizer.state[main_param] = _to_device_tree(
                    parameter_state["state"],
                    main_param.device,
                )

        zero_adapter_slot_grads(model, slot)
        for name, param in named_adapter_slot_parameters(model, slot):
            if name not in owned_names:
                continue
            if name not in retained_grads:
                raise TinkerError(
                    "checkpoint retained-gradient topology does not match this trainer",
                    category="user",
                )
            saved_grad = retained_grads[name]
            if saved_grad is not None:
                if getattr(param, "main_grad", None) is None:
                    raise TinkerError(
                        "checkpoint contains retained gradients, but this trainer has no matching gradient buffer",
                        category="server",
                    )
                param.main_grad.copy_(saved_grad.to(device=param.main_grad.device, dtype=param.main_grad.dtype))
    else:
        zero_optimizer_state_for_adapter(optimizer, model, slot)
        zero_adapter_slot_grads(model, slot)
    if dist.is_initialized():
        dist.barrier(group=get_gloo_group())


def _to_cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_tree(item) for item in value)
    return value


def _to_device_tree(value, device: torch.device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _to_device_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device_tree(item, device) for item in value)
    return value


def _owned_model_main_parameters(child):
    """Yield the model parameter paired with the parameter optimized by a child."""
    for model_group, main_group in zip(
        getattr(child, "float16_groups", ()),
        getattr(child, "fp32_from_float16_groups", ()),
        strict=True,
    ):
        yield from zip(model_group, main_group, strict=True)
    for model_group in getattr(child, "fp32_from_fp32_groups", ()):
        for model_param in model_group:
            yield model_param, model_param


def _model_parallel_coordinate(parallel_state) -> dict[str, int]:
    return {
        "tp": parallel_state.tp.rank,
        "pp": parallel_state.pp.rank,
        "cp": parallel_state.cp.rank,
        "ep": parallel_state.ep.rank,
        "etp": parallel_state.etp.rank,
    }


def _data_parallel_global_ranks(parallel_state) -> list[int]:
    if not dist.is_initialized():
        return [0]
    get_ranks = getattr(dist, "get_process_group_ranks", None)
    if get_ranks is not None:
        return list(get_ranks(parallel_state.intra_dp.group))

    coordinate = _model_parallel_coordinate(parallel_state)
    coordinates: list[dict[str, int] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(coordinates, coordinate, group=get_gloo_group())
    return [rank for rank, candidate in enumerate(coordinates) if candidate == coordinate]


def _load_optimizer_sidecars(local_path: Path, parallel_state) -> dict[int, dict[str, Any]]:
    expected_world_size = dist.get_world_size() if dist.is_initialized() else 1
    expected_coordinate = _model_parallel_coordinate(parallel_state)
    saved_by_rank = {}
    for rank in _data_parallel_global_ranks(parallel_state):
        sidecar = local_path / f"tinker_optimizer_rank{rank}.pt"
        if not sidecar.exists():
            raise TinkerError(
                f"checkpoint optimizer state {sidecar} does not exist for this trainer topology",
                category="user",
            )
        saved = torch.load(sidecar, map_location="cpu", weights_only=False)
        if saved.get("format_version") != 2 or saved.get("world_size") != expected_world_size or saved.get("topology") != expected_coordinate:
            raise TinkerError(
                "checkpoint optimizer topology does not match this trainer",
                category="user",
            )
        saved_by_rank[rank] = saved
    return saved_by_rank


def _merge_optimizer_state(saved_sidecars) -> dict[str, Any]:
    merged = {}
    for saved in saved_sidecars:
        for name, state in saved["optimizer_state"].items():
            if name in merged:
                raise TinkerError(
                    f"checkpoint contains duplicate optimizer state for {name!r}",
                    category="server",
                )
            merged[name] = state
    return merged


def _merge_retained_grads(saved_sidecars) -> dict[str, torch.Tensor | None]:
    """Redistribute retained gradients using the optimizer's whole-param owner.

    LayerWiseDistributedOptimizer leaves a slot parameter's reduced
    ``main_grad`` on the DP rank that owns its optimizer state. Different slots
    can assign the same logical parameter to different ranks, so loading the
    local rank's gradient sidecar positionally would silently lose gradients.
    """
    merged = {}
    for saved in saved_sidecars:
        for name in saved["optimizer_state"]:
            if name in merged or name not in saved["retained_grads"]:
                raise TinkerError(
                    f"checkpoint contains inconsistent retained-gradient state for {name!r}",
                    category="server",
                )
            merged[name] = saved["retained_grads"][name]
    return merged


def _optimizer_group_device(group: dict[str, Any]) -> torch.device:
    params = group["params"]
    return params[0].device if params else torch.device("cpu")


def _require_loaded_adapter(loaded_adapters: dict[str, object], model_id: str):
    try:
        return loaded_adapters[model_id]
    except KeyError:
        raise TinkerError(f"model {model_id!r} is not loaded on the trainer", category="server") from None


def _is_main_rank() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0
