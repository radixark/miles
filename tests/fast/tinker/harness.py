"""Shared fakes for the tinker gateway suite."""

import asyncio
import hashlib
import json
from pathlib import Path

import torch

from miles.backends.training_utils.checkpoint.adapter.adapter_exports import (
    write_external_adapter,
    write_sampler_adapter,
)
from miles.backends.training_utils.checkpoint.adapter.commit import write_adapter_dir
from miles.backends.training_utils.checkpoint.adapter.parametrization import (
    AdapterParametrization,
    write_parametrization,
)
from miles.backends.training_utils.checkpoint.io import write_checkpoint_dir
from miles.tinker.core.future import DONE, PENDING, RequestFuture
from miles.tinker.core.service import TinkerService
from miles.tinker.core.types import Command, CommandOp, GatewayConfig
from miles.tinker.runtime import (
    classify_lora_groups,
    commit_adapter,
    delete_adapter,
    delete_retired_adapters,
    resolve_adapter,
    resolve_checkpoint_files,
)
from miles.utils.lora.hf_lora_targets import get_hf_lora_targets

_TARGETS = get_hf_lora_targets({"model_type": "qwen3", "num_hidden_layers": 1})
ADAM = {
    "learning_rate": 1e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-12,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
}


def _weights(rank: int) -> dict[str, torch.Tensor]:
    return {
        "model.layers.0.self_attn.q_proj.lora_A": torch.zeros(rank, 4),
        "model.layers.0.self_attn.q_proj.lora_B": torch.zeros(4, rank),
    }


def _parametrization(fields: dict, target_modules=("model.layers.0.self_attn.q_proj",)) -> AdapterParametrization:
    # tiny modules stand in for the model's targets
    return AdapterParametrization(**fields, target_modules=target_modules)


def write_legacy_checkpoint(path: str, record, config: GatewayConfig) -> None:
    """A checkpoint dir as main wrote it: META.json and no commit link."""
    metadata = {
        "tenant_digest": hashlib.sha256(record.tenant.encode()).hexdigest(),
        "base_model": record.base_model,
        "lora_rank": record.lora_rank,
        "lora_alpha": record.lora_alpha,
        "lora_type": config.lora_type,
        "experts_shared_outer_loras": config.experts_shared_outer_loras,
        "train_attn": config.trains_attn,
        "train_mlp": config.trains_mlp,
        "train_unembed": config.trains_unembed,
    }
    write_checkpoint_dir(path, lambda directory: (directory / "META.json").write_text(json.dumps(metadata)))


class FakeBackend:
    """Record calls with deterministic outputs and configurable failures."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_next: Exception | dict | None = None
        self.dead = False
        self.optim_outcomes: dict[int, dict] = {}
        self.fail_on: dict[str, Exception | dict] = {}

    def _record(self, name: str, **kwargs) -> dict | None:
        self.calls.append((name, kwargs))
        failure, self.fail_next = self.fail_next, None
        if failure is None:
            failure = self.fail_on.pop(name, None)
        if isinstance(failure, Exception):
            self.dead = True
            raise failure
        return failure

    def named(self, name: str) -> list[dict]:
        return [kwargs for called, kwargs in self.calls if called == name]

    async def trainer_dead(self):
        return self.dead

    async def load_slot(self, slot, rank, alpha, source=None):
        return self._record("load_slot", slot=slot, rank=rank, alpha=alpha, source=source)

    async def unload_slot(self, slot):
        return self._record("unload_slot", slot=slot)

    async def forward_backward(self, batch_id, slot_datums, loss_fn, loss_fn_config):
        failure = self._record(
            "forward_backward",
            batch_id=batch_id,
            slot_datums=slot_datums,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
        if failure is not None:
            return failure
        return [{"loss": 1.0, "logprobs": [0.0] * datum["target_len"]} for _, datum in slot_datums]

    async def forward_only(self, batch_id, slot_datums, loss_fn, loss_fn_config):
        failure = self._record(
            "forward_only", batch_id=batch_id, slot_datums=slot_datums, loss_fn=loss_fn, loss_fn_config=loss_fn_config
        )
        if failure is not None:
            return failure
        return [{"loss": 0.0, "logprobs": [0.0] * datum["target_len"]} for _, datum in slot_datums]

    async def optim_step(self, adam_params_by_slot):
        failure = self._record("optim_step", adam_params_by_slot=adam_params_by_slot)
        if failure is not None:
            return {slot: failure for slot in adam_params_by_slot}
        return {slot: self.optim_outcomes.get(slot, {"grad_norm": 0.5 + slot}) for slot in adam_params_by_slot}

    async def save_slot(self, slot, root, name, tenant_digest, parametrization_fields):
        failure = self._record(
            "save_slot",
            slot=slot,
            root=root,
            adapter_name=name,
            tenant_digest=tenant_digest,
            parametrization_fields=parametrization_fields,
        )
        assert failure is None, "a trainer's save_slot raises; it never returns a failure"
        with write_adapter_dir(Path(root), name, tenant_digest=tenant_digest) as written:
            # the default gateway layout trains attn and mlp
            targets = ("model.layers.0.self_attn.q_proj", "model.layers.0.mlp.gate_proj")
            write_parametrization(written.path, _parametrization(parametrization_fields, targets))
        return written

    commit_adapter = staticmethod(commit_adapter)
    delete_retired_adapters = staticmethod(delete_retired_adapters)
    resolve_adapter = staticmethod(resolve_adapter)
    resolve_checkpoint_files = staticmethod(resolve_checkpoint_files)
    delete_adapter = staticmethod(delete_adapter)

    async def export_external(
        self, slot, root, name, tenant_digest, parametrization_fields, age_encryption_recipients
    ):
        failure = self._record(
            "export_external",
            slot=slot,
            root=root,
            adapter_name=name,
            tenant_digest=tenant_digest,
            parametrization_fields=parametrization_fields,
            age_encryption_recipients=age_encryption_recipients,
        )
        assert failure is None, "a trainer's export_external_slot raises; it never returns a failure"
        parametrization = _parametrization(parametrization_fields)
        with write_adapter_dir(Path(root), name, tenant_digest=tenant_digest) as written:
            write_external_adapter(
                written.path,
                _weights(parametrization.lora_rank),
                parametrization,
                _TARGETS,
                age_encryption_recipients=age_encryption_recipients,
                lora_dropout=0.0,
            )
        return written

    def lora_groups(self, target_modules):
        return classify_lora_groups(target_modules, _TARGETS)

    async def export_slot(self, slot, root, name, tenant_digest, parametrization_fields):
        failure = self._record(
            "export_slot",
            slot=slot,
            root=root,
            adapter_name=name,
            tenant_digest=tenant_digest,
            parametrization_fields=parametrization_fields,
        )
        assert failure is None, "a trainer's export_slot raises; it never returns a failure"
        parametrization = _parametrization(parametrization_fields)
        with write_adapter_dir(Path(root), name, tenant_digest=tenant_digest) as written:
            write_sampler_adapter(
                written.path, _weights(parametrization.lora_rank), parametrization, _TARGETS, lora_dropout=0.0
            )
        return written

    async def sample(self, payload, lora_name, lora_path=None):
        failure = self._record("sample", payload=payload, lora_name=lora_name, lora_path=lora_path)
        if failure is not None:
            return failure
        return {
            "sequences": [
                {"tokens": [1, 2], "logprobs": [0.0, 0.0], "stop_reason": "stop"}
                for _ in range(payload["num_samples"])
            ]
        }


def make_config(checkpoint_root, **overrides) -> GatewayConfig:
    defaults = dict(
        base_model="base",
        n_slots=2,
        checkpoint_root=str(checkpoint_root),
        vocab_size=128000,
        exports_tinker_names=True,
    )
    return GatewayConfig(**{**defaults, **overrides})


def make_service(checkpoint_root, **config_overrides) -> TinkerService:
    return TinkerService(FakeBackend(), make_config(checkpoint_root, **config_overrides))


def datum(tokens: int = 3) -> dict:
    return {
        "tokens": list(range(tokens + 1)),
        "target_tokens": list(range(1, tokens + 1)),
        "target_len": tokens,
        "weights": [1.0] * tokens,
    }


def rl_datum(tokens: int = 3) -> dict:
    """RL losses read logprobs+advantages and reject the cross-entropy weights."""
    return {
        "tokens": list(range(tokens + 1)),
        "target_tokens": list(range(1, tokens + 1)),
        "target_len": tokens,
        "sampling_logprobs": [0.0] * tokens,
        "advantages": [1.0] * tokens,
    }


def fb_payload(model_id: str, seq_id: int, datums: list[dict], loss_fn: str = "cross_entropy") -> dict:
    return {"model_id": model_id, "seq_id": seq_id, "datums": datums, "loss_fn": loss_fn, "loss_fn_config": {}}


def command(model_id: str, seq_id: int, op: str, payload: dict, arrival: int) -> Command:
    return Command(
        model_id=model_id,
        seq_id=seq_id,
        op=CommandOp(op),
        payload=payload,
        request_id=f"req-{seq_id}",
        arrival=arrival,
    )


def model_payload(service: TinkerService, tenant: str = "tenant", session_id: str | None = None, **overrides) -> dict:
    return {
        "base_model": service.config.base_model,
        "session_id": session_id or service.create_session(tenant),
        "model_seq_id": 1,
        **overrides,
    }


async def created_model(service: TinkerService, tenant: str = "tenant", session_id: str | None = None) -> str:
    request_id, model_id = service.create_model(
        tenant, model_payload(service, tenant, session_id, lora_config={"rank": 8})
    )
    future = await await_settled(service, tenant, request_id)
    assert future.state == DONE, future.error
    return model_id


async def await_settled(service: TinkerService, tenant: str, request_id: str, timeout: float = 2.0) -> RequestFuture:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        future = service.retrieve_future(tenant, request_id)
        assert future is not None, f"future {request_id} expired"
        if future.state != PENDING:
            return future
        await asyncio.sleep(0.005)
    raise AssertionError(f"future {request_id} still pending after {timeout}s")
