"""Shared fakes for the tinker gateway suite."""

import asyncio

from miles.tinker.core.future import DONE, PENDING, Future
from miles.tinker.core.service import ExecutorBackend, TinkerService
from miles.tinker.core.types import Command, CommandOp, GatewayConfig

ADAM = {
    "learning_rate": 1e-4,
    "beta1": 0.9,
    "beta2": 0.95,
    "eps": 1e-12,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
}


class FakeBackend(ExecutorBackend):
    """Record calls with deterministic outputs and configurable failures."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_next: Exception | None = None
        self.optim_outcomes: dict[int, dict] = {}
        self.fail_on: dict[str, Exception] = {}

    def _record(self, name: str, **kwargs) -> None:
        self.calls.append((name, kwargs))
        if self.fail_next is not None:
            error, self.fail_next = self.fail_next, None
            raise error
        error = self.fail_on.pop(name, None)
        if error is not None:
            raise error

    def named(self, name: str) -> list[dict]:
        return [kwargs for called, kwargs in self.calls if called == name]

    async def load_slot(self, slot, rank, alpha, ckpt_path=None, load_optimizer=True):
        self._record(
            "load_slot", slot=slot, rank=rank, alpha=alpha, ckpt_path=ckpt_path, load_optimizer=load_optimizer
        )

    async def unload_slot(self, slot):
        self._record("unload_slot", slot=slot)

    async def forward_backward(self, batch_id, slot_datums, loss_fn, loss_fn_config):
        self._record(
            "forward_backward",
            batch_id=batch_id,
            slot_datums=slot_datums,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
        return [{"loss": 1.0, "logprobs": [0.0] * datum["target_len"]} for _, datum in slot_datums]

    async def forward_only(self, batch_id, slot_datums, loss_fn, loss_fn_config):
        self._record(
            "forward_only", batch_id=batch_id, slot_datums=slot_datums, loss_fn=loss_fn, loss_fn_config=loss_fn_config
        )
        return [{"loss": 0.0, "logprobs": [0.0] * datum["target_len"]} for _, datum in slot_datums]

    async def optim_step(self, adam_params_by_slot):
        self._record("optim_step", adam_params_by_slot=adam_params_by_slot)
        return {slot: self.optim_outcomes.get(slot, {"grad_norm": 0.5 + slot}) for slot in adam_params_by_slot}

    async def zero_grads(self, slot):
        self._record("zero_grads", slot=slot)

    async def save_slot(self, slot, path):
        self._record("save_slot", slot=slot, path=path)

    async def export_slot(self, slot, rank, alpha, path):
        self._record("export_slot", slot=slot, rank=rank, alpha=alpha, path=path)

    async def push_slot(self, slot, lora_name, rank, alpha, lora_path=None):
        self._record("push_slot", slot=slot, lora_name=lora_name, rank=rank, alpha=alpha, lora_path=lora_path)

    async def sample(self, payload, lora_name, lora_path=None):
        self._record("sample", payload=payload, lora_name=lora_name, lora_path=lora_path)
        return {
            "sequences": [
                {"sequence_id": f"seq-{i}", "tokens": [1, 2], "logprobs": [0.0, 0.0], "stop_reason": "stop"}
                for i in range(payload["num_samples"])
            ]
        }


def make_config(checkpoint_root, **overrides) -> GatewayConfig:
    defaults = dict(base_model="base", n_slots=2, checkpoint_root=str(checkpoint_root))
    return GatewayConfig(**{**defaults, **overrides})


def make_service(checkpoint_root, **config_overrides) -> TinkerService:
    return TinkerService(FakeBackend(), make_config(checkpoint_root, **config_overrides))


def datum(tokens: int = 3) -> dict:
    return {"tokens": list(range(tokens + 1)), "target_len": tokens, "weights": [1.0] * tokens}


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


async def created_model(service: TinkerService, tenant: str = "tenant") -> str:
    request_id, model_id = service.create_model(
        tenant, {"base_model": service.config.base_model, "lora_config": {"rank": 8}}
    )
    future = await await_settled(service, tenant, request_id)
    assert future.state == DONE, future.error
    return model_id


async def await_settled(service: TinkerService, tenant: str, request_id: str, timeout: float = 2.0) -> Future:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        future = service.retrieve_future(tenant, request_id)
        assert future is not None, f"future {request_id} expired"
        if future.state != PENDING:
            return future
        await asyncio.sleep(0.005)
    raise AssertionError(f"future {request_id} still pending after {timeout}s")
