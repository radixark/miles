import asyncio
from collections import defaultdict
from collections.abc import Callable
from typing import Any


class FakeRemoteMethod:
    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self._fn(*args, **kwargs)


class FakeRolloutExecutor:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.generation_gates: dict[int, asyncio.Event] = {}
        self.generation_errors: list[BaseException | None] = []
        self.get = FakeRemoteMethod(self._get)
        self.eval = FakeRemoteMethod(self._eval)
        self.save = FakeRemoteMethod(self._save)
        self.dispose = FakeRemoteMethod(self._dispose)
        self.report_eval_skip = FakeRemoteMethod(self._report_eval_skip)

    async def _get(self, rollout_id: int) -> dict[str, Any]:
        self.events.append(f"generate_start:{rollout_id}")
        if (gate := self.generation_gates.get(rollout_id)) is not None:
            await gate.wait()
        if self.generation_errors and (error := self.generation_errors.pop(0)) is not None:
            self.events.append(f"generate_failed:{rollout_id}")
            raise error
        self.events.append(f"generate_done:{rollout_id}")
        return {"data_ref": f"rollout-data-{rollout_id}"}

    async def _eval(self, rollout_id: int, **_kwargs: Any) -> None:
        self.events.append(f"eval:{rollout_id}")

    async def _save(self, rollout_id: int) -> None:
        self.events.append(f"executor_save:{rollout_id}")

    async def _dispose(self) -> None:
        self.events.append("executor_dispose")

    async def _report_eval_skip(self, rollout_id: int, reason: str) -> None:
        self.events.append(f"eval_skip:{rollout_id}:{reason}")


class FakeInferenceController:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.check_weights_calls: list[dict[str, Any]] = []

    async def prepare_rollout(self, rollout_id: int) -> None:
        self.events.append(f"prepare_rollout:{rollout_id}")

    async def prepare_eval(self) -> None:
        self.events.append("prepare_eval")

    async def check_weights(self, **kwargs: Any) -> None:
        self.check_weights_calls.append(kwargs)
        self.events.append("check_weights")

    async def onload_kv(self) -> None:
        self.events.append("onload_kv")

    async def onload_weights(self) -> None:
        self.events.append("onload_weights")

    async def offload(self, tags: list[str]) -> None:
        self.events.append(f"offload:{','.join(tags)}")

    async def dispose(self) -> None:
        self.events.append("inference_dispose")


class FakeTrainingModel:
    def __init__(self, events: list[str], role: str) -> None:
        self.events = events
        self.role = role
        self.trained: list[int] = []
        self.saved: list[int] = []
        self.train_started: dict[int, asyncio.Event] = defaultdict(asyncio.Event)

    async def train(self, rollout_id: int, rollout_data: Any, external_data: Any = None) -> str:
        self.events.append(f"{self.role}_train:{rollout_id}")
        self.trained.append(rollout_id)
        self.train_started[rollout_id].set()
        return f"{self.role}-values-{rollout_id}"

    async def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        self.events.append(f"{self.role}_save:{rollout_id}")
        self.saved.append(rollout_id)

    async def export_hf(self, rollout_id: int, path: str) -> None:
        self.events.append(f"{self.role}_export_hf:{rollout_id}")

    async def reconcile_adapters(self) -> None:
        self.events.append(f"{self.role}_reconcile_adapters")

    async def offload(self) -> None:
        self.events.append(f"{self.role}_offload")

    async def onload(self) -> None:
        self.events.append(f"{self.role}_onload")

    async def clear_memory(self) -> None:
        self.events.append(f"{self.role}_clear_memory")

    async def dispose(self) -> None:
        self.events.append(f"{self.role}_dispose")
