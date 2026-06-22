"""Lightweight Ray actor for unit testing RayTrainCell/RayTrainGroup without GPU or real training.

Records all method calls so tests can verify what was dispatched.
"""

from typing import Any

import ray

from miles.backends.megatron_utils.types import TrainStepOutcome
from miles.utils.heartbeat_utils import HeartbeatStatus, SimpleHeartbeat


@ray.remote(num_gpus=0, num_cpus=0)
class DummyTrainActor:

    def __init__(self):
        self._calls: list[tuple[str, tuple, dict]] = []
        self._fail_methods: set[str] = set()
        self._train_return_value: Any = TrainStepOutcome.NORMAL
        self._heartbeat = SimpleHeartbeat()
        self._heartbeat.bump()
        self._heartbeat_fail: bool = False

    def set_fail_methods(self, methods: list[str]) -> None:
        self._fail_methods = set(methods)

    def set_train_return_value(self, value: Any) -> None:
        self._train_return_value = value

    def _record(self, method: str, args: tuple, kwargs: dict) -> None:
        self._calls.append((method, args, kwargs))
        if method in self._fail_methods:
            raise RuntimeError(f"Injected failure in {method}")

    def get_calls(self) -> list[tuple[str, tuple, dict]]:
        return list(self._calls)

    def init(self, *args: Any, **kwargs: Any) -> None:
        self._record("init", args, kwargs)

    def reconfigure_indep_dp(self, *args: Any, **kwargs: Any) -> None:
        self._record("reconfigure_indep_dp", args, kwargs)

    def send_ckpt(self, *args: Any, **kwargs: Any) -> None:
        self._record("send_ckpt", args, kwargs)

    def train(self, *args: Any, **kwargs: Any) -> Any:
        self._record("train", args, kwargs)
        return self._train_return_value

    def set_rollout_manager(self, *args: Any, **kwargs: Any) -> None:
        self._record("set_rollout_manager", args, kwargs)

    def wake_up(self) -> None:
        self._record("wake_up", (), {})

    def sleep(self) -> None:
        self._record("sleep", (), {})

    def clear_memory(self) -> None:
        self._record("clear_memory", (), {})

    def save_model(self, *args: Any, **kwargs: Any) -> None:
        self._record("save_model", args, kwargs)

    def update_weights(self) -> None:
        self._record("update_weights", (), {})

    def set_heartbeat_fail(self, fail: bool) -> None:
        self._heartbeat_fail = fail

    def set_last_active_timestamp(self, ts: float) -> None:
        self._heartbeat._status = HeartbeatStatus(
            last_active_timestamp=ts,
            bump_count=self._heartbeat._status.bump_count,
        )

    def get_heartbeat_status(self) -> HeartbeatStatus:
        if self._heartbeat_fail:
            raise RuntimeError("Injected heartbeat failure")
        return self._heartbeat.status()
