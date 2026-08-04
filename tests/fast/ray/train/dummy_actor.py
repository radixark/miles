"""Lightweight Ray actor for unit testing TrainerCell/TrainerController without GPU or real training.

Records all method calls so tests can verify what was dispatched.
"""

import os
from typing import Any

import ray

from miles.backends.megatron_utils.ft.types import TrainStepOutcome, TrainStepOutput
from miles.utils.ft_utils.heartbeat_utils import HeartbeatStatus, SimpleHeartbeat


@ray.remote(num_gpus=0, num_cpus=0)
class DummyTrainActor:

    def __init__(self):
        self._calls: list[tuple[str, tuple, dict]] = []
        self._fail_methods: set[str] = set()
        self._train_return_value: Any = TrainStepOutput(outcome=TrainStepOutcome.NORMAL)
        self._train_return_values_per_attempt: list[Any] = []
        self._update_weights_return_value: Any = None
        self._train_parallel_config: dict = {}
        self._heartbeat = SimpleHeartbeat()
        self._heartbeat.bump()
        self._heartbeat_fail: bool = False

    def set_fail_methods(self, methods: list[str]) -> None:
        self._fail_methods = set(methods)

    def set_train_return_value(self, value: Any) -> None:
        self._train_return_value = value

    def set_train_return_values_per_attempt(self, values: list[Any]) -> None:
        self._train_return_values_per_attempt = list(values)

    def set_train_parallel_config(self, config: dict) -> None:
        self._train_parallel_config = config

    def _record(self, method: str, args: tuple, kwargs: dict) -> None:
        self._calls.append((method, args, kwargs))
        if method in self._fail_methods:
            raise RuntimeError(f"Injected failure in {method}")

    def get_calls(self) -> list[tuple[str, tuple, dict]]:
        return list(self._calls)

    def init(self, *args: Any, **kwargs: Any) -> None:
        self._record("init", args, kwargs)

    def configure_master_addr_and_port(self, *args: Any, **kwargs: Any) -> None:
        self._record("configure_master_addr_and_port", args, kwargs)

    def reconfigure_indep_dp(self, *args: Any, **kwargs: Any) -> None:
        self._record("reconfigure_indep_dp", args, kwargs)

    def send_ckpt(self, *args: Any, **kwargs: Any) -> None:
        self._record("send_ckpt", args, kwargs)

    def train(self, *args: Any, **kwargs: Any) -> Any:
        self._record("train", args, kwargs)
        if self._train_return_values_per_attempt:
            return self._train_return_values_per_attempt.pop(0)
        return self._train_return_value

    def reconcile_adapters(self) -> None:
        self._record("reconcile_adapters", (), {})

    def get_train_parallel_config(self) -> dict:
        self._record("get_train_parallel_config", (), {})
        return self._train_parallel_config

    def wake_up(self) -> None:
        self._record("wake_up", (), {})

    def sleep(self) -> None:
        self._record("sleep", (), {})

    def clear_memory(self) -> None:
        self._record("clear_memory", (), {})

    def save_model(self, *args: Any, **kwargs: Any) -> None:
        self._record("save_model", args, kwargs)

    def export_hf(self, *args: Any, **kwargs: Any) -> None:
        self._record("export_hf", args, kwargs)

    def set_update_weights_return_value(self, value: Any) -> None:
        self._update_weights_return_value = value

    def update_weights(self, *args: Any, **kwargs: Any) -> Any:
        self._record("update_weights", args, kwargs)
        return self._update_weights_return_value

    def kill_self(self) -> None:
        self._record("kill_self", (), {})
        os._exit(1)

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
