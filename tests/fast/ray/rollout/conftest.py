from __future__ import annotations

import textwrap
from argparse import Namespace
from collections.abc import AsyncIterator
from typing import Any, TypeVar
from unittest.mock import MagicMock

import pytest
import ray
from tests.fast.fixtures.rollout_args_fixtures import make_args

from miles.ray.specs.inference import inference_controller_worker_name
from miles.utils import object_store
from miles.utils.args.component_rollout import InferenceRuntimeImmutState
from miles.utils.args.configs.router import RouterConfig
from miles.utils.args.custom_function import CustomFunctionConfig
from miles.utils.args.runtime import AllConfig, InferenceControllerConfig, RolloutConfig
from miles.utils.types import Sample


def make_rollout_config(**overrides: Any) -> RolloutConfig:
    """The typed config the rollout executor receives, sliced from ``make_args``."""
    return _slice_config(RolloutConfig, make_args(**overrides))


def make_inference_controller_config(args: Namespace) -> InferenceControllerConfig:
    """The typed config the inference controller receives, sliced from a ``make_args`` namespace."""
    return _slice_config(InferenceControllerConfig, args)


def _slice_config(config_class: type[_LeafConfigT], args: Namespace) -> _LeafConfigT:
    values = vars(args) | _RESOLVED_ROLLOUT_FIELDS | RouterConfig.from_args(args)
    for name in _CUSTOM_FUNCTION_FIELDS:
        if isinstance(path := values[name], str):
            values[name] = CustomFunctionConfig(path=path)
    return config_class.model_validate({name: values[name] for name in config_class.model_fields if name in values})


_LeafConfigT = TypeVar("_LeafConfigT", RolloutConfig, InferenceControllerConfig)

_RESOLVED_ROLLOUT_FIELDS: dict[str, Any] = dict(
    ci_enable_metrics_capture=False,
    eval_datasets=[],
    ckpt_step=None,
    num_layers=None,
    raw_fsdp=None,
    raw_torchtitan=None,
    lora_A_init_method="xavier",
    lora_B_init_method="zero",
    lora_sync_sends_base=True,
)

_CUSTOM_FUNCTION_FIELDS = [
    name
    for name, field in AllConfig.model_fields.items()
    if field.annotation in {CustomFunctionConfig, CustomFunctionConfig | None}
]


class FakeInferenceTopologyProvider:
    """Serves the inference controller the rollout executor asks for the observed engine topology."""

    def __init__(self, state: InferenceRuntimeImmutState) -> None:
        self._state = state

    def get_handle(self, worker_name: str) -> FakeInferenceTopologyProvider:
        assert worker_name == inference_controller_worker_name()
        return self

    async def get_inference_runtime_immut_state(self) -> InferenceRuntimeImmutState:
        return self._state


def make_sample(
    *,
    group_index: int = 0,
    index: int = 0,
    response_length: int = 4,
    reward: float | dict | None = 1.0,
    status: Sample.Status = Sample.Status.COMPLETED,
    **overrides: Any,
) -> Sample:
    """Build a Sample with sensible defaults. Token list defaults to a length
    matching ``response_length`` so loss_mask/effective_response_length checks pass."""
    s = Sample(
        group_index=group_index,
        index=index,
        prompt="prompt",
        tokens=list(range(response_length)),
        response="response",
        response_length=response_length,
        label="label",
        reward=reward,
        status=status,
    )
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


def make_samples_grouped(
    n_groups: int,
    group_size: int,
    *,
    rewards: list[float] | None = None,
    response_length: int = 4,
) -> list[Sample]:
    """Construct ``n_groups * group_size`` samples laid out group-by-group.

    If ``rewards`` is given, must have length n_groups*group_size."""
    total = n_groups * group_size
    if rewards is not None:
        assert len(rewards) == total, f"rewards must have length {total}, got {len(rewards)}"
    samples: list[Sample] = []
    for g in range(n_groups):
        for k in range(group_size):
            i = g * group_size + k
            r = rewards[i] if rewards is not None else float(k)
            samples.append(
                make_sample(
                    group_index=g,
                    index=i,
                    reward=r,
                    response_length=response_length,
                )
            )
    return samples


def make_sglang_config_yaml(
    *,
    name: str = "default",
    server_groups: list[dict] | None = None,
    update_weights: bool | None = None,
    model_path: str | None = None,
) -> str:
    """Render a small SglangConfig YAML for from_file_arg() round-trip tests."""
    server_groups = server_groups or [{"worker_type": "regular", "num_gpus": 8, "num_gpus_per_engine": 1}]
    lines = ["sglang:", f"  - name: {name}"]
    if model_path is not None:
        lines.append(f"    model_path: {model_path}")
    if update_weights is not None:
        lines.append(f"    update_weights: {str(update_weights).lower()}")
    lines.append("    server_groups:")
    for g in server_groups:
        lines.append(f"      - worker_type: {g['worker_type']}")
        lines.append(f"        num_gpus: {g['num_gpus']}")
        if "num_gpus_per_engine" in g:
            lines.append(f"        num_gpus_per_engine: {g['num_gpus_per_engine']}")
    return "\n".join(lines) + "\n"


def make_args_with_sglang_config(tmp_path, *, server_groups: list[dict] | None = None, **overrides: Any) -> Namespace:
    """Args namespace pointed at a freshly written sglang config file."""
    config_path = tmp_path / "sglang.yaml"
    config_path.write_text(make_sglang_config_yaml(server_groups=server_groups))
    return make_args(sglang_config=str(config_path), **overrides)


# --------------------------- server cell fixtures ---------------------------

_tracked_server_cells: list[Any] = []


def track_server_cell(cell: Any) -> Any:
    """Register a cell for teardown. ``ServerCell.__del__`` asserts that every cell was disposed."""
    _tracked_server_cells.append(cell)
    return cell


@pytest.fixture
async def dispose_tracked_server_cells() -> AsyncIterator[None]:
    """Dispose every cell registered through ``track_server_cell`` during the test."""
    _tracked_server_cells.clear()
    yield
    for cell in _tracked_server_cells:
        await cell.dispose()
    _tracked_server_cells.clear()


# --------------------------- ray fixtures ---------------------------


@pytest.fixture
def ray_actor_baseline(ray_local_mode):
    """Snapshot live ray actor count before / after a test; asserts no leak."""
    import ray

    def _count():
        try:
            return len([a for a in ray.util.list_named_actors() if a])
        except Exception:
            return 0

    before = _count()
    yield
    after = _count()
    assert after <= before, f"Ray actor leaked: before={before} after={after}"


@pytest.fixture(autouse=True)
def _autouse_reset_object_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the process-wide object store singleton between tests."""
    monkeypatch.setattr(object_store, "_INSTANCE", None)


@pytest.fixture(autouse=True)
def _autouse_subprocess_leak_check(monkeypatch):
    """Catch leaked router / session-server children (multiprocessing and Popen)."""
    import multiprocessing

    from miles.utils.workers import process_utils

    launched: list = []
    real_launch = process_utils.launch_bound_subprocess

    def _recording_launch(argv, *, envs):
        process = real_launch(argv, envs=envs)
        launched.append(process)
        return process

    monkeypatch.setattr(process_utils, "launch_bound_subprocess", _recording_launch)

    before = {p.pid for p in multiprocessing.active_children()}
    yield
    leaked_mp = {p.pid for p in multiprocessing.active_children()} - before
    leaked_popen = [p for p in launched if p.poll() is None]
    if leaked_mp or leaked_popen:
        # Tear down leaked children to avoid cascading test failures.
        for p in multiprocessing.active_children():
            if p.pid in leaked_mp:
                try:
                    p.terminate()
                    p.join(timeout=2)
                except Exception:
                    pass
        for p in leaked_popen:
            process_utils.terminate_process_tree(p)
        raise AssertionError(
            f"Subprocess leaked from previous test: mp={leaked_mp} popen={[p.pid for p in leaked_popen]}"
        )


def dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


FAKE_SERVICE_PORT_START = 20000


def fake_engine(host: str = "10.0.0.1", port_seed: int = 30000) -> MagicMock:
    """MagicMock that mimics the engine ``CommandActor`` enough for ``addr_allocator``.

    Mocks ``_get_free_port_block.remote(start_port, count)`` with a
    deterministic ``max(seq, start_port)`` counter so allocator tests can
    predict and assert on port assignment. A node without a cursor starts at
    20000, where the service port range starts on a host with the default
    ephemeral range. ``_get_node_ip.remote()`` is the
    node-ip probe, which the cell awaits, so it returns an awaitable just like
    a real ``ObjectRef``. It also passes ``isinstance(x, ray.actor.ActorHandle)``
    so it can be handed to ``mark_allocated_uninitialized``."""
    e = MagicMock()
    e._spec_class = ray.actor.ActorHandle
    e._port_cursor = port_seed

    async def _alloc(start_port: int | None = None, count: int = 1):
        port = max(e._port_cursor, FAKE_SERVICE_PORT_START if start_port is None else start_port)
        e._port_cursor = port + count
        return port

    async def _probe():
        return host

    e._get_free_port_block.remote.side_effect = _alloc
    e._get_node_ip.remote.side_effect = _probe
    return e
