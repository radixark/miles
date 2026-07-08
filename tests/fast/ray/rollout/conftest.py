from __future__ import annotations

import textwrap
from argparse import Namespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import ray

from miles.utils.types import Sample


def fake_actor_handle() -> MagicMock:
    """MagicMock that passes ``isinstance(x, ray.actor.ActorHandle)``.

    Setting ``_spec_class`` directly (rather than ``spec=...``) keeps
    arbitrary-attribute auto-creation working so ``actor.shutdown.remote(...)``
    chains still resolve — ``ActorHandle`` routes its methods via
    ``__getattr__`` and they don't show up as class attributes."""
    m = MagicMock()
    m._spec_class = ray.actor.ActorHandle
    return m


def make_args(**overrides: Any) -> Namespace:
    """Args namespace covering every field touched by ``miles/ray/rollout/``.
    Adding a new field is fine; deleting one likely breaks tests."""
    defaults: dict[str, Any] = dict(
        # rollout core
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        num_gpus_per_node=8,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        n_samples_per_eval_prompt=4,
        rollout_max_response_len=512,
        rollout_temperature=1.0,
        over_sampling_batch_size=None,
        rollout_global_dataset=False,
        num_rollout=1,
        # batch / training
        global_batch_size=8,
        use_dynamic_global_batch_size=False,
        disable_rollout_trim_samples=False,
        balance_data=False,
        delay_split_train_data_by_dp=False,
        # advantage / reward
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        reward_key=None,
        log_reward_category=None,
        log_passrate=False,
        # placement / colocation
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        critic_num_nodes=0,
        critic_num_gpus_per_node=0,
        use_critic=False,
        critic_train_only=False,
        # sglang router
        sglang_router_ip=None,
        sglang_router_port=None,
        sglang_router_policy=None,
        sglang_router_request_timeout_secs=600,
        sglang_dp_size=1,
        sglang_speculative_algorithm=None,
        sglang_config=None,
        sglang_model_routers=None,
        prefill_num_servers=None,
        # routers / session server
        use_miles_router=False,
        use_session_server=False,
        session_server_ip=None,
        session_server_port=None,
        # external rollout
        rollout_external=False,
        rollout_external_engine_addrs=None,
        # offload / fault tolerance
        offload_rollout=False,
        use_fault_tolerance=False,
        rollout_health_check_interval=10.0,
        rollout_health_check_timeout=30.0,
        # checkpoint / data source
        hf_checkpoint="/fake/model",
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout",
        eval_function_path="miles.rollout.sglang_rollout.eval_generate_rollout",
        data_source_path="miles.data.dummy.DummyDataSource",
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        custom_rollout_log_function_path=None,
        custom_eval_rollout_log_function_path=None,
        # debug data
        save_debug_rollout_data=None,
        load_debug_rollout_data=None,
        load_debug_rollout_data_subsample=None,
        ci_inject_rollout_data_path=None,
        ci_inject_rollout_data_start_rollout_id=None,
        ci_inject_rollout_data_min_match_ratio=0.9,
        # event checkpointing (event_logger.restore/snapshot in RolloutManager)
        save_debug_event_data=None,
        load=None,
        save=None,
        # CI
        ci_test=False,
        # dumper (sglang debug dumper integration)
        dumper_enable=False,
        dumper_inference=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


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
    """Render a small SglangConfig YAML for from_yaml() round-trip tests."""
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
def _autouse_subprocess_leak_check():
    """Catch leaked router / session-server multiprocessing children."""
    import multiprocessing

    before = {p.pid for p in multiprocessing.active_children()}
    yield
    after = {p.pid for p in multiprocessing.active_children()}
    leaked = after - before
    if leaked:
        # Tear down leaked children to avoid cascading test failures.
        for p in multiprocessing.active_children():
            if p.pid in leaked:
                try:
                    p.terminate()
                    p.join(timeout=2)
                except Exception:
                    pass
        raise AssertionError(f"Subprocess leaked from previous test: pids={leaked}")


def dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def make_dataclass_group(
    *,
    num_engines: int = 2,
    num_gpus_per_engine: int = 1,
    gpu_offset: int = 0,
):
    """Build a ``ServerGroup`` with ``pg=None`` (no actor scheduling). Each
    engine starts unallocated."""
    from miles.ray.rollout.server_engine import ServerEngine
    from miles.ray.rollout.server_group import ServerGroup

    args = make_args(num_gpus_per_node=8)
    engines = [ServerEngine() for _ in range(num_engines)]
    return ServerGroup(
        args=args,
        pg=None,
        all_engines=engines,
        num_gpus_per_engine=num_gpus_per_engine,
        has_new_engines=False,
        gpu_offset=gpu_offset,
        update_weights=True,
    )


def fake_engine(host: str = "10.0.0.1", port_seed: int = 30000) -> MagicMock:
    """MagicMock that mimics ``SGLangEngine`` enough for ``addr_allocator``.

    Mocks ``_get_current_node_ip_and_free_port.remote(start_port, consecutive)``
    with a deterministic ``max(seq, start_port)`` counter so allocator tests
    can predict and assert on port assignment."""
    e = MagicMock()
    e._port_cursor = port_seed

    def _alloc(start_port: int = 15000, consecutive: int = 1):
        port = max(e._port_cursor, start_port)
        e._port_cursor = port + consecutive
        return (host, port)

    e._get_current_node_ip_and_free_port.remote.side_effect = lambda **kw: _alloc(**kw)
    return e


@pytest.fixture
def patch_ray_get(monkeypatch):
    """Make ``ray.get(remote_call(...))`` return the MagicMock's value directly,
    so allocator tests don't need a real Ray cluster."""
    import miles.ray.rollout.addr_allocator as mod

    monkeypatch.setattr(mod.ray, "get", lambda x: x)
