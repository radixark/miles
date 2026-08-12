from __future__ import annotations

import textwrap
from argparse import ArgumentParser, Namespace
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest
import ray
from sglang_router.launch_router import RouterArgs

from miles.utils import object_store
from miles.utils.types import Sample


def make_args(**overrides: Any) -> Namespace:
    """Args namespace covering every field touched by ``miles/ray/rollout/``.
    Adding a new field is fine; deleting one likely breaks tests."""
    parser: ArgumentParser = ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    router_defaults: dict[str, Any] = vars(parser.parse_args([]))
    defaults: dict[str, Any] = dict(
        # rollout core
        rollout_num_gpus=8,
        rollout_num_gpus_per_engine=1,
        eval_num_gpus=0,
        eval_num_gpus_per_engine=1,
        eval_uses_snapshots=False,
        num_gpus_per_node=8,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        n_samples_per_eval_prompt=4,
        rollout_max_response_len=512,
        rollout_temperature=1.0,
        over_sampling_batch_size=None,
        rollout_global_dataset=False,
        num_rollout=1,
        update_weights_interval=1,
        check_weight_update_equal=False,
        check_weight_update_skip_list=None,
        # batch / training
        global_batch_size=8,
        use_dynamic_global_batch_size=False,
        wandb_always_use_train_step=False,
        disable_rollout_trim_samples=False,
        balance_data=False,
        delay_split_train_data_by_dp=False,
        # object store
        object_store_backend="ray",
        mooncake_store_init_kwargs=None,
        mooncake_replica_num=1,
        # advantage / reward
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        reward_key=None,
        log_reward_category=None,
        log_passrate=False,
        pin_rollout_manager_to_head=False,
        # placement / colocation
        debug_train_only=False,
        debug_rollout_only=False,
        debug_skip_weight_update=False,
        colocate=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=8,
        indep_dp=False,
        train_backend="megatron",
        kl_coef=0,
        use_kl_loss=False,
        use_opd=False,
        opd_type="megatron",
        train_env_vars={},
        dumper_source_patcher_config_train=None,
        offload_train=False,
        offload_train_target="cpu",
        offload_train_disk_dir="/tmp/offload",
        offload_train_disk_chunk_mb=64,
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
        sglang_pp_size=1,
        sglang_ep_size=1,
        sglang_api_key=None,
        multi_lora_n_adapters=0,
        target_modules=None,
        sglang_speculative_algorithm=None,
        sglang_config=None,
        sglang_model_routers=None,
        prefill_num_servers=None,
        # routers / session server
        use_miles_dashboard=False,
        use_miles_router=False,
        miles_router_timeout=None,
        miles_router_max_connections=None,
        miles_router_health_check_failure_threshold=3,
        sglang_server_concurrency=512,
        use_session_server=False,
        use_rollout_routing_replay=False,
        session_server_ip=None,
        session_server_port=None,
        session_server_workers=1,
        run_uuid="0123456789abcdef",
        # external rollout
        rollout_external=False,
        rollout_external_engine_addrs=None,
        # offload / fault tolerance
        offload_rollout=False,
        use_fault_tolerance=False,
        ft_components=[],
        rollout_health_check_interval=30.0,
        rollout_health_check_timeout=30.0,
        rollout_health_check_first_wait=0.0,
        rollout_health_check_failure_threshold=1,
        # engine launch command
        seed=42,
        fp16=False,
        use_rollout_indexer_replay=False,
        env_report=None,
        # checkpoint / data source
        hf_checkpoint="/fake/model",
        lora_rank=0,
        rollout_function_path="miles.rollout.sglang_rollout.generate_rollout",
        eval_function_path="miles.rollout.sglang_rollout.eval_generate_rollout",
        data_source_path="miles.data.dummy.DummyDataSource",
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        custom_rollout_log_function_path=None,
        custom_eval_rollout_log_function_path=None,
        # debug data
        save_debug_rollout_data=None,
        save_debug_trajectory_data=None,
        load_debug_rollout_data=None,
        load_debug_rollout_data_subsample=None,
        ci_inject_rollout_data_path=None,
        ci_inject_rollout_data_start_rollout_id=None,
        ci_inject_rollout_data_min_match_ratio=0.9,
        # event checkpointing (event_logger.restore/snapshot in RolloutExecutor)
        save_debug_event_data=None,
        load=None,
        save=None,
        # CI
        ci_test=False,
        # dumper (sglang debug dumper integration)
        dumper_enable=False,
        dumper_inference=False,
    )
    defaults.update(router_defaults)
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


def fake_engine(host: str = "10.0.0.1", port_seed: int = 30000) -> MagicMock:
    """MagicMock that mimics the engine ``CommandActor`` enough for ``addr_allocator``.

    Mocks ``_get_free_port_block.remote(start_port, count)`` with a
    deterministic ``max(seq, start_port)`` counter so allocator tests can
    predict and assert on port assignment. ``_get_node_ip.remote()`` is the
    node-ip probe, which the cell awaits, so it returns an awaitable just like
    a real ``ObjectRef``. It also passes ``isinstance(x, ray.actor.ActorHandle)``
    so it can be handed to ``mark_allocated_uninitialized``."""
    e = MagicMock()
    e._spec_class = ray.actor.ActorHandle
    e._port_cursor = port_seed

    def _alloc(start_port: int = 15000, count: int = 1):
        port = max(e._port_cursor, start_port)
        e._port_cursor = port + count
        return port

    async def _probe():
        return host

    e._get_free_port_block.remote.side_effect = _alloc
    e._get_node_ip.remote.side_effect = _probe
    return e


@pytest.fixture
def patch_ray_get(monkeypatch):
    """Make ``ray.get(remote_call(...))`` return the MagicMock's value directly,
    so allocator tests don't need a real Ray cluster."""
    import miles.utils.workers.addr_allocator as mod

    monkeypatch.setattr(mod.ray, "get", lambda x: x)
