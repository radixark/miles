from types import SimpleNamespace as NS

import pytest

from miles.ray.specs import weight_update_env as policy
from miles.utils.workers.worker_spec import BaseWorkerSpec, SchedulingSpec, WorkerLaunchContext


def group(*, tp=2, worker_type="regular", **overrides):
    return NS(num_gpus_per_engine=tp, worker_type=worker_type, overrides=overrides)


def model(*groups, update_weights=True):
    return NS(server_groups=list(groups), update_weights=update_weights)


def resolve(monkeypatch, *models, hip=False, channels=8, **overrides):
    monkeypatch.setattr(policy, "_deterministic_channels", lambda: channels)
    args = NS(sglang_enable_deterministic_inference=True, train_env_vars={}, **overrides)
    return policy.resolve_weight_update_env(args, NS(models=list(models)), is_hip=hip)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in (*policy._CHANNEL_KEYS, "SGLANG_DETERMINISTIC_NCCL_NCHANNELS"):
        monkeypatch.delenv(key, raising=False)


def test_tp2_and_tp1_receivers_share_the_trainers_policy(monkeypatch):
    result = resolve(monkeypatch, model(group(), group(tp=1)), model(group(), update_weights=False))
    assert set(result) == {"trainer-actor", "inference-engine-0-0", "inference-engine-0-1"}
    for env in result.values():
        assert env["NCCL_MIN_NCHANNELS"] == env["NCCL_MAX_NCHANNELS"] == "8"
        assert "NCCL_ALGO" not in env


@pytest.mark.parametrize(
    "overrides",
    [
        {"colocate": True},
        {"debug_train_only": True},
        {"debug_rollout_only": True},
        {"rollout_external": True},
        {"update_weight_transfer_mode": "disk-delta"},
        {"update_weight_transfer_mode": "p2p"},
    ],
)
def test_unaffected_launches_keep_their_environment(monkeypatch, overrides):
    assert resolve(monkeypatch, model(group()), **overrides) == {}


@pytest.mark.parametrize(
    "models",
    [
        [model(group(tp=1))],
        [model(group(worker_type="placeholder"))],
        [model(group(), update_weights=False)],
        [model(group(enable_deterministic_inference=False))],
        [model(group(device="cpu"))],
        [model(group(tp_size=1))],
        [],
    ],
)
def test_only_effective_deterministic_cuda_tp_groups_trigger_policy(monkeypatch, models):
    assert resolve(monkeypatch, *models) == {}


def test_rocm_keeps_its_existing_policy(monkeypatch):
    assert resolve(monkeypatch, model(group()), hip=True) == {}


def test_custom_sglang_count_reaches_both_roles(monkeypatch):
    result = resolve(monkeypatch, model(group()), channels=16)
    assert result["trainer-actor"]["NCCL_MAX_NCHANNELS"] == "16"
    assert result["inference-engine-0-0"]["SGLANG_DETERMINISTIC_NCCL_NCHANNELS"] == "16"


@pytest.mark.parametrize("key", policy._CHANNEL_KEYS)
def test_inherited_conflicts_fail_before_launch(monkeypatch, key):
    monkeypatch.setenv(key, "24")
    with pytest.raises(ValueError, match=f"{key}=24"):
        resolve(monkeypatch, model(group()))


def test_explicit_trainer_conflict_fails(monkeypatch):
    monkeypatch.setattr(policy, "_deterministic_channels", lambda: 8)
    args = NS(sglang_enable_deterministic_inference=True, train_env_vars={"NCCL_MIN_NCHANNELS": "24"})
    with pytest.raises(ValueError, match="trainer sets NCCL_MIN_NCHANNELS=24"):
        policy.resolve_weight_update_env(args, NS(models=[model(group())]), is_hip=False)


def test_matching_explicit_values_are_accepted(monkeypatch):
    for key in policy._CHANNEL_KEYS:
        monkeypatch.setenv(key, "8")
    assert resolve(monkeypatch, model(group()))["trainer-actor"]["NCCL_MIN_NCHANNELS"] == "8"


@pytest.mark.parametrize("count", [0, -1])
def test_invalid_count_fails(monkeypatch, count):
    with pytest.raises(ValueError, match="positive integer"):
        resolve(monkeypatch, model(group()), channels=count)


def test_older_sglang_without_channel_policy_is_unchanged(monkeypatch):
    assert resolve(monkeypatch, model(group()), channels=None) == {}


def test_worker_specs_preserve_other_environment_and_restarts(monkeypatch):
    original_env = {"NCCL_ALGO": "Ring", "OTHER": "value"}
    specs = [
        BaseWorkerSpec(
            name=name,
            port_infos=[],
            scheduling=SchedulingSpec.single(num_gpus_per_worker=1),
            env_var=lambda ctx: original_env,
        )
        for name in ["trainer-actor", "trainer-critic", "inference-engine-0-0", "session-server"]
    ]
    envs = resolve(monkeypatch, model(group()))
    updated = policy.apply_weight_update_env(specs, envs)
    ctx = WorkerLaunchContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[0])
    for _ in range(2):
        assert updated[0].env_var(ctx) == {**original_env, **envs["trainer-actor"]}
        assert updated[2].env_var(ctx) == {**original_env, **envs["inference-engine-0-0"]}
    assert updated[1] is specs[1] and updated[3] is specs[3]
    assert original_env == {"NCCL_ALGO": "Ring", "OTHER": "value"}


def test_worker_specific_overrides_cannot_undo_policy():
    with pytest.raises(ValueError, match="NCCL_MAX_NCHANNELS=24"):
        policy._worker_env(
            None,
            original=lambda ctx: {"NCCL_MAX_NCHANNELS": "24"},
            required={"NCCL_MAX_NCHANNELS": "8"},
            role="serving",
        )


def test_group_override_can_enable_determinism(monkeypatch):
    monkeypatch.setattr(policy, "_deterministic_channels", lambda: 16)
    args = NS(sglang_enable_deterministic_inference=False, train_env_vars={})
    config = NS(models=[model(group(enable_deterministic_inference=True))])
    assert policy.resolve_weight_update_env(args, config, is_hip=False)["trainer-actor"]["NCCL_MIN_NCHANNELS"] == "16"


def test_sglang_setting_supplies_its_own_default(monkeypatch):
    import sys
    from types import ModuleType

    module = ModuleType("sglang.srt.environ")
    module.envs = NS(SGLANG_DETERMINISTIC_NCCL_NCHANNELS=NS(get=lambda: 12))
    monkeypatch.setitem(sys.modules, "sglang.srt.environ", module)
    assert policy._deterministic_channels() == 12
    module.envs = NS()
    assert policy._deterministic_channels() is None


@pytest.mark.parametrize(
    "name,value",
    [
        ("enable_prefill_only_deterministic_inference", True),
        ("rl_on_policy_target", "test-target"),
        ("true_on_policy_contract", "test-contract"),
    ],
)
def test_implied_deterministic_modes_are_resolved(monkeypatch, name, value):
    monkeypatch.setattr(policy, "_deterministic_channels", lambda: 8)
    args = NS(sglang_enable_deterministic_inference=False, train_env_vars={})
    config = NS(models=[model(group(**{name: value}))])
    assert "trainer-actor" in policy.resolve_weight_update_env(args, config, is_hip=False)


def test_skipped_weight_updates_keep_existing_environment(monkeypatch):
    assert resolve(monkeypatch, model(group()), debug_skip_weight_update=True) == {}
