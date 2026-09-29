from types import SimpleNamespace as NS

import pytest

from miles.backends.training_utils.weight_update.nccl import CHANNEL_ENV_KEYS
from miles.ray.specs import weight_update_env as policy
from miles.utils.workers.worker_spec import BaseWorkerSpec, SchedulingSpec, ServeWorkerSpec, WorkerLaunchContext


def group(*, tp=2, worker_type="regular", **overrides):
    return NS(num_gpus_per_engine=tp, worker_type=worker_type, overrides=overrides)


def model(*groups, update_weights=True):
    return NS(server_groups=list(groups), update_weights=update_weights)


def resolve(monkeypatch, *models, hip=False, channels=8, **overrides):
    monkeypatch.setattr(policy, "_deterministic_channels", lambda: channels)
    args = NS(sglang_enable_deterministic_inference=True, train_env_vars={}, **overrides)
    return resolve_policy(args, NS(models=list(models)), is_hip=hip)


def resolve_policy(args, config, *, is_hip):
    return policy.resolve_weight_update_channels(
        args,
        config,
        is_hip=is_hip,
        trainer_pool_ids=["trainer-engine-actor"],
        engine_pool_ids={
            (m, g): f"inference-engine-all-{m}-{g}"
            for m, model in enumerate(config.models)
            for g, _ in enumerate(model.server_groups)
        },
    )


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in (*CHANNEL_ENV_KEYS, "SGLANG_DETERMINISTIC_NCCL_NCHANNELS"):
        monkeypatch.delenv(key, raising=False)


def test_tp2_and_tp1_receivers_share_the_trainers_policy(monkeypatch):
    result = resolve(monkeypatch, model(group(), group(tp=1)), model(group(), update_weights=False))
    assert set(result) == {"trainer-engine-actor", "inference-engine-all-0-0", "inference-engine-all-0-1"}
    assert set(result.values()) == {8}


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
    assert result["trainer-engine-actor"] == 16
    assert result["inference-engine-all-0-0"] == 16


@pytest.mark.parametrize("key", CHANNEL_ENV_KEYS)
def test_inherited_conflicts_fail_before_launch(monkeypatch, key):
    monkeypatch.setenv(key, "24")
    with pytest.raises(ValueError, match=f"{key}=24"):
        resolve(monkeypatch, model(group()))


def test_explicit_trainer_conflict_fails(monkeypatch):
    monkeypatch.setattr(policy, "_deterministic_channels", lambda: 8)
    args = NS(sglang_enable_deterministic_inference=True, train_env_vars={"NCCL_MIN_NCHANNELS": "24"})
    with pytest.raises(ValueError, match="trainer sets NCCL_MIN_NCHANNELS=24"):
        resolve_policy(args, NS(models=[model(group())]), is_hip=False)


def test_matching_explicit_values_are_accepted(monkeypatch):
    for key in CHANNEL_ENV_KEYS:
        monkeypatch.setenv(key, "8")
    assert resolve(monkeypatch, model(group()))["trainer-engine-actor"] == 8


@pytest.mark.parametrize("count", [0, -1])
def test_invalid_count_fails(monkeypatch, count):
    with pytest.raises(ValueError, match="positive integer"):
        resolve(monkeypatch, model(group()), channels=count)


def test_older_sglang_without_channel_policy_is_unchanged(monkeypatch):
    assert resolve(monkeypatch, model(group()), channels=None) == {}


def test_worker_specs_preserve_other_environment_and_restarts(monkeypatch):
    original_env = {"NCCL_ALGO": "Ring", "OTHER": "value"}
    original_args = NS(existing="value")
    original_kwargs = {"args": original_args, "rank": 0}
    specs = [
        ServeWorkerSpec(
            name="trainer-engine-actor",
            port_infos=[],
            scheduling=SchedulingSpec.single(num_gpus_per_worker=1),
            env_var=lambda ctx: original_env,
            worker_class="test.Actor",
            ctor_kwargs=lambda ctx: original_kwargs,
        ),
        *[
            BaseWorkerSpec(
                name=name,
                port_infos=[],
                scheduling=SchedulingSpec.single(num_gpus_per_worker=1),
                env_var=lambda ctx: original_env,
            )
            for name in ["trainer-engine-critic", "inference-engine-all-0-0", "session-server"]
        ],
    ]
    updated = policy.apply_weight_update_channels(specs, resolve(monkeypatch, model(group())))
    ctx = WorkerLaunchContext(cell_index=0, worker_in_cell_index=0, gpu_ids=[0])
    for _ in range(2):
        assert updated[0].env_var(ctx) == original_env
        kwargs = updated[0].ctor_kwargs(ctx)
        assert kwargs["args"]._weight_update_nccl_channels == 8
        assert kwargs["args"].existing == "value" and kwargs["rank"] == 0
        assert kwargs["args"] is not original_args
        assert updated[2].env_var(ctx) == {
            **original_env,
            "NCCL_MIN_NCHANNELS": "8",
            "NCCL_MAX_NCHANNELS": "8",
            "SGLANG_DETERMINISTIC_NCCL_NCHANNELS": "8",
        }
    assert updated[1] is specs[1] and updated[3] is specs[3]
    assert not hasattr(original_args, "_weight_update_nccl_channels")
    assert original_env == {"NCCL_ALGO": "Ring", "OTHER": "value"}


def test_worker_specific_overrides_cannot_undo_policy():
    with pytest.raises(ValueError, match="NCCL_MAX_NCHANNELS=24"):
        policy._worker_env(
            None,
            original=lambda ctx: {"NCCL_MAX_NCHANNELS": "24"},
            channels=8,
            role="serving",
        )


def test_group_override_can_enable_determinism(monkeypatch):
    monkeypatch.setattr(policy, "_deterministic_channels", lambda: 16)
    args = NS(sglang_enable_deterministic_inference=False, train_env_vars={})
    config = NS(models=[model(group(enable_deterministic_inference=True))])
    assert resolve_policy(args, config, is_hip=False)["trainer-engine-actor"] == 16


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
    assert "trainer-engine-actor" in resolve_policy(args, config, is_hip=False)


def test_skipped_weight_updates_keep_existing_environment(monkeypatch):
    assert resolve(monkeypatch, model(group()), debug_skip_weight_update=True) == {}


def test_named_deployment_pool_ids_are_preserved(monkeypatch):
    monkeypatch.setattr(policy, "_deterministic_channels", lambda: 8)
    args = NS(sglang_enable_deterministic_inference=True, train_env_vars={})
    result = policy.resolve_weight_update_channels(
        args,
        NS(models=[model(group())]),
        is_hip=False,
        trainer_pool_ids=["trainer-engine-policy-a-actor", "trainer-engine-policy-b-actor"],
        engine_pool_ids={(0, 0): "inference-engine-custom-release-0-0"},
    )
    assert result == {
        "trainer-engine-policy-a-actor": 8,
        "trainer-engine-policy-b-actor": 8,
        "inference-engine-custom-release-0-0": 8,
    }
