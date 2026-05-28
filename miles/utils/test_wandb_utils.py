from types import SimpleNamespace

from miles.utils.tracking_utils import wandb_utils


def test_wandb_rollout_submetrics_use_rollout_step(monkeypatch):
    define_calls = []

    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: define_calls.append((args, kwargs)))

    wandb_utils._init_wandb_common()

    rollout_step_patterns = [
        "rollout/response_len/*",
        "rollout/zero_std/*",
        "rollout/weight_version/*",
        "rollout/tito_session_mismatch_rate/*",
        "rollout/error_cat/*",
        "rollout/dynamic_filter/*",
        "perf/non_generation_time/*",
        "multi_turn/raw_response_length/*",
        "multi_turn/wo_obs_response_length/*",
        "multi_turn/multi_turn_metric/*",
    ]
    for pattern in rollout_step_patterns:
        assert ((pattern,), {"step_metric": "rollout/step"}) in define_calls

    eval_step_patterns = [
        "eval/*/response_len/*",
        "eval/*/zero_std/*",
        "eval/*/weight_version/*",
        "eval/*/tito_session_mismatch_rate/*",
        "eval/*/error_cat/*",
    ]
    for pattern in eval_step_patterns:
        assert ((pattern,), {"step_metric": "eval/step"}) in define_calls


def _args(**overrides):
    values = {
        "env_report": None,
        "rank": 0,
        "sglang_enable_metrics": False,
        "use_wandb": True,
        "wandb_dir": None,
        "wandb_group": "group",
        "wandb_host": None,
        "wandb_key": None,
        "wandb_mode": None,
        "wandb_project": "project",
        "wandb_random_suffix": False,
        "wandb_run_id": None,
        "wandb_team": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_primary_wandb_init_uses_extended_init_timeout(monkeypatch):
    init_calls = []

    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb_utils.wandb, "run", SimpleNamespace(id="run-id"), raising=False)

    args = _args()
    wandb_utils.init_wandb_primary(args)

    settings = init_calls[0]["settings"]
    assert settings.mode == "shared"
    assert settings.x_primary is True
    assert settings.init_timeout == 300.0
    assert args.wandb_run_id == "run-id"


def test_secondary_wandb_init_uses_extended_init_timeout(monkeypatch):
    init_calls = []

    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_calls.append(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)

    args = _args(wandb_run_id="run-id")
    wandb_utils.init_wandb_secondary(args)

    settings = init_calls[0]["settings"]
    assert settings.mode == "shared"
    assert settings.x_primary is False
    assert settings.x_update_finish_state is False
    assert settings.init_timeout == 300.0
