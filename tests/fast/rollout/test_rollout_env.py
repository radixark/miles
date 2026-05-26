from miles.ray.rollout_env import build_sglang_rollout_env_vars


def test_rollout_deepgemm_override_is_sglang_actor_local(monkeypatch):
    monkeypatch.setenv("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM", "0")
    monkeypatch.setenv("SGLANG_ROLLOUT_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM", "1")

    env_vars = build_sglang_rollout_env_vars()

    assert env_vars["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"] == "1"


def test_rollout_deepgemm_default_follows_global_env(monkeypatch):
    monkeypatch.setenv("SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM", "0")
    monkeypatch.delenv("SGLANG_ROLLOUT_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM", raising=False)

    env_vars = build_sglang_rollout_env_vars()

    assert env_vars["SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM"] == "0"
