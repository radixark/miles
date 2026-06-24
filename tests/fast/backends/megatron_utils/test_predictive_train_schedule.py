from miles.backends.megatron_utils.predictive_train_schedule import get_predictive_train_mode_for_step


def test_predictive_train_mode_skips_first_actor_step_only():
    # Paper Algorithm 3 line 3: skip predictive loss when mini-step i=1.
    # Code uses 0-indexed step_id so step_id=0 == paper's i=1.
    assert get_predictive_train_mode_for_step(role="actor", predictive_enabled=True, step_id=0) == "skip"
    assert get_predictive_train_mode_for_step(role="actor", predictive_enabled=True, step_id=1) == "compute"
    # Predictive off → always compute (= paper baseline).
    assert get_predictive_train_mode_for_step(role="actor", predictive_enabled=False, step_id=0) == "compute"
    # Critic never runs the predictive path.
    assert get_predictive_train_mode_for_step(role="critic", predictive_enabled=True, step_id=0) == "compute"
