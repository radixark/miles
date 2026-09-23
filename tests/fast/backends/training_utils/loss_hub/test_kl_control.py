from argparse import Namespace

import pytest

from miles.backends.training_utils.loss_hub.kl_control import update_adaptive_kl


def _make_args(**overrides) -> Namespace:
    defaults = dict(kl_ctrl="adaptive", kl_loss_coef=0.5, kl_target=0.1, kl_horizon=10.0, kl_ctrl_steps=1)
    return Namespace(**{**defaults, **overrides})


def test_fixed_is_a_noop():
    args = _make_args(kl_ctrl="fixed")
    before = vars(args).copy()

    update_adaptive_kl(args, {"kl_loss": 1.0})

    assert vars(args) == before


def test_missing_controller_fields_defaults_to_fixed():
    args = Namespace(kl_loss_coef=0.5)
    before = vars(args).copy()

    update_adaptive_kl(args, {"kl_loss": 1.0})

    assert vars(args) == before


@pytest.mark.parametrize(("kl", "expected_coef"), [(0.11, 0.505), (0.09, 0.495), (0.1, 0.5)])
def test_adaptive_tracks_target(kl, expected_coef):
    args = _make_args()
    loss_dict = {"kl_loss": kl, "ppo_kl": 100.0}

    update_adaptive_kl(args, loss_dict)

    assert args.kl_loss_coef == pytest.approx(expected_coef)
    assert loss_dict == {"kl_loss": kl, "ppo_kl": 100.0, "kl_loss_coef": pytest.approx(expected_coef)}


@pytest.mark.parametrize(("kl", "expected_coef"), [(1.0, 0.51), (0.0, 0.49)])
def test_adaptive_clips_relative_error(kl, expected_coef):
    args = _make_args()

    update_adaptive_kl(args, {"kl_loss": kl})

    assert args.kl_loss_coef == pytest.approx(expected_coef)


def test_adaptive_uses_configured_target():
    args = _make_args(kl_target=0.2)

    update_adaptive_kl(args, {"kl_loss": 0.22})

    assert args.kl_loss_coef == pytest.approx(0.505)


@pytest.mark.parametrize(
    ("horizon", "steps", "expected_coef"),
    [(10.0, 1, 0.51), (100.0, 1, 0.501), (10.0, 2, 0.52), (0.5, 1, 0.7)],
)
def test_adaptive_scales_by_steps_over_horizon(horizon, steps, expected_coef):
    args = _make_args(kl_horizon=horizon, kl_ctrl_steps=steps)

    update_adaptive_kl(args, {"kl_loss": 0.2})

    assert args.kl_loss_coef == pytest.approx(expected_coef)


def test_adaptive_uses_defaults_for_missing_tuning_fields():
    args = Namespace(kl_ctrl="adaptive", kl_loss_coef=0.5)

    update_adaptive_kl(args, {"kl_loss": 0.2})

    assert args.kl_loss_coef == pytest.approx(0.50001)


def test_missing_metric_leaves_state_alone():
    args = _make_args()
    before = vars(args).copy()

    update_adaptive_kl(args, {"ppo_kl": 1.0})

    assert vars(args) == before


def test_adaptive_coefficient_never_negative():
    args = _make_args(kl_horizon=1.0, kl_ctrl_steps=10)

    update_adaptive_kl(args, {"kl_loss": 0.0})

    assert args.kl_loss_coef == 0.0


def test_zero_coefficient_remains_zero():
    args = _make_args(kl_loss_coef=0.0)

    update_adaptive_kl(args, {"kl_loss": 1.0})

    assert args.kl_loss_coef == 0.0


def test_repeated_updates_use_current_coefficient():
    args = _make_args()

    update_adaptive_kl(args, {"kl_loss": 0.2})
    update_adaptive_kl(args, {"kl_loss": 0.0})

    assert args.kl_loss_coef == pytest.approx(0.4998)


def test_update_reports_the_next_coefficient_in_the_metrics():
    args = _make_args()
    loss_dict = {"kl_loss": 0.2, "pg_loss": 1.0}

    update_adaptive_kl(args, loss_dict)

    assert loss_dict == {"kl_loss": 0.2, "pg_loss": 1.0, "kl_loss_coef": pytest.approx(0.51)}
