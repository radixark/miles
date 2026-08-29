from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-fast")

from types import SimpleNamespace

from miles.ray.rollout.router_manager import _maybe_enable_router_dp_aware


def _make(dp_size: int | None, dp_aware: bool = False):
    args = SimpleNamespace(sglang_dp_size=dp_size)
    router_args = SimpleNamespace(dp_aware=dp_aware)
    return args, router_args


def test_auto_enables_when_dp_attention_on(monkeypatch):
    monkeypatch.delenv("MILES_DISABLE_AUTO_DP_AWARE", raising=False)
    args, router_args = _make(dp_size=4)
    _maybe_enable_router_dp_aware(args, router_args)
    assert router_args.dp_aware is True


def test_noop_when_dp_size_one(monkeypatch):
    monkeypatch.delenv("MILES_DISABLE_AUTO_DP_AWARE", raising=False)
    args, router_args = _make(dp_size=1)
    _maybe_enable_router_dp_aware(args, router_args)
    assert router_args.dp_aware is False


def test_noop_when_dp_size_none(monkeypatch):
    monkeypatch.delenv("MILES_DISABLE_AUTO_DP_AWARE", raising=False)
    args, router_args = _make(dp_size=None)
    _maybe_enable_router_dp_aware(args, router_args)
    assert router_args.dp_aware is False


def test_respects_explicit_dp_aware(monkeypatch):
    monkeypatch.delenv("MILES_DISABLE_AUTO_DP_AWARE", raising=False)
    args, router_args = _make(dp_size=4, dp_aware=True)
    _maybe_enable_router_dp_aware(args, router_args)
    assert router_args.dp_aware is True


def test_opt_out_warns_and_keeps_disabled(monkeypatch, caplog):
    monkeypatch.setenv("MILES_DISABLE_AUTO_DP_AWARE", "1")
    args, router_args = _make(dp_size=4)
    with caplog.at_level("WARNING"):
        _maybe_enable_router_dp_aware(args, router_args)
    assert router_args.dp_aware is False
    assert "MILES_DISABLE_AUTO_DP_AWARE" in caplog.text
