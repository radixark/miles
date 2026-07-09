import torch

from miles.utils.debug_utils.nan_scan import NanScanner


def _enabled_scanner(monkeypatch) -> NanScanner:
    monkeypatch.setenv("MILES_NAN_SCAN", "1")
    return NanScanner()


def test_disabled_is_noop(monkeypatch, capsys):
    monkeypatch.delenv("MILES_NAN_SCAN", raising=False)
    scanner = NanScanner()
    evaluated = []
    assert scanner.scan("x", torch.tensor([float("nan")])) is False
    assert scanner.scan("lazy", lambda: evaluated.append(1)) is False
    assert evaluated == [], "callable value must not be evaluated when disabled"
    scanner.scan("y", torch.ones(1))
    out = capsys.readouterr().out
    assert out.count("disabled") == 1, "one-time notice must print exactly once"


def test_tensor_scan(monkeypatch, capsys):
    scanner = _enabled_scanner(monkeypatch)
    assert scanner.scan("clean", torch.ones(4)) is False
    assert scanner.scan("dirty", torch.tensor([1.0, float("nan"), float("inf")])) is True
    out = capsys.readouterr().out
    assert "clean: shape=(4,) nan=0 inf=0" in out
    assert "dirty: shape=(3,) nan=1 inf=1" in out and "***NONFINITE***" in out


def test_quiet_prints_only_violations(monkeypatch, capsys):
    scanner = _enabled_scanner(monkeypatch)
    scanner.scan("clean", torch.ones(3), quiet=True)
    assert capsys.readouterr().out == ""
    scanner.scan("dirty", torch.tensor([float("nan")]), quiet=True)
    assert "***NONFINITE***" in capsys.readouterr().out


def test_tensor_list_aggregation(monkeypatch, capsys):
    scanner = _enabled_scanner(monkeypatch)
    found = scanner.scan("lst", [torch.ones(5), torch.tensor([float("nan")] * 3), torch.ones(2)])
    assert found is True
    out = capsys.readouterr().out
    assert "lst: tensors=3 numel=10 nan=3" in out
    assert "lst[1]:" in out, "offending element must be reported individually"


def test_dict_recursion_and_none(monkeypatch, capsys):
    scanner = _enabled_scanner(monkeypatch)
    assert scanner.scan("batch", {"a": torch.ones(2), "b": None, "c": [torch.ones(1)]}) is False
    out = capsys.readouterr().out
    assert "batch.a:" in out and "batch.c:" in out


def test_module_scan(monkeypatch, capsys):
    scanner = _enabled_scanner(monkeypatch)
    module = torch.nn.Linear(4, 4)
    module.weight.grad = torch.full_like(module.weight, float("nan"))
    assert scanner.scan("mod", module) is True
    out = capsys.readouterr().out
    assert "mod.weight.grad:" in out and "***NONFINITE***" in out
    assert "mod: tensors=3" in out and "nan=16" in out  # weight, bias, weight.grad


def test_lazy_callable_evaluated_when_enabled(monkeypatch):
    scanner = _enabled_scanner(monkeypatch)
    assert scanner.scan("lazy", lambda: [torch.ones(2), torch.tensor([float("inf")])]) is True


def test_fatal_raises(monkeypatch):
    scanner = _enabled_scanner(monkeypatch)
    try:
        scanner.scan("boom", torch.tensor([float("nan")]), fatal=True)
    except RuntimeError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("fatal=True must raise on non-finite values")


def test_scan_grad_fires_and_dedups(monkeypatch, capsys):
    scanner = _enabled_scanner(monkeypatch)
    for _ in range(2):
        t = torch.ones(3, requires_grad=True)
        y = t * torch.tensor([1.0, float("nan"), 1.0])
        scanner.scan_grad("y", y)
        (y * y).sum().backward()  # dL/dy = 2y contains NaN
    out = capsys.readouterr().out
    assert out.count("y.grad:") == 1, "once=True must dedup repeated violations"


def test_step_counter_in_output(monkeypatch, capsys):
    scanner = _enabled_scanner(monkeypatch)
    assert scanner.step() == 1
    assert scanner.step() == 2
    scanner.scan("x", torch.ones(1))
    assert "step=2 x:" in capsys.readouterr().out


def test_scan_grad_true_triages_forward_vs_backward(monkeypatch, capsys):
    scanner = _enabled_scanner(monkeypatch)

    # forward-born NaN: the value line fires
    bad = torch.tensor([1.0, float("nan")])
    assert scanner.scan("fwd", bad, grad=True) is True
    assert "fwd: shape=(2,)" in capsys.readouterr().out

    # backward-born NaN: the value line is clean, the .grad line fires
    t = torch.ones(3, requires_grad=True)
    y = t * torch.tensor([1.0, float("nan"), 1.0])
    z = torch.nan_to_num(y)  # forward value at z is clean
    assert scanner.scan("bwd", z, grad=True) is False
    (z * y).sum().backward()  # dL/dz = y contains NaN
    out = capsys.readouterr().out
    assert "bwd: shape=(3,) nan=0" in out
    assert "bwd.grad:" in out and "***NONFINITE***" in out

    # grad=True on a non-Tensor is a hook that would silently never fire; reject it
    try:
        scanner.scan("lst", [torch.ones(2)], grad=True)
    except TypeError:
        pass
    else:
        raise AssertionError("grad=True must require a Tensor")
