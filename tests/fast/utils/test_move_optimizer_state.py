import torch

from miles.utils.memory_utils import move_optimizer_state


def _model_with_partial_optimizer_state():
    """An optimizer whose state covers only one of its two parameters."""
    model = torch.nn.Linear(4, 4, bias=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()  # skips bias, which has no grad
    return model, optimizer


def test_moves_state_tensors(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    model, optimizer = _model_with_partial_optimizer_state()
    assert optimizer.state[model.weight]["exp_avg"].device.type == "cpu"

    move_optimizer_state([optimizer], "cpu")

    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"


def test_does_not_grow_state_for_parameters_that_have_none(monkeypatch):
    # optimizer.state is a defaultdict, so indexing it per parameter -- the shape
    # this helper replaced -- inserts an empty entry for every stateless parameter.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    model, optimizer = _model_with_partial_optimizer_state()
    assert model.bias not in optimizer.state
    before = len(optimizer.state)

    move_optimizer_state([optimizer], "cpu")

    assert len(optimizer.state) == before
    assert model.bias not in optimizer.state


def test_accepts_several_optimizers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    first = _model_with_partial_optimizer_state()[1]
    second = _model_with_partial_optimizer_state()[1]

    move_optimizer_state([first, second], "cpu")

    for optimizer in (first, second):
        assert any(isinstance(v, torch.Tensor) for state in optimizer.state.values() for v in state.values())


def test_empty_iterable_is_a_no_op(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: calls.append(1))

    move_optimizer_state([], "cpu")

    assert calls == [1]
