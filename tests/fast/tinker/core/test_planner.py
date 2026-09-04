"""The planner must pack compatible ready datums across streams, split on the
token budget, and serve strictly by arrival."""

from tests.fast.tinker.harness import ADAM, command, fb_payload, row

from miles.tinker.core.planner import BarrierUnit, Planner, WorkUnit
from miles.tinker.core.stream import ModelStream


def _planner_with_streams(n: int, budget: int = 1_000_000) -> tuple[Planner, list[ModelStream]]:
    planner = Planner(budget)
    streams = [ModelStream(f"model-{i}", f"tenant-{i}", slot=i) for i in range(n)]
    for stream in streams:
        planner.add_stream(stream)
    return planner, streams


def _submit_fb(stream: ModelStream, seq_id: int, arrival: int, rows: list[dict], loss_fn="cross_entropy"):
    stream.submit(
        command(
            stream.model_id, seq_id, "forward_backward", fb_payload(stream.model_id, seq_id, rows, loss_fn), arrival
        )
    )


def _submit_optim(stream: ModelStream, seq_id: int, arrival: int):
    payload = {"model_id": stream.model_id, "seq_id": seq_id, "adam_params": dict(ADAM)}
    stream.submit(command(stream.model_id, seq_id, "optim_step", payload, arrival))


def test_same_loss_class_rows_pack_across_streams():
    planner, (a, b) = _planner_with_streams(2)
    _submit_fb(a, 1, arrival=1, rows=[row(), row()])
    _submit_fb(b, 1, arrival=2, rows=[row()])

    unit = planner.next_unit()
    assert isinstance(unit, WorkUnit)
    assert len(unit.rows) == 3
    assert {ref.stream.model_id for ref in unit.rows} == {"model-0", "model-1"}


def test_different_loss_config_does_not_pack():
    planner, (a, b) = _planner_with_streams(2)
    _submit_fb(a, 1, arrival=1, rows=[row()], loss_fn="cross_entropy")
    _submit_fb(b, 1, arrival=2, rows=[row()], loss_fn="ppo")

    unit = planner.next_unit()
    assert [ref.stream.model_id for ref in unit.rows] == ["model-0"]


def test_the_token_budget_splits_a_large_request():
    planner, (a,) = _planner_with_streams(1, budget=10)
    _submit_fb(a, 1, arrival=1, rows=[row(4), row(4), row(4)])  # 5 tokens each with the appended target

    first, second = planner.next_unit(), planner.next_unit()
    assert [len(first.rows), len(second.rows)] == [2, 1]


def test_an_oversized_datum_still_ships_alone():
    planner, (a,) = _planner_with_streams(1, budget=2)
    _submit_fb(a, 1, arrival=1, rows=[row(10)])

    assert len(planner.next_unit().rows) == 1


def test_the_oldest_arrival_wins():
    planner, (a, b) = _planner_with_streams(2)
    _submit_optim(a, 1, arrival=1)
    _submit_fb(b, 1, arrival=2, rows=[row()])

    assert isinstance(planner.next_unit(), BarrierUnit)


def test_optim_barriers_merge_and_save_does_not():
    planner, (a, b, c) = _planner_with_streams(3)
    _submit_optim(a, 1, arrival=1)
    _submit_optim(b, 1, arrival=2)
    c.submit(command("model-2", 1, "save_state", {"model_id": "model-2", "seq_id": 1, "name": "x"}, arrival=3))

    merged = planner.next_unit()
    assert merged.kind == "optim_step"
    assert {stream.model_id for stream, _ in merged.entries} == {"model-0", "model-1"}

    for stream, pending in merged.entries:
        stream.finish(pending)
    save = planner.next_unit()
    assert save.kind == "save_state"
    assert len(save.entries) == 1


def test_issued_rows_are_not_reissued():
    planner, (a,) = _planner_with_streams(1)
    _submit_fb(a, 1, arrival=1, rows=[row()])

    assert len(planner.next_unit().rows) == 1
    assert planner.next_unit() is None
