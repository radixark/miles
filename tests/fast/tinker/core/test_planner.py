"""The planner selects the oldest ready seed and packs compatible work within its budget."""

from tests.fast.tinker.harness import ADAM, command, datum, fb_payload

from miles.tinker.core.model_queue import ModelRequestQueue
from miles.tinker.core.planner import BarrierUnit, BatchUnit, Planner


def _planner_with_model_queues(n: int, budget: int = 1_000_000) -> tuple[Planner, list[ModelRequestQueue]]:
    planner = Planner(budget)
    model_queues = [ModelRequestQueue(f"model-{i}", f"tenant-{i}", slot=i) for i in range(n)]
    for model_queue in model_queues:
        planner.add_model_queue(model_queue)
    return planner, model_queues


def _submit_fb(model_queue: ModelRequestQueue, seq_id: int, arrival: int, datums: list[dict], loss_fn="cross_entropy"):
    model_queue.submit(
        command(
            model_queue.model_id,
            seq_id,
            "forward_backward",
            fb_payload(model_queue.model_id, seq_id, datums, loss_fn),
            arrival,
        )
    )


def _submit_optim(model_queue: ModelRequestQueue, seq_id: int, arrival: int):
    payload = {"model_id": model_queue.model_id, "seq_id": seq_id, "adam_params": dict(ADAM)}
    model_queue.submit(command(model_queue.model_id, seq_id, "optim_step", payload, arrival))


def test_same_pack_key_datums_pack_across_model_queues():
    planner, (a, b) = _planner_with_model_queues(2)
    _submit_fb(a, 1, arrival=1, datums=[datum(), datum()])
    _submit_fb(b, 1, arrival=2, datums=[datum()])

    unit = planner.next_to_run()
    assert isinstance(unit, BatchUnit)
    assert len(unit.datums) == 3
    assert {ref.model_queue.model_id for ref in unit.datums} == {"model-0", "model-1"}


def test_different_loss_functions_do_not_pack():
    planner, (a, b) = _planner_with_model_queues(2)
    _submit_fb(a, 1, arrival=1, datums=[datum()], loss_fn="cross_entropy")
    _submit_fb(b, 1, arrival=2, datums=[datum()], loss_fn="ppo")

    unit = planner.next_to_run()
    assert [ref.model_queue.model_id for ref in unit.datums] == ["model-0"]


def test_the_token_budget_splits_a_large_request():
    planner, (a,) = _planner_with_model_queues(1, budget=10)
    _submit_fb(a, 1, arrival=1, datums=[datum(4), datum(4), datum(4)])  # 5 tokens each with the appended target

    first, second = planner.next_to_run(), planner.next_to_run()
    assert [len(first.datums), len(second.datums)] == [2, 1]


def test_an_oversized_datum_still_ships_alone():
    planner, (a,) = _planner_with_model_queues(1, budget=2)
    _submit_fb(a, 1, arrival=1, datums=[datum(10)])

    assert len(planner.next_to_run().datums) == 1


def test_the_oldest_arrival_wins():
    planner, (a, b) = _planner_with_model_queues(2)
    _submit_optim(a, 1, arrival=1)
    _submit_fb(b, 1, arrival=2, datums=[datum()])

    assert isinstance(planner.next_to_run(), BarrierUnit)


def test_optim_barriers_merge_and_save_does_not():
    planner, (a, b, c) = _planner_with_model_queues(3)
    _submit_optim(a, 1, arrival=1)
    _submit_optim(b, 1, arrival=2)
    c.submit(command("model-2", 1, "save_state", {"model_id": "model-2", "seq_id": 1, "name": "x"}, arrival=3))

    merged = planner.next_to_run()
    assert merged.op == "optim_step"
    assert {model_queue.model_id for model_queue, _ in merged.entries} == {"model-0", "model-1"}

    for model_queue, pending in merged.entries:
        model_queue.finish(pending)
    save = planner.next_to_run()
    assert save.op == "save_state"
    assert len(save.entries) == 1


def test_issued_datums_are_not_reissued():
    planner, (a,) = _planner_with_model_queues(1)
    _submit_fb(a, 1, arrival=1, datums=[datum()])

    assert len(planner.next_to_run().datums) == 1
    assert planner.next_to_run() is None


def test_forward_only_packs_only_within_one_loss():
    planner = Planner(batch_token_budget=1 << 20)
    for index, (model, loss_fn) in enumerate([("m1", "cross_entropy"), ("m2", "dro")], start=1):
        model_queue = ModelRequestQueue(model, "tenant", slot=index)
        planner.add_model_queue(model_queue)
        model_queue.submit(
            command(
                model,
                1,
                "forward_only",
                {"datums": [{"tokens": [1, 2], "target_len": 1}], "loss_fn": loss_fn, "loss_fn_config": {}},
                arrival=index,
            )
        )
    first = planner.next_to_run()
    assert [ref.request.command.payload["loss_fn"] for ref in first.datums] == [
        "cross_entropy"
    ], "a DRO request must not execute under another request's loss"
