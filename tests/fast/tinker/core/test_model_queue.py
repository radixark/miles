"""Model queues order commands and hold barriers until all preceding datums finish."""

from tests.fast.tinker.harness import command, datum, fb_payload

from miles.tinker.core.model_queue import ModelRequestQueue


def _model_queue() -> ModelRequestQueue:
    return ModelRequestQueue("model", "tenant", slot=0)


def test_out_of_order_arrival_queues_in_seq_order():
    model_queue = _model_queue()
    model_queue.submit(command("model", 2, "forward_backward", fb_payload("model", 2, [datum()]), arrival=1))
    assert not model_queue.queue, "seq 2 must wait for seq 1"
    model_queue.submit(command("model", 1, "forward_backward", fb_payload("model", 1, [datum()]), arrival=2))
    assert [pending.command.seq_id for pending in model_queue.queue] == [1, 2]


def test_a_gap_holds_later_commands():
    model_queue = _model_queue()
    model_queue.submit(command("model", 1, "forward_backward", fb_payload("model", 1, [datum()]), arrival=1))
    model_queue.submit(command("model", 3, "forward_backward", fb_payload("model", 3, [datum()]), arrival=2))
    assert [pending.command.seq_id for pending in model_queue.queue] == [1]
    model_queue.submit(command("model", 2, "forward_backward", fb_payload("model", 2, [datum()]), arrival=3))
    assert [pending.command.seq_id for pending in model_queue.queue] == [1, 2, 3]


def test_batch_runs_alternate_with_barriers():
    model_queue = _model_queue()
    model_queue.submit(command("model", 1, "forward_backward", fb_payload("model", 1, [datum()]), arrival=1))
    model_queue.submit(command("model", 2, "forward_backward", fb_payload("model", 2, [datum()]), arrival=2))
    model_queue.submit(
        command("model", 3, "optim_step", {"model_id": "model", "seq_id": 3, "adam_params": {}}, arrival=3)
    )
    model_queue.submit(command("model", 4, "forward_backward", fb_payload("model", 4, [datum()]), arrival=4))

    assert [pending.command.seq_id for pending in model_queue.open_batch_run()] == [1, 2]
    assert model_queue.ready_barrier() is None, "the barrier must wait for its batch run"

    for pending in list(model_queue.open_batch_run()):
        model_queue.finish(pending)
    assert model_queue.ready_barrier().command.seq_id == 3
    assert not model_queue.open_batch_run(), "the next batch run opens only after the barrier"

    model_queue.finish(model_queue.ready_barrier())
    assert [pending.command.seq_id for pending in model_queue.open_batch_run()] == [4]
