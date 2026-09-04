"""ModelStream must rectify HTTP arrival order into seq order and compile
batch-run|barrier structure."""

from tests.fast.tinker.harness import command, datum, fb_payload

from miles.tinker.core.stream import ModelStream


def _stream() -> ModelStream:
    return ModelStream("model", "tenant", slot=0)


def test_out_of_order_arrival_queues_in_seq_order():
    stream = _stream()
    stream.submit(command("model", 2, "forward_backward", fb_payload("model", 2, [datum()]), arrival=1))
    assert not stream.queue, "seq 2 must wait for seq 1"
    stream.submit(command("model", 1, "forward_backward", fb_payload("model", 1, [datum()]), arrival=2))
    assert [pending.command.seq_id for pending in stream.queue] == [1, 2]


def test_a_gap_holds_later_commands():
    stream = _stream()
    stream.submit(command("model", 1, "forward_backward", fb_payload("model", 1, [datum()]), arrival=1))
    stream.submit(command("model", 3, "forward_backward", fb_payload("model", 3, [datum()]), arrival=2))
    assert [pending.command.seq_id for pending in stream.queue] == [1]
    stream.submit(command("model", 2, "forward_backward", fb_payload("model", 2, [datum()]), arrival=3))
    assert [pending.command.seq_id for pending in stream.queue] == [1, 2, 3]


def test_batch_runs_alternate_with_barriers():
    stream = _stream()
    stream.submit(command("model", 1, "forward_backward", fb_payload("model", 1, [datum()]), arrival=1))
    stream.submit(command("model", 2, "forward_backward", fb_payload("model", 2, [datum()]), arrival=2))
    stream.submit(command("model", 3, "optim_step", {"model_id": "model", "seq_id": 3, "adam_params": {}}, arrival=3))
    stream.submit(command("model", 4, "forward_backward", fb_payload("model", 4, [datum()]), arrival=4))

    assert [pending.command.seq_id for pending in stream.open_batch_run()] == [1, 2]
    assert stream.ready_barrier() is None, "the barrier must wait for its batch run"

    for pending in list(stream.open_batch_run()):
        stream.finish(pending)
    assert stream.ready_barrier().command.seq_id == 3
    assert not stream.open_batch_run(), "the next batch run opens only after the barrier"

    stream.finish(stream.ready_barrier())
    assert [pending.command.seq_id for pending in stream.open_batch_run()] == [4]
