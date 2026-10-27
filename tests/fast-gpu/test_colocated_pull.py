from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["weight-update"], hardware=["hopper", "blackwell"], num_gpus=1
)

import pytest
import torch
import torch.multiprocessing as mp
import zmq

from miles.backends.training_utils.weight_update.protocols.transports.local_pull import (
    LocalPullTransport,
    get_device_uuid,
)
from miles.backends.training_utils.weight_update.protocols.transports.p2p_transport import P2PTransport
from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import (
    RemoteShard,
    RemoteWeightLocation,
)

_BUFFER_NBYTES = 1 << 20
# name -> (shape, dtype) the trainer's replica lays the param out in
_PARAMS = {"a.weight": ((1000,), torch.float32), "b.weight": ((3, 5), torch.bfloat16)}


def _run_engine_rank(socket_path, engine_param_shapes, results):
    """The engine side, as the sglang scheduler runs it: pull into this process's own params and report."""
    from sglang.srt.weight_sync.colocated_param_bytes import pull_param_bytes

    torch.cuda.set_device(0)
    params = {
        name: torch.zeros(engine_param_shapes[name], dtype=dtype, device="cuda")
        for name, (_, dtype) in _PARAMS.items()
    }
    sock = zmq.Context.instance().socket(zmq.PAIR)
    sock.setsockopt(zmq.RCVTIMEO, 60_000)
    sock.connect(socket_path)
    error = pull_param_bytes(
        sock=sock,
        device_uuid=get_device_uuid(),
        params_by_role={"target": params},
        published_nbytes_by_role={
            "target": {name: param.numel() * param.element_size() for name, param in params.items()}
        },
    )
    sock.close()
    # plain bytes: a tensor would travel as shared memory that dies with this process
    results.put(
        (error, {name: param.reshape(-1).view(torch.uint8).cpu().numpy().tobytes() for name, param in params.items()})
    )


def _sync_once(engine_param_shapes):
    """One sync between a trainer process's LocalPullTransport, written through P2PTransport as the protocol does,
    and an engine rank process on the same GPU."""
    torch.cuda.set_device(0)
    local_pull = LocalPullTransport(timeout_s=60)
    transport = P2PTransport(local_pull)
    buffer = torch.zeros(_BUFFER_NBYTES, dtype=torch.uint8, device="cuda")
    transport.register_memory(buffer)
    tensors_by_name, offset = {}, 0
    for name, (shape, dtype) in _PARAMS.items():
        nbytes = torch.Size(shape).numel() * dtype.itemsize
        tensors_by_name[name] = buffer[offset : offset + nbytes]
        offset += (nbytes + 255) // 256 * 256
    shard = RemoteShard(
        cell_id="cell-0",
        rollout_engine_ind=0,
        rollout_engine_rank=0,
        runner_role="target",
        transport_peer=local_pull.socket_path,
        weight_locations_by_name={
            name: RemoteWeightLocation(address=0, numel=tensor.numel(), element_size=1)
            for name, tensor in tensors_by_name.items()
        },
    )
    context = mp.get_context("spawn")
    results = context.Queue()
    engine = context.Process(target=_run_engine_rank, args=(local_pull.socket_path, engine_param_shapes, results))
    engine.start()

    local_pull.begin_session(
        expected_names_by_role={"target": frozenset(_PARAMS)}, stop_waiting=lambda: not engine.is_alive()
    )
    new_bytes_by_name = {
        name: torch.randint(0, 256, tensor.shape, dtype=torch.uint8, device="cuda")
        for name, tensor in tensors_by_name.items()
    }
    # the fill stays queued behind the sleep while the write starts; the engine process copies on its own stream,
    # so it reads the filled bytes only if the write waits for this stream's work
    torch.cuda._sleep(2_000_000_000)
    for name, tensor in tensors_by_name.items():
        tensor.copy_(new_bytes_by_name[name])
    (write,) = transport.write([shard], tensors_by_name, run_write=lambda shard, write: write())
    write_error = write.exception(timeout=60)
    with pytest.raises(RuntimeError, match="may still read it"):
        transport.unregister_memory(buffer)
    local_pull.end_session()
    # the engine closed its handles, so the buffer may go even after a failed write
    transport.unregister_memory(buffer)
    engine_error, engine_bytes = results.get(timeout=60)
    engine.join(timeout=60)
    return (
        write_error,
        engine_error,
        engine_bytes,
        {name: tensor.cpu().numpy().tobytes() for name, tensor in tensors_by_name.items()},
    )


def test_the_engine_rank_holds_every_byte_the_trainer_wrote():
    """The engine's post-load pass reads exactly the bytes the trainer's replica laid out, at their own params,
    even when the trainer's GPU is still filling them as the write starts."""
    write_error, engine_error, engine_bytes, sent_bytes = _sync_once(
        {name: shape for name, (shape, _) in _PARAMS.items()}
    )

    assert write_error is None and engine_error is None
    for name, sent in sent_bytes.items():
        assert engine_bytes[name] == sent, f"{name} differs from what the trainer wrote"


def test_an_engine_side_copy_error_fails_the_write_and_the_sync_still_closes():
    """A layout mismatch must fail the trainer's update rather than corrupt a neighbor, and the engine must still
    release the trainer's buffers so neither side hangs."""
    write_error, engine_error, _, _ = _sync_once({"a.weight": (1000,), "b.weight": (3, 4)})

    assert isinstance(write_error, RuntimeError)
    assert "b.weight: 30 bytes sent, 24 published" in str(write_error)
    assert "b.weight" in engine_error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
