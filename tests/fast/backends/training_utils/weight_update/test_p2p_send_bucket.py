from concurrent.futures import Future
from types import SimpleNamespace

import torch

_REGISTRY = {"w": (0x1000, 4, 2)}


class _LoggingFuture(Future):
    def __init__(self, log: list[tuple], cell_id: str, engine_rank: int):
        super().__init__()
        self._log = log
        self._cell_id = cell_id
        self._engine_rank = engine_rank
        self.set_result(None)

    def result(self, timeout: float | None = None):
        self._log.append(("await", self._cell_id, self._engine_rank))
        return super().result(timeout)


class _RecordingCellUpdater:
    def __init__(self, log: list[tuple], cell_id: str):
        self._log = log
        self.cell_id = cell_id

    def submit_write(self, engine_rank: int, names: list[str], weight_memory_registry) -> Future:
        self._log.append(("submit", self.cell_id, engine_rank, tuple(names), weight_memory_registry))
        return _LoggingFuture(self._log, self.cell_id, engine_rank)


class _RecordingReplica:
    def __init__(self, log: list[tuple], engine_rank: int):
        self._log = log
        self._engine_rank = engine_rank

    def load_weights(self, named_tensors) -> None:
        self._log.append(("load", self._engine_rank, tuple(name for name, _tensor in named_tensors)))


def _send_one_bucket(p2p, *, engine_ranks: list[int], cell_ids: list[str]) -> list[tuple]:
    log: list[tuple] = []
    cell_updaters = {cell_id: _RecordingCellUpdater(log, cell_id) for cell_id in cell_ids}
    ready_hf_tensors = [("hf.w", torch.zeros(1))]
    protocol = SimpleNamespace(
        is_sender=True,
        _shared_param_mapper=object(),
        _shared_params_dict={},
        _weight_memory_registry=_REGISTRY,
        _model_param_stager=SimpleNamespace(
            get_transfer_ready_params=lambda *_args, **_kwargs: (["w"], ready_hf_tensors)
        ),
        _transfer_engine_meta_list=[
            p2p.TransferEngineMeta(
                engine_rank=engine_rank,
                model_replica=_RecordingReplica(log, engine_rank),
                cell_updaters=[cell_updaters[cell_id] for cell_id in cell_ids],
            )
            for engine_rank in engine_ranks
        ],
    )

    bucket = [("hf.w", torch.zeros(1))]
    p2p.UpdateWeightP2P.send_bucket(protocol, bucket)

    assert bucket == []
    return log


class TestSendBucketOrdering:
    """One CPU replica per engine rank is loaded into the shared buffers, so the writes must be ordered around it."""

    def test_a_non_last_engine_rank_is_fully_written_before_the_next_one_is_loaded(self, p2p_protocol) -> None:
        """Loading the next rank overwrites the shared buffers an unfinished write is still reading."""
        log = _send_one_bucket(p2p_protocol, engine_ranks=[0, 1, 2], cell_ids=["cell-a", "cell-b"])

        assert [entry[0] for entry in log] == [
            "load",
            "submit",
            "submit",
            "await",
            "await",
            "load",
            "submit",
            "submit",
            "await",
            "await",
            "load",
            "submit",
            "submit",
        ]

    def test_the_last_engine_rank_is_left_running_in_the_background(self, p2p_protocol) -> None:
        """Nothing overwrites the buffers after the last rank, so waiting there would stall the bucket stream."""
        log = _send_one_bucket(p2p_protocol, engine_ranks=[0, 1], cell_ids=["cell-a"])

        assert [entry for entry in log if entry[0] == "await"] == [("await", "cell-a", 0)]

    def test_every_cell_of_an_engine_rank_is_submitted_before_any_of_them_is_awaited(self, p2p_protocol) -> None:
        """Awaiting each cell in turn would serialize the cells instead of writing them in parallel."""
        log = _send_one_bucket(p2p_protocol, engine_ranks=[0, 1], cell_ids=["cell-a", "cell-b"])

        first_rank = [entry for entry in log if entry[0] in ("submit", "await") and entry[2] == 0]
        assert [(entry[0], entry[1]) for entry in first_rank] == [
            ("submit", "cell-a"),
            ("submit", "cell-b"),
            ("await", "cell-a"),
            ("await", "cell-b"),
        ]

    def test_each_write_is_addressed_to_the_engine_rank_whose_replica_was_just_loaded(self, p2p_protocol) -> None:
        """A write submitted for another rank would ship that rank's shard from this rank's buffers."""
        log = _send_one_bucket(p2p_protocol, engine_ranks=[3, 7], cell_ids=["cell-a"])

        assert [(entry[0], entry[1]) for entry in log if entry[0] == "load"] == [("load", 3), ("load", 7)]
        assert [(entry[1], entry[2]) for entry in log if entry[0] == "submit"] == [("cell-a", 3), ("cell-a", 7)]

    def test_every_write_names_the_ready_params_and_the_shared_source_registry(self, p2p_protocol) -> None:
        """The names and the registry decide which pinned buffers are read, so both must reach every cell."""
        log = _send_one_bucket(p2p_protocol, engine_ranks=[0], cell_ids=["cell-a", "cell-b"])

        assert [(entry[3], entry[4]) for entry in log if entry[0] == "submit"] == [(("w",), _REGISTRY)] * 2
