from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.ray.train_batch_admission import (
    RayTrainerAdmissionAdapter,
    TrainBatchPublication,
    TrainerAdmissionReceipt,
    TrainerAdmissionStatus,
    TrainerCellCohort,
    TrainerCohort,
    TrainerCohortChangedError,
    TrainerRankReceipt,
    data_ref_ids,
    required_trainer_roles,
    validate_data_parallel_coverage,
)
from miles.utils.ray_utils import Box


class HexRef:
    def __init__(self, value: str) -> None:
        self._value = value

    def hex(self) -> str:
        return self._value


def make_args(*, use_critic: bool, num_critic_only_steps: int = 0) -> Namespace:
    return Namespace(use_critic=use_critic, num_critic_only_steps=num_critic_only_steps)


def test_data_ref_ids_are_stable_and_ordered() -> None:
    first = Box(HexRef("a"))
    second = Box(HexRef("b"))

    assert data_ref_ids([first, second]) == data_ref_ids([Box(HexRef("a")), Box(HexRef("b"))])
    assert data_ref_ids([first, second]) != data_ref_ids([second, first])


def test_data_ref_ids_reject_empty_or_non_box_refs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        data_ref_ids([])
    with pytest.raises(TypeError, match="Box"):
        data_ref_ids("not-a-box")


@pytest.mark.parametrize(
    ("args", "rollout_id", "expected"),
    [
        (make_args(use_critic=False), 4, frozenset({"actor"})),
        (make_args(use_critic=True, num_critic_only_steps=2), 1, frozenset({"critic"})),
        (make_args(use_critic=True, num_critic_only_steps=2), 2, frozenset({"actor", "critic"})),
    ],
)
def test_required_trainer_roles(args: Namespace, rollout_id: int, expected: frozenset[str]) -> None:
    assert required_trainer_roles(args, rollout_id) == expected


def test_publication_is_immutable() -> None:
    publication = TrainBatchPublication(
        manager_incarnation="manager-a",
        admission_id=3,
        rollout_id=7,
        data_ref_ids=("ref",),
        required_roles=frozenset({"actor"}),
    )

    assert publication.required_roles == frozenset({"actor"})
    with pytest.raises(AttributeError):
        publication.admission_id = 4  # type: ignore[misc]


def test_status_values_include_fail_closed_states() -> None:
    assert TrainerAdmissionStatus.PENDING.value == "pending"
    assert TrainerAdmissionStatus.COMMIT_FAILED.value == "commit_failed"
    assert TrainerAdmissionStatus.ROLLBACK_FAILED.value == "rollback_failed"


def make_publication(ref: Box, *, role: str = "actor") -> TrainBatchPublication:
    return TrainBatchPublication(
        manager_incarnation="manager-a",
        admission_id=3,
        rollout_id=7,
        data_ref_ids=data_ref_ids(ref),
        required_roles=frozenset({role}),
    )


async def test_v1_admits_every_rank_into_one_structured_cohort() -> None:
    from miles.ray.actor_group import RayTrainGroup

    ref = Box(HexRef("published"))
    publication = make_publication(ref)
    group = object.__new__(RayTrainGroup)
    group.role = "actor"
    group._actor_handles = [object(), object()]

    async def broadcast(method_name, received_publication, data_ref):
        assert method_name == "admit_train_batch"
        assert received_publication == publication
        assert data_ref == ref
        return [
            TrainerRankReceipt(publication=publication, rank=0),
            TrainerRankReceipt(publication=publication, rank=1),
        ]

    group._broadcast = broadcast

    receipt = await group.admit_train_batch(7, {"data_ref": ref, "trainer_admission": publication})

    assert receipt == TrainerAdmissionReceipt(
        publication=publication,
        role="actor",
        cohort=TrainerCohort(quorum_id=None, cells=(TrainerCellCohort(cell_index=0, ranks=(0, 1)),)),
    )


@pytest.mark.parametrize(
    "responses",
    [
        [TrainerRankReceipt(publication=make_publication(Box(HexRef("published"))), rank=0)],
        [
            TrainerRankReceipt(publication=make_publication(Box(HexRef("published"))), rank=0),
            TrainerRankReceipt(publication=make_publication(Box(HexRef("published"))), rank=0),
        ],
        [
            TrainerRankReceipt(
                publication=TrainBatchPublication(
                    manager_incarnation="foreign",
                    admission_id=3,
                    rollout_id=7,
                    data_ref_ids=data_ref_ids(Box(HexRef("published"))),
                    required_roles=frozenset({"actor"}),
                ),
                rank=0,
            ),
            TrainerRankReceipt(
                publication=TrainBatchPublication(
                    manager_incarnation="foreign",
                    admission_id=3,
                    rollout_id=7,
                    data_ref_ids=data_ref_ids(Box(HexRef("published"))),
                    required_roles=frozenset({"actor"}),
                ),
                rank=1,
            ),
        ],
    ],
)
async def test_v1_rejects_missing_duplicate_or_foreign_rank_responses(responses) -> None:
    from miles.ray.actor_group import RayTrainGroup

    ref = Box(HexRef("published"))
    publication = make_publication(ref)
    group = object.__new__(RayTrainGroup)
    group.role = "actor"
    group._actor_handles = [object(), object()]

    async def broadcast(*args, **kwargs):
        return responses

    group._broadcast = broadcast

    with pytest.raises((ValueError, RuntimeError)):
        await group.admit_train_batch(7, {"data_ref": ref, "trainer_admission": publication})


class AliveCell:
    def __init__(self, cell_index: int, ranks: tuple[int, ...]) -> None:
        self.cell_index = cell_index
        self.is_alive = True
        self._handles = [object() for _ in ranks]

    def _get_actor_handles(self):
        return self._handles

    def _snapshot_actor_handles(self):
        return tuple(self._handles)


async def test_v2_snapshots_quorum_cells_and_ranks() -> None:
    from miles.ray.train.group import RayTrainGroup

    ref = Box(HexRef("published"))
    publication = make_publication(ref)
    group = object.__new__(RayTrainGroup)
    group.role = "actor"
    group._indep_dp_quorum_id = 5
    group._cells = [AliveCell(0, (0, 1)), AliveCell(1, (0, 1))]

    async def refresh_cells(*, rollout_id):
        assert rollout_id == 7

    async def execute_all(fn_name, received_publication, data_ref, *, kill_on_failure):
        assert fn_name == "admit_train_batch"
        assert received_publication == publication
        assert data_ref == ref
        assert kill_on_failure is False
        return group._cells, [
            [TrainerRankReceipt(publication=publication, rank=0), TrainerRankReceipt(publication=publication, rank=1)],
            [TrainerRankReceipt(publication=publication, rank=0), TrainerRankReceipt(publication=publication, rank=1)],
        ]

    group._refresh_cells = refresh_cells
    group._execute_all_alive_and_catch = execute_all

    receipt = await group.admit_train_batch(7, {"data_ref": ref, "trainer_admission": publication})

    assert receipt.cohort == TrainerCohort(
        quorum_id=5,
        cells=(TrainerCellCohort(cell_index=0, ranks=(0, 1)), TrainerCellCohort(cell_index=1, ranks=(0, 1))),
    )


async def test_v2_admission_probe_never_kills_cells_on_failure() -> None:
    """Admission only proves readability, so a failing rank must stay alive for replay."""
    from miles.ray.train.group import RayTrainGroup

    ref = Box(HexRef("published"))
    publication = make_publication(ref)
    group = object.__new__(RayTrainGroup)
    group.role = "actor"
    group._indep_dp_quorum_id = 5
    group._cells = [AliveCell(0, (0,))]
    execute_kwargs: list[dict[str, object]] = []

    async def refresh_cells(**kwargs):
        pass

    async def execute_all(fn_name, *args, **kwargs):
        assert fn_name == "admit_train_batch"
        execute_kwargs.append(dict(kwargs))
        return group._cells, [[TrainerRankReceipt(publication=publication, rank=0)]]

    group._refresh_cells = refresh_cells
    group._execute_all_alive_and_catch = execute_all

    await group.admit_train_batch(7, {"data_ref": ref, "trainer_admission": publication})

    assert execute_kwargs == [{"kill_on_failure": False}]


@pytest.mark.parametrize("drift", ["quorum", "cells"])
async def test_v2_rejects_quorum_or_cell_drift(drift: str) -> None:
    from miles.ray.train.group import RayTrainGroup

    ref = Box(HexRef("published"))
    publication = make_publication(ref)
    group = object.__new__(RayTrainGroup)
    group.role = "actor"
    group._indep_dp_quorum_id = 5
    group._cells = [AliveCell(0, (0,))]

    async def refresh_cells(*, rollout_id):
        pass

    async def execute_all(*args, **kwargs):
        if drift == "quorum":
            group._indep_dp_quorum_id = 6
        if drift == "cells":
            group._cells = [AliveCell(1, (0,))]
        cells = group._cells
        return cells, []

    group._refresh_cells = refresh_cells
    group._execute_all_alive_and_catch = execute_all

    with pytest.raises(RuntimeError, match="cohort"):
        await group.admit_train_batch(7, {"data_ref": ref, "trainer_admission": publication})


async def test_v2_rejects_same_index_count_cell_replacement() -> None:
    from miles.ray.train.group import RayTrainGroup

    ref = Box(HexRef("published"))
    publication = make_publication(ref)
    original = AliveCell(0, (0,))
    replacement = AliveCell(0, (0,))
    group = object.__new__(RayTrainGroup)
    group.role = "actor"
    group._indep_dp_quorum_id = 5
    group._cells = [original]

    async def refresh_cells(**kwargs):
        pass

    async def execute_all(*args, **kwargs):
        group._cells = [replacement]
        return [original], [[TrainerRankReceipt(publication=publication, rank=0)]]

    group._refresh_cells = refresh_cells
    group._execute_all_alive_and_catch = execute_all

    with pytest.raises(RuntimeError, match="cohort"):
        await group.admit_train_batch(7, {"data_ref": ref, "trainer_admission": publication})


@pytest.mark.parametrize("ranks", [(0,), (0, 0)])
async def test_v2_rejects_partial_or_duplicate_rank_cohort(ranks: tuple[int, ...]) -> None:
    from miles.ray.train.group import RayTrainGroup

    ref = Box(HexRef("published"))
    publication = make_publication(ref)
    group = object.__new__(RayTrainGroup)
    group.role = "actor"
    group._indep_dp_quorum_id = 5
    group._cells = [AliveCell(0, (0, 1))]

    async def refresh_cells(**kwargs):
        pass

    group._refresh_cells = refresh_cells

    async def execute_all(*args, **kwargs):
        return group._cells, [[TrainerRankReceipt(publication=publication, rank=rank) for rank in ranks]]

    group._execute_all_alive_and_catch = execute_all

    with pytest.raises(RuntimeError):
        await group.admit_train_batch(7, {"data_ref": ref, "trainer_admission": publication})


def make_v1_group(responses: list[TrainerRankReceipt]):
    """Build a bare v1 trainer group whose ranks return ``responses``."""
    from miles.ray.actor_group import RayTrainGroup

    group = object.__new__(RayTrainGroup)
    group.role = "actor"
    group._actor_handles = [object() for _ in responses]

    async def broadcast(*args, **kwargs):
        return responses

    group._broadcast = broadcast
    return group


def make_v2_cell_group(cell_responses: list[list[TrainerRankReceipt]]):
    """Build a bare v2 trainer group whose cells return ``cell_responses`` in order."""
    from miles.ray.train.group import RayTrainGroup

    group = object.__new__(RayTrainGroup)
    group.role = "actor"
    group._indep_dp_quorum_id = 5
    group._cells = [
        AliveCell(cell_index, tuple(range(len(responses)))) for cell_index, responses in enumerate(cell_responses)
    ]

    async def refresh_cells(**kwargs):
        pass

    async def execute_all(*args, **kwargs):
        return group._cells, list(cell_responses)

    group._refresh_cells = refresh_cells
    group._execute_all_alive_and_catch = execute_all
    return group


def make_v2_group(responses: list[TrainerRankReceipt]):
    """Build a bare v2 trainer group whose single cell returns ``responses``."""
    return make_v2_cell_group([responses])


def make_cell_receipts(publication: TrainBatchPublication, layouts: list[tuple[int, int] | None]):
    """Return one rank receipt per entry of ``layouts`` for one cell of ``publication``."""
    return [
        TrainerRankReceipt(publication=publication, rank=rank, data_parallel=layout)
        for rank, layout in enumerate(layouts)
    ]


def make_shard_pack(layouts: list[tuple[int, int] | None], *, shards: int = 2):
    """Return a ``shards``-shard rollout data pack and the rank receipts reporting ``layouts``."""
    refs = [Box(HexRef(f"shard-{index}")) for index in range(shards)]
    publication = make_publication(refs)
    return {"data_ref": refs, "trainer_admission": publication}, make_cell_receipts(publication, layouts)


@pytest.mark.parametrize("make_group", [make_v1_group, make_v2_group])
async def test_group_rejects_a_published_shard_no_rank_read(make_group) -> None:
    pack, receipts = make_shard_pack([(0, 2), (0, 2)])

    with pytest.raises(RuntimeError, match="shard"):
        await make_group(receipts).admit_train_batch(7, pack)


@pytest.mark.parametrize("make_group", [make_v1_group, make_v2_group])
async def test_group_rejects_disagreeing_data_parallel_sizes(make_group) -> None:
    pack, receipts = make_shard_pack([(0, 2), (0, 3)])

    with pytest.raises(RuntimeError, match="from one cohort"):
        await make_group(receipts).admit_train_batch(7, pack)


@pytest.mark.parametrize("make_group", [make_v1_group, make_v2_group])
@pytest.mark.parametrize(
    ("layouts", "shards"),
    [([(0, 2), (1, 2)], 2), ([None, (1, 2)], 2), ([None, None], 2), ([(0, 1)], 1)],
)
async def test_group_admits_receipts_that_cover_every_shard(make_group, layouts, shards) -> None:
    pack, receipts = make_shard_pack(layouts, shards=shards)

    receipt = await make_group(receipts).admit_train_batch(7, pack)

    assert receipt.publication == pack["trainer_admission"]


async def test_v2_rejects_a_cell_whose_own_receipts_miss_a_shard() -> None:
    pack, _ = make_shard_pack([(0, 2), (1, 2)])
    publication = pack["trainer_admission"]
    group = make_v2_cell_group(
        [
            make_cell_receipts(publication, [(0, 2), (0, 2)]),
            make_cell_receipts(publication, [(1, 2), (1, 2)]),
        ]
    )

    with pytest.raises(RuntimeError, match="train batch shards"):
        await group.admit_train_batch(7, pack)


async def test_v2_admits_cells_that_each_cover_every_shard() -> None:
    pack, _ = make_shard_pack([(0, 2), (1, 2)])
    publication = pack["trainer_admission"]
    group = make_v2_cell_group(
        [
            make_cell_receipts(publication, [(0, 2), (1, 2)]),
            make_cell_receipts(publication, [(0, 2), (1, 2)]),
        ]
    )

    receipt = await group.admit_train_batch(7, pack)

    assert receipt.cohort.cells == (
        TrainerCellCohort(cell_index=0, ranks=(0, 1)),
        TrainerCellCohort(cell_index=1, ranks=(0, 1)),
    )


def test_coverage_rejects_a_data_parallel_size_the_publication_cannot_shard() -> None:
    publication = make_publication([Box(HexRef("shard-0")), Box(HexRef("shard-1"))])
    receipts = make_cell_receipts(publication, [(0, 3), (1, 3)])

    with pytest.raises(RuntimeError, match="data-parallel size of 3 against 2 published shards"):
        validate_data_parallel_coverage(receipts, publication)


async def test_v2_admitted_train_fails_fast_when_cell_changes_before_dispatch(monkeypatch) -> None:
    import miles.ray.train.group as train_group_mod
    from miles.ray.train.group import RayTrainGroup

    ref = Box(HexRef("published"))
    publication = make_publication(ref)
    receipt = TrainerAdmissionReceipt(
        publication=publication,
        role="actor",
        cohort=TrainerCohort(quorum_id=5, cells=(TrainerCellCohort(cell_index=0, ranks=(0,)),)),
    )
    admitted = AliveCell(0, (0,))
    replacement = AliveCell(0, (0,))
    group = object.__new__(RayTrainGroup)
    group.args = SimpleNamespace()
    group._cells = [admitted]
    group._indep_dp_quorum_id = 5
    group._train_batch_pins = {publication: (receipt, 5, (admitted,), (admitted._snapshot_actor_handles(),))}
    group._witness_allocator = None
    group._test_action_executor = SimpleNamespace(run_after_step=lambda **kwargs: None)
    attempts = []

    def validate(*args, **kwargs):
        group._cells = [replacement]
        return (admitted,), (admitted._snapshot_actor_handles(),)

    async def one_attempt(fn, **kwargs):
        attempts.append(0)
        await fn(0)

    group._validate_train_batch_pin = validate
    monkeypatch.setattr(train_group_mod.event_analyzer, "run_analysis_from_args", lambda args: None)
    monkeypatch.setattr(train_group_mod, "retry", one_attempt)

    with pytest.raises(TrainerCohortChangedError, match="changed"):
        await group.train(
            7,
            {"sample_indices": [0], "data_ref": ref, "trainer_admission": publication},
            admission_receipt=receipt,
        )

    assert attempts == [0]


async def test_v2_cell_rejects_replaced_handles_before_remote_train() -> None:
    from miles.ray.train.cell import RayTrainCell

    admitted_handle = object()
    replacement_handle = object()
    cell = object.__new__(RayTrainCell)
    cell.cell_index = 0
    cell._get_actor_handles = lambda: [replacement_handle]

    with pytest.raises(TrainerCohortChangedError, match="changed actor handles"):
        await cell.execute("train", expected_actor_handles=(admitted_handle,))


class GetResult:
    def __init__(self, value, events: list[str]) -> None:
        self.value = value
        self.events = events

    def __enter__(self):
        self.events.append("enter")
        return self.value

    def __exit__(self, exc_type, exc_value, traceback):
        self.events.append("exit")


async def test_train_actor_reads_exact_box_as_mapping_without_training(monkeypatch: pytest.MonkeyPatch) -> None:
    import miles.ray.train_actor as train_actor_mod
    from miles.ray.train_actor import TrainRayActor

    ref = Box(HexRef("published"))
    publication = make_publication(ref)
    events: list[str] = []
    monkeypatch.setattr(
        train_actor_mod.object_store,
        "get_instance",
        lambda: SimpleNamespace(get=lambda received_ref: GetResult({"input_ids": [1]}, events)),
    )
    actor = object.__new__(TrainRayActor)
    actor.args = SimpleNamespace(rank=4)

    receipt = actor.admit_train_batch(publication, ref)

    assert receipt == TrainerRankReceipt(publication=publication, rank=4)
    assert events == ["enter", "exit"]


@pytest.mark.parametrize("value", [RuntimeError("read failed"), ["not", "a", "mapping"]])
async def test_train_actor_rejects_read_failure_or_non_mapping(monkeypatch: pytest.MonkeyPatch, value) -> None:
    import miles.ray.train_actor as train_actor_mod
    from miles.ray.train_actor import TrainRayActor

    ref = Box(HexRef("published"))
    publication = make_publication(ref)

    def get(_ref):
        if isinstance(value, BaseException):
            raise value
        return GetResult(value, [])

    monkeypatch.setattr(train_actor_mod.object_store, "get_instance", lambda: SimpleNamespace(get=get))
    actor = object.__new__(TrainRayActor)
    actor.args = SimpleNamespace(rank=0)

    with pytest.raises((RuntimeError, ValueError)):
        actor.admit_train_batch(publication, ref)


async def test_train_actor_rejects_substituted_reference_before_store_read(monkeypatch: pytest.MonkeyPatch) -> None:
    import miles.ray.train_actor as train_actor_mod
    from miles.ray.train_actor import TrainRayActor

    published = Box(HexRef("published"))
    substituted = Box(HexRef("substituted"))
    publication = make_publication(published)
    reads: list[Box] = []

    def get(ref):
        reads.append(ref)
        return GetResult({"input_ids": [1]}, [])

    monkeypatch.setattr(train_actor_mod.object_store, "get_instance", lambda: SimpleNamespace(get=get))
    actor = object.__new__(TrainRayActor)
    actor.args = SimpleNamespace(rank=0)

    with pytest.raises(ValueError, match="published data reference"):
        actor.admit_train_batch(publication, substituted)
    assert reads == []


def make_sharded_actor(monkeypatch: pytest.MonkeyPatch, reads: list, *, data_parallel):
    """Build a bare TrainRayActor whose store reads are recorded in ``reads``."""
    import miles.ray.train_actor as train_actor_mod
    from miles.ray.train_actor import TrainRayActor

    def get(ref):
        reads.append(ref)
        return GetResult({"input_ids": [1]}, [])

    monkeypatch.setattr(train_actor_mod.object_store, "get_instance", lambda: SimpleNamespace(get=get))
    actor = object.__new__(TrainRayActor)
    actor.args = SimpleNamespace(rank=1)
    actor._admission_data_parallel = lambda: data_parallel
    return actor


async def test_train_actor_reads_only_its_own_data_parallel_shard(monkeypatch: pytest.MonkeyPatch) -> None:
    refs = [Box(HexRef("shard-0")), Box(HexRef("shard-1"))]
    publication = make_publication(refs)
    reads: list[Box] = []
    actor = make_sharded_actor(monkeypatch, reads, data_parallel=(1, 2))

    receipt = actor.admit_train_batch(publication, refs)

    assert receipt == TrainerRankReceipt(publication=publication, rank=1, data_parallel=(1, 2))
    assert reads == [refs[1]]


async def test_train_actor_rejects_shard_count_mismatch_before_store_read(monkeypatch: pytest.MonkeyPatch) -> None:
    refs = [Box(HexRef("shard-0")), Box(HexRef("shard-1")), Box(HexRef("shard-2"))]
    publication = make_publication(refs)
    reads: list[Box] = []
    actor = make_sharded_actor(monkeypatch, reads, data_parallel=(1, 2))

    with pytest.raises(ValueError, match="3 train batch shards for a data-parallel size of 2"):
        actor.admit_train_batch(publication, refs)
    assert reads == []


async def test_train_actor_reads_every_shard_when_data_parallel_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    refs = [Box(HexRef("shard-0")), Box(HexRef("shard-1"))]
    publication = make_publication(refs)
    reads: list[Box] = []
    actor = make_sharded_actor(monkeypatch, reads, data_parallel=None)

    receipt = actor.admit_train_batch(publication, refs)

    assert receipt == TrainerRankReceipt(publication=publication, rank=1)
    assert reads == refs


@pytest.fixture
def install_effective_dp():
    """Install a real ParallelState through the public setter, then restore the old one."""
    from miles.backends.training_utils import parallel as parallel_mod
    from miles.utils.ft_utils.process_group_utils import GroupInfo

    saved = parallel_mod._parallel_state

    def install(rank: int, size: int) -> None:
        trivial = GroupInfo(rank=0, size=1, group=None)
        parallel_mod.set_parallel_state(
            parallel_mod.ParallelState(
                intra_dp=GroupInfo(rank=rank, size=size, group=None),
                intra_dp_cp=trivial,
                cp=trivial,
                tp=trivial,
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
            )
        )

    try:
        yield install
    finally:
        parallel_mod._parallel_state = saved


def make_fsdp_actor(monkeypatch: pytest.MonkeyPatch, reads: list) -> object:
    """Build a bare FSDP trainer actor that records every store read in ``reads``."""
    import miles.ray.train_actor as train_actor_mod
    from miles.backends.fsdp_utils.actor import FSDPTrainRayActor

    def get(ref):
        reads.append(ref)
        return GetResult({"input_ids": [1]}, [])

    monkeypatch.setattr(train_actor_mod.object_store, "get_instance", lambda: SimpleNamespace(get=get))
    actor = object.__new__(FSDPTrainRayActor)
    actor.args = SimpleNamespace(rank=3)
    return actor


async def test_fsdp_actor_reads_the_shard_its_parallel_state_names(
    monkeypatch: pytest.MonkeyPatch, install_effective_dp
) -> None:
    from miles.backends.training_utils.parallel import get_parallel_state

    install_effective_dp(rank=1, size=2)
    refs = [Box(HexRef("shard-0")), Box(HexRef("shard-1"))]
    publication = make_publication(refs)
    reads: list[Box] = []
    actor = make_fsdp_actor(monkeypatch, reads)

    receipt = actor.admit_train_batch(publication, refs)

    assert receipt == TrainerRankReceipt(publication=publication, rank=3, data_parallel=(1, 2))
    assert reads == [refs[1]]
    # The same index training itself reads with, in training_utils/data.py.
    assert reads == [refs[get_parallel_state().effective_dp.rank]]


async def test_fsdp_actor_rejects_a_short_publication_before_any_store_read(
    monkeypatch: pytest.MonkeyPatch, install_effective_dp
) -> None:
    install_effective_dp(rank=1, size=2)
    refs = [Box(HexRef("shard-0"))]
    publication = make_publication(refs)
    reads: list[Box] = []
    actor = make_fsdp_actor(monkeypatch, reads)

    with pytest.raises(ValueError, match="1 train batch shards for a data-parallel size of 2"):
        actor.admit_train_batch(publication, refs)
    assert reads == []


async def test_fsdp_actor_reads_an_unsplit_publication_whole(
    monkeypatch: pytest.MonkeyPatch, install_effective_dp
) -> None:
    install_effective_dp(rank=1, size=2)
    ref = Box(HexRef("whole-batch"))
    publication = make_publication(ref)
    reads: list[Box] = []
    actor = make_fsdp_actor(monkeypatch, reads)

    receipt = actor.admit_train_batch(publication, ref)

    assert receipt == TrainerRankReceipt(publication=publication, rank=3)
    assert reads == [ref]


async def test_lost_commit_and_rollback_responses_reconcile_from_status() -> None:
    publication = TrainBatchPublication("manager-a", 3, 7, ("ref",), frozenset({"actor"}))
    receipt = TrainerAdmissionReceipt(
        publication=publication,
        role="actor",
        cohort=TrainerCohort(None, (TrainerCellCohort(0, (0,)),)),
    )

    class RemoteCall:
        def __init__(self, fn):
            self.remote = fn

    class Manager:
        def __init__(self):
            self.state = TrainerAdmissionStatus.PENDING
            self.commit_trainer_admission = RemoteCall(self.commit)
            self.rollback_trainer_admission = RemoteCall(self.rollback)
            self.get_trainer_admission_status = RemoteCall(self.status)

        async def commit(self, _publication, _receipts):
            self.state = TrainerAdmissionStatus.COMMITTED
            raise RuntimeError("lost commit response")

        async def rollback(self, _publication):
            self.state = TrainerAdmissionStatus.ROLLED_BACK
            raise RuntimeError("lost rollback response")

        async def status(self, _publication):
            return self.state

    manager = Manager()
    adapter = RayTrainerAdmissionAdapter(manager)
    assert await adapter.commit(publication, (receipt,)) is TrainerAdmissionStatus.COMMITTED
    manager.state = TrainerAdmissionStatus.PENDING
    assert await adapter.rollback(publication) is TrainerAdmissionStatus.ROLLED_BACK
