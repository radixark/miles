from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pydantic
import pytest
from tests.fast.utils.workers.reconcile.utils import FakeSource, replace_of, settle

from miles.utils.external_utils import colocate_pairing as pairing
from miles.utils.test_utils.clock import FakeClock
from miles.utils.workers.reconcile.loop import ReconcileLoop
from miles.utils.workers.reconcile.source_event import DeleteEvent, UpsertEvent

TRAINER_COMPONENT = "r-miles-run-trainer-actor"
ENGINE_COMPONENT = "r-miles-run-inference-engine"


def _layout(
    engine_cells: int, trainer_cells: int, pods_per_engine_cell: int = 1, pods_per_trainer_cell: int = 1
) -> pairing.PairingLayout:
    return pairing.PairingLayout(
        engine_cells=engine_cells,
        trainer_cells=trainer_cells,
        pods_per_engine_cell=pods_per_engine_cell,
        pods_per_trainer_cell=pods_per_trainer_cell,
    )


def _target(engine_cell_index: int, layout: pairing.PairingLayout, engine_pod_index: int = 0) -> str:
    return pairing.target_trainer_pod(
        engine_cell_index=engine_cell_index,
        engine_pod_index=engine_pod_index,
        layout=layout,
        trainer_component=TRAINER_COMPONENT,
    )


class TestPairingLayout:
    def test_refuses_a_pool_with_no_cells(self):
        """Zero cells is a values bug, and every later division would be by zero."""
        with pytest.raises(pydantic.ValidationError):
            _layout(engine_cells=0, trainer_cells=1)

    def test_refuses_a_cell_with_no_pods(self):
        """A cell is at least one pod, and the mapping divides by the engine cell width."""
        with pytest.raises(pydantic.ValidationError):
            _layout(engine_cells=1, trainer_cells=1, pods_per_engine_cell=0)

    def test_refuses_an_unknown_field(self):
        """The layout comes from rendered values, so a renamed key must not be silently ignored."""
        with pytest.raises(pydantic.ValidationError):
            pairing.PairingLayout(
                engine_cells=1,
                trainer_cells=1,
                pods_per_engine_cell=1,
                pods_per_trainer_cell=1,
                podsPerEngineCell=1,
            )


class TestTargetTrainerPod:
    def test_pairs_rank_for_rank_when_the_cells_are_the_same_width(self):
        """Equal cell widths are the identity mapping: engine (x, y) goes to trainer (x, y)."""
        layout = _layout(engine_cells=2, trainer_cells=2, pods_per_engine_cell=4, pods_per_trainer_cell=4)

        assert [_target(1, layout, engine_pod_index=index) for index in range(4)] == [
            f"{TRAINER_COMPONENT}-1",
            f"{TRAINER_COMPONENT}-1-1",
            f"{TRAINER_COMPONENT}-1-2",
            f"{TRAINER_COMPONENT}-1-3",
        ]

    def test_maps_a_single_pod_engine_by_dividing_and_taking_the_remainder(self):
        """The one-node engine case the spec writes as (x div r, x mod r), with r engines per trainer cell."""
        layout = _layout(engine_cells=8, trainer_cells=2, pods_per_engine_cell=1, pods_per_trainer_cell=4)

        assert [_target(index, layout) for index in range(8)] == [
            f"{TRAINER_COMPONENT}-{index // 4}" if index % 4 == 0 else f"{TRAINER_COMPONENT}-{index // 4}-{index % 4}"
            for index in range(8)
        ]

    def test_tiles_several_narrow_engines_across_one_trainer_cell(self):
        """A single-node engine paired with a four-node trainer cell: four engines cover it in order."""
        layout = _layout(engine_cells=8, trainer_cells=2, pods_per_engine_cell=1, pods_per_trainer_cell=4)

        assert [_target(index, layout) for index in range(4)] == [
            f"{TRAINER_COMPONENT}-0",
            f"{TRAINER_COMPONENT}-0-1",
            f"{TRAINER_COMPONENT}-0-2",
            f"{TRAINER_COMPONENT}-0-3",
        ]

    def test_moves_on_to_the_next_trainer_cell(self):
        """Engine five of eight belongs to the second trainer cell, not the first."""
        layout = _layout(engine_cells=8, trainer_cells=2, pods_per_engine_cell=1, pods_per_trainer_cell=4)

        assert _target(4, layout) == f"{TRAINER_COMPONENT}-1"

    def test_pairs_a_two_node_engine_with_half_a_four_node_trainer_cell(self):
        """The general case 1 < K_e < K_t: two engine pods land on two of the trainer cell's four."""
        layout = _layout(engine_cells=4, trainer_cells=2, pods_per_engine_cell=2, pods_per_trainer_cell=4)

        assert [_target(1, layout, engine_pod_index=index) for index in range(2)] == [
            f"{TRAINER_COMPONENT}-0-2",
            f"{TRAINER_COMPONENT}-0-3",
        ]

    def test_names_a_cells_first_pod_without_a_rank(self):
        """LeaderWorkerSet names a group's leader after the group alone, and dns follows that name."""
        assert (
            pairing.component_pod_name(component=TRAINER_COMPONENT, cell_index=3, pod_index=0)
            == f"{TRAINER_COMPONENT}-3"
        )

    def test_refuses_an_engine_wider_than_a_trainer_cell(self):
        """K_e > K_t: its extra ranks would have no trainer node to sit on, so colocate cannot hold."""
        layout = _layout(engine_cells=1, trainer_cells=1, pods_per_engine_cell=4, pods_per_trainer_cell=2)

        with pytest.raises(AssertionError, match="cannot fit"):
            _target(0, layout)

    def test_refuses_engines_that_do_not_divide_a_trainer_cell(self):
        """One engine would straddle two trainer cells, and its ranks would disagree about their peer."""
        layout = _layout(engine_cells=2, trainer_cells=1, pods_per_engine_cell=3, pods_per_trainer_cell=4)

        with pytest.raises(AssertionError, match="whole number"):
            _target(0, layout)

    def test_refuses_a_pool_larger_than_the_trainer_can_seat(self):
        """The third engine has no trainer cell left, so its weight update would transfer nothing."""
        layout = _layout(engine_cells=3, trainer_cells=2, pods_per_engine_cell=1, pods_per_trainer_cell=1)

        with pytest.raises(AssertionError, match="do not fit"):
            _target(0, layout)

    def test_refuses_an_engine_index_outside_the_pool(self):
        """A stale pod from a shrunk release must not be paired against arithmetic that no longer holds."""
        with pytest.raises(AssertionError, match="outside"):
            _target(9, _layout(engine_cells=2, trainer_cells=2))

    def test_refuses_a_pod_index_outside_its_cell(self):
        """A worker index beyond the cell width means the name was parsed against the wrong pool_id."""
        layout = _layout(engine_cells=2, trainer_cells=2, pods_per_engine_cell=2, pods_per_trainer_cell=2)

        with pytest.raises(AssertionError, match="outside"):
            _target(0, layout, engine_pod_index=5)


class TestAssertColocateSupported:
    def test_accepts_whole_node_cells_that_tile(self):
        """What the launcher checks before rendering: whole-node pods and an engine pool_id that tiles."""
        pairing.assert_colocate_supported(
            layout=_layout(engine_cells=8, trainer_cells=2, pods_per_engine_cell=1, pods_per_trainer_cell=4),
            gpus_per_engine_pod=8,
            gpus_per_trainer_pod=8,
            gpus_per_node=8,
        )

    def test_refuses_an_engine_cell_wider_than_a_trainer_cell(self):
        """K_e > K_t is refused at launch, where the message can name the values that caused it."""
        with pytest.raises(AssertionError, match="cannot fit"):
            pairing.assert_colocate_supported(
                layout=_layout(engine_cells=1, trainer_cells=1, pods_per_engine_cell=4, pods_per_trainer_cell=2),
                gpus_per_engine_pod=8,
                gpus_per_trainer_pod=8,
                gpus_per_node=8,
            )

    def test_refuses_a_sub_node_engine_cell(self):
        """The device plugin picks the cards, so an engine holding part of a node has no static base gpu id."""
        with pytest.raises(AssertionError, match="sub-node cell"):
            pairing.assert_colocate_supported(
                layout=_layout(engine_cells=1, trainer_cells=1),
                gpus_per_engine_pod=4,
                gpus_per_trainer_pod=8,
                gpus_per_node=8,
            )

    def test_refuses_a_sub_node_trainer_cell(self):
        """Two trainer cells sharing a node would leave an engine with no single cell to pair with."""
        with pytest.raises(AssertionError, match="sub-node cell"):
            pairing.assert_colocate_supported(
                layout=_layout(engine_cells=1, trainer_cells=1),
                gpus_per_engine_pod=8,
                gpus_per_trainer_pod=4,
                gpus_per_node=8,
            )


class TestReleasePatch:
    def test_pins_the_pod_to_one_node_and_removes_the_gate(self):
        """Both in one patch, so a controller restart cannot leave a pinned pod still gated."""
        patch = pairing.release_patch(node_name="gpu-7")

        assert patch[0]["value"] == {"kubernetes.io/hostname": "gpu-7"}
        assert patch[1] == {"op": "remove", "path": "/spec/schedulingGates"}

    def test_adds_one_key_when_the_pod_already_has_a_selector(self):
        """Replacing the map would drop the run's own nodeSelector, and a gated pod may only gain keys."""
        patch = pairing.release_patch(node_name="gpu-7", has_node_selector=True)

        assert patch[0] == {
            "op": "add",
            "path": "/spec/nodeSelector/kubernetes.io~1hostname",
            "value": "gpu-7",
        }

    def test_is_a_json_patch_rather_than_a_merge(self):
        """A merge patch setting the gates to an empty list is silently ignored: the list merges by name."""
        assert all("op" in operation for operation in pairing.release_patch(node_name="gpu-7"))


class TestParseLwsPodName:
    def test_reads_a_leaders_indices(self):
        """A leader carries no worker index in its name, and is worker zero of its cell."""
        indices = pairing.parse_component_pod_name(pod_name=f"{ENGINE_COMPONENT}-2", component=ENGINE_COMPONENT)

        assert (indices.cell_index, indices.pod_index) == (2, 0)

    def test_reads_a_workers_indices(self):
        """The second number is the worker index within the cell."""
        indices = pairing.parse_component_pod_name(pod_name=f"{ENGINE_COMPONENT}-2-3", component=ENGINE_COMPONENT)

        assert (indices.cell_index, indices.pod_index) == (2, 3)

    def test_returns_none_for_a_pod_of_another_pool(self):
        """Every pod of the release comes down one stream, and most of them are not engines."""
        assert pairing.parse_component_pod_name(pod_name=f"{TRAINER_COMPONENT}-0", component=ENGINE_COMPONENT) is None

    def test_returns_none_for_a_name_that_is_not_indices(self):
        """A pod merely prefixed like the pool_id must not be parsed into indices it does not have."""
        assert (
            pairing.parse_component_pod_name(pod_name=f"{ENGINE_COMPONENT}-router", component=ENGINE_COMPONENT) is None
        )


def _pod(name: str, *, node_name: str | None = None, gated: bool = True, node_selector: Any = None) -> Any:
    gates = [SimpleNamespace(name=pairing.GATE_NAME)] if gated else []
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name),
        spec=SimpleNamespace(node_name=node_name, scheduling_gates=gates, node_selector=node_selector),
    )


class FakePods:
    def __init__(self, pods: list[Any]) -> None:
        self._pods = list(pods)
        self.patched: list[tuple[str, list[dict[str, Any]]]] = []

    def pods_for(self, *, parent_key: str) -> list[Any]:
        return list(self._pods)

    async def patch(self, *, pod_name: str, patch: list[dict[str, Any]]) -> None:
        self.patched.append((pod_name, patch))


def _controller(pods: Any, layout: pairing.PairingLayout | None = None) -> pairing.PairingController:
    return pairing.PairingController(
        engine_component=ENGINE_COMPONENT,
        trainer_component=TRAINER_COMPONENT,
        layout=layout or _layout(engine_cells=2, trainer_cells=2),
        pods=pods,
    )


class TestReconcile:
    def test_releases_a_gated_engine_onto_its_trainer_node(self):
        """This is the whole point: the engine ends up where the trainer that feeds it already runs."""
        pods = FakePods(
            [_pod(f"{ENGINE_COMPONENT}-0"), _pod(f"{TRAINER_COMPONENT}-0", node_name="gpu-3", gated=False)]
        )

        asyncio.run(_controller(pods).reconcile(f"{ENGINE_COMPONENT}-0"))

        assert pods.patched == [(f"{ENGINE_COMPONENT}-0", pairing.release_patch(node_name="gpu-3"))]

    def test_keeps_a_selector_the_pod_already_carries(self):
        """The run's global nodeSelector is on the pod, and removing it makes the apiserver refuse."""
        engine = _pod(f"{ENGINE_COMPONENT}-0", node_selector={"pool": "gpu"})
        pods = FakePods([engine, _pod(f"{TRAINER_COMPONENT}-0", node_name="gpu-3", gated=False)])

        asyncio.run(_controller(pods).reconcile(f"{ENGINE_COMPONENT}-0"))

        assert pods.patched[0][1][0]["path"].endswith("kubernetes.io~1hostname")

    def test_waits_while_the_trainer_has_no_node(self):
        """Releasing now would let the scheduler put the engine anywhere, which is the bug gates prevent."""
        pods = FakePods([_pod(f"{ENGINE_COMPONENT}-0"), _pod(f"{TRAINER_COMPONENT}-0")])

        asyncio.run(_controller(pods).reconcile(f"{ENGINE_COMPONENT}-0"))

        assert pods.patched == []

    def test_waits_while_the_trainer_does_not_exist_yet(self):
        """helm creates both pool_ids at once, so an engine routinely reconciles before its trainer appears."""
        pods = FakePods([_pod(f"{ENGINE_COMPONENT}-0")])

        asyncio.run(_controller(pods).reconcile(f"{ENGINE_COMPONENT}-0"))

        assert pods.patched == []

    def test_does_nothing_for_an_engine_already_released(self):
        """Removing a gate cannot be undone, so a released pod is terminal and patching again is noise."""
        pods = FakePods(
            [
                _pod(f"{ENGINE_COMPONENT}-0", node_name="gpu-3", gated=False),
                _pod(f"{TRAINER_COMPONENT}-0", node_name="gpu-3", gated=False),
            ]
        )

        asyncio.run(_controller(pods).reconcile(f"{ENGINE_COMPONENT}-0"))

        assert pods.patched == []

    def test_does_nothing_for_an_engine_that_disappeared(self):
        """A scaled-down release deletes pods, and their queued reconciles must not resurrect anything."""
        pods = FakePods([])

        asyncio.run(_controller(pods).reconcile(f"{ENGINE_COMPONENT}-0"))

        assert pods.patched == []

    def test_is_idempotent_when_run_twice(self):
        """Delivery is at-least-once, and the second run sees a pod that is no longer gated."""
        engine = _pod(f"{ENGINE_COMPONENT}-0")
        pods = FakePods([engine, _pod(f"{TRAINER_COMPONENT}-0", node_name="gpu-3", gated=False)])
        controller = _controller(pods)

        asyncio.run(controller.reconcile(f"{ENGINE_COMPONENT}-0"))
        engine.spec.scheduling_gates = []
        asyncio.run(controller.reconcile(f"{ENGINE_COMPONENT}-0"))

        assert len(pods.patched) == 1


class TestKeyOf:
    def test_keys_an_engine_pod_by_itself(self):
        """Its own events are what drive it forward."""
        controller = _controller(FakePods([]))

        assert controller.key_of(_pod(f"{ENGINE_COMPONENT}-1")) == f"{ENGINE_COMPONENT}-1"

    def test_keys_a_trainer_pod_by_the_engine_it_unblocks(self):
        """A trainer getting a node is the event the engine is waiting for, so it must reach that key."""
        controller = _controller(FakePods([]))

        assert controller.key_of(_pod(f"{TRAINER_COMPONENT}-1")) == f"{ENGINE_COMPONENT}-1"

    def test_inverts_the_mapping_once_rather_than_per_event(self):
        """key_of runs for every pod on every relist, so a pool_id sweep per call is quadratic."""
        controller = _controller(FakePods([]))

        assert controller.engine_waiting_on(f"{TRAINER_COMPONENT}-0") == f"{ENGINE_COMPONENT}-0"
        assert controller.engine_waiting_on("r-miles-run-orchestrator-0") is None

    def test_keys_an_unrelated_pod_apart(self):
        """The orchestrator shares the release label and must not be mistaken for anything pairable."""
        controller = _controller(FakePods([]))

        assert controller.key_of(_pod("r-miles-run-orchestrator-0")).startswith("__unrelated__/")


class FakeCoreV1:
    def __init__(self) -> None:
        self.patched: list[tuple[str, list[dict[str, Any]]]] = []

    async def patch_namespaced_pod(self, *, name: str, namespace: str, body: list[dict[str, Any]]) -> None:
        self.patched.append((name, body))


class PairingHarness:
    def __init__(self, *, layout: pairing.PairingLayout | None = None) -> None:
        self.core_v1 = FakeCoreV1()
        self.source = FakeSource()
        self.clock = FakeClock()
        self.pods = pairing.StorePodApi(core_v1=self.core_v1, namespace="rl")
        self.controller = pairing.PairingController(
            engine_component=ENGINE_COMPONENT,
            trainer_component=TRAINER_COMPONENT,
            layout=layout or _layout(engine_cells=2, trainer_cells=2),
            pods=self.pods,
        )
        self.loop = ReconcileLoop(
            source=self.source,
            reconcile=self.controller.reconcile,
            key_map=self.controller.key_of,
            clock=self.clock,
            resync_period=60.0,
        )
        self.pods.read_from(loop=self.loop)

    async def start(self, *pods: Any) -> None:
        start_task = asyncio.create_task(self.loop.start())
        await settle()
        self.source.emit(replace_of(*pods))
        await settle()
        await start_task

    async def upsert(self, pod: Any) -> None:
        self.source.emit(UpsertEvent(key=pod.metadata.name, obj=pod))
        await settle()

    async def delete(self, pod: Any) -> None:
        self.source.emit(DeleteEvent(key=pod.metadata.name, last_obj=pod))
        await settle()

    def patched_names(self) -> list[str]:
        return [name for name, _ in self.core_v1.patched]


class TestEventSequences:
    async def test_releases_the_engine_when_its_trainer_lands_afterwards(self):
        """Engine first: nothing happens until the trainer's own event brings the reconcile back."""
        harness = PairingHarness()

        await harness.start(_pod(f"{ENGINE_COMPONENT}-0"), _pod(f"{TRAINER_COMPONENT}-0"))
        assert harness.core_v1.patched == []

        await harness.upsert(_pod(f"{TRAINER_COMPONENT}-0", node_name="gpu-3", gated=False))

        assert harness.core_v1.patched == [
            (f"{ENGINE_COMPONENT}-0", pairing.release_patch(node_name="gpu-3")),
        ]
        await harness.loop.stop()

    async def test_releases_the_engine_when_the_trainer_was_there_first(self):
        """Trainer first: the engine's own arrival finds a placed trainer and releases immediately."""
        harness = PairingHarness()

        await harness.start(_pod(f"{TRAINER_COMPONENT}-1", node_name="gpu-9", gated=False))
        await harness.upsert(_pod(f"{ENGINE_COMPONENT}-1"))

        assert harness.core_v1.patched == [
            (f"{ENGINE_COMPONENT}-1", pairing.release_patch(node_name="gpu-9")),
        ]
        await harness.loop.stop()

    async def test_leaves_the_engine_gated_while_the_trainer_has_no_node(self):
        """A trainer pod exists long before it is scheduled, and its node is the only thing worth waiting on."""
        harness = PairingHarness()

        await harness.start(_pod(f"{ENGINE_COMPONENT}-0"))
        await harness.upsert(_pod(f"{TRAINER_COMPONENT}-0"))

        assert harness.core_v1.patched == []
        await harness.loop.stop()

    async def test_does_not_patch_an_engine_that_is_already_released(self):
        """A controller restart relists a world where the gate is gone, which is a terminal state."""
        harness = PairingHarness()

        await harness.start(
            _pod(f"{ENGINE_COMPONENT}-0", node_name="gpu-3", gated=False),
            _pod(f"{TRAINER_COMPONENT}-0", node_name="gpu-3", gated=False),
        )

        assert harness.core_v1.patched == []
        await harness.loop.stop()

    async def test_does_not_patch_twice_when_the_release_is_observed_back(self):
        """The patch produces its own watch event, and re-entering must not repeat the write."""
        harness = PairingHarness()

        await harness.start(
            _pod(f"{ENGINE_COMPONENT}-0"), _pod(f"{TRAINER_COMPONENT}-0", node_name="gpu-3", gated=False)
        )
        await harness.upsert(_pod(f"{ENGINE_COMPONENT}-0", node_name="gpu-3", gated=False))

        assert harness.patched_names() == [f"{ENGINE_COMPONENT}-0"]
        await harness.loop.stop()

    async def test_re_enters_safely_after_a_crash_before_the_patch_landed(self):
        """A controller that died before patching sees the same gated pod again and finishes the job."""
        first = PairingHarness()
        await first.start(
            _pod(f"{ENGINE_COMPONENT}-0"), _pod(f"{TRAINER_COMPONENT}-0", node_name="gpu-3", gated=False)
        )
        await first.loop.stop()

        second = PairingHarness()
        await second.start(
            _pod(f"{ENGINE_COMPONENT}-0"), _pod(f"{TRAINER_COMPONENT}-0", node_name="gpu-3", gated=False)
        )

        assert second.patched_names() == [f"{ENGINE_COMPONENT}-0"]
        await second.loop.stop()

    async def test_resync_releases_an_engine_whose_trainer_event_was_missed(self):
        """The backstop for a lost event: the next resync reconciles every key from the store again."""
        harness = PairingHarness()
        trainer = _pod(f"{TRAINER_COMPONENT}-0", gated=False)
        await harness.start(_pod(f"{ENGINE_COMPONENT}-0"), trainer)
        assert harness.core_v1.patched == []

        trainer.spec.node_name = "gpu-3"
        await harness.clock.elapse(60.0)
        await settle()

        assert harness.patched_names() == [f"{ENGINE_COMPONENT}-0"]
        await harness.loop.stop()

    async def test_a_deleted_engine_is_never_patched(self):
        """Scale-down deletes gated pods, and a queued reconcile must not write to a name that is gone."""
        harness = PairingHarness()
        engine = _pod(f"{ENGINE_COMPONENT}-0")

        await harness.start(engine, _pod(f"{TRAINER_COMPONENT}-0"))
        await harness.delete(engine)
        await harness.upsert(_pod(f"{TRAINER_COMPONENT}-0", node_name="gpu-3", gated=False))

        assert harness.core_v1.patched == []
        await harness.loop.stop()
