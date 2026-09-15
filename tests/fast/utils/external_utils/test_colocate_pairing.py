from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

import pydantic
import pytest
from tests.fast.utils.workers.reconcile.utils import FakeSource, replace_of, settle

from miles.utils.external_utils.colocate_pairing import pods as pairing_pods
from miles.utils.external_utils.colocate_pairing.config import InferencePool, PairingConfig, PairingLayout
from miles.utils.external_utils.colocate_pairing.controller import PairingController, _target_trainer_pod
from miles.utils.external_utils.colocate_pairing.pods import PodCoordinate
from miles.utils.external_utils.command_utils.helm_backend.launcher.values.colocate import _assert_colocate_supported
from miles.utils.test_utils.clock import FakeClock
from miles.utils.workers.reconcile.loop import ReconcileLoop
from miles.utils.workers.reconcile.source_event import DeleteEvent, UpsertEvent
from miles.utils.workers.worker_provider.kubernetes.helm.env import DEFAULT_LABEL_KEYS

TRAINER_POOL_ID = "trainer-engine-actor"
INFERENCE_POOL_ID = "inference-inference"
_OBJECT_NAME_PREFIX = "r-miles-run-"


GPUS_PER_NODE = 8


def _layout(
    num_inference_cells: int,
    num_trainer_cells: int,
    num_pods_per_inference_cell: int = 1,
    num_pods_per_trainer_cell: int = 1,
    gpu_offset: int = 0,
    num_gpus_per_node: int = GPUS_PER_NODE,
    num_gpus_per_inference_pod: int | None = None,
) -> PairingLayout:
    return PairingLayout(
        num_inference_cells=num_inference_cells,
        num_trainer_cells=num_trainer_cells,
        num_pods_per_inference_cell=num_pods_per_inference_cell,
        num_pods_per_trainer_cell=num_pods_per_trainer_cell,
        num_gpus_per_node=num_gpus_per_node,
        num_gpus_per_inference_pod=(
            num_gpus_per_node if num_gpus_per_inference_pod is None else num_gpus_per_inference_pod
        ),
        gpu_offset=gpu_offset,
    )


def _sub_node_layout(gpu_offset: int = 0, num_inference_cells: int = 4) -> PairingLayout:
    return _layout(
        num_inference_cells=num_inference_cells,
        num_trainer_cells=1,
        num_pods_per_inference_cell=1,
        num_pods_per_trainer_cell=2,
        num_gpus_per_inference_pod=4,
        gpu_offset=gpu_offset,
    )


def _target(inference_cell_index: int, layout: PairingLayout, inference_pod_index: int = 0) -> PodCoordinate:
    return _target_trainer_pod(
        inference_cell_index=inference_cell_index,
        inference_pod_index=inference_pod_index,
        layout=layout,
        trainer_pool_id=TRAINER_POOL_ID,
    )


def _coordinate(cell_index: int, pod_index: int = 0, pool_id: str = TRAINER_POOL_ID) -> PodCoordinate:
    return PodCoordinate(pool_id=pool_id, cell_index=cell_index, pod_in_cell_index=pod_index)


def _key(pool_id: str, cell_index: int, pod_index: int = 0) -> str:
    return _coordinate(cell_index, pod_index, pool_id).key


def _pod_name(pool_id: str, cell_index: int, pod_index: int = 0) -> str:
    suffix = str(cell_index) if pod_index == 0 else f"{cell_index}-{pod_index}"
    return f"{_OBJECT_NAME_PREFIX}{pool_id}-{suffix}"


class TestPairingLayout:
    def test_refuses_a_pool_with_no_cells(self):
        """Zero cells is a values bug, and every later division would be by zero."""
        with pytest.raises(pydantic.ValidationError):
            _layout(num_inference_cells=0, num_trainer_cells=1)

    def test_refuses_a_cell_with_no_pods(self):
        """A cell is at least one pod, and the mapping divides by the inference cell width."""
        with pytest.raises(pydantic.ValidationError):
            _layout(num_inference_cells=1, num_trainer_cells=1, num_pods_per_inference_cell=0)

    def test_refuses_a_layout_that_never_says_how_wide_an_inference_pod_is(self):
        """Without the pod width the pairing cannot tell a whole-node pod from four sub-node ones."""
        with pytest.raises(pydantic.ValidationError):
            PairingLayout(
                num_inference_cells=1,
                num_trainer_cells=1,
                num_pods_per_inference_cell=1,
                num_pods_per_trainer_cell=1,
                num_gpus_per_node=GPUS_PER_NODE,
                gpu_offset=0,
            )

    def test_refuses_an_inference_pod_with_no_gpus(self):
        """A zero-gpu pod would divide by zero in the mapping and claim no gpu of the trainer's."""
        with pytest.raises(pydantic.ValidationError):
            _layout(num_inference_cells=1, num_trainer_cells=1, num_gpus_per_inference_pod=0)

    def test_counts_the_gpus_every_inference_pod_of_the_pool_holds(self):
        """The fit check is in gpus now, so the pool's width is cells times pods times the pod's gpus."""
        assert _sub_node_layout().total_inference_gpus == 16

    def test_counts_the_gpus_the_trainer_pool_holds_as_whole_nodes(self):
        """A trainer pod is always a whole node, which is what the inference pool is measured against."""
        assert _sub_node_layout().total_trainer_gpus == 16

    def test_refuses_an_unknown_field(self):
        """The layout comes from rendered values, so a renamed key must not be silently ignored."""
        with pytest.raises(pydantic.ValidationError):
            PairingLayout(
                num_inference_cells=1,
                num_trainer_cells=1,
                num_pods_per_inference_cell=1,
                num_pods_per_trainer_cell=1,
                num_gpus_per_node=GPUS_PER_NODE,
                num_gpus_per_inference_pod=GPUS_PER_NODE,
                gpu_offset=0,
                podsPerInferenceCell=1,
            )


_WHOLE_NODE_GRID = dict(
    num_inference_cells=(1, 2, 3),
    num_trainer_cells=(1, 2),
    num_pods_per_inference_cell=(1, 2, 3),
    num_pods_per_trainer_cell=(1, 2, 4),
)


def _whole_node_cases() -> list[dict[str, int]]:
    return [
        dict(zip(_WHOLE_NODE_GRID, values, strict=True)) for values in itertools.product(*_WHOLE_NODE_GRID.values())
    ]


def _pod_granular_layout_is_legal(*, gpu_offset: int, **case: int) -> bool:
    pods_per_inference_cell = case["num_pods_per_inference_cell"]
    pods_per_trainer_cell = case["num_pods_per_trainer_cell"]
    pod_offset = gpu_offset // GPUS_PER_NODE
    return (
        pods_per_inference_cell <= pods_per_trainer_cell
        and pods_per_trainer_cell % pods_per_inference_cell == 0
        and gpu_offset % GPUS_PER_NODE == 0
        and pod_offset % pods_per_inference_cell == 0
        and pod_offset + case["num_inference_cells"] * pods_per_inference_cell
        <= case["num_trainer_cells"] * pods_per_trainer_cell
    )


def _gpu_granular_layout_is_legal(*, gpu_offset: int, **case: int) -> bool:
    try:
        _layout(gpu_offset=gpu_offset, num_gpus_per_inference_pod=GPUS_PER_NODE, **case)
    except pydantic.ValidationError:
        return False
    return True


class TestWholeNodeLayoutsAreJudgedExactlyAsTheyWereInPods:
    @pytest.mark.parametrize("gpu_offset", [0, 4, 8, 12, 16, 24, 32])
    def test_accepts_and_refuses_the_same_layouts_the_pod_granular_rules_did(self, gpu_offset: int):
        """The gpu rewrite has to be a pure generalisation: a whole-node pool sees no change in verdict."""
        disagreements = [
            case
            for case in _whole_node_cases()
            if _gpu_granular_layout_is_legal(gpu_offset=gpu_offset, **case)
            != _pod_granular_layout_is_legal(gpu_offset=gpu_offset, **case)
        ]

        assert disagreements == []

    def test_the_grid_it_compares_reaches_both_verdicts(self):
        """A grid that only ever accepts, or only ever refuses, would make the comparison above vacuous."""
        verdicts = {_gpu_granular_layout_is_legal(gpu_offset=8, **case) for case in _whole_node_cases()}

        assert verdicts == {True, False}


class TestTargetTrainerPod:
    def test_pairs_rank_for_rank_when_the_cells_are_the_same_width(self):
        """Equal cell widths are the identity mapping: inference (x, y) goes to trainer (x, y)."""
        layout = _layout(
            num_inference_cells=2, num_trainer_cells=2, num_pods_per_inference_cell=4, num_pods_per_trainer_cell=4
        )

        assert [_target(1, layout, inference_pod_index=index) for index in range(4)] == [
            _coordinate(1, index) for index in range(4)
        ]

    def test_maps_a_single_pod_inference_by_dividing_and_taking_the_remainder(self):
        """The one-node inference case the spec writes as (x div r, x mod r), with r inferences per trainer cell."""
        layout = _layout(
            num_inference_cells=8, num_trainer_cells=2, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=4
        )

        assert [_target(index, layout) for index in range(8)] == [
            _coordinate(index // 4, index % 4) for index in range(8)
        ]

    def test_tiles_several_narrow_inference_pods_across_one_trainer_cell(self):
        """A single-node inference paired with a four-node trainer cell: four inferences cover it in order."""
        layout = _layout(
            num_inference_cells=8, num_trainer_cells=2, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=4
        )

        assert [_target(index, layout) for index in range(4)] == [_coordinate(0, index) for index in range(4)]

    def test_moves_on_to_the_next_trainer_cell(self):
        """Inference five of eight belongs to the second trainer cell, not the first."""
        layout = _layout(
            num_inference_cells=8, num_trainer_cells=2, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=4
        )

        assert _target(4, layout) == _coordinate(1)

    def test_pairs_a_two_node_inference_with_half_a_four_node_trainer_cell(self):
        """The general case 1 < K_e < K_t: two inference pods land on two of the trainer cell's four."""
        layout = _layout(
            num_inference_cells=4, num_trainer_cells=2, num_pods_per_inference_cell=2, num_pods_per_trainer_cell=4
        )

        assert [_target(1, layout, inference_pod_index=index) for index in range(2)] == [
            _coordinate(0, 2),
            _coordinate(0, 3),
        ]

    def test_seats_two_sub_node_inference_pods_on_one_trainer_pod(self):
        """Half-node inference pods share the node their trainer pod holds, two of them per trainer pod."""
        layout = _sub_node_layout()

        assert [_target(index, layout) for index in range(4)] == [
            _coordinate(0, 0),
            _coordinate(0, 0),
            _coordinate(0, 1),
            _coordinate(0, 1),
        ]

    def test_seats_four_quarter_node_inference_pods_on_one_trainer_pod(self):
        """The pod width alone decides how many inference pods a trainer's node seats, down to a quarter."""
        layout = _layout(
            num_inference_cells=8,
            num_trainer_cells=1,
            num_pods_per_inference_cell=1,
            num_pods_per_trainer_cell=2,
            num_gpus_per_inference_pod=2,
        )

        assert [_target(index, layout) for index in range(8)] == [_coordinate(0, index // 4) for index in range(8)]

    def test_refuses_an_inference_wider_than_a_trainer_cell(self):
        """K_e > K_t: its extra ranks would have no trainer node to sit on, so colocate cannot hold."""
        with pytest.raises(pydantic.ValidationError, match="cannot fit"):
            _layout(
                num_inference_cells=1, num_trainer_cells=1, num_pods_per_inference_cell=4, num_pods_per_trainer_cell=2
            )

    def test_refuses_inference_cells_that_do_not_divide_a_trainer_cell(self):
        """One inference would straddle two trainer cells, and its ranks would disagree about their peer."""
        with pytest.raises(pydantic.ValidationError, match="whole number"):
            _layout(
                num_inference_cells=2, num_trainer_cells=1, num_pods_per_inference_cell=3, num_pods_per_trainer_cell=4
            )

    def test_refuses_a_pool_larger_than_the_trainer_can_seat(self):
        """The third inference has no trainer cell left, so its weight update would transfer nothing."""
        with pytest.raises(pydantic.ValidationError, match="do not fit"):
            _layout(
                num_inference_cells=3, num_trainer_cells=2, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=1
            )

    def test_refuses_an_inference_index_outside_the_pool(self):
        """A stale pod from a shrunk release must not be paired against arithmetic that no longer holds."""
        with pytest.raises(AssertionError, match="outside"):
            _target(9, _layout(num_inference_cells=2, num_trainer_cells=2))

    def test_refuses_a_pod_index_outside_its_cell(self):
        """A worker index beyond the cell width means the name was parsed against the wrong pool_id."""
        layout = _layout(
            num_inference_cells=2, num_trainer_cells=2, num_pods_per_inference_cell=2, num_pods_per_trainer_cell=2
        )

        with pytest.raises(AssertionError, match="outside"):
            _target(0, layout, inference_pod_index=5)


def _target_by_cell_division(
    inference_cell_index: int, inference_pod_index: int, layout: PairingLayout
) -> PodCoordinate:
    inferences_per_trainer_cell = layout.num_pods_per_trainer_cell // layout.num_pods_per_inference_cell
    trainer_cell_index = inference_cell_index // inferences_per_trainer_cell
    offset_within_cell = inference_cell_index % inferences_per_trainer_cell
    return _coordinate(
        trainer_cell_index, offset_within_cell * layout.num_pods_per_inference_cell + inference_pod_index
    )


def _all_targets(layout: PairingLayout) -> list[PodCoordinate]:
    return [
        _target(cell_index, layout, inference_pod_index=pod_index)
        for cell_index in range(layout.num_inference_cells)
        for pod_index in range(layout.num_pods_per_inference_cell)
    ]


class TestGpuOffsetPairing:
    @pytest.mark.parametrize(
        "layout",
        [
            _layout(
                num_inference_cells=8, num_trainer_cells=2, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=4
            ),
            _layout(
                num_inference_cells=4, num_trainer_cells=2, num_pods_per_inference_cell=2, num_pods_per_trainer_cell=4
            ),
            _layout(
                num_inference_cells=2, num_trainer_cells=2, num_pods_per_inference_cell=4, num_pods_per_trainer_cell=4
            ),
        ],
        ids=["narrow", "half-cell", "whole-cell"],
    )
    def test_a_pool_at_the_first_gpu_maps_exactly_as_dividing_by_cells_did(self, layout):
        """The offset form has to be a generalisation, so at offset zero it must reproduce the old mapping."""
        assert _all_targets(layout) == [
            _target_by_cell_division(cell_index, pod_index, layout)
            for cell_index in range(layout.num_inference_cells)
            for pod_index in range(layout.num_pods_per_inference_cell)
        ]

    def test_an_offset_pool_starts_at_the_trainer_pod_holding_its_first_gpu(self):
        """The offset is what makes a second pool land beside the trainers it actually reads weights from."""
        layout = _layout(
            num_inference_cells=2,
            num_trainer_cells=1,
            num_pods_per_inference_cell=1,
            num_pods_per_trainer_cell=4,
            gpu_offset=16,
        )

        assert _all_targets(layout) == [_coordinate(0, 2), _coordinate(0, 3)]

    def test_an_offset_crosses_into_the_next_trainer_cell(self):
        """Trainer cells are numbered off the same flat gpu range, so an offset can carry a pool into the next."""
        layout = _layout(
            num_inference_cells=1,
            num_trainer_cells=2,
            num_pods_per_inference_cell=1,
            num_pods_per_trainer_cell=2,
            gpu_offset=16,
        )

        assert _all_targets(layout) == [_coordinate(1)]

    def test_refuses_an_offset_that_starts_inside_a_node(self):
        """Half a node is not a whole-node pod, and the inference would want gpus of two trainer pods at once."""
        with pytest.raises(pydantic.ValidationError, match="is not a whole number of its own"):
            _layout(num_inference_cells=1, num_trainer_cells=2, num_pods_per_trainer_cell=2, gpu_offset=4)

    def test_refuses_an_offset_that_starts_inside_a_sub_node_pod(self):
        """The offset is counted in the pool's own pods, so two gpus in is half a four-gpu pod, not a start."""
        with pytest.raises(pydantic.ValidationError, match="is not a whole number of its own"):
            _sub_node_layout(gpu_offset=2, num_inference_cells=1)

    def test_starts_a_sub_node_pool_on_the_trainer_pod_holding_the_offset_gpu(self):
        """A pool offset by one sub-node pod still lands on the trainer pod whose node holds that gpu."""
        layout = _sub_node_layout(gpu_offset=4, num_inference_cells=3)

        assert _all_targets(layout) == [_coordinate(0, 0), _coordinate(0, 1), _coordinate(0, 1)]

    def test_refuses_an_offset_that_splits_an_inference_cell_across_trainer_cells(self):
        """An offset of half a cell leaves the pool unaligned, so a later cell of it would span two trainer cells."""
        with pytest.raises(pydantic.ValidationError, match="so its cells would straddle trainer cells"):
            _layout(
                num_inference_cells=1,
                num_trainer_cells=2,
                num_pods_per_inference_cell=2,
                num_pods_per_trainer_cell=4,
                gpu_offset=8,
            )

    def test_allows_an_offset_of_whole_inference_cells(self):
        """Offsetting by a whole two-pod cell keeps every later cell inside one trainer cell."""
        layout = _layout(
            num_inference_cells=1,
            num_trainer_cells=2,
            num_pods_per_inference_cell=2,
            num_pods_per_trainer_cell=4,
            gpu_offset=16,
        )

        assert _all_targets(layout) == [_coordinate(0, 2), _coordinate(0, 3)]

    def test_refuses_a_pool_that_the_offset_pushes_past_the_last_trainer_pod(self):
        """Its last inference would sit on a gpu no trainer holds, and a weight update would transfer nothing."""
        with pytest.raises(pydantic.ValidationError, match="do not fit"):
            _layout(
                num_inference_cells=2,
                num_trainer_cells=1,
                num_pods_per_inference_cell=1,
                num_pods_per_trainer_cell=2,
                gpu_offset=8,
            )


class TestLayoutPairs:
    def test_allows_trainer_cells_that_seat_no_inference(self):
        """A prefill pool_id on its own nodes leaves trainer gpus with no inference, which is a legal run."""
        _layout(num_inference_cells=2, num_trainer_cells=4)

    def test_allows_a_trainer_cell_seating_fewer_inference_pods_than_it_could(self):
        """Half a trainer cell may run inferences and the other half none, which is still rank-for-rank."""
        _layout(num_inference_cells=2, num_trainer_cells=1, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=4)

    def test_refuses_more_inference_cells_than_the_trainer_cells_seat(self):
        """The surplus inferences would pair with a trainer cell the run never created."""
        with pytest.raises(pydantic.ValidationError, match="do not fit"):
            _layout(
                num_inference_cells=8, num_trainer_cells=1, num_pods_per_inference_cell=1, num_pods_per_trainer_cell=4
            )

    def test_refuses_an_inference_cell_that_straddles_two_trainer_cells(self):
        """No single trainer cell then owns the inference, so healing one would leave the other half live."""
        with pytest.raises(pydantic.ValidationError, match="whole number"):
            _layout(
                num_inference_cells=1, num_trainer_cells=1, num_pods_per_inference_cell=2, num_pods_per_trainer_cell=3
            )


class TestGpuWidthPairs:
    def test_refuses_an_inference_pod_that_does_not_divide_a_node(self):
        """Three gpus of an eight-gpu node means a later pod of the pool would straddle two trainer pods."""
        with pytest.raises(pydantic.ValidationError, match="does not divide a"):
            _layout(num_inference_cells=1, num_trainer_cells=1, num_gpus_per_inference_pod=3)

    def test_refuses_an_inference_pod_wider_than_a_node(self):
        """A pod wider than the node it is pinned to would need gpus the trainer pod beside it holds."""
        with pytest.raises(pydantic.ValidationError, match="does not divide a"):
            _layout(num_inference_cells=1, num_trainer_cells=2, num_gpus_per_inference_pod=16)

    def test_refuses_sub_node_pods_that_overrun_the_trainer_gpus_by_less_than_a_node(self):
        """Counting in pods would round this down and miss it: the last half-node pod has no trainer gpu."""
        with pytest.raises(pydantic.ValidationError, match="do not fit in the trainer's"):
            _sub_node_layout(gpu_offset=4, num_inference_cells=4)

    def test_allows_sub_node_pods_that_exactly_fill_the_trainer_gpus(self):
        """The fit check is inclusive, so a pool covering every trainer gpu is the legal maximum."""
        assert _sub_node_layout(num_inference_cells=4).total_inference_gpus == 16


class TestAssertColocateSupported:
    def test_accepts_whole_node_cells_that_tile(self):
        """What the launcher checks before rendering: whole-node pods and an inference pool_id that tiles."""
        _assert_colocate_supported(
            num_gpus_per_node=GPUS_PER_NODE,
            gpus_per_inference_pod=8,
            gpus_per_trainer_pod=8,
        )

    def test_refuses_a_sub_node_inference_cell(self):
        """The device plugin picks the cards, so an inference holding part of a node has no static base gpu id."""
        with pytest.raises(AssertionError, match="sub-node cell"):
            _assert_colocate_supported(
                num_gpus_per_node=GPUS_PER_NODE,
                gpus_per_inference_pod=4,
                gpus_per_trainer_pod=8,
            )

    def test_refuses_a_sub_node_trainer_cell(self):
        """Two trainer cells sharing a node would leave an inference with no single cell to pair with."""
        with pytest.raises(AssertionError, match="sub-node cell"):
            _assert_colocate_supported(
                num_gpus_per_node=GPUS_PER_NODE,
                gpus_per_inference_pod=8,
                gpus_per_trainer_pod=4,
            )


class TestPoolsClaimDistinctGpus:
    def test_refuses_two_pools_that_want_the_same_trainer_gpus(self):
        """Only one inference can hold a node's gpus, and the second would land on gpus already taken."""
        with pytest.raises(pydantic.ValidationError, match="both claim the trainer's gpu 8"):
            _config(
                [
                    _inference_pool(_layout(num_inference_cells=2, num_trainer_cells=1, num_pods_per_trainer_cell=4)),
                    _inference_pool(
                        _layout(
                            num_inference_cells=2,
                            num_trainer_cells=1,
                            num_pods_per_trainer_cell=4,
                            gpu_offset=8,
                        ),
                        pool_id=DECODE_POOL_ID,
                    ),
                ]
            )

    def test_refuses_two_sub_node_pools_that_want_the_same_gpu_of_one_node(self):
        """Sub-node pools share a node on purpose, so the overlap check has to be per gpu, not per node."""
        with pytest.raises(pydantic.ValidationError, match="both claim the trainer's gpu 4"):
            _config(
                [
                    _inference_pool(_sub_node_layout(num_inference_cells=2)),
                    _inference_pool(_sub_node_layout(gpu_offset=4, num_inference_cells=1), pool_id=DECODE_POOL_ID),
                ]
            )

    def test_allows_two_sub_node_pools_that_split_one_node_between_them(self):
        """Two half-node pools on one trainer node is the point of sub-node cells, not a collision."""
        _config(
            [
                _inference_pool(_sub_node_layout(num_inference_cells=1)),
                _inference_pool(_sub_node_layout(gpu_offset=4, num_inference_cells=1), pool_id=DECODE_POOL_ID),
            ]
        )


class TestReleasePatch:
    def test_pins_the_pod_to_one_node_and_removes_the_gate(self):
        """Both in one patch, so a controller restart cannot leave a pinned pod still gated."""
        patch = pairing_pods.release_patch(node_name="gpu-7", gates=[pairing_pods._GATE_NAME], has_node_selector=False)

        assert patch[0]["value"] == {"kubernetes.io/hostname": "gpu-7"}
        assert patch[1:] == [
            {"op": "test", "path": "/spec/schedulingGates/0/name", "value": pairing_pods._GATE_NAME},
            {"op": "remove", "path": "/spec/schedulingGates/0"},
        ]

    def test_adds_one_key_when_the_pod_already_has_a_selector(self):
        """Replacing the map would drop the run's own nodeSelector, and a gated pod may only gain keys."""
        patch = pairing_pods.release_patch(node_name="gpu-7", gates=[pairing_pods._GATE_NAME], has_node_selector=True)

        assert patch[0] == {
            "op": "add",
            "path": "/spec/nodeSelector/kubernetes.io~1hostname",
            "value": "gpu-7",
        }

    def test_removes_only_its_own_gate(self):
        """Dropping the whole list would release a pod another controller is still deliberately holding back."""
        gates = ["other.io/first", pairing_pods._GATE_NAME, "other.io/last"]

        patch = pairing_pods.release_patch(node_name="gpu-7", gates=gates, has_node_selector=False)

        assert patch[1:] == [
            {"op": "test", "path": "/spec/schedulingGates/1/name", "value": pairing_pods._GATE_NAME},
            {"op": "remove", "path": "/spec/schedulingGates/1"},
        ]

    def test_is_a_json_patch_rather_than_a_merge(self):
        """A merge patch setting the gates to an empty list is silently ignored: the list merges by name."""
        patch = pairing_pods.release_patch(node_name="gpu-7", gates=[pairing_pods._GATE_NAME], has_node_selector=False)

        assert all("op" in operation for operation in patch)


class TestCoordinateOf:
    def test_reads_the_pool_and_the_indices_the_chart_labelled(self):
        """The same three labels the worker provider reads, so both sides identify a pod the same way."""
        assert pairing_pods.coordinate_of(_pod(INFERENCE_POOL_ID, 2, 3)) == _coordinate(2, 3, INFERENCE_POOL_ID)

    def test_does_not_read_the_name(self):
        """A pod name is the release's object name plus indices; only the labels are the contract."""
        renamed = _pod(INFERENCE_POOL_ID, 2, 3)
        renamed.metadata.name = "something-else-entirely"

        assert pairing_pods.coordinate_of(renamed) == _coordinate(2, 3, INFERENCE_POOL_ID)

    def test_returns_none_for_a_pod_the_run_did_not_label(self):
        """Every pod of the release comes down one stream, and most of them belong to no pool."""
        assert pairing_pods.coordinate_of(_unlabelled_pod("r-miles-run-orchestrator-0")) is None


def _pod(
    pool_id: str,
    cell_index: int = 0,
    pod_index: int = 0,
    *,
    node_name: str | None = None,
    gated: bool = True,
    node_selector: Any = None,
) -> Any:
    pod = _unlabelled_pod(
        _pod_name(pool_id, cell_index, pod_index),
        node_name=node_name,
        gated=gated,
        node_selector=node_selector,
    )
    pod.metadata.labels = {
        DEFAULT_LABEL_KEYS.pool_id: pool_id,
        DEFAULT_LABEL_KEYS.cell_index: str(cell_index),
        DEFAULT_LABEL_KEYS.pod_in_cell_index: str(pod_index),
    }
    return pod


def _unlabelled_pod(name: str, *, node_name: str | None = None, gated: bool = True, node_selector: Any = None) -> Any:
    gates = [SimpleNamespace(name=pairing_pods._GATE_NAME)] if gated else []
    return SimpleNamespace(
        metadata=SimpleNamespace(name=name, uid=f"uid-{name}", labels={}, annotations={}, deletion_timestamp=None),
        spec=SimpleNamespace(node_name=node_name, scheduling_gates=gates, node_selector=node_selector, subdomain=None),
        status=SimpleNamespace(pod_ip=None, conditions=[], container_statuses=[]),
    )


class FakeLoop:
    def __init__(self, pods: list[Any]) -> None:
        self._pods = list(pods)

    def get_by_parent(self, parent_key: str) -> list[Any]:
        return list(self._pods)


def _attached(controller: PairingController, pods: list[Any]) -> PairingController:
    controller.set_loop(FakeLoop(pods))
    return controller


def _inference_pool(layout: PairingLayout, pool_id: str = INFERENCE_POOL_ID) -> InferencePool:
    return InferencePool(pool_id=pool_id, layout=layout)


def _config(pools: list[InferencePool]) -> PairingConfig:
    return PairingConfig(namespace="rl", release="r", trainer_pool_id=TRAINER_POOL_ID, inference_pools=pools)


def _controller(core_v1: Any, layout: PairingLayout | None = None) -> PairingController:
    pools = [_inference_pool(layout or _layout(num_inference_cells=2, num_trainer_cells=2))]
    return PairingController(config=_config(pools), core_v1=core_v1)


class TestReconcile:
    def test_releases_a_gated_inference_onto_its_trainer_node(self):
        """This is the whole point: the inference ends up where the trainer that feeds it already runs."""
        core_v1 = FakeCoreV1()
        pods = [_pod(INFERENCE_POOL_ID, 0), _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False)]

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(INFERENCE_POOL_ID, 0)))

        assert core_v1.patched == [
            (
                _pod_name(INFERENCE_POOL_ID, 0),
                pairing_pods.release_patch(
                    node_name="gpu-3", gates=[pairing_pods._GATE_NAME], has_node_selector=False
                ),
            )
        ]

    def test_keeps_a_selector_the_pod_already_carries(self):
        """The run's global nodeSelector is on the pod, and removing it makes the apiserver refuse."""
        inference = _pod(INFERENCE_POOL_ID, 0, node_selector={"pool": "gpu"})
        core_v1 = FakeCoreV1()
        pods = [inference, _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False)]

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(INFERENCE_POOL_ID, 0)))

        assert core_v1.patched[0][1][0]["path"].endswith("kubernetes.io~1hostname")

    def test_waits_while_the_trainer_has_no_node(self):
        """Releasing now would let the scheduler put the inference anywhere, which is the bug gates prevent."""
        core_v1 = FakeCoreV1()
        pods = [_pod(INFERENCE_POOL_ID, 0), _pod(TRAINER_POOL_ID, 0)]

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(INFERENCE_POOL_ID, 0)))

        assert core_v1.patched == []

    def test_waits_while_the_trainer_does_not_exist_yet(self):
        """helm creates both pool_ids at once, so an inference routinely reconciles before its trainer appears."""
        core_v1 = FakeCoreV1()
        pods = [_pod(INFERENCE_POOL_ID, 0)]

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(INFERENCE_POOL_ID, 0)))

        assert core_v1.patched == []

    def test_does_nothing_for_an_inference_already_released(self):
        """Removing a gate cannot be undone, so a released pod is terminal and patching again is noise."""
        core_v1 = FakeCoreV1()
        pods = [
            _pod(INFERENCE_POOL_ID, 0, node_name="gpu-3", gated=False),
            _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False),
        ]

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(INFERENCE_POOL_ID, 0)))

        assert core_v1.patched == []

    def test_does_nothing_for_an_inference_that_disappeared(self):
        """A scaled-down release deletes pods, and their queued reconciles must not resurrect anything."""
        core_v1 = FakeCoreV1()
        pods = []

        asyncio.run(_attached(_controller(core_v1), pods).reconcile(_key(INFERENCE_POOL_ID, 0)))

        assert core_v1.patched == []

    def test_is_idempotent_when_run_twice(self):
        """Delivery is at-least-once, and the second run sees a pod that is no longer gated."""
        inference = _pod(INFERENCE_POOL_ID, 0)
        core_v1 = FakeCoreV1()
        pods = [inference, _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False)]
        controller = _attached(_controller(core_v1), pods)

        asyncio.run(controller.reconcile(_key(INFERENCE_POOL_ID, 0)))
        inference.spec.scheduling_gates = []
        asyncio.run(controller.reconcile(_key(INFERENCE_POOL_ID, 0)))

        assert len(core_v1.patched) == 1


class TestKeyOf:
    def test_keys_an_inference_pod_by_itself(self):
        """Its own events are what drive it forward."""
        controller = _controller(FakeCoreV1())

        assert controller.key_of(_pod(INFERENCE_POOL_ID, 1)) == _key(INFERENCE_POOL_ID, 1)

    def test_keys_a_trainer_pod_by_the_inference_it_unblocks(self):
        """A trainer getting a node is the event the inference is waiting for, so it must reach that key."""
        controller = _controller(FakeCoreV1())

        assert controller.key_of(_pod(TRAINER_POOL_ID, 1)) == _key(INFERENCE_POOL_ID, 1)

    def test_keys_a_trainer_that_seats_no_inference_apart(self):
        """A trainer cell the run left empty must not be routed to an inference key nothing waits on."""
        controller = _controller(FakeCoreV1(), _layout(num_inference_cells=1, num_trainer_cells=4))

        assert controller.key_of(_pod(TRAINER_POOL_ID, 3)).startswith("__unrelated__/")

    def test_keys_an_unrelated_pod_apart(self):
        """The orchestrator shares the release label and must not be mistaken for anything pairable."""
        controller = _controller(FakeCoreV1())

        assert controller.key_of(_unlabelled_pod("r-miles-run-orchestrator-0")).startswith("__unrelated__/")


DECODE_POOL_ID = "inference-inference-decode"


def _two_pool_controller(core_v1: Any) -> PairingController:
    pools = [
        _inference_pool(_layout(num_inference_cells=2, num_trainer_cells=1, num_pods_per_trainer_cell=4)),
        _inference_pool(
            _layout(num_inference_cells=2, num_trainer_cells=1, num_pods_per_trainer_cell=4, gpu_offset=16),
            pool_id=DECODE_POOL_ID,
        ),
    ]
    return PairingController(config=_config(pools), core_v1=core_v1)


class TestSeveralInferencePools:
    def test_each_pool_lands_on_the_trainer_pods_its_own_offset_names(self):
        """One controller drives every colocated pool_id, and the offset is all that tells them apart."""
        controller = _two_pool_controller(FakeCoreV1())

        assert [controller.key_of(_pod(TRAINER_POOL_ID, 0, index)) for index in (1, 2)] == [
            _key(INFERENCE_POOL_ID, 1),
            _key(DECODE_POOL_ID, 0),
        ]

    def test_releases_a_pod_of_the_second_pool_onto_its_own_trainer(self):
        """A prefill/decode run has two gated pool_ids, and reconcile has to know which layout a pod follows."""
        core_v1 = FakeCoreV1()
        pods = [_pod(DECODE_POOL_ID, 1), _pod(TRAINER_POOL_ID, 0, 3, node_name="gpu-4", gated=False)]

        asyncio.run(_attached(_two_pool_controller(core_v1), pods).reconcile(_key(DECODE_POOL_ID, 1)))

        assert core_v1.patched == [
            (
                _pod_name(DECODE_POOL_ID, 1),
                pairing_pods.release_patch(
                    node_name="gpu-4", gates=[pairing_pods._GATE_NAME], has_node_selector=False
                ),
            )
        ]

    def test_keys_every_pool_by_its_own_pods(self):
        """key_of runs over one stream of pods, so a pod of either pool_id has to route to itself."""
        controller = _two_pool_controller(FakeCoreV1())

        assert controller.key_of(_pod(DECODE_POOL_ID, 1)) == _key(DECODE_POOL_ID, 1)

    def test_refuses_two_pools_that_want_the_same_trainer_pod(self):
        """Only one inference can hold a node's gpus, and the second would wait on a trainer already taken."""
        with pytest.raises(AssertionError, match="same trainer pod"):
            PairingController(
                config=_config(
                    [
                        _inference_pool(
                            _layout(num_inference_cells=2, num_trainer_cells=1, num_pods_per_trainer_cell=4)
                        ),
                        _inference_pool(
                            _layout(
                                num_inference_cells=2,
                                num_trainer_cells=1,
                                num_pods_per_trainer_cell=4,
                                gpu_offset=8,
                            ),
                            pool_id=DECODE_POOL_ID,
                        ),
                    ]
                ),
                core_v1=FakeCoreV1(),
            )


class TestPairingConfig:
    def test_reads_back_what_the_chart_serialised(self):
        """The chart passes one json blob, and a field lost in transit would pair pods with the wrong trainer."""
        config = PairingConfig(
            namespace="rl",
            release="run",
            trainer_pool_id=TRAINER_POOL_ID,
            inference_pools=[
                InferencePool(
                    pool_id=DECODE_POOL_ID,
                    layout=_layout(
                        num_inference_cells=2, num_trainer_cells=1, num_pods_per_trainer_cell=4, gpu_offset=16
                    ),
                )
            ],
        )

        restored = PairingConfig.model_validate_json(config.model_dump_json())

        assert restored == config

    def test_refuses_a_config_that_names_no_pool(self):
        """A pairing controller with nothing to pair would sit there while every inference pod stays gated."""
        with pytest.raises(pydantic.ValidationError):
            PairingConfig(namespace="rl", release="run", trainer_pool_id=TRAINER_POOL_ID, inference_pools=[])

    def test_refuses_a_layout_the_chart_should_never_render(self):
        """The json crosses a process boundary, so the invariants have to be rechecked on the reading side."""
        payload = (
            '{"namespace": "rl", "release": "run", "trainer_pool_id": "t", "inference_pools": '
            '[{"pool_id": "d", "layout": {"num_inference_cells": 1, "num_trainer_cells": 1, '
            '"num_pods_per_inference_cell": 4, "num_pods_per_trainer_cell": 2, "num_gpus_per_node": 8, '
            '"num_gpus_per_inference_pod": 8, "gpu_offset": 0}}]}'
        )

        with pytest.raises(pydantic.ValidationError, match="cannot fit"):
            PairingConfig.model_validate_json(payload)

    def test_refuses_a_layout_whose_sub_node_pods_overrun_the_trainer(self):
        """The gpu-level fit check has to survive the json boundary too, or the surplus pod waits forever."""
        payload = (
            '{"namespace": "rl", "release": "run", "trainer_pool_id": "t", "inference_pools": '
            '[{"pool_id": "d", "layout": {"num_inference_cells": 5, "num_trainer_cells": 1, '
            '"num_pods_per_inference_cell": 1, "num_pods_per_trainer_cell": 2, "num_gpus_per_node": 8, '
            '"num_gpus_per_inference_pod": 4, "gpu_offset": 0}}]}'
        )

        with pytest.raises(pydantic.ValidationError, match="do not fit in the trainer's"):
            PairingConfig.model_validate_json(payload)


class FakeCoreV1:
    def __init__(self) -> None:
        self.patched: list[tuple[str, list[dict[str, Any]]]] = []

    async def patch_namespaced_pod(self, *, name: str, namespace: str, body: list[dict[str, Any]]) -> None:
        self.patched.append((name, body))


class PairingHarness:
    def __init__(self, *, layout: PairingLayout | None = None) -> None:
        self.core_v1 = FakeCoreV1()
        self.source = FakeSource()
        self.clock = FakeClock()
        pools = [_inference_pool(layout or _layout(num_inference_cells=2, num_trainer_cells=2))]
        self.controller = PairingController(config=_config(pools), core_v1=self.core_v1)
        self.loop = ReconcileLoop(
            source=self.source,
            reconcile=self.controller.reconcile,
            key_map=self.controller.key_of,
            clock=self.clock,
            resync_period=60.0,
        )
        self.controller.set_loop(self.loop)

    @asynccontextmanager
    async def running(self, *pods: Any) -> AsyncIterator[PairingHarness]:
        start_task = asyncio.create_task(self.loop.start())
        await settle()
        self.source.emit(replace_of(*pods))
        await settle()
        await start_task
        try:
            yield self
        finally:
            await self.loop.stop()

    async def upsert(self, pod: Any) -> None:
        self.source.emit(UpsertEvent(key=pod.metadata.name, obj=pod))
        await settle()

    async def delete(self, pod: Any) -> None:
        self.source.emit(DeleteEvent(key=pod.metadata.name, last_obj=pod))
        await settle()

    def patched_names(self) -> list[str]:
        return [name for name, _ in self.core_v1.patched]


class TestEventSequences:
    async def test_releases_the_inference_when_its_trainer_lands_afterwards(self):
        """Inference first: nothing happens until the trainer's own event brings the reconcile back."""
        harness = PairingHarness()

        async with harness.running(_pod(INFERENCE_POOL_ID, 0), _pod(TRAINER_POOL_ID, 0)):
            assert harness.core_v1.patched == []

            await harness.upsert(_pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False))

            assert harness.core_v1.patched == [
                (
                    _pod_name(INFERENCE_POOL_ID, 0),
                    pairing_pods.release_patch(
                        node_name="gpu-3", gates=[pairing_pods._GATE_NAME], has_node_selector=False
                    ),
                ),
            ]

    async def test_releases_the_inference_when_the_trainer_was_there_first(self):
        """Trainer first: the inference's own arrival finds a placed trainer and releases immediately."""
        harness = PairingHarness()

        async with harness.running(_pod(TRAINER_POOL_ID, 1, node_name="gpu-9", gated=False)):
            await harness.upsert(_pod(INFERENCE_POOL_ID, 1))

            assert harness.core_v1.patched == [
                (
                    _pod_name(INFERENCE_POOL_ID, 1),
                    pairing_pods.release_patch(
                        node_name="gpu-9", gates=[pairing_pods._GATE_NAME], has_node_selector=False
                    ),
                ),
            ]

    async def test_leaves_the_inference_gated_while_the_trainer_has_no_node(self):
        """A trainer pod exists long before it is scheduled, and its node is the only thing worth waiting on."""
        harness = PairingHarness()

        async with harness.running(_pod(INFERENCE_POOL_ID, 0)):
            await harness.upsert(_pod(TRAINER_POOL_ID, 0))

            assert harness.core_v1.patched == []

    async def test_does_not_patch_an_inference_that_is_already_released(self):
        """A controller restart relists a world where the gate is gone, which is a terminal state."""
        harness = PairingHarness()

        async with harness.running(
            _pod(INFERENCE_POOL_ID, 0, node_name="gpu-3", gated=False),
            _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False),
        ):
            assert harness.core_v1.patched == []

    async def test_does_not_patch_twice_when_the_release_is_observed_back(self):
        """The patch produces its own watch event, and re-entering must not repeat the write."""
        harness = PairingHarness()

        async with harness.running(
            _pod(INFERENCE_POOL_ID, 0), _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False)
        ):
            await harness.upsert(_pod(INFERENCE_POOL_ID, 0, node_name="gpu-3", gated=False))

            assert harness.patched_names() == [_pod_name(INFERENCE_POOL_ID, 0)]

    async def test_re_enters_safely_after_a_crash_before_the_patch_landed(self):
        """A controller that died before patching sees the same gated pod again and finishes the job."""
        crashed = PairingHarness()
        async with crashed.running(
            _pod(INFERENCE_POOL_ID, 0), _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False)
        ):
            pass

        restarted = PairingHarness()
        async with restarted.running(
            _pod(INFERENCE_POOL_ID, 0), _pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False)
        ):
            assert restarted.patched_names() == [_pod_name(INFERENCE_POOL_ID, 0)]

    async def test_resync_releases_an_inference_whose_trainer_event_was_missed(self):
        """The backstop for a lost event: the next resync reconciles every key from the store again."""
        harness = PairingHarness()
        trainer = _pod(TRAINER_POOL_ID, 0, gated=False)

        async with harness.running(_pod(INFERENCE_POOL_ID, 0), trainer):
            assert harness.core_v1.patched == []

            trainer.spec.node_name = "gpu-3"
            await harness.clock.elapse(60.0)
            await settle()

            assert harness.patched_names() == [_pod_name(INFERENCE_POOL_ID, 0)]

    async def test_a_deleted_inference_is_never_patched(self):
        """Scale-down deletes gated pods, and a queued reconcile must not write to a name that is gone."""
        harness = PairingHarness()
        inference = _pod(INFERENCE_POOL_ID, 0)

        async with harness.running(inference, _pod(TRAINER_POOL_ID, 0)):
            await harness.delete(inference)
            await harness.upsert(_pod(TRAINER_POOL_ID, 0, node_name="gpu-3", gated=False))

            assert harness.core_v1.patched == []
