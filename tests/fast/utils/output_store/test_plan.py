"""Seg / Segments / StorePlan: edits read only refs; materialize reads objects given as plain dicts."""

from argparse import Namespace
from typing import NamedTuple

import numpy as np
import pytest
import torch

from miles.utils.output_store.plan import (
    FIELDS,
    MASK_LENGTHS,
    MASK_LOGPROBS,
    MASK_TOKEN_IDS,
    IndexerTopkSegments,
    RoutedExpertsSegments,
    SamplingMaskSegments,
    Seg,
    StorePlan,
    StorePlanError,
    combine_plans,
)

R3, INDEXER, MASK = (cls.name for cls in FIELDS)
ALL = {R3, INDEXER, MASK}


class Spec(NamedTuple):
    name: str
    dtype: str
    shape: tuple[int, ...]


class Ref(NamedTuple):
    key: str
    fields: tuple[Spec, ...]


def ref(key: str, *, seq_rows: int | None = None, mask_rows: int | None = None) -> Ref:
    specs = []
    if seq_rows is not None:
        specs.append(Spec(R3, "int32", (seq_rows, 2, 4)))
        specs.append(Spec(INDEXER, "int32", (seq_rows, 2, 8)))
    if mask_rows is not None:
        specs.append(Spec(MASK_LENGTHS, "int32", (mask_rows,)))
        specs.append(Spec(MASK_TOKEN_IDS, "int32", (3 * mask_rows,)))
        specs.append(Spec(MASK_LOGPROBS, "float32", (3 * mask_rows,)))
    return Ref(key, tuple(specs))


def response(key: str, *, prompt: int, output: int, requested=ALL, **kwargs) -> StorePlan:
    return StorePlan.from_response(
        ref(key, seq_rows=prompt + output - 1, mask_rows=output),
        requested=requested,
        num_tokens=prompt + output,
        response_length=output,
        **kwargs,
    )


def segs(plan: StorePlan, name: str) -> list[tuple[str | None, int, int]]:
    return [(None if seg.ref is None else seg.ref.key, seg.start, seg.end) for seg in plan.field(name).segs]


class TestFromResponse:
    def test_requested_fields_take_every_row_of_the_object(self):
        plan = response("a", prompt=3, output=2)
        assert segs(plan, R3) == [("a", 0, 4)] and segs(plan, INDEXER) == [("a", 0, 4)]
        assert segs(plan, MASK) == [("a", 0, 2)]
        assert [r.key for r in plan.owned_refs] == ["a"]
        assert isinstance(plan.field(R3), RoutedExpertsSegments)
        assert isinstance(plan.field(INDEXER), IndexerTopkSegments)
        assert isinstance(plan.field(MASK), SamplingMaskSegments)

    def test_unrequested_fields_are_absent_and_requested_ones_must_be_stored(self):
        plan = response("a", prompt=3, output=2, requested={MASK})
        assert list(plan.fields) == [MASK]
        with pytest.raises(StorePlanError, match="asked for routed_experts"):
            StorePlan.from_response(ref("a", mask_rows=2), requested={R3}, num_tokens=5, response_length=2)

    def test_a_no_output_abort_gets_an_empty_mask_without_a_ref(self):
        plan = StorePlan.from_response(None, requested={MASK}, num_tokens=3, response_length=0, no_output_abort=True)
        assert plan.field(MASK) == SamplingMaskSegments() and plan.owned_refs == ()
        with pytest.raises(StorePlanError, match="asked for sampling_mask"):
            StorePlan.from_response(None, requested={MASK}, num_tokens=3, response_length=0)

    def test_row_counts_are_checked_against_the_sample(self):
        with pytest.raises(StorePlanError, match="has 4 rows; expected 5"):
            StorePlan.from_response(ref("a", seq_rows=4), requested={R3}, num_tokens=6, response_length=2)
        with pytest.raises(StorePlanError, match="has 3 rows; expected 2"):
            StorePlan.from_response(ref("a", mask_rows=3), requested={MASK}, num_tokens=5, response_length=2)

    def test_legacy_stop_edge_row_is_dropped_from_forward_fields_only(self):
        plan = StorePlan.from_response(
            ref("a", seq_rows=5, mask_rows=2), requested=ALL, num_tokens=5, response_length=2, allow_extra_seq_row=True
        )
        assert segs(plan, R3) == [("a", 0, 4)] and segs(plan, MASK) == [("a", 0, 2)]
        with pytest.raises(StorePlanError, match="has 3 rows; expected 2"):
            StorePlan.from_response(
                ref("a", mask_rows=3), requested={MASK}, num_tokens=5, response_length=2, allow_extra_seq_row=True
            )


class TestEdits:
    def test_next_response_replaces_forward_rows_and_appends_mask_rows(self):
        first = response("a", prompt=3, output=2)
        second = response("b", prompt=5, output=3)  # its forward covered the whole sequence again

        plan = first.append_response(second)

        assert segs(plan, R3) == [("b", 0, 7)] and segs(plan, MASK) == [("a", 0, 2), ("b", 0, 3)]
        assert [r.key for r in plan.owned_refs] == ["a", "b"]  # the replaced object stays owned

    def test_session_additional_r3_appends_from_the_covered_row(self):
        rows = RoutedExpertsSegments()
        rows = rows.append(RoutedExpertsSegments((Seg(ref("a"), 0, 4),)), seq_start=0)
        rows = rows.append(RoutedExpertsSegments((Seg(ref("b"), 0, 6),)), seq_start=4)
        assert [(s.ref.key, s.start, s.end) for s in rows.segs] == [("a", 0, 4), ("b", 0, 6)]
        assert rows.num_rows() == 10

    def test_a_tool_observation_adds_forced_mask_rows_only(self):
        plan = response("a", prompt=3, output=2).insert_tokens(3)
        assert segs(plan, R3) == [("a", 0, 4)] and segs(plan, MASK) == [("a", 0, 2), (None, 0, 3)]
        assert plan.insert_tokens(0) == plan

    def test_session_merge_puts_forced_rows_between_turns_and_keeps_the_later_forward(self):
        turn1 = response("a", prompt=3, output=2)
        turn2 = response("b", prompt=8, output=1)  # 3 prompt + 2 output + 3 observation tokens

        plan = turn1.merge_turn(turn2, observation_tokens=3)

        assert segs(plan, R3) == [("b", 0, 8)]
        assert segs(plan, MASK) == [("a", 0, 2), (None, 0, 3), ("b", 0, 1)]
        plan.validate(num_tokens=9, response_length=6)

    def test_strip_tail_cuts_across_segments_and_keeps_every_ref(self):
        plan = response("a", prompt=3, output=2).merge_turn(response("b", prompt=8, output=1), observation_tokens=3)

        plan = plan.strip_tail(2)

        assert segs(plan, R3) == [("b", 0, 6)] and segs(plan, MASK) == [("a", 0, 2), (None, 0, 2)]
        assert [r.key for r in plan.owned_refs] == ["a", "b"]
        plan.validate(num_tokens=7, response_length=4)

    def test_adopted_refs_are_owned_but_unused(self):
        plan = response("a", prompt=3, output=2).adopt_refs([ref("x"), ref("a")])
        assert [r.key for r in plan.owned_refs] == ["a", "x"] and segs(plan, R3) == [("a", 0, 4)]

    def test_strict_field_rule(self):
        with_mask = response("a", prompt=3, output=2)
        without = response("b", prompt=5, output=1, requested={R3, INDEXER})
        with pytest.raises(StorePlanError, match="cannot combine"):
            with_mask.append_response(without)
        assert combine_plans(None, None, lambda a, b: a) is None
        with pytest.raises(StorePlanError, match="with output store data and one without"):
            combine_plans(with_mask, None, lambda a, b: a)


class TestValidate:
    def test_rows_must_match_the_sample(self):
        plan = response("a", prompt=3, output=2)
        plan.validate(num_tokens=5, response_length=2)
        with pytest.raises(StorePlanError, match="routed_experts has 4 rows; expected 5"):
            plan.validate(num_tokens=6, response_length=2)
        with pytest.raises(StorePlanError, match="sampling_mask has 2 rows; expected 3"):
            plan.validate(num_tokens=5, response_length=3)


class TestTable:
    def test_round_trip_keeps_fields_forced_rows_empty_fields_and_unused_refs(self):
        plan = response("a", prompt=3, output=2).merge_turn(response("b", prompt=8, output=1), observation_tokens=3)
        plan = plan.adopt_refs([ref("x")]).with_field(INDEXER, IndexerTopkSegments())

        table, refs = plan.to_table()
        back = StorePlan.from_table(table, refs)

        assert table.dtype == np.int32 and table.shape[1] == 4
        assert table.tolist() == [[0, 1, 0, 8], [1, -1, 0, 0], [2, 0, 0, 2], [2, -1, 0, 3], [2, 1, 0, 1]]
        assert back == plan and [r.key for r in refs] == ["a", "b", "x"]

    def test_a_plan_without_fields_is_an_empty_table(self):
        plan = StorePlan(fields={}, owned_refs=(ref("a"),))
        table, refs = plan.to_table()
        assert table.shape == (0, 4) and StorePlan.from_table(table, refs) == plan


# ============================== materialize ==============================


def args(**overrides) -> Namespace:
    values = dict(
        rollout_sampling_logprobs_mode="selected", rollout_top_logprobs_num=0, rollout_indexer_topk_num_streams=None
    )
    return Namespace(**(values | overrides))


def r3(rows: int, seed: int, streams: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).integers(1, 64, size=(rows, streams, 4), dtype=np.int32)


def mask_arrays(supports: list[list[int]], logprobs: list[list[float]] | None = None) -> dict[str, np.ndarray]:
    arrays = {
        MASK_LENGTHS: np.asarray([len(s) for s in supports], dtype=np.int32),
        MASK_TOKEN_IDS: np.asarray([t for s in supports for t in s], dtype=np.int32),
    }
    if logprobs is not None:
        arrays[MASK_LOGPROBS] = np.asarray([v for r in logprobs for v in r], dtype=np.float32)
    return arrays


def rows_of(name: str, *segs: tuple[str, int, int]):
    cls = {cls.name: cls for cls in FIELDS}[name]
    return cls(tuple(Seg(None if key is None else ref(key), start, end) for key, start, end in segs))


class TestMaterializeForward:
    def test_rows_are_concatenated_across_objects_in_plan_order(self):
        a, b = r3(4, 1), r3(6, 2)
        objects = {"a": {R3: a}, "b": {R3: b}}

        columns = rows_of(R3, ("a", 0, 4), ("b", 0, 6)).materialize(objects, response_tokens=np.empty(0), args=args())

        np.testing.assert_array_equal(columns["rollout_routed_experts"], np.concatenate([a, b]))
        assert columns["rollout_routed_experts"].dtype == np.int32
        partial = rows_of(R3, ("b", 1, 3)).materialize(objects, response_tokens=np.empty(0), args=args())
        np.testing.assert_array_equal(partial["rollout_routed_experts"], b[1:3])

    def test_all_zero_routed_experts_are_rejected(self):
        objects = {"a": {R3: np.zeros((3, 2, 4), dtype=np.int32)}}
        with pytest.raises(StorePlanError, match="all zeros"):
            rows_of(R3, ("a", 0, 3)).materialize(objects, response_tokens=np.empty(0), args=args())

    def test_indexer_streams_must_match_the_model(self):
        objects = {"a": {INDEXER: r3(3, 3, streams=2)}}
        columns = rows_of(INDEXER, ("a", 0, 3)).materialize(objects, response_tokens=np.empty(0), args=args())
        assert columns["rollout_indexer_topk"].shape == (3, 2, 4)
        with pytest.raises(StorePlanError, match="2 streams but the model has 4"):
            rows_of(INDEXER, ("a", 0, 3)).materialize(
                objects, response_tokens=np.empty(0), args=args(rollout_indexer_topk_num_streams=4)
            )


class TestMaterializeMask:
    def test_forced_rows_hold_their_token_and_columns_match_the_inline_layout(self):
        objects = {"a": mask_arrays([[5, 6], [7, 8, 9]]), "b": mask_arrays([[11], [12, 13]])}
        # Sampled 6, 8; forced 40, 41; then row 1 of object b (sampled 13).
        plan = rows_of(MASK, ("a", 0, 2), (None, 0, 2), ("b", 1, 2))

        columns = plan.materialize(objects, response_tokens=np.asarray([6, 8, 40, 41, 13]), args=args())

        assert columns["rollout_sampling_mask_ids"].tolist() == [5, 6, 7, 8, 9, 40, 41, 12, 13]
        assert columns["rollout_sampling_mask_offsets"].tolist() == [0, 2, 5, 6, 7, 9]
        assert columns["rollout_sampling_mask_ids"].dtype == torch.int32
        assert columns["rollout_sampling_mask_offsets"].dtype == torch.int64
        assert "rollout_topk_token_ids" not in columns

    def test_support_mode_builds_the_top_k_candidates_with_forced_rows_empty(self):
        objects = {"a": mask_arrays([[5, 6], [7, 8, 9]], [[-0.5, -1.0], [-0.1, -0.2, -0.3]])}
        plan = rows_of(MASK, ("a", 0, 2), (None, 0, 1))
        support = args(rollout_sampling_logprobs_mode="support", rollout_top_logprobs_num=3)

        columns = plan.materialize(objects, response_tokens=np.asarray([6, 8, 40]), args=support)

        assert columns["rollout_sampling_mask_ids"].tolist() == [5, 6, 7, 8, 9, 40]
        assert columns["rollout_topk_token_ids"].tolist() == [[5, 6, -1], [7, 8, 9], [-1, -1, -1]]
        np.testing.assert_array_equal(
            columns["rollout_topk_log_probs"][1], np.asarray([-0.1, -0.2, -0.3], dtype=np.float32)
        )
        assert np.isneginf(columns["rollout_topk_log_probs"][2]).all()
        assert columns["rollout_topk_token_ids"].dtype == np.int32

    def test_a_mask_of_forced_rows_only_or_no_rows_needs_no_object(self):
        forced = rows_of(MASK, (None, 0, 2)).materialize({}, response_tokens=np.asarray([3, 4]), args=args())
        assert forced["rollout_sampling_mask_ids"].tolist() == [3, 4]
        assert forced["rollout_sampling_mask_offsets"].tolist() == [0, 1, 2]
        empty = SamplingMaskSegments().materialize({}, response_tokens=np.empty(0, dtype=np.int64), args=args())
        assert empty["rollout_sampling_mask_ids"].tolist() == [] and empty[
            "rollout_sampling_mask_offsets"
        ].tolist() == [0]

    def test_the_sampled_token_must_be_in_its_support(self):
        objects = {"a": mask_arrays([[5, 6], [7, 7]], [[-0.5, -1.0], [-0.1, -0.2]])}
        plan = rows_of(MASK, ("a", 0, 2))
        with pytest.raises(StorePlanError, match="absent from its sampling support"):
            plan.materialize(objects, response_tokens=np.asarray([9, 7]), args=args())
        with pytest.raises(StorePlanError, match="exactly once"):
            plan.materialize(
                objects, response_tokens=np.asarray([6, 7]), args=args(rollout_sampling_logprobs_mode="support")
            )
        with pytest.raises(StorePlanError, match="at least one token"):
            rows_of(MASK, ("b", 0, 1)).materialize(
                {"b": mask_arrays([[]])}, response_tokens=np.asarray([1]), args=args()
            )

    def test_a_support_wider_than_k_is_rejected(self):
        objects = {"a": mask_arrays([[5, 6, 7]], [[-0.1, -0.2, -0.3]])}
        with pytest.raises(StorePlanError, match="exceeds --rollout-top-logprobs-num"):
            rows_of(MASK, ("a", 0, 1)).materialize(
                objects,
                response_tokens=np.asarray([6]),
                args=args(rollout_sampling_logprobs_mode="support", rollout_top_logprobs_num=2),
            )


def test_plan_materialize_merges_every_fields_columns():
    plan = StorePlan.from_response(ref("a", seq_rows=4, mask_rows=2), requested=ALL, num_tokens=5, response_length=2)
    objects = {"a": {R3: r3(4, 1), INDEXER: r3(4, 2), **mask_arrays([[5, 6], [8]])}}

    columns = plan.materialize(objects, response_tokens=np.asarray([6, 8]), args=args())

    assert set(columns) == {
        "rollout_routed_experts",
        "rollout_indexer_topk",
        "rollout_sampling_mask_ids",
        "rollout_sampling_mask_offsets",
    }
