"""Seg / Segments / StorePlan: edits read only refs; materialize reads objects given as plain dicts."""

import json
import sys
import types
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import pytest
import torch

from miles.utils.output_store.plan import (
    FIELDS,
    MASK_LENGTHS,
    MASK_SUPPORT_LOGPROBS,
    MASK_TOKEN_IDS,
    IndexerTopkSegments,
    RoutedExpertsSegments,
    SamplingMaskSegments,
    Seg,
    StorePlan,
    StorePlanError,
    combine_plans,
    store_reads,
    store_ref_from_dict,
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

    def to_builtins(self) -> dict:
        # As StoreRef.to_builtins: tuple-typed fields stay tuples.
        return {"key": self.key, "fields": tuple(spec._asdict() for spec in self.fields)}

    @classmethod
    def from_builtins(cls, value: dict) -> "Ref":
        return cls(value["key"], tuple(Spec(f["name"], f["dtype"], tuple(f["shape"])) for f in value["fields"]))


def fake_store_refs(monkeypatch) -> None:
    """CPU CI has no radix_transfer_plugins: refs read back as ``Ref`` wherever they are decoded."""
    module = types.ModuleType("radix_transfer_plugins")
    module.StoreRef = Ref
    monkeypatch.setitem(sys.modules, "radix_transfer_plugins", module)


def ref(key: str, *, seq_rows: int | None = None, mask_rows: int | None = None) -> Ref:
    specs = []
    if seq_rows is not None:
        specs.append(Spec(R3, "int32", (seq_rows, 2, 4)))
        specs.append(Spec(INDEXER, "int32", (seq_rows, 2, 8)))
    if mask_rows is not None:
        specs.append(Spec(MASK_LENGTHS, "int32", (mask_rows,)))
        specs.append(Spec(MASK_TOKEN_IDS, "int32", (3 * mask_rows,)))
        specs.append(Spec(MASK_SUPPORT_LOGPROBS, "float32", (3 * mask_rows,)))
    return Ref(key, tuple(specs))


def request(names=ALL, **extra) -> dict:
    """The request fields that ask SGLang for ``names``."""
    return {cls.request_key: True for cls in FIELDS if cls.name in names} | extra


def response(key: str, *, prompt: int, output: int, requested=ALL, **kwargs) -> StorePlan:
    return StorePlan.from_response(
        ref(key, seq_rows=prompt + output - 1, mask_rows=output),
        request=request(requested),
        num_tokens=prompt + output,
        response_length=output,
        **kwargs,
    )


def segs(plan: StorePlan, name: str) -> list[tuple[str | None, int, int]]:
    return [(None if seg.ref is None else seg.ref.key, seg.start, seg.end) for seg in plan.fields[name].segs]


class TestFromResponse:
    def test_requested_fields_take_every_row_of_the_object(self):
        plan = response("a", prompt=3, output=2)
        assert segs(plan, R3) == [("a", 0, 4)] and segs(plan, INDEXER) == [("a", 0, 4)]
        assert segs(plan, MASK) == [("a", 0, 2)]
        assert [r.key for r in plan.owned_refs] == ["a"]
        assert isinstance(plan.fields[R3], RoutedExpertsSegments)
        assert isinstance(plan.fields[INDEXER], IndexerTopkSegments)
        assert isinstance(plan.fields[MASK], SamplingMaskSegments)

    def test_unrequested_fields_are_absent_and_requested_ones_must_be_stored(self):
        plan = response("a", prompt=3, output=2, requested={MASK})
        assert list(plan.fields) == [MASK]
        with pytest.raises(StorePlanError, match="asked for routed_experts"):
            StorePlan.from_response(ref("a", mask_rows=2), request=request({R3}), num_tokens=5, response_length=2)

    def test_a_no_output_abort_may_store_nothing(self):
        # Aborted in SGLang's waiting queue: no forward ran, and no ref came back.
        plan = StorePlan.from_response(None, request=request(), num_tokens=3, response_length=0, no_output_abort=True)
        assert plan.fields == {
            R3: RoutedExpertsSegments(),
            INDEXER: IndexerTopkSegments(),
            MASK: SamplingMaskSegments(),
        }
        assert plan.owned_refs == ()
        # Aborted after its prefill: the forward rows are stored, the mask is not.
        prefilled = StorePlan.from_response(
            ref("a", seq_rows=2), request=request({R3, MASK}), num_tokens=3, response_length=0, no_output_abort=True
        )
        assert segs(prefilled, R3) == [("a", 0, 2)] and segs(prefilled, MASK) == []
        with pytest.raises(StorePlanError, match="asked for routed_experts"):
            StorePlan.from_response(None, request=request(), num_tokens=3, response_length=0)
        with pytest.raises(StorePlanError, match="asked for routed_experts"):
            StorePlan.from_response(
                ref("a", mask_rows=0),
                request=request({R3, MASK}),
                num_tokens=3,
                response_length=0,
                no_output_abort=True,
            )

    def test_row_counts_are_checked_against_the_sample(self):
        with pytest.raises(StorePlanError, match="has 4 rows; expected 5"):
            StorePlan.from_response(ref("a", seq_rows=4), request=request({R3}), num_tokens=6, response_length=2)
        with pytest.raises(StorePlanError, match="has 3 rows; expected 2"):
            StorePlan.from_response(ref("a", mask_rows=3), request=request({MASK}), num_tokens=5, response_length=2)

    def test_an_incremental_forward_covers_only_its_new_rows(self):
        # The object of routed experts starts at sequence position 4; indexer top-k always covers all of it.
        b = Ref("b", (Spec(R3, "int32", (6, 2, 4)), Spec(INDEXER, "int32", (10, 2, 8))))
        plan = StorePlan.from_response(
            b, request=request({R3, INDEXER}, routed_experts_start_len=4), num_tokens=11, response_length=1
        )
        assert segs(plan, R3) == [(None, 0, 4), ("b", 0, 6)] and segs(plan, INDEXER) == [("b", 0, 10)]
        plan.validate(num_tokens=11, response_length=1)
        with pytest.raises(StorePlanError, match="has 6 rows; expected 7"):
            StorePlan.from_response(
                b, request=request({R3}, routed_experts_start_len=3), num_tokens=11, response_length=1
            )


class TestEdits:
    def test_next_response_replaces_forward_rows_and_appends_mask_rows(self):
        first = response("a", prompt=3, output=2)
        second = response("b", prompt=5, output=3)  # its forward covered the whole sequence again

        plan = first.append_response(second)

        assert segs(plan, R3) == [("b", 0, 7)] and segs(plan, MASK) == [("a", 0, 2), ("b", 0, 3)]
        assert [r.key for r in plan.owned_refs] == ["a", "b"]  # the replaced object stays owned

    def test_an_incremental_forward_keeps_the_rows_before_it(self):
        turn1 = response("a", prompt=3, output=2, requested={R3, MASK})  # forward rows 0..3
        # 3 observation tokens, then a turn over 8 input tokens and 1 output token (forward rows 0..7),
        # whose object stores rows 4..7 only.
        b = Ref(
            "b", (Spec(R3, "int32", (4, 2, 4)), Spec(MASK_LENGTHS, "int32", (1,)), Spec(MASK_TOKEN_IDS, "int32", (3,)))
        )
        turn2 = StorePlan.from_response(
            b, request=request({R3, MASK}, routed_experts_start_len=4), num_tokens=9, response_length=1
        )

        plan = turn1.merge_turn(turn2, observation_tokens=3)

        assert segs(plan, R3) == [("a", 0, 4), ("b", 0, 4)]
        assert segs(plan, MASK) == [("a", 0, 2), (None, 0, 3), ("b", 0, 1)]
        plan.validate(num_tokens=9, response_length=6)

    def test_an_incremental_forward_needs_every_row_before_it(self):
        # Rows 0..3 came before, but the next object starts at position 5: a row is missing.
        before = response("a", prompt=3, output=2, requested={R3})
        later = StorePlan.from_response(
            Ref("b", (Spec(R3, "int32", (3, 2, 4)),)),
            request=request({R3}, routed_experts_start_len=5),
            num_tokens=9,
            response_length=1,
        )
        with pytest.raises(StorePlanError, match="continues from row 5, but only 4 rows precede it"):
            before.append_response(later)
        # Nothing precedes the first response of a sample.
        with pytest.raises(StorePlanError, match="continues from row 5, but only 0 rows precede it"):
            RoutedExpertsSegments().append(later.fields[R3])

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


class TestReads:
    def test_only_the_arrays_the_rows_come_from_are_read_once_per_object(self):
        plan = response("a", prompt=3, output=2).merge_turn(response("b", prompt=8, output=1), observation_tokens=3)
        plan = plan.adopt_refs([ref("x", seq_rows=4)])

        reads = store_reads([plan, plan], support_topk=2)

        # "a"'s forward rows were replaced by "b"'s; "x" is owned but unused.
        assert {key: names for key, (_, names) in reads.items()} == {
            "b": [R3, INDEXER, MASK_LENGTHS, MASK_TOKEN_IDS, MASK_SUPPORT_LOGPROBS],
            "a": [MASK_LENGTHS, MASK_TOKEN_IDS, MASK_SUPPORT_LOGPROBS],
        }
        assert reads["b"][0] is plan.fields[R3].segs[0].ref

    def test_plans_sharing_an_object_read_the_union_of_their_arrays(self):
        merged = response("a", prompt=3, output=2).merge_turn(response("b", prompt=8, output=1), observation_tokens=3)
        # Another sample whose forward rows still come from "a".
        alone = response("a", prompt=3, output=2)

        reads = store_reads([merged, alone], support_topk=2)

        assert reads["a"][1] == [MASK_LENGTHS, MASK_TOKEN_IDS, MASK_SUPPORT_LOGPROBS, R3, INDEXER]

    def test_support_log_probs_are_read_only_in_support_mode(self):
        # A selected-mode request: SGLang stores no support log-probs.
        selected = Ref("s", (Spec(MASK_LENGTHS, "int32", (2,)), Spec(MASK_TOKEN_IDS, "int32", (3,))))
        plan = StorePlan.from_response(selected, request=request({MASK}), num_tokens=5, response_length=2)

        assert store_reads([plan], support_topk=0)["s"][1] == [MASK_LENGTHS, MASK_TOKEN_IDS]
        # Asked for though the ref lacks them, so the read fails instead of materialize.
        assert store_reads([plan], support_topk=2)["s"][1] == [MASK_LENGTHS, MASK_TOKEN_IDS, MASK_SUPPORT_LOGPROBS]
        assert store_reads([StorePlan(fields={}, owned_refs=(ref("x"),))], support_topk=2) == {}


class TestValidate:
    def test_rows_must_match_the_sample(self):
        plan = response("a", prompt=3, output=2)
        plan.validate(num_tokens=5, response_length=2)
        with pytest.raises(StorePlanError, match="routed_experts has 4 rows; expected 5"):
            plan.validate(num_tokens=6, response_length=2)
        with pytest.raises(StorePlanError, match="sampling_mask has 2 rows; expected 3"):
            plan.validate(num_tokens=5, response_length=3)


class TestDict:
    @pytest.fixture(autouse=True)
    def _fake_refs(self, monkeypatch):
        fake_store_refs(monkeypatch)

    def test_round_trip_keeps_fields_forced_rows_empty_fields_and_unused_refs(self):
        plan = response("a", prompt=3, output=2).merge_turn(response("b", prompt=8, output=1), observation_tokens=3)
        plan = plan.adopt_refs([ref("x")])
        plan = replace(plan, fields={**plan.fields, INDEXER: IndexerTopkSegments()})

        value = plan.to_dict()

        assert value["fields"] == {
            R3: [["b", 0, 8]],
            INDEXER: [],
            MASK: [["a", 0, 2], [None, 0, 3], ["b", 0, 1]],
        }
        assert [r["key"] for r in value["refs"]] == ["a", "b", "x"]
        assert StorePlan.from_dict(value) == plan

    def test_the_dict_holds_json_types_only(self):
        # The train dict's msgpack codec packs with strict_types, which rejects tuples and numpy scalars.
        value = response("a", prompt=3, output=2).insert_tokens(2).to_dict()
        assert json.loads(json.dumps(value)) == value  # a tuple would come back as an unequal list

    def test_a_plan_without_fields_round_trips(self):
        plan = StorePlan(fields={}, owned_refs=(ref("a"),))
        assert StorePlan.from_dict(plan.to_dict()) == plan


def test_no_ref_stays_none():
    assert store_ref_from_dict(None) is None


def test_store_refs_round_trip():
    rtp = pytest.importorskip("radix_transfer_plugins")
    spec = rtp.FieldSpec(name=R3, dtype="int32", shape=(4, 2, 4), codec="raw", nbytes=128)
    plan = StorePlan.from_response(
        rtp.StoreRef(backend="mooncake", key="k", fields=(spec,)),
        request=request({R3}),
        num_tokens=5,
        response_length=2,
    )
    value = plan.to_dict()
    assert json.loads(json.dumps(value)) == value
    assert StorePlan.from_dict(value) == plan


# ============================== materialize ==============================


def opts(*, support_topk: int = 0, indexer_streams: int | None = None) -> dict:
    return dict(support_topk=support_topk, indexer_streams=indexer_streams)


def r3(rows: int, seed: int, streams: int = 2) -> np.ndarray:
    return np.random.default_rng(seed).integers(1, 64, size=(rows, streams, 4), dtype=np.int32)


def mask_arrays(supports: list[list[int]], logprobs: list[list[float]] | None = None) -> dict[str, np.ndarray]:
    arrays = {
        MASK_LENGTHS: np.asarray([len(s) for s in supports], dtype=np.int32),
        MASK_TOKEN_IDS: np.asarray([t for s in supports for t in s], dtype=np.int32),
    }
    if logprobs is not None:
        arrays[MASK_SUPPORT_LOGPROBS] = np.asarray([v for r in logprobs for v in r], dtype=np.float32)
    return arrays


def rows_of(name: str, *segs: tuple[str, int, int]):
    cls = {cls.name: cls for cls in FIELDS}[name]
    return cls(tuple(Seg(None if key is None else ref(key), start, end) for key, start, end in segs))


class TestMaterializeForward:
    def test_rows_are_concatenated_across_objects_in_plan_order(self):
        a, b = r3(4, 1), r3(6, 2)
        objects = {"a": {R3: a}, "b": {R3: b}}

        columns = rows_of(R3, ("a", 0, 4), ("b", 0, 6)).materialize(objects, response_tokens=np.empty(0), **opts())

        np.testing.assert_array_equal(columns["rollout_routed_experts"], np.concatenate([a, b]))
        assert columns["rollout_routed_experts"].dtype == np.int32
        partial = rows_of(R3, ("b", 1, 3)).materialize(objects, response_tokens=np.empty(0), **opts())
        np.testing.assert_array_equal(partial["rollout_routed_experts"], b[1:3])

    def test_all_zero_routed_experts_are_rejected_per_object(self):
        objects = {"a": {R3: r3(4, 1)}, "b": {R3: np.zeros((3, 2, 4), dtype=np.int32)}}
        with pytest.raises(StorePlanError, match="all zeros"):
            rows_of(R3, ("a", 0, 4), ("b", 0, 3)).materialize(objects, response_tokens=np.empty(0), **opts())
        # Expert 0 is valid: only a whole object of zeros is rejected, not the zero rows a plan keeps.
        objects["b"][R3] = np.concatenate([np.zeros((2, 2, 4), dtype=np.int32), r3(1, 2)])
        rows_of(R3, ("a", 0, 4), ("b", 0, 2)).materialize(objects, response_tokens=np.empty(0), **opts())

    def test_indexer_streams_must_match_the_model(self):
        objects = {"a": {INDEXER: r3(3, 3, streams=2)}}
        columns = rows_of(INDEXER, ("a", 0, 3)).materialize(objects, response_tokens=np.empty(0), **opts())
        assert columns["rollout_indexer_topk"].shape == (3, 2, 4)
        with pytest.raises(StorePlanError, match="2 streams but the model has 4"):
            rows_of(INDEXER, ("a", 0, 3)).materialize(objects, response_tokens=np.empty(0), **opts(indexer_streams=4))


class TestMaterializeMask:
    def test_forced_rows_hold_their_token_and_columns_match_the_inline_layout(self):
        objects = {"a": mask_arrays([[5, 6], [7, 8, 9]]), "b": mask_arrays([[11], [12, 13]])}
        # Sampled 6, 8; forced 40, 41; then row 1 of object b (sampled 13).
        plan = rows_of(MASK, ("a", 0, 2), (None, 0, 2), ("b", 1, 2))

        columns = plan.materialize(objects, response_tokens=np.asarray([6, 8, 40, 41, 13]), **opts())

        assert columns["rollout_sampling_mask_ids"].tolist() == [5, 6, 7, 8, 9, 40, 41, 12, 13]
        assert columns["rollout_sampling_mask_offsets"].tolist() == [0, 2, 5, 6, 7, 9]
        assert columns["rollout_sampling_mask_ids"].dtype == torch.int32
        assert columns["rollout_sampling_mask_offsets"].dtype == torch.int64
        assert "rollout_topk_token_ids" not in columns

    def test_support_mode_builds_the_top_k_candidates_with_forced_rows_empty(self):
        objects = {"a": mask_arrays([[5, 6], [7, 8, 9]], [[-0.5, -1.0], [-0.1, -0.2, -0.3]])}
        plan = rows_of(MASK, ("a", 0, 2), (None, 0, 1))
        support = opts(support_topk=3)

        columns = plan.materialize(objects, response_tokens=np.asarray([6, 8, 40]), **support)

        assert columns["rollout_sampling_mask_ids"].tolist() == [5, 6, 7, 8, 9, 40]
        assert columns["rollout_topk_token_ids"].tolist() == [[5, 6, -1], [7, 8, 9], [-1, -1, -1]]
        np.testing.assert_array_equal(
            columns["rollout_topk_log_probs"][1], np.asarray([-0.1, -0.2, -0.3], dtype=np.float32)
        )
        assert np.isneginf(columns["rollout_topk_log_probs"][2]).all()
        assert columns["rollout_topk_token_ids"].dtype == np.int32

    def test_a_mask_of_forced_rows_only_or_no_rows_needs_no_object(self):
        forced = rows_of(MASK, (None, 0, 2)).materialize({}, response_tokens=np.asarray([3, 4]), **opts())
        assert forced["rollout_sampling_mask_ids"].tolist() == [3, 4]
        assert forced["rollout_sampling_mask_offsets"].tolist() == [0, 1, 2]
        empty = SamplingMaskSegments().materialize({}, response_tokens=np.empty(0, dtype=np.int64), **opts())
        assert empty["rollout_sampling_mask_ids"].tolist() == [] and empty[
            "rollout_sampling_mask_offsets"
        ].tolist() == [0]

    def test_without_support_log_probs_the_mask_needs_none_stored(self):
        # Support launch mode without top-k collection sends selected-mode requests: no log-probs stored.
        objects = {"a": mask_arrays([[5, 6], [7, 6, 6]])}
        columns = rows_of(MASK, ("a", 0, 2)).materialize(objects, response_tokens=np.asarray([6, 6]), **opts())
        assert columns["rollout_sampling_mask_ids"].tolist() == [5, 6, 7, 6, 6]
        assert "rollout_topk_token_ids" not in columns

    def test_the_sampled_token_must_be_in_its_support(self):
        objects = {"a": mask_arrays([[5, 6], [7, 7]], [[-0.5, -1.0], [-0.1, -0.2]])}
        plan = rows_of(MASK, ("a", 0, 2))
        with pytest.raises(StorePlanError, match="absent from its sampling support"):
            plan.materialize(objects, response_tokens=np.asarray([9, 7]), **opts())
        with pytest.raises(StorePlanError, match="exactly once"):
            plan.materialize(objects, response_tokens=np.asarray([6, 7]), **opts(support_topk=2))
        with pytest.raises(StorePlanError, match="at least one token"):
            rows_of(MASK, ("b", 0, 1)).materialize({"b": mask_arrays([[]])}, response_tokens=np.asarray([1]), **opts())

    def test_a_support_wider_than_k_is_rejected(self):
        objects = {"a": mask_arrays([[5, 6, 7]], [[-0.1, -0.2, -0.3]])}
        with pytest.raises(StorePlanError, match="exceeds --rollout-top-logprobs-num"):
            rows_of(MASK, ("a", 0, 1)).materialize(
                objects,
                response_tokens=np.asarray([6]),
                **opts(support_topk=2),
            )


def test_plan_materialize_merges_every_fields_columns():
    plan = StorePlan.from_response(
        ref("a", seq_rows=4, mask_rows=2), request=request(), num_tokens=5, response_length=2
    )
    objects = {"a": {R3: r3(4, 1), INDEXER: r3(4, 2), **mask_arrays([[5, 6], [8]])}}

    columns = plan.materialize(objects, response_tokens=np.asarray([6, 8]), **opts())

    assert set(columns) == {
        "rollout_routed_experts",
        "rollout_indexer_topk",
        "rollout_sampling_mask_ids",
        "rollout_sampling_mask_offsets",
    }
