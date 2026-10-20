"""ReplayIndex: one parser for inline and output store replay fields, and segment-only rollout edits."""

import numpy as np
import pytest
from tests.fast.fixtures.output_store_fixtures import b64, make_store, mask_arrays, put_response

from miles.utils.replay_index import (
    INLINE,
    STORE,
    ReplayDataError,
    ReplayIndex,
    compute_train_columns,
    pop_train_columns,
    response_has_routed_experts,
    sampled_logprobs_from_meta_info,
)

LAYERS, TOPK = 3, 2


def _rows(n: int, start: int = 1) -> np.ndarray:
    return np.arange(start, start + n * LAYERS * TOPK, dtype=np.int32).reshape(n, LAYERS, TOPK)


def _parse(meta_info, *, via_store, rows=4, num_output_tokens=2, support=False, mask=True, extra_row=False):
    return ReplayIndex.from_meta_info(
        meta_info,
        via_store=via_store,
        routed_experts_rows=rows,
        num_layers=LAYERS,
        indexer_topk_rows=rows,
        indexer_num_streams=None,
        return_sampling_mask=mask,
        support_logprobs=support,
        num_output_tokens=num_output_tokens,
        aborted=False,
        allow_extra_routed_experts_row=extra_row,
    )


def _responses(store, *, support: bool):
    """The same response inline and through the store, as SGLang returns it in each log-prob mode."""
    routed, indexer = _rows(4), _rows(4, start=100)
    supports = [[7, 3], [8]]
    support_logprobs = [[-0.1, -2.0], [-0.3]]
    # The sampled tokens' log-probs stay inline either way.
    if support:
        sampled = {"output_token_sampling_selected_logprobs": [-0.1, -0.3]}
    else:
        sampled = {"output_token_sampling_logprobs": [-0.1, -0.3]}
    inline = {
        "routed_experts": b64(routed),
        "indexer_topk": b64(indexer),
        "indexer_topk_num_layers": LAYERS,
        "output_token_sampling_mask": supports,
        **sampled,
    }
    if support:
        inline["output_token_sampling_logprobs"] = support_logprobs
    arrays = mask_arrays(supports, support_logprobs if support else None)
    ref = put_response(store, routed_experts=routed, indexer_topk=indexer, **arrays)
    stored = {"output_store_ref": ref, "output_token_sampling_mask_length": 2, **sampled}
    return inline, stored


def _fetch_all(store, index: ReplayIndex) -> dict:
    fetched = {}
    for ref, names in index.store_reads():
        (stash,) = store.get_many([ref], stash_type=_stash_type())
        for name in names:
            fetched[(ref.key, name)] = np.array(getattr(stash, name))
    return fetched


def _stash_type():
    from radix_transfer_plugins.stashes.token_output import TokenOutputStash

    return TokenOutputStash


def _mask_arrays(index: ReplayIndex, fetched: dict) -> list[tuple]:
    return [
        (
            piece.num_rows,
            *(None if a is None else a.tolist() for a in (piece.lengths, piece.token_ids, piece.logprobs)),
        )
        for piece in index.sampling_mask.mask_pieces(fetched)
    ]


@pytest.mark.parametrize("support", [False, True])
def test_inline_and_store_responses_index_the_same_data(support):
    store, _ = make_store()
    inline_meta, stored_meta = _responses(store, support=support)
    inline = _parse(inline_meta, via_store=False, support=support)
    stored = _parse(stored_meta, via_store=True, support=support)

    assert inline.routed_experts.source_kind == INLINE and stored.routed_experts.source_kind == STORE
    fetched = _fetch_all(store, stored)
    for name in ("routed_experts", "indexer_topk"):
        np.testing.assert_array_equal(
            getattr(inline, name).materialize_rows({}), getattr(stored, name).materialize_rows(fetched)
        )
    assert _mask_arrays(inline, {}) == _mask_arrays(stored, fetched)


def test_store_flag_with_inline_arrays_points_at_the_router():
    store, _ = make_store()
    inline_meta, _ = _responses(store, support=False)
    with pytest.raises(ReplayDataError, match="router .* may have dropped return_outputs_via_store"):
        _parse(inline_meta, via_store=True)


def test_inline_response_without_the_store_flag_parses_as_before():
    store, _ = make_store()
    inline_meta, stored_meta = _responses(store, support=False)
    assert _parse(inline_meta, via_store=False).routed_experts.num_rows == 4
    with pytest.raises(ReplayDataError, match="did not ask for one"):
        _parse(stored_meta, via_store=False)


def test_ref_of_another_backend_is_rejected_with_its_name():
    store, _ = make_store()
    _, stored_meta = _responses(store, support=False)
    stored_meta["output_store_ref"] = {**stored_meta["output_store_ref"], "backend": "disk"}
    with pytest.raises(ReplayDataError, match="backend 'disk' is not supported"):
        _parse(stored_meta, via_store=True)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(rows=5), "stored routed_experts has 4 rows; expected 5"),
        (dict(num_output_tokens=3), "sampling support length 2 != token length 3"),
        (dict(support=True), r"without \['output_token_sampling_logprobs'\]"),
    ],
)
def test_store_structure_is_checked_on_arrival_without_reading(kwargs, match):
    store, backend = make_store()
    _, stored_meta = _responses(store, support=False)
    with pytest.raises(ReplayDataError, match=match):
        _parse(stored_meta, via_store=True, **kwargs)
    assert backend.reads == []


def test_legacy_stop_edge_row_is_dropped_for_inline_and_store():
    store, _ = make_store()
    routed = _rows(5)
    ref = put_response(store, routed_experts=routed)
    inline = _parse({"routed_experts": b64(routed)}, via_store=False, mask=False, extra_row=True)
    stored = _parse({"output_store_ref": ref}, via_store=True, mask=False, extra_row=True)
    np.testing.assert_array_equal(inline.routed_experts.materialize_rows({}), routed[:4])
    np.testing.assert_array_equal(stored.routed_experts.materialize_rows(_fetch_all(store, stored)), routed[:4])


def test_edits_rearrange_segments_and_read_nothing():
    store, backend = make_store()
    _, first_meta = _responses(store, support=False)
    _, second_meta = _responses(store, support=False)
    first = _parse(first_meta, via_store=True)
    second = _parse(second_meta, via_store=True, rows=4)

    merged = first.merge_turn(second, observation_tokens=3)
    assert merged.routed_experts is second.routed_experts
    assert merged.sampling_mask.num_rows == 2 + 3 + 2
    assert [s.kind for s in merged.sampling_mask.segments] == [STORE, "forced", STORE]
    stripped = merged.strip_tail(3)
    assert stripped.sampling_mask.num_rows == 4 and stripped.routed_experts.num_rows == 1
    added = first.routed_experts.concat(second.routed_experts).prefix(6)
    assert [(s.start, s.stop) for s in added.segments] == [(0, 4), (0, 2)]
    assert backend.reads == []


def test_one_sample_cannot_mix_inline_and_store_sources():
    store, _ = make_store()
    inline_meta, stored_meta = _responses(store, support=False)
    inline, stored = _parse(inline_meta, via_store=False), _parse(stored_meta, via_store=True)
    with pytest.raises(ReplayDataError, match="mixes replay data sources"):
        inline.merge_turn(stored, observation_tokens=1)
    with pytest.raises(ReplayDataError, match="mixes replay data sources"):
        inline.routed_experts.concat(stored.routed_experts)


def test_wire_round_trip_keeps_store_inline_and_forced_segments():
    store, _ = make_store()
    inline_meta, stored_meta = _responses(store, support=False)
    stored = _parse(stored_meta, via_store=True).append_forced_tokens(2, response_length=2)
    inline = _parse(inline_meta, via_store=False).strip_tail(1)

    for index in (stored, inline):
        descriptor, arrays = index.to_wire()
        restored = ReplayIndex.from_wire(descriptor, arrays)
        assert restored.to_wire()[0] == descriptor
        fetched = _fetch_all(store, index)
        assert _mask_arrays(restored, fetched) == _mask_arrays(index, fetched)
    descriptor, arrays = stored.to_wire()
    assert arrays == {} and [list(entry) for entry in descriptor["sampling_mask"]] == [
        ["store", "start", "stop"],
        ["forced"],
    ]


def test_train_columns_mix_inline_and_store_samples():
    store, _ = make_store()
    inline_meta, stored_meta = _responses(store, support=True)
    indexes = [
        _parse(inline_meta, via_store=False, support=True),
        _parse(stored_meta, via_store=True, support=True),
        ReplayIndex(),
    ]

    columns = compute_train_columns(indexes)
    assert [len(a) for a in columns["rollout_routed_experts"]] == [4, 0, 0]
    assert columns["rollout_sampling_mask_logprobs"][0].dtype == np.float32
    restored = pop_train_columns(dict(columns))
    np.testing.assert_array_equal(restored[0].routed_experts.materialize_rows({}), _rows(4))
    assert restored[1].routed_experts.source_kind == STORE and restored[2].is_empty()
    assert compute_train_columns([ReplayIndex(), ReplayIndex()]) == {}


def test_response_has_routed_experts_reads_no_data():
    store, _ = make_store()
    inline_meta, stored_meta = _responses(store, support=False)
    assert response_has_routed_experts(inline_meta) and response_has_routed_experts(stored_meta)
    assert not response_has_routed_experts({"output_store_ref": put_response(store, indexer_topk=_rows(2))})


@pytest.mark.parametrize("support", [False, True])
def test_sampled_logprobs_stay_inline_with_and_without_the_store(support):
    store, _ = make_store()
    inline_meta, stored_meta = _responses(store, support=support)
    for meta_info in (inline_meta, stored_meta):
        values = sampled_logprobs_from_meta_info(
            meta_info, output_token_ids=[7, 8], support_logprobs=support, aborted=False
        )
        assert values == pytest.approx([-0.1, -0.3])
    if support:
        # Inline support rows: the sampled token's entry is picked from its row, not SGLang's pick.
        picked = sampled_logprobs_from_meta_info(
            inline_meta | {"output_token_sampling_selected_logprobs": [0.0, 0.0]},
            output_token_ids=[7, 8],
            support_logprobs=True,
            aborted=False,
        )
        assert picked == pytest.approx([-0.1, -0.3])


def test_missing_sampled_logprobs_fail_unless_the_request_aborted_empty():
    with pytest.raises(ValueError, match="missing the sampling mask or output_token_sampling_logprobs"):
        sampled_logprobs_from_meta_info({}, output_token_ids=[1], support_logprobs=True, aborted=False)
    with pytest.raises(ValueError, match="missing the sampling mask or output_token_sampling_selected_logprobs"):
        sampled_logprobs_from_meta_info(
            {"output_store_ref": {}}, output_token_ids=[1], support_logprobs=True, aborted=False
        )
    assert sampled_logprobs_from_meta_info({}, output_token_ids=[], support_logprobs=False, aborted=True) == []


def test_train_descriptor_survives_strict_msgpack():
    # The Mooncake object store packs the train dict's replay_index column with strict msgpack types.
    msgpack = pytest.importorskip("msgpack")
    store, _ = make_store()
    _, stored_meta = _responses(store, support=False)
    index = _parse(stored_meta, via_store=True).append_forced_tokens(1, response_length=2)

    (descriptor,) = compute_train_columns([index])["replay_index"]
    restored = msgpack.unpackb(msgpack.packb(descriptor, use_bin_type=True, strict_types=True), raw=False)
    assert ReplayIndex.from_wire(restored, {}).to_wire()[0] == descriptor
