import asyncio
import logging
import shutil
import socket
import subprocess
import time
from argparse import Namespace

import numpy as np
import pybase64
import pytest
from tests.fast.fixtures.output_store_fixtures import FakeMooncakeStore, output_store_ref
from tests.fast.fixtures.score_centering_fixtures import _args, _meta

from miles.rollout.generate_utils.generate_endpoint_utils import compute_request_payload, update_sample_from_response
from miles.rollout.generate_utils.output_store import (
    OUTPUT_STORE_REF_KEY,
    ReplayOutputs,
    read_replay_outputs,
    resolve_replay_outputs,
)
from miles.utils import object_store
from miles.utils.object_store import RayObjectStore
from miles.utils.types import Sample

ROUTED_EXPERTS = np.arange(2 * 3 * 4, dtype=np.int32).reshape(2, 3, 4)


def _ref(**fields: tuple[str, list[int]]) -> dict:
    return output_store_ref({"type": "mooncake_dataproto_ref", "id": 1}, **fields)


class TestReadReplayOutputs:
    def test_copies_list_rows_into_owned_arrays_then_releases_and_removes(self):
        store = FakeMooncakeStore(
            {
                "routed_experts": [ROUTED_EXPERTS.tolist()],
                "indexer_topk": [[]],
                "output_token_sampling_mask_lengths": [[2, 1]],
                "output_token_sampling_mask_token_ids": [[5, 6, 7]],
                "output_token_sampling_logprobs": [[-0.5, -1.0]],
            }
        )
        ref = _ref(
            routed_experts=("int32", [2, 3, 4]),
            indexer_topk=("int32", [0, 3, 8]),
            output_token_sampling_mask_lengths=("int32", [2]),
            output_token_sampling_mask_token_ids=("int32", [3]),
            output_token_sampling_logprobs=("float32", [2]),
        )

        replay = read_replay_outputs(ref, store=store)

        np.testing.assert_array_equal(replay.routed_experts, ROUTED_EXPERTS)
        assert replay.routed_experts.dtype == np.int32
        assert replay.indexer_topk.shape == (0, 3, 8)
        assert replay.sampling_logprobs.dtype == np.float32
        np.testing.assert_array_equal(replay.sampling_mask_token_ids, [5, 6, 7])
        assert store.released
        assert store.removed == [ref["handle"]]

    @pytest.mark.parametrize(
        ("ref", "match"),
        [
            ({"handle": {}}, "'handle' and 'fields'"),
            (_ref(hidden_states=("float32", [2])), "unknown field"),
            (_ref(routed_experts=("int64", [2, 3, 4])), "must be int32 with 3 dims"),
            (_ref(routed_experts=("int32", [6, 4])), "must be int32 with 3 dims"),
            (_ref(output_token_sampling_mask_lengths=("int32", [2])), "all or none"),
        ],
    )
    def test_rejects_a_malformed_ref_before_touching_the_store(self, ref, match):
        store = FakeMooncakeStore({})

        with pytest.raises(ValueError, match=match):
            read_replay_outputs(ref, store=store)
        assert store.removed == []

    def test_a_bundle_that_disagrees_with_its_ref_fails_and_is_still_removed(self):
        """Reading failed, so nobody else will ever remove the object."""
        store = FakeMooncakeStore({"routed_experts": [ROUTED_EXPERTS.tolist()]})
        ref = _ref(routed_experts=("int32", [2, 4, 3]))

        with pytest.raises(ValueError, match="has shape"):
            read_replay_outputs(ref, store=store)
        assert store.removed == [ref["handle"]]

    def test_a_failed_removal_is_logged_with_the_handle_without_failing_the_read(self, caplog):
        store = FakeMooncakeStore(
            {"routed_experts": [ROUTED_EXPERTS.tolist()]}, remove_error=RuntimeError("master unreachable")
        )

        with caplog.at_level(logging.ERROR):
            replay = read_replay_outputs(_ref(routed_experts=("int32", [2, 3, 4])), store=store)

        np.testing.assert_array_equal(replay.routed_experts, ROUTED_EXPERTS)
        assert '"id": 1' in caplog.text

    def test_a_ray_object_store_cannot_read_a_mooncake_handle(self):
        with pytest.raises(ValueError, match="--object-store-backend mooncake"):
            read_replay_outputs(_ref(routed_experts=("int32", [2, 3, 4])), store=RayObjectStore(frees_objects=False))


class TestResolveReplayOutputs:
    def test_an_inline_response_needs_no_object_store(self, monkeypatch):
        monkeypatch.setattr(object_store, "_INSTANCE", None)

        assert asyncio.run(resolve_replay_outputs({"routed_experts": "AAAA"})) is None

    def test_reads_through_the_process_object_store(self, monkeypatch):
        store = FakeMooncakeStore({"routed_experts": [ROUTED_EXPERTS.tolist()]})
        monkeypatch.setattr(object_store, "_INSTANCE", store)
        meta_info = {OUTPUT_STORE_REF_KEY: _ref(routed_experts=("int32", [2, 3, 4]))}

        replay = asyncio.run(resolve_replay_outputs(meta_info))

        assert isinstance(replay, ReplayOutputs)
        np.testing.assert_array_equal(replay.routed_experts, ROUTED_EXPERTS)


def _replay_args(**overrides: object) -> Namespace:
    return _args(
        **{
            "rollout_top_logprobs_num": 0,
            "rollout_max_response_len": 20,
            "rollout_max_context_len": None,
            "use_rollout_routing_replay": True,
            "use_rollout_indexer_replay": True,
            "sglang_speculative_algorithm": None,
            "sglang_output_store_backend": "mooncake",
            "num_layers": 3,
            **overrides,
        }
    )


class TestRequestOptIn:
    def test_a_training_request_with_replay_outputs_opts_in(self):
        payload, _ = compute_request_payload(_replay_args(), [0, 1], {})

        assert payload["return_outputs_via_store"] is True

    @pytest.mark.parametrize(
        ("overrides", "evaluation"),
        [
            ({}, True),
            ({"sglang_output_store_backend": "none"}, False),
            ({"use_rollout_routing_replay": False, "use_rollout_indexer_replay": False}, False),
        ],
        ids=["evaluation", "backend-off", "no-replay-outputs"],
    )
    def test_other_requests_stay_inline(self, overrides, evaluation):
        payload, _ = compute_request_payload(_replay_args(**overrides), [0, 1], {}, evaluation=evaluation)

        assert "return_outputs_via_store" not in payload

    def test_an_sglang_without_the_flag_never_opts_in(self):
        args = _replay_args()
        del args.sglang_output_store_backend

        payload, _ = compute_request_payload(args, [0, 1], {})

        assert "return_outputs_via_store" not in payload


def test_inline_and_output_store_responses_build_the_same_sample(monkeypatch):
    """The output store changes only how the replay arrays travel, never the Sample they produce."""
    args = _replay_args()
    payload, _ = compute_request_payload(args, [0, 1], {})
    indexer_topk = np.arange(2 * 2 * 5, dtype=np.int32).reshape(2, 2, 5)
    base_meta = _meta([2], [0.5])

    def encode(array: np.ndarray) -> str:
        return pybase64.b64encode(array.tobytes()).decode("ascii")

    inline_meta = {
        **base_meta,
        "routed_experts": encode(ROUTED_EXPERTS),
        "indexer_topk": encode(indexer_topk),
        "indexer_topk_num_layers": 2,
    }
    store_meta = {
        **base_meta,
        OUTPUT_STORE_REF_KEY: _ref(routed_experts=("int32", [2, 3, 4]), indexer_topk=("int32", [2, 2, 5])),
    }
    store = FakeMooncakeStore({"routed_experts": [ROUTED_EXPERTS.tolist()], "indexer_topk": [indexer_topk.tolist()]})
    monkeypatch.setattr(object_store, "_INSTANCE", store)

    samples = []
    for meta_info in (inline_meta, store_meta):
        sample = Sample()
        asyncio.run(update_sample_from_response(args, sample, payload, {"text": "2", "meta_info": meta_info}))
        samples.append(sample)

    inline, via_store = samples
    np.testing.assert_array_equal(via_store.rollout_routed_experts, inline.rollout_routed_experts)
    np.testing.assert_array_equal(via_store.rollout_indexer_topk, inline.rollout_indexer_topk)
    assert (via_store.tokens, via_store.rollout_log_probs) == (inline.tokens, inline.rollout_log_probs)
    assert len(store.removed) == 1


def _mooncake_available() -> bool:
    if shutil.which("mooncake_master") is None:
        return False
    try:
        from mooncake.structured_object_store import FieldSchema, MooncakeBundleTransfer, export_ref  # noqa: F401
    except ImportError:
        return False
    return True


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.skipif(not _mooncake_available(), reason="mooncake with structured_object_store API not installed")
class TestRealMooncakeRoundTrip:
    @pytest.fixture(scope="class")
    def mooncake_master_port(self):
        port = _free_port()
        master = subprocess.Popen(
            ["mooncake_master", "--rpc_port", str(port), "--metrics_port", str(_free_port())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
        yield port
        master.terminate()
        master.wait(timeout=10)

    def test_reads_and_removes_a_bundle_written_like_the_sglang_output_store(self, mooncake_master_port, monkeypatch):
        from mooncake.structured_object_store import FieldSchema, export_ref

        monkeypatch.setattr(object_store, "_INSTANCE", None)
        store = object_store.init_instance(
            Namespace(
                object_store_backend="mooncake",
                mooncake_store_init_kwargs={
                    "protocol": "tcp",
                    "master_server_address": f"127.0.0.1:{mooncake_master_port}",
                    "local_hostname": "127.0.0.1",
                    "global_segment_size": "64mb",
                    "local_buffer_size": "64mb",
                },
                mooncake_replica_num=1,
            ),
            contribute_segment=True,
        )
        fields = {
            "routed_experts": ROUTED_EXPERTS,
            "output_token_sampling_mask_lengths": np.array([2, 1], dtype=np.int32),
            "output_token_sampling_mask_token_ids": np.array([5, 6, 7], dtype=np.int32),
            "output_token_sampling_logprobs": np.array([-0.5, -1.0], dtype=np.float32),
        }
        # The SGLang producer's put: one typed_ragged row per field, under Miles' key prefix.
        bundle_ref = store._transfer.put(
            {name: [array] for name, array in fields.items()},
            type="dict",
            field_schemas={
                name: FieldSchema(
                    codec="typed_ragged",
                    nullable=False,
                    metadata={"section": "non_tensor_batch", "dtype": str(array.dtype)},
                )
                for name, array in fields.items()
            },
        )
        output_store_ref = {
            "handle": export_ref(bundle_ref),
            "fields": {name: {"dtype": str(a.dtype), "shape": list(a.shape)} for name, a in fields.items()},
        }

        replay = read_replay_outputs(output_store_ref, store=store)

        np.testing.assert_array_equal(replay.routed_experts, ROUTED_EXPERTS)
        np.testing.assert_array_equal(replay.sampling_mask_lengths, [2, 1])
        np.testing.assert_array_equal(replay.sampling_logprobs, np.array([-0.5, -1.0], dtype=np.float32))
        with pytest.raises(Exception):  # noqa: B017 - mooncake surfaces missing keys as varying exception types
            read_replay_outputs(output_store_ref, store=store)
