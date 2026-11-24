import asyncio
import json
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
import torch

from miles.backends.training_utils.data.rollout import prepare_rollout_data
from miles.backends.training_utils.parallel import ParallelState, set_parallel_state
from miles.backends.training_utils.replay.data import attach_replay_data, register_replay_list_sequential
from miles.ray.rollout.rollout_executor import _compact_train_data_routed_experts
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compact_routed_experts_array,
    decode_routed_experts_buffer,
    decode_routed_experts_from_response_async,
    get_routed_experts_from_response,
    update_sample_from_response,
)
from miles.utils import http_utils
from miles.utils.ft_utils.process_group_utils import GroupInfo
from miles.utils.replay_base import RoutingReplayManager
from miles.utils.types import Sample


class _StubResponse:
    def __init__(self, body: bytes):
        self.content = body
        self.status_code = 200
        self.json_calls = 0

    def raise_for_status(self) -> None:
        return None

    def json(self):
        self.json_calls += 1
        return json.loads(self.content)

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")


class _StubClient:
    def __init__(self, response: _StubResponse):
        self._response = response

    async def post(self, url, json=None, headers=None):
        return self._response


def test_post_offloads_large_json_decoding_to_worker_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    offloaded_calls: list[int] = []
    orig_to_thread = asyncio.to_thread

    async def tracked_to_thread(func, *args, **kwargs):
        offloaded_calls.append(len(args[0]) if args else 0)
        return await orig_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(http_utils.asyncio, "to_thread", tracked_to_thread)

    small_body = json.dumps({"text": "ok", "meta_info": {"routed_experts": "AAAA"}}).encode("utf-8")
    small_resp = _StubResponse(small_body)
    small_out = asyncio.run(http_utils._post(_StubClient(small_resp), "http://127.0.0.1:8000/generate", {"x": 1}))
    assert small_out["text"] == "ok"
    assert small_resp.json_calls == 1
    assert offloaded_calls == []

    large_payload = {"text": "ok", "meta_info": {"routed_experts": "A" * 100_000}}
    large_body = json.dumps(large_payload).encode("utf-8")
    assert len(large_body) >= http_utils._ASYNC_JSON_DECODE_THRESHOLD_BYTES
    large_resp = _StubResponse(large_body)
    large_out = asyncio.run(http_utils._post(_StubClient(large_resp), "http://127.0.0.1:8000/generate", {"x": 1}))
    assert large_out == large_payload
    assert large_resp.json_calls == 0
    assert offloaded_calls == [len(large_body)]

    bad_large_body = b"not-json-" * 9000
    assert len(bad_large_body) >= http_utils._ASYNC_JSON_DECODE_THRESHOLD_BYTES
    bad_resp = _StubResponse(bad_large_body)
    bad_out = asyncio.run(http_utils._post(_StubClient(bad_resp), "http://127.0.0.1:8000/generate", {"x": 1}))
    assert bad_out == bad_resp.text


@pytest.mark.parametrize("wire_dtype_name,np_dtype", [("uint8", np.uint8), ("int16", np.int16), ("int32", np.int32)])
def test_routed_experts_wire_dtypes_decode_and_preserve_or_widen(wire_dtype_name: str, np_dtype) -> None:
    arr = np.array([[[0, 1, 255], [12, 34, 200]], [[5, 6, 7], [8, 9, 10]]], dtype=np_dtype)
    encoded = pybase64.b64encode(arr.tobytes()).decode("ascii")
    output = {"meta_info": {"routed_experts": encoded, "routed_experts_dtype": wire_dtype_name}}
    args = Namespace(num_layers=2)

    widened = get_routed_experts_from_response(args, output, num_tokens=2)
    assert widened.dtype == np.int32
    np.testing.assert_array_equal(widened, arr.astype(np.int32))

    preserved = get_routed_experts_from_response(args, output, num_tokens=2, preserve_wire_dtype=True)
    assert preserved.dtype == np_dtype
    np.testing.assert_array_equal(preserved, arr)


def test_routed_experts_invalid_wire_dtype_and_invalid_payloads_raise() -> None:
    good = pybase64.b64encode(np.ones((2, 2, 2), dtype=np.int32).tobytes()).decode("ascii")
    with pytest.raises(ValueError, match="Unsupported routed_experts_dtype"):
        decode_routed_experts_buffer(good, num_tokens=2, num_layers=2, wire_dtype="float16")

    zeros = pybase64.b64encode(np.zeros((2, 2, 2), dtype=np.uint8).tobytes()).decode("ascii")
    with pytest.raises(AssertionError, match="all zeros"):
        decode_routed_experts_buffer(zeros, num_tokens=2, num_layers=2, wire_dtype="uint8")

    with pytest.raises(ValueError):
        decode_routed_experts_buffer("not valid base64\n", num_tokens=1, num_layers=1)


def test_compact_routed_experts_preserves_minus_one_padding_and_wide_ids() -> None:
    u8_case = np.array([[[0, 255]]], dtype=np.int32)
    compacted_u8 = compact_routed_experts_array(u8_case)
    assert compacted_u8.dtype == np.uint8
    np.testing.assert_array_equal(compacted_u8, u8_case)

    padded_case = np.array([[[-1, 255]]], dtype=np.int32)
    compacted_padded = compact_routed_experts_array(padded_case)
    assert compacted_padded.dtype == np.int16
    assert int(compacted_padded[0, 0, 0]) == -1
    assert int(compacted_padded[0, 0, 1]) == 255

    over_256 = np.array([[[0, 384]]], dtype=np.int32)
    compacted_i16 = compact_routed_experts_array(over_256)
    assert compacted_i16.dtype == np.int16
    np.testing.assert_array_equal(compacted_i16, over_256)

    wide_case = np.array([[[0, 40_000]]], dtype=np.int32)
    compacted_i32 = compact_routed_experts_array(wide_case)
    assert compacted_i32.dtype == np.int32
    np.testing.assert_array_equal(compacted_i32, wide_case)


def test_large_routed_experts_decode_and_batch_compaction_offload_to_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offloaded_fns: list[str] = []
    orig_to_thread = asyncio.to_thread

    async def tracked_to_thread(func, *args, **kwargs):
        offloaded_fns.append(func.__name__)
        return await orig_to_thread(func, *args, **kwargs)

    monkeypatch.setattr("miles.rollout.generate_utils.generate_endpoint_utils.asyncio.to_thread", tracked_to_thread)
    monkeypatch.setattr("miles.ray.rollout.rollout_executor.asyncio.to_thread", tracked_to_thread)

    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, size=(256, 16, 8), dtype=np.int32)
    encoded = pybase64.b64encode(arr.tobytes()).decode("ascii")
    assert len(encoded) >= 65536

    args = Namespace(
        num_layers=16,
        rollout_top_logprobs_num=0,
        sglang=SimpleNamespace(common_value=lambda _k: None),
        sglang_speculative_algorithm=None,
        max_seq_len=None,
    )
    sample = Sample(tokens=list(range(200)))
    payload = {"input_ids": list(sample.tokens)}
    output = {
        "text": "ok",
        "meta_info": {
            "prompt_tokens": 200,
            "completion_tokens": 57,
            "output_token_logprobs": [(-0.1, i, None) for i in range(57)],
            "finish_reason": {"type": "stop"},
            "routed_experts": encoded,
            "routed_experts_dtype": "int32",
        },
    }
    asyncio.run(update_sample_from_response(args, sample, payload, output))
    assert "decode_routed_experts_buffer" in offloaded_fns
    assert sample.rollout_routed_experts.dtype == np.int32
    np.testing.assert_array_equal(sample.rollout_routed_experts, arr)

    compact_direct = asyncio.run(decode_routed_experts_from_response_async(args, output, 256, compact=True))
    assert compact_direct.dtype == np.uint8
    np.testing.assert_array_equal(compact_direct, arr)

    train_data = {"rollout_routed_experts": [sample.rollout_routed_experts]}
    asyncio.run(_compact_train_data_routed_experts(train_data))
    assert "_compact_routed_experts_list" in offloaded_fns
    assert train_data["rollout_routed_experts"][0].dtype == np.uint8
    assert train_data["rollout_routed_experts"][0].nbytes == sample.rollout_routed_experts.nbytes // 4


def test_trainer_unpack_and_replay_widen_narrow_dtypes_for_pad_and_gather(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("miles.utils.replay_base.current_platform.current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("miles.utils.replay_base.current_platform.dynamic_shapes", False)
    single_group = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=single_group,
            intra_dp_cp=single_group,
            cp=single_group,
            tp=single_group,
            pp=single_group,
            ep=single_group,
            etp=single_group,
            indep_dp=single_group,
        ),
    )

    u8_experts = np.array([[[0, 255], [3, 7]], [[1, 254], [5, 9]]], dtype=np.uint8)
    i16_experts = np.array([[[-1, -1], [4, 256]], [[2, 300], [-1, -1]]], dtype=np.int16)
    rollout_data = {
        "tokens": [[10, 11, 12], [20, 21, 22]],
        "loss_masks": [[1, 1], [1, 1]],
        "response_lengths": [2, 2],
        "total_lengths": [3, 3],
        "rollout_routed_experts": [u8_experts, i16_experts],
    }
    args = SimpleNamespace(
        enable_witness=False,
        multimodal_token_space_processing=False,
        debug_rollout_only=False,
        qkv_format="thd",
        data_pad_size_multiplier=4,
        allgather_cp=False,
        train_backend="torchtitan",
        save_debug_train_data=None,
        backend=SimpleNamespace(compress_ratios=None, sequence_parallel=False),
    )
    prepared = prepare_rollout_data(args, rollout_data, sample_device=torch.device("cpu"))
    assert [t.dtype for t in prepared["rollout_routed_experts"]] == [torch.int32, torch.int32]

    # Also verify attach_replay_data widens raw uint8/int16 tensors directly so pad_func(-1)
    # never wraps to 255 and scores.gather(1, top_indices) succeeds.
    raw_rollout_data = {
        "tokens": [torch.tensor([10, 11, 12]), torch.tensor([20, 21, 22])],
        "rollout_routed_experts": [torch.from_numpy(u8_experts), torch.from_numpy(i16_experts)],
    }

    class _Iter:
        @staticmethod
        def microbatch_indices(_n: int):
            return [[0, 1]]

    manager = RoutingReplayManager()
    manager.enabled = True
    manager.register_replay_list_func = register_replay_list_sequential
    r0 = manager.create_replay(stream_idx=0)
    r1 = manager.create_replay(stream_idx=1)
    attach_replay_data(
        manager,
        args=args,
        data_iterator=[_Iter()],
        num_microbatches=[1],
        rollout_data=raw_rollout_data,
    )
    manager.stage = "replay_forward"
    manager.set_current(r0)
    topk_fn = manager.get_topk_fn(lambda s, k: torch.topk(s, k, dim=-1), return_probs=True)
    scores = torch.randn(8, 384)
    probs0, idx0 = topk_fn(scores, 2)
    assert idx0.dtype == torch.int32
    assert probs0.shape == (8, 2)
    assert idx0[0].tolist() == [0, 255]
    # Position 2 is end-of-sample padding (-1 filled by fill_replay_padding -> [0, 1], NOT [255, 255]!)
    assert idx0[2].tolist() == [0, 1]
    # Position 3 is sample 1 row 0 ([-1, -1] -> [0, 1])
    assert idx0[3].tolist() == [0, 1]
    assert idx0[4].tolist() == [2, 300]

    manager.set_current(r1)
    probs1, idx1 = topk_fn(scores, 2)
    assert probs1.shape == (8, 2)
    assert idx1[3].tolist() == [4, 256]
    manager.clear_all()
