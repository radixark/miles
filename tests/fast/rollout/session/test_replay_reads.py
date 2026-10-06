import asyncio
import logging

import numpy as np
import pytest
from tests.fast.fixtures.output_store_fixtures import FakeMooncakeStore, output_store_ref

from miles.rollout.generate_utils.output_store import OUTPUT_STORE_REF_KEY, ReplayOutputs
from miles.rollout.session.replay_reads import ReplayReader, ReplayReadError, wait_for_replay_reads
from miles.rollout.session.types import SessionRecord

pytestmark = pytest.mark.asyncio

HANDLE = {"type": "mooncake_dataproto_ref", "id": 3}
ROUTED_EXPERTS = np.arange(2 * 2 * 2, dtype=np.int32).reshape(2, 2, 2)


def _response(meta_info) -> dict:
    return {"choices": [{"meta_info": meta_info}]}


def _record(read: asyncio.Future | None) -> SessionRecord:
    record = SessionRecord(
        timestamp=0.0, method="POST", path="/v1/chat/completions", request={}, response={}, status_code=200
    )
    record.attach_replay_read(read)
    return record


async def test_a_response_without_a_ref_starts_no_read():
    reader = ReplayReader(FakeMooncakeStore({}))

    assert reader.start(_response({"output_token_logprobs": []})) is None
    assert reader.start(_response("left for extract_completion to reject")) is None


async def test_a_read_yields_owned_arrays_and_removes_the_bundle():
    store = FakeMooncakeStore({"routed_experts": [ROUTED_EXPERTS.tolist()]})
    reader = ReplayReader(store)

    read = reader.start(
        _response({OUTPUT_STORE_REF_KEY: output_store_ref(HANDLE, routed_experts=("int32", [2, 2, 2]))})
    )

    np.testing.assert_array_equal((await read).routed_experts, ROUTED_EXPERTS)
    assert store.removed == [HANDLE]


async def test_a_failed_read_is_logged_even_when_nobody_waits_for_it(caplog):
    """A response the session discarded still has its read; its failure must not vanish."""
    store = FakeMooncakeStore({"routed_experts": [[1]]})
    reader = ReplayReader(store)

    with caplog.at_level(logging.ERROR):
        read = reader.start(
            _response({OUTPUT_STORE_REF_KEY: output_store_ref(HANDLE, routed_experts=("int32", [2, 2, 2]))})
        )
        await asyncio.wait([read])
        await asyncio.sleep(0)

    assert "Reading an output-store bundle" in caplog.text
    assert store.removed == [HANDLE]


async def test_waiting_covers_records_committed_meanwhile():
    loop = asyncio.get_running_loop()
    first, second = loop.create_future(), loop.create_future()
    records = [_record(first), _record(None)]
    waiter = asyncio.create_task(wait_for_replay_reads(lambda: records))
    await asyncio.sleep(0)

    records.append(_record(second))
    first.set_result(ReplayOutputs())
    await asyncio.sleep(0)
    assert not waiter.done()

    second.set_result(ReplayOutputs())
    await waiter


async def test_a_failed_read_fails_the_wait_instead_of_leaving_missing_arrays():
    failed = asyncio.get_running_loop().create_future()
    failed.set_exception(RuntimeError("master unreachable"))

    with pytest.raises(ReplayReadError, match="master unreachable"):
        await wait_for_replay_reads(lambda: [_record(failed)])
