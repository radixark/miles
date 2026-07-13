"""Unit tests for the minimal IPC channel (miles/rollout/session/ipc.py)."""

import asyncio
import socket

import pytest

from miles.rollout.session.ipc import (
    _LEN,
    _MAX_FRAME,
    IpcChannelClosed,
    IpcError,
    decode_envelope,
    encode_envelope,
    open_unix_channel,
)


async def _make_pair(handler=None, *, client_on_close=None):
    """A connected (client, server) channel pair over a socketpair."""
    s_client, s_server = socket.socketpair()
    server = await open_unix_channel(s_server, request_handler=handler)
    client = await open_unix_channel(s_client, on_close=client_on_close)
    return client, server


async def _close(*channels):
    for ch in channels:
        await ch.aclose()


def test_envelope_roundtrip():
    meta = {"op": "chat", "status": 200, "headers": {"content-type": "application/json"}}
    body = b'{"x":1}\x00\xff'
    m2, b2 = decode_envelope(encode_envelope(meta, body))
    assert m2 == meta
    assert b2 == body
    m3, b3 = decode_envelope(encode_envelope({}, b""))
    assert m3 == {}
    assert b3 == b""


async def test_request_reply_roundtrip():
    async def echo(payload):
        return payload

    client, server = await _make_pair(echo)
    try:
        assert await asyncio.wait_for(client.request(b"hello"), 2) == b"hello"
        assert await asyncio.wait_for(client.request(b""), 2) == b""
        assert await asyncio.wait_for(client.request(b"\x00\xff\x01"), 2) == b"\x00\xff\x01"
    finally:
        await _close(client, server)


async def test_multiplexing_replies_match_requests():
    async def echo_after(payload):
        # payload = b"<delay_ms>:<token>"; later-sent requests use shorter delays
        delay_ms, _, token = payload.partition(b":")
        await asyncio.sleep(int(delay_ms) / 1000)
        return token

    client, server = await _make_pair(echo_after)
    try:
        # decreasing delays → out-of-order completion; each future must still get ITS reply
        reqs = [f"{(10 - i) * 10}:token-{i}".encode() for i in range(10)]
        results = await asyncio.wait_for(asyncio.gather(*(client.request(r) for r in reqs)), 5)
        assert results == [f"token-{i}".encode() for i in range(10)]
    finally:
        await _close(client, server)


async def test_handler_error_becomes_ipc_error_and_channel_survives():
    async def maybe_boom(payload):
        if payload == b"x":
            raise ValueError("kaboom")
        return payload

    client, server = await _make_pair(maybe_boom)
    try:
        with pytest.raises(IpcError, match="kaboom"):
            await asyncio.wait_for(client.request(b"x"), 2)
        # a handler error must not kill the channel
        assert await asyncio.wait_for(client.request(b"ok"), 2) == b"ok"
    finally:
        await _close(client, server)


async def test_peer_close_fails_pending_and_calls_on_close():
    closed = asyncio.Event()

    async def never(payload):
        await asyncio.sleep(60)

    client, server = await _make_pair(never, client_on_close=closed.set)
    try:
        task = asyncio.ensure_future(client.request(b"x"))
        await asyncio.sleep(0.05)
        await server.aclose()  # peer closes → client reader hits EOF
        with pytest.raises(IpcChannelClosed):
            await asyncio.wait_for(task, 2)
        await asyncio.wait_for(closed.wait(), 2)
        # a request on a closed channel fails fast
        with pytest.raises(IpcChannelClosed):
            await client.request(b"y")
    finally:
        await _close(client, server)


async def test_cancel_then_reuse_drops_late_reply():
    async def slow_echo(payload):
        await asyncio.sleep(0.2)
        return payload

    client, server = await _make_pair(slow_echo)
    try:
        task = asyncio.ensure_future(client.request(b"first"))
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # the cancelled request's late reply must be dropped, not delivered to the next request
        assert await asyncio.wait_for(client.request(b"second"), 2) == b"second"
    finally:
        await _close(client, server)


async def test_oversized_frame_length_tears_down():
    s_client, s_peer = socket.socketpair()
    closed = asyncio.Event()
    client = await open_unix_channel(s_client, on_close=closed.set)
    try:
        # peer declares a frame length beyond the sanity cap → tear down, never allocate it
        s_peer.sendall(_LEN.pack(_MAX_FRAME + 1))
        await asyncio.wait_for(closed.wait(), 2)
        with pytest.raises(IpcChannelClosed):
            await client.request(b"x")
    finally:
        await client.aclose()
        s_peer.close()
