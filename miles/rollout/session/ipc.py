# doc-dev: docs/developer/multi-process-session-server.md
"""Minimal framed, multiplexed RPC between the session router and one session worker.

- One :class:`IpcChannel` wraps one connected stream socket (a ``socket.socketpair`` end); the router side is the client (:meth:`IpcChannel.request`), the worker side is the server (a ``request_handler`` coroutine).
- Concurrent requests over the one socket are multiplexed by ``request_id``: replies are matched by id, not arrival order, so a reply never waits for an earlier request to finish *processing*. Writes are FIFO whole frames, so it can still queue behind a large frame already being sent (see the trailing note).
- Wire format — one whole message per frame, no chunking:

    frame: u64 length | u64 request_id | u8 type | payload bytes
           └ length counts everything after it (the 9-byte header + payload) ┘

- ``type`` is REQUEST / REPLY / ERROR; a REPLY carries the handler's return bytes, an ERROR a UTF-8 error message.
- The payload is opaque to the channel; the worker/router layer packs it with :func:`encode_envelope` as ``u64 meta_len | meta_json | raw_body`` (raw body, no base64).
- The shared ``OP_*`` session-op protocol and the request/envelope codecs live here so neither the router nor the worker imports the other's module.
- Teardown (EOF / read-write error / ``aclose``) fails every pending ``request()`` with ``IpcChannelClosed``, so a peer death surfaces as an exception, never a hang; ``request()`` is cancel-safe (a late reply is dropped, not mis-delivered).

Deliberately minimal — no round-robin scheduling across requests, no configurable frame/body size limit; the single ``_MAX_FRAME`` cap is a corruption guard (rejects a garbage length before allocating), not a size feature, so a large body (e.g. a multi-GiB ``GET /sessions`` records dump) crosses as one frame. A frame that would exceed the cap is refused at send time as a per-request failure (ERROR frame / ``IpcError``) — it must never reach the peer's reader, where an oversized length is indistinguishable from corruption and tears down the whole channel.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

_LEN = struct.Struct(">Q")  # frame length / envelope meta_len prefix (u64)
_HDR = struct.Struct(">QB")  # request_id (u64) + type (u8)
_HDR_LEN = _HDR.size

# Reject a frame whose declared length exceeds this before allocating, so a
# corrupt/garbage u64 length cannot trigger an absurd readexactly. Generous vs
# the largest real body: a full-records GET measures ~3.15 GiB per session at
# the production shape (50 turns x ~64 MiB avg accumulated-R3 responses) and
# grows with turns x body size.
_MAX_FRAME = 64 * 1024 * 1024 * 1024  # 64 GiB

_TYPE_REQUEST = 1
_TYPE_REPLY = 2
_TYPE_ERROR = 3


class IpcError(Exception):
    """A peer request handler raised; surfaced on the caller's request()."""


class IpcChannelClosed(Exception):
    """The channel was torn down (EOF / connection error / aclose)."""


def encode_envelope(meta: dict, body: bytes) -> bytes:
    """Pack ``meta`` (JSON) + raw ``body`` (no base64) into one buffer."""
    meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return _LEN.pack(len(meta_bytes)) + meta_bytes + body


def decode_envelope(buf: bytes) -> tuple[dict, bytes]:
    """Inverse of :func:`encode_envelope`."""
    (meta_len,) = _LEN.unpack_from(buf, 0)
    start = _LEN.size
    meta = json.loads(buf[start : start + meta_len])
    return meta, buf[start + meta_len :]


# Session op protocol shared by the router (encodes requests) and the worker
# (decodes them). Kept here so neither side imports the other's heavy module.
OP_HEALTH = "health"
OP_CREATE = "create"
OP_GET = "get"
OP_DELETE = "delete"
OP_CHAT = "chat"
OP_PROXY = "proxy"


def encode_request(
    op: str,
    *,
    session_id: str = "",
    path: str = "",
    method: str = "",
    query: str = "",
    headers: dict | None = None,
    body: bytes = b"",
) -> bytes:
    """Encode a session request for the worker (op + routing/HTTP metadata + raw body)."""
    meta = {
        "op": op,
        "session_id": session_id,
        "path": path,
        "method": method,
        "query": query,
        "headers": headers or {},
    }
    return encode_envelope(meta, body)


class IpcChannel:
    """Framed multiplexed RPC over one connected stream socket.

    Role is set by ``request_handler``: None ⇒ client (calls ``request()``; a
    stray inbound REQUEST is logged and dropped); else ⇒ server (runs the handler
    per REQUEST; a stray REPLY for an unknown id is dropped). Teardown (EOF /
    error / ``aclose()``) fails every pending ``request()`` with IpcChannelClosed,
    so a peer death surfaces as that exception, never a hang.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        request_handler: Callable[[bytes], Awaitable[bytes]] | None = None,
        on_close: Callable[[], None] | None = None,
    ):
        self._reader = reader
        self._writer = writer
        self._request_handler = request_handler
        self._on_close = on_close
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._send_q: asyncio.Queue[bytes] = asyncio.Queue()
        self._handler_tasks: set[asyncio.Task] = set()
        self._reader_task: asyncio.Task | None = None
        self._writer_task: asyncio.Task | None = None
        self._closed = False

    def start(self) -> None:
        self._reader_task = asyncio.ensure_future(self._reader_loop())
        self._writer_task = asyncio.ensure_future(self._writer_loop())

    async def request(self, payload: bytes) -> bytes:
        """Send a request and await the peer's reply bytes.

        Cancel-safe: if the awaiting coroutine is cancelled, the pending future
        is removed in ``finally`` so a late reply is dropped, not mis-delivered.

        Raises IpcChannelClosed if the channel is/goes down, IpcError if the peer
        handler raised.
        """
        if self._closed:
            raise IpcChannelClosed("channel is closed")
        request_id = self._next_id
        self._next_id += 1
        fut = asyncio.get_running_loop().create_future()
        self._pending[request_id] = fut
        try:
            self._enqueue(_TYPE_REQUEST, request_id, payload)
            return await fut
        finally:
            self._pending.pop(request_id, None)

    def _enqueue(self, mtype: int, request_id: int, payload: bytes) -> None:
        if self._closed:
            return
        # Refuse an oversized frame HERE, as a per-request failure: on the send
        # side this raises to the caller (client request()) or becomes an ERROR
        # frame (server handler). If it were sent, the peer's reader could not
        # tell it from a corrupt length and would tear down the whole channel —
        # silently poisoning every other session on this worker.
        if _HDR_LEN + len(payload) > _MAX_FRAME:
            raise IpcError(f"frame too large: {_HDR_LEN + len(payload)} bytes > _MAX_FRAME={_MAX_FRAME}")
        frame = _HDR.pack(request_id, mtype) + payload
        self._send_q.put_nowait(_LEN.pack(len(frame)) + frame)

    async def _writer_loop(self) -> None:
        # finally-teardown: however this loop exits (expected socket error or an
        # unexpected bug), the channel must be torn down so pending futures fail
        # instead of hanging. The exit reason is carried into IpcChannelClosed.
        reason = "writer loop exited"
        try:
            while True:
                frame = await self._send_q.get()
                self._writer.write(frame)
                await self._writer.drain()
        except asyncio.CancelledError:
            raise
        except (ConnectionError, OSError) as exc:
            reason = f"write failed: {exc!r}"
        except Exception as exc:
            reason = f"writer loop crashed: {exc!r}"
        finally:
            self._teardown(IpcChannelClosed(reason))

    async def _reader_loop(self) -> None:
        reason = "reader loop exited"
        try:
            while True:
                (length,) = _LEN.unpack(await self._reader.readexactly(_LEN.size))
                if length < _HDR_LEN or length > _MAX_FRAME:
                    raise IpcError(f"invalid frame length {length}")
                frame = await self._reader.readexactly(length)
                request_id, mtype = _HDR.unpack_from(frame, 0)
                self._dispatch(mtype, request_id, frame[_HDR_LEN:])
        except asyncio.CancelledError:
            raise
        except asyncio.IncompleteReadError:
            reason = "peer closed connection"
        except (IpcError, ConnectionError, OSError) as exc:
            reason = f"read failed: {exc!r}"
        except Exception as exc:
            reason = f"reader loop crashed: {exc!r}"
        finally:
            self._teardown(IpcChannelClosed(reason))

    def _dispatch(self, mtype: int, request_id: int, payload: bytes) -> None:
        if mtype == _TYPE_REPLY:
            fut = self._pending.get(request_id)
            if fut is not None and not fut.done():
                fut.set_result(payload)
        elif mtype == _TYPE_ERROR:
            fut = self._pending.get(request_id)
            if fut is not None and not fut.done():
                fut.set_exception(IpcError(payload.decode("utf-8", "replace")))
        elif mtype == _TYPE_REQUEST:
            if self._request_handler is None:
                logger.error("IpcChannel received a request but has no handler; dropping")
                return
            task = asyncio.ensure_future(self._run_handler(request_id, payload))
            self._handler_tasks.add(task)
            task.add_done_callback(self._handler_tasks.discard)
        else:
            logger.error("IpcChannel received unknown frame type %d; dropping", mtype)

    async def _run_handler(self, request_id: int, payload: bytes) -> None:
        try:
            result = await self._request_handler(payload)
            self._enqueue(_TYPE_REPLY, request_id, result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # handler error → ERROR frame back to caller
            logger.exception("IpcChannel request handler failed")
            self._enqueue(_TYPE_ERROR, request_id, f"{type(exc).__name__}: {exc}".encode())

    def _teardown(self, exc: Exception) -> None:
        if self._closed:
            return
        self._closed = True
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()
        for task in (self._reader_task, self._writer_task):
            if task is not None and task is not asyncio.current_task():
                task.cancel()
        for task in list(self._handler_tasks):
            task.cancel()
        try:
            self._writer.close()
        except Exception:
            pass
        if self._on_close is not None:
            try:
                self._on_close()
            except Exception:
                logger.exception("IpcChannel on_close callback failed")

    async def aclose(self) -> None:
        self._teardown(IpcChannelClosed("channel closed by aclose()"))


async def open_unix_channel(
    sock,
    *,
    request_handler: Callable[[bytes], Awaitable[bytes]] | None = None,
    on_close: Callable[[], None] | None = None,
) -> IpcChannel:
    """Wrap a connected socket (e.g. a ``socket.socketpair`` end) in an IpcChannel."""
    reader, writer = await asyncio.open_connection(sock=sock)
    channel = IpcChannel(reader, writer, request_handler=request_handler, on_close=on_close)
    channel.start()
    return channel
