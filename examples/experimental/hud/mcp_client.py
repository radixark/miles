"""Minimal synchronous MCP client for HUD environment images.

Why not the ``mcp`` SDK: the images HUD's published tasksets pin (v5-era,
``hud-mcp-python-sdk`` 3.13.x speaking protocol 2025-06-18) reject the
``initialize`` shape that mcp>=2 clients send, and mcp>=2 is what the training
image already has installed for other packages. This speaks the four calls an
environment adapter needs, over streamable HTTP, with no version coupling.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

PROTOCOL_VERSION = "2025-06-18"


class McpError(RuntimeError):
    """A JSON-RPC error, or a malformed reply."""


@dataclass
class ToolResult:
    """The parts of an MCP ``tools/call`` result an env adapter cares about."""

    text: str
    image_b64: str | None
    structured: dict | None

    def payload(self) -> dict:
        """Structured content, falling back to parsing the text block.

        HUD's graders return their EvaluationResult in ``structuredContent``,
        but older builds only put the JSON in a text block.
        """
        if isinstance(self.structured, dict):
            return self.structured
        try:
            parsed = json.loads(self.text)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}


class SyncMCP:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._url = base_url.rstrip("/") + "/mcp"
        self._client = httpx.Client(timeout=timeout)
        self._session_id: str | None = None
        self._next_id = 0

    # ---- wire ----

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self._session_id:
            h["Mcp-Session-Id"] = self._session_id
        return h

    def _post(self, payload: dict) -> httpx.Response:
        return self._client.post(self._url, json=payload, headers=self._headers())

    @staticmethod
    def _extract(body: str, req_id: int) -> dict:
        """Pull the result for *req_id* out of a JSON or SSE response body."""
        for line in body.splitlines():
            if line.startswith("data:"):
                line = line[len("data:") :]
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") != req_id:
                continue
            if "error" in msg:
                raise McpError(str(msg["error"]))
            return msg.get("result", {})
        raise McpError(f"no reply for id={req_id} in {body[:200]!r}")

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        self._next_id += 1
        req_id = self._next_id
        resp = self._post({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}})
        resp.raise_for_status()
        return self._extract(resp.text, req_id)

    # ---- api ----

    def initialize(self, retries: int = 8, delay: float = 4.0) -> dict:
        """Handshake, retrying while the environment's services come up.

        One session per sandbox: the fork these images carry runs its
        ``@mcp.initialize`` hook only for the first session of a process and
        then awaits ``None`` for later ones, so a second session gets -32602.
        """
        last: Exception | None = None
        for _ in range(retries):
            try:
                self._next_id += 1
                req_id = self._next_id
                resp = self._post(
                    {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": PROTOCOL_VERSION,
                            "capabilities": {},
                            "clientInfo": {"name": "miles-hud", "version": "0.1"},
                        },
                    }
                )
                resp.raise_for_status()
                if sid := resp.headers.get("mcp-session-id"):
                    self._session_id = sid
                result = self._extract(resp.text, req_id)
                self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})
                return result
            except Exception as e:  # noqa: BLE001 - services may still be booting
                last = e
                time.sleep(delay)
        raise McpError(f"initialize failed after {retries} attempts: {last}")

    def list_tools(self) -> list[dict]:
        return self._rpc("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> ToolResult:
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        texts: list[str] = []
        image: str | None = None
        for item in result.get("content", []):
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif item.get("type") == "image":
                image = item.get("data")
        return ToolResult("\n".join(texts), image, result.get("structuredContent"))

    def close(self) -> None:
        self._client.close()
