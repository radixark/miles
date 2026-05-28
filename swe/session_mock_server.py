#!/usr/bin/env python3
"""在 30000 端口监听，用于测试 miles-session Service / 跨 Pod 连通性。

用法（在 worker-0 上，与 miles-session targetPort 一致）:
    python3 session_mock_server.py

其它 Pod 测试:
    curl -s http://miles-session.nlp-train.svc.cluster.local:30000/health
"""

from __future__ import annotations

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [mock] %(message)s")
logger = logging.getLogger(__name__)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        logger.info("%s - %s", self.client_address[0], fmt % args)

    def _ok(self) -> None:
        body = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._ok()

    def do_POST(self) -> None:
        self._ok()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=30000)
    args = p.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("listening on %s:%s", args.host, args.port)
    srv.serve_forever()


if __name__ == "__main__":
    main()
