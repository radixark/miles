"""ImageReward as an HTTP service for `reward.py`: score PNG bytes against a prompt. It renders nothing.

It runs outside Miles' environment, since ImageReward needs transformers 4.x: a venv with `image-reward==1.5`, `torch`
and `pillow` is enough (README.md has the pins we ran it with). One process serves one GPU and scores one image at a
time; it fits beside other work on a GPU.

    CUDA_VISIBLE_DEVICES=7 python examples/experimental/prompt_enhancer/scorer_server.py --port 18200

POST /score {"image_b64": <base64 PNG>, "prompt": <text>} answers {"image_reward": <float>}; a body it cannot read
answers 400.
"""

import argparse
import base64
import binascii
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import ImageReward as RM
import torch
from PIL import Image, UnidentifiedImageError


class _Scorer:
    def __init__(self, device: str):
        self.model = RM.load("ImageReward-v1.0", device=device)
        self.lock = threading.Lock()

    def score(self, png: bytes, prompt: str) -> float:
        image = Image.open(io.BytesIO(png)).convert("RGB")
        with self.lock, torch.no_grad():
            return float(self.model.score(prompt, [image]))


def _handler(scorer: _Scorer):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/score":
                self.send_error(404)
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                png = base64.b64decode(body["image_b64"], validate=True)
                reply = {"image_reward": scorer.score(png, body["prompt"])}
            except (ValueError, KeyError, TypeError, binascii.Error, UnidentifiedImageError) as e:
                self.send_error(400, explain=repr(e))
                return
            self._send(200, reply)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True})
            else:
                self.send_error(404)

        def _send(self, status: int, reply: dict):
            data = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _handler(_Scorer(args.device)))
    print(f"scorer server ready on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
