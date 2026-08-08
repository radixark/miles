"""Standalone launcher + console host for the miles multi-LoRA service.

Runs OUTSIDE the service so the UI exists before the server does: serves
``ui/console.html``, reverse-proxies ``/health`` + ``/v1/*`` to the service
API, and exposes ``/launcher/*`` to start/stop the service itself. One
process, stdlib + fastapi/httpx/uvicorn (all in the training image).

Usage (devbox):
  python ui/launcher.py --port 8067 --save-dir /personal/demo_v1/save \
      --extra-serve-args "--dump-details /personal/demo_v1/dump --use-miles-dashboard"

Then open http://127.0.0.1:8067/ui (over an SSH port-forward) and press
"launch miles server".
"""

import argparse
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSOLE = Path(__file__).resolve().parent / "console.html"


@dataclass
class LauncherConfig:
    port: int = 8067
    api_url: str = "http://127.0.0.1:8068"
    save_dir: str = "/tmp/multi_lora_ui"
    default_slots: int = 4
    extra_serve_args: str = ""
    log_path: str = "/tmp/miles_serve_ui.log"
    repo: Path = field(default_factory=lambda: REPO_ROOT)


def create_app(cfg: LauncherConfig) -> FastAPI:
    app = FastAPI(title="Miles Multi-LoRA Launcher")
    client = httpx.AsyncClient(base_url=cfg.api_url, timeout=httpx.Timeout(30.0, connect=1.5))
    state: dict = {"proc": None}

    async def api_healthy() -> bool:
        try:
            r = await client.get("/health", timeout=1.5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def proc_alive() -> bool:
        return state["proc"] is not None and state["proc"].poll() is None

    def log_tail(lines: int = 20) -> str:
        try:
            with open(cfg.log_path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 8192))
                return b"\n".join(f.read().splitlines()[-lines:]).decode(errors="replace")
        except OSError:
            return ""

    @app.get("/ui")
    async def serve_console():
        # no-cache: the console is edited live; a reload must always fetch it fresh
        return FileResponse(CONSOLE, media_type="text/html", headers={"Cache-Control": "no-cache"})

    @app.get("/launcher/status")
    async def status():
        up = await api_healthy()
        return {
            "serverUp": up,
            "launching": (not up) and proc_alive(),
            "defaultSlots": cfg.default_slots,
            "apiUrl": cfg.api_url,
            "logTail": "" if up else log_tail(),
        }

    @app.post("/launcher/start")
    async def start(request: Request):
        if await api_healthy():
            return {"status": "already-up"}
        if proc_alive():
            return {"status": "already-starting"}
        body = {}
        try:
            body = await request.json()
        except Exception:
            pass
        slots = int(body.get("slots") or cfg.default_slots)
        cmd = (
            f"python examples/multi_lora/run_multi_lora.py serve "
            f"--n-adapters {slots} --save-dir {shlex.quote(cfg.save_dir)}"
        )
        if cfg.extra_serve_args:
            cmd += f" --extra-args {shlex.quote(cfg.extra_serve_args)}"
        log = open(cfg.log_path, "a")
        # New session so the service outlives the launcher if the launcher dies.
        state["proc"] = subprocess.Popen(
            ["bash", "-lc", cmd], cwd=cfg.repo, stdout=log, stderr=log,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        return {"status": "starting", "slots": slots, "log": cfg.log_path}

    @app.post("/launcher/stop")
    async def stop():
        # Blunt but reliable: the serve driver fans out over Ray, so killing
        # the child alone leaves workers behind. Bracketed patterns avoid the
        # pkill-matches-its-own-shell trap.
        subprocess.run(["bash", "-lc", 'pkill -f "[t]rain_multi_lora" || true'], timeout=30)
        subprocess.run(["bash", "-lc", "ray stop --force >/dev/null 2>&1 || true"], timeout=120)
        state["proc"] = None
        return {"status": "stopping"}

    @app.get("/health")
    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "DELETE", "PUT", "PATCH"])
    async def proxy(request: Request, path: str = ""):
        url = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        try:
            upstream = await client.request(
                request.method, url, content=await request.body(),
                headers={"content-type": request.headers.get("content-type", "application/json")},
            )
            return Response(upstream.content, status_code=upstream.status_code,
                            media_type=upstream.headers.get("content-type"))
        except httpx.HTTPError as e:
            return JSONResponse(
                {"error": {"code": 502, "status": "UNAVAILABLE",
                           "message": f"miles server unreachable ({e.__class__.__name__}) — launch it first"}},
                status_code=502,
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8067)
    parser.add_argument("--api-url", default="http://127.0.0.1:8068")
    parser.add_argument("--save-dir", default="/tmp/multi_lora_ui")
    parser.add_argument("--default-slots", type=int, default=4)
    parser.add_argument("--extra-serve-args", default="")
    parser.add_argument("--log-path", default="/tmp/miles_serve_ui.log")
    args = parser.parse_args()
    cfg = LauncherConfig(
        port=args.port, api_url=args.api_url, save_dir=args.save_dir,
        default_slots=args.default_slots, extra_serve_args=args.extra_serve_args,
        log_path=args.log_path,
    )
    uvicorn.run(create_app(cfg), host="0.0.0.0", port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
