#!/usr/bin/env python3
# Copied from sgl-project/sglang-torchtpu .github/tpu-ci/run_cloudbuild.py (main 807e555d43).
"""Create and run a Cloud Build via the REST API, print the pushed image digest.

`gcloud builds submit` runs a client-side staging / service-usage pre-flight that is
incompatible with Workload Identity Federation credentials: it fails accessing the
``<project>_cloudbuild`` bucket even though the service account can write it (proven
by `gcloud storage cp` succeeding with the same token). This talks to the Cloud Build
REST API directly instead -- the source is pre-uploaded to GCS by the caller and we
POST a Build that points at it, so that broken pre-flight never runs.

When the build config sets ``logsBucket``, the build's GCS log object is streamed
to stderr while the build runs, so the full Cloud Build log lands in the GitHub
Actions job log. External collaborators cannot open this project's Cloud Console /
Cloud Logging, so the Actions job log is the only build log they can read.

``--max-attempts N`` re-creates the build after a retriable terminal status
(FAILURE / INTERNAL_ERROR / TIMEOUT / EXPIRED -- not CANCELLED) to absorb
intermittent failures such as network flakes while pulling base images.

Progress goes to stderr; the bare image digest (``sha256:...``) is the ONLY thing
printed to stdout, so callers can capture it with ``digest="$(... )"``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://cloudbuild.googleapis.com/v1"
GCS_API = "https://storage.googleapis.com/storage/v1"
_TERMINAL = {"SUCCESS", "FAILURE", "INTERNAL_ERROR", "TIMEOUT", "CANCELLED", "EXPIRED"}
# CANCELLED is deliberate (a human stopped it), everything else is worth one more try.
_RETRIABLE = {"FAILURE", "INTERNAL_ERROR", "TIMEOUT", "EXPIRED"}


def _log(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def _load_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:  # not guaranteed on the runner
        import subprocess

        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyyaml"], check=True)
        import yaml
    with open(path) as f:
        return yaml.safe_load(f)


class _LogTail:
    """Incrementally copy a build's GCS log object to stderr as it grows.

    Cloud Build appends to ``<logsBucket>/log-<build_id>.txt`` while the build
    runs; ranged reads pick up only the new bytes. Read failures are non-fatal:
    forwarding the log is best-effort and never affects the build result.
    """

    def __init__(self, logs_bucket: str, build_id: str, token: str) -> None:
        path = logs_bucket.removeprefix("gs://")
        bucket, _, prefix = path.partition("/")
        obj = f"{prefix}/log-{build_id}.txt" if prefix else f"log-{build_id}.txt"
        self._url = f"{GCS_API}/b/{bucket}/o/{urllib.parse.quote(obj, safe='')}?alt=media"
        self._token = token
        self._offset = 0
        self._warned = False

    def poll(self) -> None:
        req = urllib.request.Request(
            self._url,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Range": f"bytes={self._offset}-",
            },
        )
        try:
            with urllib.request.urlopen(req) as r:
                body = r.read()
                if r.status == 200 and self._offset:  # server ignored the Range
                    body = body[self._offset :]
        except urllib.error.HTTPError as e:
            # 404: log object not created yet; 416: no bytes past offset yet.
            if e.code not in (404, 416) and not self._warned:
                _log(f"(log tail unavailable: HTTP {e.code}; use logUrl instead)")
                self._warned = True
            return
        except urllib.error.URLError:
            return
        if body:
            self._offset += len(body)
            sys.stderr.write(body.decode("utf-8", errors="replace"))
            sys.stderr.flush()


def _create_build(call, project: str, build: dict) -> str | None:
    """POST the build; return its id, or None on a transient (retriable) error."""
    code, resp = call("POST", f"{API}/projects/{project}/builds", build)
    if code != 200:
        _log(f"builds.create failed: HTTP {code}")
        _log(json.dumps(resp, indent=2)[:2000])
        if code == 429 or code >= 500:
            return None
        sys.exit(1)  # 4xx: config/auth problem, retrying cannot help
    build_id = resp["metadata"]["build"]["id"]
    _log(f"build {build_id} created; logs: {resp['metadata']['build'].get('logUrl', '?')}")
    return build_id


def _wait_for_build(call, project: str, build_id: str, token: str) -> tuple[str | None, dict]:
    """Poll until the build reaches a terminal status, streaming its GCS log."""
    status = last = None
    tail: _LogTail | None = None
    b: dict = {}
    for _ in range(240):  # 240 * 15s = up to 60 min
        code, b = call("GET", f"{API}/projects/{project}/builds/{build_id}")
        if code != 200:
            _log(f"builds.get HTTP {code}: {json.dumps(b)[:300]}")
            time.sleep(15)
            continue
        if tail is None and b.get("logsBucket"):
            tail = _LogTail(b["logsBucket"], build_id, token)
        if tail is not None:
            tail.poll()
        status = b.get("status")
        if status != last:
            _log(f"  status: {status}")
            last = status
        if status in _TERMINAL:
            break
        time.sleep(15)
    if tail is not None:  # the log object can trail the terminal status slightly
        time.sleep(5)
        tail.poll()
    return status, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--config", required=True, help="Cloud Build config YAML")
    ap.add_argument("--bucket", required=True, help="GCS bucket of the staged source")
    ap.add_argument("--object", required=True, help="GCS object of the staged source")
    ap.add_argument("--tag", required=True, help="value for the _TAG substitution")
    ap.add_argument(
        "--substitutions",
        nargs="*",
        default=[],
        help="additional substitutions in KEY=VALUE format",
    )
    ap.add_argument("--token-file", required=True, help="file holding a bearer token")
    ap.add_argument("--timeout", default="3600s", help="build timeout")
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="total build attempts; >1 retries retriable failures",
    )
    args = ap.parse_args()

    with open(args.token_file) as f:
        token = f.read().strip()

    def call(method: str, url: str, body: dict | None = None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.load(e)
            except Exception:
                return e.code, {"error": e.read().decode()[:500]}

    cfg = _load_config(args.config)
    subs = dict(cfg.get("substitutions", {}))
    subs["_TAG"] = args.tag
    for item in args.substitutions:
        if "=" in item:
            k, v = item.split("=", 1)
            subs[k] = v
    build = {
        "source": {"storageSource": {"bucket": args.bucket, "object": args.object}},
        "steps": cfg["steps"],
        "images": cfg.get("images", []),
        "options": cfg.get("options", {}),
        "substitutions": subs,
        "timeout": args.timeout,
    }
    if "serviceAccount" in cfg:  # run the build as the configured SA (tpu-ci)
        build["serviceAccount"] = cfg["serviceAccount"]
    if "logsBucket" in cfg:  # GCS log copy, streamed to stderr by _LogTail
        build["logsBucket"] = cfg["logsBucket"]

    status = None
    b: dict = {}
    for attempt in range(1, args.max_attempts + 1):
        _log(
            f"builds.create (source gs://{args.bucket}/{args.object}, "
            f"_TAG={args.tag}, attempt {attempt}/{args.max_attempts})"
        )
        build_id = _create_build(call, args.project, build)
        if build_id is not None:
            status, b = _wait_for_build(call, args.project, build_id, token)
            if status == "SUCCESS":
                break
        if attempt < args.max_attempts and (build_id is None or status in _RETRIABLE):
            _log(f"attempt {attempt} ended with {status or 'create error'}; " "retrying in 30s")
            status, b = None, {}
            time.sleep(30)
            continue
        break

    if status != "SUCCESS":
        _log(f"BUILD {status or 'CREATE_FAILED'}")
        if b.get("failureInfo"):
            _log("failureInfo: " + json.dumps(b["failureInfo"]))
        _log("logUrl: " + b.get("logUrl", "?"))
        sys.exit(1)

    images = b.get("results", {}).get("images", [])
    digest = images[0]["digest"] if images else ""
    if not digest:
        _log("build SUCCESS but no image digest in results")
        sys.exit(1)
    _log(f"SUCCESS: {images[0].get('name', '')}@{digest}")
    print(digest)  # stdout: bare digest only


if __name__ == "__main__":
    main()
