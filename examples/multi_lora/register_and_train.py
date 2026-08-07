"""Register LoRA adapter runs against a live multi-LoRA service (started via
``run_multi_lora.py serve``) and watch them train to completion.

Usage:
  python examples/multi_lora/register_and_train.py \\
      --api-url http://127.0.0.1:8068 \\
      --adapter gsm8k=examples/multi_lora/adapters/gsm8k.yaml \\
      --adapter dapo_math=examples/multi_lora/adapters/dapo_math.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import httpx
import yaml

TERMINAL_STATES = {"CLEANUP", "COMPLETED"}


def parse_adapter_spec(spec: str) -> tuple[str, dict]:
    """NAME=path/to/adapter.yaml -> (name, config dict for the register call)."""
    name, sep, path = spec.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=path/to/adapter.yaml, got {spec!r}")
    with open(Path(path)) as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "data" not in config:
        raise argparse.ArgumentTypeError(f"{path}: an adapter yaml must at least set 'data'")
    return name, config


class AdapterServiceClient:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.http = httpx.Client(timeout=30.0)

    def runs(self) -> dict[str, dict]:
        response = self.http.get(f"{self.api_url}/adapter_runs")
        response.raise_for_status()
        return {status["name"]: status for status in response.json()["adapters"]}

    def register(self, name: str, config: dict, retry_window_s: float = 120.0) -> None:
        """Register a run; a name whose previous run is still cleaning up frees
        after that run's final checkpoint lands, so retry briefly."""
        deadline = time.time() + retry_window_s
        while True:
            response = self.http.post(f"{self.api_url}/adapter_runs", json={"name": name, "config": config})
            if response.status_code == 200:
                print(f"registered '{name}': {response.json()}")
                return
            if time.time() >= deadline:
                raise RuntimeError(f"registering '{name}' failed: {response.status_code} {response.text[:200]}")
            print(f"register '{name}' rejected ({response.status_code}); retrying: {response.text[:120]}")
            time.sleep(5.0)

    def deregister(self, name: str) -> None:
        self.http.delete(f"{self.api_url}/adapter_runs/{name}")


def watch(client: AdapterServiceClient, names: list[str], poll_s: float, timeout_s: float) -> int:
    """Print a status line per poll until every named run leaves the registry
    (or reaches a terminal state). Returns the number of unfinished runs."""
    pending = set(names)
    deadline = time.time() + timeout_s
    while pending and time.time() < deadline:
        runs = client.runs()
        done = {name for name in pending if name not in runs or runs[name].get("state") in TERMINAL_STATES}
        for name in sorted(done):
            print(f"'{name}' completed")
        pending -= done
        if pending:
            line = "  ".join(
                f"{name}[{run.get('state')}] slot={run.get('slot')} step={run.get('step')} v{run.get('version')}"
                for name, run in sorted(runs.items())
                if name in pending
            )
            print(line or f"waiting for promotion: {sorted(pending)}")
            time.sleep(poll_s)
    return len(pending)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-url", required=True, help="controller API, e.g. http://HOST:8068")
    parser.add_argument(
        "--adapter",
        dest="adapters",
        action="append",
        type=parse_adapter_spec,
        required=True,
        metavar="NAME=YAML",
        help="adapter run to register (repeatable); YAML follows the adapter.yaml schema",
    )
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=3600.0, help="max seconds to wait for all runs to finish")
    args = parser.parse_args()

    client = AdapterServiceClient(args.api_url)
    names = [name for name, _ in args.adapters]
    try:
        for name, config in args.adapters:
            client.register(name, config)
        unfinished = watch(client, names, args.poll_interval, args.timeout)
        if unfinished:
            print(f"timed out with {unfinished} run(s) still active", file=sys.stderr)
            return 1
        print(
            "all adapter runs completed; checkpoints are under each run's 'save' dir "
            "(or the trainer's --save root by default)"
        )
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted; deregistering this client's runs")
        for name in names:
            client.deregister(name)
        return 130


if __name__ == "__main__":
    sys.exit(main())
