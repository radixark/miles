"""Smoke test for multi-LoRA service mode: register/deregister against a running
trainer, using step counts as the race-free progress signal.

Usage: python examples/multi_lora/service_smoke.py --api-url http://HOST:8068 \\
    --data /root/gsm8k/train.parquet --input-key messages --label-key label --rm-type math
"""

import argparse
import sys
import time

import httpx

POLL_INTERVAL_S = 5.0


class SmokeFailure(Exception):
    pass


class ServiceClient:
    def __init__(self, api_url: str, timeout_s: float):
        self.api_url = api_url.rstrip("/")
        self.timeout_s = timeout_s
        self.http = httpx.Client(timeout=30.0)

    def active_adapters(self) -> dict:
        response = self.http.get(f"{self.api_url}/adapter_runs")
        response.raise_for_status()
        return {
            status["name"]: {"slot": status["slot"], "version": status["version"], "step": status["step"]}
            for status in response.json()["adapters"]
            if status["state"] == "ACTIVE"
        }

    def register(self, name: str, config: dict) -> httpx.Response:
        return self.http.post(f"{self.api_url}/adapter_runs", json={"name": name, "config": config})

    def deregister(self, name: str) -> None:
        response = self.http.delete(f"{self.api_url}/adapter_runs/{name}")
        response.raise_for_status()

    def wait_for(self, description: str, predicate) -> dict:
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            try:
                adapters = self.active_adapters()
            except httpx.HTTPError as e:
                print(f"  ... api not reachable yet ({e})")
                adapters = None
            if adapters is not None:
                if predicate(adapters):
                    print(f"  ok: {description}  (active={adapters})")
                    return adapters
                print(f"  waiting for {description}  (active={adapters})")
            time.sleep(POLL_INTERVAL_S)
        raise SmokeFailure(f"timed out after {self.timeout_s}s waiting for: {description}")

    def wait_for_step(self, name: str, min_step: int) -> None:
        self.wait_for(
            f"'{name}' to reach step {min_step}",
            lambda adapters: name in adapters and adapters[name]["step"] >= min_step,
        )

    def register_when_allowed(self, name: str, config: dict) -> None:
        """Registration is rejected while a same-named adapter is cleaning up;
        retry until the name frees."""
        deadline = time.time() + self.timeout_s
        while time.time() < deadline:
            response = self.register(name, config)
            if response.status_code == 200:
                print(f"  ok: registered '{name}'")
                return
            print(f"  register '{name}' rejected ({response.status_code}): {response.text[:200]}")
            time.sleep(POLL_INTERVAL_S)
        raise SmokeFailure(f"timed out registering '{name}'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True, help="controller API listener, e.g. http://host:8068")
    parser.add_argument("--data", required=True, help="prompt dataset path for the test adapters")
    parser.add_argument("--input-key", default="text")
    parser.add_argument("--label-key", default="label")
    parser.add_argument("--rm-type", default="math")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--save", default=None, help="per-adapter save dir root override (default: trainer --save)")
    parser.add_argument("--steps", type=int, default=2, help="training steps to wait for per phase")
    parser.add_argument(
        "--num-step-smoke",
        type=int,
        default=1,
        help="num_step used by the auto-deregister smoke adapter",
    )
    parser.add_argument("--timeout", type=float, default=1800.0, help="per-phase timeout in seconds")
    args = parser.parse_args()

    def config(name: str) -> dict:
        cfg = {
            "rank": args.rank,
            "alpha": args.alpha,
            "data": args.data,
            "input_key": args.input_key,
            "label_key": args.label_key,
            "rm_type": args.rm_type,
        }
        if args.save:
            cfg["save"] = f"{args.save}/{name}"
        return cfg

    client = ServiceClient(args.api_url, args.timeout)
    try:
        print("phase 1: api reachable, no active adapters expected")
        client.wait_for("api reachable", lambda adapters: True)

        print("phase 2: register smoke_auto with num_step; expect auto-deregister after committed steps")
        auto_cfg = config("smoke_auto")
        auto_cfg["num_step"] = args.num_step_smoke
        client.register_when_allowed("smoke_auto", auto_cfg)
        client.wait_for_step("smoke_auto", args.num_step_smoke)
        client.wait_for("'smoke_auto' auto-deregistered", lambda adapters: "smoke_auto" not in adapters)

        print("phase 3: register smoke_a; expect promotion + training progress")
        client.register_when_allowed("smoke_a", config("smoke_a"))
        client.wait_for_step("smoke_a", args.steps)

        print("phase 4: register smoke_b mid-run; both must train")
        client.register_when_allowed("smoke_b", config("smoke_b"))
        client.wait_for_step("smoke_b", args.steps)

        print("phase 5: deregister smoke_a mid-run; smoke_b must keep training")
        step_b = client.active_adapters()["smoke_b"]["step"]
        client.deregister("smoke_a")
        client.wait_for("'smoke_a' gone from active set", lambda adapters: "smoke_a" not in adapters)
        client.wait_for_step("smoke_b", step_b + 1)

        print("phase 6: re-register the name smoke_a (waits out cleanup, reuses slot)")
        client.register_when_allowed("smoke_a", config("smoke_a"))
        client.wait_for_step("smoke_a", 1)

        print("phase 7: deregister everything; service should drain to idle")
        client.deregister("smoke_a")
        client.deregister("smoke_b")
        client.wait_for("no active adapters", lambda adapters: not adapters)

        print("SMOKE TEST PASSED")
        return 0
    except SmokeFailure as failure:
        print(f"SMOKE TEST FAILED: {failure}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
