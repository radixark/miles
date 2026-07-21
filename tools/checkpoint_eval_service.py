"""Standalone checkpoint eval service: watch --save-hf output, pin an sglang server
to each complete snapshot, run the miles eval datasets, log at the snapshot's step.
No Ray; restarts resume from a ledger file next to the watch dir.

Example::

    python tools/checkpoint_eval_service.py \\
        --watch-dir /ckpt/exp/hf --hf-checkpoint /models/Qwen3.5-4B --tp 1 \\
        --eval-prompt-data aime /data/aime-2024.jsonl --rm-type dapo --reward-key score
"""

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

from miles.rollout.checkpoint_eval import retarget_args
from miles.utils.hf_config import is_complete_hf_export, looks_like_hf_checkpoint
from miles.utils.http_utils import init_http_client, post

logger = logging.getLogger("checkpoint_eval_service")

QUIESCENCE_SECS = 120.0

# Fields the eval path consumes that are not service flags; values match miles defaults.
EVAL_ARG_DEFAULTS = dict(
    chat_template_path=None,
    custom_generate_function_path=None,
    custom_eval_rollout_log_function_path=None,
    custom_rm_path=None,
    group_rm=False,
    partial_rollout=False,
    mask_offpolicy_in_partial_rollout=False,
    multimodal_keys=None,
    metadata_key="metadata",
    tool_key=None,
    apply_chat_template_kwargs=None,
    rollout_stop=None,
    rollout_stop_token_ids=None,
    rollout_skip_special_tokens=True,
    rollout_max_context_len=None,
    rollout_seed=42,
    rm_url=None,
    sglang_enable_deterministic_inference=False,
    sglang_speculative_algorithm=None,
    sglang_server_concurrency=512,
    use_distributed_post=False,
    use_rollout_routing_replay=False,
    use_rollout_indexer_replay=False,
    use_opd=False,
    lora_rank=0,
    lora_adapter_path=None,
    log_passrate=False,
    log_reward_category=None,
    advantage_estimator="grpo",
    load_debug_rollout_data=None,
    wandb_always_use_train_step=False,
    eval_num_gpus=0,  # the service talks to its own server; no in-job fleet
    ci_test=False,
)


def parse_service_args() -> Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # snapshot source
    parser.add_argument("--watch-dir", type=str, required=True, help="Directory containing HF snapshot subdirs.")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--min-rollout-id", type=int, default=0)
    parser.add_argument(
        "--catchup",
        type=str,
        choices=["all", "latest"],
        default="all",
        help="On startup/backlog: eval every unconsumed snapshot, or skip to the newest.",
    )
    parser.add_argument("--once", action="store_true", help="Process the current backlog and exit.")
    # server
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="Base checkpoint (tokenizer/arch source).")
    parser.add_argument("--server-url", type=str, default=None, help="Attach to a running sglang server.")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--server-port", type=int, default=31000)
    parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.8)
    # eval datasets (mirrors the miles eval surface)
    parser.add_argument("--eval-prompt-data", type=str, nargs="+", default=None)
    parser.add_argument("--eval-config", type=str, default=None)
    parser.add_argument("--eval-input-key", type=str, default=None)
    parser.add_argument("--eval-label-key", type=str, default=None)
    parser.add_argument("--eval-tool-key", type=str, default=None)
    parser.add_argument("--n-samples-per-eval-prompt", type=int, default=1)
    parser.add_argument("--eval-temperature", type=float, default=None)
    parser.add_argument("--eval-top-p", type=float, default=None)
    parser.add_argument("--eval-top-k", type=int, default=None)
    parser.add_argument("--eval-max-response-len", type=int, default=None)
    parser.add_argument("--eval-max-prompt-len", type=int, default=None)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-top-k", type=int, default=-1)
    parser.add_argument("--rollout-max-response-len", type=int, default=8192)
    parser.add_argument("--input-key", type=str, default="prompt")
    parser.add_argument("--label-key", type=str, default=None)
    parser.add_argument("--apply-chat-template", action="store_true", default=False)
    parser.add_argument("--rm-type", type=str, default=None)
    parser.add_argument("--reward-key", type=str, default=None)
    parser.add_argument("--eval-reward-key", type=str, default=None)
    # step mapping (only needed with --wandb-always-use-train-step trainers)
    parser.add_argument("--rollout-batch-size", type=int, default=None)
    parser.add_argument("--n-samples-per-prompt", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    # tracking
    parser.add_argument("--wandb-mode", type=str, choices=["shared", "separate", "off"], default="off")
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--wandb-key", type=str, default=None)
    parser.add_argument("--wandb-host", type=str, default=None)
    return parser.parse_args()


def build_eval_namespace(service_args: Namespace, server_ip: str, server_port: int) -> Namespace:
    from miles.utils.arguments import _resolve_eval_datasets

    args = Namespace(**EVAL_ARG_DEFAULTS)
    for key, value in vars(service_args).items():
        setattr(args, key, value)
    args.eval_datasets = _resolve_eval_datasets(args)
    return retarget_args(args, server_ip, server_port, service_args.num_gpus, service_args.tp)


class SnapshotLedger:
    def __init__(self, watch_dir: Path):
        self.path = watch_dir / ".eval_service_state.json"
        self.consumed: set[int] = set()
        if self.path.exists():
            self.consumed = set(json.loads(self.path.read_text()).get("consumed", []))

    def mark(self, rollout_id: int) -> None:
        self.consumed.add(rollout_id)
        self.path.write_text(json.dumps({"consumed": sorted(self.consumed)}))


def find_ready_snapshots(watch_dir: Path, min_rollout_id: int, consumed: set[int]) -> list[tuple[int, Path]]:
    ready = []
    for child in watch_dir.iterdir():
        if not child.is_dir():
            continue
        match = re.search(r"(\d+)", child.name)
        if match is None:
            continue
        rollout_id = int(match.group(1))
        if rollout_id < min_rollout_id or rollout_id in consumed:
            continue
        if is_complete_hf_export(child):
            ready.append((rollout_id, child))
        elif looks_like_hf_checkpoint(child):
            # Pre-marker checkpoint: accept once quiescent.
            newest_mtime = max(p.stat().st_mtime for p in child.iterdir())
            if time.time() - newest_mtime > QUIESCENCE_SECS:
                ready.append((rollout_id, child))
    return sorted(ready)


def launch_server(service_args: Namespace) -> tuple[subprocess.Popen | None, str, int]:
    if service_args.server_url is not None:
        url = service_args.server_url.removeprefix("http://")
        ip, port = url.split(":")
        return None, ip, int(port)

    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        service_args.hf_checkpoint,
        "--tp",
        str(service_args.tp),
        "--host",
        "127.0.0.1",
        "--port",
        str(service_args.server_port),
        "--mem-fraction-static",
        str(service_args.sglang_mem_fraction_static),
        "--trust-remote-code",
    ]
    logger.info(f"Launching sglang server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    return proc, "127.0.0.1", service_args.server_port


async def wait_server_healthy(ip: str, port: int, timeout: float = 1800.0) -> None:
    import httpx

    deadline = time.time() + timeout
    async with httpx.AsyncClient() as client:
        while time.time() < deadline:
            try:
                # /health_generate runs a tiny generation; 200 body is not JSON.
                response = await client.get(f"http://{ip}:{port}/health_generate", timeout=60)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(5)
    raise TimeoutError(f"sglang server at {ip}:{port} not healthy after {timeout}s")


async def eval_snapshot(args: Namespace, state, cache: dict, rollout_id: int, snapshot: Path) -> None:
    from miles.ray.rollout.metrics import log_eval_rollout_data
    from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
    from miles.utils.http_utils import get

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    start = time.time()
    weight_version = str(rollout_id)
    await post(
        f"{url}/update_weights_from_disk",
        {"model_path": str(snapshot), "weight_version": weight_version},
    )
    info = await get(f"{url}/model_info")
    if str(info.get("weight_version")) != weight_version:
        raise RuntimeError(
            f"weight_version pin failed: engine reports {info.get('weight_version')}, expected {weight_version}"
        )

    results = await run_eval_datasets(state, cache)
    extra = {"eval/duration_seconds": time.time() - start}
    log_eval_rollout_data(rollout_id, args, results, extra)


def init_service_tracking(args: Namespace) -> None:
    if args.wandb_mode == "off":
        args.use_wandb = False
        return
    from miles.utils.tracking_utils.tracking import init_tracking

    args.use_wandb = True
    if args.wandb_mode == "shared" and not args.wandb_run_id:
        raise ValueError("--wandb-mode shared requires --wandb-run-id of the training run")
    init_tracking(args, primary=args.wandb_mode == "separate")


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    service_args = parse_service_args()
    watch_dir = Path(service_args.watch_dir)
    if not watch_dir.is_dir():
        raise FileNotFoundError(f"--watch-dir {watch_dir} does not exist")

    proc, server_ip, server_port = launch_server(service_args)
    try:
        args = build_eval_namespace(service_args, server_ip, server_port)
        init_http_client(args)
        await wait_server_healthy(server_ip, server_port)
        init_service_tracking(args)

        from miles.rollout.inference_rollout.inference_rollout_common import GenerateState

        state = GenerateState(args)
        cache: dict = {}
        ledger = SnapshotLedger(watch_dir)

        while True:
            ready = find_ready_snapshots(watch_dir, service_args.min_rollout_id, ledger.consumed)
            if ready and service_args.catchup == "latest":
                for rollout_id, _ in ready[:-1]:
                    ledger.mark(rollout_id)
                ready = ready[-1:]
            for rollout_id, snapshot in ready:
                logger.info(f"Evaluating snapshot {snapshot} (rollout_id={rollout_id})")
                try:
                    await eval_snapshot(args, state, cache, rollout_id, snapshot)
                    ledger.mark(rollout_id)
                except Exception:
                    logger.exception(f"Eval of {snapshot} failed; will retry next scan")
            if service_args.once:
                break
            await asyncio.sleep(service_args.poll_interval)
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    asyncio.run(main())
