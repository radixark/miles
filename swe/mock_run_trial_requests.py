#!/usr/bin/env python3
"""为 server._run_trial 构建虚拟 RunRequest，并可选地执行校验或完整 trial。

用法:
    # 仅打印各场景的 request JSON（不调用 Harbor）
    python mock_run_trial_requests.py --list

    # 跑校验类场景（无效 instance_id、TaskNotFound 等，不启动完整 Docker trial）
    python mock_run_trial_requests.py --dry-run

    # 对指定场景真正调用 _run_trial（需要 Harbor、Docker、可连通的 LLM）
    HARBOR_TASKS_DIR=/fs/nlp-intern/houjue/harbor_project/datasets/swe-bench-verified \\
        python mock_run_trial_requests.py --execute --scenario mini_swe_openai

    # 并发模拟 N 个请求（默认直连 _run_trial；加 --http 则打正在运行的 server）
    python mock_run_trial_requests.py --execute --scenario mini_swe_openai \\
        --count 8 --concurrency 4 --scan-tasks 8

    # 经 HTTP 打 server（会走 server 内置 semaphore）
    python mock_run_trial_requests.py --execute --http http://127.0.0.1:11000 \\
        --count 8 --concurrency 8 --scan-tasks 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import server
from server import RunRequest, _run_trial  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.getenv("MOCK_LLM_BASE_URL", "http://172.31.72.24:31720/v1/chat/completions")
DEFAULT_MODEL = os.getenv("MOCK_MODEL_NAME", "openai/model")
DEFAULT_INSTANCE = os.getenv("MOCK_INSTANCE_ID", "astropy__astropy-12907")
DEFAULT_TASKS_DIR = os.getenv(
    "HARBOR_TASKS_DIR",
    "/fs/nlp-intern/yangchengyi/harbor/datasets/swegym",
)


def _base_kwargs(**overrides: Any) -> dict[str, Any]:
    """与 swe_agent_function.run 发往 /run 的字段对齐的最小合法 payload。"""
    return {
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "sampling_params": {"temperature": 1.0, "max_tokens": 40960},
        "api_key": "dummy",
        "instance_id": DEFAULT_INSTANCE,
        "agent_name": "mini-swe-agent",
        **overrides,
    }


@dataclass(frozen=True)
class MockScenario:
    name: str
    description: str
    builder: Callable[[], RunRequest]
    expected_exit_status: str | None = None
    needs_full_trial: bool = False


def build_mini_swe_openai() -> RunRequest:
    return RunRequest(**_base_kwargs())


def build_mini_swe_hosted_vllm() -> RunRequest:
    return RunRequest(**_base_kwargs(model="hosted_vllm/Qwen3-8B"))


def build_terminus_host_agent() -> RunRequest:
    return RunRequest(
        **_base_kwargs(agent_name="terminus-2", model="openai/gpt-4o-mini"),
    )


def build_with_metadata_extras() -> RunRequest:
    return RunRequest(
        **_base_kwargs(
            max_seq_len=16384,
            session_server_id="router-pod-0:30000",
            session_server_instance_id="worker-3",
            custom_tag="mock-rollout",
        ),
    )


def build_invalid_empty_instance_id() -> RunRequest:
    return RunRequest(**_base_kwargs(instance_id=""))


def build_invalid_bad_chars() -> RunRequest:
    return RunRequest(**_base_kwargs(instance_id="../etc/passwd"))


def build_invalid_path_traversal() -> RunRequest:
    return RunRequest(**_base_kwargs(instance_id=".."))


def build_invalid_task_not_found() -> RunRequest:
    return RunRequest(**_base_kwargs(instance_id="nonexistent__task-00000"))


def build_valid_swe_instance() -> RunRequest:
    return RunRequest(**_base_kwargs(instance_id="astropy__astropy-12907"))


SCENARIOS: list[MockScenario] = [
    MockScenario(
        "mini_swe_openai",
        "标准 mini-swe-agent + openai/{model}，Docker 环境",
        build_mini_swe_openai,
        needs_full_trial=True,
    ),
    MockScenario(
        "mini_swe_hosted_vllm",
        "hosted_vllm 模型名，触发 model_info 配置",
        build_mini_swe_hosted_vllm,
        needs_full_trial=True,
    ),
    MockScenario(
        "terminus_host_agent",
        "terminus-2 宿主机 agent",
        build_terminus_host_agent,
        needs_full_trial=True,
    ),
    MockScenario(
        "with_metadata_extras",
        "带 extra 字段（Pydantic extra=allow）",
        build_with_metadata_extras,
        needs_full_trial=True,
    ),
    MockScenario(
        "valid_swe_instance",
        "合法 SWE-bench instance_id",
        build_valid_swe_instance,
        needs_full_trial=True,
    ),
    MockScenario(
        "invalid_empty_instance_id",
        "空 instance_id",
        build_invalid_empty_instance_id,
        expected_exit_status="InvalidInstanceId",
    ),
    MockScenario(
        "invalid_bad_chars",
        "instance_id 含非法字符",
        build_invalid_bad_chars,
        expected_exit_status="InvalidInstanceId",
    ),
    MockScenario(
        "invalid_path_traversal",
        "路径穿越 instance_id",
        build_invalid_path_traversal,
        expected_exit_status="InvalidInstanceId",
    ),
    MockScenario(
        "invalid_task_not_found",
        "格式合法但任务目录不存在",
        build_invalid_task_not_found,
        expected_exit_status="TaskNotFound",
    ),
]


def _scenario_map() -> dict[str, MockScenario]:
    return {s.name: s for s in SCENARIOS}


def request_to_dict(req: RunRequest) -> dict[str, Any]:
    return req.model_dump(mode="json")


def _select_scenarios(names: list[str] | None) -> list[MockScenario]:
    if not names:
        return list(SCENARIOS)
    sm = _scenario_map()
    unknown = [n for n in names if n not in sm]
    if unknown:
        raise SystemExit(f"未知场景: {unknown}\n可用: {list(sm.keys())}")
    return [sm[n] for n in names]


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


async def _run_one(scenario: MockScenario, instance_id: str | None = None) -> dict[str, Any]:
    req = scenario.builder()
    if instance_id is not None:
        req = req.model_copy(update={"instance_id": instance_id})
    result = await _run_trial(req)
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "request": request_to_dict(req),
        "result": result,
    }


def _list_task_instance_ids(tasks_dir: Path, limit: int) -> list[str]:
    if not tasks_dir.is_dir():
        raise SystemExit(f"HARBOR_TASKS_DIR 不存在或不是目录: {tasks_dir}")
    ids = sorted(
        p.name
        for p in tasks_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )
    if not ids:
        raise SystemExit(f"任务目录为空: {tasks_dir}")
    return ids[:limit]


def _resolve_instance_ids(
    *,
    count: int,
    explicit: list[str] | None,
    scan_tasks: int | None,
    tasks_dir: Path,
    fallback: str,
) -> list[str]:
    if explicit:
        if scan_tasks:
            logger.warning("--instance-ids 与 --scan-tasks 同时指定，忽略 --scan-tasks")
        base = explicit
    elif scan_tasks:
        base = _list_task_instance_ids(tasks_dir, scan_tasks)
    else:
        base = [fallback]
    if len(base) >= count:
        return base[:count]
    # 不足时循环复用（同一 instance 并发压测）
    return [base[i % len(base)] for i in range(count)]


async def _http_post_run(server_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = server_url.rstrip("/") + "/run"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    def _sync() -> dict[str, Any]:
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            return {
                "reward": 0.0,
                "exit_status": f"HTTPError:{e.code}",
                "agent_metrics": {},
                "eval_report": {"error": detail},
            }

    return await asyncio.to_thread(_sync)


@dataclass
class _JobResult:
    job_id: int
    instance_id: str
    started_at: float
    finished_at: float
    ok: bool
    exit_status: str
    reward: float
    error: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


async def _run_job(
    job_id: int,
    scenario: MockScenario,
    instance_id: str,
    *,
    sem: asyncio.Semaphore,
    http_url: str | None,
    use_server_semaphore: bool,
) -> _JobResult:
    req = scenario.builder()
    req = req.model_copy(update={"instance_id": instance_id})
    payload = request_to_dict(req)
    t0 = time.perf_counter()

    async def _execute() -> dict[str, Any]:
        if http_url:
            return await _http_post_run(http_url, payload)
        if use_server_semaphore:
            async with server.get_semaphore():
                return await _run_trial(req)
        return await _run_trial(req)

    try:
        async with sem:
            result = await _execute()
        t1 = time.perf_counter()
        return _JobResult(
            job_id=job_id,
            instance_id=instance_id,
            started_at=t0,
            finished_at=t1,
            ok=True,
            exit_status=str(result.get("exit_status", "")),
            reward=float(result.get("reward", 0.0)),
            detail={"request": payload, "result": result},
        )
    except Exception as e:
        t1 = time.perf_counter()
        logger.exception("job %s failed: %s", job_id, e)
        return _JobResult(
            job_id=job_id,
            instance_id=instance_id,
            started_at=t0,
            finished_at=t1,
            ok=False,
            exit_status="Exception",
            reward=0.0,
            error=f"{type(e).__name__}: {e}",
            detail={"request": payload},
        )


def _init_server_semaphore(concurrency: int) -> None:
    server._semaphore = asyncio.Semaphore(concurrency)  # noqa: SLF001
    logger.info("已初始化 server semaphore，max_concurrent=%s", concurrency)


def _print_concurrent_summary(results: list[_JobResult], wall_s: float) -> None:
    ok = sum(1 for r in results if r.ok)
    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.exit_status] = by_status.get(r.exit_status, 0) + 1

    print("\n" + "=" * 60)
    print(f"并发汇总: total={len(results)} ok={ok} wall_time={wall_s:.2f}s")
    print(f"exit_status 分布: {by_status}")
    durations = [r.finished_at - r.started_at for r in results]
    if durations:
        print(
            f"单请求耗时(s): min={min(durations):.2f} "
            f"max={max(durations):.2f} avg={sum(durations)/len(durations):.2f}"
        )
    print("=" * 60)
    for r in sorted(results, key=lambda x: x.job_id):
        dt = r.finished_at - r.started_at
        line = (
            f"  [#{r.job_id:03d}] {r.instance_id} "
            f"exit={r.exit_status!r} reward={r.reward} time={dt:.2f}s"
        )
        if r.error:
            line += f" err={r.error}"
        print(line)


async def cmd_concurrent_execute(
    names: list[str] | None,
    *,
    count: int,
    concurrency: int,
    instance_ids: list[str] | None,
    scan_tasks: int | None,
    http_url: str | None,
    use_server_semaphore: bool,
    dump_json: bool,
) -> int:
    os.environ.setdefault("HARBOR_TASKS_DIR", DEFAULT_TASKS_DIR)
    tasks_dir = Path(os.environ["HARBOR_TASKS_DIR"]).resolve()

    scenarios = _select_scenarios(names)
    if not scenarios:
        raise SystemExit("未选择 scenario，例如 --scenario mini_swe_openai")
    scenario = scenarios[0]
    if len(scenarios) > 1:
        logger.warning("多个 scenario 指定时仅使用第一个: %s", scenario.name)
    if not scenario.needs_full_trial:
        logger.info("使用快速失败/校验类场景 %s 做并发连通性测试", scenario.name)

    ids = _resolve_instance_ids(
        count=count,
        explicit=instance_ids,
        scan_tasks=scan_tasks,
        tasks_dir=tasks_dir,
        fallback=DEFAULT_INSTANCE,
    )

    if http_url:
        use_server_semaphore = False
        logger.info("HTTP 模式: %s（并发由 server 端 semaphore 限制）", http_url)
    elif use_server_semaphore:
        _init_server_semaphore(concurrency)

    client_sem = asyncio.Semaphore(concurrency)
    logger.info(
        "发起并发请求: count=%s client_concurrency=%s scenario=%s instances(sample)=%s",
        count,
        concurrency,
        scenario.name,
        ids[:3],
    )

    wall_t0 = time.perf_counter()
    results = await asyncio.gather(
        *[
            _run_job(
                job_id=i,
                scenario=scenario,
                instance_id=ids[i],
                sem=client_sem,
                http_url=http_url,
                use_server_semaphore=use_server_semaphore,
            )
            for i in range(count)
        ]
    )
    wall_s = time.perf_counter() - wall_t0
    _print_concurrent_summary(results, wall_s)

    if dump_json:
        _print_json(
            [
                {
                    "job_id": r.job_id,
                    "instance_id": r.instance_id,
                    "ok": r.ok,
                    "exit_status": r.exit_status,
                    "reward": r.reward,
                    "duration_sec": r.finished_at - r.started_at,
                    "error": r.error,
                    **r.detail,
                }
                for r in results
            ]
        )
    return 0


async def cmd_list(names: list[str] | None) -> int:
    for s in _select_scenarios(names):
        req = s.builder()
        print(f"\n=== {s.name}: {s.description} ===")
        if s.expected_exit_status:
            print(f"    (dry-run 期望 exit_status={s.expected_exit_status!r})")
        if s.needs_full_trial:
            print("    (需 --execute 才会跑完整 Harbor trial)")
        _print_json(request_to_dict(req))
    return 0


async def cmd_dry_run(names: list[str] | None) -> int:
    os.environ.setdefault("HARBOR_TASKS_DIR", DEFAULT_TASKS_DIR)
    scenarios = _select_scenarios(names)
    validation = [s for s in scenarios if s.expected_exit_status is not None]
    if not validation:
        print("没有可 dry-run 的校验场景；请去掉 --scenario 或选用 invalid_* 场景", file=sys.stderr)
        return 2

    failed = 0
    for s in validation:
        out = await _run_one(s)
        _print_json(out)
        got = out["result"]["exit_status"]
        if got != s.expected_exit_status:
            logger.error("%s: expected %r, got %r", s.name, s.expected_exit_status, got)
            failed += 1
    return 1 if failed else 0


async def cmd_execute(
    names: list[str] | None,
    *,
    count: int,
    concurrency: int,
    instance_ids: list[str] | None,
    scan_tasks: int | None,
    http_url: str | None,
    use_server_semaphore: bool,
    dump_json: bool,
) -> int:
    if count > 1 or concurrency > 1 or scan_tasks or (instance_ids and len(instance_ids) > 1):
        return await cmd_concurrent_execute(
            names,
            count=count,
            concurrency=concurrency,
            instance_ids=instance_ids,
            scan_tasks=scan_tasks,
            http_url=http_url,
            use_server_semaphore=use_server_semaphore,
            dump_json=dump_json,
        )

    os.environ.setdefault("HARBOR_TASKS_DIR", DEFAULT_TASKS_DIR)
    scenarios = _select_scenarios(names)
    trial = [s for s in scenarios if s.needs_full_trial or s.expected_exit_status is None]
    if not trial:
        trial = scenarios

    inst = instance_ids[0] if instance_ids else None
    for s in trial:
        logger.info("Executing trial: %s", s.name)
        out = await _run_one(s, instance_id=inst)
        _print_json(out)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="构建并测试 server._run_trial 的虚拟 RunRequest")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--list", action="store_true", help="打印 request JSON，不调用 _run_trial")
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="仅跑会快速失败的校验场景（默认）",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="执行完整 Harbor trial（需 Docker + LLM）",
    )
    parser.add_argument("--scenario", nargs="+", metavar="NAME", help="限定场景名")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="并发模式下总请求数（默认 1）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="客户端同时 in-flight 的请求上限（默认 4）",
    )
    parser.add_argument(
        "--scan-tasks",
        type=int,
        metavar="N",
        help="从 HARBOR_TASKS_DIR 取前 N 个任务目录作为 instance_id",
    )
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        metavar="ID",
        help="显式指定 instance_id 列表（不足 count 时循环复用）",
    )
    parser.add_argument(
        "--http",
        metavar="URL",
        help="POST 到运行中的 server（如 http://127.0.0.1:11000），而非直连 _run_trial",
    )
    parser.add_argument(
        "--use-server-semaphore",
        action="store_true",
        help="直连 _run_trial 时模拟 server 的 asyncio.Semaphore 限流",
    )
    parser.add_argument(
        "--dump-json",
        action="store_true",
        help="并发结束后额外打印完整 JSON 结果",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.count < 1:
        parser.error("--count 必须 >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency 必须 >= 1")

    if args.list:
        raise SystemExit(asyncio.run(cmd_list(args.scenario)))
    if args.execute:
        raise SystemExit(
            asyncio.run(
                cmd_execute(
                    args.scenario,
                    count=args.count,
                    concurrency=args.concurrency,
                    instance_ids=args.instance_ids,
                    scan_tasks=args.scan_tasks,
                    http_url=args.http,
                    use_server_semaphore=args.use_server_semaphore,
                    dump_json=args.dump_json,
                ),
            ),
        )
    # 默认 --dry-run
    raise SystemExit(asyncio.run(cmd_dry_run(args.scenario)))


if __name__ == "__main__":
    main()
