---
name: debug-session-server
description: Debug Miles session-server and agentic fully-async rollout issues — active sessions/work accumulation, low SGLang inflight despite high rollout-batch-size, AgentError from Harbor, session debug dumps, TITO/tokenization errors. Use when running swe/run-qwen35-4B-agentic-async.py (or similar agentic async scripts with --use-session-server), session-server metrics look wrong, or agentic training stalls with exit_status AgentError.
user_invocable: true
---

# Debug Session Server（Agentic Fully-Async）

针对 **Qwen3.5 agentic fully-async**（如 `swe/run-qwen35-4B-agentic-async.py`）中 session-server、Harbor agent、SGLang 打不满、active work 堆积、AgentError 的系统性排查流程。

## 架构（请求链路）

```text
AsyncRolloutWorker (rollout_batch_size 个并发 trial)
  → agentic_tool_call.generate
    → OpenAIEndpointTracer  POST /sessions  (session-server :30000)
      → swe_agent_function.run  POST agent-server /run (:11000)
        → Harbor Trial (Docker/bash，大部分时间在等环境)
          → LLM 调用  base_url/sessions/{id}/v1/...  (回到 session-server)
            → session-server proxy → miles/sglang router → SGLang engines
```

**三个 inflight 概念不要混：**

| 指标 | 含义 | 典型位置 |
|------|------|----------|
| `active_tasks` / rollout 并发 | `AsyncRolloutWorker` 同时在跑的 agent trial 数 | 上限 = `--rollout-batch-size` |
| `inflight_chat` | session-server 正在 proxy 的 chat completion 数 | 日志 `[session-server] ... inflight_chat=` |
| SGLang running reqs | GPU 上实际 decode/prefill 请求 | SGLang metrics / `nvidia-smi` |

Agent trial 很长且大部分时间在 **bash/Docker**，所以 `rollout-batch-size=32` 时 `inflight_chat≈10`、SGLang GPU 打不满 **可能是正常现象**，不一定是 session-server bug。

## 快速诊断清单

复制并逐项检查：

```text
Progress:
- [ ] 1. 确认 session-server 存活与 active_sessions
- [ ] 2. 对比 rollout 并发 vs session inflight vs SGLang 利用率
- [ ] 3. 查 AgentError 来源（Harbor vs session vs SGLang）
- [ ] 4. 检查 session debug dump / wandb session/* 指标
- [ ] 5. 验证 K8s 网络（agent-server → session-server）
- [ ] 6. 判断是泄漏、并发瓶颈还是 agent 环境慢
```

---

## Step 1: Session-server 健康检查

Session-server 随 RolloutManager 在 **训练 worker-0** 启动（`--session-server-port 30000`）。

在 **训练 Pod** 内执行：

```bash
# 健康与 instance id（agent 侧会校验）
curl -s http://127.0.0.1:30000/health | jq .

# 内存中 session 快照（排查 split-brain / 泄漏）
curl -s http://127.0.0.1:30000/debug/sessions | jq .
```

关注字段：
- `active_count` / `active_sessions` — 持续增长且不回落 → **session 泄漏或 DELETE 失败**
- `session_server_instance_id` — agent 请求里的 instance_id 必须匹配，否则 404
- `closing_count` — 长时间 >0 说明 DELETE 卡在等 lock

**日志关键字**（RolloutManager / session-server stdout）：

```text
[session-server] CREATE session_id=
[session-server] CHAT start ... active_count=
[session-server] DELETE done ... active_count=
[session-server] REQUEST ARRIVED: ... inflight_chat=
[session-debug] dumped ... debug event to
```

---

## Step 2: 解读「active work 1500+ / inflight ~10 / SGLang 打不满」

### 2a. active work / active_sessions 持续涨到 1000+

**已知行为：** `OpenAIEndpointTracer.collect_records()` 在 **成功路径不 DELETE session**（仅 timeout/collect_failed 会 `_try_delete_session`）。长跑后 `active_count` 单调上升是预期现象，会导致：

- session-server 内存涨（每个 session 存完整 trajectory + records）
- 最终 OOM / 响应变慢 / 间接 AgentError

**验证：**

```bash
# 每隔几分钟采样
watch -n 30 'curl -s http://127.0.0.1:30000/health | jq .active_sessions'
```

若单调上升 → 按「Step 6 修复方向」处理（短期：重启 rollout；长期：补 DELETE 或定期清理）。

### 2b. rollout-batch-size=32 但 inflight_chat≈10

正常原因（按优先级）：

1. **Agent 环境占时** — Harbor trial 在跑 bash/测试，不在调 LLM
2. **Agent server 并发** — `AGENT_MAX_CONCURRENT`（yaml 里 64；`swe/server.py` 默认 8）
3. **Session-server 单进程** — 所有 trial 共用一个 uvicorn
4. **SGLang KV slot 上限** — Qwen3.5-4B dense 1 GPU/engine，32k ctx 时 per-engine slot ≈ `12 / (32768/8192) = 3`

脚本里计算的 SGLang 参数（`run-qwen35-4B-agentic-async.py`）：

```python
ctx_scale = sglang_context_length // sglang_context_ref_len  # 32768/8192 = 4
sglang_per_engine_slots = sglang_short_ctx_slots_per_engine // ctx_scale  # 12//4 = 3
cluster_max_connections = sglang_per_engine_slots * rollout_gpus  # 例：3 * 16 = 48
```

注意：`--sglang-server-concurrency` / `--sglang-max-running-requests` 在脚本里 **被注释掉**，client semaphore 用默认 512，但 **engine KV 才是硬上限**。

### 2c. SGLang GPU 利用率低

先区分 **agent-bound** vs **inference-bound**：

| 信号 | agent-bound | inference-bound |
|------|-------------|-----------------|
| inflight_chat 低 | ✓ | |
| session active 高、chat 少 | ✓（trial 多但在跑 bash） | |
| inflight_chat 接近 rollout_batch | | ✓ |
| SGLang queue 满 / 429 | | ✓ |

---

## Step 3: AgentError 排查

`exit_status=AgentError` 来自 Harbor（`swe/server.py` `_extract_exit_status`），表示 trial 抛异常（非 timeout）。

**分层查：**

```bash
# 1) Agent server 日志（Harbor trial 栈）
# 在 agent-server Pod/机器上看 swe/server.py 输出

# 2) Miles rollout 日志
# [session=...] Agent function failed
# Returned aborted group ... to data buffer

# 3) Session debug dumps
ls -lt /home/yangchengyi/data/debug_msgs/<wandb_run_name>/ | head
```

**Wandb / Prometheus session 指标**（每 rollout step）：

```text
session/session_error_count/samples_affected
session/session_error_count/events/<reason>
session/session_error_count/samples/error_type/<type>
```

常见 `reason`：`collect_timeout`、`SessionNotFoundError`、`TokenizationError`、`UpstreamResponseError`、`backend_status_502`、`closed_during_proxy`。

---

## Step 4: 分析 session debug dump

Dump 目录：`--session-debug-dump-dir`（默认 `/home/yangchengyi/data/debug_msgs/<wandb_run_name>/`）。

每个 JSON 含：`reason`、`registry.active_count`、`trajectory.debug_interactions`、异常栈。

**离线复现 SGLang 行为：**

```bash
cd /fs/nlp-intern/yangchengyi/miles
python scripts/tools/replay_debug_messages_to_sglang.py \
  --debug-json /home/yangchengyi/data/debug_msgs/<run>/<file>.json \
  --model-path /home/yangchengyi/data/models/Qwen3.5-4B
```

**检查 Qwen3.5 tool parser：**

```bash
python scripts/tools/check_qwen35_tool_parser.py
```

脚本使用 `--sglang-tool-call-parser qwen3_coder`；parser 不匹配会导致 tool call 解析失败 → upstream 400/异常。

---

## Step 5: K8s 网络（方案 A）

Harbor agent **不在训练 Pod 内**，必须通过可达地址访问 session-server：

| 变量 | 作用 |
|------|------|
| `MILES_ROUTER_EXTERNAL_HOST` | 把 session URL 改写成 agent 能访问的 host:port |
| `AGENT_SERVER_URL` | Miles → Harbor（如 `http://10.255.116.6:11000`） |
| `AGENT_MAX_CONCURRENT` | Harbor 侧 trial 并发 |

**Split-brain 症状：** agent 打到旧 session-server / 错误 Pod → `SessionNotFoundError`、404、`session_server_instance_id` 不匹配。

验证 agent 侧能否访问 session：

```bash
# 在 agent-server 所在网络
curl -s http://<MILES_ROUTER_EXTERNAL_HOST>:30000/health | jq .
```

训练 Pod 需带 label `app: miles-session`，Service 指向 worker-0（见 `swe/qwen35_4B_single_node.yaml` 注释）。

---

## Step 6: 修复与调参方向

按症状选动作，**不要同时改太多 knob**：

### Session 泄漏 / active_count 1000+

- **短期**：重启 rollout job（`pkill` / 重新 submit）
- **长期**：在 `collect_records` 成功路径补 `_try_delete_session()`；或 session-server 加 TTL GC

### 想提高 SGLang 利用率

1. 提高 **真正在调 LLM 的 trial 数**：`--rollout-batch-size`（受 agent/docker 资源限制）
2. 提高 **agent 并发**：`AGENT_MAX_CONCURRENT`（注意 Docker 磁盘/CPU）
3. 降低 **ctx 换 slot**：减小 `--sglang-context-length` 或增大 `--sglang-short-ctx-slots-per-engine`（受 GPU 内存限制）
4. 取消注释并显式传入 `--sglang-server-concurrency`（与 per-engine slot 对齐，见脚本 print 的 `cluster_connections`）

### AgentError 批量出现

1. 先看最新 debug dump 的 `reason` 和 Harbor 栈
2. 查 `session_error_count/events/*` 哪个 reason 突增
3. 若是 `collect_timeout`（300s）：agent  hung 或 session-server 过载
4. 若是 `TokenizationError` / `tito_session_mismatch`：查 `--tito-model qwen35` 与 chat template

### Fully-async 特有

- `AsyncRolloutWorker.output_queue` maxsize=1000 — 完成 trial 积压时训练侧 drain 慢
- `max_weight_staleness 2` — stale group 回 buffer，会 **inflate** 有效 active work
- worker loop `await asyncio.sleep(1)` — 补 task 有最多 1s 延迟（通常不是主因）

---

## 参考文件

| 文件 | 内容 |
|------|------|
| `swe/run-qwen35-4B-agentic-async.py` | 启动参数、SGLang slot 计算、session-server 配置 |
| `miles/rollout/session/session_server.py` | Session server 入口 |
| `miles/rollout/session/sessions.py` | 路由、inflight_chat、debug dump |
| `miles/rollout/generate_hub/agentic_tool_call.py` | Agent generate 主流程 |
| `miles/rollout/generate_utils/openai_endpoint_utils.py` | Session 创建/collect/delete |
| `swe/swe_agent_function.py` | Miles → Harbor `/run` |
| `swe/server.py` | Harbor agent server、AgentError |
| `examples/fully_async/fully_async_rollout.py` | AsyncRolloutWorker 并发模型 |
| `scripts/tools/replay_debug_messages_to_sglang.py` | 离线复现 |
| `scripts/tools/check_qwen35_tool_parser.py` | Tool parser 检查 |

## 报告模板

排查完成后给用户/自己留档：

```markdown
## Session-server debug summary

**Run:** <wandb_run_name>
**Symptom:** active_sessions=?, inflight_chat=?, rollout_batch_size=?, AgentError rate=?

### Findings
- Bottleneck: [agent-bound | session-leak | sglang-kv | network | tito]
- Evidence: <log lines / metric / dump file>

### Actions taken
- ...

### Recommended next
- ...
```
