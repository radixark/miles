# Default Arguments That Should Be Required — miles.utils.ft

## Summary

Audited all 162 Python files under `miles/utils/ft/`. Found 10 cases where default arguments mask required parameters or hide configuration errors. The most dangerous pattern is `TrainingRankRoster(run_id=None)` which silently rejects all training rank registrations. The most widespread pattern is `ft_id: str = ""` on internal-only functions whose callers always pass a value.

## Findings

### 1. TrainingRankRoster(run_id=None) — silent registration rejection

- **位置：** `controller/subsystem_hub/training_rank_roster.py` → `TrainingRankRoster.__init__()`
- **参数：** `run_id: str | None = None`
- **问题：** If `run_id` is None, `register_training_rank()` silently rejects **every** registration because `run_id != self.run_id` is True for any non-None `run_id` (any real string `!= None`). The controller appears healthy but has zero rank information — no rank placement, no PIDs, no per-rank scrape targets. `warn_if_incomplete()` never fires because `expected_world_size` stays None.
- **调用方：** The production caller (`controller/controller.py:119`) **always** passes `run_id=run_id`. Test fixtures in `test_training_rank_roster.py` pass it. But `test_status.py` creates 10+ `TrainingRankRoster()` instances without `run_id` for status rendering tests — these are safe because they don't test registration, but the pattern is fragile.
- **建议：** Make `run_id` required. Tests that don't need registration can pass `run_id="unused"`.

### 2. TickLoop(on_new_run=None) — critical callback always passed

- **位置：** `controller/tick_loop.py` → `TickLoop.__init__()`
- **参数：** `on_new_run: Callable[[str], None] | None = None`
- **问题：** `on_new_run` triggers `_activate_run()` which switches TrainingRankRoster, updates MiniWandb's active run, and resets state. Without it, after every restart the controller continues using the old run's rank roster and metrics — training ranks from the new run are silently rejected (see Finding #1's mechanism), MiniWandb discards new log_step calls, and detectors operate on stale data. The function `_wrap_on_new_run` returns None if input is None, and `stop_and_submit` checks `if on_new_run is not None` so there's no crash — just completely silent data corruption.
- **调用方：** The only production caller (`controller/factory.py:161`) **always** passes `on_new_run=instance._activate_run`. Test code (`test_tick_loop.py`) may omit it.
- **建议：** Make required. Tests can pass a no-op `lambda run_id: None`.

### 3. TickLoop(rank_pids_provider=None) — stack trace diagnostics silently disabled

- **位置：** `controller/tick_loop.py` → `TickLoop.__init__()`
- **参数：** `rank_pids_provider: Callable[[str], dict[int, int]] | None = None`
- **问题：** This provides per-node rank→PID mapping needed by `StackTraceClusterExecutor` to run py-spy. If None, the diagnostic pipeline receives no PIDs and `StackTraceNodeExecutor.run()` returns `_fail("no PIDs provided")` for every node. Stack trace diagnostics — critical for hang root-cause analysis — silently produce failure results that look like py-spy failures rather than configuration errors.
- **调用方：** The production caller (`controller/factory.py:162`) **always** passes a lambda that reads from `training_rank_roster_box`. Test callers may omit.
- **建议：** Make required. Tests can pass `lambda node_id: {}`.

### 4. _build_platform_components(ft_id="", k8s_label_prefix="") — internal function, always receives values

- **位置：** `factories/controller.py` → `_build_platform_components()`
- **参数：** `ft_id: str = ""`, `k8s_label_prefix: str = ""`
- **问题：** This is a private (`_` prefix) helper. Its only production caller (`build_ft_controller`, line 93) **always** passes both `ft_id` and `k8s_label_prefix`. An empty `ft_id` propagates to `RayMainJob` which sets `MILES_FT_ID=""` in the job's env vars — downstream code like `get_ft_id()` returns `""`, and actor name resolution produces the singleton name `"ft_controller"` instead of a scoped one, silently breaking multi-instance isolation.
- **调用方：** `build_ft_controller` (always passes both); 3 test callsites pass platform only (relying on defaults).
- **建议：** Make `ft_id` required. Test callsites that test invalid-platform error paths can pass `ft_id="test"`.

### 5. RayMainJob(ft_id="", k8s_label_prefix="") — always constructed with values

- **位置：** `adapters/impl/ray/main_job.py` → `RayMainJob.__init__()`
- **参数：** `ft_id: str = ""`, `k8s_label_prefix: str = ""`
- **问题：** The only constructor callsite (`_build_platform_components`, line 211) always passes both. `ft_id` is injected into the job's runtime_env as `MILES_FT_ID`; `k8s_label_prefix` as `MILES_FT_K8S_LABEL_PREFIX`. Empty strings silently propagate to the training process where they cause singleton-mode behavior instead of scoped behavior. Since `RayMainJob` is an internal class (not public API), the defaults serve no purpose.
- **调用方：** `_build_platform_components` (always passes both). No other callers.
- **建议：** Remove defaults for both `ft_id` and `k8s_label_prefix` — make them required keyword arguments.

### 6. _FtNodeAgentActorCls(node_id="", ft_id="") — always constructed with values

- **位置：** `adapters/impl/ray/node_agent_actor.py` → `_FtNodeAgentActorCls.__init__()`
- **参数：** `node_id: str = ""`, `ft_id: str = ""`
- **问题：** The only caller (`ensure_node_agent`, via `actor_kwargs`) always passes both `node_id` (from `ray.get_runtime_context().get_node_id()`) and `ft_id` (from resolved ft_id). An empty `node_id` would create an agent with identity `""`, which silently breaks node-based routing: the controller's `NodeAgentRegistry` would store it under key `""`, diagnostic pipeline would skip it (or target it incorrectly), and `mark_node_bad("")` would attempt to label a non-existent K8s node.
- **调用方：** `ensure_node_agent` in `factories/embedded_agent.py:104` (always passes both).
- **建议：** Remove defaults for both. The constructor already uses `*` (keyword-only), so callers must name them.

### 7. K8sNodeManager(namespace="") — empty namespace causes API failures

- **位置：** `adapters/impl/k8s_node_manager.py` → `K8sNodeManager.__init__()`
- **参数：** `namespace: str = ""`
- **问题：** An empty namespace string is passed to `list_namespaced_pod(namespace="")` and `patch_node(name=...)` — the K8s API accepts it but returns results from the default namespace or fails unpredictably depending on RBAC. The constructor stores it without validation. The only production callsite (`_build_platform_components`, line 208) validates `namespace` before constructing, but the class itself accepts the invalid value.
- **调用方：** `_build_platform_components` always reads from env and validates. Tests may construct directly.
- **建议：** Add validation in `__init__`: `if not namespace: raise ValueError("namespace must be non-empty")`. Or make it required with no default.

### 8. DiagnosticOrchestrator(node_agent_registry=None, pipeline=None) — both required in practice

- **位置：** `controller/diagnostics/orchestrator.py` → `DiagnosticOrchestrator.__init__()`
- **参数：** `node_agent_registry: NodeAgentRegistry | None = None`, `pipeline: list[ClusterExecutorProtocol] | None = None`
- **问题：** With both defaults, you get an empty registry and empty pipeline. `run_diagnostic_pipeline()` returns "no diagnostics configured (empty pipeline)" with `bad_node_ids=[]`. The controller treats this as "all healthy" — a false negative. In production (`controller/factory.py:80`), both `node_agent_registry` and `pipeline` are **always** passed. In tests, either `node_agents` or `node_agent_registry` plus `pipeline` are always passed. The all-None path exists only to make the constructor lenient, but it masks a wiring error.
- **调用方：** Production: always passes both. Tests: always pass node_agents + pipeline (or just pipeline).
- **建议：** Make `pipeline` required (no default). Keep `node_agent_registry` and `node_agents` optional (either-or pattern is intentional).

### 9. RayControllerClient(ft_id="") — always passed by factory callers

- **位置：** `adapters/impl/ray/controller_client.py` → `RayControllerClient.__init__()`
- **参数：** `ft_id: str = ""`
- **问题：** The factory callers (`build_tracking_agent`, `build_training_rank_agent` in `factories/embedded_agent.py`) always pass `ft_id=ft_id or get_ft_id()`. Test code always passes `ft_id="test-ft"` or similar. The empty-string default is ambiguous: it means both "the singleton controller" (valid) and "caller forgot to pass ft_id" (bug). If a caller constructs `RayControllerClient()` without ft_id in a multi-instance deployment, it silently connects to `"ft_controller"` instead of `"ft_controller_{ft_id}"` — wrong controller instance, silent data corruption.
- **调用方：** `factories/embedded_agent.py` (always passes ft_id or get_ft_id()); all test callers pass explicit values.
- **建议：** Make `ft_id` a required parameter. The factories already resolve it before constructing.

### 10. TrainingRankRoster(scrape_target_manager=None) vs. FtController._activate_run always passes it

- **位置：** `controller/subsystem_hub/training_rank_roster.py` → `TrainingRankRoster.__init__()`
- **参数：** `scrape_target_manager: ScrapeTargetManagerProtocol | None = None`
- **问题：** If None, it falls back to `NullScrapeTargetManager()` which silently discards all `add_scrape_target` / `remove_scrape_target` calls. This means per-rank Prometheus metrics (iteration counter, phase gauge) are never scraped. The hang detector, which relies on heartbeat changes from these rank-level metrics, would see no data and report "no heartbeat data available" — a false negative that masks a real hang. The production caller (`controller.py:121`) always passes `scrape_target_manager=self._scrape_target_manager`.
- **建议：** Make `scrape_target_manager` required. Tests that don't need scraping can pass `NullScrapeTargetManager()` explicitly — this documents the intent.

## Not flagged (intentional patterns)

The following were examined and found to be intentional:

- **Detector `config: XxxConfig | None = None`** (HangDetector, MfuDeclineDetector, etc.) — standard "use defaults" pattern, well-handled by `config or XxxConfig()`.
- **`notifier: NotifierProtocol | None = None`** — genuinely optional, None means "no notifications".
- **`controller_exporter: ControllerExporter | None = None`** — falls back to NullControllerExporter, documented null-object pattern.
- **Factory `*_override` params** (build_ft_controller) — standard DI override pattern for tests.
- **`build_tracking_agent(ft_id="")`, `build_training_rank_agent(ft_id="")`, `ensure_node_agent(ft_id="")`** — public API factories that auto-detect from env vars via `ft_id or get_ft_id()`.
- **`FtTrackingAgent(controller_client=None)`**, **`FtTrainingRankAgent(controller_client=None)`** — intentional graceful degradation; None means "operate silently".
- **`metadata_provider: AgentMetadataProvider | None = None`** — genuinely optional, None means "no metadata".
- **`node_metadata: dict[str, str] | None = None`** on protocol methods — genuinely optional auxiliary data.
- **`label_filters: ... | None = None`** on metric query methods — None means "no filtering".
- **Pydantic model fields with `= []` / `= {}`** (DiagnosticPipelineResult, RestartState, etc.) — Pydantic v2 creates new instances per model, safe.
- **`graceful_degrade(default=None, msg=None)`** — decorator parameters, intentionally optional.
- **`retry_async(sleep_fn=None, per_call_timeout=None)`** — None means "use asyncio.sleep" / "no per-call timeout".
- **CLI parameters** (`cli/launch.py`) — Typer CLI options, defaults are inherent to CLI design.
