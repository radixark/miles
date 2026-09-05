# Rollout cell health under saturation

A periodic generation probe competes with rollout requests for a scheduler slot.
A full engine can reject or time out this probe even while serving useful work.
For Trainium engines, select liveness for periodic Miles cell checks:

```bash
# Miles trainer/controller launch arguments (this branch must be installed):
--rollout-health-check-endpoint health

# Environment of every SGLang engine, set before startup:
SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
```

The ten Trainium 16k rollout engines already use that environment setting.
Applying the Miles client change requires restarting/redeploying the trainer's
controller with the flag; it does not require restarting these engines.
The flag only matters when rollout fault-tolerance health checking is enabled.

Periodic checks now use `/health`, without reserving a rollout slot or reducing
rollout concurrency. `/health_generate` remains a real generation probe; explicit
calls can still wait or fail under saturation. Startup `probe_server_healthy` and
`wait_server_healthy` retain generation checks. Do not redirect all generation
probes at the proxy: that would silently weaken startup readiness.

The default remains `health_generate` for existing Miles deployments. The H200
fork may differ from this public Miles base (`d2fc97ce581577e255e494801d7568747d5a10d7`):
port the client `health` method, endpoint argument, and periodic cell-checker
selection together. Set the new flag when launching H200 Miles. No trainer fork
has been modified remotely by this change.

## Failure semantics

SGLang's generation-free `/health` still reports 503 while starting or gracefully
exiting. HTTP errors, timeout, and connection failures still fail the Miles probe.
However, liveness alone does not prove an otherwise responsive engine's accelerator
is making progress. Retain generation request timeouts and explicit readiness
checks; no guarantee is made that liveness detects every device stall.

## Validation

- 69 API-client tests passed, including a simulated full 16-slot engine with 64
  liveness probes, a rejected generation probe, timeout propagation, and real 503
  propagation.
- The actual cell-checker function was exercised separately with its real health
  checker and API client: 64 liveness requests, unhealthy status propagation, and
  unchanged default generation endpoint passed.
- The full Ray controller test suite was not run in this environment because Ray
  is unavailable. A regression test is included for that suite.
- No live rollout workers were saturated or restarted for these CPU tests.

API-client command (isolated test environment):

```bash
python -m pytest --confcutdir=tests/fast/backends/sglang_utils \
  tests/fast/backends/sglang_utils/test_sglang_api_client.py -q -o asyncio_mode=auto
```
