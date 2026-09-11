# Sandbox smoke: one episode on a real sandbox API

`run.py` runs one (connector, backend, agent, benchmark) combination against
the real platform; PASS iff the verifier returns reward 1.0. With the default
agent `golden` — the task's own reference solution, executed by the
connector's own mechanism — no GPU, no model and no session server are
involved, so what a run proves is exactly the platform round trip: image
resolution, sandbox create, exec, verifier, teardown.

How a provider plugs in and how to add one:
[Adding a Sandbox Provider](../../docs/developer/adding-a-sandbox-provider.md).
This file is the tool's own reference.

| flag | values | notes |
| --- | --- | --- |
| `--connector` | `harbor`, (`openenv` next) | which integration carries the episode |
| `--backend` | `e2b`, `daytona`, `modal`, ... | passed through; the connector validates |
| `--agent` | `golden` (default), or a harness name | a harness needs `--base-url`: a live session-server URL for full token fidelity, or any OpenAI-compatible endpoint when only the harness↔sandbox plumbing is under test — but never a third-party model API, which cannot sit behind the session server and so proves nothing about the training path |
| `--benchmark` | `tb2` (default) | Terminal-Bench-2, cloned on first use; `TB2_TASKS_DIR` points at an existing checkout, `--task` overrides the preset `fix-git` instance |

TB2 task directories are native Harbor tasks carrying prebuilt official
images, so a checkout is directly usable and no image is built from a
Dockerfile here. (If you point `--tasks-dir` at a Dockerfile-built task set
instead, note that E2B Cloud builds templates as a non-root user, so `RUN`
layers needing root fail there.)

```bash
# uv, not pip: the branch carries a uv-workspace dependency pip cannot resolve
uv pip install "harbor[e2b] @ git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0"
mkdir -p ~/.config/e2b && echo e2b_... > ~/.config/e2b/api_key
# the key FILE, not an exported var: this is the credential path training uses,
# so a smoke run exercises it too (E2B_API_KEY in the env would shadow it)
python scripts/sandbox_smoke/run.py --connector harbor --backend e2b
```

Endpoints and the other providers' credentials:
[Sandbox Providers](../../docs/user-guide/sandbox-providers.md). Run it from
any machine holding the provider key.

`tests/e2e/agentic/test_sandbox_golden.py` is the same episode as a registered
test, one case per provider, for CI to collect; it calls this script rather
than reimplementing it.
