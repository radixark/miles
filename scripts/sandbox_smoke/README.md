# Sandbox smoke: golden episodes on real sandbox APIs

`run.py` runs one (connector, backend, agent, benchmark) combination against
the real platform; PASS iff the verifier returns reward 1.0. With the default
agent `golden` — the task's own reference solution, executed by the
connector's own mechanism — no GPU, no model and no session server are
involved, so what a run proves is exactly the platform round trip: image
build, sandbox create, exec, verifier, teardown. The offline unit and
contract tests cannot see that layer; the GPU e2e suite covers the layers
above it (harness, session server, training).

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
pip install "harbor[e2b] @ git+https://github.com/harbor-framework/harbor@harbor-miles-v0.20.0"
mkdir -p ~/.config/e2b && echo e2b_... > ~/.config/e2b/api_key
# the key FILE, not an exported var: this is the credential path training uses,
# so a smoke run exercises it too (E2B_API_KEY in the env would shadow it)
# self-hosted E2B-compatible endpoint instead of E2B Cloud:
# export E2B_API_URL=http://<server>:8000 E2B_SANDBOX_URL=http://<server>:8000
python scripts/sandbox_smoke/run.py --connector harbor --backend e2b
```

Other backends follow the same key-file contract (`DAYTONA_API_KEY` /
`~/.config/daytona/api_key`, ...; see `miles/rollout/agentic/credentials.py`).

Run it from any machine holding the provider key — deliberately not a
scheduled workflow, so no sandbox credential lives in repository secrets.

## Exercised so far

| connector | backend | agent | task | last run |
| --- | --- | --- | --- | --- |
| harbor | e2b | golden | tb2/fix-git | 2026-09-02 PASS |
| harbor | daytona | golden | tb2/fix-git | 2026-09-02 PASS |
