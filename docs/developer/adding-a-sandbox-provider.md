---
title: Adding a Sandbox Provider
description: How a cloud sandbox provider plugs into Miles, and what it takes to add one.
---

Connectors that run a task in a container (Harbor, OpenEnv) get that container
from a sandbox provider. A provider is **one entry in `PROVIDER_CREDENTIALS`**
(`miles/rollout/agentic/credentials.py`) — no connector code branches on it.
The in-process Harbor path passes `HARBOR_ENV_TYPE` straight to Harbor; the
golden test runs one case per registry entry, and the rollout e2e takes its
provider from `HARBOR_ENV_TYPE`. Setting a provider up as a user is
[Sandbox Providers](/user-guide/sandbox-providers).

## Adding one

1. **The registry entry.** What credential a worker needs, the path-valued
   variable the launcher forwards in place of the secret, the address variables
   that are safe to forward by value, and the SDK. Field semantics are
   documented on `PROVIDER_CREDENTIALS` itself.
2. **A golden episode** on the real platform:
   `python scripts/sandbox_smoke/run.py --connector harbor --backend <name>`.
   This is the only step that proves the platform round trip — image
   resolution, sandbox create, exec, verifier, teardown — and nothing below it
   is claimable without one.
3. **A GPU rollout**: `HARBOR_ENV_TYPE=<name>` with
   `tests/e2e/agentic/test_harbor_rollout.py`. This is the bar for the
   [provider table](/user-guide/sandbox-providers).
4. **A cell in that table**, plus a setup section on the same page if the
   provider needs anything beyond a credential.

Nothing in step 1 needs a test change: the golden episode runs as one case per
registry entry, and the rollout e2e reads the provider from the environment.

## The two tests

| | `tests/e2e/agentic/test_sandbox_golden.py` | `tests/e2e/agentic/test_harbor_rollout.py` |
| --- | --- | --- |
| Proves | the platform round trip | harness, session server, TITO recording, reward |
| Needs | a provider credential | that, plus 2 GPUs and the model |
| Provider axis | one case per registry entry | `HARBOR_ENV_TYPE` |
| In CI | registered, skipped without a credential | registered, `disabled` |

The golden test drives `scripts/sandbox_smoke/run.py`, which stays the entry
point for the combinations CI does not fix — another agent harness, another
task, another connector. Its README covers those axes.
