# Snapshots

Expected results for the repository's snapshot tests. Everything here is
generated: never edit a file by hand, regenerate it and review the diff.

| Directory | Produced by | Contains |
| --- | --- | --- |
| `launch_scripts/sh/` | `tests/fast/launch_scripts/test_sh_launch_scripts.py` | every external command each `scripts/**.sh` and `examples/**.sh` launcher issues, including the full `ray job submit` argv |
| `launch_scripts/py/` | `tests/fast/launch_scripts/test_py_launch_scripts.py` | every shell command each `scripts/run_*.py` entrypoint builds |
| `helm_backend/` | `tests/fast/utils/external_utils/command_utils/helm_backend/test_launch_snapshot.py` | the helm argv a Kubernetes launch issues, and the `run:` values file it generates |

Regenerate after an intentional change:

```bash
MILES_UPDATE_LAUNCH_SCRIPT_SNAPSHOTS=1 pytest tests/fast/launch_scripts tests/fast/utils/external_utils
```

The recordings are reproducible on any machine: the launchers run under a
shimmed PATH with a frozen environment, and absolute paths are replaced by
`<REPO_ROOT>` / `<SANDBOX>` placeholders.
