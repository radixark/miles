# Snapshots

Expected results for the repository's snapshot tests. Everything here is
generated: never edit a file by hand, regenerate it and review the diff.

| Directory | Produced by | Contains |
| --- | --- | --- |
| `launch_scripts/sh/` | `tests/fast/launch_scripts/test_sh_launch_scripts.py` | every external command each `scripts/**.sh` and `examples/**.sh` launcher issues, including the full `ray job submit` argv |
| `launch_scripts/py/` | `tests/fast/launch_scripts/test_py_launch_scripts.py` | every shell command each `scripts/run_*.py` entrypoint builds |
| `charts/miles-run/` | `tests/fast/charts/miles_run/test_snapshot.py` | two cross-sections of one synthetic run, whose argv the real miles parser parses: the values the launcher generates from its specs, and the manifests those values render |
| `helm_backend/` | `tests/fast/utils/external_utils/command_utils/helm_backend/test_launch_snapshot.py` | the helm argv a Kubernetes launch issues, and the `run:` values file it generates |

Regenerate after an intentional change:

```bash
MILES_UPDATE_LAUNCH_SCRIPT_SNAPSHOTS=1 pytest tests/fast/launch_scripts tests/fast/charts tests/fast/utils/external_utils
```

Chart changes break only `charts/`; launcher changes break only `helm_backend/`. Each side is
recorded once, so a failure names which half moved.

The recordings are reproducible on any machine: the launchers run under a
shimmed PATH with a frozen environment, and absolute paths are replaced by
`<REPO_ROOT>` / `<SANDBOX>` placeholders. The seed sglang draws afresh on every
render is replaced by `<RANDOM_SEED>`, so a re-recording only shows real moves.
