# Snapshots

Expected results for the repository's snapshot tests. Everything here is
generated: never edit a file by hand, regenerate it and review the diff.

| Directory | Produced by | Contains |
| --- | --- | --- |
| `launch_scripts/sh/` | `tests/manual/launch_scripts/test_sh_launch_scripts.py` | every external command each `scripts/**.sh` and `examples/**.sh` launcher issues, including the full `ray job submit` argv |
| `launch_scripts/py/` | `tests/manual/launch_scripts/test_py_launch_scripts.py` | every shell command each `scripts/run_*.py` entrypoint builds |
| `launch_scripts/self_executing/` | `tests/manual/launch_scripts/test_self_executing_launchers.py` | the `ray job submit` argv of the launchers that build their own command line |
| `model_args/` | `tests/manual/launch_scripts/test_model_args.py` | the expanded argv of every `scripts/models/*.py` model definition |
| `charts/miles-run/` | `tests/fast/charts/miles_run/test_snapshot.py` | two cross-sections of one synthetic run, whose argv the real miles parser parses: the values the launcher generates from its specs, and the manifests those values render |
| `helm_backend/` | `tests/fast/utils/external_utils/command_utils/helm_backend/test_launch_snapshot.py` | the helm argv a Kubernetes launch issues, and the `run:` values file it generates |

The `launch_scripts/` and `model_args/` tests live under `tests/manual/`, which the CI runner does not discover
(see `_DISCOVERY_ROOTS` in `tests/ci/ci_register.py`), so they run only when
invoked by hand. Run them after touching a launcher or a model definition:

```bash
pytest tests/manual/launch_scripts
```

Regenerate after an intentional change:

```bash
MILES_UPDATE_LAUNCH_SCRIPT_SNAPSHOTS=1 pytest tests/manual/launch_scripts tests/fast/charts tests/fast/utils/external_utils
```

Chart changes break only `charts/`; launcher changes break only `helm_backend/`. Each side is
recorded once, so a failure names which half moved.

The recordings are reproducible on any machine: the launchers run under a
shimmed PATH with a frozen environment, and absolute paths are replaced by
`<REPO_ROOT>` / `<SANDBOX>` placeholders. The seed sglang draws afresh on every
render is replaced by `<RANDOM_SEED>`, so a re-recording only shows real moves.
