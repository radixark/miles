# Snapshots

Expected results for the repository's snapshot tests. Everything here is
generated: never edit a file by hand, regenerate it and review the diff.

| Directory | Produced by | Contains |
| --- | --- | --- |
| `launch_scripts/sh/` | `tests/manual/launch_scripts/test_sh_launch_scripts.py` | every external command each `scripts/**.sh` and `examples/**.sh` launcher issues, including the full `ray job submit` argv |
| `launch_scripts/py/` | `tests/manual/launch_scripts/test_py_launch_scripts.py` | every shell command each `scripts/run_*.py` entrypoint builds |
| `launch_scripts/self_executing/` | `tests/manual/launch_scripts/test_self_executing_launchers.py` | the `ray job submit` argv of the launchers that build their own command line |
| `model_args/` | `tests/manual/launch_scripts/test_model_args.py` | the expanded argv of every `scripts/models/*.py` model definition |

These tests live under `tests/manual/`, which the CI runner does not discover
(see `_DISCOVERY_ROOTS` in `tests/ci/ci_register.py`), so they run only when
invoked by hand. Run them after touching a launcher or a model definition:

```bash
pytest tests/manual/launch_scripts
```

Regenerate after an intentional change:

```bash
MILES_UPDATE_LAUNCH_SCRIPT_SNAPSHOTS=1 pytest tests/manual/launch_scripts
```

The recordings are reproducible on any machine: the launchers run under a
shimmed PATH with a frozen environment, and absolute paths are replaced by
`<REPO_ROOT>` / `<SANDBOX>` placeholders.
