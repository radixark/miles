# Must-Read Skills Before Modifying Components

Before modifying the following, read the listed skill first.

- **A file carrying a `# doc-dev:` sentinel, or a document such a sentinel names** -- today the PR, release, bot, and image workflows under `.github/workflows/`, `docker/build.py` and the Dockerfiles, `tests/ci/metric_history/**`, and their pages under `docs/developer/ci/` -- → [`doc-dev`](../skills/doc-dev/SKILL.md). Grep for `doc-dev:` before editing anything under `.github/workflows/` or `docker/`; the code and its document change in the same PR.
- **A file split, function move, module extraction, or rename** presented as behavior-preserving → [`mechanical-refactor-verify`](../skills/mechanical-refactor-verify/SKILL.md)
- **`est_time=` literals in `register_cuda_ci` calls** under `tests/` → [`ci-e2e-time-tune`](../skills/ci-e2e-time-tune/SKILL.md); estimates come from complete CI logs, not guesses.
