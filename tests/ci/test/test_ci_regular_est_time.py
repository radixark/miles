"""A regular CUDA test fits in 2400 s.

A `register_cuda_ci` test with neither `nightly=True` nor a `long` or `ft-long`
label keeps `est_time` at or below 2400 s; a longer one has its workload cut
until it fits. A disabled registration is checked when it is re-enabled.
"""

from pathlib import Path

from tests.ci.ci_register import HWBackend, collect_tests, discover_ci_files, register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

REPO_ROOT = Path(__file__).resolve().parents[3]

_REGULAR_EST_TIME_CAP_S = 2400
_LONG_LABELS = {"long", "ft-long"}


def test_regular_cuda_tests_fit_the_cap(monkeypatch):
    # discover_ci_files() globs repo-relative paths; pin cwd to the checkout.
    monkeypatch.chdir(REPO_ROOT)
    over = sorted(
        (r.filename, r.est_time)
        for r in collect_tests(discover_ci_files())
        if r.backend is HWBackend.CUDA
        and r.disabled is None
        and not r.nightly
        and not _LONG_LABELS & set(r.labels)
        and r.est_time > _REGULAR_EST_TIME_CAP_S
    )
    assert over == [], (
        f"these regular CUDA tests exceed est_time={_REGULAR_EST_TIME_CAP_S}; cut their workload until "
        f"they fit: {over}"
    )
