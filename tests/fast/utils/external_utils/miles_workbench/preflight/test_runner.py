from __future__ import annotations

import logging
import subprocess

import pytest

from miles.utils.external_utils.miles_workbench.preflight import checkers as checkers_module
from miles.utils.external_utils.miles_workbench.preflight.runner import (
    _namespace_listing_checkers,
    _warn_about_foreign_objects,
)


class TestNamespaceForeignObjectWarnings:
    def test_a_foreign_secret_omitted_by_get_all_is_still_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A secret omitted from kubectl get all still produces a foreign-object warning."""

        def run_raw(*args: str) -> subprocess.CompletedProcess[str]:
            output = "secret/foreign-secret\n" if args[1] == "secret" else ""
            return subprocess.CompletedProcess(args=list(args), returncode=0, stdout=output, stderr="")

        monkeypatch.setattr(checkers_module.Kubectl, "run_raw", staticmethod(run_raw))
        listings = _namespace_listing_checkers("rl", release="my-release")
        for listing in listings:
            listing.check()

        with caplog.at_level(logging.WARNING, logger="miles.utils.external_utils.miles_workbench.preflight.utils"):
            _warn_about_foreign_objects(namespace="rl", listings=listings)

        assert "namespace rl also holds secret/foreign-secret" in caplog.text
