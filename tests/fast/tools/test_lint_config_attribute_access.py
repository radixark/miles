from pathlib import Path

import pytest

from tools.lint_config_attribute_access import _violations, main


class TestScanRoots:
    @pytest.mark.parametrize("root", ["miles", "miles_plugins", "examples"])
    @pytest.mark.parametrize("access", ['getattr(args, "missing", False)', 'hasattr(args, "missing")'])
    def test_unexempted_access_is_rejected_in_every_source_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, root: str, access: str
    ) -> None:
        """All production roots reject dynamic access without a reason."""
        path = tmp_path / root / "nested" / "consumer.py"
        path.parent.mkdir(parents=True)
        path.write_text(f"value = {access}\n")
        monkeypatch.chdir(tmp_path)

        with pytest.raises(SystemExit, match=rf"{root}/nested/consumer.py:1: dynamic attribute access"):
            main()


class TestExemptions:
    @pytest.mark.parametrize(
        "source",
        [
            "value = getattr(result, field)  # config-access-exempt: external schema selects fields by name\n",
            "value = getattr(\n    result, field\n)  # config-access-exempt: external schema selects fields by name\n",
        ],
    )
    def test_a_specific_inline_reason_allows_single_and_multiline_calls(self, tmp_path: Path, source: str) -> None:
        """A reason on the same logical statement permits intentional reflection."""
        path = tmp_path / "consumer.py"
        path.write_text(source)

        assert _violations(path) == []

    @pytest.mark.parametrize(
        "source",
        [
            'value = getattr(args, "missing")  # config-access-exempt:\n',
            'value = getattr(args, "missing")  # config-access-exempt: runtime reflection is required\n',
            '# config-access-exempt: external schema selects fields by name\nvalue = getattr(args, "missing")\n',
            'reason = "config-access-exempt: external schema"; value = getattr(args, "missing")\n',
            'value = getattr(result, field)  # config-access-exempt: external schema selects fields by name\nother = hasattr(args, "missing")\n',
        ],
    )
    def test_empty_generic_or_unrelated_reasons_do_not_exempt_access(self, tmp_path: Path, source: str) -> None:
        """Exemptions require a specific comment on the offending statement."""
        path = tmp_path / "consumer.py"
        path.write_text(source)

        [violation] = _violations(path)
        assert "dynamic attribute access needs a specific inline exemption" in violation
