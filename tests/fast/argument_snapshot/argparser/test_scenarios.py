import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

import argparse

from pydantic import JsonValue
from tests.fast.argument_snapshot.argparser import scenarios

from miles.utils.test_utils.snapshot_yaml import assert_snapshot_equal


class TestParsedScenarioValues:
    @pytest.mark.parametrize(
        ("base", "changed"),
        [(False, 0), (1, 1.0), ({"items": [False]}, {"items": [0]})],
    )
    def test_variants_preserve_complete_values_and_numeric_types(
        self, monkeypatch: pytest.MonkeyPatch, base: JsonValue, changed: JsonValue
    ) -> None:
        """Parsed variants retain type changes that ordinary Python equality would discard."""
        parser = argparse.ArgumentParser()
        for name in (
            "rollout-batch-size",
            "train-backend",
            "num-rollout",
            "actor-num-gpus-per-node",
            "micro-batch-size",
        ):
            parser.add_argument(f"--{name}")
        for flag in (
            "--no-sglang-lora-use-virtual-experts",
            "--sglang-tp-size",
            "--eval-sglang-enable-metrics",
            "--no-eval-sglang-enable-metrics",
        ):
            parser.add_argument(flag, nargs="?", const=changed, type=lambda value: changed, dest="value")
        parser.set_defaults(value=base, unchanged=None)
        monkeypatch.setattr(scenarios, "parse_args_and_get_parser", lambda: (None, parser))

        captured = scenarios._capture_scenario(scenarios._Scenario(backend="fsdp"))
        result = captured["parsed"]

        defaults = captured["schema"]["effective_defaults"]
        assert_snapshot_equal(expected=base, actual=defaults["value"])
        assert "unchanged" in defaults and defaults["unchanged"] is None
        assert_snapshot_equal(expected=base, actual=result["minimal"]["value"])
        for name in ("lora_disabled", "sglang_alias", "eval_true", "eval_false"):
            assert_snapshot_equal(expected=changed, actual=result[name]["value"])
            assert "unchanged" in result[name] and result[name]["unchanged"] is None
            assert result[name]["train_backend"] == "fsdp"
