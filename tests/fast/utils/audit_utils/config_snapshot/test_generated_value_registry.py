import pytest

from miles.utils.audit_utils.config_snapshot.enabled import SNAPSHOT_TESTING_ENABLED

if not SNAPSHOT_TESTING_ENABLED:
    pytest.skip("Snapshot testing is disabled by op48-277", allow_module_level=True)

import json


from miles.utils.audit_utils.config_snapshot.generated_value_registry import (
    GENERATED_VALUES_ENV_VAR,
    GeneratedValueRegistry,
)
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


class TestGeneratedValueRegistryTransport:
    def test_disabled_capture_ignores_even_malformed_inherited_provenance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ordinary training must not parse stale snapshot-only environment payloads."""
        monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR, raising=False)
        monkeypatch.setenv(GENERATED_VALUES_ENV_VAR, "not json")
        registry = GeneratedValueRegistry()
        registry.register(kind="host", value="worker")

        assert registry.read() == []
        assert registry.to_env() == {}

    def test_child_process_round_trip_preserves_names_and_continues_numbering(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A child process extends inherited provenance without reassigning existing tokens."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, "/capture")
        monkeypatch.delenv(GENERATED_VALUES_ENV_VAR, raising=False)
        parent = GeneratedValueRegistry()
        parent.register(kind="host", value="first")
        parent.register(kind="port", value="1234")
        monkeypatch.setenv(GENERATED_VALUES_ENV_VAR, parent.to_env()[GENERATED_VALUES_ENV_VAR])
        child = GeneratedValueRegistry()
        child.register(kind="host", value="second")

        assert [(entry.kind, entry.name, entry.value) for entry in child.read()] == [
            ("host", "0000", "first"),
            ("port", "0000", "1234"),
            ("host", "0001", "second"),
        ]
        assert len(parent.read()) == 2

    def test_named_conflicts_fail_without_replacing_the_original_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A generated token must never silently change its meaning within a process."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, "/capture")
        monkeypatch.delenv(GENERATED_VALUES_ENV_VAR, raising=False)
        registry = GeneratedValueRegistry()
        registry.register(kind="host", name="router", value="first")
        registry.register(kind="host", name="router", value="first")
        with pytest.raises(ValueError, match="host/router"):
            registry.register(kind="host", name="router", value="second")

        assert [entry.value for entry in registry.read()] == ["first"]
        assert json.loads(registry.to_env()[GENERATED_VALUES_ENV_VAR])[0]["value"] == "first"

    def test_reading_entries_does_not_expose_the_registry_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A consumer clearing its returned list cannot erase later snapshot provenance."""
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, "/capture")
        monkeypatch.delenv(GENERATED_VALUES_ENV_VAR, raising=False)
        registry = GeneratedValueRegistry()
        registry.register(kind="host", value="worker")
        registry.read().clear()

        assert len(registry.read()) == 1
