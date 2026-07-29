import os

import pytest
from tests.fast.import_isolation_utils import modules_imported_by

from miles.utils.function_registry import FunctionRegistry, function_registry, load_function


def _fn_a():
    return "a"


def _fn_b():
    return "b"


class TestFunctionRegistry:
    def test_register_and_get(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            assert registry.get("my_fn") is _fn_a

    def test_register_duplicate_raises(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            with pytest.raises(AssertionError):
                with registry.temporary("my_fn", _fn_b):
                    pass

    def test_unregister(self):
        registry = FunctionRegistry()
        with registry.temporary("my_fn", _fn_a):
            assert registry.get("my_fn") is _fn_a
        assert registry.get("my_fn") is None

    def test_temporary_cleanup_on_exception(self):
        registry = FunctionRegistry()
        with pytest.raises(RuntimeError):
            with registry.temporary("temp_fn", _fn_a):
                raise RuntimeError("test")
        assert registry.get("temp_fn") is None


class TestLoadFunction:
    def test_load_from_module(self):
        import os.path

        assert load_function("os.path.join") is os.path.join

    def test_load_none_returns_none(self):
        assert load_function(None) is None

    def test_load_from_registry(self):
        with function_registry.temporary("test:my_fn", _fn_a):
            assert load_function("test:my_fn") is _fn_a

    def test_registry_takes_precedence(self):
        with function_registry.temporary("os.path.join", _fn_b):
            assert load_function("os.path.join") is _fn_b
        assert load_function("os.path.join") is os.path.join

    def test_sync_required_rejects_a_non_callable_registry_entry(self):
        """A path that resolves to a non-callable must fail loudly before a synchronous caller invokes it."""
        with function_registry.temporary("test:not_callable", [1, 2, 3]):
            with pytest.raises(ValueError, match="did not resolve to a callable"):
                load_function("test:not_callable", sync_required=True)


class TestFunctionRegistryImportWeight:
    def test_importing_function_registry_does_not_import_ray(self):
        """load_function lives here rather than in miles.utils.misc so importers avoid ray."""
        modules = modules_imported_by("miles.utils.function_registry")

        assert "ray" not in modules
        assert "miles.utils.misc" not in modules
