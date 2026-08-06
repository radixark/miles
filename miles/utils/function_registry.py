import importlib
from contextlib import contextmanager


# Mainly used for test purpose where `load_function` needs to load many in-flight generated functions
class FunctionRegistry:
    def __init__(self):
        self._registry: dict[str, object] = {}

    @contextmanager
    def temporary(self, name: str, fn: object):
        self._register(name, fn)
        try:
            yield
        finally:
            self._unregister(name)

    def get(self, name: str) -> object | None:
        return self._registry.get(name)

    def _register(self, name: str, fn: object) -> None:
        assert name not in self._registry
        self._registry[name] = fn

    def _unregister(self, name: str) -> None:
        assert name in self._registry
        self._registry.pop(name)


function_registry = FunctionRegistry()


# TODO may rename to `load_object` since it can be used to load things like tool_specs
def load_function(path):
    """
    Load a function from registry or module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :return: The function object.
    """
    if path is None:
        return None

    registered = function_registry.get(path)
    if registered is not None:
        return registered

    module_path, _, attr = path.rpartition(".")
    module = importlib.import_module(module_path)
    return getattr(module, attr)
