import importlib
import inspect
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
def load_function(path, *, sync_required=False):
    """
    Load a function from registry or module.
    :param path: The path to the function, e.g. "module.submodule.function".
    :param sync_required: Reject coroutine functions, for callers that run the
        loaded function synchronously on an event loop.
    :return: The function object.
    """
    if not path:
        return None

    fn = function_registry.get(path)
    if fn is None:
        module_path, _, attr = path.rpartition(".")
        module = importlib.import_module(module_path)
        fn = getattr(module, attr)
    if sync_required:
        if not callable(fn):
            raise ValueError(f"load_function({path!r}) did not resolve to a callable")
        if inspect.iscoroutinefunction(fn):
            raise ValueError(f"load_function({path!r}) resolved to an async function; a synchronous one is required")
    return fn
