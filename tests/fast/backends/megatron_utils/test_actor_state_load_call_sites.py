import ast

from tests.fast.source_scan import FRAMEWORK_ROOT

# reading the source, not the module: importing it pulls in torch_memory_saver, which a cpu shard
# does not have, and a collection error there takes down every other test in the same shard
_SOURCE = FRAMEWORK_ROOT / "backends" / "megatron_utils" / "actor.py"
_CORE_METHOD = "_load_state_core"
_LOAD_FUNCTION = "load_model_state"
_INIT_METHOD = "init"
_SLEEP_METHOD = "sleep"
_WEIGHT_UPDATER_ATTRIBUTE = "weight_updater"


def _functions_calling(name: str, *, as_method: bool) -> list[str]:
    tree = ast.parse(_SOURCE.read_text())
    calling: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            called = inner.func.attr if as_method and isinstance(inner.func, ast.Attribute) else None
            if not as_method and isinstance(inner.func, ast.Name):
                called = inner.func.id
            if called == name:
                calling.append(node.name)
    return sorted(set(calling))


class TestWhereTheTrainerLoadsItsState:
    def test_the_state_load_is_reached_from_one_function_only(self):
        """A second call site would let a reload drift from what init does, which is what this op exists to stop."""
        assert _functions_calling(_LOAD_FUNCTION, as_method=False) == [_CORE_METHOD]

    def test_init_reaches_its_state_load_through_that_function(self):
        """The invariant is worth nothing if init stopped going through the function everything else goes through."""
        assert "init" in _functions_calling(_CORE_METHOD, as_method=True)


def _function(name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(_SOURCE.read_text())
    (found,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    return found


def _method_call_lines(function_name: str, method: str) -> list[int]:
    return sorted(
        inner.lineno
        for inner in ast.walk(_function(function_name))
        if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == method
    )


def _attribute_assignment_lines(function_name: str, attribute: str) -> list[int]:
    return sorted(
        target.lineno
        for node in ast.walk(_function(function_name))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and target.attr == attribute
    )


class TestWhenTheTrainerBuildsItsWeightUpdater:
    def test_the_weight_updater_is_built_after_the_state_load(self):
        """The checkpoint load lifts expert_bias back to fp32, so a snapshot taken before it records a stale dtype."""
        (built,) = _attribute_assignment_lines(_INIT_METHOD, _WEIGHT_UPDATER_ATTRIBUTE)

        assert max(_method_call_lines(_INIT_METHOD, _CORE_METHOD)) < built

    def test_the_trainer_only_offloads_once_the_weight_updater_exists(self):
        """Building it needs a live cuda context and live process groups, which the offload sleep tears down."""
        (built,) = _attribute_assignment_lines(_INIT_METHOD, _WEIGHT_UPDATER_ATTRIBUTE)

        assert max(_method_call_lines(_INIT_METHOD, _SLEEP_METHOD)) > built

    def test_the_reusable_load_never_offloads_by_itself(self):
        """A reload runs it with the trainer awake, and an offload there would strand the caller asleep."""
        assert _method_call_lines(_CORE_METHOD, _SLEEP_METHOD) == []
