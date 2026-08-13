from __future__ import annotations

import ast
import functools
from pathlib import Path

from tests.fast.source_scan import FRAMEWORK_ROOT, REPO_ROOT, shipped_modules

from miles.utils.workers.rpc.common.wire_types import PICKLED_HATCH_MARKER

EXCLUDED_DIRS = (REPO_ROOT / "tests",)

HATCH_NAME = "Pickled"

HATCH_DEFINITION = FRAMEWORK_ROOT / "utils" / "workers" / "rpc" / "common" / "wire_types.py"

RETURN_ANNOTATION = "return"

PICKLED_PARAMETERS = {
    ("miles/ray/train/group.py", "TrainerController.init", "args"),
    ("miles/ray/train_actor.py", "TrainRayActor.init", "args"),
    ("miles/backends/megatron_utils/actor.py", "MegatronTrainRayActor.init", "args"),
    ("miles/backends/fsdp_utils/actor.py", "FSDPTrainRayActor.init", "args"),
}


def _hatch_aliases(tree: ast.Module) -> set[str]:
    aliases = {HATCH_NAME}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            aliases.update(alias.asname or alias.name for alias in node.names if alias.name == HATCH_NAME)
        elif isinstance(node, ast.Assign) and _is_pickled(node.value, aliases):
            aliases.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return aliases


def _is_pickled(annotation: ast.expr | None, aliases: set[str]) -> bool:
    if annotation is None:
        return False
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id in aliases:
            return True
        if isinstance(node, ast.Attribute) and node.attr == HATCH_NAME:
            return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _is_pickled_text(node.value, aliases):
            return True
    return False


def _is_pickled_text(text: str, aliases: set[str]) -> bool:
    try:
        parsed = ast.parse(text.strip(), mode="eval")
    except SyntaxError:
        return False
    return _is_pickled(parsed.body, aliases)


def _pickled_annotations_in(source: str, *, label: str) -> set[tuple[str, str, str]]:
    tree = ast.parse(source)
    aliases = _hatch_aliases(tree)
    found: set[tuple[str, str, str]] = set()
    for class_node in ast.walk(tree):
        if not isinstance(class_node, ast.ClassDef):
            continue
        for node in ast.walk(class_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = node.args
                for parameter in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]:
                    if _is_pickled(parameter.annotation, aliases):
                        found.add((label, f"{class_node.name}.{node.name}", parameter.arg))
                if _is_pickled(node.returns, aliases):
                    found.add((label, f"{class_node.name}.{node.name}", RETURN_ANNOTATION))
            elif isinstance(node, ast.AnnAssign) and _is_pickled(node.annotation, aliases):
                found.add((label, class_node.name, ast.unparse(node.target)))
    return found


def _pickled_annotations_of(path: Path) -> set[tuple[str, str, str]]:
    return _pickled_annotations_in(path.read_text(), label=str(path.relative_to(REPO_ROOT)))


@functools.cache
def _all_pickled_annotations() -> frozenset[tuple[str, str, str]]:
    return frozenset(
        entry for path in shipped_modules(exclude_dirs=EXCLUDED_DIRS) for entry in _pickled_annotations_of(path)
    )


class TestThePickledHatchStaysWhitelisted:
    def test_nothing_uses_the_hatch_without_being_listed(self):
        """The hatch punches a hole in the strict wire typing, so every user of it is reviewed here."""
        unlisted = sorted(_all_pickled_annotations() - PICKLED_PARAMETERS)

        assert unlisted == [], (
            f"{unlisted} use the Pickled escape hatch; it is reserved for the argparse Namespace a trainer "
            f"is built from, so give the parameter a wire type instead"
        )

    def test_the_list_names_no_parameter_that_no_longer_uses_it(self):
        """A stale entry is an invitation to reach for pickle where a wire type already exists."""
        stale = sorted(PICKLED_PARAMETERS - _all_pickled_annotations())

        assert stale == []

    def test_every_listed_parameter_is_the_args_namespace(self):
        """Only args was authorized; anything else crossing as pickle is a different decision."""
        assert {parameter for _path, _method, parameter in PICKLED_PARAMETERS} == {"args"}

    def test_the_hatch_carries_a_greppable_marker_and_a_todo(self):
        """It is temporary, and the way it gets reclaimed is by finding it."""
        source = HATCH_DEFINITION.read_text()

        assert f"TODO({PICKLED_HATCH_MARKER})" in source
        assert "arguments subsystem" in source


class TestTheShapesThatUsedToCarryTheHatchThrough:
    def test_a_union_is_seen(self):
        """`Pickled | None` is the same hole with an extra bar in it."""
        source = "class W:\n    def go(self, args: Pickled | None) -> None: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", "args")}

    def test_a_subscript_is_seen(self):
        """A container of pickled values pickles just as much as one value does."""
        source = "class W:\n    def go(self, args: list[Pickled]) -> None: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", "args")}

    def test_an_aliased_import_is_seen(self):
        """Renaming the hatch on the way in does not make the parameter wire-typed."""
        source = "from miles.utils.workers.rpc.common.wire_types import Pickled as _P\n"
        source += "class W:\n    def go(self, args: _P) -> None: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", "args")}

    def test_a_module_level_alias_is_seen(self):
        """A second name for the same annotation is still the same annotation."""
        source = "Blob = Pickled\nclass W:\n    def go(self, args: Blob) -> None: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", "args")}

    def test_a_string_annotation_is_seen(self):
        """Quoting an annotation is how a forward reference is written, not how a rule is escaped."""
        source = 'class W:\n    def go(self, args: "Pickled | None") -> None: ...\n'

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", "args")}

    def test_a_method_defined_under_a_type_checking_guard_is_seen(self):
        """A method nested in any block is still a method the served class exposes."""
        source = "class W:\n    if TYPE_CHECKING:\n        def go(self, args: Pickled) -> None: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", "args")}

    def test_a_return_annotation_is_seen(self):
        """A method that answers pickle sends it in the direction the parameter check never looks."""
        source = "class W:\n    def go(self) -> Pickled: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "W.go", RETURN_ANNOTATION)}

    def test_a_wire_model_field_is_seen(self):
        """A hatch inside a model field crosses the wire on every call that carries the model."""
        source = "class M(StrictBaseModel):\n    args: Pickled\n"

        assert _pickled_annotations_in(source, label="w.py") == {("w.py", "M", "args")}

    def test_a_wire_typed_parameter_is_not_reported(self):
        """A check that reports everything would make the ledger meaningless."""
        source = "class W:\n    def go(self, args: int) -> None: ...\n"

        assert _pickled_annotations_in(source, label="w.py") == set()
