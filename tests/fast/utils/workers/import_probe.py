from __future__ import annotations

import sys

IMPORTED_MODULES_SEPARATOR = ","

# Everything here lands in sys.modules through site initialization, before the
# entrypoint runs a line of its own, so none of it says anything about what the
# entrypoint imports. nvidia_cutlass_dsl arrives via a .pth in dist-packages.
ALLOWED_LIGHT_ENTRYPOINT_IMPORTS = frozenset(
    {
        "__main__",
        "miles",
        "tests",
        "sitecustomize",
        "usercustomize",
        "_distutils_hack",
        "_virtualenv",
        "nvidia_cutlass_dsl",
    }
)

# Real dependencies the light entrypoint is allowed to import before the
# env-var hook runs; each entry is a deliberate decision that the module is
# acceptably light, unlike the site-initialization noise listed above.
ACCEPTED_LIGHT_ENTRYPOINT_DEPENDENCIES = frozenset(
    {
        "pydantic",
        "pydantic_core",
        "annotated_types",
        "typing_extensions",
        "typing_inspection",
    }
)


_INSTALLER_SHIM_PREFIX = "__editable__"
_SYSCONFIGDATA_PREFIX = "_sysconfigdata"


def imported_top_level_modules() -> set[str]:
    top_level_names = {name.partition(".")[0] for name in sys.modules}
    return {
        name
        for name in top_level_names
        if name not in sys.stdlib_module_names
        and not name.startswith(_INSTALLER_SHIM_PREFIX)
        and not name.startswith(_SYSCONFIGDATA_PREFIX)
    }


def report_imported_top_level_modules() -> str:
    return IMPORTED_MODULES_SEPARATOR.join(sorted(imported_top_level_modules()))


def unexpected_light_entrypoint_imports(reported: str) -> list[str]:
    imported = {name for name in reported.split(IMPORTED_MODULES_SEPARATOR) if name}
    return sorted(imported - ALLOWED_LIGHT_ENTRYPOINT_IMPORTS - ACCEPTED_LIGHT_ENTRYPOINT_DEPENDENCIES)
