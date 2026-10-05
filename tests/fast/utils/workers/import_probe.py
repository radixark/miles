from __future__ import annotations

import sys

IMPORTED_MODULES_SEPARATOR = ","

FORBIDDEN_LIGHT_ENTRYPOINT_IMPORTS = frozenset(
    {"torch", "megatron", "sglang", "vllm", "transformers", "deepspeed", "ray", "uvicorn"}
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
    return sorted(imported & FORBIDDEN_LIGHT_ENTRYPOINT_IMPORTS)
