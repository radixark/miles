from __future__ import annotations

import subprocess
import sys

from tests.fast.source_scan import REPO_ROOT, imported_modules, imports_package, relative_paths, shipped_modules

# torchmonarch is an optional dependency: only the modules of the monarch cluster backend may import it, and
# nothing a ray or kubernetes run imports may import them at module level.
MONARCH_USING_MODULES = {
    "miles/utils/workers/monarch/runtime.py": "the runtime spawns every cell as a monarch proc mesh",
    "miles/utils/workers/monarch/manager.py": "the worker manager is a monarch actor that owns every cell",
    "miles/utils/workers/monarch/handles.py": "worker and cell handles that send monarch endpoint calls",
}


def _monarch_using_module_paths() -> list[str]:
    paths = shipped_modules(exclude_dirs=(REPO_ROOT / "tests",))
    return sorted(relative_paths(path for path in paths if imports_package(imported_modules(path), "monarch")))


class TestMonarchIsAnOptionalDependency:
    def test_no_module_outside_the_backend_imports_monarch(self):
        unlisted = sorted(set(_monarch_using_module_paths()) - set(MONARCH_USING_MODULES))

        assert unlisted == [], f"{unlisted} import monarch; keep it inside miles/utils/workers/monarch/"

    def test_the_list_names_no_module_that_stopped_using_monarch(self):
        assert sorted(set(MONARCH_USING_MODULES) - set(_monarch_using_module_paths())) == []

    def test_the_ray_path_imports_without_monarch(self):
        """A ray run must not need torchmonarch installed: block it and import what a ray driver imports."""
        code = (
            "import sys; sys.modules['monarch'] = None\n"
            "import miles.ray.wiring, miles.utils.workers.backend_capability.factory\n"
            "import miles.utils.external_utils.command_utils.base_backend\n"
        )
        completed = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)

        assert completed.returncode == 0, completed.stderr[-2000:]
