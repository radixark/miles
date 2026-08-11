import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_ROOT = REPO_ROOT / "miles"
REPLACEABLE_PACKAGE = "miles.utils.external_utils"

_TOOLING_DIRS = (
    FRAMEWORK_ROOT / "utils" / "external_utils",
    FRAMEWORK_ROOT / "utils" / "debug_utils",
    FRAMEWORK_ROOT / "utils" / "test_utils",
)

TRAIN_ONLY_SUBCOMMAND = "train"

ORCHESTRATION_SCRIPTS = ("train.py", "train_async.py", "train_multi_lora_async.py")

BACKEND_CAPABILITY_FN = "create_backend_capability"

UPPER_LAYER_MODULES = (
    "kubernetes",
    "kubernetes_asyncio",
    "miles.ray.specs.bootstrap",
    "miles.ray.specs.entrypoint",
    "miles.ray.wiring",
)

UPPER_LAYER_NAMES = (
    "ClusterBackend",
    "KubernetesWorkerProvider",
    "KubernetesBackendCapability",
    "compute_helm_backend_capability",
    "RayBackendCapability",
    "RayWorkerManager",
    "compute_ctor_kwargs",
    "compute_specs",
    "create_backend_capability",
    "get_backend_capability",
    "create_worker_backend_capability",
)

UPPER_LAYER_EXEMPTIONS = {
    "miles/ray/specs": "the composition root of a worker process: a spec says what its worker is built from",
    "miles/ray/wiring.py": "the glue layer holding the driver process's single fork between the backends",
    "train.py": "orchestration script: its first lines are the driver process's composition root",
    "train_async.py": "orchestration script: its first lines are the driver process's composition root",
    "train_multi_lora_async.py": "orchestration script: its first lines are the driver process's composition root",
    "miles/utils/workers/worker_provider": "the infrastructure that owns every provider implementation",
    "miles/utils/workers/serving/serve_inner.py": "the composition root of a served worker process",
    "miles/utils/workers/ray_worker_manager.py": "the composition root of a worker process an actor wraps",
    "miles/utils/workers/reconcile/k8s_api.py": "the kubernetes client the observing provider is written against",
    "miles/utils/arguments.py": "declares the --cluster-backend flag the composition roots read",
    "miles/utils/tracking_utils/base.py": "the prometheus collector is a ray actor and has no kubernetes form",
}


def _framework_modules() -> list[Path]:
    return sorted(
        path
        for path in FRAMEWORK_ROOT.rglob("*.py")
        if not any(path.is_relative_to(directory) for directory in _TOOLING_DIRS) and "__pycache__" not in path.parts
    )


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    return imported


def _layered_modules() -> list[Path]:
    candidates = [*_framework_modules(), *(REPO_ROOT / name for name in ORCHESTRATION_SCRIPTS)]
    return [path for path in candidates if not _is_exempt(path)]


def _is_exempt(path: Path) -> bool:
    relative = path.relative_to(REPO_ROOT).as_posix()
    return any(relative == exempt or relative.startswith(f"{exempt}/") for exempt in UPPER_LAYER_EXEMPTIONS)


def _exempted_files(exemption: str) -> list[Path]:
    target = REPO_ROOT / exemption
    if not target.is_dir():
        return [target]
    return sorted(path for path in target.rglob("*.py") if "__pycache__" not in path.parts)


def _upper_layer_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names if _is_upper_layer_module(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None and node.level == 0 and _is_upper_layer_module(node.module):
                found.append(node.module)
            found.extend(alias.name for alias in node.names if alias.name in UPPER_LAYER_NAMES)
    return found


def _is_upper_layer_module(module: str) -> bool:
    return any(module == name or module.startswith(f"{name}.") for name in UPPER_LAYER_MODULES)


def _calls_of(path: Path, name: str) -> list[ast.Call]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


class TestLayering:
    def test_only_the_listed_files_learn_which_backend_the_workers_come_from(self):
        """Every other module is handed the abstractions it needs, so knowing the backend would be a second answer."""
        offenders = [
            f"{path.relative_to(REPO_ROOT)} imports {name}"
            for path in _layered_modules()
            for name in _upper_layer_imports(path)
        ]

        assert offenders == [], offenders

    def test_the_check_sees_the_upper_layer_knowledge_the_glue_layer_holds(self):
        """A check that finds nothing anywhere would pass on a codebase that reaches upwards from everywhere."""
        assert _upper_layer_imports(FRAMEWORK_ROOT / "ray" / "wiring.py") != []

    @pytest.mark.parametrize("exemption", sorted(UPPER_LAYER_EXEMPTIONS))
    def test_an_exemption_still_names_a_file_of_this_repo(self, exemption: str):
        """An exemption nobody removes outlives the file it was written for and quietly widens the rule."""
        assert (REPO_ROOT / exemption).exists()

    @pytest.mark.parametrize("exemption", sorted(UPPER_LAYER_EXEMPTIONS))
    def test_an_exemption_still_covers_a_file_that_reaches_upwards(self, exemption: str):
        """Once the code it was written for stops reaching upwards, the exemption only shelters the next arrival."""
        reaching = [path for path in _exempted_files(exemption) if _upper_layer_imports(path)]

        assert reaching != [], f"{exemption} no longer reaches upwards: {UPPER_LAYER_EXEMPTIONS[exemption]}"

    @pytest.mark.parametrize("script", ORCHESTRATION_SCRIPTS)
    def test_an_orchestration_script_forks_the_backend_exactly_once(self, script: str):
        """The whole run hangs off one factory, and a second one would observe the same workers twice."""
        assert len(_calls_of(REPO_ROOT / script, BACKEND_CAPABILITY_FN)) == 1


class TestImportDirection:
    def test_the_framework_never_imports_the_replaceable_deployment_code(self):
        """A platform replacing the charts must be able to drop this half; an import would make it load anyway."""
        offenders = [
            f"{path.relative_to(REPO_ROOT)} imports {module}"
            for path in _framework_modules()
            for module in _imported_modules(path)
            if module == REPLACEABLE_PACKAGE or module.startswith(f"{REPLACEABLE_PACKAGE}.")
        ]

        assert offenders == [], offenders

    def test_the_replaceable_code_may_import_the_framework(self):
        """The dependency has to point one way, and this is the way it points."""
        launcher = (
            FRAMEWORK_ROOT
            / "utils"
            / "external_utils"
            / "command_utils"
            / "helm_backend"
            / "launcher"
            / "entrypoint.py"
        )

        assert any(module.startswith("miles.utils.workers") for module in _imported_modules(launcher))


class TestLaunchScriptContract:
    @pytest.mark.parametrize("script", sorted((REPO_ROOT / "scripts").glob("run_*.py")), ids=lambda path: path.name)
    def test_a_train_subcommand_only_trains(self, script):
        """The Kubernetes backend runs this subcommand in a pod, where preparation has already happened."""
        tree = ast.parse(script.read_text(), filename=str(script))
        train = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == TRAIN_ONLY_SUBCOMMAND),
            None,
        )
        if train is None:
            pytest.skip(f"{script.name} has no {TRAIN_ONLY_SUBCOMMAND} subcommand, so it stays Ray only")

        called = {
            node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            for node in ast.walk(train)
            if isinstance(node, ast.Call)
        }
        prepared = {name for name in called if "prepare" in name or "download" in name or "convert" in name}

        assert prepared == set(), f"{script.name} prepares data inside {TRAIN_ONLY_SUBCOMMAND}: {prepared}"
