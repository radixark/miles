import importlib.util
import shlex
import sys
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODEL_SCRIPT_DIR = _REPO_ROOT / "scripts" / "models"


# ==================== loading a model script ====================


def load_model_args(model_type: str, model_script_dir: Path | None = None, **kwargs: object) -> str:
    """Collapse scripts/models/<model_type>.py to one line; a newline would truncate the shell's read -ra."""
    path = (model_script_dir or _MODEL_SCRIPT_DIR) / f"{model_type}.py"
    assert path.exists(), f"No model args script at {path}"
    sys.modules.setdefault("model_args_utils", sys.modules[__name__])
    module = _import_module_from_path(path, f"miles_model_args_{path.stem.replace('.', '_').replace('-', '_')}")
    args = " ".join(module.model_args(**kwargs).split())
    assert args, f"{path} declared no model args"
    return args


def load_sibling_model_args(model_script: str, model_type: str, **kwargs: object) -> str:
    """Load the model a variant is derived from, out of the same checkout as the variant itself."""
    return load_model_args(model_type, model_script_dir=Path(model_script).resolve().parent, **kwargs)


def shell_safe_model_args(model_type: str | None) -> str:
    """For callers splicing the args into a command line, where --moe-layer-freq [1,1,1] is a glob."""
    if model_type is None:
        return ""
    return " ".join(shlex.quote(token) for token in load_model_args(model_type).split())


# ==================== what a model script may call ====================


def moe_layer_freq(*, nlayers: int, first_k_dense_replace: int) -> str:
    """Render megatron's --moe-layer-freq pattern: the first K layers dense, the rest MoE."""
    dense = min(first_k_dense_replace, nlayers)
    return "[" + ",".join(["0"] * dense + ["1"] * (nlayers - dense)) + "]"


# ==================== importing a file by path ====================


def _import_module_from_path(path: Path, module_name: str) -> ModuleType:
    """Import a python file that is not reachable as a dotted module path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None, f"Cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[module_name]
    return module


# ==================== command line ====================


if __name__ == "__main__":
    (_MODEL_TYPE,) = sys.argv[1:]
    print(load_model_args(_MODEL_TYPE))
