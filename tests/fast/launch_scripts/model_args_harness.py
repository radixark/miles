from tests.fast.launch_scripts.sh_harness import REPO_ROOT

from miles.utils.external_utils.model_args_utils import load_model_args

MODEL_SCRIPT_DIR = REPO_ROOT / "scripts" / "models"


def iter_model_types() -> list[str]:
    return sorted(path.stem for path in MODEL_SCRIPT_DIR.glob("*.py"))


def expand_model_args(model_type: str) -> list[str]:
    """Only the producer changed here; the golden files still hold what the shell era expanded to."""
    return load_model_args(model_type).split()
