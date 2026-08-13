import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "tests/e2e/short/test_qwen2.5_0.5B_external_rollout.py"


def load_external_rollout_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("external_rollout_e2e", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
