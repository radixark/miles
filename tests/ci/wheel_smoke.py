"""An installed Miles wheel imports and parses args for FSDP and torchtitan with Megatron hidden.

Run from outside the checkout, so ``miles`` is the wheel: ``python tests/ci/wheel_smoke.py <checkout>``
"""

import importlib
import importlib.util
import sys
import sysconfig
from pathlib import Path

MEGATRON_ONLY = ("megatron", "mbridge", "transformer_engine", "apex")
BACKENDS = ("fsdp", "torchtitan")
# loaded by name at run time, so importing the drivers does not reach them
RUNTIME_MODULES = (
    "miles.rollout.inference_rollout.inference_rollout_train",
    "miles.rollout.inference_rollout.inference_rollout_eval",
    "miles.rollout.sglang_rollout",  # MILES_USE_LEGACY_ROLLOUT_V1=1
    "miles.rollout.session.server",
)
ENTRY_SCRIPTS = ("train.py", "train_async.py")


def _check_miles_is_installed() -> None:
    import miles

    site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()
    origin = Path(miles.__file__).resolve()
    assert origin.is_relative_to(site_packages), f"miles resolved to {origin}, not the wheel in {site_packages}"


def _load_entry_scripts(checkout: Path) -> None:
    for script in ENTRY_SCRIPTS:
        spec = importlib.util.spec_from_file_location(f"_wheel_smoke_{Path(script).stem}", checkout / script)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))


def _parse_args(backend: str) -> None:
    from miles.utils.arguments import parse_args

    sys.argv = ["train.py", "--train-backend", backend]
    sys.argv += ["--num-rollout", "1", "--rollout-batch-size", "1", "--rollout-max-response-len", "128"]
    args = parse_args()
    assert args.train_backend == backend


def _check_chat_templates() -> None:
    from miles.utils.chat_template_utils import TEMPLATE_DIR

    assert sorted(TEMPLATE_DIR.glob("*.jinja")), f"no chat templates shipped in {TEMPLATE_DIR}"


def main() -> None:
    checkout = Path(sys.argv[1]).resolve()
    # None: imports raise and find_spec returns None, as for an uninstalled package
    for name in MEGATRON_ONLY:
        sys.modules[name] = None

    _check_miles_is_installed()
    _load_entry_scripts(checkout)
    for backend in BACKENDS:
        importlib.import_module(f"miles.backends.{backend}_utils.actor")
        _parse_args(backend)
    for module in RUNTIME_MODULES:
        importlib.import_module(module)
    _check_chat_templates()

    print(f"wheel smoke passed: {', '.join(BACKENDS)} without Megatron")


if __name__ == "__main__":
    main()
