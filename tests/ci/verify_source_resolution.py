import os
from importlib.util import find_spec
from pathlib import Path


MODULE_ROOT_ENV = {
    "miles": "GITHUB_WORKSPACE",
    "miles_plugins": "GITHUB_WORKSPACE",
    "sglang": "SGLANG_SOURCE_ROOT",
    "megatron.core": "MEGATRON_SOURCE_ROOT",
    "megatron.training": "MEGATRON_SOURCE_ROOT",
}


def main() -> None:
    module_roots = {name: Path(os.environ[root_env]).resolve() for name, root_env in MODULE_ROOT_ENV.items()}
    module_roots["radix_transfer_plugins"] = (
        Path(os.environ["SGLANG_SOURCE_ROOT"]).resolve().parent / "radix_transfer_plugins"
    )
    for module_name, expected_root in module_roots.items():
        spec = find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"cannot resolve {module_name}")
        origin = Path(spec.origin).resolve()
        try:
            origin.relative_to(expected_root)
        except ValueError as exc:
            raise RuntimeError(f"{module_name} resolved to {origin}, expected {expected_root}") from exc
        print(f"{module_name}: {origin}")


if __name__ == "__main__":
    main()
