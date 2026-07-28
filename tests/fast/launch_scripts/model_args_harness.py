import subprocess

from tests.fast.launch_scripts.sh_harness import REPO_ROOT

MODEL_SCRIPT_DIR = REPO_ROOT / "scripts" / "models"

_ENV_WITHOUT_THE_MODEL_ARGS_KNOBS = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "/root",
    "LANG": "C",
    "LC_ALL": "C",
}


def iter_model_types() -> list[str]:
    return sorted(path.stem for path in MODEL_SCRIPT_DIR.glob("*.sh"))


def expand_model_args(model_type: str) -> list[str]:
    """The golden files are taken from this shell expansion; whatever replaces it must reproduce them."""
    script = MODEL_SCRIPT_DIR / f"{model_type}.sh"
    result = subprocess.run(
        f'source "{script}" && printf "%s\\n" "${{MODEL_ARGS[@]}}"',
        shell=True,
        executable="/bin/bash",
        env=_ENV_WITHOUT_THE_MODEL_ARGS_KNOBS,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()
