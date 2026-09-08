"""Sandbox-provider credentials and endpoints for agent-function launchers.

The contract every sandbox backend shares: rollout workers read a provider
credential from a key file; the launcher forwards only the file PATH (never
the value, which would ride ray's runtime_env in plaintext) and address-like
variables by value. A set key env var is rejected: env supply used to win
silently over a missing or unread file, which is how a worker-side resolver
gap passed every unit test until the first GPU e2e run.

PROVIDER_CREDENTIALS holds one entry per backend in
openenv_sandbox_common.AGENT_MODULES; a provider is added there, not by
growing a branch:

  # 2026-09-09, tianqi, file-only sandbox credentials (#3111)
  key_env_vars   vars that must be UNSET; a set value is rejected (env supply
                 used to silently mask a missing file resolver). Modal's
                 pair is both halves
  # end
  file_env_var   the path-valued var the launcher forwards instead of secrets
  forward        addresses/selectors, safe to forward by value
  target         (var, label, default description) echoed so a launch says
                 which endpoint/environment it will actually use
  sdk_min_version  optional; the floor the extra in setup.py declares,
                 enforced by preflight_sdk
"""

import importlib
import importlib.metadata
import os
from pathlib import Path

PROVIDER_CREDENTIALS = {
    "daytona": {
        "provider": "Daytona",
        "key_env_vars": ("DAYTONA_API_KEY",),
        "file_env_var": "DAYTONA_API_KEY_FILE",
        "arg_attr": "daytona_api_key_file",
        "default_path": "~/.config/daytona/api_key",
        "provision_hint": "mkdir -p ~/.config/daytona && echo dtn_... > ~/.config/daytona/api_key",
        "sdk": "daytona",
        "sdk_hint": "pip install daytona (or pip install -e '<OpenEnv>/envs/tbench2_env[daytona]')",
        "forward": (),
        "target": None,
    },
    "e2b": {
        "provider": "E2B",
        "key_env_vars": ("E2B_API_KEY",),
        "file_env_var": "E2B_API_KEY_FILE",
        "arg_attr": "e2b_api_key_file",
        "default_path": "~/.config/e2b/api_key",
        "provision_hint": "mkdir -p ~/.config/e2b && echo <key> > ~/.config/e2b/api_key"
        "  # AgentENV accepts any non-empty key today",
        "sdk": "e2b",
        "sdk_hint": "pip install -e '<miles>[e2b]'",
        # Older releases send the template name as `alias`, which self-hosted
        # AgentENV does not read; the failure is a bare 400 at the first bake.
        "sdk_min_version": "2.12",
        # E2B_API_URL unset means E2B Cloud; set, it usually points at a
        # self-hosted AgentENV deployment.
        "forward": ("E2B_API_URL", "E2B_SANDBOX_URL", "E2B_DOMAIN", "OPENENV_E2B_URL_SCHEME"),
        "target": ("E2B_API_URL", "endpoint", "E2B Cloud (default)"),
    },
    "modal": {
        "provider": "Modal",
        # 2026-09-09, tianqi, file-only sandbox credentials (#3111)
        # Modal has no single API key: the SDK reads the config file whose path
        # MODAL_CONFIG_PATH names. Token env vars are rejected, not used.
        # end
        "key_env_vars": ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
        "file_env_var": "MODAL_CONFIG_PATH",
        "arg_attr": "modal_config_file",
        "default_path": "~/.modal.toml",
        "provision_hint": "uv tool install modal && modal token new  # writes ~/.modal.toml",
        "sdk": "modal",
        "sdk_hint": "pip install modal",
        "forward": ("MODAL_PROFILE", "MODAL_ENVIRONMENT", "OPENENV_MODAL_APP"),
        "target": ("MODAL_ENVIRONMENT", "workspace environment", "the profile's default"),
    },
}


def forward_address(env: dict[str, str], var: str, value: str) -> None:
    """Forward one address-like var by value, refusing one that carries a secret.

    What ``forward`` names are endpoints and selectors, which is why they may
    ride ray's runtime_env at all while a credential may not (only its PATH is
    forwarded — see sandbox_key_supply). A URL defeats that distinction by
    smuggling a credential through userinfo (``https://user:token@host``), and
    runtime_env is echoed into driver logs and persisted in job metadata in
    plaintext, so refuse it rather than forward it. Nothing legitimately
    forwarded here — a hostname, a scheme, a profile or app name — contains an
    '@', so the check needs no URL parsing to be precise.
    """
    if "@" in value:
        raise ValueError(
            f"{var} looks like it embeds credentials ('@'), and anything forwarded "
            "to rollout workers is logged in plaintext by ray. Put the credential "
            # 2026-09-09, tianqi, file-only sandbox credentials (#3111)
            "in the provider's key file and leave a bare address here."
            # end
        )
    env[var] = value


def sandbox_key_supply(
    env: dict[str, str],
    *,
    provider: str,
    key_env_vars: tuple[str, ...],
    file_env_var: str,
    arg_path: str,
    default_path: str,
    provision_hint: str,
) -> None:
    # 2026-09-09, tianqi, file-only sandbox credentials (#3111)
    """Key-supply contract, shared by the sandbox backends: rollout workers
    read the provider credential from a file they can read (a dotfile, K8s
    Secret mount, or shared-FS path). The launcher forwards only the file PATH,
    never the value: worker env rides ray's runtime_env, which
    exec_command_cpu echoes into driver logs and ray persists in job metadata,
    all in plaintext. A set key env var is rejected so it cannot silently
    satisfy preflight while the file path is missing or unread."""
    key_file = Path(arg_path or default_path).expanduser()
    set_vars = tuple(var for var in key_env_vars if os.environ.get(var, "").strip())
    if set_vars:
        names = " + ".join(set_vars)
        raise ValueError(_file_only_env_message(names, key_file, file_env_var, provision_hint))
    try:
        key_present = bool(key_file.read_text(encoding="utf-8").strip())
    except OSError:
        key_present = False
    if key_present:
        env[file_env_var] = str(key_file)
        print(
            f"{provider} credential supply: file {key_file} "
            "(readable here; forwarding the path, workers read it themselves)",
            flush=True,
        )
    elif arg_path:
        # An explicitly configured path that doesn't resolve on the launcher
        # is a config error; failing every episode later is far worse.
        raise ValueError(f"{file_env_var}={arg_path} is missing or empty")
    else:
        raise ValueError(
            f"the {provider} sandbox mode needs credentials: put them in a file "
            f"({key_file}; {file_env_var} overrides). Provision the file with:\n"
            f"  {provision_hint}"
        )
    # end


def preflight_sdk(module: str, install_hint: str, min_version: str | None = None) -> None:
    """Preflight the lazily-imported provider SDK. Without this, a missing
    install only surfaces inside each episode's sandbox start, where the
    failed sample is aborted, the group dropped, and the rollout loop refills
    forever — a silent GPU-burning churn instead of a launch-time error.
    *min_version*, when given, is the floor the extra in setup.py declares;
    checking it here catches an SDK installed by hand or preinstalled in an
    image, whose failure mode would otherwise be a provider error with no hint
    that the version is the cause."""
    try:
        importlib.import_module(module)
    except ImportError as e:
        raise RuntimeError(
            f"this sandbox mode needs the {module} SDK in the rollout process's environment: {install_hint}"
        ) from e
    if min_version is None:
        return
    try:
        # Looking the distribution up by the module name assumes the two match,
        # which holds for every backend that sets a floor today (e2b).
        installed = importlib.metadata.version(module)
    except importlib.metadata.PackageNotFoundError:
        # Importable but no dist-info (a vendored copy): the version cannot be
        # verified, and blocking a possibly-fine install is worse than leaving
        # a too-old one to fail at the provider with its own error.
        print(f"{module} has no package metadata; cannot verify {module}>={min_version}", flush=True)
        return
    if _version_tuple(installed) < _version_tuple(min_version):
        raise RuntimeError(f"this sandbox mode needs {module}>={min_version} (installed: {installed}): {install_hint}")


def _version_tuple(version: str) -> tuple[int, ...]:
    """Numeric leading components of a version string, enough to compare
    release floors like 2.12 against 2.46.0 or 2.12.0rc1."""
    parts = []
    for piece in version.split("."):
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _file_only_env_message(names: str, key_file: Path, file_env_var: str, provision_hint: str | None = None) -> str:
    # 2026-09-09, tianqi, file-only sandbox credentials (#3111)
    msg = (
        f"{names} is set; sandbox credentials are file-only "
        f"(env supply silently masked a missing file resolver). "
        f"Unset {names} and put the key in {key_file} ({file_env_var} overrides)"
    )
    if provision_hint:
        msg += f". Provision the file with:\n  {provision_hint}"
    return msg
    # end


def resolve_provider_api_key(env_var: str, file_env_var: str, default_path: str) -> str:
    # 2026-09-09, tianqi, file-only sandbox credentials (#3111)
    """A provider API key from the key file only.

    The file indirection (*file_env_var*, default *default_path*) exists so
    launchers can hand rollout workers a PATH instead of the secret itself:
    anything a launcher forwards rides ray's runtime_env, which is echoed into
    driver logs and persisted in job metadata in plaintext. *env_var* is
    not a supply path: if it is set, this errors and names the file to create,
    rather than silently winning over a missing or unread file.
    """
    key_file = Path(os.environ.get(file_env_var, "").strip() or default_path).expanduser()
    if os.environ.get(env_var, "").strip():
        raise RuntimeError(_file_only_env_message(env_var, key_file, file_env_var))
    try:
        key = key_file.read_text(encoding="utf-8").strip()
    except OSError:
        key = ""
    if not key:
        raise RuntimeError(f"no API key: {key_file} is missing or empty ({file_env_var} overrides)")
    return key
    # end
