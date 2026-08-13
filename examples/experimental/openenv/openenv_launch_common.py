"""Shared launch helpers for the OpenEnv tbench2 learning launchers.

``run-openenv-tbench2.py`` (GLM-4.7-Flash) is the launcher in this example;
sibling per-model launchers (e.g. a DeepSeek-V4-Flash variant) reuse the same
agentic adapter and differ only in the model-family serving/training profile.
The model-agnostic fragments (process cleanup, GRPO/optimizer/rollout/agent
flags, W&B + Prometheus wiring, and the OpenEnv env-var plumbing) live here so
those launchers cannot silently drift apart. Each launcher keeps only its own
perf/sglang/misc profile and its ``ScriptArgs`` defaults.
"""

import importlib
import os
import subprocess
import time
from pathlib import Path
from typing import Protocol

import openenv_sandbox_common as sandbox_common


class LaunchArgs(Protocol):
    """The config fields the shared helpers read (satisfied by each launcher's ScriptArgs)."""

    prompt_data: str
    rollout_batch_size: int
    n_samples_per_prompt: int
    max_seq_len: int
    global_batch_size: int

    openenv_env_url: str
    agent_model_name: str
    openenv_max_turns: int
    openenv_max_rollout_time_seconds: int
    openenv_tb2_tasks_dir: str
    openenv_sandbox_backend: str
    daytona_api_key_file: str
    e2b_api_key_file: str
    modal_config_file: str
    router_external_host: str
    miles_host_ip: str

    wandb_key: str
    wandb_project: str
    wandb_team: str
    wandb_run_name: str

    use_prometheus: bool
    prometheus_port: int
    prometheus_run_name: str


def cleanup() -> None:
    """Kill old Ray jobs and stale processes to free GPU resources."""
    my_pid = os.getpid()
    ppid = os.getppid()
    print(f"Cleanup starting (pid={my_pid}, ppid={ppid})")
    targets = ["sglang", "train.py", "MegatronTrain"]
    exclude = f"grep -v '^{my_pid}$' | grep -v '^{ppid}$'"
    for t in targets:
        subprocess.run(
            f"pgrep -f '{t}' | {exclude} | xargs -r kill 2>/dev/null || true",
            shell=True,
        )
    time.sleep(5)
    print(f"Cleanup complete (pid={my_pid}) — old processes killed.")


def rollout_args(args: LaunchArgs) -> str:
    return (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--metadata-key metadata "
        "--rollout-shuffle "
        "--num-rollout 40 "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        "--rollout-temperature 0.8 "
        "--rollout-max-response-len 8192 "
        f"--max-seq-len {args.max_seq_len} "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
    )


def grpo_args() -> str:
    return (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.01 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )


def optimizer_args() -> str:
    return (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )


def resolve_sandbox_backend(args: LaunchArgs) -> str:
    """The per-episode sandbox backend in effect, or "" for the shared env server.

    Names and aliases resolve through openenv_sandbox_common, the canonical
    registry, so the accepted set is never enumerated twice: "agentenv" is an
    accepted alias for "e2b", because AgentENV
    (https://github.com/kvcache-ai/AgentENV) is a self-hosted Firecracker
    microVM platform whose native API is the E2B API, so it runs on the e2b
    backend with E2B_API_URL/E2B_SANDBOX_URL pointed at it.

    The two settings that turn this mode on come as a pair — a task checkout
    to build images from, and the provider to build them on — so naming one
    without the other is an error rather than a guess.
    """
    raw = (getattr(args, "openenv_sandbox_backend", "") or "").strip()
    tasks_dir = args.openenv_tb2_tasks_dir
    if not raw and not tasks_dir:
        return ""
    if not tasks_dir:
        raise ValueError(
            "sandbox backends build per-task images: set openenv_tb2_tasks_dir to a terminal-bench-2 checkout"
        )
    if not raw:
        raise ValueError(
            "openenv_tb2_tasks_dir selects per-episode sandboxes: set openenv_sandbox_backend "
            "(OPENENV_SANDBOX_BACKEND) to the provider to run them on"
        )
    return sandbox_common.resolve_backend(raw)


def agent_args(tito_model: str, sandbox_backend: str = "") -> str:
    """Agentic-rollout wiring. The TITO surface differs across models; the
    agent function decides where episodes run — per-episode sandboxes on
    whichever backend the launcher resolves (see resolve_sandbox_backend), else
    the one shared env server."""
    agent_fn = sandbox_common.AGENT_FUNCTIONS.get(sandbox_backend, "openenv_agent_function.run")
    return (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        f"--custom-agent-function-path {agent_fn} "
        "--custom-rm-path openenv_generate.reward_func "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--tito-model {tito_model} "
        "--use-session-server "
        "--session-server-port 30000 "
    )


def wandb_args(args: LaunchArgs) -> str:
    if not args.wandb_key:
        return ""
    out = (
        "--use-wandb "
        f"--wandb-project {args.wandb_project} "
        f"--wandb-group {args.wandb_run_name} "
        f"--wandb-key {args.wandb_key} "
    )
    if args.wandb_team:
        out += f"--wandb-team {args.wandb_team} "
    return out


def prometheus_args(args: LaunchArgs) -> str:
    if not args.use_prometheus:
        return ""
    return (
        "--use-prometheus "
        f"--prometheus-port {args.prometheus_port} "
        f"--prometheus-run-name {args.prometheus_run_name} "
    )


def base_env_vars(args: LaunchArgs, script_dir: str, megatron_path: str, miles_root: str) -> dict[str, str]:
    return {
        "PYTHONPATH": f"{megatron_path}:{script_dir}:{miles_root}",
        "OPENENV_ENV_URL": args.openenv_env_url,
        "OPENENV_MAX_TURNS": str(args.openenv_max_turns),
        "OPENENV_MAX_ROLLOUT_TIME_SECONDS": str(args.openenv_max_rollout_time_seconds),
        "AGENT_MODEL_NAME": args.agent_model_name,
    }


def apply_optional_env_vars(env: dict[str, str], args: LaunchArgs) -> None:
    """Add host-rewrite / Daytona-sandbox env vars when the args request them."""
    if args.miles_host_ip:
        env["MILES_HOST_IP"] = args.miles_host_ip
    if args.router_external_host:
        env["MILES_ROUTER_EXTERNAL_HOST"] = args.router_external_host
    backend = resolve_sandbox_backend(args)
    if backend:
        spec = _PROVIDER_CREDENTIALS[backend]
        _sandbox_key_supply(
            env,
            provider=spec["provider"],
            key_env_vars=spec["key_env_vars"],
            file_env_var=spec["file_env_var"],
            arg_path=getattr(args, spec["arg_attr"], "") or "",
            default_path=spec["default_path"],
            provision_hint=spec["provision_hint"],
        )
        _preflight_sdk(spec["sdk"], spec["sdk_hint"])
        # Addresses, not secrets: the SDK reads these from the environment on
        # every worker, so forward whatever is set here BY VALUE.
        for var in spec["forward"]:
            value = os.environ.get(var, "").strip()
            if value:
                _forward_address(env, var, value)
        if spec["target"]:
            var, label, default_desc = spec["target"]
            print(f"openenv: {spec['provider']} {label}: {env.get(var, default_desc)}", flush=True)
        # Preflight the env package the recipe bakes into each task image —
        # shared by every sandbox backend. The import check catches a missing
        # install; the source probe catches an install that imports fine but
        # lacks the server features the sandbox backends score through (canonical
        # tests/test.sh evaluate, TB2_WITHHOLD_TESTS) — that one would not
        # even fail per-episode, it would silently mis-score every episode.
        try:
            import tbench2_env
        except ImportError as e:
            raise RuntimeError(
                "the sandbox modes need tbench2_env in the rollout "
                "process's environment: pip install -e '<OpenEnv>/envs/tbench2_env' "
                "from the checkout described in this directory's README"
            ) from e
        server_src = Path(tbench2_env.__file__).resolve().parent / "server" / "tbench2_env_environment.py"
        src_text = server_src.read_text(encoding="utf-8") if server_src.is_file() else ""
        if "TB2_WITHHOLD_TESTS" not in src_text:
            raise RuntimeError(
                "the installed tbench2_env server lacks the native-evaluate "
                "contract (canonical test.sh scoring / TB2_WITHHOLD_TESTS): "
                "install from an OpenEnv checkout at or after the #1012 merge "
                "(04d259ea6) — see this directory's README"
            )
        env["OPENENV_TB2_TASKS_DIR"] = args.openenv_tb2_tasks_dir


# Per-provider credential shape and the address-like env vars to forward.
# One entry per backend in openenv_sandbox_common.AGENT_MODULES; a provider is
# added here, not by growing a branch.
#   key_env_vars   what a worker must ALL have for the env-supply path to work
#                  (Modal's credential is a token PAIR, not one key)
#   file_env_var   the path-valued var the launcher forwards instead of secrets
#   forward        addresses/selectors, safe to forward by value
#   target         (var, label, default description) echoed so a launch says
#                  which endpoint/environment it will actually use
_PROVIDER_CREDENTIALS = {
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
        "sdk_hint": "pip install e2b",
        # E2B_API_URL unset means E2B Cloud; set, it usually points at a
        # self-hosted AgentENV deployment.
        "forward": ("E2B_API_URL", "E2B_SANDBOX_URL", "E2B_DOMAIN", "OPENENV_E2B_URL_SCHEME"),
        "target": ("E2B_API_URL", "endpoint", "E2B Cloud (default)"),
    },
    "modal": {
        "provider": "Modal",
        # Modal has no single API key: the SDK wants both token halves, or the
        # config file whose path MODAL_CONFIG_PATH names.
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


def _forward_address(env: dict[str, str], var: str, value: str) -> None:
    """Forward one address-like var by value, refusing one that carries a secret.

    What ``forward`` names are endpoints and selectors, which is why they may
    ride ray's runtime_env at all while a credential may not (only its PATH is
    forwarded — see _sandbox_key_supply). A URL defeats that distinction by
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
            "in the provider's key file (or the worker environment) and leave a "
            "bare address here."
        )
    env[var] = value


def _sandbox_key_supply(
    env: dict[str, str],
    *,
    provider: str,
    key_env_vars: tuple[str, ...],
    file_env_var: str,
    arg_path: str,
    default_path: str,
    provision_hint: str,
) -> None:
    """Key-supply contract, shared by the sandbox backends: rollout workers
    get the provider credential from their OWN environment (e.g.
    platform-injected) or from a file they can read (a dotfile, K8s Secret
    mount, or shared-FS path). The launcher forwards only the file PATH, never
    the value: worker env rides ray's runtime_env, which exec_command_cpu
    echoes into driver logs and ray persists in job metadata, all in
    plaintext."""
    key_file = Path(arg_path or default_path).expanduser()
    try:
        key_present = bool(key_file.read_text(encoding="utf-8").strip())
    except OSError:
        key_present = False
    # Either supply is fine; neither is fully verifiable from here (the
    # launcher cannot probe worker nodes), so echo which one is in effect.
    # A provider whose credential is several variables (Modal's token pair) is
    # only satisfied by having ALL of them; a partial set is a misconfiguration
    # that would fail every episode.
    names = " + ".join(key_env_vars)
    if key_present:
        env[file_env_var] = str(key_file)
        print(
            f"openenv: {provider} credential supply: file {key_file} "
            "(readable here; forwarding the path, workers read it themselves)",
            flush=True,
        )
    elif arg_path:
        # An explicitly configured path that doesn't resolve on the launcher
        # is a config error; failing every episode later is far worse.
        raise ValueError(f"{file_env_var}={arg_path} is missing or empty")
    elif all(os.environ.get(var, "").strip() for var in key_env_vars):
        print(
            f"openenv: {provider} credential supply: worker environment ({names} "
            "set here; workers are assumed to have them in their own env — "
            "single-host inheritance or platform-injected pod env)",
            flush=True,
        )
    else:
        raise ValueError(
            f"the {provider} sandbox mode needs credentials: put them in a file "
            f"({key_file}; {file_env_var} overrides) or in the "
            f"environment as {names}. Provision the file with:\n"
            f"  {provision_hint}"
        )


def _preflight_sdk(module: str, install_hint: str) -> None:
    """Preflight the lazily-imported provider SDK. Without this, a missing
    install only surfaces inside each episode's sandbox start, where the
    failed sample is aborted, the group dropped, and the rollout loop refills
    forever — a silent GPU-burning churn instead of a launch-time error."""
    try:
        importlib.import_module(module)
    except ImportError as e:
        raise RuntimeError(
            f"this sandbox mode needs the {module} SDK in the rollout process's environment: {install_hint}"
        ) from e
