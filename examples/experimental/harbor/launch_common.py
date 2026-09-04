"""The Harbor-side launcher wiring, shared by this example's launchers.

Separate from run.py so it imports without the trainer's heavy dependencies:
the launchers pull in miles' command utils, while this module needs only the
credential contract.
"""

import os
from pathlib import Path

from miles.rollout.agentic.credentials import PROVIDER_CREDENTIALS, provision_provider

HARBOR_EXAMPLE_DIR = Path(__file__).resolve().parent
# reward_func / RolloutFn (agent-metric aggregation) are reused from the agent-server example
HARBOR_DOCKER_EXAMPLE_DIR = HARBOR_EXAMPLE_DIR.parents[1] / "swe-agent-harbor-docker"


def agentic_pythonpath_dirs() -> list[str]:
    """Directories every Harbor launcher puts on the workers' PYTHONPATH: the
    agent function here, and the reward hook it reuses from the agent-server
    example."""
    return [str(HARBOR_EXAMPLE_DIR), str(HARBOR_DOCKER_EXAMPLE_DIR)]


def agentic_train_args(*, tito_model: str, session_server_workers: int) -> str:
    """The agentic wiring every Harbor launcher passes to train.py.

    One copy, shared by the recipes and the GPU e2e, so the flags the test
    exercises are the flags the recipes ship. Model-specific values come from
    the caller.
    """
    return (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path harbor_agent_function.run "
        "--custom-rm-path generate.reward_func "
        "--rollout-function-path generate.RolloutFn "
        "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted "
        f"--tito-model {tito_model} "
        "--use-session-server "
        "--session-server-port 30000 "
        f"--session-server-workers {session_server_workers} "
    )


def harbor_env_vars(args) -> dict[str, str]:
    """The Harbor-side environment for rollout workers, preflighted on the launcher.

    HARBOR_ENV_TYPE is passed through untouched (Harbor validates it). The
    provider's credential goes by key-file PATH, its endpoint variables by
    value, on the contract every sandbox backend shares.
    """
    # normalized the same way the agent function reads it, so the guards
    # below see the value the worker will act on
    env_type = args.harbor_env_type.strip().lower()
    if not env_type:
        raise ValueError(
            "set --harbor-env-type / HARBOR_ENV_TYPE (e.g. e2b, daytona): in-process trials need a sandbox backend the worker can reach"
        )
    if env_type == "docker":
        raise ValueError(
            "docker needs a Docker daemon next to Trial.run(); use examples/swe-agent-harbor-docker (agent server) for it"
        )
    try:
        import harbor  # noqa: F401  # the fork branch, see README
    except ImportError as e:
        raise RuntimeError(
            "harbor is not importable in the rollout process's environment; see the README install line"
        ) from e

    env = {
        "HARBOR_ENV_TYPE": env_type,
        "HARBOR_TASKS_DIR": args.harbor_tasks_dir,
        "HARBOR_TRIALS_DIR": args.harbor_trials_dir,
        "AGENT_MODEL_NAME": args.agent_model_name,
        "AGENT_TIMEOUT": str(args.agent_timeout),
        "MILES_ROUTER_EXTERNAL_HOST": args.router_external_host,
    }
    if args.harbor_env_kwargs:
        env["HARBOR_ENV_KWARGS"] = args.harbor_env_kwargs
    if spec := PROVIDER_CREDENTIALS.get(env_type):
        provision_provider(env, spec, arg_path=getattr(args, spec["arg_attr"], "") or "")
    else:
        # Any other Harbor backend still passes straight through; there is just
        # no credential wiring known here, so the worker environment must carry
        # whatever that provider's SDK reads.
        print(
            f"harbor: no credential wiring known for {env_type!r}; "
            "assuming the worker environment carries the provider's credentials",
            flush=True,
        )
    # per-server knobs, forwarded when set
    for var in (
        "AGENT_MAX_INPUT_TOKENS",
        "AGENT_MAX_OUTPUT_TOKENS",
        "AGENT_TRIAL_TIMEOUT",
        "HARBOR_MAX_SEQ_LEN",
        "HARBOR_AGENT_MAX_ITERATIONS",
        "HARBOR_RESPONSE_LENGTH_POLICY",
        "HARBOR_TERMINUS_2_ENABLE_SUMMARIZE",
        "HARBOR_TERMINUS_2_LINEAR_HISTORY",
        "HARBOR_OVERRIDE_MEMORY_MB",
        "HARBOR_TIMEOUT_MULTIPLIER",
        "HARBOR_VERIFIER_TIMEOUT_SEC",
        "HARBOR_ENV_BUILD_TIMEOUT_MULTIPLIER",
        "HARBOR_AGENT_ALLOWED_HOSTS",
    ):
        if value := os.environ.get(var, "").strip():
            env[var] = value
    return env
