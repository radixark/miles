import logging
import os
import time
from copy import deepcopy

import wandb

from miles.utils.env_report import decode_env_report

logger = logging.getLogger(__name__)


# Cross-region clusters (e.g. MBZUAI Abu Dhabi ↔ wandb cloud in US-West) can
# exceed wandb's 90s default init_timeout during first-attach handshakes in
# shared mode, especially when many Ray actors initialize concurrently. Raise
# the ceiling on both primary and secondary init paths. Matches the wandb
# docs guidance for high-latency environments.
WANDB_INIT_TIMEOUT_SECS: float = float(os.environ.get("WANDB_INIT_TIMEOUT_SECS", "300"))

# Retry wandb.init on transient CommError/UsageError to survive short
# wandb-cloud flakiness during boot. Exponential backoff keeps the retry
# budget bounded while giving enough headroom for a 5-minute blip.
WANDB_INIT_RETRY_ATTEMPTS: int = int(os.environ.get("WANDB_INIT_RETRY_ATTEMPTS", "3"))
WANDB_INIT_RETRY_BACKOFF_SECS: float = float(os.environ.get("WANDB_INIT_RETRY_BACKOFF_SECS", "5"))


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def _wandb_init_with_retry(init_kwargs: dict, *, role: str):
    """Call wandb.init(**init_kwargs) with exponential-backoff retry on
    transient wandb-cloud failures.

    Shared-mode init in online mode makes several HTTPS round-trips to wandb
    cloud. A single transient CommError during boot (cross-region packet
    loss, rate-limit burst, DNS hiccup) would otherwise abort the whole run.
    We retry a bounded number of times with exponential backoff; fall
    through to raise on terminal failure so the caller surfaces it.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, WANDB_INIT_RETRY_ATTEMPTS + 1):
        try:
            return wandb.init(**init_kwargs)
        except wandb.errors.CommError as exc:  # type: ignore[attr-defined]
            last_exc = exc
        except wandb.errors.UsageError as exc:  # type: ignore[attr-defined]
            last_exc = exc
        except Exception as exc:  # unexpected; re-raise immediately
            logger.error("wandb.init (%s) failed with non-retryable %s: %s", role, type(exc).__name__, exc)
            raise
        wait = WANDB_INIT_RETRY_BACKOFF_SECS * (2 ** (attempt - 1))
        logger.warning(
            "wandb.init (%s) attempt %d/%d failed: %s. Retrying in %.1fs.",
            role, attempt, WANDB_INIT_RETRY_ATTEMPTS, last_exc, wait,
        )
        time.sleep(wait)
    logger.error("wandb.init (%s) exhausted %d retries; giving up", role, WANDB_INIT_RETRY_ATTEMPTS)
    assert last_exc is not None
    raise last_exc


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        # init_timeout default (90s) is too tight for cross-region clusters.
        # x_label tags the primary so secondary attachment can be audited in
        # the wandb UI's console-logs filter.
        init_kwargs["settings"] = wandb.Settings(
            mode="shared",
            x_primary=True,
            init_timeout=WANDB_INIT_TIMEOUT_SECS,
            x_label=f"rank_{getattr(args, 'rank', 0)}_primary",
        )

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    _wandb_init_with_retry(init_kwargs, role="primary")

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    if env_report_raw := args.env_report:
        if launcher_report := decode_env_report(env_report_raw):
            output["launcher_env_report"] = launcher_report

    return output


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args, router_addr=None):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        # Same init_timeout treatment as primary; cross-region latency +
        # concurrent actor bursts routinely exceed the 90s default for the
        # secondary's run-attach HTTPS round-trip.
        # x_label per-rank tagging is the standard per wandb distributed-
        # training docs (unique label per writer).
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
            init_timeout=WANDB_INIT_TIMEOUT_SECS,
            x_label=f"rank_{getattr(args, 'rank', 0)}_secondary",
        )

    if args.sglang_enable_metrics and router_addr is not None:
        logger.info(f"Forward SGLang metrics at {router_addr} to WandB.")
        settings_kwargs |= dict(
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": args.__dict__,
        "resume": "allow",
        # reinit=True removed: shared-mode semantics permit concurrent writers
        # to the same run natively. Setting reinit=True triggered stale-state
        # warnings from the wandb-core on secondary actors, without any
        # functional benefit once the primary owns the run lifecycle.
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    _wandb_init_with_retry(init_kwargs, role="secondary")

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
