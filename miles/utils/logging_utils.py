import logging
import warnings

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    configure_raise_unawaited_coroutine()


def configure_raise_unawaited_coroutine() -> None:
    """Turn 'coroutine was never awaited' warnings into errors.

    Python emits RuntimeWarning when a coroutine is called but never awaited.
    By default this is easy to miss. This makes it crash immediately.
    """
    warnings.filterwarnings("error", category=RuntimeWarning, message="coroutine .* was never awaited")
