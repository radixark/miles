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

    configure_strict_async_warnings()


def configure_strict_async_warnings() -> None:
    """Turn common async misuse warnings into errors.

    Catches two silent-failure patterns:
    - Coroutine called but never awaited (silently dropped)
    - asyncio.Task created but reference lost (silently GC'd and cancelled)
    """
    warnings.filterwarnings("error", category=RuntimeWarning, message="coroutine .* was never awaited")
    warnings.filterwarnings("error", category=RuntimeWarning, message=".*Task.*was destroyed but it is pending")
