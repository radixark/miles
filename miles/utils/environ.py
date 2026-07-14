import os

_printed_experimental_rollout_refactor = False


def enable_experimental_rollout_refactor() -> bool:
    result = bool(int(os.environ.get("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "0")))

    global _printed_experimental_rollout_refactor
    if result and not _printed_experimental_rollout_refactor:
        print("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 is enabled (experimental feature)")
        _printed_experimental_rollout_refactor = True

    return result


_printed_experimental_ft_trainer = False


def enable_experimental_ft_trainer() -> bool:
    raw = os.environ.get("MILES_EXPERIMENTAL_FT_TRAINER", "0").lower()
    result = raw in ("1", "true", "on", "yes")

    global _printed_experimental_ft_trainer
    if result and not _printed_experimental_ft_trainer:
        print("MILES_EXPERIMENTAL_FT_TRAINER=1 is enabled (experimental feature)")
        _printed_experimental_ft_trainer = True

    return result
