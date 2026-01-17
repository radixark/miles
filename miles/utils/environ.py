import os


def get_experimental_rollout_refactor() -> bool:
    return bool(int(os.environ.get("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", "0")))
