import os

TIMEOUT_SCALE_ENV = "MILES_TEST_TIMEOUT_SCALE"


def scaled_timeout(seconds: float) -> float:
    return seconds * float(os.environ.get(TIMEOUT_SCALE_ENV, "1"))
