import os

import pytest

from tests.fast.fixtures.generation_fixtures import generation_env
from tests.fast.fixtures.rollout_fixtures import rollout_env

_ = rollout_env, generation_env


@pytest.fixture(autouse=True)
def enable_experimental_rollout_refactor():
    os.environ["MILES_EXPERIMENTAL_ROLLOUT_REFACTOR"] = "1"
    yield
    os.environ.pop("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", None)
