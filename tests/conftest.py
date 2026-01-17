import os

import pytest

from tests.fixtures.generation_fixtures import generation_env
from tests.fixtures.rollout_integration import rollout_integration_env

_ = rollout_integration_env, generation_env


@pytest.fixture(autouse=True)
def enable_experimental_rollout_refactor():
    os.environ["MILES_EXPERIMENTAL_ROLLOUT_REFACTOR"] = "1"
    yield
    os.environ.pop("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", None)
