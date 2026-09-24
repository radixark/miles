"""Load Workplace modules the same way as the rollout plugin loader."""

import sys

from tests.fast.examples.experimental.nemo_gym_workspace_assistant import EXAMPLE_DIR

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
