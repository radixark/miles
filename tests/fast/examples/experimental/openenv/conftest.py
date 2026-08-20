"""Put the openenv example's modules on sys.path for this package's tests.

A rollout loads them by path (``--custom-agent-function-path``), never as a
package, so they import each other by bare name (``import tb2_sandbox_recipe as
recipe``) and the tests must reach them the same way. One insert here replaces
the copy each test file used to carry next to its own imports.
"""

import sys

from . import EXAMPLE_DIR

if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))
