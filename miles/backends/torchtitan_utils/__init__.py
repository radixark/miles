from miles.backends.torchtitan_utils import compat

compat.install()

from miles.backends.torchtitan_utils.actor import TorchtitanTrainRayActor  # noqa: E402

__all__ = ["TorchtitanTrainRayActor"]
