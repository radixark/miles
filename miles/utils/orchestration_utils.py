from argparse import Namespace

from ray.actor import ActorHandle

from miles.ray.wiring import launch_worker_manager
from miles.utils import object_store
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.logging_utils import configure_logger
from miles.utils.tracking_utils.tracking import init_tracking


def init_orchestration_script(args: Namespace) -> ActorHandle | None:
    configure_logger(args, source=SimpleProcessIdentity(component="main"))
    maybe_start_periodic_pyspy_dump()
    init_tracking(args)
    worker_manager = launch_worker_manager(args)
    object_store.init_instance(args, contribute_segment=False)
    return worker_manager
