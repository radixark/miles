from argparse import Namespace
from functools import partial

from ray.actor import ActorHandle

from miles.ray.wiring import launch_worker_manager, shutdown_worker_manager
from miles.utils import object_store
from miles.utils.async_utils import Disposer
from miles.utils.audit_utils.event_logger import checkpoint as event_logger_checkpoint
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.logging_utils import configure_logger
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking


def init_orchestration_script(args: Namespace, *, disposer: Disposer) -> ActorHandle | None:
    event_logger_checkpoint.restore(args)
    configure_logger(args, source=SimpleProcessIdentity(component="main"))
    maybe_start_periodic_pyspy_dump()
    init_tracking(args)
    disposer.add(finish_tracking)
    worker_manager = launch_worker_manager(args)
    disposer.add(partial(shutdown_worker_manager, worker_manager))
    object_store.init_instance(args, contribute_segment=False)
    return worker_manager
