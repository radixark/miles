from .tracking import TrackingManager

_manager = TrackingManager()


def init_tracking(args, primary: bool = True, **kwargs):
    _manager.init(args, primary=primary, **kwargs)


def log(args, metrics, step_key: str):
    step = metrics.get(step_key)
    _manager.log(metrics, step=step)


def finish_tracking():
    _manager.finish()
