import logging

import torch

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        # Keep DeepEP's buffers out of the memory saver's region for the duration of __init__, then put the flag
        # back to whatever it was. Restoring True unconditionally clobbered a caller that had the region off (#807).
        # torch_memory_saver.disable() is not usable here: it asserts the region is on when entered.
        cdll = torch_memory_saver._impl._binary_wrapper.cdll if torch_memory_saver._impl is not None else None
        was_interesting = cdll.tms_get_interesting_region() if cdll is not None else None
        if cdll is not None:
            cdll.tms_set_interesting_region(False)
        try:
            old_init(self, *args, **kwargs)
            torch.cuda.synchronize()
        finally:
            if cdll is not None:
                cdll.tms_set_interesting_region(was_interesting)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    import miles_plugins.megatron_bridge  # noqa: F401
except Exception as _e:  # best-effort; not every environment uses megatron.bridge
    logging.warning("miles megatron.bridge plugins failed to load: %s", _e)

logging.getLogger("megatron").setLevel(logging.WARNING)
