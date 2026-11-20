import functools
import os

if os.environ.get("KIMI_EVAL_SYSTEM_DNS") == "1":
    try:
        import pyqwest
    except ImportError:
        pass
    else:
        pyqwest.HTTPTransport = functools.partial(pyqwest.HTTPTransport, use_system_dns=True)
        pyqwest.SyncHTTPTransport = functools.partial(pyqwest.SyncHTTPTransport, use_system_dns=True)
