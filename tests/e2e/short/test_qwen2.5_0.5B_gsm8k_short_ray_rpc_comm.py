import importlib.util
import os
from pathlib import Path
from types import ModuleType

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=450, suite="stage-c-8-gpu-h100", labels=["short", "mooncake", "rpc-comm"])


def _load_base_test() -> ModuleType:
    path = Path(__file__).with_name("test_qwen2.5_0.5B_gsm8k_short.py")
    spec = importlib.util.spec_from_file_location("gsm8k_short_base_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    base = _load_base_test()
    base.prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    base.execute(comm_backend=base.WorkerCommBackend.RPC, test_file=__file__)
