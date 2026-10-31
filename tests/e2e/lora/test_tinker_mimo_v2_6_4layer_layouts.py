"""The MiMo-V2.6 multi-LoRA checks of `test_tinker_mimo_v2_6_4layer.py` on the adapter layouts the gateway does not
default to."""

import asyncio
import os

from scripts.run_mimo_v2_6_flash_tinker import prepare
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.lora.test_tinker_mimo_v2_6_4layer import MODEL_NAME, SERVE_CMD, _check, _run_tenants
from tests.e2e.lora.tinker_gateway import running_gateway

# weekly only: `long` is outside run-ci-lora / run-ci-multi-lora and the nightly scope
register_cuda_ci(est_time=1500, suite="stage-c-8-gpu-h100", labels=["long"], hardware=["hopper"], num_gpus=8)

# (lora_type, experts_shared_outer_loras)
_LAYOUTS = [
    ("canonical_lora", True),
    ("lora", False),
    ("lora", True),
]


if __name__ == "__main__":
    prepare(model_name=MODEL_NAME)
    # The gateway and the Tinker clients talk over loopback.
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    for lora_type, shared_outer in _LAYOUTS:
        name = f"{lora_type}{' + shared-outer' if shared_outer else ''}"
        print(f"[mimo-layouts] ===== layout: {name} =====", flush=True)
        extra_args = f"--lora-type {lora_type}" + (" --experts-shared-outer-loras" if shared_outer else "")
        with running_gateway(f'{SERVE_CMD} --extra-args "{extra_args}"') as base_url:
            _check(asyncio.run(_run_tenants(base_url)))
        print(f"[mimo-layouts] ===== layout PASSED: {name} =====", flush=True)
