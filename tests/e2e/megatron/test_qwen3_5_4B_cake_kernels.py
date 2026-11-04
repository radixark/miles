"""E2E: Qwen3.5-4B full-parameter GRPO with every Cake kernel family the model can run.

The recipe of ``test_qwen3_5_4B_gdn_cake.py`` (TP 2 with sequence parallelism, 4 GPUs, colocated) with the Cake
backends of four families of the miles-ext-kernels submodule on together:

- ``--linear-attention-backend cake``: the chunked GDN kernels;
- ``--linear-attention-conv-backend cake``: the GDN short conv;
- ``--linear-attention-output-norm-backend cake``: the SiLU-gated output RMSNorm;
- ``--full-attention-backend cake``: the packed causal GQA core attention of the full-attention layers.

``MILES_GDN_CAKE_VARIANT`` (``det`` by default, or ``fast``) pins the variant of every family that has one. The GQA
kernels sum dQ with atomics, so ``replay`` runs of this test are not bit-identical; the GDN test checks that.

Not here: ``--linear-attention-input-norm cake`` (the Cake RMSNorm does not serve Qwen3.5-4B's hidden size 2560), and
the families of other models. The other ``MILES_E2E_*`` knobs and the paths are those of the GDN test. Needs
Blackwell (SM100a / SM103a) GPUs and the checked-out submodule.
"""

import os

from tests.ci.ci_register import register_cuda_ci
from tests.e2e.megatron import test_qwen3_5_4B_gdn_cake as gdn_e2e

register_cuda_ci(
    est_time=2400,
    suite="stage-c-4-gpu-b200",
    labels=["megatron"],
    hardware=["blackwell"],
    num_gpus=4,
)

CAKE_ARGS = (
    f"--linear-attention-conv-backend cake --cake-conv1d-variant {gdn_e2e.VARIANT} "
    f"--linear-attention-output-norm-backend cake --cake-gated-rmsnorm-variant {gdn_e2e.VARIANT} "
    "--full-attention-backend cake "
)


if __name__ == "__main__":
    if gdn_e2e.BACKEND != "cake":
        raise ValueError(f"this test runs the cake backends; unset MILES_GDN_BACKEND (got {gdn_e2e.BACKEND!r})")
    stage = os.environ.get("MILES_E2E_STAGE", "all")
    if stage in ("prepare", "all"):
        gdn_e2e.prepare()
    if stage in ("train", "all"):
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(proxy_var, None)
        gdn_e2e.execute(CAKE_ARGS, test_file=__file__, run_name="cake-kernels")
