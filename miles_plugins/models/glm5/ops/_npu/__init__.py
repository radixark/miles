# Copyright (c) Huawei Technologies Co., Ltd. 2026.
#
# NPU (Ascend A3 / mlir-ascend) drop-in dispatch for miles' tilelang ops.
#
# Re-exports the 4 `*_interface` functions of glm5 ops, but routed to NPU
# kernels in this subpackage. Auto-imported when miles' interface modules
# detect `torch.is_npu` / `torch_npu` and substitute themselves.
#
# Adaptation notes:
#   * Our kernels skip the `cu_seqlen_*` causal mask (varlen handling) and
#     instead expect the wrapper to apply masking on the output.
#   * R-KA-13 / R-KA-14 / R-KA-15 OPEN bugs are worked around at this level:
#       - R-KA-13: vsub schedule-locality fix is baked into the bwd kernel
#       - R-KA-14: indexer_bwd is invoked per-seq-position (SEQ=1 grid)
#       - R-KA-15: indexer_bwd short-circuits when grad_scores.abs().max() < 1e-30
from . import indexer as indexer  # noqa: F401
from . import sparse_mla as sparse_mla  # noqa: F401
