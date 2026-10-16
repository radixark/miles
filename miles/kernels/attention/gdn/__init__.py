"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Gated DeltaNet (GDN) chunked training kernels.

``cake/``: the generated deterministic ``chunk_gated_delta_rule`` for Blackwell (SM100a / SM103a), selected by
``--linear-attention-backend cake`` (see :func:`miles_plugins.models.linear_attn.gdn_kernel`).
"""
