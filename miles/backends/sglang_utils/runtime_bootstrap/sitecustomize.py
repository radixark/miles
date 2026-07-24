"""Bootstrap Miles runtime patches in freshly spawned SGLang workers."""

from __future__ import annotations

import os

if os.environ.get("MILES_DSV4_TOP_RUNTIME_PATCHES", "0") == "1":
    from miles.backends.sglang_utils.dsv4_top_patches import (
        apply_dsv4_top_sglang_patches,
    )

    apply_dsv4_top_sglang_patches()
