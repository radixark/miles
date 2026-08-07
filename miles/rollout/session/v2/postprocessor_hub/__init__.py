"""Post-process hooks for the session samples op — this is the customization point.

Point ``--session-sample-postprocessor-path`` at any ``fn(leaf_samples,
session_metadata) -> list[Sample]``. The post-process hook finalizes the
surviving set into training samples; it runs strictly after pick,
synchronously inside the session server process. It never drops samples —
selection is the picker's job (see ``picker_hub``).

One module per hook, file named after the function. Custom post-processor
ideas: reassign shared-node mask ownership by reward, broadcast advantages
across branches.
"""

from miles.rollout.session.v2.postprocessor_hub.default_postprocess import default_postprocess

__all__ = ["default_postprocess"]
