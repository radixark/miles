"""Compatibility imports for the original Qwen-only entrypoint.

Use prefix_tree.transformers for any compatible full-attention causal LM.
"""

from miles.backends.training_utils.prefix_tree.transformers import TreeSFTBatch, enable_transformers_prefix_tree

enable_qwen_prefix_tree = enable_transformers_prefix_tree

__all__ = ["TreeSFTBatch", "enable_qwen_prefix_tree"]
