"""Triton kernels for Qwen3.8-Next (QSA sparse attention, HC, PLE, table gather).

First-call JIT is seconds per family (measured: ~1.9s for HC and for PLE at the
real widths, T is do_not_specialize) and lands in the node-local triton cache;
no warmup machinery, same as every other kernel-bearing model plugin.
"""
