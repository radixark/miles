#!/usr/bin/env python3
"""Run a miles test with aiter shuffle_scale monkey-patch for older containers."""

import sys

# Monkey-patch aiter.ops.shuffle before anything imports from sglang
import aiter.ops.shuffle as _shuffle

if not hasattr(_shuffle, "shuffle_scale"):
    import functools

    @functools.wraps(_shuffle.shuffle_scale_a16w4)
    def shuffle_scale(src, experts_cnt=None, is_guinterleave=False, gate_up=False):
        """Compatibility shim: older aiter has shuffle_scale_a16w4 only."""
        return _shuffle.shuffle_scale_a16w4(src, experts_cnt, gate_up)

    _shuffle.shuffle_scale = shuffle_scale
    print("[patch] Added shuffle_scale compatibility shim to aiter.ops.shuffle", flush=True)

# Execute the target test file
test_file = sys.argv[1]
sys.argv = [test_file]
exec(open(test_file).read())
