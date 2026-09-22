"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Native loading for the generated chunked KDA training-backward kernels (SM100a / SM103a).

The registry is filled mechanically from the Cake compiler's resolved production builds:
one device/launcher source pair per backward stage and architecture under ``csrc/<arch>/``.
Each stage builds on first use as a torch CUDA extension (``nvcc`` + ``ninja``; cached under
``TORCH_EXTENSIONS_DIR``). Launchers encode TMA descriptors on the host on every call into a
caller-owned device workspace, so tensor addresses may change between calls.
"""

import os
from functools import cache
from pathlib import Path

import torch

_CSRC = Path(__file__).resolve().parent / "csrc"
_ARCH_FLAGS = {
    "sm_100a": "-gencode=arch=compute_100a,code=sm_100a",
    "sm_103a": "-gencode=arch=compute_103a,code=sm_103a",
}
SUPPORTED_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
# torch's extension build disables the half / bf16 conversion operators; the generated device code uses them.
_CONVERSION_FLAGS = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
)

# MODULES[stage][arch] -> generated module record (mechanically filled).
MODULES = {
    "prep": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_ff540e4f27805de6a473",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_ff540e4f27805de6a473_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_ff540e4f27805de6a473",
            "arg_plan": (
                ("buffer", "g_raw"),
                ("buffer", "q_norm"),
                ("buffer", "k_norm"),
                ("buffer", "v"),
                ("buffer", "beta"),
                ("buffer", "A_log"),
                ("buffer", "dt_bias"),
                ("buffer", "aqk"),
                ("buffer", "akk"),
                ("buffer", "do"),
                ("buffer", "chunk_bos"),
                ("buffer", "chunk_len"),
                ("buffer", "gk_out"),
                ("buffer", "vb_out"),
                ("buffer", "kb_out"),
                ("buffer", "qg_out"),
                ("buffer", "kg_out"),
                ("buffer", "ke_out"),
                ("buffer", "qe_out"),
                ("buffer", "aqk_tril"),
                ("buffer", "akk_int"),
                ("buffer", "do_int"),
                ("buffer", "v_int"),
                ("buffer", "beta_int"),
                ("parameter", "num_qk_heads"),
                ("parameter", "num_heads"),
                ("parameter", "group"),
                ("parameter", "copy_inputs"),
                ("parameter", "lower_bound"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_100a/cake_kda_chunk_train_ff540e4f27805de6a473_kernel.cu",
                "sm_100a/cake_kda_chunk_train_ff540e4f27805de6a473_binding.cu",
            ),
            "block": (1024, 1, 1),
            "dynamic_smem_bytes": 4096,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_7e4f8f2c7082ace9d490",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_7e4f8f2c7082ace9d490_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_7e4f8f2c7082ace9d490",
            "arg_plan": (
                ("buffer", "g_raw"),
                ("buffer", "q_norm"),
                ("buffer", "k_norm"),
                ("buffer", "v"),
                ("buffer", "beta"),
                ("buffer", "A_log"),
                ("buffer", "dt_bias"),
                ("buffer", "aqk"),
                ("buffer", "akk"),
                ("buffer", "do"),
                ("buffer", "chunk_bos"),
                ("buffer", "chunk_len"),
                ("buffer", "gk_out"),
                ("buffer", "vb_out"),
                ("buffer", "kb_out"),
                ("buffer", "qg_out"),
                ("buffer", "kg_out"),
                ("buffer", "ke_out"),
                ("buffer", "qe_out"),
                ("buffer", "aqk_tril"),
                ("buffer", "akk_int"),
                ("buffer", "do_int"),
                ("buffer", "v_int"),
                ("buffer", "beta_int"),
                ("parameter", "num_qk_heads"),
                ("parameter", "num_heads"),
                ("parameter", "group"),
                ("parameter", "copy_inputs"),
                ("parameter", "lower_bound"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_103a/cake_kda_chunk_train_7e4f8f2c7082ace9d490_kernel.cu",
                "sm_103a/cake_kda_chunk_train_7e4f8f2c7082ace9d490_binding.cu",
            ),
            "block": (1024, 1, 1),
            "dynamic_smem_bytes": 4096,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "wy": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_3b51f3e4ba289b50b26b",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_3b51f3e4ba289b50b26b_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_3b51f3e4ba289b50b26b",
            "arg_plan": (
                ("tma_buffer", "akk_tma"),
                ("tma_buffer", "vb_tma"),
                ("tma_buffer", "kb_tma"),
                ("buffer", "u_out"),
                ("buffer", "w_out"),
                ("parameter", "num_heads"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_100a/cake_kda_chunk_train_3b51f3e4ba289b50b26b_kernel.cu",
                "sm_100a/cake_kda_chunk_train_3b51f3e4ba289b50b26b_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "tma_workspace_bytes": 384,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_10b6ed8813cfdf8b5f58",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_10b6ed8813cfdf8b5f58_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_10b6ed8813cfdf8b5f58",
            "arg_plan": (
                ("tma_buffer", "akk_tma"),
                ("tma_buffer", "vb_tma"),
                ("tma_buffer", "kb_tma"),
                ("buffer", "u_out"),
                ("buffer", "w_out"),
                ("parameter", "num_heads"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_103a/cake_kda_chunk_train_10b6ed8813cfdf8b5f58_kernel.cu",
                "sm_103a/cake_kda_chunk_train_10b6ed8813cfdf8b5f58_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "tma_workspace_bytes": 384,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "fwdh": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_c8390113a9dc4a276918",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_c8390113a9dc4a276918_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_c8390113a9dc4a276918",
            "arg_plan": (
                ("tma_buffer", "w_tma"),
                ("tma_buffer", "kg_tma"),
                ("buffer", "u"),
                ("buffer", "gk"),
                ("buffer", "h_out"),
                ("buffer", "v_new"),
                ("parameter", "num_heads"),
                ("buffer", "seq_chunk_start"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_100a/cake_kda_chunk_train_c8390113a9dc4a276918_kernel.cu",
                "sm_100a/cake_kda_chunk_train_c8390113a9dc4a276918_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 91136,
            "tma_workspace_bytes": 256,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_d312d299b61305de0ee9",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_d312d299b61305de0ee9_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_d312d299b61305de0ee9",
            "arg_plan": (
                ("tma_buffer", "w_tma"),
                ("tma_buffer", "kg_tma"),
                ("buffer", "u"),
                ("buffer", "gk"),
                ("buffer", "h_out"),
                ("buffer", "v_new"),
                ("parameter", "num_heads"),
                ("buffer", "seq_chunk_start"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_103a/cake_kda_chunk_train_d312d299b61305de0ee9_kernel.cu",
                "sm_103a/cake_kda_chunk_train_d312d299b61305de0ee9_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 91136,
            "tma_workspace_bytes": 256,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "dav": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_6ebf680f71eddf9e04c5",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_6ebf680f71eddf9e04c5_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_6ebf680f71eddf9e04c5",
            "arg_plan": (
                ("tma_buffer", "do_tma"),
                ("tma_buffer", "vnew_tma"),
                ("tma_buffer", "aqk_tma"),
                ("buffer", "dAqk"),
                ("buffer", "dv1"),
                ("parameter", "num_heads"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_100a/cake_kda_chunk_train_6ebf680f71eddf9e04c5_kernel.cu",
                "sm_100a/cake_kda_chunk_train_6ebf680f71eddf9e04c5_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "tma_workspace_bytes": 384,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_bf3c4ed073ca4bcdb25f",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_bf3c4ed073ca4bcdb25f_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_bf3c4ed073ca4bcdb25f",
            "arg_plan": (
                ("tma_buffer", "do_tma"),
                ("tma_buffer", "vnew_tma"),
                ("tma_buffer", "aqk_tma"),
                ("buffer", "dAqk"),
                ("buffer", "dv1"),
                ("parameter", "num_heads"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_103a/cake_kda_chunk_train_bf3c4ed073ca4bcdb25f_kernel.cu",
                "sm_103a/cake_kda_chunk_train_bf3c4ed073ca4bcdb25f_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "tma_workspace_bytes": 384,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "dhu": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_895185f20c4f0c47d54e",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_895185f20c4f0c47d54e_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_895185f20c4f0c47d54e",
            "arg_plan": (
                ("tma_buffer", "kg_tma"),
                ("tma_buffer", "qg_tma"),
                ("tma_buffer", "w_tma"),
                ("tma_buffer", "do_tma"),
                ("buffer", "dv1"),
                ("buffer", "gk"),
                ("buffer", "dh_out"),
                ("buffer", "dv2"),
                ("parameter", "num_heads"),
                ("buffer", "seq_chunk_start"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_100a/cake_kda_chunk_train_895185f20c4f0c47d54e_kernel.cu",
                "sm_100a/cake_kda_chunk_train_895185f20c4f0c47d54e_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 140288,
            "tma_workspace_bytes": 512,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_40585b0ca6924dc66dfc",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_40585b0ca6924dc66dfc_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_40585b0ca6924dc66dfc",
            "arg_plan": (
                ("tma_buffer", "kg_tma"),
                ("tma_buffer", "qg_tma"),
                ("tma_buffer", "w_tma"),
                ("tma_buffer", "do_tma"),
                ("buffer", "dv1"),
                ("buffer", "gk"),
                ("buffer", "dh_out"),
                ("buffer", "dv2"),
                ("parameter", "num_heads"),
                ("buffer", "seq_chunk_start"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_103a/cake_kda_chunk_train_40585b0ca6924dc66dfc_kernel.cu",
                "sm_103a/cake_kda_chunk_train_40585b0ca6924dc66dfc_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 140288,
            "tma_workspace_bytes": 512,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "dqkg": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_a0c5df001cdd3ee221b8",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_a0c5df001cdd3ee221b8_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_a0c5df001cdd3ee221b8",
            "arg_plan": (
                ("tma_buffer", "do_tma"),
                ("tma_buffer", "vn_tma"),
                ("tma_buffer", "dv2_tma"),
                ("tma_buffer", "v_tma"),
                ("tma_buffer", "h_tma"),
                ("tma_buffer", "dh_tma"),
                ("tma_buffer", "akk_tma"),
                ("buffer", "h_ptr"),
                ("buffer", "dh_ptr"),
                ("buffer", "k_ptr"),
                ("buffer", "q_ptr"),
                ("buffer", "v_ptr"),
                ("buffer", "gk"),
                ("buffer", "beta"),
                ("buffer", "dq_out"),
                ("buffer", "dk_out"),
                ("buffer", "dg_out"),
                ("buffer", "db_out"),
                ("buffer", "dAkk_out"),
                ("buffer", "dv_out"),
                ("parameter", "num_heads"),
                ("buffer", "chunk_bos"),
                ("buffer", "chunk_len"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_100a/cake_kda_chunk_train_a0c5df001cdd3ee221b8_kernel.cu",
                "sm_100a/cake_kda_chunk_train_a0c5df001cdd3ee221b8_binding.cu",
            ),
            "block": (320, 1, 1),
            "dynamic_smem_bytes": 174080,
            "tma_workspace_bytes": 896,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_cf73aef6c0cf9990f7d3",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_cf73aef6c0cf9990f7d3_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_cf73aef6c0cf9990f7d3",
            "arg_plan": (
                ("tma_buffer", "do_tma"),
                ("tma_buffer", "vn_tma"),
                ("tma_buffer", "dv2_tma"),
                ("tma_buffer", "v_tma"),
                ("tma_buffer", "h_tma"),
                ("tma_buffer", "dh_tma"),
                ("tma_buffer", "akk_tma"),
                ("buffer", "h_ptr"),
                ("buffer", "dh_ptr"),
                ("buffer", "k_ptr"),
                ("buffer", "q_ptr"),
                ("buffer", "v_ptr"),
                ("buffer", "gk"),
                ("buffer", "beta"),
                ("buffer", "dq_out"),
                ("buffer", "dk_out"),
                ("buffer", "dg_out"),
                ("buffer", "db_out"),
                ("buffer", "dAkk_out"),
                ("buffer", "dv_out"),
                ("parameter", "num_heads"),
                ("buffer", "chunk_bos"),
                ("buffer", "chunk_len"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_103a/cake_kda_chunk_train_cf73aef6c0cf9990f7d3_kernel.cu",
                "sm_103a/cake_kda_chunk_train_cf73aef6c0cf9990f7d3_binding.cu",
            ),
            "block": (320, 1, 1),
            "dynamic_smem_bytes": 174080,
            "tma_workspace_bytes": 896,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "intra": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_e2d6d8171e14c1b6f265",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_e2d6d8171e14c1b6f265_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_e2d6d8171e14c1b6f265",
            "arg_plan": (
                ("buffer", "dAqk"),
                ("buffer", "dAkk"),
                ("buffer", "gk"),
                ("buffer", "k_e"),
                ("buffer", "q_e"),
                ("buffer", "beta"),
                ("buffer", "dq_f"),
                ("buffer", "dk_f"),
                ("buffer", "dg_f"),
                ("buffer", "db_f"),
                ("buffer", "dq_out"),
                ("buffer", "dk_out"),
                ("buffer", "dg_out"),
                ("buffer", "db_out"),
                ("parameter", "num_heads"),
                ("parameter", "num_qk_heads"),
                ("parameter", "group"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_100a/cake_kda_chunk_train_e2d6d8171e14c1b6f265_kernel.cu",
                "sm_100a/cake_kda_chunk_train_e2d6d8171e14c1b6f265_binding.cu",
            ),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 103680,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_528f64208e08f0916d5c",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_528f64208e08f0916d5c_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_528f64208e08f0916d5c",
            "arg_plan": (
                ("buffer", "dAqk"),
                ("buffer", "dAkk"),
                ("buffer", "gk"),
                ("buffer", "k_e"),
                ("buffer", "q_e"),
                ("buffer", "beta"),
                ("buffer", "dq_f"),
                ("buffer", "dk_f"),
                ("buffer", "dg_f"),
                ("buffer", "db_f"),
                ("buffer", "dq_out"),
                ("buffer", "dk_out"),
                ("buffer", "dg_out"),
                ("buffer", "db_out"),
                ("parameter", "num_heads"),
                ("parameter", "num_qk_heads"),
                ("parameter", "group"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_103a/cake_kda_chunk_train_528f64208e08f0916d5c_kernel.cu",
                "sm_103a/cake_kda_chunk_train_528f64208e08f0916d5c_binding.cu",
            ),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 103680,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "gate_epilogue": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_a0fd9d08ef18560bcb1d",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_a0fd9d08ef18560bcb1d_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_a0fd9d08ef18560bcb1d",
            "arg_plan": (
                ("buffer", "dg_intra"),
                ("buffer", "g_raw"),
                ("buffer", "db_total"),
                ("buffer", "beta_raw"),
                ("buffer", "A_log"),
                ("buffer", "dt_bias"),
                ("buffer", "chunk_bos"),
                ("buffer", "chunk_len"),
                ("buffer", "dg_out"),
                ("buffer", "dbeta"),
                ("buffer", "dA_part"),
                ("buffer", "dbias_part"),
                ("parameter", "num_heads"),
                ("parameter", "lower_bound"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_100a/cake_kda_chunk_train_a0fd9d08ef18560bcb1d_kernel.cu",
                "sm_100a/cake_kda_chunk_train_a0fd9d08ef18560bcb1d_binding.cu",
            ),
            "block": (1024, 1, 1),
            "dynamic_smem_bytes": 12288,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_7746fbe47fa5f1b86d36",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_7746fbe47fa5f1b86d36_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_7746fbe47fa5f1b86d36",
            "arg_plan": (
                ("buffer", "dg_intra"),
                ("buffer", "g_raw"),
                ("buffer", "db_total"),
                ("buffer", "beta_raw"),
                ("buffer", "A_log"),
                ("buffer", "dt_bias"),
                ("buffer", "chunk_bos"),
                ("buffer", "chunk_len"),
                ("buffer", "dg_out"),
                ("buffer", "dbeta"),
                ("buffer", "dA_part"),
                ("buffer", "dbias_part"),
                ("parameter", "num_heads"),
                ("parameter", "lower_bound"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_103a/cake_kda_chunk_train_7746fbe47fa5f1b86d36_kernel.cu",
                "sm_103a/cake_kda_chunk_train_7746fbe47fa5f1b86d36_binding.cu",
            ),
            "block": (1024, 1, 1),
            "dynamic_smem_bytes": 12288,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "qk_epilogue": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_c3d9a1fa3ac24245d6d4",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_c3d9a1fa3ac24245d6d4_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_c3d9a1fa3ac24245d6d4",
            "arg_plan": (
                ("buffer", "dq_intra"),
                ("buffer", "dk_intra"),
                ("buffer", "q_norm"),
                ("buffer", "k_norm"),
                ("buffer", "q_rstd"),
                ("buffer", "k_rstd"),
                ("buffer", "chunk_bos"),
                ("buffer", "chunk_len"),
                ("buffer", "dq_out"),
                ("buffer", "dk_out"),
                ("parameter", "num_qk_heads"),
                ("parameter", "num_v_heads"),
                ("parameter", "group"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_100a/cake_kda_chunk_train_c3d9a1fa3ac24245d6d4_kernel.cu",
                "sm_100a/cake_kda_chunk_train_c3d9a1fa3ac24245d6d4_binding.cu",
            ),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 0,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_44978dc5595f5354f436",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_44978dc5595f5354f436_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_44978dc5595f5354f436",
            "arg_plan": (
                ("buffer", "dq_intra"),
                ("buffer", "dk_intra"),
                ("buffer", "q_norm"),
                ("buffer", "k_norm"),
                ("buffer", "q_rstd"),
                ("buffer", "k_rstd"),
                ("buffer", "chunk_bos"),
                ("buffer", "chunk_len"),
                ("buffer", "dq_out"),
                ("buffer", "dk_out"),
                ("parameter", "num_qk_heads"),
                ("parameter", "num_v_heads"),
                ("parameter", "group"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_103a/cake_kda_chunk_train_44978dc5595f5354f436_kernel.cu",
                "sm_103a/cake_kda_chunk_train_44978dc5595f5354f436_binding.cu",
            ),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 0,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "finalize": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_2a3743d77da552346275",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_2a3743d77da552346275_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_2a3743d77da552346275",
            "arg_plan": (
                ("buffer", "dA_part"),
                ("buffer", "dbias_part"),
                ("buffer", "dA_log"),
                ("buffer", "dt_bias_grad"),
                ("parameter", "num_chunks"),
                ("parameter", "num_heads"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_100a/cake_kda_chunk_train_2a3743d77da552346275_kernel.cu",
                "sm_100a/cake_kda_chunk_train_2a3743d77da552346275_binding.cu",
            ),
            "block": (128, 1, 1),
            "dynamic_smem_bytes": 128,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_eeab91af7e13dac4193a",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_eeab91af7e13dac4193a_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_eeab91af7e13dac4193a",
            "arg_plan": (
                ("buffer", "dA_part"),
                ("buffer", "dbias_part"),
                ("buffer", "dA_log"),
                ("buffer", "dt_bias_grad"),
                ("parameter", "num_chunks"),
                ("parameter", "num_heads"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_103a/cake_kda_chunk_train_eeab91af7e13dac4193a_kernel.cu",
                "sm_103a/cake_kda_chunk_train_eeab91af7e13dac4193a_binding.cu",
            ),
            "block": (128, 1, 1),
            "dynamic_smem_bytes": 128,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
}


def device_arch(device=None):
    """Map the current CUDA device to the exported architecture tag."""
    capability = torch.cuda.get_device_capability(device)
    try:
        return SUPPORTED_CAPABILITIES[capability]
    except KeyError:
        raise NotImplementedError(
            f"the deterministic chunked KDA training backward requires SM100a or SM103a, got {capability}"
        ) from None


@cache
def load(stage, arch):
    """Build (once per extension cache) and import the native module for ``stage`` on ``arch``."""
    from torch.utils.cpp_extension import load as load_extension

    record = MODULES[stage][arch]
    return load_extension(
        name=record["cache_name"],
        sources=[str(_CSRC / relative) for relative in record["sources"]],
        extra_include_paths=[str(_CSRC)],
        extra_cuda_cflags=[_ARCH_FLAGS[arch], "-O3", *_CONVERSION_FLAGS, *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        verbose=os.environ.get("MILES_KDA_CHUNK_TRAIN_VERBOSE_BUILD", "0") == "1",
    )


class NativeKernel:
    """One generated stage; named bindings are mapped onto the exported argument plan.

    ``launch(grid=(x, y, z), **bindings)`` binds tensors and scalars by their
    exported names.  The caller-owned TMA descriptor workspace and the grid
    are filled in automatically; descriptors are re-encoded on every call.
    """

    def __init__(self, stage, arch=None, device=None):
        self.stage = stage
        self.arch = arch or device_arch(device)
        record = MODULES[stage][self.arch]
        self.record = record
        native = load(stage, self.arch)
        self._call = getattr(native, record["ffi_entry"])
        self._arg_plan = tuple(tuple(item) for item in record["arg_plan"])
        workspace_bytes = int(record["tma_workspace_bytes"])
        self.descriptor_storage = (
            torch.empty(
                workspace_bytes,
                dtype=torch.uint8,
                device=device if device is not None else "cuda",
            )
            if workspace_bytes
            else None
        )

    def launch(self, *, grid, **bindings):
        grid = tuple(int(g) for g in grid) + (1,) * (3 - len(grid))
        args = []
        used = set()
        for kind, name in self._arg_plan:
            if kind == "grid":
                args.append(grid[("grid_x", "grid_y", "grid_z").index(name)])
            elif kind == "workspace":
                args.append(self.descriptor_storage)
            else:
                if name not in bindings:
                    raise KeyError(f"{self.stage}: missing binding {name!r}")
                args.append(bindings[name])
                used.add(name)
        unexpected = set(bindings) - used
        if unexpected:
            raise KeyError(f"{self.stage}: unexpected bindings {sorted(unexpected)!r}")
        return self._call(*args)


@cache
def kernel(stage, arch):
    """Return the process-wide NativeKernel for ``stage`` on ``arch``."""
    return NativeKernel(stage, arch)
