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

import operator
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
            "name": "cake_kda_chunk_train_35c13d4668c0cafffff2",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_35c13d4668c0cafffff2_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_35c13d4668c0cafffff2",
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
                "sm_100a/cake_kda_chunk_train_35c13d4668c0cafffff2_kernel.cu",
                "sm_100a/cake_kda_chunk_train_35c13d4668c0cafffff2_binding.cu",
            ),
            "block": (1024, 1, 1),
            "dynamic_smem_bytes": 36864,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_fc6d8e35d5abf17f5ea6",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_fc6d8e35d5abf17f5ea6_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_fc6d8e35d5abf17f5ea6",
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
                "sm_103a/cake_kda_chunk_train_fc6d8e35d5abf17f5ea6_kernel.cu",
                "sm_103a/cake_kda_chunk_train_fc6d8e35d5abf17f5ea6_binding.cu",
            ),
            "block": (1024, 1, 1),
            "dynamic_smem_bytes": 36864,
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
    "fwdh_slices": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_db5273405f7c20c25dda",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_db5273405f7c20c25dda_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_db5273405f7c20c25dda",
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
                "sm_100a/cake_kda_chunk_train_db5273405f7c20c25dda_kernel.cu",
                "sm_100a/cake_kda_chunk_train_db5273405f7c20c25dda_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 78848,
            "tma_workspace_bytes": 256,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_1e1ee370f02052b881ce",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_1e1ee370f02052b881ce_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_1e1ee370f02052b881ce",
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
                "sm_103a/cake_kda_chunk_train_1e1ee370f02052b881ce_kernel.cu",
                "sm_103a/cake_kda_chunk_train_1e1ee370f02052b881ce_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 78848,
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
    "dhu_slices": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_7bec085839eca4d8944f",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_7bec085839eca4d8944f_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_7bec085839eca4d8944f",
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
                "sm_100a/cake_kda_chunk_train_7bec085839eca4d8944f_kernel.cu",
                "sm_100a/cake_kda_chunk_train_7bec085839eca4d8944f_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 119808,
            "tma_workspace_bytes": 512,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_f36b958b2e98912bca25",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_f36b958b2e98912bca25_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_f36b958b2e98912bca25",
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
                "sm_103a/cake_kda_chunk_train_f36b958b2e98912bca25_kernel.cu",
                "sm_103a/cake_kda_chunk_train_f36b958b2e98912bca25_binding.cu",
            ),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 119808,
            "tma_workspace_bytes": 512,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
    "dqkg": {
        "sm_100a": {
            "name": "cake_kda_chunk_train_ceac272b26b20ca455f8",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_ceac272b26b20ca455f8_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_ceac272b26b20ca455f8",
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
                ("parameter", "num_items"),
                ("buffer", "chunk_bos"),
                ("buffer", "chunk_len"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_100a/cake_kda_chunk_train_ceac272b26b20ca455f8_kernel.cu",
                "sm_100a/cake_kda_chunk_train_ceac272b26b20ca455f8_binding.cu",
            ),
            "block": (576, 1, 1),
            "dynamic_smem_bytes": 231936,
            "tma_workspace_bytes": 896,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_72ccdb777dc2a90c17d7",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_72ccdb777dc2a90c17d7_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_72ccdb777dc2a90c17d7",
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
                ("parameter", "num_items"),
                ("buffer", "chunk_bos"),
                ("buffer", "chunk_len"),
                ("parameter", "scale"),
                ("workspace", "tma_descriptor_workspace"),
                ("grid", "grid_x"),
                ("grid", "grid_y"),
                ("grid", "grid_z"),
            ),
            "sources": (
                "sm_103a/cake_kda_chunk_train_72ccdb777dc2a90c17d7_kernel.cu",
                "sm_103a/cake_kda_chunk_train_72ccdb777dc2a90c17d7_binding.cu",
            ),
            "block": (576, 1, 1),
            "dynamic_smem_bytes": 231936,
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
            "name": "cake_kda_chunk_train_5bc05c8f3ffbd9e5ae38",
            "arch": "sm_100a",
            "cache_name": "cake_kda_chunk_train_5bc05c8f3ffbd9e5ae38_sm_100a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_5bc05c8f3ffbd9e5ae38",
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
                "sm_100a/cake_kda_chunk_train_5bc05c8f3ffbd9e5ae38_kernel.cu",
                "sm_100a/cake_kda_chunk_train_5bc05c8f3ffbd9e5ae38_binding.cu",
            ),
            "block": (128, 1, 1),
            "dynamic_smem_bytes": 128,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
        "sm_103a": {
            "name": "cake_kda_chunk_train_cef7dce854d291418931",
            "arch": "sm_103a",
            "cache_name": "cake_kda_chunk_train_cef7dce854d291418931_sm_103a",
            "kernel_symbol": "kernel_cake_kda_chunk_train_cef7dce854d291418931",
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
                "sm_103a/cake_kda_chunk_train_cef7dce854d291418931_kernel.cu",
                "sm_103a/cake_kda_chunk_train_cef7dce854d291418931_binding.cu",
            ),
            "block": (128, 1, 1),
            "dynamic_smem_bytes": 128,
            "tma_workspace_bytes": 0,
            "ffi_entry": "run",
            "compile_flags": ("--use_fast_math",),
        },
    },
}


_DEVICE_ARCH: dict[int, str] = {}


def device_arch(device=None):
    """Map the current CUDA device to the exported architecture tag (cached per device index: the
    capability query costs ~10 us per call and the backward asks once per call)."""
    index = torch.cuda.current_device() if device is None else (device if isinstance(device, int) else device.index)
    if index is None:
        index = torch.cuda.current_device()
    arch = _DEVICE_ARCH.get(index)
    if arch is None:
        capability = torch.cuda.get_device_capability(index)
        try:
            arch = SUPPORTED_CAPABILITIES[capability]
        except KeyError:
            raise NotImplementedError(
                f"the deterministic chunked KDA training backward requires SM100a or SM103a, got {capability}"
            ) from None
        _DEVICE_ARCH[index] = arch
    return arch


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


_GRID_AXES = ("grid_x", "grid_y", "grid_z")
# argument-plan segment kinds (see _plan_segments)
_BINDINGS, _GRID, _WORKSPACE = "bindings", "grid", "workspace"


def _is_binding(kind):
    return kind != "grid" and kind != "workspace"


def _bindings_getter(names):
    """``bindings -> tuple of the values of names, in order`` (``itemgetter`` unwraps a single key)."""
    if len(names) == 1:
        (name,) = names

        def single(bindings):
            return (bindings[name],)

        return single
    return operator.itemgetter(*names)


def _plan_segments(arg_plan, descriptor_storage):
    """Compile an exported argument plan into positional segments for :meth:`NativeKernel.launch`.

    Consecutive named entries (buffers, TMA buffers, parameters) collapse into one
    ``(_BINDINGS, getter)`` segment whose getter pulls their values out of the caller's bindings in
    plan order with a single ``itemgetter`` call; ``(_GRID, axis)`` takes one launch-grid component
    and ``(_WORKSPACE, tensor)`` the kernel's fixed descriptor workspace.  The concatenated segments
    reproduce the plan's positional order exactly.
    """
    segments = []
    run = []

    def flush():
        if run:
            segments.append((_BINDINGS, _bindings_getter(tuple(run))))
            run.clear()

    for kind, name in arg_plan:
        if kind == "grid":
            flush()
            segments.append((_GRID, _GRID_AXES.index(name)))
        elif kind == "workspace":
            flush()
            segments.append((_WORKSPACE, descriptor_storage))
        else:
            run.append(name)
    flush()
    return tuple(segments)


class NativeKernel:
    """One generated stage; named bindings are mapped onto the exported argument plan.

    ``launch(grid=(x, y, z), **bindings)`` binds tensors and scalars by their
    exported names.  The caller-owned TMA descriptor workspace and the grid
    are filled in automatically; descriptors are re-encoded on every call.
    The plan is compiled once here into positional segments so a launch costs
    one set comparison of the binding names and a few ``itemgetter`` calls.
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
        self._binding_names = frozenset(name for kind, name in self._arg_plan if _is_binding(kind))
        self._segments = _plan_segments(self._arg_plan, self.descriptor_storage)

    def launch(self, *, grid, **bindings):
        if bindings.keys() != self._binding_names:
            self._raise_binding_mismatch(bindings)
        if len(grid) != 3:
            grid = tuple(grid) + (1,) * (3 - len(grid))
        args = []
        for kind, payload in self._segments:
            if kind is _BINDINGS:
                args.extend(payload(bindings))
            elif kind is _GRID:
                args.append(int(grid[payload]))
            else:
                args.append(payload)
        return self._call(*args)

    def _raise_binding_mismatch(self, bindings):
        for kind, name in self._arg_plan:
            if _is_binding(kind) and name not in bindings:
                raise KeyError(f"{self.stage}: missing binding {name!r}")
        unexpected = set(bindings) - self._binding_names
        raise KeyError(f"{self.stage}: unexpected bindings {sorted(unexpected)!r}")


@cache
def kernel(stage, arch):
    """Return the process-wide NativeKernel for ``stage`` on ``arch``."""
    return NativeKernel(stage, arch)
