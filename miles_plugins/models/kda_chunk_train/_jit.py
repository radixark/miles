"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0

Native loading for the generated chunked KDA training kernels (SM100a / SM103a): the forward stages
(``fwd_prep`` / ``fwd_intra`` / ``fwd_o``), the serial walks shared by both directions (``wy``, ``fwdh``,
``dhu`` and their four-slice variants) and the backward stages.

The registry is filled mechanically from the Cake compiler's resolved production builds
(the Cake export recipe's ``modules`` subcommand, from ``registry_sm_100a.json`` / ``registry_sm_103a.json``):
one device/launcher source pair per stage and architecture under ``csrc/<arch>/``. Each stage builds on
first use as a torch CUDA extension (``nvcc`` + ``ninja``; cached under ``TORCH_EXTENSIONS_DIR``).

Host ABI: by-value ``grid_constant`` tensor maps. Every TMA operand (plan kind ``tma_buffer``) is a
``const __grid_constant__ CUtensorMap`` kernel parameter that the launcher encodes on the host on every
call from the tensor it is bound to, so tensor addresses may change between calls and no caller-owned
descriptor workspace exists (the pointer ABI of earlier exports, with its ``tma_workspace_bytes`` and
``workspace`` plan entries, is gone). The ``fwdh`` / ``dhu`` plans end in the four fp32 state slots of the
serial walks (``h0``/``use_h0``/``ht_out``/``store_ht`` and ``dht``/``use_dht``/``dh0_out``/``store_dh0``); a
flag-off slot is bound to a cached one-element zero the kernels never dereference.
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
    "fwd_prep": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_2a129056513a41b96997',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_2a129056513a41b96997_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_2a129056513a41b96997',
            "arg_plan": (
                ('buffer', 'q'),
                ('buffer', 'k'),
                ('buffer', 'v'),
                ('buffer', 'g_raw'),
                ('buffer', 'beta_logits'),
                ('buffer', 'A_log'),
                ('buffer', 'dt_bias'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'q_norm_out'),
                ('buffer', 'k_norm_out'),
                ('buffer', 'q_rstd_out'),
                ('buffer', 'k_rstd_out'),
                ('buffer', 'beta_out'),
                ('buffer', 'gk_out'),
                ('buffer', 'vb_out'),
                ('buffer', 'kb_out'),
                ('buffer', 'kg_out'),
                ('buffer', 'qg_out'),
                ('parameter', 'num_qk_heads'),
                ('parameter', 'num_heads'),
                ('parameter', 'group'),
                ('parameter', 'write_qg'),
                ('parameter', 'lower_bound'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_2a129056513a41b96997_kernel.cu', 'sm_100a/cake_kda_chunk_train_2a129056513a41b96997_binding.cu'),
            "block": (512, 1, 1),
            "dynamic_smem_bytes": 36864,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_19feab58a3e430f33a04',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_19feab58a3e430f33a04_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_19feab58a3e430f33a04',
            "arg_plan": (
                ('buffer', 'q'),
                ('buffer', 'k'),
                ('buffer', 'v'),
                ('buffer', 'g_raw'),
                ('buffer', 'beta_logits'),
                ('buffer', 'A_log'),
                ('buffer', 'dt_bias'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'q_norm_out'),
                ('buffer', 'k_norm_out'),
                ('buffer', 'q_rstd_out'),
                ('buffer', 'k_rstd_out'),
                ('buffer', 'beta_out'),
                ('buffer', 'gk_out'),
                ('buffer', 'vb_out'),
                ('buffer', 'kb_out'),
                ('buffer', 'kg_out'),
                ('buffer', 'qg_out'),
                ('parameter', 'num_qk_heads'),
                ('parameter', 'num_heads'),
                ('parameter', 'group'),
                ('parameter', 'write_qg'),
                ('parameter', 'lower_bound'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_19feab58a3e430f33a04_kernel.cu', 'sm_103a/cake_kda_chunk_train_19feab58a3e430f33a04_binding.cu'),
            "block": (512, 1, 1),
            "dynamic_smem_bytes": 36864,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "fwd_intra": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_96dafe35a2814c11fd5d',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_96dafe35a2814c11fd5d_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_96dafe35a2814c11fd5d',
            "arg_plan": (
                ('buffer', 'q_norm'),
                ('buffer', 'k_norm'),
                ('buffer', 'gk'),
                ('buffer', 'beta'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'aqk_ext'),
                ('buffer', 'akk_ext'),
                ('buffer', 'aqk_int'),
                ('buffer', 'akk_int'),
                ('parameter', 'num_heads'),
                ('parameter', 'num_qk_heads'),
                ('parameter', 'group'),
                ('parameter', 'write_int'),
                ('parameter', 'scale'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_96dafe35a2814c11fd5d_kernel.cu', 'sm_100a/cake_kda_chunk_train_96dafe35a2814c11fd5d_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 108544,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_6f7288f457dcde3fc286',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_6f7288f457dcde3fc286_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_6f7288f457dcde3fc286',
            "arg_plan": (
                ('buffer', 'q_norm'),
                ('buffer', 'k_norm'),
                ('buffer', 'gk'),
                ('buffer', 'beta'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'aqk_ext'),
                ('buffer', 'akk_ext'),
                ('buffer', 'aqk_int'),
                ('buffer', 'akk_int'),
                ('parameter', 'num_heads'),
                ('parameter', 'num_qk_heads'),
                ('parameter', 'group'),
                ('parameter', 'write_int'),
                ('parameter', 'scale'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_6f7288f457dcde3fc286_kernel.cu', 'sm_103a/cake_kda_chunk_train_6f7288f457dcde3fc286_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 108544,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "fwd_o": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_3e582854664e0a954494',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_3e582854664e0a954494_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_3e582854664e0a954494',
            "arg_plan": (
                ('tma_buffer', 'qg_tma'),
                ('tma_buffer', 'h_tma'),
                ('tma_buffer', 'aqk_tma'),
                ('tma_buffer', 'vn_tma'),
                ('buffer', 'o_out'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('parameter', 'num_heads'),
                ('parameter', 'scale'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_3e582854664e0a954494_kernel.cu', 'sm_100a/cake_kda_chunk_train_3e582854664e0a954494_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 74752,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_db4610672288aca7194e',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_db4610672288aca7194e_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_db4610672288aca7194e',
            "arg_plan": (
                ('tma_buffer', 'qg_tma'),
                ('tma_buffer', 'h_tma'),
                ('tma_buffer', 'aqk_tma'),
                ('tma_buffer', 'vn_tma'),
                ('buffer', 'o_out'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('parameter', 'num_heads'),
                ('parameter', 'scale'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_db4610672288aca7194e_kernel.cu', 'sm_103a/cake_kda_chunk_train_db4610672288aca7194e_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 74752,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "prep": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_555284359eb58d1fd6eb',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_555284359eb58d1fd6eb_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_555284359eb58d1fd6eb',
            "arg_plan": (
                ('buffer', 'g_raw'),
                ('buffer', 'q_norm'),
                ('buffer', 'k_norm'),
                ('buffer', 'v'),
                ('buffer', 'beta'),
                ('buffer', 'A_log'),
                ('buffer', 'dt_bias'),
                ('buffer', 'aqk'),
                ('buffer', 'akk'),
                ('buffer', 'do'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'gk_out'),
                ('buffer', 'vb_out'),
                ('buffer', 'kb_out'),
                ('buffer', 'qg_out'),
                ('buffer', 'kg_out'),
                ('buffer', 'ke_out'),
                ('buffer', 'qe_out'),
                ('buffer', 'aqk_tril'),
                ('buffer', 'akk_int'),
                ('buffer', 'do_int'),
                ('buffer', 'v_int'),
                ('buffer', 'beta_int'),
                ('parameter', 'num_qk_heads'),
                ('parameter', 'num_heads'),
                ('parameter', 'group'),
                ('parameter', 'copy_inputs'),
                ('parameter', 'lower_bound'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_555284359eb58d1fd6eb_kernel.cu', 'sm_100a/cake_kda_chunk_train_555284359eb58d1fd6eb_binding.cu'),
            "block": (1024, 1, 1),
            "dynamic_smem_bytes": 36864,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_4ebec0b3e85c45c1dd4f',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_4ebec0b3e85c45c1dd4f_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_4ebec0b3e85c45c1dd4f',
            "arg_plan": (
                ('buffer', 'g_raw'),
                ('buffer', 'q_norm'),
                ('buffer', 'k_norm'),
                ('buffer', 'v'),
                ('buffer', 'beta'),
                ('buffer', 'A_log'),
                ('buffer', 'dt_bias'),
                ('buffer', 'aqk'),
                ('buffer', 'akk'),
                ('buffer', 'do'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'gk_out'),
                ('buffer', 'vb_out'),
                ('buffer', 'kb_out'),
                ('buffer', 'qg_out'),
                ('buffer', 'kg_out'),
                ('buffer', 'ke_out'),
                ('buffer', 'qe_out'),
                ('buffer', 'aqk_tril'),
                ('buffer', 'akk_int'),
                ('buffer', 'do_int'),
                ('buffer', 'v_int'),
                ('buffer', 'beta_int'),
                ('parameter', 'num_qk_heads'),
                ('parameter', 'num_heads'),
                ('parameter', 'group'),
                ('parameter', 'copy_inputs'),
                ('parameter', 'lower_bound'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_4ebec0b3e85c45c1dd4f_kernel.cu', 'sm_103a/cake_kda_chunk_train_4ebec0b3e85c45c1dd4f_binding.cu'),
            "block": (1024, 1, 1),
            "dynamic_smem_bytes": 36864,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "wy": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_6d530164f28c11579deb',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_6d530164f28c11579deb_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_6d530164f28c11579deb',
            "arg_plan": (
                ('tma_buffer', 'akk_tma'),
                ('tma_buffer', 'vb_tma'),
                ('tma_buffer', 'kb_tma'),
                ('buffer', 'u_out'),
                ('buffer', 'w_out'),
                ('parameter', 'num_heads'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_6d530164f28c11579deb_kernel.cu', 'sm_100a/cake_kda_chunk_train_6d530164f28c11579deb_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_804c1f41fc7936726b12',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_804c1f41fc7936726b12_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_804c1f41fc7936726b12',
            "arg_plan": (
                ('tma_buffer', 'akk_tma'),
                ('tma_buffer', 'vb_tma'),
                ('tma_buffer', 'kb_tma'),
                ('buffer', 'u_out'),
                ('buffer', 'w_out'),
                ('parameter', 'num_heads'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_804c1f41fc7936726b12_kernel.cu', 'sm_103a/cake_kda_chunk_train_804c1f41fc7936726b12_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "fwdh": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_ff94179b81a382873c7c',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_ff94179b81a382873c7c_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_ff94179b81a382873c7c',
            "arg_plan": (
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'kg_tma'),
                ('buffer', 'u'),
                ('buffer', 'gk'),
                ('buffer', 'h_out'),
                ('buffer', 'v_new'),
                ('parameter', 'num_heads'),
                ('buffer', 'seq_chunk_start'),
                ('buffer', 'h0'),
                ('parameter', 'use_h0'),
                ('buffer', 'ht_out'),
                ('parameter', 'store_ht'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_ff94179b81a382873c7c_kernel.cu', 'sm_100a/cake_kda_chunk_train_ff94179b81a382873c7c_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 91136,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_c59100ebf990cc5666fa',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_c59100ebf990cc5666fa_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_c59100ebf990cc5666fa',
            "arg_plan": (
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'kg_tma'),
                ('buffer', 'u'),
                ('buffer', 'gk'),
                ('buffer', 'h_out'),
                ('buffer', 'v_new'),
                ('parameter', 'num_heads'),
                ('buffer', 'seq_chunk_start'),
                ('buffer', 'h0'),
                ('parameter', 'use_h0'),
                ('buffer', 'ht_out'),
                ('parameter', 'store_ht'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_c59100ebf990cc5666fa_kernel.cu', 'sm_103a/cake_kda_chunk_train_c59100ebf990cc5666fa_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 91136,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "fwdh_slices": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_9c8372341e0f1aa98235',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_9c8372341e0f1aa98235_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_9c8372341e0f1aa98235',
            "arg_plan": (
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'kg_tma'),
                ('buffer', 'u'),
                ('buffer', 'gk'),
                ('buffer', 'h_out'),
                ('buffer', 'v_new'),
                ('parameter', 'num_heads'),
                ('buffer', 'seq_chunk_start'),
                ('buffer', 'h0'),
                ('parameter', 'use_h0'),
                ('buffer', 'ht_out'),
                ('parameter', 'store_ht'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_9c8372341e0f1aa98235_kernel.cu', 'sm_100a/cake_kda_chunk_train_9c8372341e0f1aa98235_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 78848,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_34c513b4dae1a5ee444d',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_34c513b4dae1a5ee444d_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_34c513b4dae1a5ee444d',
            "arg_plan": (
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'kg_tma'),
                ('buffer', 'u'),
                ('buffer', 'gk'),
                ('buffer', 'h_out'),
                ('buffer', 'v_new'),
                ('parameter', 'num_heads'),
                ('buffer', 'seq_chunk_start'),
                ('buffer', 'h0'),
                ('parameter', 'use_h0'),
                ('buffer', 'ht_out'),
                ('parameter', 'store_ht'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_34c513b4dae1a5ee444d_kernel.cu', 'sm_103a/cake_kda_chunk_train_34c513b4dae1a5ee444d_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 78848,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "dav": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_56bb6bf3b7eb54f5ed03',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_56bb6bf3b7eb54f5ed03_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_56bb6bf3b7eb54f5ed03',
            "arg_plan": (
                ('tma_buffer', 'do_tma'),
                ('tma_buffer', 'vnew_tma'),
                ('tma_buffer', 'aqk_tma'),
                ('buffer', 'dAqk'),
                ('buffer', 'dv1'),
                ('parameter', 'num_heads'),
                ('parameter', 'scale'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_56bb6bf3b7eb54f5ed03_kernel.cu', 'sm_100a/cake_kda_chunk_train_56bb6bf3b7eb54f5ed03_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_4b35f4d82aa290fffcbe',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_4b35f4d82aa290fffcbe_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_4b35f4d82aa290fffcbe',
            "arg_plan": (
                ('tma_buffer', 'do_tma'),
                ('tma_buffer', 'vnew_tma'),
                ('tma_buffer', 'aqk_tma'),
                ('buffer', 'dAqk'),
                ('buffer', 'dv1'),
                ('parameter', 'num_heads'),
                ('parameter', 'scale'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_4b35f4d82aa290fffcbe_kernel.cu', 'sm_103a/cake_kda_chunk_train_4b35f4d82aa290fffcbe_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "dhu": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_d626f8a1d51fb8354a8a',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_d626f8a1d51fb8354a8a_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_d626f8a1d51fb8354a8a',
            "arg_plan": (
                ('tma_buffer', 'kg_tma'),
                ('tma_buffer', 'qg_tma'),
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'do_tma'),
                ('buffer', 'dv1'),
                ('buffer', 'gk'),
                ('buffer', 'dh_out'),
                ('buffer', 'dv2'),
                ('parameter', 'num_heads'),
                ('buffer', 'seq_chunk_start'),
                ('parameter', 'scale'),
                ('buffer', 'dht'),
                ('parameter', 'use_dht'),
                ('buffer', 'dh0_out'),
                ('parameter', 'store_dh0'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_d626f8a1d51fb8354a8a_kernel.cu', 'sm_100a/cake_kda_chunk_train_d626f8a1d51fb8354a8a_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 140288,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_5610797e1f62cd9ef123',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_5610797e1f62cd9ef123_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_5610797e1f62cd9ef123',
            "arg_plan": (
                ('tma_buffer', 'kg_tma'),
                ('tma_buffer', 'qg_tma'),
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'do_tma'),
                ('buffer', 'dv1'),
                ('buffer', 'gk'),
                ('buffer', 'dh_out'),
                ('buffer', 'dv2'),
                ('parameter', 'num_heads'),
                ('buffer', 'seq_chunk_start'),
                ('parameter', 'scale'),
                ('buffer', 'dht'),
                ('parameter', 'use_dht'),
                ('buffer', 'dh0_out'),
                ('parameter', 'store_dh0'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_5610797e1f62cd9ef123_kernel.cu', 'sm_103a/cake_kda_chunk_train_5610797e1f62cd9ef123_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 140288,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "dhu_slices": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_c00669e1c022bbebc1e3',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_c00669e1c022bbebc1e3_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_c00669e1c022bbebc1e3',
            "arg_plan": (
                ('tma_buffer', 'kg_tma'),
                ('tma_buffer', 'qg_tma'),
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'do_tma'),
                ('buffer', 'dv1'),
                ('buffer', 'gk'),
                ('buffer', 'dh_out'),
                ('buffer', 'dv2'),
                ('parameter', 'num_heads'),
                ('buffer', 'seq_chunk_start'),
                ('parameter', 'scale'),
                ('buffer', 'dht'),
                ('parameter', 'use_dht'),
                ('buffer', 'dh0_out'),
                ('parameter', 'store_dh0'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_c00669e1c022bbebc1e3_kernel.cu', 'sm_100a/cake_kda_chunk_train_c00669e1c022bbebc1e3_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 119808,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_b1eab85922d7198a10e7',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_b1eab85922d7198a10e7_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_b1eab85922d7198a10e7',
            "arg_plan": (
                ('tma_buffer', 'kg_tma'),
                ('tma_buffer', 'qg_tma'),
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'do_tma'),
                ('buffer', 'dv1'),
                ('buffer', 'gk'),
                ('buffer', 'dh_out'),
                ('buffer', 'dv2'),
                ('parameter', 'num_heads'),
                ('buffer', 'seq_chunk_start'),
                ('parameter', 'scale'),
                ('buffer', 'dht'),
                ('parameter', 'use_dht'),
                ('buffer', 'dh0_out'),
                ('parameter', 'store_dh0'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_b1eab85922d7198a10e7_kernel.cu', 'sm_103a/cake_kda_chunk_train_b1eab85922d7198a10e7_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 119808,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "dqkg": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_81634e6718626a21f444',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_81634e6718626a21f444_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_81634e6718626a21f444',
            "arg_plan": (
                ('tma_buffer', 'do_tma'),
                ('tma_buffer', 'vn_tma'),
                ('tma_buffer', 'dv2_tma'),
                ('tma_buffer', 'v_tma'),
                ('tma_buffer', 'h_tma'),
                ('tma_buffer', 'dh_tma'),
                ('tma_buffer', 'akk_tma'),
                ('tma_buffer', 'gk_tma'),
                ('tma_buffer', 'k_tma'),
                ('tma_buffer', 'q_tma'),
                ('buffer', 'beta'),
                ('buffer', 'dq_out'),
                ('buffer', 'dk_out'),
                ('buffer', 'dg_out'),
                ('buffer', 'db_out'),
                ('buffer', 'dAkk_out'),
                ('buffer', 'dv_out'),
                ('parameter', 'num_heads'),
                ('parameter', 'num_items'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('parameter', 'scale'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_81634e6718626a21f444_kernel.cu', 'sm_100a/cake_kda_chunk_train_81634e6718626a21f444_binding.cu'),
            "block": (576, 1, 1),
            "dynamic_smem_bytes": 232192,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_f602f33b0cfba58ee79f',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_f602f33b0cfba58ee79f_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_f602f33b0cfba58ee79f',
            "arg_plan": (
                ('tma_buffer', 'do_tma'),
                ('tma_buffer', 'vn_tma'),
                ('tma_buffer', 'dv2_tma'),
                ('tma_buffer', 'v_tma'),
                ('tma_buffer', 'h_tma'),
                ('tma_buffer', 'dh_tma'),
                ('tma_buffer', 'akk_tma'),
                ('tma_buffer', 'gk_tma'),
                ('tma_buffer', 'k_tma'),
                ('tma_buffer', 'q_tma'),
                ('buffer', 'beta'),
                ('buffer', 'dq_out'),
                ('buffer', 'dk_out'),
                ('buffer', 'dg_out'),
                ('buffer', 'db_out'),
                ('buffer', 'dAkk_out'),
                ('buffer', 'dv_out'),
                ('parameter', 'num_heads'),
                ('parameter', 'num_items'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('parameter', 'scale'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_f602f33b0cfba58ee79f_kernel.cu', 'sm_103a/cake_kda_chunk_train_f602f33b0cfba58ee79f_binding.cu'),
            "block": (576, 1, 1),
            "dynamic_smem_bytes": 232192,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "intra": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_6a5d6c1241cd1cb26f35',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_6a5d6c1241cd1cb26f35_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_6a5d6c1241cd1cb26f35',
            "arg_plan": (
                ('buffer', 'dAqk'),
                ('buffer', 'dAkk'),
                ('buffer', 'gk'),
                ('buffer', 'k_e'),
                ('buffer', 'q_e'),
                ('buffer', 'beta'),
                ('buffer', 'dq_f'),
                ('buffer', 'dk_f'),
                ('buffer', 'dg_f'),
                ('buffer', 'db_f'),
                ('buffer', 'dq_out'),
                ('buffer', 'dk_out'),
                ('buffer', 'dg_out'),
                ('buffer', 'db_out'),
                ('parameter', 'num_heads'),
                ('parameter', 'num_qk_heads'),
                ('parameter', 'group'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_6a5d6c1241cd1cb26f35_kernel.cu', 'sm_100a/cake_kda_chunk_train_6a5d6c1241cd1cb26f35_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 103680,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_910d2888fd05cc82dda5',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_910d2888fd05cc82dda5_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_910d2888fd05cc82dda5',
            "arg_plan": (
                ('buffer', 'dAqk'),
                ('buffer', 'dAkk'),
                ('buffer', 'gk'),
                ('buffer', 'k_e'),
                ('buffer', 'q_e'),
                ('buffer', 'beta'),
                ('buffer', 'dq_f'),
                ('buffer', 'dk_f'),
                ('buffer', 'dg_f'),
                ('buffer', 'db_f'),
                ('buffer', 'dq_out'),
                ('buffer', 'dk_out'),
                ('buffer', 'dg_out'),
                ('buffer', 'db_out'),
                ('parameter', 'num_heads'),
                ('parameter', 'num_qk_heads'),
                ('parameter', 'group'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_910d2888fd05cc82dda5_kernel.cu', 'sm_103a/cake_kda_chunk_train_910d2888fd05cc82dda5_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 103680,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "intra_qk": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_0e2f4d27043ca9191405',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_0e2f4d27043ca9191405_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_0e2f4d27043ca9191405',
            "arg_plan": (
                ('buffer', 'dAqk'),
                ('buffer', 'dAkk'),
                ('buffer', 'gk'),
                ('buffer', 'k_e'),
                ('buffer', 'q_e'),
                ('buffer', 'beta'),
                ('buffer', 'dq_f'),
                ('buffer', 'dk_f'),
                ('buffer', 'dg_f'),
                ('buffer', 'db_f'),
                ('buffer', 'q_rstd'),
                ('buffer', 'k_rstd'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'dq_ext'),
                ('buffer', 'dk_ext'),
                ('buffer', 'dg_out'),
                ('buffer', 'db_out'),
                ('parameter', 'num_heads'),
                ('parameter', 'num_qk_heads'),
                ('parameter', 'group'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_0e2f4d27043ca9191405_kernel.cu', 'sm_100a/cake_kda_chunk_train_0e2f4d27043ca9191405_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 172288,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_735a2f1a18af1976e147',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_735a2f1a18af1976e147_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_735a2f1a18af1976e147',
            "arg_plan": (
                ('buffer', 'dAqk'),
                ('buffer', 'dAkk'),
                ('buffer', 'gk'),
                ('buffer', 'k_e'),
                ('buffer', 'q_e'),
                ('buffer', 'beta'),
                ('buffer', 'dq_f'),
                ('buffer', 'dk_f'),
                ('buffer', 'dg_f'),
                ('buffer', 'db_f'),
                ('buffer', 'q_rstd'),
                ('buffer', 'k_rstd'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'dq_ext'),
                ('buffer', 'dk_ext'),
                ('buffer', 'dg_out'),
                ('buffer', 'db_out'),
                ('parameter', 'num_heads'),
                ('parameter', 'num_qk_heads'),
                ('parameter', 'group'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_735a2f1a18af1976e147_kernel.cu', 'sm_103a/cake_kda_chunk_train_735a2f1a18af1976e147_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 172288,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "gate_epilogue": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_f91bf44f4e9c3cb63367',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_f91bf44f4e9c3cb63367_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_f91bf44f4e9c3cb63367',
            "arg_plan": (
                ('buffer', 'dg_intra'),
                ('buffer', 'g_raw'),
                ('buffer', 'db_total'),
                ('buffer', 'beta_raw'),
                ('buffer', 'A_log'),
                ('buffer', 'dt_bias'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'dg_out'),
                ('buffer', 'dbeta'),
                ('buffer', 'dA_part'),
                ('buffer', 'dbias_part'),
                ('parameter', 'num_heads'),
                ('parameter', 'lower_bound'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_f91bf44f4e9c3cb63367_kernel.cu', 'sm_100a/cake_kda_chunk_train_f91bf44f4e9c3cb63367_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 12288,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_84e63dc1103dd28dcf85',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_84e63dc1103dd28dcf85_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_84e63dc1103dd28dcf85',
            "arg_plan": (
                ('buffer', 'dg_intra'),
                ('buffer', 'g_raw'),
                ('buffer', 'db_total'),
                ('buffer', 'beta_raw'),
                ('buffer', 'A_log'),
                ('buffer', 'dt_bias'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'dg_out'),
                ('buffer', 'dbeta'),
                ('buffer', 'dA_part'),
                ('buffer', 'dbias_part'),
                ('parameter', 'num_heads'),
                ('parameter', 'lower_bound'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_84e63dc1103dd28dcf85_kernel.cu', 'sm_103a/cake_kda_chunk_train_84e63dc1103dd28dcf85_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 12288,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "qk_epilogue": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_ed63bfdc4318dc8172b8',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_ed63bfdc4318dc8172b8_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_ed63bfdc4318dc8172b8',
            "arg_plan": (
                ('buffer', 'dq_intra'),
                ('buffer', 'dk_intra'),
                ('buffer', 'q_norm'),
                ('buffer', 'k_norm'),
                ('buffer', 'q_rstd'),
                ('buffer', 'k_rstd'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'dq_out'),
                ('buffer', 'dk_out'),
                ('parameter', 'num_qk_heads'),
                ('parameter', 'num_v_heads'),
                ('parameter', 'group'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_ed63bfdc4318dc8172b8_kernel.cu', 'sm_100a/cake_kda_chunk_train_ed63bfdc4318dc8172b8_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 0,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_c8a8c7efbbf9c91cc5a9',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_c8a8c7efbbf9c91cc5a9_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_c8a8c7efbbf9c91cc5a9',
            "arg_plan": (
                ('buffer', 'dq_intra'),
                ('buffer', 'dk_intra'),
                ('buffer', 'q_norm'),
                ('buffer', 'k_norm'),
                ('buffer', 'q_rstd'),
                ('buffer', 'k_rstd'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'dq_out'),
                ('buffer', 'dk_out'),
                ('parameter', 'num_qk_heads'),
                ('parameter', 'num_v_heads'),
                ('parameter', 'group'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_c8a8c7efbbf9c91cc5a9_kernel.cu', 'sm_103a/cake_kda_chunk_train_c8a8c7efbbf9c91cc5a9_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 0,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "finalize": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_c015a75646dfde618b59',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_c015a75646dfde618b59_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_c015a75646dfde618b59',
            "arg_plan": (
                ('buffer', 'dA_part'),
                ('buffer', 'dbias_part'),
                ('buffer', 'dA_log'),
                ('buffer', 'dt_bias_grad'),
                ('parameter', 'num_chunks'),
                ('parameter', 'num_heads'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_c015a75646dfde618b59_kernel.cu', 'sm_100a/cake_kda_chunk_train_c015a75646dfde618b59_binding.cu'),
            "block": (128, 1, 1),
            "dynamic_smem_bytes": 128,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_a689781926e5d075cda3',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_a689781926e5d075cda3_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_a689781926e5d075cda3',
            "arg_plan": (
                ('buffer', 'dA_part'),
                ('buffer', 'dbias_part'),
                ('buffer', 'dA_log'),
                ('buffer', 'dt_bias_grad'),
                ('parameter', 'num_chunks'),
                ('parameter', 'num_heads'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_a689781926e5d075cda3_kernel.cu', 'sm_103a/cake_kda_chunk_train_a689781926e5d075cda3_binding.cu'),
            "block": (128, 1, 1),
            "dynamic_smem_bytes": 128,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
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
_BINDINGS, _GRID = "bindings", "grid"
_BINDING_KINDS = frozenset({"buffer", "tma_buffer", "parameter"})


def _is_binding(kind):
    return kind != "grid"


def _bindings_getter(names):
    """``bindings -> tuple of the values of names, in order`` (``itemgetter`` unwraps a single key)."""
    if len(names) == 1:
        (name,) = names

        def single(bindings):
            return (bindings[name],)

        return single
    return operator.itemgetter(*names)


def _plan_segments(stage, arg_plan):
    """Compile an exported argument plan into positional segments for :meth:`NativeKernel.launch`.

    Consecutive named entries (buffers, TMA buffers, parameters) collapse into one
    ``(_BINDINGS, getter)`` segment whose getter pulls their values out of the caller's bindings in
    plan order with a single ``itemgetter`` call; ``(_GRID, axis)`` takes one launch-grid component.
    The concatenated segments reproduce the plan's positional order exactly. Any other kind (the
    retired pointer ABI's ``workspace``) is a stale record: the stage must be re-exported.
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
        elif kind in _BINDING_KINDS:
            run.append(name)
        else:
            raise ValueError(
                f"{stage}: argument plan entry ({kind!r}, {name!r}) is not part of the by-value host ABI; "
                "re-export the stage with the Cake export recipe"
            )
    flush()
    return tuple(segments)


class NativeKernel:
    """One generated stage; named bindings are mapped onto the exported argument plan.

    ``launch(grid=(x, y, z), **bindings)`` binds tensors and scalars by their exported names; the
    grid is filled in automatically and the TMA descriptors are encoded by the launcher on every
    call (by-value ABI). The plan is compiled once here into positional segments so a launch
    costs one set comparison of the binding names and a few ``itemgetter`` calls.
    """

    def __init__(self, stage, arch=None, device=None):
        self.stage = stage
        self.arch = arch or device_arch(device)
        record = MODULES[stage][self.arch]
        self.record = record
        native = load(stage, self.arch)
        self._call = getattr(native, record["ffi_entry"])
        self._arg_plan = tuple(tuple(item) for item in record["arg_plan"])
        self._binding_names = frozenset(name for kind, name in self._arg_plan if _is_binding(kind))
        self._segments = _plan_segments(stage, self._arg_plan)

    def launch(self, *, grid, **bindings):
        if bindings.keys() != self._binding_names:
            self._raise_binding_mismatch(bindings)
        if len(grid) != 3:
            grid = tuple(grid) + (1,) * (3 - len(grid))
        args = []
        for kind, payload in self._segments:
            if kind is _BINDINGS:
                args.extend(payload(bindings))
            else:
                args.append(int(grid[payload]))
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
