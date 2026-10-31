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
            "name": 'cake_kda_chunk_train_69b883035d7d40a04b21',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_69b883035d7d40a04b21_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_69b883035d7d40a04b21',
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
            "sources": ('sm_100a/cake_kda_chunk_train_69b883035d7d40a04b21_kernel.cu', 'sm_100a/cake_kda_chunk_train_69b883035d7d40a04b21_binding.cu'),
            "block": (512, 1, 1),
            "dynamic_smem_bytes": 36864,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_3c3395129b304ecfa116',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_3c3395129b304ecfa116_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_3c3395129b304ecfa116',
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
            "sources": ('sm_103a/cake_kda_chunk_train_3c3395129b304ecfa116_kernel.cu', 'sm_103a/cake_kda_chunk_train_3c3395129b304ecfa116_binding.cu'),
            "block": (512, 1, 1),
            "dynamic_smem_bytes": 36864,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "fwd_prep_split": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_8b107b137babcf2346af',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_8b107b137babcf2346af_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_8b107b137babcf2346af',
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
            "sources": ('sm_100a/cake_kda_chunk_train_8b107b137babcf2346af_kernel.cu', 'sm_100a/cake_kda_chunk_train_8b107b137babcf2346af_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 36864,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_3db5e7fed0b308870c8e',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_3db5e7fed0b308870c8e_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_3db5e7fed0b308870c8e',
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
            "sources": ('sm_103a/cake_kda_chunk_train_3db5e7fed0b308870c8e_kernel.cu', 'sm_103a/cake_kda_chunk_train_3db5e7fed0b308870c8e_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 36864,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "fwd_intra": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_8ab61e185dab19951404',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_8ab61e185dab19951404_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_8ab61e185dab19951404',
            "arg_plan": (
                ('buffer', 'q_norm'),
                ('buffer', 'k_norm'),
                ('buffer', 'gk'),
                ('buffer', 'beta'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'aqk_ext'),
                ('buffer', 'akk_ext'),
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
            "sources": ('sm_100a/cake_kda_chunk_train_8ab61e185dab19951404_kernel.cu', 'sm_100a/cake_kda_chunk_train_8ab61e185dab19951404_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 75008,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_96104b0c21ba69292026',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_96104b0c21ba69292026_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_96104b0c21ba69292026',
            "arg_plan": (
                ('buffer', 'q_norm'),
                ('buffer', 'k_norm'),
                ('buffer', 'gk'),
                ('buffer', 'beta'),
                ('buffer', 'chunk_bos'),
                ('buffer', 'chunk_len'),
                ('buffer', 'aqk_ext'),
                ('buffer', 'akk_ext'),
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
            "sources": ('sm_103a/cake_kda_chunk_train_96104b0c21ba69292026_kernel.cu', 'sm_103a/cake_kda_chunk_train_96104b0c21ba69292026_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 75008,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "fwd_o": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_e4a0717606b0fdc9a1db',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_e4a0717606b0fdc9a1db_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_e4a0717606b0fdc9a1db',
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
                ('parameter', 'num_chunks'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_e4a0717606b0fdc9a1db_kernel.cu', 'sm_100a/cake_kda_chunk_train_e4a0717606b0fdc9a1db_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 222208,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_f054d882d5db914528e3',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_f054d882d5db914528e3_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_f054d882d5db914528e3',
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
                ('parameter', 'num_chunks'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_f054d882d5db914528e3_kernel.cu', 'sm_103a/cake_kda_chunk_train_f054d882d5db914528e3_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 222208,
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
            "name": 'cake_kda_chunk_train_fe3a009dcba2d0f52cdb',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_fe3a009dcba2d0f52cdb_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_fe3a009dcba2d0f52cdb',
            "arg_plan": (
                ('tma_buffer', 'akk_tma'),
                ('tma_buffer', 'vb_tma'),
                ('tma_buffer', 'kb_tma'),
                ('tma_buffer', 'u_tma'),
                ('tma_buffer', 'w_tma'),
                ('parameter', 'num_heads'),
                ('parameter', 'num_pairs'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_fe3a009dcba2d0f52cdb_kernel.cu', 'sm_100a/cake_kda_chunk_train_fe3a009dcba2d0f52cdb_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 230400,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_8668de69eaeb4941130d',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_8668de69eaeb4941130d_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_8668de69eaeb4941130d',
            "arg_plan": (
                ('tma_buffer', 'akk_tma'),
                ('tma_buffer', 'vb_tma'),
                ('tma_buffer', 'kb_tma'),
                ('tma_buffer', 'u_tma'),
                ('tma_buffer', 'w_tma'),
                ('parameter', 'num_heads'),
                ('parameter', 'num_pairs'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_8668de69eaeb4941130d_kernel.cu', 'sm_103a/cake_kda_chunk_train_8668de69eaeb4941130d_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 230400,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "fwdh": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_57f6b2d8df0d0d73743b',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_57f6b2d8df0d0d73743b_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_57f6b2d8df0d0d73743b',
            "arg_plan": (
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'kg_tma'),
                ('tma_buffer', 'u'),
                ('buffer', 'gk'),
                ('buffer', 'h_out'),
                ('buffer', 'v_new'),
                ('parameter', 'num_heads'),
                ('buffer', 'seq_chunk_start'),
                ('buffer', 'h0'),
                ('parameter', 'use_h0'),
                ('buffer', 'ht_out'),
                ('parameter', 'store_ht'),
                ('buffer', 'seq_order'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_57f6b2d8df0d0d73743b_kernel.cu', 'sm_100a/cake_kda_chunk_train_57f6b2d8df0d0d73743b_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 107520,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_d822b3616306ebda721f',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_d822b3616306ebda721f_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_d822b3616306ebda721f',
            "arg_plan": (
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'kg_tma'),
                ('tma_buffer', 'u'),
                ('buffer', 'gk'),
                ('buffer', 'h_out'),
                ('buffer', 'v_new'),
                ('parameter', 'num_heads'),
                ('buffer', 'seq_chunk_start'),
                ('buffer', 'h0'),
                ('parameter', 'use_h0'),
                ('buffer', 'ht_out'),
                ('parameter', 'store_ht'),
                ('buffer', 'seq_order'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_d822b3616306ebda721f_kernel.cu', 'sm_103a/cake_kda_chunk_train_d822b3616306ebda721f_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 107520,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "fwdh_slices": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_2ab2fff7a5263e9f4ca6',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_2ab2fff7a5263e9f4ca6_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_2ab2fff7a5263e9f4ca6',
            "arg_plan": (
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'kg_tma'),
                ('tma_buffer', 'u'),
                ('buffer', 'gk'),
                ('buffer', 'h_out'),
                ('buffer', 'v_new'),
                ('parameter', 'num_heads'),
                ('buffer', 'seq_chunk_start'),
                ('buffer', 'h0'),
                ('parameter', 'use_h0'),
                ('buffer', 'ht_out'),
                ('parameter', 'store_ht'),
                ('buffer', 'seq_order'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_2ab2fff7a5263e9f4ca6_kernel.cu', 'sm_100a/cake_kda_chunk_train_2ab2fff7a5263e9f4ca6_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 87040,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_f68ce9eb3897f734dc73',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_f68ce9eb3897f734dc73_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_f68ce9eb3897f734dc73',
            "arg_plan": (
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'kg_tma'),
                ('tma_buffer', 'u'),
                ('buffer', 'gk'),
                ('buffer', 'h_out'),
                ('buffer', 'v_new'),
                ('parameter', 'num_heads'),
                ('buffer', 'seq_chunk_start'),
                ('buffer', 'h0'),
                ('parameter', 'use_h0'),
                ('buffer', 'ht_out'),
                ('parameter', 'store_ht'),
                ('buffer', 'seq_order'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_f68ce9eb3897f734dc73_kernel.cu', 'sm_103a/cake_kda_chunk_train_f68ce9eb3897f734dc73_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 87040,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "dav": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_0a36097711803e9e0e62',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_0a36097711803e9e0e62_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_0a36097711803e9e0e62',
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
            "sources": ('sm_100a/cake_kda_chunk_train_0a36097711803e9e0e62_kernel.cu', 'sm_100a/cake_kda_chunk_train_0a36097711803e9e0e62_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_8d0f775689402de93103',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_8d0f775689402de93103_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_8d0f775689402de93103',
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
            "sources": ('sm_103a/cake_kda_chunk_train_8d0f775689402de93103_kernel.cu', 'sm_103a/cake_kda_chunk_train_8d0f775689402de93103_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 99328,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "dhu": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_a6e7fec0bd7abf4f8b9b',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_a6e7fec0bd7abf4f8b9b_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_a6e7fec0bd7abf4f8b9b',
            "arg_plan": (
                ('tma_buffer', 'kg_tma'),
                ('tma_buffer', 'qg_tma'),
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'do_tma'),
                ('tma_buffer', 'dv1'),
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
                ('buffer', 'seq_order'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_a6e7fec0bd7abf4f8b9b_kernel.cu', 'sm_100a/cake_kda_chunk_train_a6e7fec0bd7abf4f8b9b_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 156672,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_dcb67494a3dc9297a9af',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_dcb67494a3dc9297a9af_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_dcb67494a3dc9297a9af',
            "arg_plan": (
                ('tma_buffer', 'kg_tma'),
                ('tma_buffer', 'qg_tma'),
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'do_tma'),
                ('tma_buffer', 'dv1'),
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
                ('buffer', 'seq_order'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_dcb67494a3dc9297a9af_kernel.cu', 'sm_103a/cake_kda_chunk_train_dcb67494a3dc9297a9af_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 156672,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "dhu_slices": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_0a288150218d7841deec',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_0a288150218d7841deec_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_0a288150218d7841deec',
            "arg_plan": (
                ('tma_buffer', 'kg_tma'),
                ('tma_buffer', 'qg_tma'),
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'do_tma'),
                ('tma_buffer', 'dv1'),
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
                ('buffer', 'seq_order'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_100a/cake_kda_chunk_train_0a288150218d7841deec_kernel.cu', 'sm_100a/cake_kda_chunk_train_0a288150218d7841deec_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 128000,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_ab8fa57cfd3ea6694431',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_ab8fa57cfd3ea6694431_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_ab8fa57cfd3ea6694431',
            "arg_plan": (
                ('tma_buffer', 'kg_tma'),
                ('tma_buffer', 'qg_tma'),
                ('tma_buffer', 'w_tma'),
                ('tma_buffer', 'do_tma'),
                ('tma_buffer', 'dv1'),
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
                ('buffer', 'seq_order'),
                ('grid', 'grid_x'),
                ('grid', 'grid_y'),
                ('grid', 'grid_z'),
            ),
            "sources": ('sm_103a/cake_kda_chunk_train_ab8fa57cfd3ea6694431_kernel.cu', 'sm_103a/cake_kda_chunk_train_ab8fa57cfd3ea6694431_binding.cu'),
            "block": (192, 1, 1),
            "dynamic_smem_bytes": 128000,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
    },
    "dqkg": {
        "sm_100a": {
            "name": 'cake_kda_chunk_train_248d3768bc898d7ef999',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_248d3768bc898d7ef999_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_248d3768bc898d7ef999',
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
                ('tma_buffer', 'dk_tma'),
                ('tma_buffer', 'dg_tma'),
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
            "sources": ('sm_100a/cake_kda_chunk_train_248d3768bc898d7ef999_kernel.cu', 'sm_100a/cake_kda_chunk_train_248d3768bc898d7ef999_binding.cu'),
            "block": (512, 1, 1),
            "dynamic_smem_bytes": 232192,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_cbee91362344530999af',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_cbee91362344530999af_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_cbee91362344530999af',
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
                ('tma_buffer', 'dk_tma'),
                ('tma_buffer', 'dg_tma'),
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
            "sources": ('sm_103a/cake_kda_chunk_train_cbee91362344530999af_kernel.cu', 'sm_103a/cake_kda_chunk_train_cbee91362344530999af_binding.cu'),
            "block": (512, 1, 1),
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
            "name": 'cake_kda_chunk_train_bc64bdfcc9881a76a5f8',
            "arch": 'sm_100a',
            "cache_name": 'cake_kda_chunk_train_bc64bdfcc9881a76a5f8_sm_100a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_bc64bdfcc9881a76a5f8',
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
            "sources": ('sm_100a/cake_kda_chunk_train_bc64bdfcc9881a76a5f8_kernel.cu', 'sm_100a/cake_kda_chunk_train_bc64bdfcc9881a76a5f8_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 175360,
            "ffi_entry": 'run',
            "compile_flags": ('--use_fast_math',),
        },
        "sm_103a": {
            "name": 'cake_kda_chunk_train_ffc657e8d5d517bcf526',
            "arch": 'sm_103a',
            "cache_name": 'cake_kda_chunk_train_ffc657e8d5d517bcf526_sm_103a',
            "kernel_symbol": 'kernel_cake_kda_chunk_train_ffc657e8d5d517bcf526',
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
            "sources": ('sm_103a/cake_kda_chunk_train_ffc657e8d5d517bcf526_kernel.cu', 'sm_103a/cake_kda_chunk_train_ffc657e8d5d517bcf526_binding.cu'),
            "block": (256, 1, 1),
            "dynamic_smem_bytes": 175360,
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
        self._call = getattr(
            native, record["ffi_entry"]
        )  # config-access-exempt: the build record names the native module's FFI entry point
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
