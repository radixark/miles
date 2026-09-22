/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_TOTALS_OFF 0
#define SMEM_TOTALS_STAGE_BYTES 4096
#define SMEM_TOTALS_STRIDE 4096
#define SMEM_TOTAL 4096
#define THREADS 1024

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(1024) void
kernel_cake_kda_chunk_train_ff540e4f27805de6a473(__nv_bfloat16* __restrict__ g_raw, __nv_bfloat16* __restrict__ q_norm, __nv_bfloat16* __restrict__ k_norm, __nv_bfloat16* __restrict__ v, float* __restrict__ beta, float* __restrict__ A_log, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ aqk, __nv_bfloat16* __restrict__ akk, __nv_bfloat16* __restrict__ do_, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float* __restrict__ gk_out, __nv_bfloat16* __restrict__ vb_out, __nv_bfloat16* __restrict__ kb_out, __nv_bfloat16* __restrict__ qg_out, __nv_bfloat16* __restrict__ kg_out, __nv_bfloat16* __restrict__ ke_out, __nv_bfloat16* __restrict__ qe_out, __nv_bfloat16* __restrict__ aqk_tril, __nv_bfloat16* __restrict__ akk_int, __nv_bfloat16* __restrict__ do_int, __nv_bfloat16* __restrict__ v_int, float* __restrict__ beta_int, int num_qk_heads, int num_heads, int group, int copy_inputs, float lower_bound)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* totals = reinterpret_cast<float*>(smem_raw + 0);
    const int totals_addr = smem + 0;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int head = blockIdx.y;
    int qk_head = head / group;
    int dim = tid % 128;
    int tgrp = tid / 128;
    long long ext0 = (long long)chunk_bos[chunk];
    int clen = chunk_len[chunk];
    long long int0 = (long long)chunk * 64;
    int clast = clen - 1;
    if (clast < 0) {
        clast = 0;
    }
    float _expf_0 = __expf(A_log[head]);
    float rate = _expf_0;
    float bias = dt_bias[head * 128 + dim];
    float gates[8];
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        int t_u = tgrp * 8 + u;
        int t_c = t_u;
        if (t_u > clast) {
            t_c = clast;
        }
        long long idx_u = ((ext0 + (long long)t_c) * (long long)num_heads + (long long)head) * 128 + (long long)dim;
        float x_u = (float)g_raw[idx_u] + bias;
        float _expf_1 = __expf(-(rate * x_u));
        float _rcp_0 = approx_rcp(1.0f + _expf_1);
        float gate_l = lower_bound * _rcp_0;
        float gate_u = 0.0f;
        if (t_u < clen) {
            gate_u = gate_l;
        }
        gates[u] = gate_u;
    }
    float local = 0.0f;
    #pragma unroll
    for (int u_1 = 0; u_1 < 8; u_1++) {
        local = local + gates[u_1];
        gates[u_1] = local;
    }
    totals[tgrp * 128 + dim] = local;
    __syncthreads();
    float offset = 0.0f;
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        float tg = totals[g * 128 + dim];
        if (tgrp > g) {
            offset = offset + tg;
        }
    }
    float total = 0.0f;
    #pragma unroll
    for (int g_1 = 0; g_1 < 8; g_1++) {
        total = total + totals[g_1 * 128 + dim];
    }
    float gk_n = total * 1.4426950408889634f;
    #pragma unroll
    for (int u_2 = 0; u_2 < 8; u_2++) {
        int t_u_1 = tgrp * 8 + u_2;
        int t_c_1 = t_u_1;
        if (t_u_1 > clast) {
            t_c_1 = clast;
        }
        long long irow = int0 + (long long)t_u_1;
        long long iidx = (irow * (long long)num_heads + (long long)head) * 128 + (long long)dim;
        float gkt = (offset + gates[u_2]) * 1.4426950408889634f;
        gk_out[iidx] = gkt;
        float _exp2_0 = approx_exp2(gkt);
        float e_g = _exp2_0;
        long long erow = ext0 + (long long)t_c_1;
        long long eidx = (erow * (long long)num_heads + (long long)head) * 128 + (long long)dim;
        long long qk_u = (erow * (long long)num_qk_heads + (long long)qk_head) * 128 + (long long)dim;
        float bt_l = beta[erow * (long long)num_heads + (long long)head];
        float kv_l = (float)k_norm[qk_u];
        float qv_l = (float)q_norm[qk_u];
        float vv_l = (float)v[eidx];
        float bt = 0.0f;
        float kv = 0.0f;
        float qv = 0.0f;
        float vv = 0.0f;
        if (t_u_1 < clen) {
            bt = bt_l;
            kv = kv_l;
            qv = qv_l;
            vv = vv_l;
        }
        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(vv * bt);
        vb_out[iidx] = _cvt_bf16_0;
        __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(kv * bt);
        float _cvt_f32_0 = __bfloat162float(_cvt_bf16_1);
        __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_cvt_f32_0 * e_g);
        kb_out[iidx] = _cvt_bf16_2;
        __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(qv * e_g);
        qg_out[iidx] = _cvt_bf16_3;
        float _exp2_1 = approx_exp2(gk_n - gkt);
        __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(kv * _exp2_1);
        kg_out[iidx] = _cvt_bf16_4;
        __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(kv);
        ke_out[iidx] = _cvt_bf16_5;
        __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(qv);
        qe_out[iidx] = _cvt_bf16_6;
    }
    if (copy_inputs != 0) {
        #pragma unroll
        for (int u_3 = 0; u_3 < 8; u_3++) {
            int t_u_2 = tgrp * 8 + u_3;
            int t_c_2 = t_u_2;
            if (t_u_2 > clast) {
                t_c_2 = clast;
            }
            long long irow_1 = int0 + (long long)t_u_2;
            long long iidx_1 = (irow_1 * (long long)num_heads + (long long)head) * 128 + (long long)dim;
            long long erow_1 = ext0 + (long long)t_c_2;
            long long eidx_1 = (erow_1 * (long long)num_heads + (long long)head) * 128 + (long long)dim;
            float vc_l = (float)v[eidx_1];
            float do_l = (float)do_[eidx_1];
            float bc_l = beta[erow_1 * (long long)num_heads + (long long)head];
            float vc = 0.0f;
            float dc = 0.0f;
            float bc = 0.0f;
            if (t_u_2 < clen) {
                vc = vc_l;
                dc = do_l;
                bc = bc_l;
            }
            __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(vc);
            v_int[iidx_1] = _cvt_bf16_7;
            __nv_bfloat16 _cvt_bf16_8 = __float2bfloat16(dc);
            do_int[iidx_1] = _cvt_bf16_8;
            if (dim == 0) {
                beta_int[irow_1 * (long long)num_heads + (long long)head] = bc;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int e = r * 1024 + tid;
        int t_a = e / 64;
        int s_a = e - t_a * 64;
        int t_ac = t_a;
        if (t_a > clast) {
            t_ac = clast;
        }
        long long iidx_a = ((int0 + (long long)t_a) * (long long)num_heads + (long long)head) * 64 + (long long)s_a;
        long long eidx_a = ((ext0 + (long long)t_ac) * (long long)num_heads + (long long)head) * 64 + (long long)s_a;
        __nv_bfloat16 raw = aqk[eidx_a];
        __nv_bfloat16 _cvt_bf16_9 = __float2bfloat16(0.0f);
        __nv_bfloat16 val = _cvt_bf16_9;
        if (t_a < clen) {
            if (s_a <= t_a) {
                val = raw;
            }
        }
        aqk_tril[iidx_a] = val;
    }
    if (copy_inputs != 0) {
        #pragma unroll
        for (int r_1 = 0; r_1 < 4; r_1++) {
            int e_1 = r_1 * 1024 + tid;
            int t_a_1 = e_1 / 64;
            int s_a_1 = e_1 - t_a_1 * 64;
            int t_ac_1 = t_a_1;
            if (t_a_1 > clast) {
                t_ac_1 = clast;
            }
            long long iidx_a_1 = ((int0 + (long long)t_a_1) * (long long)num_heads + (long long)head) * 64 + (long long)s_a_1;
            long long eidx_a_1 = ((ext0 + (long long)t_ac_1) * (long long)num_heads + (long long)head) * 64 + (long long)s_a_1;
            __nv_bfloat16 rawk = akk[eidx_a_1];
            __nv_bfloat16 _cvt_bf16_10 = __float2bfloat16(0.0f);
            __nv_bfloat16 valk = _cvt_bf16_10;
            if (t_a_1 < clen) {
                valk = rawk;
            }
            akk_int[iidx_a_1] = valk;
        }
    }
}

} // extern "C"
