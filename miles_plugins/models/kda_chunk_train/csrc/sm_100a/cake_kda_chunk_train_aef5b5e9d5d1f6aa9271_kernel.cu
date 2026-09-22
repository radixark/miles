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
#define SMEM_PART_A_S_OFF 4096
#define SMEM_PART_A_S_STAGE_BYTES 4096
#define SMEM_PART_A_S_STRIDE 4096
#define SMEM_PART_B_S_OFF 8192
#define SMEM_PART_B_S_STAGE_BYTES 4096
#define SMEM_PART_B_S_STRIDE 4096
#define SMEM_TOTAL 12288
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(256, 4) void
kernel_cake_kda_chunk_train_aef5b5e9d5d1f6aa9271(float* __restrict__ dg_intra, __nv_bfloat16* __restrict__ g_raw, float* __restrict__ db_total, __nv_bfloat16* __restrict__ beta_raw, float* __restrict__ A_log, float* __restrict__ dt_bias, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ dg_out, __nv_bfloat16* __restrict__ dbeta, float* __restrict__ dA_part, float* __restrict__ dbias_part, int num_heads, float lower_bound)
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
    float* part_a_s = reinterpret_cast<float*>(smem_raw + 4096);
    const int part_a_s_addr = smem + 4096;
    float* part_b_s = reinterpret_cast<float*>(smem_raw + 8192);
    const int part_b_s_addr = smem + 8192;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int head = blockIdx.y;
    int tgrp = tid / 32;
    int quad = tid - tgrp * 32;
    int d0 = quad * 4;
    long long row0 = (long long)chunk * 64;
    long long ext0 = (long long)chunk_bos[chunk];
    int clen = chunk_len[chunk];
    float _expf_0 = __expf(A_log[head]);
    float rate = _expf_0;
    float bias[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(dt_bias + (long long)(head * 128 + d0));
        bias[0 + 0] = _v4.x;
        bias[0 + 1] = _v4.y;
        bias[0 + 2] = _v4.z;
        bias[0 + 3] = _v4.w;
    }
    int rs = num_heads * 128;
    int t0 = tgrp * 8;
    long long ibase = ((row0 + (long long)t0) * (long long)num_heads + (long long)head) * 128 + (long long)d0;
    long long ebase = ((ext0 + (long long)t0) * (long long)num_heads + (long long)head) * 128 + (long long)d0;
    int sq = d0 * 4;
    float dgs[32];
    #pragma unroll
    for (int u = 0; u < 8; u++) {
        {
            float4 _v4 = *reinterpret_cast<const float4*>(dg_intra + ibase + (long long)(rs * u));
            dgs[u * 4 + 0] = _v4.x;
            dgs[u * 4 + 1] = _v4.y;
            dgs[u * 4 + 2] = _v4.z;
            dgs[u * 4 + 3] = _v4.w;
        }
    }
    int tok_b = tid - 32;
    if (tok_b >= 0) {
        if (tok_b < clen) {
            long long ebeta = (ext0 + (long long)tok_b) * (long long)num_heads + (long long)head;
            long long ibeta = (row0 + (long long)tok_b) * (long long)num_heads + (long long)head;
            float _expf_1 = __expf(-(float)beta_raw[ebeta]);
            float _rcp_0 = approx_rcp(1.0f + _expf_1);
            float sb = _rcp_0;
            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(db_total[ibeta] * sb * (1.0f - sb));
            dbeta[ebeta] = _cvt_bf16_0;
        }
    }
    float suffix[4];
    suffix[0] = 0.0f;
    suffix[1] = 0.0f;
    suffix[2] = 0.0f;
    suffix[3] = 0.0f;
    #pragma unroll
    for (int uu = 0; uu < 8; uu++) {
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            suffix[e] = suffix[e] + dgs[(7 - uu) * 4 + e];
            dgs[(7 - uu) * 4 + e] = suffix[e];
        }
    }
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(totals_addr + (unsigned int)(tgrp * 512) + (unsigned int)sq), "f"(suffix[0]), "f"(suffix[1]), "f"(suffix[2]), "f"(suffix[3]) : "memory");
    __syncthreads();
    float offset[4];
    offset[0] = 0.0f;
    offset[1] = 0.0f;
    offset[2] = 0.0f;
    offset[3] = 0.0f;
    #pragma unroll
    for (int gg = 0; gg < 8; gg++) {
        const int g = 7 - gg;
        float tg[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tg[0])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 3]))
            : "r"(totals_addr + (unsigned int)(g * 512) + (unsigned int)sq));
        if (g > tgrp) {
            #pragma unroll
            for (int e_1 = 0; e_1 < 4; e_1++) {
                offset[e_1] = offset[e_1] + tg[e_1];
            }
        }
    }
    #pragma unroll
    for (int u_1 = 0; u_1 < 8; u_1++) {
        #pragma unroll
        for (int e_2 = 0; e_2 < 4; e_2++) {
            dgs[u_1 * 4 + e_2] = offset[e_2] + dgs[u_1 * 4 + e_2];
        }
    }
    float part_a[4];
    float part_b[4];
    part_a[0] = 0.0f;
    part_a[1] = 0.0f;
    part_a[2] = 0.0f;
    part_a[3] = 0.0f;
    part_b[0] = 0.0f;
    part_b[1] = 0.0f;
    part_b[2] = 0.0f;
    part_b[3] = 0.0f;
    #pragma unroll
    for (int u_2 = 0; u_2 < 8; u_2++) {
        int t_u = t0 + u_2;
        long long eidx_u = ebase + (long long)(rs * u_2);
        float graw[4];
        graw[0] = 0.0f;
        graw[1] = 0.0f;
        graw[2] = 0.0f;
        graw[3] = 0.0f;
        if (t_u < clen) {
            {
                uint2 _vld_2;
                _vld_2 = *reinterpret_cast<const uint2*>(g_raw + eidx_u);
                uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&graw[0 + _pair * 2])[0]), "=f"((&graw[0 + _pair * 2])[1])
                        : "r"(_vpairs_2[_pair]));
                }
            }
        }
        float dgo[4];
        #pragma unroll
        for (int e_3 = 0; e_3 < 4; e_3++) {
            float dyg = dgs[u_2 * 4 + e_3];
            float x = 0.0f;
            if (t_u < clen) {
                x = graw[e_3] + bias[e_3];
            }
            float _expf_2 = __expf(-(rate * x));
            float _rcp_1 = approx_rcp(1.0f + _expf_2);
            float sig = _rcp_1;
            float dsig = sig * (1.0f - sig);
            float dg_raw = dyg * (lower_bound * dsig) * rate;
            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(dg_raw);
            __nv_bfloat16 dgb = _cvt_bf16_1;
            float _cvt_f32_0 = __bfloat162float(dgb);
            dgo[e_3] = _cvt_f32_0;
            float _fma_0 = __fmaf_rn(dg_raw, x, part_a[e_3]);
            part_a[e_3] = _fma_0;
            float _cvt_f32_1 = __bfloat162float(dgb);
            part_b[e_3] = part_b[e_3] + _cvt_f32_1;
        }
        if (t_u < clen) {
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(dgo[0 + 0], dgo[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(dgo[0 + 2], dgo[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(dg_out + eidx_u))[0]) = _pk2;
            }
        }
    }
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(part_b_s_addr + (unsigned int)(tgrp * 512) + (unsigned int)sq), "f"(part_b[0]), "f"(part_b[1]), "f"(part_b[2]), "f"(part_b[3]) : "memory");
    asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(part_a_s_addr + (unsigned int)(tgrp * 512) + (unsigned int)sq), "f"(part_a[0]), "f"(part_a[1]), "f"(part_a[2]), "f"(part_a[3]) : "memory");
    __syncthreads();
    if (tgrp == 0) {
        float sum_a[4];
        float sum_b[4];
        sum_a[0] = 0.0f;
        sum_a[1] = 0.0f;
        sum_a[2] = 0.0f;
        sum_a[3] = 0.0f;
        sum_b[0] = 0.0f;
        sum_b[1] = 0.0f;
        sum_b[2] = 0.0f;
        sum_b[3] = 0.0f;
        #pragma unroll
        for (int g_1 = 0; g_1 < 8; g_1++) {
            float pa[4];
            float pb[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&pa[0])), "=r"(*reinterpret_cast<uint32_t*>(&pa[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pa[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pa[(0) + 3]))
                : "r"(part_a_s_addr + (unsigned int)(g_1 * 512) + (unsigned int)sq));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&pb[0])), "=r"(*reinterpret_cast<uint32_t*>(&pb[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pb[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pb[(0) + 3]))
                : "r"(part_b_s_addr + (unsigned int)(g_1 * 512) + (unsigned int)sq));
            #pragma unroll
            for (int e_4 = 0; e_4 < 4; e_4++) {
                sum_a[e_4] = sum_a[e_4] + pa[e_4];
                sum_b[e_4] = sum_b[e_4] + pb[e_4];
            }
        }
        long long part_index = ((long long)chunk * (long long)num_heads + (long long)head) * 128 + (long long)d0;
        {
            float4 _v4 = make_float4(sum_a[0 + 0], sum_a[0 + 1], sum_a[0 + 2], sum_a[0 + 3]);
            *reinterpret_cast<float4*>(dA_part + part_index + 0) = _v4;
        }
        {
            float4 _v4 = make_float4(sum_b[0 + 0], sum_b[0 + 1], sum_b[0 + 2], sum_b[0 + 3]);
            *reinterpret_cast<float4*>(dbias_part + part_index + 0) = _v4;
        }
    }
}

} // extern "C"
