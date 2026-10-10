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
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_gdn_chunk_train_4423345a9505351dc8de(__nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, float* __restrict__ rstd_q, float* __restrict__ rstd_k, __nv_bfloat16* __restrict__ dq_hv, __nv_bfloat16* __restrict__ dk_hv, __nv_bfloat16* __restrict__ dk2, float* __restrict__ dg1, float* __restrict__ dg2, __nv_bfloat16* __restrict__ dq_out, __nv_bfloat16* __restrict__ dk_out, float* __restrict__ dg_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, int normalize_qk)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int hv = blockIdx.y;
    int tok0 = chunk_start[chunk];
    int n_valid = chunk_len[chunk];
    int group = num_v_heads / num_heads;
    int hq = hv / group;
    int lane_0 = lane;
    if (hv - hq * group == 0) {
        float dq_frag[4];
        float dk_frag[4];
        float pair[4];
        #pragma unroll
        for (int r = 0; r < 16; r++) {
            int row = warp * 16 + r;
            if (row < n_valid) {
                long long tok = (long long)(tok0 + row);
                dq_frag[0] = 0.0f;
                dq_frag[1] = 0.0f;
                dq_frag[2] = 0.0f;
                dq_frag[3] = 0.0f;
                dk_frag[0] = 0.0f;
                dk_frag[1] = 0.0f;
                dk_frag[2] = 0.0f;
                dk_frag[3] = 0.0f;
                #pragma unroll 1
                for (int gi = 0; gi < group; gi++) {
                    long long vbase = (tok * (long long)num_v_heads + (long long)(hq * group + gi)) * 128 + (long long)(lane_0 * 4);
                    {
                        uint2 _vld_0;
                        _vld_0 = *reinterpret_cast<const uint2*>(dq_hv + vbase);
                        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&pair[0 + _pair * 2])[0]), "=f"((&pair[0 + _pair * 2])[1])
                                : "r"(_vpairs_0[_pair]));
                        }
                    }
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        dq_frag[_la] = dq_frag[_la] + pair[_la];
                    {
                        uint2 _vld_1;
                        _vld_1 = *reinterpret_cast<const uint2*>(dk_hv + vbase);
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&pair[0 + _pair * 2])[0]), "=f"((&pair[0 + _pair * 2])[1])
                                : "r"(_vpairs_1[_pair]));
                        }
                    }
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        dk_frag[_la] = dk_frag[_la] + pair[_la];
                    {
                        uint2 _vld_2;
                        _vld_2 = *reinterpret_cast<const uint2*>(dk2 + vbase);
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
                        #pragma unroll
                        for (int _pair = 0; _pair < 2; _pair++) {
                            asm volatile(
                                "{\n\t"
                                "shl.b32 %0, %2, 16;\n\t"
                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                "}\n"
                                : "=f"((&pair[0 + _pair * 2])[0]), "=f"((&pair[0 + _pair * 2])[1])
                                : "r"(_vpairs_2[_pair]));
                        }
                    }
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        dk_frag[_la] = dk_frag[_la] + pair[_la];
                }
                long long qbase = (tok * (long long)num_heads + (long long)hq) * 128 + (long long)(lane_0 * 4);
                float y_q[4];
                float y_k[4];
                {
                    uint2 _vld_3;
                    _vld_3 = *reinterpret_cast<const uint2*>(qn + qbase);
                    uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&y_q[0 + _pair * 2])[0]), "=f"((&y_q[0 + _pair * 2])[1])
                            : "r"(_vpairs_3[_pair]));
                    }
                }
                {
                    uint2 _vld_4;
                    _vld_4 = *reinterpret_cast<const uint2*>(kn + qbase);
                    uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&y_k[0 + _pair * 2])[0]), "=f"((&y_k[0 + _pair * 2])[1])
                            : "r"(_vpairs_4[_pair]));
                    }
                }
                float inner_q = 0.0f;
                float inner_k = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    float _fma_0 = __fmaf_rn(dq_frag[i], y_q[i], inner_q);
                    inner_q = _fma_0;
                    float _fma_1 = __fmaf_rn(dk_frag[i], y_k[i], inner_k);
                    inner_k = _fma_1;
                }
                float _warp_reduce_0 = inner_q;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                inner_q = _warp_reduce_0;
                float _warp_reduce_1 = inner_k;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                inner_k = _warp_reduce_1;
                float rq = rstd_q[tok * (long long)num_heads + (long long)hq];
                float rk = rstd_k[tok * (long long)num_heads + (long long)hq];
                if (normalize_qk != 0) {
                    #pragma unroll
                    for (int i2 = 0; i2 < 4; i2++) {
                        dq_frag[i2] = dq_frag[i2] * rq - inner_q * y_q[i2] * rq;
                        dk_frag[i2] = dk_frag[i2] * rk - inner_k * y_k[i2] * rk;
                    }
                }
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(dq_frag[0 + 0], dq_frag[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(dq_frag[0 + 2], dq_frag[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(dq_out))[qbase]) = _pk2;
                }
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(dk_frag[0 + 0], dk_frag[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(dk_frag[0 + 2], dk_frag[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(dk_out))[qbase]) = _pk2;
                }
            }
        }
    }
    if (warp == 0) {
        float d_lo = 0.0f;
        float d_hi = 0.0f;
        long long gate_index_lo = (long long)(tok0 + lane_0) * (long long)num_v_heads + (long long)hv;
        long long gate_index_hi = (long long)(tok0 + lane_0 + 32) * (long long)num_v_heads + (long long)hv;
        if (lane_0 < n_valid) {
            d_lo = dg1[gate_index_lo] + dg2[gate_index_lo];
        }
        if (n_valid > lane_0 + 32) {
            d_hi = dg1[gate_index_hi] + dg2[gate_index_hi];
        }
        float self_lo = d_lo;
        float self_hi = d_hi;
        #pragma unroll
        for (int delta = 0; delta < 5; delta++) {
            const int shift = 1 << delta;
            float _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, d_lo, shift, 32);
            float up_lo = _shfl_up_0;
            float _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, d_hi, shift, 32);
            float up_hi = _shfl_up_1;
            if (lane_0 >= shift) {
                d_lo = d_lo + up_lo;
                d_hi = d_hi + up_hi;
            }
        }
        float _shfl_0;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(d_lo), "r"(31));
        float total_lo = _shfl_0;
        float _shfl_1;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_1) : "f"(d_hi), "r"(31));
        float total_hi = _shfl_1;
        float total = total_lo + total_hi;
        if (lane_0 < n_valid) {
            dg_out[gate_index_lo] = total - d_lo + self_lo;
        }
        if (n_valid > lane_0 + 32) {
            dg_out[gate_index_hi] = total - total_lo - d_hi + self_hi;
        }
    }
}

} // extern "C"
