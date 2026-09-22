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
kernel_cake_dsa_train_62f0409153cad182a75c(float* __restrict__ partial, int* __restrict__ order, int* __restrict__ seg_start, int* __restrict__ seg_count, float* __restrict__ out, int dim, int keys_per_cta, int num_out)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int tid_0 = tid;
    int lanes = 128 / keys_per_cta;
    int key = blockIdx.x * keys_per_cta + tid_0 / lanes;
    int lane_c = tid_0 % lanes;
    if (key < num_out) {
        int count = seg_count[key];
        if (count > 0) {
            int start = seg_start[key];
            int chunks = dim / 4;
            int chunk_iters = (chunks + lanes - 1) / lanes;
            int quads = count / 4;
            float acc[4];
            float t0[4];
            float t1[4];
            float t2[4];
            float t3[4];
            #pragma unroll 1
            for (int it = 0; it < chunk_iters; it++) {
                int c = it * lanes + lane_c;
                if (c < chunks) {
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    #pragma unroll 1
                    for (int pq = 0; pq < quads; pq++) {
                        int base = start + pq * 4;
                        int r0 = order[base];
                        int r1 = order[base + 1];
                        int r2 = order[base + 2];
                        int r3 = order[base + 3];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r0 * (long long)dim + (long long)(c * 4));
                            t0[0 + 0] = _v4.x;
                            t0[0 + 1] = _v4.y;
                            t0[0 + 2] = _v4.z;
                            t0[0 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r1 * (long long)dim + (long long)(c * 4));
                            t1[0 + 0] = _v4.x;
                            t1[0 + 1] = _v4.y;
                            t1[0 + 2] = _v4.z;
                            t1[0 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r2 * (long long)dim + (long long)(c * 4));
                            t2[0 + 0] = _v4.x;
                            t2[0 + 1] = _v4.y;
                            t2[0 + 2] = _v4.z;
                            t2[0 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r3 * (long long)dim + (long long)(c * 4));
                            t3[0 + 0] = _v4.x;
                            t3[0 + 1] = _v4.y;
                            t3[0 + 2] = _v4.z;
                            t3[0 + 3] = _v4.w;
                        }
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t0[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t1[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t2[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t3[_la];
                    }
                    #pragma unroll 1
                    for (int pt = quads * 4; pt < count; pt++) {
                        int rt = order[start + pt];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)rt * (long long)dim + (long long)(c * 4));
                            t0[0 + 0] = _v4.x;
                            t0[0 + 1] = _v4.y;
                            t0[0 + 2] = _v4.z;
                            t0[0 + 3] = _v4.w;
                        }
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t0[_la];
                    }
                    long long out_index = (long long)key * (long long)dim + (long long)(c * 4);
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(out + out_index);
                        t0[0 + 0] = _v4.x;
                        t0[0 + 1] = _v4.y;
                        t0[0 + 2] = _v4.z;
                        t0[0 + 3] = _v4.w;
                    }
                    #pragma unroll
                    for (int _la = 0; _la < 4; _la++)
                        acc[_la] = acc[_la] + t0[_la];
                    {
                        float4 _v4 = make_float4(acc[0 + 0], acc[0 + 1], acc[0 + 2], acc[0 + 3]);
                        *reinterpret_cast<float4*>(out + out_index) = _v4;
                    }
                }
            }
        }
    }
}

} // extern "C"
