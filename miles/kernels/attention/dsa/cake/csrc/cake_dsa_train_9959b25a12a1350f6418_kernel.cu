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
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
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

__global__ __launch_bounds__(THREADS) void
kernel_cake_dsa_train_9959b25a12a1350f6418(float* __restrict__ partial, int* __restrict__ order, int* __restrict__ seg_start, int* __restrict__ seg_count, float* __restrict__ out, int dim, int keys_per_cta, int num_out)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

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
            int groups = count / 16;
            int tail0 = start + groups * 16;
            int tail1 = start + count;
            float acc[4];
            float t0[4];
            float t1[4];
            float t2[4];
            float t3[4];
            float t4[4];
            float t5[4];
            float t6[4];
            float t7[4];
            float t8[4];
            float t9[4];
            float t10[4];
            float t11[4];
            float t12[4];
            float t13[4];
            float t14[4];
            float t15[4];
            int r[16];
            #pragma unroll 1
            for (int it = 0; it < chunk_iters; it++) {
                int c = it * lanes + lane_c;
                if (c < chunks) {
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    if (groups > 0) {
                        r[0] = order[start];
                        r[1] = order[start + 1];
                        r[2] = order[start + 2];
                        r[3] = order[start + 3];
                        r[4] = order[start + 4];
                        r[5] = order[start + 5];
                        r[6] = order[start + 6];
                        r[7] = order[start + 7];
                        r[8] = order[start + 8];
                        r[9] = order[start + 9];
                        r[10] = order[start + 10];
                        r[11] = order[start + 11];
                        r[12] = order[start + 12];
                        r[13] = order[start + 13];
                        r[14] = order[start + 14];
                        r[15] = order[start + 15];
                    }
                    #pragma unroll 1
                    for (int g = 0; g < groups; g++) {
                        int r0 = r[0];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r0 * (long long)dim + (long long)(c * 4));
                            t0[0 + 0] = _v4.x;
                            t0[0 + 1] = _v4.y;
                            t0[0 + 2] = _v4.z;
                            t0[0 + 3] = _v4.w;
                        }
                        int r1 = r[1];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r1 * (long long)dim + (long long)(c * 4));
                            t1[0 + 0] = _v4.x;
                            t1[0 + 1] = _v4.y;
                            t1[0 + 2] = _v4.z;
                            t1[0 + 3] = _v4.w;
                        }
                        int r2 = r[2];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r2 * (long long)dim + (long long)(c * 4));
                            t2[0 + 0] = _v4.x;
                            t2[0 + 1] = _v4.y;
                            t2[0 + 2] = _v4.z;
                            t2[0 + 3] = _v4.w;
                        }
                        int r3 = r[3];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r3 * (long long)dim + (long long)(c * 4));
                            t3[0 + 0] = _v4.x;
                            t3[0 + 1] = _v4.y;
                            t3[0 + 2] = _v4.z;
                            t3[0 + 3] = _v4.w;
                        }
                        int r4 = r[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r4 * (long long)dim + (long long)(c * 4));
                            t4[0 + 0] = _v4.x;
                            t4[0 + 1] = _v4.y;
                            t4[0 + 2] = _v4.z;
                            t4[0 + 3] = _v4.w;
                        }
                        int r5 = r[5];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r5 * (long long)dim + (long long)(c * 4));
                            t5[0 + 0] = _v4.x;
                            t5[0 + 1] = _v4.y;
                            t5[0 + 2] = _v4.z;
                            t5[0 + 3] = _v4.w;
                        }
                        int r6 = r[6];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r6 * (long long)dim + (long long)(c * 4));
                            t6[0 + 0] = _v4.x;
                            t6[0 + 1] = _v4.y;
                            t6[0 + 2] = _v4.z;
                            t6[0 + 3] = _v4.w;
                        }
                        int r7 = r[7];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r7 * (long long)dim + (long long)(c * 4));
                            t7[0 + 0] = _v4.x;
                            t7[0 + 1] = _v4.y;
                            t7[0 + 2] = _v4.z;
                            t7[0 + 3] = _v4.w;
                        }
                        int r8 = r[8];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r8 * (long long)dim + (long long)(c * 4));
                            t8[0 + 0] = _v4.x;
                            t8[0 + 1] = _v4.y;
                            t8[0 + 2] = _v4.z;
                            t8[0 + 3] = _v4.w;
                        }
                        int r9 = r[9];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r9 * (long long)dim + (long long)(c * 4));
                            t9[0 + 0] = _v4.x;
                            t9[0 + 1] = _v4.y;
                            t9[0 + 2] = _v4.z;
                            t9[0 + 3] = _v4.w;
                        }
                        int r10 = r[10];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r10 * (long long)dim + (long long)(c * 4));
                            t10[0 + 0] = _v4.x;
                            t10[0 + 1] = _v4.y;
                            t10[0 + 2] = _v4.z;
                            t10[0 + 3] = _v4.w;
                        }
                        int r11 = r[11];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r11 * (long long)dim + (long long)(c * 4));
                            t11[0 + 0] = _v4.x;
                            t11[0 + 1] = _v4.y;
                            t11[0 + 2] = _v4.z;
                            t11[0 + 3] = _v4.w;
                        }
                        int r12 = r[12];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r12 * (long long)dim + (long long)(c * 4));
                            t12[0 + 0] = _v4.x;
                            t12[0 + 1] = _v4.y;
                            t12[0 + 2] = _v4.z;
                            t12[0 + 3] = _v4.w;
                        }
                        int r13 = r[13];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r13 * (long long)dim + (long long)(c * 4));
                            t13[0 + 0] = _v4.x;
                            t13[0 + 1] = _v4.y;
                            t13[0 + 2] = _v4.z;
                            t13[0 + 3] = _v4.w;
                        }
                        int r14 = r[14];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r14 * (long long)dim + (long long)(c * 4));
                            t14[0 + 0] = _v4.x;
                            t14[0 + 1] = _v4.y;
                            t14[0 + 2] = _v4.z;
                            t14[0 + 3] = _v4.w;
                        }
                        int r15 = r[15];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial + (long long)r15 * (long long)dim + (long long)(c * 4));
                            t15[0 + 0] = _v4.x;
                            t15[0 + 1] = _v4.y;
                            t15[0 + 2] = _v4.z;
                            t15[0 + 3] = _v4.w;
                        }
                        if (groups > g + 1) {
                            int nb = start + (g + 1) * 16;
                            r[0] = order[nb];
                            r[1] = order[nb + 1];
                            r[2] = order[nb + 2];
                            r[3] = order[nb + 3];
                            r[4] = order[nb + 4];
                            r[5] = order[nb + 5];
                            r[6] = order[nb + 6];
                            r[7] = order[nb + 7];
                            r[8] = order[nb + 8];
                            r[9] = order[nb + 9];
                            r[10] = order[nb + 10];
                            r[11] = order[nb + 11];
                            r[12] = order[nb + 12];
                            r[13] = order[nb + 13];
                            r[14] = order[nb + 14];
                            r[15] = order[nb + 15];
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
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t4[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t5[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t6[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t7[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t8[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t9[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t10[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t11[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t12[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t13[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t14[_la];
                        #pragma unroll
                        for (int _la = 0; _la < 4; _la++)
                            acc[_la] = acc[_la] + t15[_la];
                    }
                    #pragma unroll 1
                    for (int pt = tail0; pt < tail1; pt++) {
                        int rt = order[pt];
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
