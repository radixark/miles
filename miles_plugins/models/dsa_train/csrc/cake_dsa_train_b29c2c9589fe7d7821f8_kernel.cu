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
#define SMEM_S_Q_OFF 0
#define SMEM_S_Q_STAGE_BYTES 16384
#define SMEM_S_Q_STRIDE 16384
#define SMEM_S_KV_OFF 16384
#define SMEM_S_KV_STAGE_BYTES 65536
#define SMEM_S_KV_STRIDE 65536
#define SMEM_S_P_OFF 147456
#define SMEM_S_P_STAGE_BYTES 2048
#define SMEM_S_P_STRIDE 2048
#define SMEM_S_IDX_OFF 149504
#define SMEM_S_IDX_STAGE_BYTES 512
#define SMEM_S_IDX_STRIDE 512
#define SMEM_S_RED_OFF 150016
#define SMEM_S_RED_STAGE_BYTES 512
#define SMEM_S_RED_STRIDE 512
#define SMEM_TOTAL 150528
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_dsa_train_b29c2c9589fe7d7821f8(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ kv, float* __restrict__ sink, int* __restrict__ indices, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, int heads, int topk, float scale_log2)
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
    __nv_bfloat16* s_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_q_addr = smem + 0;
    __nv_bfloat16* s_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_kv_addr = smem + 16384;
    __nv_bfloat16* s_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 147456);
    const int s_p_addr = smem + 147456;
    int* s_idx = reinterpret_cast<int*>(smem_raw + 149504);
    const int s_idx_addr = smem + 149504;
    float* s_red = reinterpret_cast<float*>(smem_raw + 150016);
    const int s_red_addr = smem + 150016;

    // === Task calls (dependency order) ===
    int row = blockIdx.x;
    int hb = blockIdx.y;
    int tid_0 = tid;
    int lane_1 = lane;
    int warp_2 = warp;
    int hg = warp_2 / 4;
    int kw = warp_2 % 4;
    int h0 = hb * 16;
    int r_a = lane_1 / 4;
    int r_b = r_a + 8;
    int col_q = lane_1 % 4 * 2;
    int num_blocks = topk / 64;
    long long idx_base = (long long)row * (long long)topk;
    int head_a = h0 + hg * 16 + r_a;
    int head_b = head_a + 8;
    float m_a = -1073741824.0f;
    float m_b = -1073741824.0f;
    float l_a = 0.0f;
    float l_b = 0.0f;
    float acc_o[64];
    float acc_s[8];
    unsigned int a_frag[4];
    unsigned int b_frag[4];
    unsigned int b_frag_t[4];
    acc_o[0] = 0.0f;
    acc_o[1] = 0.0f;
    acc_o[2] = 0.0f;
    acc_o[3] = 0.0f;
    acc_o[4] = 0.0f;
    acc_o[5] = 0.0f;
    acc_o[6] = 0.0f;
    acc_o[7] = 0.0f;
    acc_o[8] = 0.0f;
    acc_o[9] = 0.0f;
    acc_o[10] = 0.0f;
    acc_o[11] = 0.0f;
    acc_o[12] = 0.0f;
    acc_o[13] = 0.0f;
    acc_o[14] = 0.0f;
    acc_o[15] = 0.0f;
    acc_o[16] = 0.0f;
    acc_o[17] = 0.0f;
    acc_o[18] = 0.0f;
    acc_o[19] = 0.0f;
    acc_o[20] = 0.0f;
    acc_o[21] = 0.0f;
    acc_o[22] = 0.0f;
    acc_o[23] = 0.0f;
    acc_o[24] = 0.0f;
    acc_o[25] = 0.0f;
    acc_o[26] = 0.0f;
    acc_o[27] = 0.0f;
    acc_o[28] = 0.0f;
    acc_o[29] = 0.0f;
    acc_o[30] = 0.0f;
    acc_o[31] = 0.0f;
    acc_o[32] = 0.0f;
    acc_o[33] = 0.0f;
    acc_o[34] = 0.0f;
    acc_o[35] = 0.0f;
    acc_o[36] = 0.0f;
    acc_o[37] = 0.0f;
    acc_o[38] = 0.0f;
    acc_o[39] = 0.0f;
    acc_o[40] = 0.0f;
    acc_o[41] = 0.0f;
    acc_o[42] = 0.0f;
    acc_o[43] = 0.0f;
    acc_o[44] = 0.0f;
    acc_o[45] = 0.0f;
    acc_o[46] = 0.0f;
    acc_o[47] = 0.0f;
    acc_o[48] = 0.0f;
    acc_o[49] = 0.0f;
    acc_o[50] = 0.0f;
    acc_o[51] = 0.0f;
    acc_o[52] = 0.0f;
    acc_o[53] = 0.0f;
    acc_o[54] = 0.0f;
    acc_o[55] = 0.0f;
    acc_o[56] = 0.0f;
    acc_o[57] = 0.0f;
    acc_o[58] = 0.0f;
    acc_o[59] = 0.0f;
    acc_o[60] = 0.0f;
    acc_o[61] = 0.0f;
    acc_o[62] = 0.0f;
    acc_o[63] = 0.0f;
    if (tid_0 < 64) {
        s_idx[tid_0] = indices[idx_base + (long long)tid_0];
    }
    __syncthreads();
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)(tid % 64 * 8 / 64 * 2048 + (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 ^ (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + tid / 64)) * 512 + (long long)(tid % 64 * 8))), "r"((h0 + tid / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((128 + tid) % 64 * 8 / 64 * 2048 + ((128 + tid) / 64 * 128 + (128 + tid) % 64 * 8 % 64 * 2 ^ ((128 + tid) / 64 * 128 + (128 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (128 + tid) / 64)) * 512 + (long long)((128 + tid) % 64 * 8))), "r"((h0 + (128 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((256 + tid) % 64 * 8 / 64 * 2048 + ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 ^ ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (256 + tid) / 64)) * 512 + (long long)((256 + tid) % 64 * 8))), "r"((h0 + (256 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((384 + tid) % 64 * 8 / 64 * 2048 + ((384 + tid) / 64 * 128 + (384 + tid) % 64 * 8 % 64 * 2 ^ ((384 + tid) / 64 * 128 + (384 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (384 + tid) / 64)) * 512 + (long long)((384 + tid) % 64 * 8))), "r"((h0 + (384 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((512 + tid) % 64 * 8 / 64 * 2048 + ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 ^ ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (512 + tid) / 64)) * 512 + (long long)((512 + tid) % 64 * 8))), "r"((h0 + (512 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((640 + tid) % 64 * 8 / 64 * 2048 + ((640 + tid) / 64 * 128 + (640 + tid) % 64 * 8 % 64 * 2 ^ ((640 + tid) / 64 * 128 + (640 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (640 + tid) / 64)) * 512 + (long long)((640 + tid) % 64 * 8))), "r"((h0 + (640 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((768 + tid) % 64 * 8 / 64 * 2048 + ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 ^ ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (768 + tid) / 64)) * 512 + (long long)((768 + tid) % 64 * 8))), "r"((h0 + (768 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((896 + tid) % 64 * 8 / 64 * 2048 + ((896 + tid) / 64 * 128 + (896 + tid) % 64 * 8 % 64 * 2 ^ ((896 + tid) / 64 * 128 + (896 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (896 + tid) / 64)) * 512 + (long long)((896 + tid) % 64 * 8))), "r"((h0 + (896 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)(tid % 64 * 8 / 64 * 8192 + (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 ^ (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[tid / 64] * 512 + (long long)(tid % 64 * 8))), "r"((s_idx[tid / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((128 + tid) % 64 * 8 / 64 * 8192 + ((128 + tid) / 64 * 128 + (128 + tid) % 64 * 8 % 64 * 2 ^ ((128 + tid) / 64 * 128 + (128 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(128 + tid) / 64] * 512 + (long long)((128 + tid) % 64 * 8))), "r"((s_idx[(128 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((256 + tid) % 64 * 8 / 64 * 8192 + ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 ^ ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(256 + tid) / 64] * 512 + (long long)((256 + tid) % 64 * 8))), "r"((s_idx[(256 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((384 + tid) % 64 * 8 / 64 * 8192 + ((384 + tid) / 64 * 128 + (384 + tid) % 64 * 8 % 64 * 2 ^ ((384 + tid) / 64 * 128 + (384 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(384 + tid) / 64] * 512 + (long long)((384 + tid) % 64 * 8))), "r"((s_idx[(384 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((512 + tid) % 64 * 8 / 64 * 8192 + ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 ^ ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(512 + tid) / 64] * 512 + (long long)((512 + tid) % 64 * 8))), "r"((s_idx[(512 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((640 + tid) % 64 * 8 / 64 * 8192 + ((640 + tid) / 64 * 128 + (640 + tid) % 64 * 8 % 64 * 2 ^ ((640 + tid) / 64 * 128 + (640 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(640 + tid) / 64] * 512 + (long long)((640 + tid) % 64 * 8))), "r"((s_idx[(640 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((768 + tid) % 64 * 8 / 64 * 8192 + ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 ^ ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(768 + tid) / 64] * 512 + (long long)((768 + tid) % 64 * 8))), "r"((s_idx[(768 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((896 + tid) % 64 * 8 / 64 * 8192 + ((896 + tid) / 64 * 128 + (896 + tid) % 64 * 8 % 64 * 2 ^ ((896 + tid) / 64 * 128 + (896 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(896 + tid) / 64] * 512 + (long long)((896 + tid) % 64 * 8))), "r"((s_idx[(896 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((1024 + tid) % 64 * 8 / 64 * 8192 + ((1024 + tid) / 64 * 128 + (1024 + tid) % 64 * 8 % 64 * 2 ^ ((1024 + tid) / 64 * 128 + (1024 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(1024 + tid) / 64] * 512 + (long long)((1024 + tid) % 64 * 8))), "r"((s_idx[(1024 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((1152 + tid) % 64 * 8 / 64 * 8192 + ((1152 + tid) / 64 * 128 + (1152 + tid) % 64 * 8 % 64 * 2 ^ ((1152 + tid) / 64 * 128 + (1152 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(1152 + tid) / 64] * 512 + (long long)((1152 + tid) % 64 * 8))), "r"((s_idx[(1152 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((1280 + tid) % 64 * 8 / 64 * 8192 + ((1280 + tid) / 64 * 128 + (1280 + tid) % 64 * 8 % 64 * 2 ^ ((1280 + tid) / 64 * 128 + (1280 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(1280 + tid) / 64] * 512 + (long long)((1280 + tid) % 64 * 8))), "r"((s_idx[(1280 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((1408 + tid) % 64 * 8 / 64 * 8192 + ((1408 + tid) / 64 * 128 + (1408 + tid) % 64 * 8 % 64 * 2 ^ ((1408 + tid) / 64 * 128 + (1408 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(1408 + tid) / 64] * 512 + (long long)((1408 + tid) % 64 * 8))), "r"((s_idx[(1408 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((1536 + tid) % 64 * 8 / 64 * 8192 + ((1536 + tid) / 64 * 128 + (1536 + tid) % 64 * 8 % 64 * 2 ^ ((1536 + tid) / 64 * 128 + (1536 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(1536 + tid) / 64] * 512 + (long long)((1536 + tid) % 64 * 8))), "r"((s_idx[(1536 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((1664 + tid) % 64 * 8 / 64 * 8192 + ((1664 + tid) / 64 * 128 + (1664 + tid) % 64 * 8 % 64 * 2 ^ ((1664 + tid) / 64 * 128 + (1664 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(1664 + tid) / 64] * 512 + (long long)((1664 + tid) % 64 * 8))), "r"((s_idx[(1664 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((1792 + tid) % 64 * 8 / 64 * 8192 + ((1792 + tid) / 64 * 128 + (1792 + tid) % 64 * 8 % 64 * 2 ^ ((1792 + tid) / 64 * 128 + (1792 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(1792 + tid) / 64] * 512 + (long long)((1792 + tid) % 64 * 8))), "r"((s_idx[(1792 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((1920 + tid) % 64 * 8 / 64 * 8192 + ((1920 + tid) / 64 * 128 + (1920 + tid) % 64 * 8 % 64 * 2 ^ ((1920 + tid) / 64 * 128 + (1920 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(1920 + tid) / 64] * 512 + (long long)((1920 + tid) % 64 * 8))), "r"((s_idx[(1920 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((2048 + tid) % 64 * 8 / 64 * 8192 + ((2048 + tid) / 64 * 128 + (2048 + tid) % 64 * 8 % 64 * 2 ^ ((2048 + tid) / 64 * 128 + (2048 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(2048 + tid) / 64] * 512 + (long long)((2048 + tid) % 64 * 8))), "r"((s_idx[(2048 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((2176 + tid) % 64 * 8 / 64 * 8192 + ((2176 + tid) / 64 * 128 + (2176 + tid) % 64 * 8 % 64 * 2 ^ ((2176 + tid) / 64 * 128 + (2176 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(2176 + tid) / 64] * 512 + (long long)((2176 + tid) % 64 * 8))), "r"((s_idx[(2176 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((2304 + tid) % 64 * 8 / 64 * 8192 + ((2304 + tid) / 64 * 128 + (2304 + tid) % 64 * 8 % 64 * 2 ^ ((2304 + tid) / 64 * 128 + (2304 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(2304 + tid) / 64] * 512 + (long long)((2304 + tid) % 64 * 8))), "r"((s_idx[(2304 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((2432 + tid) % 64 * 8 / 64 * 8192 + ((2432 + tid) / 64 * 128 + (2432 + tid) % 64 * 8 % 64 * 2 ^ ((2432 + tid) / 64 * 128 + (2432 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(2432 + tid) / 64] * 512 + (long long)((2432 + tid) % 64 * 8))), "r"((s_idx[(2432 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((2560 + tid) % 64 * 8 / 64 * 8192 + ((2560 + tid) / 64 * 128 + (2560 + tid) % 64 * 8 % 64 * 2 ^ ((2560 + tid) / 64 * 128 + (2560 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(2560 + tid) / 64] * 512 + (long long)((2560 + tid) % 64 * 8))), "r"((s_idx[(2560 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((2688 + tid) % 64 * 8 / 64 * 8192 + ((2688 + tid) / 64 * 128 + (2688 + tid) % 64 * 8 % 64 * 2 ^ ((2688 + tid) / 64 * 128 + (2688 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(2688 + tid) / 64] * 512 + (long long)((2688 + tid) % 64 * 8))), "r"((s_idx[(2688 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((2816 + tid) % 64 * 8 / 64 * 8192 + ((2816 + tid) / 64 * 128 + (2816 + tid) % 64 * 8 % 64 * 2 ^ ((2816 + tid) / 64 * 128 + (2816 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(2816 + tid) / 64] * 512 + (long long)((2816 + tid) % 64 * 8))), "r"((s_idx[(2816 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((2944 + tid) % 64 * 8 / 64 * 8192 + ((2944 + tid) / 64 * 128 + (2944 + tid) % 64 * 8 % 64 * 2 ^ ((2944 + tid) / 64 * 128 + (2944 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(2944 + tid) / 64] * 512 + (long long)((2944 + tid) % 64 * 8))), "r"((s_idx[(2944 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((3072 + tid) % 64 * 8 / 64 * 8192 + ((3072 + tid) / 64 * 128 + (3072 + tid) % 64 * 8 % 64 * 2 ^ ((3072 + tid) / 64 * 128 + (3072 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(3072 + tid) / 64] * 512 + (long long)((3072 + tid) % 64 * 8))), "r"((s_idx[(3072 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((3200 + tid) % 64 * 8 / 64 * 8192 + ((3200 + tid) / 64 * 128 + (3200 + tid) % 64 * 8 % 64 * 2 ^ ((3200 + tid) / 64 * 128 + (3200 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(3200 + tid) / 64] * 512 + (long long)((3200 + tid) % 64 * 8))), "r"((s_idx[(3200 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((3328 + tid) % 64 * 8 / 64 * 8192 + ((3328 + tid) / 64 * 128 + (3328 + tid) % 64 * 8 % 64 * 2 ^ ((3328 + tid) / 64 * 128 + (3328 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(3328 + tid) / 64] * 512 + (long long)((3328 + tid) % 64 * 8))), "r"((s_idx[(3328 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((3456 + tid) % 64 * 8 / 64 * 8192 + ((3456 + tid) / 64 * 128 + (3456 + tid) % 64 * 8 % 64 * 2 ^ ((3456 + tid) / 64 * 128 + (3456 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(3456 + tid) / 64] * 512 + (long long)((3456 + tid) % 64 * 8))), "r"((s_idx[(3456 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((3584 + tid) % 64 * 8 / 64 * 8192 + ((3584 + tid) / 64 * 128 + (3584 + tid) % 64 * 8 % 64 * 2 ^ ((3584 + tid) / 64 * 128 + (3584 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(3584 + tid) / 64] * 512 + (long long)((3584 + tid) % 64 * 8))), "r"((s_idx[(3584 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((3712 + tid) % 64 * 8 / 64 * 8192 + ((3712 + tid) / 64 * 128 + (3712 + tid) % 64 * 8 % 64 * 2 ^ ((3712 + tid) / 64 * 128 + (3712 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(3712 + tid) / 64] * 512 + (long long)((3712 + tid) % 64 * 8))), "r"((s_idx[(3712 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((3840 + tid) % 64 * 8 / 64 * 8192 + ((3840 + tid) / 64 * 128 + (3840 + tid) % 64 * 8 % 64 * 2 ^ ((3840 + tid) / 64 * 128 + (3840 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(3840 + tid) / 64] * 512 + (long long)((3840 + tid) % 64 * 8))), "r"((s_idx[(3840 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((3968 + tid) % 64 * 8 / 64 * 8192 + ((3968 + tid) / 64 * 128 + (3968 + tid) % 64 * 8 % 64 * 2 ^ ((3968 + tid) / 64 * 128 + (3968 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(3968 + tid) / 64] * 512 + (long long)((3968 + tid) % 64 * 8))), "r"((s_idx[(3968 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    #pragma unroll 1
    for (int blk = 0; blk < num_blocks; blk++) {
        int st = blk % 2;
        int nst = 1 - st;
        if (num_blocks > blk + 1) {
            if (tid_0 < 64) {
                s_idx[nst * 64 + tid_0] = indices[idx_base + (long long)((blk + 1) * 64) + (long long)tid_0];
            }
        }
        __syncthreads();
        if (num_blocks > blk + 1) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)(tid % 64 * 8 / 64 * 8192 + (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 ^ (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + tid / 64] * 512 + (long long)(tid % 64 * 8))), "r"((s_idx[nst * 64 + tid / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((128 + tid) % 64 * 8 / 64 * 8192 + ((128 + tid) / 64 * 128 + (128 + tid) % 64 * 8 % 64 * 2 ^ ((128 + tid) / 64 * 128 + (128 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (128 + tid) / 64] * 512 + (long long)((128 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (128 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((256 + tid) % 64 * 8 / 64 * 8192 + ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 ^ ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (256 + tid) / 64] * 512 + (long long)((256 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (256 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((384 + tid) % 64 * 8 / 64 * 8192 + ((384 + tid) / 64 * 128 + (384 + tid) % 64 * 8 % 64 * 2 ^ ((384 + tid) / 64 * 128 + (384 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (384 + tid) / 64] * 512 + (long long)((384 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (384 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((512 + tid) % 64 * 8 / 64 * 8192 + ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 ^ ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (512 + tid) / 64] * 512 + (long long)((512 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (512 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((640 + tid) % 64 * 8 / 64 * 8192 + ((640 + tid) / 64 * 128 + (640 + tid) % 64 * 8 % 64 * 2 ^ ((640 + tid) / 64 * 128 + (640 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (640 + tid) / 64] * 512 + (long long)((640 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (640 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((768 + tid) % 64 * 8 / 64 * 8192 + ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 ^ ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (768 + tid) / 64] * 512 + (long long)((768 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (768 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((896 + tid) % 64 * 8 / 64 * 8192 + ((896 + tid) / 64 * 128 + (896 + tid) % 64 * 8 % 64 * 2 ^ ((896 + tid) / 64 * 128 + (896 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (896 + tid) / 64] * 512 + (long long)((896 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (896 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((1024 + tid) % 64 * 8 / 64 * 8192 + ((1024 + tid) / 64 * 128 + (1024 + tid) % 64 * 8 % 64 * 2 ^ ((1024 + tid) / 64 * 128 + (1024 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (1024 + tid) / 64] * 512 + (long long)((1024 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (1024 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((1152 + tid) % 64 * 8 / 64 * 8192 + ((1152 + tid) / 64 * 128 + (1152 + tid) % 64 * 8 % 64 * 2 ^ ((1152 + tid) / 64 * 128 + (1152 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (1152 + tid) / 64] * 512 + (long long)((1152 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (1152 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((1280 + tid) % 64 * 8 / 64 * 8192 + ((1280 + tid) / 64 * 128 + (1280 + tid) % 64 * 8 % 64 * 2 ^ ((1280 + tid) / 64 * 128 + (1280 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (1280 + tid) / 64] * 512 + (long long)((1280 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (1280 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((1408 + tid) % 64 * 8 / 64 * 8192 + ((1408 + tid) / 64 * 128 + (1408 + tid) % 64 * 8 % 64 * 2 ^ ((1408 + tid) / 64 * 128 + (1408 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (1408 + tid) / 64] * 512 + (long long)((1408 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (1408 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((1536 + tid) % 64 * 8 / 64 * 8192 + ((1536 + tid) / 64 * 128 + (1536 + tid) % 64 * 8 % 64 * 2 ^ ((1536 + tid) / 64 * 128 + (1536 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (1536 + tid) / 64] * 512 + (long long)((1536 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (1536 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((1664 + tid) % 64 * 8 / 64 * 8192 + ((1664 + tid) / 64 * 128 + (1664 + tid) % 64 * 8 % 64 * 2 ^ ((1664 + tid) / 64 * 128 + (1664 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (1664 + tid) / 64] * 512 + (long long)((1664 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (1664 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((1792 + tid) % 64 * 8 / 64 * 8192 + ((1792 + tid) / 64 * 128 + (1792 + tid) % 64 * 8 % 64 * 2 ^ ((1792 + tid) / 64 * 128 + (1792 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (1792 + tid) / 64] * 512 + (long long)((1792 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (1792 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((1920 + tid) % 64 * 8 / 64 * 8192 + ((1920 + tid) / 64 * 128 + (1920 + tid) % 64 * 8 % 64 * 2 ^ ((1920 + tid) / 64 * 128 + (1920 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (1920 + tid) / 64] * 512 + (long long)((1920 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (1920 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((2048 + tid) % 64 * 8 / 64 * 8192 + ((2048 + tid) / 64 * 128 + (2048 + tid) % 64 * 8 % 64 * 2 ^ ((2048 + tid) / 64 * 128 + (2048 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (2048 + tid) / 64] * 512 + (long long)((2048 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (2048 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((2176 + tid) % 64 * 8 / 64 * 8192 + ((2176 + tid) / 64 * 128 + (2176 + tid) % 64 * 8 % 64 * 2 ^ ((2176 + tid) / 64 * 128 + (2176 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (2176 + tid) / 64] * 512 + (long long)((2176 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (2176 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((2304 + tid) % 64 * 8 / 64 * 8192 + ((2304 + tid) / 64 * 128 + (2304 + tid) % 64 * 8 % 64 * 2 ^ ((2304 + tid) / 64 * 128 + (2304 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (2304 + tid) / 64] * 512 + (long long)((2304 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (2304 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((2432 + tid) % 64 * 8 / 64 * 8192 + ((2432 + tid) / 64 * 128 + (2432 + tid) % 64 * 8 % 64 * 2 ^ ((2432 + tid) / 64 * 128 + (2432 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (2432 + tid) / 64] * 512 + (long long)((2432 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (2432 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((2560 + tid) % 64 * 8 / 64 * 8192 + ((2560 + tid) / 64 * 128 + (2560 + tid) % 64 * 8 % 64 * 2 ^ ((2560 + tid) / 64 * 128 + (2560 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (2560 + tid) / 64] * 512 + (long long)((2560 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (2560 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((2688 + tid) % 64 * 8 / 64 * 8192 + ((2688 + tid) / 64 * 128 + (2688 + tid) % 64 * 8 % 64 * 2 ^ ((2688 + tid) / 64 * 128 + (2688 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (2688 + tid) / 64] * 512 + (long long)((2688 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (2688 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((2816 + tid) % 64 * 8 / 64 * 8192 + ((2816 + tid) / 64 * 128 + (2816 + tid) % 64 * 8 % 64 * 2 ^ ((2816 + tid) / 64 * 128 + (2816 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (2816 + tid) / 64] * 512 + (long long)((2816 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (2816 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((2944 + tid) % 64 * 8 / 64 * 8192 + ((2944 + tid) / 64 * 128 + (2944 + tid) % 64 * 8 % 64 * 2 ^ ((2944 + tid) / 64 * 128 + (2944 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (2944 + tid) / 64] * 512 + (long long)((2944 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (2944 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((3072 + tid) % 64 * 8 / 64 * 8192 + ((3072 + tid) / 64 * 128 + (3072 + tid) % 64 * 8 % 64 * 2 ^ ((3072 + tid) / 64 * 128 + (3072 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (3072 + tid) / 64] * 512 + (long long)((3072 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (3072 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((3200 + tid) % 64 * 8 / 64 * 8192 + ((3200 + tid) / 64 * 128 + (3200 + tid) % 64 * 8 % 64 * 2 ^ ((3200 + tid) / 64 * 128 + (3200 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (3200 + tid) / 64] * 512 + (long long)((3200 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (3200 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((3328 + tid) % 64 * 8 / 64 * 8192 + ((3328 + tid) / 64 * 128 + (3328 + tid) % 64 * 8 % 64 * 2 ^ ((3328 + tid) / 64 * 128 + (3328 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (3328 + tid) / 64] * 512 + (long long)((3328 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (3328 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((3456 + tid) % 64 * 8 / 64 * 8192 + ((3456 + tid) / 64 * 128 + (3456 + tid) % 64 * 8 % 64 * 2 ^ ((3456 + tid) / 64 * 128 + (3456 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (3456 + tid) / 64] * 512 + (long long)((3456 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (3456 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((3584 + tid) % 64 * 8 / 64 * 8192 + ((3584 + tid) / 64 * 128 + (3584 + tid) % 64 * 8 % 64 * 2 ^ ((3584 + tid) / 64 * 128 + (3584 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (3584 + tid) / 64] * 512 + (long long)((3584 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (3584 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((3712 + tid) % 64 * 8 / 64 * 8192 + ((3712 + tid) / 64 * 128 + (3712 + tid) % 64 * 8 % 64 * 2 ^ ((3712 + tid) / 64 * 128 + (3712 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (3712 + tid) / 64] * 512 + (long long)((3712 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (3712 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((3840 + tid) % 64 * 8 / 64 * 8192 + ((3840 + tid) / 64 * 128 + (3840 + tid) % 64 * 8 % 64 * 2 ^ ((3840 + tid) / 64 * 128 + (3840 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (3840 + tid) / 64] * 512 + (long long)((3840 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (3840 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 65536) + (unsigned int)((3968 + tid) % 64 * 8 / 64 * 8192 + ((3968 + tid) / 64 * 128 + (3968 + tid) % 64 * 8 % 64 * 2 ^ ((3968 + tid) / 64 * 128 + (3968 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 64 + (3968 + tid) / 64] * 512 + (long long)((3968 + tid) % 64 * 8))), "r"((s_idx[nst * 64 + (3968 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        int kv_base = s_kv_addr + (unsigned int)(st * 65536);
        #pragma unroll
        for (int kb = 0; kb < 32; kb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(s_q_addr + (unsigned int)((kb * 16 + lane / 16 * 8) / 64 * 2048 + ((hg * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 ^ ((hg * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                : "r"(kv_base + ((kb * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((kw * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((kw * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc_s[0]), "=f"(acc_s[1]), "=f"(acc_s[2]), "=f"(acc_s[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(((kb == 0) ? 0.0f : acc_s[0])), "f"(((kb == 0) ? 0.0f : acc_s[1])), "f"(((kb == 0) ? 0.0f : acc_s[2])), "f"(((kb == 0) ? 0.0f : acc_s[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"(acc_s[4]), "=f"(acc_s[(4) + 1]), "=f"(acc_s[(4) + 2]), "=f"(acc_s[(4) + 3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(((kb == 0) ? 0.0f : acc_s[4])), "f"(((kb == 0) ? 0.0f : acc_s[(4) + 1])), "f"(((kb == 0) ? 0.0f : acc_s[(4) + 2])), "f"(((kb == 0) ? 0.0f : acc_s[(4) + 3])));
        }
        #pragma unroll
        for (int nb = 0; nb < 2; nb++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int kcol = kw * 16 + nb * 8 + col_q + e;
                int slot_valid = s_idx[st * 64 + kcol];
                acc_s[nb * 4 + e] = ((slot_valid >= 0) ? acc_s[nb * 4 + e] : -CAKE_INF);
                acc_s[nb * 4 + 2 + e] = ((slot_valid >= 0) ? acc_s[nb * 4 + 2 + e] : -CAKE_INF);
            }
        }
        float _max_0 = max_noftz(acc_s[0], acc_s[1]);
        float _max_1 = max_noftz(acc_s[4], acc_s[5]);
        float _max_2 = max_noftz(_max_0, _max_1);
        float mx_a = _max_2;
        float _max_3 = max_noftz(acc_s[2], acc_s[3]);
        float _max_4 = max_noftz(acc_s[6], acc_s[7]);
        float _max_5 = max_noftz(_max_3, _max_4);
        float mx_b = _max_5;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, mx_a, 1);
        float _max_6 = max_noftz(mx_a, _shfl_xor_0);
        mx_a = _max_6;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, mx_a, 2);
        float _max_7 = max_noftz(mx_a, _shfl_xor_1);
        mx_a = _max_7;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, mx_b, 1);
        float _max_8 = max_noftz(mx_b, _shfl_xor_2);
        mx_b = _max_8;
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, mx_b, 2);
        float _max_9 = max_noftz(mx_b, _shfl_xor_3);
        mx_b = _max_9;
        if (lane_1 % 4 == 0) {
            s_red[kw * 16 + hg * 16 + r_a] = mx_a;
            s_red[kw * 16 + hg * 16 + r_b] = mx_b;
        }
        __syncthreads();
        float m_blk_a = s_red[hg * 16 + r_a];
        float m_blk_b = s_red[hg * 16 + r_b];
        #pragma unroll
        for (int w = 1; w < 4; w++) {
            float _max_10 = max_noftz(m_blk_a, s_red[w * 16 + hg * 16 + r_a]);
            m_blk_a = _max_10;
            float _max_11 = max_noftz(m_blk_b, s_red[w * 16 + hg * 16 + r_b]);
            m_blk_b = _max_11;
        }
        float _max_12 = max_noftz(m_a, m_blk_a);
        float m_new_a = _max_12;
        float _max_13 = max_noftz(m_b, m_blk_b);
        float m_new_b = _max_13;
        float _exp2_0 = approx_exp2((m_a - m_new_a) * scale_log2);
        float alpha_a = _exp2_0;
        float _exp2_1 = approx_exp2((m_b - m_new_b) * scale_log2);
        float alpha_b = _exp2_1;
        float sum_a = 0.0f;
        float sum_b = 0.0f;
        #pragma unroll
        for (int nb2 = 0; nb2 < 2; nb2++) {
            #pragma unroll
            for (int e2 = 0; e2 < 2; e2++) {
                int pcol = kw * 16 + nb2 * 8 + col_q + e2;
                float _exp2_2 = approx_exp2(acc_s[nb2 * 4 + e2] * scale_log2 - m_new_a * scale_log2);
                float p_a = _exp2_2;
                float _exp2_3 = approx_exp2(acc_s[nb2 * 4 + 2 + e2] * scale_log2 - m_new_b * scale_log2);
                float p_b = _exp2_3;
                sum_a = sum_a + p_a;
                sum_b = sum_b + p_b;
                {
                    __nv_bfloat16 _bval_0 = __float2bfloat16_rn(p_a);
                    uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                    uint32_t _addr_0 = static_cast<uint32_t>(s_p_addr + (unsigned int)(pcol / 64 * 2048 + ((hg * 16 + r_a) * 128 + pcol % 64 * 2 ^ ((hg * 16 + r_a) * 128 + pcol % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                }
                {
                    __nv_bfloat16 _bval_1 = __float2bfloat16_rn(p_b);
                    uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                    uint32_t _addr_1 = static_cast<uint32_t>(s_p_addr + (unsigned int)(pcol / 64 * 2048 + ((hg * 16 + r_b) * 128 + pcol % 64 * 2 ^ ((hg * 16 + r_b) * 128 + pcol % 64 * 2 >> 7 & 7) << 4)));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
                }
            }
        }
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, sum_a, 1);
        sum_a = sum_a + _shfl_xor_4;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, sum_a, 2);
        sum_a = sum_a + _shfl_xor_5;
        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, sum_b, 1);
        sum_b = sum_b + _shfl_xor_6;
        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, sum_b, 2);
        sum_b = sum_b + _shfl_xor_7;
        if (lane_1 % 4 == 0) {
            s_red[64 + kw * 16 + hg * 16 + r_a] = sum_a;
            s_red[64 + kw * 16 + hg * 16 + r_b] = sum_b;
        }
        __syncthreads();
        float l_blk_a = s_red[64 + hg * 16 + r_a];
        float l_blk_b = s_red[64 + hg * 16 + r_b];
        #pragma unroll
        for (int w2 = 1; w2 < 4; w2++) {
            l_blk_a = l_blk_a + s_red[64 + w2 * 16 + hg * 16 + r_a];
            l_blk_b = l_blk_b + s_red[64 + w2 * 16 + hg * 16 + r_b];
        }
        l_a = l_a * alpha_a + l_blk_a;
        l_b = l_b * alpha_b + l_blk_b;
        m_a = m_new_a;
        m_b = m_new_b;
        #pragma unroll
        for (int nb3 = 0; nb3 < 16; nb3++) {
            acc_o[nb3 * 4] = acc_o[nb3 * 4] * alpha_a;
            acc_o[nb3 * 4 + 1] = acc_o[nb3 * 4 + 1] * alpha_a;
            acc_o[nb3 * 4 + 2] = acc_o[nb3 * 4 + 2] * alpha_b;
            acc_o[nb3 * 4 + 3] = acc_o[nb3 * 4 + 3] * alpha_b;
        }
        #pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(s_p_addr + (unsigned int)((ks * 16 + lane / 16 * 8) / 64 * 2048 + ((hg * 16 + lane % 16) * 128 + (ks * 16 + lane / 16 * 8) % 64 * 2 ^ ((hg * 16 + lane % 16) * 128 + (ks * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            #pragma unroll
            for (int nb4 = 0; nb4 < 8; nb4++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_t[0]), "=r"(b_frag_t[1]), "=r"(b_frag_t[2]), "=r"(b_frag_t[3])
                    : "r"(kv_base + ((kw * 128 + nb4 * 16 + lane / 16 * 8) / 64 * 8192 + ((ks * 16 + lane % 16) * 128 + (kw * 128 + nb4 * 16 + lane / 16 * 8) % 64 * 2 ^ ((ks * 16 + lane % 16) * 128 + (kw * 128 + nb4 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_o + nb4 * 8)[0]), "+f"((acc_o + nb4 * 8)[1]), "+f"((acc_o + nb4 * 8)[2]), "+f"((acc_o + nb4 * 8)[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_t[0]), "r"(b_frag_t[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_o + nb4 * 8 + 4)[0]), "+f"((acc_o + nb4 * 8 + 4)[1]), "+f"((acc_o + nb4 * 8 + 4)[2]), "+f"((acc_o + nb4 * 8 + 4)[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_t[2]), "r"(b_frag_t[(2) + 1]));
            }
        }
    }
    float sink_a = -CAKE_INF;
    float sink_b = -CAKE_INF;
    if (head_a < heads) {
        sink_a = sink[head_a];
    }
    if (head_b < heads) {
        sink_b = sink[head_b];
    }
    float m_sc_a = m_a * scale_log2;
    float m_sc_b = m_b * scale_log2;
    float _max_14 = max_noftz(m_sc_a, sink_a * 1.4426950408889634f);
    float m_fin_a = _max_14;
    float _max_15 = max_noftz(m_sc_b, sink_b * 1.4426950408889634f);
    float m_fin_b = _max_15;
    float _exp2_4 = approx_exp2(m_sc_a - m_fin_a);
    float fin_a = _exp2_4;
    float _exp2_5 = approx_exp2(m_sc_b - m_fin_b);
    float fin_b = _exp2_5;
    float _exp2_6 = approx_exp2(sink_a * 1.4426950408889634f - m_fin_a);
    l_a = l_a * fin_a + _exp2_6;
    float _exp2_7 = approx_exp2(sink_b * 1.4426950408889634f - m_fin_b);
    l_b = l_b * fin_b + _exp2_7;
    float _max_16 = max_noftz(l_a, 1e-30f);
    l_a = _max_16;
    float _max_17 = max_noftz(l_b, 1e-30f);
    l_b = _max_17;
    float inv_a = fin_a / l_a;
    float inv_b = fin_b / l_b;
    long long out_a = ((long long)row * (long long)heads + (long long)head_a) * 512 + (long long)(kw * 128) + (long long)col_q;
    long long out_b = ((long long)row * (long long)heads + (long long)head_b) * 512 + (long long)(kw * 128) + (long long)col_q;
    #pragma unroll
    for (int nb5 = 0; nb5 < 16; nb5++) {
        acc_o[nb5 * 4] = acc_o[nb5 * 4] * inv_a;
        acc_o[nb5 * 4 + 1] = acc_o[nb5 * 4 + 1] * inv_a;
        acc_o[nb5 * 4 + 2] = acc_o[nb5 * 4 + 2] * inv_b;
        acc_o[nb5 * 4 + 3] = acc_o[nb5 * 4 + 3] * inv_b;
        if (head_a < heads) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc_o[nb5 * 4 + 0], acc_o[nb5 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out))[out_a + (long long)(nb5 * 8)]) = _pk;
            }
        }
        if (head_b < heads) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc_o[nb5 * 4 + 2 + 0], acc_o[nb5 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(out))[out_b + (long long)(nb5 * 8)]) = _pk;
            }
        }
    }
    if (kw == 0) {
        if (lane_1 % 4 == 0) {
            if (head_a < heads) {
                float _log2_0;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(l_a));
                lse[(long long)row * (long long)heads + (long long)head_a] = _log2_0 + m_fin_a;
            }
            if (head_b < heads) {
                float _log2_1;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(l_b));
                lse[(long long)row * (long long)heads + (long long)head_b] = _log2_1 + m_fin_b;
            }
        }
    }
}

} // extern "C"
