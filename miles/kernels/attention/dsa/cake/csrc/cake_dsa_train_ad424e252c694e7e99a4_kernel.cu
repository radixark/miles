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
#define SMEM_S_Q_STAGE_BYTES 32768
#define SMEM_S_Q_STRIDE 32768
#define SMEM_S_DO_OFF 32768
#define SMEM_S_DO_STAGE_BYTES 32768
#define SMEM_S_DO_STRIDE 32768
#define SMEM_S_KV_OFF 65536
#define SMEM_S_KV_STAGE_BYTES 32768
#define SMEM_S_KV_STRIDE 32768
#define SMEM_S_P_OFF 131072
#define SMEM_S_P_STAGE_BYTES 2048
#define SMEM_S_P_STRIDE 2048
#define SMEM_S_DS_OFF 135168
#define SMEM_S_DS_STAGE_BYTES 2048
#define SMEM_S_DS_STRIDE 2048
#define SMEM_S_DP_OFF 139264
#define SMEM_S_DP_STAGE_BYTES 4096
#define SMEM_S_DP_STRIDE 4096
#define SMEM_S_IDX_OFF 143360
#define SMEM_S_IDX_STAGE_BYTES 256
#define SMEM_S_IDX_STRIDE 256
#define SMEM_S_LSE_OFF 143616
#define SMEM_S_LSE_STAGE_BYTES 128
#define SMEM_S_LSE_STRIDE 128
#define SMEM_S_DELTA_OFF 143744
#define SMEM_S_DELTA_STAGE_BYTES 128
#define SMEM_S_DELTA_STRIDE 128
#define SMEM_TOTAL 143872
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_dsa_train_ad424e252c694e7e99a4(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ kv, __nv_bfloat16* __restrict__ do_, int* __restrict__ indices, float* __restrict__ lse, float* __restrict__ delta, __nv_bfloat16* __restrict__ dq, float* __restrict__ partial, int heads, int topk, int num_head_blocks, float sm_scale, float scale_log2)
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
    __nv_bfloat16* s_do = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_do_addr = smem + 32768;
    __nv_bfloat16* s_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_kv_addr = smem + 65536;
    __nv_bfloat16* s_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 131072);
    const int s_p_addr = smem + 131072;
    __nv_bfloat16* s_ds = reinterpret_cast<__nv_bfloat16*>(smem_raw + 135168);
    const int s_ds_addr = smem + 135168;
    float* s_dp = reinterpret_cast<float*>(smem_raw + 139264);
    const int s_dp_addr = smem + 139264;
    int* s_idx = reinterpret_cast<int*>(smem_raw + 143360);
    const int s_idx_addr = smem + 143360;
    float* s_lse = reinterpret_cast<float*>(smem_raw + 143616);
    const int s_lse_addr = smem + 143616;
    float* s_delta = reinterpret_cast<float*>(smem_raw + 143744);
    const int s_delta_addr = smem + 143744;

    // === Task calls (dependency order) ===
    int row = blockIdx.x;
    int hb = blockIdx.y;
    int tid_0 = tid;
    int lane_1 = lane;
    int warp_2 = warp;
    int grp = warp_2 / 4;
    int wl = warp_2 % 4;
    int hg = wl / 2;
    int kw = wl % 2;
    int hgq = warp_2 / 4;
    int cq = warp_2 % 4;
    int mt = warp_2 / 4;
    int cw = warp_2 % 4;
    int h0 = hb * 32;
    int r_a = lane_1 / 4;
    int r_b = r_a + 8;
    int col_q = lane_1 % 4 * 2;
    int num_blocks = topk / 32;
    long long idx_base = (long long)row * (long long)topk;
    long long prow_base = ((long long)row * (long long)num_head_blocks + (long long)hb) * (long long)topk;
    float acc_s[8];
    float acc_dq[64];
    float acc_dkv[64];
    unsigned int a_frag[4];
    unsigned int a_frag_t[4];
    unsigned int b_frag[4];
    unsigned int b_frag_t[4];
    unsigned int b_frag_x2[2];
    acc_dq[0] = 0.0f;
    acc_dq[1] = 0.0f;
    acc_dq[2] = 0.0f;
    acc_dq[3] = 0.0f;
    acc_dq[4] = 0.0f;
    acc_dq[5] = 0.0f;
    acc_dq[6] = 0.0f;
    acc_dq[7] = 0.0f;
    acc_dq[8] = 0.0f;
    acc_dq[9] = 0.0f;
    acc_dq[10] = 0.0f;
    acc_dq[11] = 0.0f;
    acc_dq[12] = 0.0f;
    acc_dq[13] = 0.0f;
    acc_dq[14] = 0.0f;
    acc_dq[15] = 0.0f;
    acc_dq[16] = 0.0f;
    acc_dq[17] = 0.0f;
    acc_dq[18] = 0.0f;
    acc_dq[19] = 0.0f;
    acc_dq[20] = 0.0f;
    acc_dq[21] = 0.0f;
    acc_dq[22] = 0.0f;
    acc_dq[23] = 0.0f;
    acc_dq[24] = 0.0f;
    acc_dq[25] = 0.0f;
    acc_dq[26] = 0.0f;
    acc_dq[27] = 0.0f;
    acc_dq[28] = 0.0f;
    acc_dq[29] = 0.0f;
    acc_dq[30] = 0.0f;
    acc_dq[31] = 0.0f;
    acc_dq[32] = 0.0f;
    acc_dq[33] = 0.0f;
    acc_dq[34] = 0.0f;
    acc_dq[35] = 0.0f;
    acc_dq[36] = 0.0f;
    acc_dq[37] = 0.0f;
    acc_dq[38] = 0.0f;
    acc_dq[39] = 0.0f;
    acc_dq[40] = 0.0f;
    acc_dq[41] = 0.0f;
    acc_dq[42] = 0.0f;
    acc_dq[43] = 0.0f;
    acc_dq[44] = 0.0f;
    acc_dq[45] = 0.0f;
    acc_dq[46] = 0.0f;
    acc_dq[47] = 0.0f;
    acc_dq[48] = 0.0f;
    acc_dq[49] = 0.0f;
    acc_dq[50] = 0.0f;
    acc_dq[51] = 0.0f;
    acc_dq[52] = 0.0f;
    acc_dq[53] = 0.0f;
    acc_dq[54] = 0.0f;
    acc_dq[55] = 0.0f;
    acc_dq[56] = 0.0f;
    acc_dq[57] = 0.0f;
    acc_dq[58] = 0.0f;
    acc_dq[59] = 0.0f;
    acc_dq[60] = 0.0f;
    acc_dq[61] = 0.0f;
    acc_dq[62] = 0.0f;
    acc_dq[63] = 0.0f;
    if (tid_0 < 32) {
        s_idx[tid_0] = indices[idx_base + (long long)tid_0];
    }
    if (tid_0 < 32) {
        int stat_h = h0 + tid_0;
        float lse_v = CAKE_INF;
        float delta_v = 0.0f;
        if (stat_h < heads) {
            lse_v = lse[(long long)row * (long long)heads + (long long)stat_h];
            delta_v = delta[(long long)row * (long long)heads + (long long)stat_h];
        }
        s_lse[tid_0] = lse_v;
        s_delta[tid_0] = delta_v;
    }
    __syncthreads();
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)(tid % 64 * 8 / 64 * 4096 + (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 ^ (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + tid / 64)) * 512 + (long long)(tid % 64 * 8))), "r"((h0 + tid / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((256 + tid) % 64 * 8 / 64 * 4096 + ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 ^ ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (256 + tid) / 64)) * 512 + (long long)((256 + tid) % 64 * 8))), "r"((h0 + (256 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((512 + tid) % 64 * 8 / 64 * 4096 + ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 ^ ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (512 + tid) / 64)) * 512 + (long long)((512 + tid) % 64 * 8))), "r"((h0 + (512 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((768 + tid) % 64 * 8 / 64 * 4096 + ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 ^ ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (768 + tid) / 64)) * 512 + (long long)((768 + tid) % 64 * 8))), "r"((h0 + (768 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((1024 + tid) % 64 * 8 / 64 * 4096 + ((1024 + tid) / 64 * 128 + (1024 + tid) % 64 * 8 % 64 * 2 ^ ((1024 + tid) / 64 * 128 + (1024 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (1024 + tid) / 64)) * 512 + (long long)((1024 + tid) % 64 * 8))), "r"((h0 + (1024 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((1280 + tid) % 64 * 8 / 64 * 4096 + ((1280 + tid) / 64 * 128 + (1280 + tid) % 64 * 8 % 64 * 2 ^ ((1280 + tid) / 64 * 128 + (1280 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (1280 + tid) / 64)) * 512 + (long long)((1280 + tid) % 64 * 8))), "r"((h0 + (1280 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((1536 + tid) % 64 * 8 / 64 * 4096 + ((1536 + tid) / 64 * 128 + (1536 + tid) % 64 * 8 % 64 * 2 ^ ((1536 + tid) / 64 * 128 + (1536 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (1536 + tid) / 64)) * 512 + (long long)((1536 + tid) % 64 * 8))), "r"((h0 + (1536 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((1792 + tid) % 64 * 8 / 64 * 4096 + ((1792 + tid) / 64 * 128 + (1792 + tid) % 64 * 8 % 64 * 2 ^ ((1792 + tid) / 64 * 128 + (1792 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(q + (((long long)row * (long long)heads + (long long)(h0 + (1792 + tid) / 64)) * 512 + (long long)((1792 + tid) % 64 * 8))), "r"((h0 + (1792 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)(tid % 64 * 8 / 64 * 4096 + (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 ^ (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)row * (long long)heads + (long long)(h0 + tid / 64)) * 512 + (long long)(tid % 64 * 8))), "r"((h0 + tid / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((256 + tid) % 64 * 8 / 64 * 4096 + ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 ^ ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)row * (long long)heads + (long long)(h0 + (256 + tid) / 64)) * 512 + (long long)((256 + tid) % 64 * 8))), "r"((h0 + (256 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((512 + tid) % 64 * 8 / 64 * 4096 + ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 ^ ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)row * (long long)heads + (long long)(h0 + (512 + tid) / 64)) * 512 + (long long)((512 + tid) % 64 * 8))), "r"((h0 + (512 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((768 + tid) % 64 * 8 / 64 * 4096 + ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 ^ ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)row * (long long)heads + (long long)(h0 + (768 + tid) / 64)) * 512 + (long long)((768 + tid) % 64 * 8))), "r"((h0 + (768 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((1024 + tid) % 64 * 8 / 64 * 4096 + ((1024 + tid) / 64 * 128 + (1024 + tid) % 64 * 8 % 64 * 2 ^ ((1024 + tid) / 64 * 128 + (1024 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)row * (long long)heads + (long long)(h0 + (1024 + tid) / 64)) * 512 + (long long)((1024 + tid) % 64 * 8))), "r"((h0 + (1024 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((1280 + tid) % 64 * 8 / 64 * 4096 + ((1280 + tid) / 64 * 128 + (1280 + tid) % 64 * 8 % 64 * 2 ^ ((1280 + tid) / 64 * 128 + (1280 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)row * (long long)heads + (long long)(h0 + (1280 + tid) / 64)) * 512 + (long long)((1280 + tid) % 64 * 8))), "r"((h0 + (1280 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((1536 + tid) % 64 * 8 / 64 * 4096 + ((1536 + tid) / 64 * 128 + (1536 + tid) % 64 * 8 % 64 * 2 ^ ((1536 + tid) / 64 * 128 + (1536 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)row * (long long)heads + (long long)(h0 + (1536 + tid) / 64)) * 512 + (long long)((1536 + tid) % 64 * 8))), "r"((h0 + (1536 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((1792 + tid) % 64 * 8 / 64 * 4096 + ((1792 + tid) / 64 * 128 + (1792 + tid) % 64 * 8 % 64 * 2 ^ ((1792 + tid) / 64 * 128 + (1792 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)row * (long long)heads + (long long)(h0 + (1792 + tid) / 64)) * 512 + (long long)((1792 + tid) % 64 * 8))), "r"((h0 + (1792 + tid) / 64 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)(tid % 64 * 8 / 64 * 4096 + (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 ^ (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[tid / 64] * 512 + (long long)(tid % 64 * 8))), "r"((s_idx[tid / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((256 + tid) % 64 * 8 / 64 * 4096 + ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 ^ ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(256 + tid) / 64] * 512 + (long long)((256 + tid) % 64 * 8))), "r"((s_idx[(256 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((512 + tid) % 64 * 8 / 64 * 4096 + ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 ^ ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(512 + tid) / 64] * 512 + (long long)((512 + tid) % 64 * 8))), "r"((s_idx[(512 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((768 + tid) % 64 * 8 / 64 * 4096 + ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 ^ ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(768 + tid) / 64] * 512 + (long long)((768 + tid) % 64 * 8))), "r"((s_idx[(768 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((1024 + tid) % 64 * 8 / 64 * 4096 + ((1024 + tid) / 64 * 128 + (1024 + tid) % 64 * 8 % 64 * 2 ^ ((1024 + tid) / 64 * 128 + (1024 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(1024 + tid) / 64] * 512 + (long long)((1024 + tid) % 64 * 8))), "r"((s_idx[(1024 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((1280 + tid) % 64 * 8 / 64 * 4096 + ((1280 + tid) / 64 * 128 + (1280 + tid) % 64 * 8 % 64 * 2 ^ ((1280 + tid) / 64 * 128 + (1280 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(1280 + tid) / 64] * 512 + (long long)((1280 + tid) % 64 * 8))), "r"((s_idx[(1280 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((1536 + tid) % 64 * 8 / 64 * 4096 + ((1536 + tid) / 64 * 128 + (1536 + tid) % 64 * 8 % 64 * 2 ^ ((1536 + tid) / 64 * 128 + (1536 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(1536 + tid) / 64] * 512 + (long long)((1536 + tid) % 64 * 8))), "r"((s_idx[(1536 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kv_addr + (unsigned int)((1792 + tid) % 64 * 8 / 64 * 4096 + ((1792 + tid) / 64 * 128 + (1792 + tid) % 64 * 8 % 64 * 2 ^ ((1792 + tid) / 64 * 128 + (1792 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[(1792 + tid) / 64] * 512 + (long long)((1792 + tid) % 64 * 8))), "r"((s_idx[(1792 + tid) / 64] >= 0) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    float lse_a = s_lse[hg * 16 + r_a];
    float lse_b = s_lse[hg * 16 + r_b];
    float delta_a = s_delta[hg * 16 + r_a];
    float delta_b = s_delta[hg * 16 + r_b];
    #pragma unroll 1
    for (int blk = 0; blk < num_blocks; blk++) {
        int st = blk % 2;
        int nst = 1 - st;
        if (num_blocks > blk + 1) {
            if (tid_0 < 32) {
                s_idx[nst * 32 + tid_0] = indices[idx_base + (long long)((blk + 1) * 32) + (long long)tid_0];
            }
        }
        __syncthreads();
        if (num_blocks > blk + 1) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 32768) + (unsigned int)(tid % 64 * 8 / 64 * 4096 + (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 ^ (tid / 64 * 128 + tid % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 32 + tid / 64] * 512 + (long long)(tid % 64 * 8))), "r"((s_idx[nst * 32 + tid / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 32768) + (unsigned int)((256 + tid) % 64 * 8 / 64 * 4096 + ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 ^ ((256 + tid) / 64 * 128 + (256 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 32 + (256 + tid) / 64] * 512 + (long long)((256 + tid) % 64 * 8))), "r"((s_idx[nst * 32 + (256 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 32768) + (unsigned int)((512 + tid) % 64 * 8 / 64 * 4096 + ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 ^ ((512 + tid) / 64 * 128 + (512 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 32 + (512 + tid) / 64] * 512 + (long long)((512 + tid) % 64 * 8))), "r"((s_idx[nst * 32 + (512 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 32768) + (unsigned int)((768 + tid) % 64 * 8 / 64 * 4096 + ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 ^ ((768 + tid) / 64 * 128 + (768 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 32 + (768 + tid) / 64] * 512 + (long long)((768 + tid) % 64 * 8))), "r"((s_idx[nst * 32 + (768 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 32768) + (unsigned int)((1024 + tid) % 64 * 8 / 64 * 4096 + ((1024 + tid) / 64 * 128 + (1024 + tid) % 64 * 8 % 64 * 2 ^ ((1024 + tid) / 64 * 128 + (1024 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 32 + (1024 + tid) / 64] * 512 + (long long)((1024 + tid) % 64 * 8))), "r"((s_idx[nst * 32 + (1024 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 32768) + (unsigned int)((1280 + tid) % 64 * 8 / 64 * 4096 + ((1280 + tid) / 64 * 128 + (1280 + tid) % 64 * 8 % 64 * 2 ^ ((1280 + tid) / 64 * 128 + (1280 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 32 + (1280 + tid) / 64] * 512 + (long long)((1280 + tid) % 64 * 8))), "r"((s_idx[nst * 32 + (1280 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 32768) + (unsigned int)((1536 + tid) % 64 * 8 / 64 * 4096 + ((1536 + tid) / 64 * 128 + (1536 + tid) % 64 * 8 % 64 * 2 ^ ((1536 + tid) / 64 * 128 + (1536 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 32 + (1536 + tid) / 64] * 512 + (long long)((1536 + tid) % 64 * 8))), "r"((s_idx[nst * 32 + (1536 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_kv_addr + (unsigned int)(nst * 32768) + (unsigned int)((1792 + tid) % 64 * 8 / 64 * 4096 + ((1792 + tid) / 64 * 128 + (1792 + tid) % 64 * 8 % 64 * 2 ^ ((1792 + tid) / 64 * 128 + (1792 + tid) % 64 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kv + ((long long)s_idx[nst * 32 + (1792 + tid) / 64] * 512 + (long long)((1792 + tid) % 64 * 8))), "r"((s_idx[nst * 32 + (1792 + tid) / 64] >= 0) ? 16 : 0));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        int kv_base = s_kv_addr + (unsigned int)(st * 32768);
        if (grp == 0) {
            #pragma unroll
            for (int kb = 0; kb < 32; kb++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(s_q_addr + (unsigned int)((kb * 16 + lane / 16 * 8) / 64 * 4096 + ((hg * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 ^ ((hg * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                    : "r"(kv_base + ((kb * 16 + lane % 16 / 8 * 8) / 64 * 4096 + ((kw * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((kw * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(acc_s[0]), "=f"(acc_s[1]), "=f"(acc_s[2]), "=f"(acc_s[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(((kb == 0) ? 0.0f : acc_s[0])), "f"(((kb == 0) ? 0.0f : acc_s[1])), "f"(((kb == 0) ? 0.0f : acc_s[2])), "f"(((kb == 0) ? 0.0f : acc_s[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(acc_s[4]), "=f"(acc_s[(4) + 1]), "=f"(acc_s[(4) + 2]), "=f"(acc_s[(4) + 3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(((kb == 0) ? 0.0f : acc_s[4])), "f"(((kb == 0) ? 0.0f : acc_s[(4) + 1])), "f"(((kb == 0) ? 0.0f : acc_s[(4) + 2])), "f"(((kb == 0) ? 0.0f : acc_s[(4) + 3])));
            }
        } else {
            #pragma unroll
            for (int kb2 = 0; kb2 < 32; kb2++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                    : "r"(s_do_addr + (unsigned int)((kb2 * 16 + lane / 16 * 8) / 64 * 4096 + ((hg * 16 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 ^ ((hg * 16 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                    : "r"(kv_base + ((kb2 * 16 + lane % 16 / 8 * 8) / 64 * 4096 + ((kw * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb2 * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((kw * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb2 * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(acc_s[0]), "=f"(acc_s[1]), "=f"(acc_s[2]), "=f"(acc_s[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(((kb2 == 0) ? 0.0f : acc_s[0])), "f"(((kb2 == 0) ? 0.0f : acc_s[1])), "f"(((kb2 == 0) ? 0.0f : acc_s[2])), "f"(((kb2 == 0) ? 0.0f : acc_s[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(acc_s[4]), "=f"(acc_s[(4) + 1]), "=f"(acc_s[(4) + 2]), "=f"(acc_s[(4) + 3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(((kb2 == 0) ? 0.0f : acc_s[4])), "f"(((kb2 == 0) ? 0.0f : acc_s[(4) + 1])), "f"(((kb2 == 0) ? 0.0f : acc_s[(4) + 2])), "f"(((kb2 == 0) ? 0.0f : acc_s[(4) + 3])));
            }
            #pragma unroll
            for (int nb = 0; nb < 2; nb++) {
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int dcol = kw * 16 + nb * 8 + col_q + e;
                    s_dp[(hg * 16 + r_a) * 32 + dcol] = acc_s[nb * 4 + e];
                    s_dp[(hg * 16 + r_b) * 32 + dcol] = acc_s[nb * 4 + 2 + e];
                }
            }
        }
        __syncthreads();
        if (grp == 0) {
            #pragma unroll
            for (int nb2 = 0; nb2 < 2; nb2++) {
                #pragma unroll
                for (int e2 = 0; e2 < 2; e2++) {
                    int pcol = kw * 16 + nb2 * 8 + col_q + e2;
                    int slot_ok = s_idx[st * 32 + pcol];
                    float _exp2_0 = approx_exp2(acc_s[nb2 * 4 + e2] * scale_log2 - lse_a);
                    float p_raw_a = _exp2_0;
                    float _exp2_1 = approx_exp2(acc_s[nb2 * 4 + 2 + e2] * scale_log2 - lse_b);
                    float p_raw_b = _exp2_1;
                    float p_a = ((slot_ok >= 0) ? p_raw_a : 0.0f);
                    float p_b = ((slot_ok >= 0) ? p_raw_b : 0.0f);
                    float ds_a = p_a * (s_dp[(hg * 16 + r_a) * 32 + pcol] - delta_a) * sm_scale;
                    float ds_b = p_b * (s_dp[(hg * 16 + r_b) * 32 + pcol] - delta_b) * sm_scale;
                    {
                        __nv_bfloat16 _bval_0 = __float2bfloat16_rn(p_a);
                        uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                        uint32_t _addr_0 = static_cast<uint32_t>(s_p_addr + (unsigned int)(pcol / 64 * 4096 + ((hg * 16 + r_a) * 128 + pcol % 64 * 2 ^ ((hg * 16 + r_a) * 128 + pcol % 64 * 2 >> 7 & 7) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_1 = __float2bfloat16_rn(p_b);
                        uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                        uint32_t _addr_1 = static_cast<uint32_t>(s_p_addr + (unsigned int)(pcol / 64 * 4096 + ((hg * 16 + r_b) * 128 + pcol % 64 * 2 ^ ((hg * 16 + r_b) * 128 + pcol % 64 * 2 >> 7 & 7) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_2 = __float2bfloat16_rn(ds_a);
                        uint16_t _bits_2 = *(uint16_t*)&_bval_2;
                        uint32_t _addr_2 = static_cast<uint32_t>(s_ds_addr + (unsigned int)(pcol / 64 * 4096 + ((hg * 16 + r_a) * 128 + pcol % 64 * 2 ^ ((hg * 16 + r_a) * 128 + pcol % 64 * 2 >> 7 & 7) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_2), "h"(_bits_2) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_3 = __float2bfloat16_rn(ds_b);
                        uint16_t _bits_3 = *(uint16_t*)&_bval_3;
                        uint32_t _addr_3 = static_cast<uint32_t>(s_ds_addr + (unsigned int)(pcol / 64 * 4096 + ((hg * 16 + r_b) * 128 + pcol % 64 * 2 ^ ((hg * 16 + r_b) * 128 + pcol % 64 * 2 >> 7 & 7) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_3), "h"(_bits_3) : "memory");
                    }
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(s_ds_addr + (unsigned int)((ks * 16 + lane / 16 * 8) / 64 * 4096 + ((hgq * 16 + lane % 16) * 128 + (ks * 16 + lane / 16 * 8) % 64 * 2 ^ ((hgq * 16 + lane % 16) * 128 + (ks * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            #pragma unroll
            for (int nb3 = 0; nb3 < 8; nb3++) {
                if (2 * nb3 + 1 < 16) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag_t[0]), "=r"(b_frag_t[1]), "=r"(b_frag_t[2]), "=r"(b_frag_t[3])
                        : "r"(kv_base + ((cq * 128 + nb3 * 16 + lane / 16 * 8) / 64 * 4096 + ((ks * 16 + lane % 16) * 128 + (cq * 128 + nb3 * 16 + lane / 16 * 8) % 64 * 2 ^ ((ks * 16 + lane % 16) * 128 + (cq * 128 + nb3 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc_dq + nb3 * 8)[0]), "+f"((acc_dq + nb3 * 8)[1]), "+f"((acc_dq + nb3 * 8)[2]), "+f"((acc_dq + nb3 * 8)[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_t[0]), "r"(b_frag_t[1]));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc_dq + nb3 * 8 + 4)[0]), "+f"((acc_dq + nb3 * 8 + 4)[1]), "+f"((acc_dq + nb3 * 8 + 4)[2]), "+f"((acc_dq + nb3 * 8 + 4)[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_t[2]), "r"(b_frag_t[(2) + 1]));
                } else {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(b_frag_x2[0]), "=r"(b_frag_x2[1])
                        : "r"(kv_base + ((cq * 128 + nb3 * 16) / 64 * 4096 + ((ks * 16 + lane % 16) * 128 + (cq * 128 + nb3 * 16) % 64 * 2 ^ ((ks * 16 + lane % 16) * 128 + (cq * 128 + nb3 * 16) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc_dq + nb3 * 8)[0]), "+f"((acc_dq + nb3 * 8)[1]), "+f"((acc_dq + nb3 * 8)[2]), "+f"((acc_dq + nb3 * 8)[3])
                        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_x2[0]), "r"(b_frag_x2[1]));
                }
            }
        }
        #pragma unroll
        for (int ps = 0; ps < 1; ps++) {
            int win0 = (cw + ps) * 128;
            #pragma unroll
            for (int hs = 0; hs < 2; hs++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag_t[0]), "=r"(a_frag_t[1]), "=r"(a_frag_t[2]), "=r"(a_frag_t[3])
                    : "r"(s_ds_addr + (unsigned int)((mt * 16 + lane % 16 / 8 * 8) / 64 * 4096 + ((hs * 16 + 8 * (lane / 16) + lane % 8) * 128 + (mt * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((hs * 16 + 8 * (lane / 16) + lane % 8) * 128 + (mt * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                #pragma unroll
                for (int nb4 = 0; nb4 < 8; nb4++) {
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(b_frag_t[0]), "=r"(b_frag_t[1]), "=r"(b_frag_t[2]), "=r"(b_frag_t[3])
                        : "r"(s_q_addr + (unsigned int)((win0 + nb4 * 16 + lane / 16 * 8) / 64 * 4096 + ((hs * 16 + lane % 16) * 128 + (win0 + nb4 * 16 + lane / 16 * 8) % 64 * 2 ^ ((hs * 16 + lane % 16) * 128 + (win0 + nb4 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"((acc_dkv + nb4 * 8)[0]), "=f"((acc_dkv + nb4 * 8)[1]), "=f"((acc_dkv + nb4 * 8)[2]), "=f"((acc_dkv + nb4 * 8)[3])
                        : "r"(a_frag_t[0]), "r"(a_frag_t[1]), "r"(a_frag_t[2]), "r"(a_frag_t[3]), "r"(b_frag_t[0]), "r"(b_frag_t[1]), "f"(((hs == 0) ? 0.0f : (acc_dkv + nb4 * 8)[0])), "f"(((hs == 0) ? 0.0f : (acc_dkv + nb4 * 8)[1])), "f"(((hs == 0) ? 0.0f : (acc_dkv + nb4 * 8)[2])), "f"(((hs == 0) ? 0.0f : (acc_dkv + nb4 * 8)[3])));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"((acc_dkv + nb4 * 8 + 4)[0]), "=f"((acc_dkv + nb4 * 8 + 4)[1]), "=f"((acc_dkv + nb4 * 8 + 4)[2]), "=f"((acc_dkv + nb4 * 8 + 4)[3])
                        : "r"(a_frag_t[0]), "r"(a_frag_t[1]), "r"(a_frag_t[2]), "r"(a_frag_t[3]), "r"(b_frag_t[2]), "r"(b_frag_t[(2) + 1]), "f"(((hs == 0) ? 0.0f : (acc_dkv + nb4 * 8 + 4)[0])), "f"(((hs == 0) ? 0.0f : (acc_dkv + nb4 * 8 + 4)[1])), "f"(((hs == 0) ? 0.0f : (acc_dkv + nb4 * 8 + 4)[2])), "f"(((hs == 0) ? 0.0f : (acc_dkv + nb4 * 8 + 4)[3])));
                }
            }
            #pragma unroll
            for (int hs2 = 0; hs2 < 2; hs2++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag_t[0]), "=r"(a_frag_t[1]), "=r"(a_frag_t[2]), "=r"(a_frag_t[3])
                    : "r"(s_p_addr + (unsigned int)((mt * 16 + lane % 16 / 8 * 8) / 64 * 4096 + ((hs2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (mt * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((hs2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (mt * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                #pragma unroll
                for (int nb5 = 0; nb5 < 8; nb5++) {
                    if (win0 + nb5 * 16 < 512) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag_t[0]), "=r"(b_frag_t[1]), "=r"(b_frag_t[2]), "=r"(b_frag_t[3])
                            : "r"(s_do_addr + (unsigned int)((win0 + nb5 * 16 + lane / 16 * 8) / 64 * 4096 + ((hs2 * 16 + lane % 16) * 128 + (win0 + nb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((hs2 * 16 + lane % 16) * 128 + (win0 + nb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"((acc_dkv + nb5 * 8)[0]), "+f"((acc_dkv + nb5 * 8)[1]), "+f"((acc_dkv + nb5 * 8)[2]), "+f"((acc_dkv + nb5 * 8)[3])
                            : "r"(a_frag_t[0]), "r"(a_frag_t[1]), "r"(a_frag_t[2]), "r"(a_frag_t[3]), "r"(b_frag_t[0]), "r"(b_frag_t[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"((acc_dkv + nb5 * 8 + 4)[0]), "+f"((acc_dkv + nb5 * 8 + 4)[1]), "+f"((acc_dkv + nb5 * 8 + 4)[2]), "+f"((acc_dkv + nb5 * 8 + 4)[3])
                            : "r"(a_frag_t[0]), "r"(a_frag_t[1]), "r"(a_frag_t[2]), "r"(a_frag_t[3]), "r"(b_frag_t[2]), "r"(b_frag_t[(2) + 1]));
                    }
                }
            }
            long long prow_a = (prow_base + (long long)(blk * 32 + mt * 16 + r_a)) * 512 + (long long)win0 + (long long)col_q;
            long long prow_b = prow_a + 4096;
            #pragma unroll
            for (int nb6 = 0; nb6 < 16; nb6++) {
                {
                    float2 _v2 = make_float2(acc_dkv[nb6 * 4 + 0], acc_dkv[nb6 * 4 + 1]);
                    *reinterpret_cast<float2*>(partial + prow_a + (long long)(nb6 * 8)) = _v2;
                }
                {
                    float2 _v2 = make_float2(acc_dkv[nb6 * 4 + 2 + 0], acc_dkv[nb6 * 4 + 2 + 1]);
                    *reinterpret_cast<float2*>(partial + prow_b + (long long)(nb6 * 8)) = _v2;
                }
            }
        }
    }
    int head_qa = h0 + hgq * 16 + r_a;
    int head_qb = head_qa + 8;
    long long dq_a = ((long long)row * (long long)heads + (long long)head_qa) * 512 + (long long)(cq * 128) + (long long)col_q;
    long long dq_b = ((long long)row * (long long)heads + (long long)head_qb) * 512 + (long long)(cq * 128) + (long long)col_q;
    #pragma unroll
    for (int nb7 = 0; nb7 < 16; nb7++) {
        if (head_qa < heads) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc_dq[nb7 * 4 + 0], acc_dq[nb7 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dq))[dq_a + (long long)(nb7 * 8)]) = _pk;
            }
        }
        if (head_qb < heads) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc_dq[nb7 * 4 + 2 + 0], acc_dq[nb7 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dq))[dq_b + (long long)(nb7 * 8)]) = _pk;
            }
        }
    }
}

} // extern "C"
