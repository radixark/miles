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
#define SMEM_S_Q_STAGE_BYTES 8192
#define SMEM_S_Q_STRIDE 8192
#define SMEM_S_K_OFF 8192
#define SMEM_S_K_STAGE_BYTES 8192
#define SMEM_S_K_STRIDE 8192
#define SMEM_S_G_OFF 24576
#define SMEM_S_G_STAGE_BYTES 2048
#define SMEM_S_G_STRIDE 2048
#define SMEM_S_G_LO_OFF 28672
#define SMEM_S_G_LO_STAGE_BYTES 2048
#define SMEM_S_G_LO_STRIDE 2048
#define SMEM_S_W_OFF 32768
#define SMEM_S_W_STAGE_BYTES 128
#define SMEM_S_W_STRIDE 128
#define SMEM_S_GRAD_OFF 32896
#define SMEM_S_GRAD_STAGE_BYTES 256
#define SMEM_S_GRAD_STRIDE 256
#define SMEM_S_IDX_OFF 33152
#define SMEM_S_IDX_STAGE_BYTES 256
#define SMEM_S_IDX_STRIDE 256
#define SMEM_S_DW_OFF 33408
#define SMEM_S_DW_STAGE_BYTES 256
#define SMEM_S_DW_STRIDE 256
#define SMEM_TOTAL 33664
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_dsa_train_bfcdd38b8b2b9195ae2f(__nv_bfloat16* __restrict__ index_q, __nv_bfloat16* __restrict__ index_k, float* __restrict__ weights, int* __restrict__ topk_indices, float* __restrict__ grad_scores, __nv_bfloat16* __restrict__ d_index_q, float* __restrict__ d_weights, float* __restrict__ partial, int heads, int topk)
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
    __nv_bfloat16* s_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 8192);
    const int s_k_addr = smem + 8192;
    __nv_bfloat16* s_g = reinterpret_cast<__nv_bfloat16*>(smem_raw + 24576);
    const int s_g_addr = smem + 24576;
    __nv_bfloat16* s_g_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 28672);
    const int s_g_lo_addr = smem + 28672;
    float* s_w = reinterpret_cast<float*>(smem_raw + 32768);
    const int s_w_addr = smem + 32768;
    float* s_grad = reinterpret_cast<float*>(smem_raw + 32896);
    const int s_grad_addr = smem + 32896;
    int* s_idx = reinterpret_cast<int*>(smem_raw + 33152);
    const int s_idx_addr = smem + 33152;
    float* s_dw = reinterpret_cast<float*>(smem_raw + 33408);
    const int s_dw_addr = smem + 33408;

    // === Task calls (dependency order) ===
    int row = blockIdx.x;
    int tid_0 = tid;
    int lane_1 = lane;
    int warp_2 = warp;
    int mt = warp_2 % 2;
    int nh = warp_2 / 2;
    int hg = warp_2 / 2;
    int cg = warp_2 % 2;
    int r_a = lane_1 / 4;
    int r_b = r_a + 8;
    int col_q = lane_1 % 4 * 2;
    int num_blocks = topk / 32;
    long long idx_base = (long long)row * (long long)topk;
    long long q_row0 = (long long)row * (long long)heads;
    float acc_l[8];
    float acc_dq[32];
    float acc_dk[32];
    float dw_part[4];
    unsigned int a_frag[4];
    unsigned int a_frag_lo[4];
    unsigned int a_frag_t[4];
    unsigned int a_frag_t_lo[4];
    unsigned int b_frag[4];
    unsigned int b_frag_t[4];
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
    dw_part[0] = 0.0f;
    dw_part[1] = 0.0f;
    dw_part[2] = 0.0f;
    dw_part[3] = 0.0f;
    if (tid_0 < 32) {
        float w_val = 0.0f;
        if (tid_0 < heads) {
            w_val = weights[q_row0 + (long long)tid_0];
        }
        s_w[tid_0] = w_val;
    }
    if (tid_0 < 32) {
        s_idx[tid_0] = topk_indices[idx_base + (long long)tid_0];
        s_grad[tid_0] = grad_scores[idx_base + (long long)tid_0];
    }
    __syncthreads();
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)(tid % 16 * 8 / 64 * 4096 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_q + ((q_row0 + (long long)(tid / 16)) * 128 + (long long)(tid % 16 * 8))), "r"((tid / 16 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((128 + tid) % 16 * 8 / 64 * 4096 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_q + ((q_row0 + (long long)((128 + tid) / 16)) * 128 + (long long)((128 + tid) % 16 * 8))), "r"(((128 + tid) / 16 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 4096 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_q + ((q_row0 + (long long)((256 + tid) / 16)) * 128 + (long long)((256 + tid) % 16 * 8))), "r"(((256 + tid) / 16 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((384 + tid) % 16 * 8 / 64 * 4096 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_q + ((q_row0 + (long long)((384 + tid) / 16)) * 128 + (long long)((384 + tid) % 16 * 8))), "r"(((384 + tid) / 16 < heads) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_k_addr + (unsigned int)(tid % 16 * 8 / 64 * 4096 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((long long)s_idx[tid / 16] * 128 + (long long)(tid % 16 * 8))), "r"((s_idx[tid / 16] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_k_addr + (unsigned int)((128 + tid) % 16 * 8 / 64 * 4096 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((long long)s_idx[(128 + tid) / 16] * 128 + (long long)((128 + tid) % 16 * 8))), "r"((s_idx[(128 + tid) / 16] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_k_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 4096 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((long long)s_idx[(256 + tid) / 16] * 128 + (long long)((256 + tid) % 16 * 8))), "r"((s_idx[(256 + tid) / 16] >= 0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_k_addr + (unsigned int)((384 + tid) % 16 * 8 / 64 * 4096 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((long long)s_idx[(384 + tid) / 16] * 128 + (long long)((384 + tid) % 16 * 8))), "r"((s_idx[(384 + tid) / 16] >= 0) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    #pragma unroll 1
    for (int blk = 0; blk < num_blocks; blk++) {
        int st = blk % 2;
        int nst = 1 - st;
        if (num_blocks > blk + 1) {
            if (tid_0 < 32) {
                s_idx[nst * 32 + tid_0] = topk_indices[idx_base + (long long)((blk + 1) * 32) + (long long)tid_0];
                s_grad[nst * 32 + tid_0] = grad_scores[idx_base + (long long)((blk + 1) * 32) + (long long)tid_0];
            }
        }
        __syncthreads();
        if (num_blocks > blk + 1) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_k_addr + (unsigned int)(nst * 8192) + (unsigned int)(tid % 16 * 8 / 64 * 4096 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((long long)s_idx[nst * 32 + tid / 16] * 128 + (long long)(tid % 16 * 8))), "r"((s_idx[nst * 32 + tid / 16] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_k_addr + (unsigned int)(nst * 8192) + (unsigned int)((128 + tid) % 16 * 8 / 64 * 4096 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((long long)s_idx[nst * 32 + (128 + tid) / 16] * 128 + (long long)((128 + tid) % 16 * 8))), "r"((s_idx[nst * 32 + (128 + tid) / 16] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_k_addr + (unsigned int)(nst * 8192) + (unsigned int)((256 + tid) % 16 * 8 / 64 * 4096 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((long long)s_idx[nst * 32 + (256 + tid) / 16] * 128 + (long long)((256 + tid) % 16 * 8))), "r"((s_idx[nst * 32 + (256 + tid) / 16] >= 0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_k_addr + (unsigned int)(nst * 8192) + (unsigned int)((384 + tid) % 16 * 8 / 64 * 4096 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((long long)s_idx[nst * 32 + (384 + tid) / 16] * 128 + (long long)((384 + tid) % 16 * 8))), "r"((s_idx[nst * 32 + (384 + tid) / 16] >= 0) ? 16 : 0));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        int k_base = s_k_addr + (unsigned int)(st * 8192);
        #pragma unroll
        for (int kb = 0; kb < 8; kb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(k_base + ((kb * 16 + lane / 16 * 8) / 64 * 4096 + ((mt * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 ^ ((mt * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            #pragma unroll
            for (int nb2 = 0; nb2 < 1; nb2++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                    : "r"(s_q_addr + (unsigned int)((kb * 16 + lane % 16 / 8 * 8) / 64 * 4096 + ((nh * 16 + nb2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((nh * 16 + nb2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_l + nb2 * 8)[0]), "=f"((acc_l + nb2 * 8)[1]), "=f"((acc_l + nb2 * 8)[2]), "=f"((acc_l + nb2 * 8)[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(((kb == 0) ? 0.0f : (acc_l + nb2 * 8)[0])), "f"(((kb == 0) ? 0.0f : (acc_l + nb2 * 8)[1])), "f"(((kb == 0) ? 0.0f : (acc_l + nb2 * 8)[2])), "f"(((kb == 0) ? 0.0f : (acc_l + nb2 * 8)[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_l + nb2 * 8 + 4)[0]), "=f"((acc_l + nb2 * 8 + 4)[1]), "=f"((acc_l + nb2 * 8 + 4)[2]), "=f"((acc_l + nb2 * 8 + 4)[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(((kb == 0) ? 0.0f : (acc_l + nb2 * 8 + 4)[0])), "f"(((kb == 0) ? 0.0f : (acc_l + nb2 * 8 + 4)[1])), "f"(((kb == 0) ? 0.0f : (acc_l + nb2 * 8 + 4)[2])), "f"(((kb == 0) ? 0.0f : (acc_l + nb2 * 8 + 4)[3])));
            }
        }
        float grad_a = s_grad[st * 32 + mt * 16 + r_a];
        float grad_b = s_grad[st * 32 + mt * 16 + r_b];
        #pragma unroll
        for (int nb3 = 0; nb3 < 2; nb3++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int hcol = nh * 16 + nb3 * 8 + col_q + e;
                float w_h = s_w[hcol];
                float l_a = acc_l[nb3 * 4 + e];
                float l_b = acc_l[nb3 * 4 + 2 + e];
                float _max_0 = max_noftz(l_a, 0.0f);
                float _max_1 = max_noftz(l_b, 0.0f);
                dw_part[nb3 * 2 + e] = dw_part[nb3 * 2 + e] + grad_a * _max_0 + grad_b * _max_1;
                float g_a = ((l_a > 0.0f) ? grad_a * w_h : 0.0f);
                float g_b = ((l_b > 0.0f) ? grad_b * w_h : 0.0f);
                int off_ga = hcol / 64 * 4096 + ((mt * 16 + r_a) * 128 + hcol % 64 * 2 ^ ((mt * 16 + r_a) * 128 + hcol % 64 * 2 >> 7 & 7) << 4);
                int off_gb = hcol / 64 * 4096 + ((mt * 16 + r_b) * 128 + hcol % 64 * 2 ^ ((mt * 16 + r_b) * 128 + hcol % 64 * 2 >> 7 & 7) << 4);
                {
                    __nv_bfloat16 _bval_0 = __float2bfloat16_rn(g_a);
                    uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                    uint32_t _addr_0 = static_cast<uint32_t>(s_g_addr + (unsigned int)off_ga);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                }
                {
                    __nv_bfloat16 _bval_1 = __float2bfloat16_rn(g_b);
                    uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                    uint32_t _addr_1 = static_cast<uint32_t>(s_g_addr + (unsigned int)off_gb);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
                }
                {
                    __nv_bfloat16 _bval_2 = __float2bfloat16_rn(g_a - (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_g) + off_ga)[0]);
                    uint16_t _bits_2 = *(uint16_t*)&_bval_2;
                    uint32_t _addr_2 = static_cast<uint32_t>(s_g_lo_addr + (unsigned int)off_ga);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_2), "h"(_bits_2) : "memory");
                }
                {
                    __nv_bfloat16 _bval_3 = __float2bfloat16_rn(g_b - (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_g) + off_gb)[0]);
                    uint16_t _bits_3 = *(uint16_t*)&_bval_3;
                    uint32_t _addr_3 = static_cast<uint32_t>(s_g_lo_addr + (unsigned int)off_gb);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_3), "h"(_bits_3) : "memory");
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag_t[0]), "=r"(a_frag_t[1]), "=r"(a_frag_t[2]), "=r"(a_frag_t[3])
                : "r"(s_g_addr + (unsigned int)((hg * 16 + lane % 16 / 8 * 8) / 64 * 4096 + ((ks * 16 + 8 * (lane / 16) + lane % 8) * 128 + (hg * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((ks * 16 + 8 * (lane / 16) + lane % 8) * 128 + (hg * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag_t_lo[0]), "=r"(a_frag_t_lo[1]), "=r"(a_frag_t_lo[2]), "=r"(a_frag_t_lo[3])
                : "r"(s_g_lo_addr + (unsigned int)((hg * 16 + lane % 16 / 8 * 8) / 64 * 4096 + ((ks * 16 + 8 * (lane / 16) + lane % 8) * 128 + (hg * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((ks * 16 + 8 * (lane / 16) + lane % 8) * 128 + (hg * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            #pragma unroll
            for (int nb4 = 0; nb4 < 4; nb4++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_t[0]), "=r"(b_frag_t[1]), "=r"(b_frag_t[2]), "=r"(b_frag_t[3])
                    : "r"(k_base + ((cg * 64 + nb4 * 16 + lane / 16 * 8) / 64 * 4096 + ((ks * 16 + lane % 16) * 128 + (cg * 64 + nb4 * 16 + lane / 16 * 8) % 64 * 2 ^ ((ks * 16 + lane % 16) * 128 + (cg * 64 + nb4 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_dq + nb4 * 8)[0]), "+f"((acc_dq + nb4 * 8)[1]), "+f"((acc_dq + nb4 * 8)[2]), "+f"((acc_dq + nb4 * 8)[3])
                    : "r"(a_frag_t[0]), "r"(a_frag_t[1]), "r"(a_frag_t[2]), "r"(a_frag_t[3]), "r"(b_frag_t[0]), "r"(b_frag_t[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_dq + nb4 * 8 + 4)[0]), "+f"((acc_dq + nb4 * 8 + 4)[1]), "+f"((acc_dq + nb4 * 8 + 4)[2]), "+f"((acc_dq + nb4 * 8 + 4)[3])
                    : "r"(a_frag_t[0]), "r"(a_frag_t[1]), "r"(a_frag_t[2]), "r"(a_frag_t[3]), "r"(b_frag_t[2]), "r"(b_frag_t[(2) + 1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_dq + nb4 * 8)[0]), "+f"((acc_dq + nb4 * 8)[1]), "+f"((acc_dq + nb4 * 8)[2]), "+f"((acc_dq + nb4 * 8)[3])
                    : "r"(a_frag_t_lo[0]), "r"(a_frag_t_lo[1]), "r"(a_frag_t_lo[2]), "r"(a_frag_t_lo[3]), "r"(b_frag_t[0]), "r"(b_frag_t[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_dq + nb4 * 8 + 4)[0]), "+f"((acc_dq + nb4 * 8 + 4)[1]), "+f"((acc_dq + nb4 * 8 + 4)[2]), "+f"((acc_dq + nb4 * 8 + 4)[3])
                    : "r"(a_frag_t_lo[0]), "r"(a_frag_t_lo[1]), "r"(a_frag_t_lo[2]), "r"(a_frag_t_lo[3]), "r"(b_frag_t[2]), "r"(b_frag_t[(2) + 1]));
            }
        }
        #pragma unroll
        for (int hs = 0; hs < 2; hs++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(s_g_addr + (unsigned int)((hs * 16 + lane / 16 * 8) / 64 * 4096 + ((mt * 16 + lane % 16) * 128 + (hs * 16 + lane / 16 * 8) % 64 * 2 ^ ((mt * 16 + lane % 16) * 128 + (hs * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag_lo[0]), "=r"(a_frag_lo[1]), "=r"(a_frag_lo[2]), "=r"(a_frag_lo[3])
                : "r"(s_g_lo_addr + (unsigned int)((hs * 16 + lane / 16 * 8) / 64 * 4096 + ((mt * 16 + lane % 16) * 128 + (hs * 16 + lane / 16 * 8) % 64 * 2 ^ ((mt * 16 + lane % 16) * 128 + (hs * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            #pragma unroll
            for (int nb5 = 0; nb5 < 4; nb5++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_t[0]), "=r"(b_frag_t[1]), "=r"(b_frag_t[2]), "=r"(b_frag_t[3])
                    : "r"(s_q_addr + (unsigned int)((nh * 64 + nb5 * 16 + lane / 16 * 8) / 64 * 4096 + ((hs * 16 + lane % 16) * 128 + (nh * 64 + nb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((hs * 16 + lane % 16) * 128 + (nh * 64 + nb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_dk + nb5 * 8)[0]), "=f"((acc_dk + nb5 * 8)[1]), "=f"((acc_dk + nb5 * 8)[2]), "=f"((acc_dk + nb5 * 8)[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_t[0]), "r"(b_frag_t[1]), "f"(((hs == 0) ? 0.0f : (acc_dk + nb5 * 8)[0])), "f"(((hs == 0) ? 0.0f : (acc_dk + nb5 * 8)[1])), "f"(((hs == 0) ? 0.0f : (acc_dk + nb5 * 8)[2])), "f"(((hs == 0) ? 0.0f : (acc_dk + nb5 * 8)[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_dk + nb5 * 8 + 4)[0]), "=f"((acc_dk + nb5 * 8 + 4)[1]), "=f"((acc_dk + nb5 * 8 + 4)[2]), "=f"((acc_dk + nb5 * 8 + 4)[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_t[2]), "r"(b_frag_t[(2) + 1]), "f"(((hs == 0) ? 0.0f : (acc_dk + nb5 * 8 + 4)[0])), "f"(((hs == 0) ? 0.0f : (acc_dk + nb5 * 8 + 4)[1])), "f"(((hs == 0) ? 0.0f : (acc_dk + nb5 * 8 + 4)[2])), "f"(((hs == 0) ? 0.0f : (acc_dk + nb5 * 8 + 4)[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_dk + nb5 * 8)[0]), "+f"((acc_dk + nb5 * 8)[1]), "+f"((acc_dk + nb5 * 8)[2]), "+f"((acc_dk + nb5 * 8)[3])
                    : "r"(a_frag_lo[0]), "r"(a_frag_lo[1]), "r"(a_frag_lo[2]), "r"(a_frag_lo[3]), "r"(b_frag_t[0]), "r"(b_frag_t[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_dk + nb5 * 8 + 4)[0]), "+f"((acc_dk + nb5 * 8 + 4)[1]), "+f"((acc_dk + nb5 * 8 + 4)[2]), "+f"((acc_dk + nb5 * 8 + 4)[3])
                    : "r"(a_frag_lo[0]), "r"(a_frag_lo[1]), "r"(a_frag_lo[2]), "r"(a_frag_lo[3]), "r"(b_frag_t[2]), "r"(b_frag_t[(2) + 1]));
            }
        }
        long long prow_a = (idx_base + (long long)(blk * 32 + mt * 16 + r_a)) * 128 + (long long)(nh * 64) + (long long)col_q;
        long long prow_b = prow_a + 1024;
        #pragma unroll
        for (int nb6 = 0; nb6 < 8; nb6++) {
            {
                float2 _v2 = make_float2(acc_dk[nb6 * 4 + 0], acc_dk[nb6 * 4 + 1]);
                *reinterpret_cast<float2*>(partial + prow_a + (long long)(nb6 * 8)) = _v2;
            }
            {
                float2 _v2 = make_float2(acc_dk[nb6 * 4 + 2 + 0], acc_dk[nb6 * 4 + 2 + 1]);
                *reinterpret_cast<float2*>(partial + prow_b + (long long)(nb6 * 8)) = _v2;
            }
        }
    }
    int head_a = hg * 16 + r_a;
    int head_b = head_a + 8;
    long long dq_a = (q_row0 + (long long)head_a) * 128 + (long long)(cg * 64) + (long long)col_q;
    long long dq_b = (q_row0 + (long long)head_b) * 128 + (long long)(cg * 64) + (long long)col_q;
    #pragma unroll
    for (int nb7 = 0; nb7 < 8; nb7++) {
        if (head_a < heads) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc_dq[nb7 * 4 + 0], acc_dq[nb7 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(d_index_q))[dq_a + (long long)(nb7 * 8)]) = _pk;
            }
        }
        if (head_b < heads) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc_dq[nb7 * 4 + 2 + 0], acc_dq[nb7 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(d_index_q))[dq_b + (long long)(nb7 * 8)]) = _pk;
            }
        }
    }
    #pragma unroll
    for (int nb8 = 0; nb8 < 4; nb8++) {
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, dw_part[nb8], 4);
        dw_part[nb8] = dw_part[nb8] + _shfl_xor_0;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, dw_part[nb8], 8);
        dw_part[nb8] = dw_part[nb8] + _shfl_xor_1;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, dw_part[nb8], 16);
        dw_part[nb8] = dw_part[nb8] + _shfl_xor_2;
    }
    if (lane_1 < 4) {
        #pragma unroll
        for (int nb9 = 0; nb9 < 2; nb9++) {
            #pragma unroll
            for (int e2 = 0; e2 < 2; e2++) {
                s_dw[mt * 32 + nh * 16 + nb9 * 8 + col_q + e2] = dw_part[nb9 * 2 + e2];
            }
        }
    }
    __syncthreads();
    if (tid_0 < 32) {
        if (tid_0 < heads) {
            float dw_total = s_dw[tid_0];
            #pragma unroll
            for (int m2 = 1; m2 < 2; m2++) {
                dw_total = dw_total + s_dw[m2 * 32 + tid_0];
            }
            d_weights[q_row0 + (long long)tid_0] = dw_total;
        }
    }
}

} // extern "C"
