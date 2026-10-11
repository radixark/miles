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
#define SMEM_S_DO_OFF 0
#define SMEM_S_DO_STAGE_BYTES 16384
#define SMEM_S_DO_STRIDE 16384
#define SMEM_S_VN_OFF 16384
#define SMEM_S_VN_STAGE_BYTES 16384
#define SMEM_S_VN_STRIDE 16384
#define SMEM_S_QN_OFF 65536
#define SMEM_S_QN_STAGE_BYTES 16384
#define SMEM_S_QN_STRIDE 16384
#define SMEM_S_KN_OFF 16384
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 16384
#define SMEM_S_DV_OFF 81920
#define SMEM_S_DV_STAGE_BYTES 16384
#define SMEM_S_DV_STRIDE 16384
#define SMEM_S_H_OFF 32768
#define SMEM_S_H_STAGE_BYTES 32768
#define SMEM_S_H_STRIDE 32768
#define SMEM_S_DH_OFF 65536
#define SMEM_S_DH_STAGE_BYTES 32768
#define SMEM_S_DH_STRIDE 32768
#define SMEM_S_DS_OFF 98304
#define SMEM_S_DS_STAGE_BYTES 8192
#define SMEM_S_DS_STRIDE 8192
#define SMEM_S_G_OFF 106496
#define SMEM_S_G_STAGE_BYTES 256
#define SMEM_S_G_STRIDE 256
#define SMEM_S_DGQ_OFF 106752
#define SMEM_S_DGQ_STAGE_BYTES 512
#define SMEM_S_DGQ_STRIDE 512
#define SMEM_S_DGK_OFF 107264
#define SMEM_S_DGK_STAGE_BYTES 512
#define SMEM_S_DGK_STRIDE 512
#define SMEM_S_RED_OFF 107776
#define SMEM_S_RED_STAGE_BYTES 128
#define SMEM_S_RED_STRIDE 128
#define SMEM_TOTAL 107904
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_chunk_train_f6b472d809aaf3b883ca(__nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ v_new, __nv_bfloat16* __restrict__ do_, __nv_bfloat16* __restrict__ dv2, __nv_bfloat16* __restrict__ h, __nv_bfloat16* __restrict__ dh, float* __restrict__ g_cs, __nv_bfloat16* __restrict__ dq_out, __nv_bfloat16* __restrict__ dk_out, __nv_bfloat16* __restrict__ dw_out, float* __restrict__ dg_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, float scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* s_do = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_DO_OFF);
    const int s_do_addr = smem + SMEM_S_DO_OFF;
    __nv_bfloat16* s_vn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_VN_OFF);
    const int s_vn_addr = smem + SMEM_S_VN_OFF;
    __nv_bfloat16* s_qn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_QN_OFF);
    const int s_qn_addr = smem + SMEM_S_QN_OFF;
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_KN_OFF);
    const int s_kn_addr = smem + SMEM_S_KN_OFF;
    __nv_bfloat16* s_dv = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_DV_OFF);
    const int s_dv_addr = smem + SMEM_S_DV_OFF;
    __nv_bfloat16* s_h = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_H_OFF);
    const int s_h_addr = smem + SMEM_S_H_OFF;
    __nv_bfloat16* s_dh = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_DH_OFF);
    const int s_dh_addr = smem + SMEM_S_DH_OFF;
    __nv_bfloat16* s_ds = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_DS_OFF);
    const int s_ds_addr = smem + SMEM_S_DS_OFF;
    float* s_g = reinterpret_cast<float*>(smem_raw + SMEM_S_G_OFF);
    const int s_g_addr = smem + SMEM_S_G_OFF;
    float* s_dgq = reinterpret_cast<float*>(smem_raw + SMEM_S_DGQ_OFF);
    const int s_dgq_addr = smem + SMEM_S_DGQ_OFF;
    float* s_dgk = reinterpret_cast<float*>(smem_raw + SMEM_S_DGK_OFF);
    const int s_dgk_addr = smem + SMEM_S_DGK_OFF;
    float* s_red = reinterpret_cast<float*>(smem_raw + SMEM_S_RED_OFF);
    const int s_red_addr = smem + SMEM_S_RED_OFF;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int hv = blockIdx.y;
    int tok0 = chunk_start[chunk];
    int n_valid = chunk_len[chunk];
    int group = num_v_heads / num_heads;
    int hq = hv / group;
    int lane_0 = lane;
    int warp_1 = warp;
    int tid_2 = tid;
    int rb = warp_1 % 4;
    int nh = warp_1 / 4;
    int n0 = nh * 64;
    int s0 = nh * 32;
    int row_a = rb * 16 + lane_0 / 4;
    int row_b = row_a + 8;
    int col_q = lane_0 % 4 * 2;
    int r0 = tid / 16;
    int c0 = tid % 16 * 8;
    long long row_stride = (long long)num_v_heads * 128;
    long long src_row = (long long)(tok0 + r0) * row_stride + (long long)hv * 128 + (long long)c0;
    int dst0 = s_do_addr + (unsigned int)(c0 / 64 * 8192 + (r0 * 128 + c0 % 64 * 2 ^ (r0 * 128 + c0 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0), "l"(do_ + src_row), "r"((n_valid > r0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 2048), "l"(do_ + (src_row + 16 * row_stride)), "r"((n_valid > r0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 4096), "l"(do_ + (src_row + 32 * row_stride)), "r"((n_valid > r0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 6144), "l"(do_ + (src_row + 48 * row_stride)), "r"((n_valid > r0 + 48) ? 16 : 0));
    int r0_3 = tid / 16;
    int c0_4 = tid % 16 * 8;
    long long row_stride_5 = (long long)num_v_heads * 128;
    long long src_row_6 = (long long)(tok0 + r0_3) * row_stride_5 + (long long)hv * 128 + (long long)c0_4;
    int dst0_7 = s_vn_addr + (unsigned int)(c0_4 / 64 * 8192 + (r0_3 * 128 + c0_4 % 64 * 2 ^ (r0_3 * 128 + c0_4 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7), "l"(v_new + src_row_6), "r"((n_valid > r0_3) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 2048), "l"(v_new + (src_row_6 + 16 * row_stride_5)), "r"((n_valid > r0_3 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 4096), "l"(v_new + (src_row_6 + 32 * row_stride_5)), "r"((n_valid > r0_3 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 6144), "l"(v_new + (src_row_6 + 48 * row_stride_5)), "r"((n_valid > r0_3 + 48) ? 16 : 0));
    long long state_base = ((long long)chunk * (long long)num_v_heads + (long long)hv) * 16384;
    #pragma unroll
    for (int it = 0; it < 8; it++) {
        int idx = it * 256 + tid_2;
        int h_row = idx / 16;
        int h_col = idx % 16 * 8;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(s_h_addr + (unsigned int)(h_col / 64 * 16384 + (h_row * 128 + h_col % 64 * 2 ^ (h_row * 128 + h_col % 64 * 2 >> 7 & 7) << 4))), "l"(h + (state_base + (long long)h_row * 128 + (long long)h_col)));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(s_dh_addr + (unsigned int)(h_col / 64 * 16384 + (h_row * 128 + h_col % 64 * 2 ^ (h_row * 128 + h_col % 64 * 2 >> 7 & 7) << 4))), "l"(dh + (state_base + (long long)h_row * 128 + (long long)h_col)));
    }
    asm volatile("cp.async.commit_group;");
    if (tid_2 < 64) {
        float gate_val = 0.0f;
        if (tid_2 < n_valid) {
            gate_val = g_cs[(long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv];
        }
        s_g[tid_2] = gate_val;
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    unsigned int a_frag_k[4];
    unsigned int a_frag_mn[4];
    unsigned int b_frag_k[4];
    unsigned int b_frag_mn[4];
    float acc_s[16];
    float acc[32];
    unsigned int pair_a[1];
    unsigned int pair_b[1];
    float g_a = s_g[row_a];
    float g_b = s_g[row_b];
    int _max_0 = ((n_valid) > (1) ? (n_valid) : (1));
    float g_last = s_g[_max_0 - 1];
    long long out_a = ((long long)(tok0 + row_a) * (long long)num_v_heads + (long long)hv) * 128 + (long long)n0 + (long long)col_q;
    long long out_b = ((long long)(tok0 + row_b) * (long long)num_v_heads + (long long)hv) * 128 + (long long)n0 + (long long)col_q;
    #pragma unroll
    for (int kb = 0; kb < 8; kb++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_do_addr + (unsigned int)((kb * 16 + lane / 16 * 8) / 64 * 8192 + ((rb * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 ^ ((rb * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb = 0; nb < 2; nb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                : "r"(s_vn_addr + (unsigned int)((kb * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((s0 + nb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((s0 + nb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb * 8)[0]), "=f"((acc_s + nb * 8)[1]), "=f"((acc_s + nb * 8)[2]), "=f"((acc_s + nb * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8)[0])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8)[1])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8)[2])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb * 8 + 4)[0]), "=f"((acc_s + nb * 8 + 4)[1]), "=f"((acc_s + nb * 8 + 4)[2]), "=f"((acc_s + nb * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8 + 4)[0])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8 + 4)[1])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8 + 4)[2])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8 + 4)[3])));
        }
    }
    #pragma unroll
    for (int nb2 = 0; nb2 < 4; nb2++) {
        #pragma unroll
        for (int e = 0; e < 2; e++) {
            int col = s0 + nb2 * 8 + col_q + e;
            float g_col = s_g[col];
            float val_a = 0.0f;
            float val_b = 0.0f;
            if (col <= row_a) {
                if (row_a < n_valid) {
                    float _exp2_0 = approx_exp2(g_a - g_col);
                    val_a = acc_s[nb2 * 4 + e] * _exp2_0 * scale;
                }
            }
            if (col <= row_b) {
                if (row_b < n_valid) {
                    float _exp2_1 = approx_exp2(g_b - g_col);
                    val_b = acc_s[nb2 * 4 + 2 + e] * _exp2_1 * scale;
                }
            }
            {
                __nv_bfloat16 _bval_0 = __float2bfloat16_rn(val_a);
                uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                uint32_t _addr_0 = static_cast<uint32_t>((s_ds_addr + (unsigned int)(row_a * 128 + col * 2 ^ (row_a * 128 + col * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
            }
            {
                __nv_bfloat16 _bval_1 = __float2bfloat16_rn(val_b);
                uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                uint32_t _addr_1 = static_cast<uint32_t>((s_ds_addr + (unsigned int)(row_b * 128 + col * 2 ^ (row_b * 128 + col * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
            }
        }
    }
    #pragma unroll
    for (int kb5 = 0; kb5 < 8; kb5++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_vn_addr + (unsigned int)((kb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((rb * 16 + lane % 16) * 128 + (kb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((rb * 16 + lane % 16) * 128 + (kb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb9 = 0; nb9 < 4; nb9++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_dh_addr + (unsigned int)((n0 + nb9 * 16 + lane / 16 * 8) / 64 * 16384 + ((kb5 * 16 + lane % 16) * 128 + (n0 + nb9 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb5 * 16 + lane % 16) * 128 + (n0 + nb9 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb9 * 8)[0]), "=f"((acc + nb9 * 8)[1]), "=f"((acc + nb9 * 8)[2]), "=f"((acc + nb9 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8)[0])), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8)[1])), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8)[2])), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb9 * 8 + 4)[0]), "=f"((acc + nb9 * 8 + 4)[1]), "=f"((acc + nb9 * 8 + 4)[2]), "=f"((acc + nb9 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8 + 4)[0])), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8 + 4)[1])), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8 + 4)[2])), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8 + 4)[3])));
        }
    }
    __syncthreads();
    int r0_8 = tid / 16;
    int c0_9 = tid % 16 * 8;
    long long row_stride_10 = (long long)num_heads * 128;
    long long src_row_11 = (long long)(tok0 + r0_8) * row_stride_10 + (long long)hq * 128 + (long long)c0_9;
    int dst0_12 = s_kn_addr + (unsigned int)(c0_9 / 64 * 8192 + (r0_8 * 128 + c0_9 % 64 * 2 ^ (r0_8 * 128 + c0_9 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12), "l"(kn + src_row_11), "r"((n_valid > r0_8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 2048), "l"(kn + (src_row_11 + 16 * row_stride_10)), "r"((n_valid > r0_8 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 4096), "l"(kn + (src_row_11 + 32 * row_stride_10)), "r"((n_valid > r0_8 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 6144), "l"(kn + (src_row_11 + 48 * row_stride_10)), "r"((n_valid > r0_8 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    float hdh = 0.0f;
    int h_row2 = tid_2 / 2;
    int h_col2 = tid_2 % 2 * 64;
    unsigned int word_h[4];
    unsigned int word_dh[4];
    #pragma unroll
    for (int it2 = 0; it2 < 8; it2++) {
        int e_off = (h_col2 + it2 * 8) / 64 * 16384 + (h_row2 * 128 + (h_col2 + it2 * 8) % 64 * 2 ^ (h_row2 * 128 + (h_col2 + it2 * 8) % 64 * 2 >> 7 & 7) << 4);
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&word_h[0])), "=r"(*reinterpret_cast<uint32_t*>(&word_h[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&word_h[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&word_h[(0) + 3]))
            : "r"(s_h_addr + (unsigned int)e_off));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&word_dh[0])), "=r"(*reinterpret_cast<uint32_t*>(&word_dh[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&word_dh[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&word_dh[(0) + 3]))
            : "r"(s_dh_addr + (unsigned int)e_off));
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float _fma_0 = __fmaf_rn(__uint_as_float(word_h[j] << 16), __uint_as_float(word_dh[j] << 16), hdh);
            hdh = _fma_0;
            float _fma_1 = __fmaf_rn(__uint_as_float(word_h[j] & 4294901760u), __uint_as_float(word_dh[j] & 4294901760u), hdh);
            hdh = _fma_1;
        }
    }
    float _warp_reduce_0 = hdh;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
    hdh = _warp_reduce_0;
    if (lane_0 == 0) {
        s_red[warp_1] = hdh;
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    int r0_13 = tid / 16;
    int c0_14 = tid % 16 * 8;
    long long row_stride_15 = (long long)num_heads * 128;
    long long src_row_16 = (long long)(tok0 + r0_13) * row_stride_15 + (long long)hq * 128 + (long long)c0_14;
    int dst0_17 = s_qn_addr + (unsigned int)(c0_14 / 64 * 8192 + (r0_13 * 128 + c0_14 % 64 * 2 ^ (r0_13 * 128 + c0_14 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17), "l"(qn + src_row_16), "r"((n_valid > r0_13) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17 + 2048), "l"(qn + (src_row_16 + 16 * row_stride_15)), "r"((n_valid > r0_13 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17 + 4096), "l"(qn + (src_row_16 + 32 * row_stride_15)), "r"((n_valid > r0_13 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17 + 6144), "l"(qn + (src_row_16 + 48 * row_stride_15)), "r"((n_valid > r0_13 + 48) ? 16 : 0));
    int r0_18 = tid / 16;
    int c0_19 = tid % 16 * 8;
    long long row_stride_20 = (long long)num_v_heads * 128;
    long long src_row_21 = (long long)(tok0 + r0_18) * row_stride_20 + (long long)hv * 128 + (long long)c0_19;
    int dst0_22 = s_dv_addr + (unsigned int)(c0_19 / 64 * 8192 + (r0_18 * 128 + c0_19 % 64 * 2 ^ (r0_18 * 128 + c0_19 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_22), "l"(dv2 + src_row_21), "r"((n_valid > r0_18) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_22 + 2048), "l"(dv2 + (src_row_21 + 16 * row_stride_20)), "r"((n_valid > r0_18 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_22 + 4096), "l"(dv2 + (src_row_21 + 32 * row_stride_20)), "r"((n_valid > r0_18 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_22 + 6144), "l"(dv2 + (src_row_21 + 48 * row_stride_20)), "r"((n_valid > r0_18 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    float decay_a = 0.0f;
    float decay_b = 0.0f;
    if (row_a < n_valid) {
        float _exp2_2 = approx_exp2(g_last - g_a);
        decay_a = _exp2_2;
    }
    if (row_b < n_valid) {
        float _exp2_3 = approx_exp2(g_last - g_b);
        decay_b = _exp2_3;
    }
    float dg_last_part = 0.0f;
    #pragma unroll
    for (int nb10 = 0; nb10 < 8; nb10++) {
        acc[nb10 * 4] = acc[nb10 * 4] * decay_a;
        acc[nb10 * 4 + 1] = acc[nb10 * 4 + 1] * decay_a;
        acc[nb10 * 4 + 2] = acc[nb10 * 4 + 2] * decay_b;
        acc[nb10 * 4 + 3] = acc[nb10 * 4 + 3] * decay_b;
        int kcol = n0 + nb10 * 8 + col_q;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(s_kn_addr + (unsigned int)(kcol / 64 * 8192 + (row_a * 128 + kcol % 64 * 2 ^ (row_a * 128 + kcol % 64 * 2 >> 7 & 7) << 4))));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(s_kn_addr + (unsigned int)(kcol / 64 * 8192 + (row_b * 128 + kcol % 64 * 2 ^ (row_b * 128 + kcol % 64 * 2 >> 7 & 7) << 4))));
        float _fma_2 = __fmaf_rn(acc[nb10 * 4], __uint_as_float(pair_a[0] << 16), dg_last_part);
        dg_last_part = _fma_2;
        float _fma_3 = __fmaf_rn(acc[nb10 * 4 + 2], __uint_as_float(pair_b[0] << 16), dg_last_part);
        dg_last_part = _fma_3;
        float _fma_4 = __fmaf_rn(acc[nb10 * 4 + 1], __uint_as_float(pair_a[0] & 4294901760u), dg_last_part);
        dg_last_part = _fma_4;
        float _fma_5 = __fmaf_rn(acc[nb10 * 4 + 3], __uint_as_float(pair_b[0] & 4294901760u), dg_last_part);
        dg_last_part = _fma_5;
    }
    float _warp_reduce_1 = dg_last_part;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
    dg_last_part = _warp_reduce_1;
    if (lane_0 == 0) {
        s_red[8 + warp_1] = dg_last_part;
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    #pragma unroll
    for (int kb6 = 0; kb6 < 4; kb6++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_mn[0]), "=r"(a_frag_mn[1]), "=r"(a_frag_mn[2]), "=r"(a_frag_mn[3])
            : "r"(s_ds_addr + (unsigned int)((rb * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((kb6 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (rb * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((kb6 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (rb * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb11 = 0; nb11 < 4; nb11++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_qn_addr + (unsigned int)((n0 + nb11 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb6 * 16 + lane % 16) * 128 + (n0 + nb11 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb6 * 16 + lane % 16) * 128 + (n0 + nb11 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc + nb11 * 8)[0]), "+f"((acc + nb11 * 8)[1]), "+f"((acc + nb11 * 8)[2]), "+f"((acc + nb11 * 8)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc + nb11 * 8 + 4)[0]), "+f"((acc + nb11 * 8 + 4)[1]), "+f"((acc + nb11 * 8 + 4)[2]), "+f"((acc + nb11 * 8 + 4)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
        }
    }
    float dgk_a = 0.0f;
    float dgk_b = 0.0f;
    #pragma unroll
    for (int nb12 = 0; nb12 < 8; nb12++) {
        int kcol2 = n0 + nb12 * 8 + col_q;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(s_kn_addr + (unsigned int)(kcol2 / 64 * 8192 + (row_a * 128 + kcol2 % 64 * 2 ^ (row_a * 128 + kcol2 % 64 * 2 >> 7 & 7) << 4))));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(s_kn_addr + (unsigned int)(kcol2 / 64 * 8192 + (row_b * 128 + kcol2 % 64 * 2 ^ (row_b * 128 + kcol2 % 64 * 2 >> 7 & 7) << 4))));
        float _fma_6 = __fmaf_rn(acc[nb12 * 4], __uint_as_float(pair_a[0] << 16), dgk_a);
        dgk_a = _fma_6;
        float _fma_7 = __fmaf_rn(acc[nb12 * 4 + 1], __uint_as_float(pair_a[0] & 4294901760u), dgk_a);
        dgk_a = _fma_7;
        float _fma_8 = __fmaf_rn(acc[nb12 * 4 + 2], __uint_as_float(pair_b[0] << 16), dgk_b);
        dgk_b = _fma_8;
        float _fma_9 = __fmaf_rn(acc[nb12 * 4 + 3], __uint_as_float(pair_b[0] & 4294901760u), dgk_b);
        dgk_b = _fma_9;
        if (row_a < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb12 * 4 + 0], acc[nb12 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dk_out))[out_a + (long long)(nb12 * 8)]) = _pk;
            }
        }
        if (row_b < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb12 * 4 + 2 + 0], acc[nb12 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dk_out))[out_b + (long long)(nb12 * 8)]) = _pk;
            }
        }
    }
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, dgk_a, 1);
    dgk_a = dgk_a + _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, dgk_a, 2);
    dgk_a = dgk_a + _shfl_xor_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, dgk_b, 1);
    dgk_b = dgk_b + _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, dgk_b, 2);
    dgk_b = dgk_b + _shfl_xor_3;
    if (lane_0 % 4 == 0) {
        s_dgk[nh * 64 + row_a] = dgk_a;
        s_dgk[nh * 64 + row_b] = dgk_b;
    }
    #pragma unroll
    for (int kb2 = 0; kb2 < 8; kb2++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_do_addr + (unsigned int)((kb2 * 16 + lane / 16 * 8) / 64 * 8192 + ((rb * 16 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 ^ ((rb * 16 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb3 = 0; nb3 < 4; nb3++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_h_addr + (unsigned int)((n0 + nb3 * 16 + lane / 16 * 8) / 64 * 16384 + ((kb2 * 16 + lane % 16) * 128 + (n0 + nb3 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb2 * 16 + lane % 16) * 128 + (n0 + nb3 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb3 * 8)[0]), "=f"((acc + nb3 * 8)[1]), "=f"((acc + nb3 * 8)[2]), "=f"((acc + nb3 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[0])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[1])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[2])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb3 * 8 + 4)[0]), "=f"((acc + nb3 * 8 + 4)[1]), "=f"((acc + nb3 * 8 + 4)[2]), "=f"((acc + nb3 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[0])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[1])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[2])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[3])));
        }
    }
    float _exp2_4 = approx_exp2(g_a);
    float gain_a = _exp2_4 * scale;
    float _exp2_5 = approx_exp2(g_b);
    float gain_b = _exp2_5 * scale;
    #pragma unroll
    for (int nb4 = 0; nb4 < 8; nb4++) {
        acc[nb4 * 4] = acc[nb4 * 4] * gain_a;
        acc[nb4 * 4 + 1] = acc[nb4 * 4 + 1] * gain_a;
        acc[nb4 * 4 + 2] = acc[nb4 * 4 + 2] * gain_b;
        acc[nb4 * 4 + 3] = acc[nb4 * 4 + 3] * gain_b;
    }
    #pragma unroll
    for (int kb3 = 0; kb3 < 4; kb3++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_ds_addr + (unsigned int)((kb3 * 16 + lane / 16 * 8) / 64 * 8192 + ((rb * 16 + lane % 16) * 128 + (kb3 * 16 + lane / 16 * 8) % 64 * 2 ^ ((rb * 16 + lane % 16) * 128 + (kb3 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb5 = 0; nb5 < 4; nb5++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_kn_addr + (unsigned int)((n0 + nb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb3 * 16 + lane % 16) * 128 + (n0 + nb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb3 * 16 + lane % 16) * 128 + (n0 + nb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc + nb5 * 8)[0]), "+f"((acc + nb5 * 8)[1]), "+f"((acc + nb5 * 8)[2]), "+f"((acc + nb5 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc + nb5 * 8 + 4)[0]), "+f"((acc + nb5 * 8 + 4)[1]), "+f"((acc + nb5 * 8 + 4)[2]), "+f"((acc + nb5 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
        }
    }
    float dgq_a = 0.0f;
    float dgq_b = 0.0f;
    #pragma unroll
    for (int nb6 = 0; nb6 < 8; nb6++) {
        int qcol = n0 + nb6 * 8 + col_q;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(s_qn_addr + (unsigned int)(qcol / 64 * 8192 + (row_a * 128 + qcol % 64 * 2 ^ (row_a * 128 + qcol % 64 * 2 >> 7 & 7) << 4))));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(s_qn_addr + (unsigned int)(qcol / 64 * 8192 + (row_b * 128 + qcol % 64 * 2 ^ (row_b * 128 + qcol % 64 * 2 >> 7 & 7) << 4))));
        float _fma_10 = __fmaf_rn(acc[nb6 * 4], __uint_as_float(pair_a[0] << 16), dgq_a);
        dgq_a = _fma_10;
        float _fma_11 = __fmaf_rn(acc[nb6 * 4 + 1], __uint_as_float(pair_a[0] & 4294901760u), dgq_a);
        dgq_a = _fma_11;
        float _fma_12 = __fmaf_rn(acc[nb6 * 4 + 2], __uint_as_float(pair_b[0] << 16), dgq_b);
        dgq_b = _fma_12;
        float _fma_13 = __fmaf_rn(acc[nb6 * 4 + 3], __uint_as_float(pair_b[0] & 4294901760u), dgq_b);
        dgq_b = _fma_13;
        if (row_a < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb6 * 4 + 0], acc[nb6 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dq_out))[out_a + (long long)(nb6 * 8)]) = _pk;
            }
        }
        if (row_b < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb6 * 4 + 2 + 0], acc[nb6 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dq_out))[out_b + (long long)(nb6 * 8)]) = _pk;
            }
        }
    }
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, dgq_a, 1);
    dgq_a = dgq_a + _shfl_xor_4;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, dgq_a, 2);
    dgq_a = dgq_a + _shfl_xor_5;
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, dgq_b, 1);
    dgq_b = dgq_b + _shfl_xor_6;
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, dgq_b, 2);
    dgq_b = dgq_b + _shfl_xor_7;
    if (lane_0 % 4 == 0) {
        s_dgq[nh * 64 + row_a] = dgq_a;
        s_dgq[nh * 64 + row_b] = dgq_b;
    }
    #pragma unroll
    for (int kb4 = 0; kb4 < 8; kb4++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_dv_addr + (unsigned int)((kb4 * 16 + lane / 16 * 8) / 64 * 8192 + ((rb * 16 + lane % 16) * 128 + (kb4 * 16 + lane / 16 * 8) % 64 * 2 ^ ((rb * 16 + lane % 16) * 128 + (kb4 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb7 = 0; nb7 < 4; nb7++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_h_addr + (unsigned int)((n0 + nb7 * 16 + lane / 16 * 8) / 64 * 16384 + ((kb4 * 16 + lane % 16) * 128 + (n0 + nb7 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb4 * 16 + lane % 16) * 128 + (n0 + nb7 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb7 * 8)[0]), "=f"((acc + nb7 * 8)[1]), "=f"((acc + nb7 * 8)[2]), "=f"((acc + nb7 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8)[0])), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8)[1])), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8)[2])), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb7 * 8 + 4)[0]), "=f"((acc + nb7 * 8 + 4)[1]), "=f"((acc + nb7 * 8 + 4)[2]), "=f"((acc + nb7 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8 + 4)[0])), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8 + 4)[1])), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8 + 4)[2])), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8 + 4)[3])));
        }
    }
    #if __CUDA_ARCH__ >= 1000
    const float2 _scale2_2 = {-1.0f, -1.0f};
    #pragma unroll
    for (int _ls = 0; _ls < 16; _ls++)
        mul_f32x2_inplace(&reinterpret_cast<float2*>(acc)[_ls], _scale2_2);
    #else
    #pragma unroll
    for (int _ls = 0; _ls < 32; _ls++) {
        acc[_ls] = acc[_ls] * -1.0f;
    }
    #endif
    #pragma unroll
    for (int nb8 = 0; nb8 < 8; nb8++) {
        if (row_a < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb8 * 4 + 0], acc[nb8 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dw_out))[out_a + (long long)(nb8 * 8)]) = _pk;
            }
        }
        if (row_b < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb8 * 4 + 2 + 0], acc[nb8 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dw_out))[out_b + (long long)(nb8 * 8)]) = _pk;
            }
        }
    }
    __syncthreads();
    if (tid_2 < n_valid) {
        float hdh_sum = 0.0f;
        float part_sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            hdh_sum = hdh_sum + s_red[r];
        }
        #pragma unroll
        for (int r2 = 0; r2 < 8; r2++) {
            part_sum = part_sum + s_red[8 + r2];
        }
        float _exp2_6 = approx_exp2(g_last);
        float dg_last = hdh_sum * _exp2_6 + part_sum;
        float dg_val = s_dgq[tid_2] + s_dgq[64 + tid_2] - (s_dgk[tid_2] + s_dgk[64 + tid_2]);
        if (tid_2 == n_valid - 1) {
            dg_val = dg_val + dg_last;
        }
        dg_out[(long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv] = dg_val;
    }
}

} // extern "C"
