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
#define SMEM_S_W_OFF 0
#define SMEM_S_W_STAGE_BYTES 16384
#define SMEM_S_W_STRIDE 40960
#define SMEM_S_KN_OFF 16384
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 40960
#define SMEM_S_U_OFF 32768
#define SMEM_S_U_STAGE_BYTES 8192
#define SMEM_S_U_STRIDE 40960
#define SMEM_S_G_OFF 81920
#define SMEM_S_G_STAGE_BYTES 512
#define SMEM_S_G_STRIDE 512
#define SMEM_S_STAGE_OFF 82432
#define SMEM_S_STAGE_STAGE_BYTES 8192
#define SMEM_S_STAGE_STRIDE 8192
#define SMEM_S_RED_OFF 90624
#define SMEM_S_RED_STAGE_BYTES 16
#define SMEM_S_RED_STRIDE 16
#define SMEM_S_X32_OFF 0
#define SMEM_S_X32_STAGE_BYTES 16384
#define SMEM_S_X32_STRIDE 16384
#define SMEM_S_XHI_OFF 16384
#define SMEM_S_XHI_STAGE_BYTES 16384
#define SMEM_S_XHI_STRIDE 16384
#define SMEM_S_XLO_OFF 32768
#define SMEM_S_XLO_STAGE_BYTES 16384
#define SMEM_S_XLO_STRIDE 16384
#define SMEM_S_MHI_OFF 49152
#define SMEM_S_MHI_STAGE_BYTES 8192
#define SMEM_S_MHI_STRIDE 8192
#define SMEM_S_MLO_OFF 57344
#define SMEM_S_MLO_STAGE_BYTES 8192
#define SMEM_S_MLO_STRIDE 8192
#define SMEM_S_M32_OFF 65536
#define SMEM_S_M32_STAGE_BYTES 16384
#define SMEM_S_M32_STRIDE 16384
#define SMEM_TOTAL 90752
#define THREADS 64

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

__global__ __launch_bounds__(THREADS) void
kernel_cake_gdn_chunk_train_bfc4f664df8010d3a111(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ u, float* __restrict__ g_cs, float* __restrict__ h0, __nv_bfloat16* __restrict__ h_out, __nv_bfloat16* __restrict__ v_new, float* __restrict__ final_state, float* __restrict__ seg_state, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seg_chunk_start, int* __restrict__ seg_seq, int* __restrict__ seq_chunk_start, int num_heads, int num_v_heads, int use_initial_state, int store_final_state, int* __restrict__ seg_flag)
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
    __nv_bfloat16* s_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_W_OFF);
    const int s_w_addr = smem + SMEM_S_W_OFF;
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_KN_OFF);
    const int s_kn_addr = smem + SMEM_S_KN_OFF;
    __nv_bfloat16* s_u = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_U_OFF);
    const int s_u_addr = smem + SMEM_S_U_OFF;
    float* s_g = reinterpret_cast<float*>(smem_raw + SMEM_S_G_OFF);
    const int s_g_addr = smem + SMEM_S_G_OFF;
    int* s_stage = reinterpret_cast<int*>(smem_raw + SMEM_S_STAGE_OFF);
    const int s_stage_addr = smem + SMEM_S_STAGE_OFF;
    float* s_red = reinterpret_cast<float*>(smem_raw + SMEM_S_RED_OFF);
    const int s_red_addr = smem + SMEM_S_RED_OFF;
    float* s_x32 = reinterpret_cast<float*>(smem_raw + SMEM_S_X32_OFF);
    const int s_x32_addr = smem + SMEM_S_X32_OFF;
    __nv_bfloat16* s_xhi = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_XHI_OFF);
    const int s_xhi_addr = smem + SMEM_S_XHI_OFF;
    __nv_bfloat16* s_xlo = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_XLO_OFF);
    const int s_xlo_addr = smem + SMEM_S_XLO_OFF;
    __nv_bfloat16* s_mhi = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_MHI_OFF);
    const int s_mhi_addr = smem + SMEM_S_MHI_OFF;
    __nv_bfloat16* s_mlo = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_MLO_OFF);
    const int s_mlo_addr = smem + SMEM_S_MLO_OFF;
    float* s_m32 = reinterpret_cast<float*>(smem_raw + SMEM_S_M32_OFF);
    const int s_m32_addr = smem + SMEM_S_M32_OFF;

    // === Task calls (dependency order) ===
    int seg = blockIdx.x;
    int hv = blockIdx.y;
    int v0 = blockIdx.z * 32;
    int c_begin = seg_chunk_start[seg];
    int c_end = seg_chunk_start[seg + 1];
    int seq = seg_seq[seg];
    int is_first = 0;
    int is_last = 0;
    if (seq >= 0) {
        if (c_begin == seq_chunk_start[seq]) {
            is_first = 1;
        }
        if (c_end == seq_chunk_start[seq + 1]) {
            is_last = 1;
        }
    }
    int group = num_v_heads / num_heads;
    int hq = hv / group;
    int lane_0 = lane;
    int warp_1 = warp;
    int tid_2 = tid;
    int quad = lane_0 % 4;
    int vwarp = warp_1;
    int khalf = 0;
    int kbase = khalf * 128;
    int vloc_a = vwarp * 16 + lane_0 / 4;
    int vrow_a = v0 + vloc_a;
    int vrow_b = vrow_a + 8;
    long long tok_stride = (long long)num_v_heads * 128;
    int stage_base = s_stage_addr + (unsigned int)(warp_1 * 4096);
    int stage_u32_base = warp_1 * 1024;
    unsigned int vw[16];
    float acc_h[64];
    float acc_v[32];
    unsigned int hpk[32];
    unsigned int vpk[16];
    unsigned int u_frag[4];
    unsigned int b_frag_k[4];
    unsigned int b_frag_mn[4];
    float g_next[1];
    float g_far[1];
    int _max_0 = ((seq) > (0) ? (seq) : (0));
    long long in_base = ((long long)_max_0 * (long long)num_v_heads + (long long)hv) * 16384;
    int _max_1 = ((seq) > (0) ? (seq) : (0));
    long long out_base = ((long long)_max_1 * (long long)num_v_heads + (long long)hv) * 16384;
    int ident = 0;
    int active = 1;
    if (is_first != 0) {
        if (use_initial_state != 0) {
            #pragma unroll
            for (int nb = 0; nb < 16; nb++) {
                int kcol = kbase + nb * 8 + quad * 2;
                acc_h[nb * 4] = h0[in_base + (long long)kcol * 128 + (long long)vrow_a];
                acc_h[nb * 4 + 1] = h0[in_base + (long long)(kcol + 1) * 128 + (long long)vrow_a];
                acc_h[nb * 4 + 2] = h0[in_base + (long long)kcol * 128 + (long long)vrow_b];
                acc_h[nb * 4 + 3] = h0[in_base + (long long)(kcol + 1) * 128 + (long long)vrow_b];
            }
        } else {
            acc_h[0] = 0.0f;
            acc_h[1] = 0.0f;
            acc_h[2] = 0.0f;
            acc_h[3] = 0.0f;
            acc_h[4] = 0.0f;
            acc_h[5] = 0.0f;
            acc_h[6] = 0.0f;
            acc_h[7] = 0.0f;
            acc_h[8] = 0.0f;
            acc_h[9] = 0.0f;
            acc_h[10] = 0.0f;
            acc_h[11] = 0.0f;
            acc_h[12] = 0.0f;
            acc_h[13] = 0.0f;
            acc_h[14] = 0.0f;
            acc_h[15] = 0.0f;
            acc_h[16] = 0.0f;
            acc_h[17] = 0.0f;
            acc_h[18] = 0.0f;
            acc_h[19] = 0.0f;
            acc_h[20] = 0.0f;
            acc_h[21] = 0.0f;
            acc_h[22] = 0.0f;
            acc_h[23] = 0.0f;
            acc_h[24] = 0.0f;
            acc_h[25] = 0.0f;
            acc_h[26] = 0.0f;
            acc_h[27] = 0.0f;
            acc_h[28] = 0.0f;
            acc_h[29] = 0.0f;
            acc_h[30] = 0.0f;
            acc_h[31] = 0.0f;
            acc_h[32] = 0.0f;
            acc_h[33] = 0.0f;
            acc_h[34] = 0.0f;
            acc_h[35] = 0.0f;
            acc_h[36] = 0.0f;
            acc_h[37] = 0.0f;
            acc_h[38] = 0.0f;
            acc_h[39] = 0.0f;
            acc_h[40] = 0.0f;
            acc_h[41] = 0.0f;
            acc_h[42] = 0.0f;
            acc_h[43] = 0.0f;
            acc_h[44] = 0.0f;
            acc_h[45] = 0.0f;
            acc_h[46] = 0.0f;
            acc_h[47] = 0.0f;
            acc_h[48] = 0.0f;
            acc_h[49] = 0.0f;
            acc_h[50] = 0.0f;
            acc_h[51] = 0.0f;
            acc_h[52] = 0.0f;
            acc_h[53] = 0.0f;
            acc_h[54] = 0.0f;
            acc_h[55] = 0.0f;
            acc_h[56] = 0.0f;
            acc_h[57] = 0.0f;
            acc_h[58] = 0.0f;
            acc_h[59] = 0.0f;
            acc_h[60] = 0.0f;
            acc_h[61] = 0.0f;
            acc_h[62] = 0.0f;
            acc_h[63] = 0.0f;
        }
    }
    if (is_first == 0) {
        if (seq < 0) {
            acc_h[0] = 0.0f;
            acc_h[1] = 0.0f;
            acc_h[2] = 0.0f;
            acc_h[3] = 0.0f;
            acc_h[4] = 0.0f;
            acc_h[5] = 0.0f;
            acc_h[6] = 0.0f;
            acc_h[7] = 0.0f;
            acc_h[8] = 0.0f;
            acc_h[9] = 0.0f;
            acc_h[10] = 0.0f;
            acc_h[11] = 0.0f;
            acc_h[12] = 0.0f;
            acc_h[13] = 0.0f;
            acc_h[14] = 0.0f;
            acc_h[15] = 0.0f;
            acc_h[16] = 0.0f;
            acc_h[17] = 0.0f;
            acc_h[18] = 0.0f;
            acc_h[19] = 0.0f;
            acc_h[20] = 0.0f;
            acc_h[21] = 0.0f;
            acc_h[22] = 0.0f;
            acc_h[23] = 0.0f;
            acc_h[24] = 0.0f;
            acc_h[25] = 0.0f;
            acc_h[26] = 0.0f;
            acc_h[27] = 0.0f;
            acc_h[28] = 0.0f;
            acc_h[29] = 0.0f;
            acc_h[30] = 0.0f;
            acc_h[31] = 0.0f;
            acc_h[32] = 0.0f;
            acc_h[33] = 0.0f;
            acc_h[34] = 0.0f;
            acc_h[35] = 0.0f;
            acc_h[36] = 0.0f;
            acc_h[37] = 0.0f;
            acc_h[38] = 0.0f;
            acc_h[39] = 0.0f;
            acc_h[40] = 0.0f;
            acc_h[41] = 0.0f;
            acc_h[42] = 0.0f;
            acc_h[43] = 0.0f;
            acc_h[44] = 0.0f;
            acc_h[45] = 0.0f;
            acc_h[46] = 0.0f;
            acc_h[47] = 0.0f;
            acc_h[48] = 0.0f;
            acc_h[49] = 0.0f;
            acc_h[50] = 0.0f;
            acc_h[51] = 0.0f;
            acc_h[52] = 0.0f;
            acc_h[53] = 0.0f;
            acc_h[54] = 0.0f;
            acc_h[55] = 0.0f;
            acc_h[56] = 0.0f;
            acc_h[57] = 0.0f;
            acc_h[58] = 0.0f;
            acc_h[59] = 0.0f;
            acc_h[60] = 0.0f;
            acc_h[61] = 0.0f;
            acc_h[62] = 0.0f;
            acc_h[63] = 0.0f;
        } else {
            long long nvh = (long long)num_v_heads;
            long long hv64 = (long long)hv;
            int lane_1 = lane;
            int warp_2 = warp;
            int quad_3 = lane_1 % 4;
            int row_q = lane_1 / 4;
            int vloc_b = vloc_a + 8;
            int vrow_b_4 = vrow_a + 8;
            int p = seg - 1;
            int p_first = 1;
            if (p > 0) {
                if (seg_seq[p - 1] == seq) {
                    p_first = 0;
                }
            }
            int has_m = seg_flag[(long long)p * nvh + hv64];
            if (p_first != 0) {
                if (use_initial_state == 0) {
                    has_m = 0;
                }
            }
            int stop = 0;
            #pragma unroll 1
            for (int it = 0; it < seg; it++) {
                if (stop == 0) {
                    if (has_m == 0) {
                        stop = 1;
                    } else if (p_first != 0) {
                        stop = 1;
                    } else {
                        p = p - 1;
                        p_first = 1;
                        if (p > 0) {
                            if (seg_seq[p - 1] == seq) {
                                p_first = 0;
                            }
                        }
                        has_m = seg_flag[(long long)p * nvh + hv64];
                        if (p_first != 0) {
                            if (use_initial_state == 0) {
                                has_m = 0;
                            }
                        }
                    }
                }
            }
            int q0 = p + 1;
            if (has_m != 0) {
                q0 = p;
            }
            int n_steps = seg - q0;
            long long pre_base_p = ((long long)p * nvh + hv64) * 32768;
            if (n_steps == 0) {
                #pragma unroll
                for (int nb_1 = 0; nb_1 < 16; nb_1++) {
                    int kcol_1 = kbase + nb_1 * 8 + quad_3 * 2;
                    acc_h[nb_1 * 4] = seg_state[pre_base_p + (long long)kcol_1 * 256 + (long long)vrow_a];
                    acc_h[nb_1 * 4 + 1] = seg_state[pre_base_p + (long long)(kcol_1 + 1) * 256 + (long long)vrow_a];
                    acc_h[nb_1 * 4 + 2] = seg_state[pre_base_p + (long long)kcol_1 * 256 + (long long)vrow_b_4];
                    acc_h[nb_1 * 4 + 3] = seg_state[pre_base_p + (long long)(kcol_1 + 1) * 256 + (long long)vrow_b_4];
                }
            }
            if (n_steps > 0) {
                float acc[16];
                unsigned int a_hi[4];
                unsigned int a_lo[4];
                unsigned int b_hi[4];
                unsigned int b_lo[4];
                int m0 = warp_2 * 16;
                int nb0 = 0;
                if (has_m != 0) {
                    long long init_base = ((long long)seq * nvh + hv64) * 16384;
                    int idx = tid * 4;
                    int row = idx / 32;
                    int col = idx % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row * 128 + (long long)col)));
                    int idx_0 = (64 + tid) * 4;
                    int row_1 = idx_0 / 32;
                    int col_2 = idx_0 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_0 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_1 * 128 + (long long)col_2)));
                    int idx_3 = (128 + tid) * 4;
                    int row_4 = idx_3 / 32;
                    int col_5 = idx_3 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_3 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_4 * 128 + (long long)col_5)));
                    int idx_6 = (192 + tid) * 4;
                    int row_7 = idx_6 / 32;
                    int col_8 = idx_6 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_6 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_7 * 128 + (long long)col_8)));
                    int idx_9 = (256 + tid) * 4;
                    int row_10 = idx_9 / 32;
                    int col_11 = idx_9 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_9 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_10 * 128 + (long long)col_11)));
                    int idx_12 = (320 + tid) * 4;
                    int row_13 = idx_12 / 32;
                    int col_14 = idx_12 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_12 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_13 * 128 + (long long)col_14)));
                    int idx_15 = (384 + tid) * 4;
                    int row_16 = idx_15 / 32;
                    int col_17 = idx_15 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_15 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_16 * 128 + (long long)col_17)));
                    int idx_18 = (448 + tid) * 4;
                    int row_19 = idx_18 / 32;
                    int col_20 = idx_18 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_18 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_19 * 128 + (long long)col_20)));
                    int idx_21 = (512 + tid) * 4;
                    int row_22 = idx_21 / 32;
                    int col_23 = idx_21 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_21 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_22 * 128 + (long long)col_23)));
                    int idx_24 = (576 + tid) * 4;
                    int row_25 = idx_24 / 32;
                    int col_26 = idx_24 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_24 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_25 * 128 + (long long)col_26)));
                    int idx_27 = (640 + tid) * 4;
                    int row_28 = idx_27 / 32;
                    int col_29 = idx_27 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_27 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_28 * 128 + (long long)col_29)));
                    int idx_30 = (704 + tid) * 4;
                    int row_31 = idx_30 / 32;
                    int col_32 = idx_30 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_30 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_31 * 128 + (long long)col_32)));
                    int idx_33 = (768 + tid) * 4;
                    int row_34 = idx_33 / 32;
                    int col_35 = idx_33 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_33 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_34 * 128 + (long long)col_35)));
                    int idx_36 = (832 + tid) * 4;
                    int row_37 = idx_36 / 32;
                    int col_38 = idx_36 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_36 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_37 * 128 + (long long)col_38)));
                    int idx_39 = (896 + tid) * 4;
                    int row_40 = idx_39 / 32;
                    int col_41 = idx_39 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_39 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_40 * 128 + (long long)col_41)));
                    int idx_42 = (960 + tid) * 4;
                    int row_43 = idx_42 / 32;
                    int col_44 = idx_42 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_42 * 4)), "l"(h0 + (init_base + (long long)v0 + (long long)row_43 * 128 + (long long)col_44)));
                } else {
                    int idx_1 = tid * 4;
                    int row_2 = idx_1 / 32;
                    int col_1 = idx_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_2 * 256 + (long long)col_1)));
                    int idx_0_1 = (64 + tid) * 4;
                    int row_1_1 = idx_0_1 / 32;
                    int col_2_1 = idx_0_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_0_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_1_1 * 256 + (long long)col_2_1)));
                    int idx_3_1 = (128 + tid) * 4;
                    int row_4_1 = idx_3_1 / 32;
                    int col_5_1 = idx_3_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_3_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_4_1 * 256 + (long long)col_5_1)));
                    int idx_6_1 = (192 + tid) * 4;
                    int row_7_1 = idx_6_1 / 32;
                    int col_8_1 = idx_6_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_6_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_7_1 * 256 + (long long)col_8_1)));
                    int idx_9_1 = (256 + tid) * 4;
                    int row_10_1 = idx_9_1 / 32;
                    int col_11_1 = idx_9_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_9_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_10_1 * 256 + (long long)col_11_1)));
                    int idx_12_1 = (320 + tid) * 4;
                    int row_13_1 = idx_12_1 / 32;
                    int col_14_1 = idx_12_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_12_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_13_1 * 256 + (long long)col_14_1)));
                    int idx_15_1 = (384 + tid) * 4;
                    int row_16_1 = idx_15_1 / 32;
                    int col_17_1 = idx_15_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_15_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_16_1 * 256 + (long long)col_17_1)));
                    int idx_18_1 = (448 + tid) * 4;
                    int row_19_1 = idx_18_1 / 32;
                    int col_20_1 = idx_18_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_18_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_19_1 * 256 + (long long)col_20_1)));
                    int idx_21_1 = (512 + tid) * 4;
                    int row_22_1 = idx_21_1 / 32;
                    int col_23_1 = idx_21_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_21_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_22_1 * 256 + (long long)col_23_1)));
                    int idx_24_1 = (576 + tid) * 4;
                    int row_25_1 = idx_24_1 / 32;
                    int col_26_1 = idx_24_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_24_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_25_1 * 256 + (long long)col_26_1)));
                    int idx_27_1 = (640 + tid) * 4;
                    int row_28_1 = idx_27_1 / 32;
                    int col_29_1 = idx_27_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_27_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_28_1 * 256 + (long long)col_29_1)));
                    int idx_30_1 = (704 + tid) * 4;
                    int row_31_1 = idx_30_1 / 32;
                    int col_32_1 = idx_30_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_30_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_31_1 * 256 + (long long)col_32_1)));
                    int idx_33_1 = (768 + tid) * 4;
                    int row_34_1 = idx_33_1 / 32;
                    int col_35_1 = idx_33_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_33_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_34_1 * 256 + (long long)col_35_1)));
                    int idx_36_1 = (832 + tid) * 4;
                    int row_37_1 = idx_36_1 / 32;
                    int col_38_1 = idx_36_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_36_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_37_1 * 256 + (long long)col_38_1)));
                    int idx_39_1 = (896 + tid) * 4;
                    int row_40_1 = idx_39_1 / 32;
                    int col_41_1 = idx_39_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_39_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_40_1 * 256 + (long long)col_41_1)));
                    int idx_42_1 = (960 + tid) * 4;
                    int row_43_1 = idx_42_1 / 32;
                    int col_44_1 = idx_42_1 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_42_1 * 4)), "l"(seg_state + (pre_base_p + (long long)v0 + (long long)row_43_1 * 256 + (long long)col_44_1)));
                }
                asm volatile("cp.async.commit_group;");
                asm volatile("cp.async.wait_group 0;");
                __syncthreads();
                float vals[4];
                unsigned int hi[2];
                unsigned int lo[2];
                float res[4];
                #pragma unroll 4
                for (int i = 0; i < 16; i++) {
                    int idx_2 = (i * 64 + tid) * 4;
                    int row_3 = idx_2 / 32;
                    int col_3 = idx_2 % 32;
                    vals[0] = s_x32[idx_2];
                    vals[1] = s_x32[idx_2 + 1];
                    vals[2] = s_x32[idx_2 + 2];
                    vals[3] = s_x32[idx_2 + 3];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
                        hi[_lp] = *(uint32_t*)&_bf2;
                    }
                    res[0] = vals[0] - __uint_as_float(hi[0] << 16);
                    res[1] = vals[1] - __uint_as_float(hi[0] & 4294901760u);
                    res[2] = vals[2] - __uint_as_float(hi[1] << 16);
                    res[3] = vals[3] - __uint_as_float(hi[1] & 4294901760u);
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res[_lp*2 + 0], res[_lp*2+1 + 0]));
                        lo[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_xhi_addr + (unsigned int)(col_3 / 64 * 16384 + (row_3 * 128 + col_3 % 64 * 2 ^ (row_3 * 128 + col_3 % 64 * 2 >> 7 & 7) << 4))), "r"(hi[0]), "r"(hi[1]) : "memory");
                    asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_xlo_addr + (unsigned int)(col_3 / 64 * 16384 + (row_3 * 128 + col_3 % 64 * 2 ^ (row_3 * 128 + col_3 % 64 * 2 >> 7 & 7) << 4))), "r"(lo[0]), "r"(lo[1]) : "memory");
                }
                __syncthreads();
                #pragma unroll 1
                for (int st = 0; st < n_steps; st++) {
                    int q = q0 + st;
                    long long pre_base_q = ((long long)q * nvh + hv64) * 32768;
                    int idx_4 = tid * 4;
                    int row_5 = idx_4 / 32;
                    int col_4 = idx_4 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_4 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_5 * 256 + (long long)col_4)));
                    int idx_0_2 = (64 + tid) * 4;
                    int row_1_2 = idx_0_2 / 32;
                    int col_2_2 = idx_0_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_0_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_1_2 * 256 + (long long)col_2_2)));
                    int idx_3_2 = (128 + tid) * 4;
                    int row_4_2 = idx_3_2 / 32;
                    int col_5_2 = idx_3_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_3_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_4_2 * 256 + (long long)col_5_2)));
                    int idx_6_2 = (192 + tid) * 4;
                    int row_7_2 = idx_6_2 / 32;
                    int col_8_2 = idx_6_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_6_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_7_2 * 256 + (long long)col_8_2)));
                    int idx_9_2 = (256 + tid) * 4;
                    int row_10_2 = idx_9_2 / 32;
                    int col_11_2 = idx_9_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_9_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_10_2 * 256 + (long long)col_11_2)));
                    int idx_12_2 = (320 + tid) * 4;
                    int row_13_2 = idx_12_2 / 32;
                    int col_14_2 = idx_12_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_12_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_13_2 * 256 + (long long)col_14_2)));
                    int idx_15_2 = (384 + tid) * 4;
                    int row_16_2 = idx_15_2 / 32;
                    int col_17_2 = idx_15_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_15_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_16_2 * 256 + (long long)col_17_2)));
                    int idx_18_2 = (448 + tid) * 4;
                    int row_19_2 = idx_18_2 / 32;
                    int col_20_2 = idx_18_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_18_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_19_2 * 256 + (long long)col_20_2)));
                    int idx_21_2 = (512 + tid) * 4;
                    int row_22_2 = idx_21_2 / 32;
                    int col_23_2 = idx_21_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_21_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_22_2 * 256 + (long long)col_23_2)));
                    int idx_24_2 = (576 + tid) * 4;
                    int row_25_2 = idx_24_2 / 32;
                    int col_26_2 = idx_24_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_24_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_25_2 * 256 + (long long)col_26_2)));
                    int idx_27_2 = (640 + tid) * 4;
                    int row_28_2 = idx_27_2 / 32;
                    int col_29_2 = idx_27_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_27_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_28_2 * 256 + (long long)col_29_2)));
                    int idx_30_2 = (704 + tid) * 4;
                    int row_31_2 = idx_30_2 / 32;
                    int col_32_2 = idx_30_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_30_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_31_2 * 256 + (long long)col_32_2)));
                    int idx_33_2 = (768 + tid) * 4;
                    int row_34_2 = idx_33_2 / 32;
                    int col_35_2 = idx_33_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_33_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_34_2 * 256 + (long long)col_35_2)));
                    int idx_36_2 = (832 + tid) * 4;
                    int row_37_2 = idx_36_2 / 32;
                    int col_38_2 = idx_36_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_36_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_37_2 * 256 + (long long)col_38_2)));
                    int idx_39_2 = (896 + tid) * 4;
                    int row_40_2 = idx_39_2 / 32;
                    int col_41_2 = idx_39_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_39_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_40_2 * 256 + (long long)col_41_2)));
                    int idx_42_2 = (960 + tid) * 4;
                    int row_43_2 = idx_42_2 / 32;
                    int col_44_2 = idx_42_2 % 32;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_x32_addr + (unsigned int)(idx_42_2 * 4)), "l"(seg_state + (pre_base_q + (long long)v0 + (long long)row_43_2 * 256 + (long long)col_44_2)));
                    int idx_45 = tid * 4;
                    int row_46 = idx_45 / 128;
                    int col_47 = idx_45 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_45 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_46 * 256 + (long long)col_47)));
                    int idx_48 = (64 + tid) * 4;
                    int row_49 = idx_48 / 128;
                    int col_50 = idx_48 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_48 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_49 * 256 + (long long)col_50)));
                    int idx_51 = (128 + tid) * 4;
                    int row_52 = idx_51 / 128;
                    int col_53 = idx_51 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_51 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_52 * 256 + (long long)col_53)));
                    int idx_54 = (192 + tid) * 4;
                    int row_55 = idx_54 / 128;
                    int col_56 = idx_54 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_54 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_55 * 256 + (long long)col_56)));
                    int idx_57 = (256 + tid) * 4;
                    int row_58 = idx_57 / 128;
                    int col_59 = idx_57 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_57 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_58 * 256 + (long long)col_59)));
                    int idx_60 = (320 + tid) * 4;
                    int row_61 = idx_60 / 128;
                    int col_62 = idx_60 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_60 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_61 * 256 + (long long)col_62)));
                    int idx_63 = (384 + tid) * 4;
                    int row_64 = idx_63 / 128;
                    int col_65 = idx_63 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_63 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_64 * 256 + (long long)col_65)));
                    int idx_66 = (448 + tid) * 4;
                    int row_67 = idx_66 / 128;
                    int col_68 = idx_66 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_66 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_67 * 256 + (long long)col_68)));
                    int idx_69 = (512 + tid) * 4;
                    int row_70 = idx_69 / 128;
                    int col_71 = idx_69 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_69 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_70 * 256 + (long long)col_71)));
                    int idx_72 = (576 + tid) * 4;
                    int row_73 = idx_72 / 128;
                    int col_74 = idx_72 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_72 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_73 * 256 + (long long)col_74)));
                    int idx_75 = (640 + tid) * 4;
                    int row_76 = idx_75 / 128;
                    int col_77 = idx_75 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_75 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_76 * 256 + (long long)col_77)));
                    int idx_78 = (704 + tid) * 4;
                    int row_79 = idx_78 / 128;
                    int col_80 = idx_78 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_78 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_79 * 256 + (long long)col_80)));
                    int idx_81 = (768 + tid) * 4;
                    int row_82 = idx_81 / 128;
                    int col_83 = idx_81 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_81 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_82 * 256 + (long long)col_83)));
                    int idx_84 = (832 + tid) * 4;
                    int row_85 = idx_84 / 128;
                    int col_86 = idx_84 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_84 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_85 * 256 + (long long)col_86)));
                    int idx_87 = (896 + tid) * 4;
                    int row_88 = idx_87 / 128;
                    int col_89 = idx_87 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_87 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_88 * 256 + (long long)col_89)));
                    int idx_90 = (960 + tid) * 4;
                    int row_91 = idx_90 / 128;
                    int col_92 = idx_90 % 128;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(s_m32_addr + (unsigned int)(idx_90 * 4)), "l"(seg_state + (pre_base_q + 128 + (long long)row_91 * 256 + (long long)col_92)));
                    asm volatile("cp.async.commit_group;");
                    #pragma unroll
                    for (int sl = 0; sl < 4; sl++) {
                        asm volatile("cp.async.wait_group 0;");
                        __syncthreads();
                        float vals_0[4];
                        unsigned int hi_1[2];
                        unsigned int lo_2[2];
                        float res_3[4];
                        #pragma unroll 4
                        for (int i_1 = 0; i_1 < 16; i_1++) {
                            int idx_1_1 = (i_1 * 64 + tid) * 4;
                            int row_2_1 = idx_1_1 / 128;
                            int col_3_1 = idx_1_1 % 128;
                            vals_0[0] = s_m32[idx_1_1];
                            vals_0[1] = s_m32[idx_1_1 + 1];
                            vals_0[2] = s_m32[idx_1_1 + 2];
                            vals_0[3] = s_m32[idx_1_1 + 3];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0[_lp*2 + 0], vals_0[_lp*2+1 + 0]));
                                hi_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            res_3[0] = vals_0[0] - __uint_as_float(hi_1[0] << 16);
                            res_3[1] = vals_0[1] - __uint_as_float(hi_1[0] & 4294901760u);
                            res_3[2] = vals_0[2] - __uint_as_float(hi_1[1] << 16);
                            res_3[3] = vals_0[3] - __uint_as_float(hi_1[1] & 4294901760u);
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3[_lp*2 + 0], res_3[_lp*2+1 + 0]));
                                lo_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mhi_addr + (unsigned int)(col_3_1 / 64 * 4096 + (row_2_1 * 128 + col_3_1 % 64 * 2 ^ (row_2_1 * 128 + col_3_1 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1[0]), "r"(hi_1[1]) : "memory");
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_mlo_addr + (unsigned int)(col_3_1 / 64 * 4096 + (row_2_1 * 128 + col_3_1 % 64 * 2 ^ (row_2_1 * 128 + col_3_1 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2[0]), "r"(lo_2[1]) : "memory");
                        }
                        __syncthreads();
                        if (sl + 1 < 4) {
                            int idx_1_2 = tid * 4;
                            int row_2_2 = idx_1_2 / 128;
                            int col_3_2 = idx_1_2 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_1_2 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_2_2 * 256 + (long long)col_3_2)));
                            int idx_4_1 = (64 + tid) * 4;
                            int row_5_1 = idx_4_1 / 128;
                            int col_6 = idx_4_1 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_4_1 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_5_1 * 256 + (long long)col_6)));
                            int idx_7 = (128 + tid) * 4;
                            int row_8 = idx_7 / 128;
                            int col_9 = idx_7 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_7 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_8 * 256 + (long long)col_9)));
                            int idx_10 = (192 + tid) * 4;
                            int row_11 = idx_10 / 128;
                            int col_12 = idx_10 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_10 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_11 * 256 + (long long)col_12)));
                            int idx_13 = (256 + tid) * 4;
                            int row_14 = idx_13 / 128;
                            int col_15 = idx_13 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_13 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_14 * 256 + (long long)col_15)));
                            int idx_16 = (320 + tid) * 4;
                            int row_17 = idx_16 / 128;
                            int col_18 = idx_16 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_16 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_17 * 256 + (long long)col_18)));
                            int idx_19 = (384 + tid) * 4;
                            int row_20 = idx_19 / 128;
                            int col_21 = idx_19 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_19 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_20 * 256 + (long long)col_21)));
                            int idx_22 = (448 + tid) * 4;
                            int row_23 = idx_22 / 128;
                            int col_24 = idx_22 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_22 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_23 * 256 + (long long)col_24)));
                            int idx_25 = (512 + tid) * 4;
                            int row_26 = idx_25 / 128;
                            int col_27 = idx_25 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_25 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_26 * 256 + (long long)col_27)));
                            int idx_28 = (576 + tid) * 4;
                            int row_29 = idx_28 / 128;
                            int col_30 = idx_28 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_28 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_29 * 256 + (long long)col_30)));
                            int idx_31 = (640 + tid) * 4;
                            int row_32 = idx_31 / 128;
                            int col_33 = idx_31 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_31 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_32 * 256 + (long long)col_33)));
                            int idx_34 = (704 + tid) * 4;
                            int row_35 = idx_34 / 128;
                            int col_36 = idx_34 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_34 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_35 * 256 + (long long)col_36)));
                            int idx_37 = (768 + tid) * 4;
                            int row_38 = idx_37 / 128;
                            int col_39 = idx_37 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_37 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_38 * 256 + (long long)col_39)));
                            int idx_40 = (832 + tid) * 4;
                            int row_41 = idx_40 / 128;
                            int col_42 = idx_40 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_40 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_41 * 256 + (long long)col_42)));
                            int idx_43 = (896 + tid) * 4;
                            int row_44 = idx_43 / 128;
                            int col_45 = idx_43 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_43 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_44 * 256 + (long long)col_45)));
                            int idx_46 = (960 + tid) * 4;
                            int row_47 = idx_46 / 128;
                            int col_48 = idx_46 % 128;
                            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                                :: "r"(s_m32_addr + (unsigned int)(idx_46 * 4)), "l"(seg_state + (pre_base_q + (long long)((sl + 1) * 32 * 256) + 128 + (long long)row_47 * 256 + (long long)col_48)));
                            asm volatile("cp.async.commit_group;");
                        }
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        acc[4] = 0.0f;
                        acc[5] = 0.0f;
                        acc[6] = 0.0f;
                        acc[7] = 0.0f;
                        acc[8] = 0.0f;
                        acc[9] = 0.0f;
                        acc[10] = 0.0f;
                        acc[11] = 0.0f;
                        acc[12] = 0.0f;
                        acc[13] = 0.0f;
                        acc[14] = 0.0f;
                        acc[15] = 0.0f;
                        #pragma unroll
                        for (int kk = 0; kk < 8; kk++) {
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(a_hi[0]), "=r"(a_hi[1]), "=r"(a_hi[2]), "=r"(a_hi[3])
                                : "r"(s_mhi_addr + (unsigned int)((kk * 16 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (kk * 16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (kk * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                                : "memory");
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(a_lo[0]), "=r"(a_lo[1]), "=r"(a_lo[2]), "=r"(a_lo[3])
                                : "r"(s_mlo_addr + (unsigned int)((kk * 16 + lane / 16 * 8) / 64 * 4096 + ((m0 + lane % 16) * 128 + (kk * 16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (kk * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                                : "memory");
                            #pragma unroll
                            for (int j = 0; j < 2; j++) {
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(b_hi[0]), "=r"(b_hi[1]), "=r"(b_hi[2]), "=r"(b_hi[3])
                                    : "r"(s_xhi_addr + (unsigned int)(((nb0 + j) * 16 + lane / 16 * 8) / 64 * 16384 + ((kk * 16 + lane % 16) * 128 + ((nb0 + j) * 16 + lane / 16 * 8) % 64 * 2 ^ ((kk * 16 + lane % 16) * 128 + ((nb0 + j) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                                    : "memory");
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(b_lo[0]), "=r"(b_lo[1]), "=r"(b_lo[2]), "=r"(b_lo[3])
                                    : "r"(s_xlo_addr + (unsigned int)(((nb0 + j) * 16 + lane / 16 * 8) / 64 * 16384 + ((kk * 16 + lane % 16) * 128 + ((nb0 + j) * 16 + lane / 16 * 8) % 64 * 2 ^ ((kk * 16 + lane % 16) * 128 + ((nb0 + j) * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                                    : "memory");
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                    : "+f"((acc + j * 8)[0]), "+f"((acc + j * 8)[1]), "+f"((acc + j * 8)[2]), "+f"((acc + j * 8)[3])
                                    : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                    : "+f"((acc + j * 8 + 4)[0]), "+f"((acc + j * 8 + 4)[1]), "+f"((acc + j * 8 + 4)[2]), "+f"((acc + j * 8 + 4)[3])
                                    : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                    : "+f"((acc + j * 8)[0]), "+f"((acc + j * 8)[1]), "+f"((acc + j * 8)[2]), "+f"((acc + j * 8)[3])
                                    : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[0]), "r"(b_lo[1]));
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                    : "+f"((acc + j * 8 + 4)[0]), "+f"((acc + j * 8 + 4)[1]), "+f"((acc + j * 8 + 4)[2]), "+f"((acc + j * 8 + 4)[3])
                                    : "r"(a_hi[0]), "r"(a_hi[1]), "r"(a_hi[2]), "r"(a_hi[3]), "r"(b_lo[2]), "r"(b_lo[(2) + 1]));
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                    : "+f"((acc + j * 8)[0]), "+f"((acc + j * 8)[1]), "+f"((acc + j * 8)[2]), "+f"((acc + j * 8)[3])
                                    : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[0]), "r"(b_hi[1]));
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                    : "+f"((acc + j * 8 + 4)[0]), "+f"((acc + j * 8 + 4)[1]), "+f"((acc + j * 8 + 4)[2]), "+f"((acc + j * 8 + 4)[3])
                                    : "r"(a_lo[0]), "r"(a_lo[1]), "r"(a_lo[2]), "r"(a_lo[3]), "r"(b_hi[2]), "r"(b_hi[(2) + 1]));
                            }
                        }
                        int krow = sl * 32 + m0 + row_q;
                        #pragma unroll
                        for (int j2 = 0; j2 < 2; j2++) {
                            #pragma unroll
                            for (int half = 0; half < 2; half++) {
                                int colv = (nb0 + j2) * 16 + half * 8 + quad_3 * 2;
                                int ia = krow * 32 + colv;
                                int ib = (krow + 8) * 32 + colv;
                                s_x32[ia] = acc[j2 * 8 + half * 4] + s_x32[ia];
                                s_x32[ia + 1] = acc[j2 * 8 + half * 4 + 1] + s_x32[ia + 1];
                                s_x32[ib] = acc[j2 * 8 + half * 4 + 2] + s_x32[ib];
                                s_x32[ib + 1] = acc[j2 * 8 + half * 4 + 3] + s_x32[ib + 1];
                            }
                        }
                    }
                    __syncthreads();
                    if (n_steps > st + 1) {
                        float vals_0_1[4];
                        unsigned int hi_1_1[2];
                        unsigned int lo_2_1[2];
                        float res_3_1[4];
                        #pragma unroll 4
                        for (int i_2 = 0; i_2 < 16; i_2++) {
                            int idx_1_3 = (i_2 * 64 + tid) * 4;
                            int row_2_3 = idx_1_3 / 32;
                            int col_3_3 = idx_1_3 % 32;
                            vals_0_1[0] = s_x32[idx_1_3];
                            vals_0_1[1] = s_x32[idx_1_3 + 1];
                            vals_0_1[2] = s_x32[idx_1_3 + 2];
                            vals_0_1[3] = s_x32[idx_1_3 + 3];
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals_0_1[_lp*2 + 0], vals_0_1[_lp*2+1 + 0]));
                                hi_1_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            res_3_1[0] = vals_0_1[0] - __uint_as_float(hi_1_1[0] << 16);
                            res_3_1[1] = vals_0_1[1] - __uint_as_float(hi_1_1[0] & 4294901760u);
                            res_3_1[2] = vals_0_1[2] - __uint_as_float(hi_1_1[1] << 16);
                            res_3_1[3] = vals_0_1[3] - __uint_as_float(hi_1_1[1] & 4294901760u);
                            #pragma unroll
                            for (int _lp = 0; _lp < 2; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(res_3_1[_lp*2 + 0], res_3_1[_lp*2+1 + 0]));
                                lo_2_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_xhi_addr + (unsigned int)(col_3_3 / 64 * 16384 + (row_2_3 * 128 + col_3_3 % 64 * 2 ^ (row_2_3 * 128 + col_3_3 % 64 * 2 >> 7 & 7) << 4))), "r"(hi_1_1[0]), "r"(hi_1_1[1]) : "memory");
                            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_xlo_addr + (unsigned int)(col_3_3 / 64 * 16384 + (row_2_3 * 128 + col_3_3 % 64 * 2 ^ (row_2_3 * 128 + col_3_3 % 64 * 2 >> 7 & 7) << 4))), "r"(lo_2_1[0]), "r"(lo_2_1[1]) : "memory");
                        }
                        __syncthreads();
                    }
                }
                #pragma unroll
                for (int nb5 = 0; nb5 < 16; nb5++) {
                    int kcol5 = kbase + nb5 * 8 + quad_3 * 2;
                    acc_h[nb5 * 4] = s_x32[kcol5 * 32 + vloc_a];
                    acc_h[nb5 * 4 + 1] = s_x32[(kcol5 + 1) * 32 + vloc_a];
                    acc_h[nb5 * 4 + 2] = s_x32[kcol5 * 32 + vloc_b];
                    acc_h[nb5 * 4 + 3] = s_x32[(kcol5 + 1) * 32 + vloc_b];
                }
                __syncthreads();
            }
        }
    }
    int num_chunks = c_end - c_begin;
    int c_read = c_begin;
    if (c_end <= c_begin) {
        c_read = 0;
    }
    int seq_tok0 = chunk_start[c_read];
    int _max_2 = ((c_end - 1) > (0) ? (c_end - 1) : (0));
    int c_last = _max_2;
    int seq_len = chunk_start[c_last] + chunk_len[c_last] - seq_tok0;
    int stage_off = 0;
    int g_off = 0;
    int _min_0 = ((seq_len) < (64) ? (seq_len) : (64));
    int r0 = tid / 8;
    int c0 = tid % 8 * 16;
    long long row_stride = (long long)num_v_heads * 128;
    long long src_row = (long long)(seq_tok0 + r0) * row_stride + (long long)hv * 128 + (long long)c0;
    int dst0 = s_w_addr + (unsigned int)stage_off + (unsigned int)(c0 / 64 * 8192 + (r0 * 128 + c0 % 64 * 2 ^ (r0 * 128 + c0 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0), "l"(w + src_row), "r"((_min_0 > r0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 1024), "l"(w + (src_row + 8 * row_stride)), "r"((_min_0 > r0 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 2048), "l"(w + (src_row + 16 * row_stride)), "r"((_min_0 > r0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 3072), "l"(w + (src_row + 24 * row_stride)), "r"((_min_0 > r0 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 4096), "l"(w + (src_row + 32 * row_stride)), "r"((_min_0 > r0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 5120), "l"(w + (src_row + 40 * row_stride)), "r"((_min_0 > r0 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 6144), "l"(w + (src_row + 48 * row_stride)), "r"((_min_0 > r0 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 7168), "l"(w + (src_row + 56 * row_stride)), "r"((_min_0 > r0 + 56) ? 16 : 0));
    int dst0_3 = s_w_addr + (unsigned int)stage_off + (unsigned int)((c0 + 8) / 64 * 8192 + (r0 * 128 + (c0 + 8) % 64 * 2 ^ (r0 * 128 + (c0 + 8) % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_3), "l"(w + (src_row + 8)), "r"((_min_0 > r0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_3 + 1024), "l"(w + (src_row + 8 * row_stride + 8)), "r"((_min_0 > r0 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_3 + 2048), "l"(w + (src_row + 16 * row_stride + 8)), "r"((_min_0 > r0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_3 + 3072), "l"(w + (src_row + 24 * row_stride + 8)), "r"((_min_0 > r0 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_3 + 4096), "l"(w + (src_row + 32 * row_stride + 8)), "r"((_min_0 > r0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_3 + 5120), "l"(w + (src_row + 40 * row_stride + 8)), "r"((_min_0 > r0 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_3 + 6144), "l"(w + (src_row + 48 * row_stride + 8)), "r"((_min_0 > r0 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_3 + 7168), "l"(w + (src_row + 56 * row_stride + 8)), "r"((_min_0 > r0 + 56) ? 16 : 0));
    int r0_4 = tid / 8;
    int c0_5 = tid % 8 * 16;
    long long row_stride_6 = (long long)num_heads * 128;
    long long src_row_7 = (long long)(seq_tok0 + r0_4) * row_stride_6 + (long long)hq * 128 + (long long)c0_5;
    int dst0_8 = s_kn_addr + (unsigned int)stage_off + (unsigned int)(c0_5 / 64 * 8192 + (r0_4 * 128 + c0_5 % 64 * 2 ^ (r0_4 * 128 + c0_5 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_8), "l"(kn + src_row_7), "r"((_min_0 > r0_4) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_8 + 1024), "l"(kn + (src_row_7 + 8 * row_stride_6)), "r"((_min_0 > r0_4 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_8 + 2048), "l"(kn + (src_row_7 + 16 * row_stride_6)), "r"((_min_0 > r0_4 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_8 + 3072), "l"(kn + (src_row_7 + 24 * row_stride_6)), "r"((_min_0 > r0_4 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_8 + 4096), "l"(kn + (src_row_7 + 32 * row_stride_6)), "r"((_min_0 > r0_4 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_8 + 5120), "l"(kn + (src_row_7 + 40 * row_stride_6)), "r"((_min_0 > r0_4 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_8 + 6144), "l"(kn + (src_row_7 + 48 * row_stride_6)), "r"((_min_0 > r0_4 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_8 + 7168), "l"(kn + (src_row_7 + 56 * row_stride_6)), "r"((_min_0 > r0_4 + 56) ? 16 : 0));
    int dst0_9 = s_kn_addr + (unsigned int)stage_off + (unsigned int)((c0_5 + 8) / 64 * 8192 + (r0_4 * 128 + (c0_5 + 8) % 64 * 2 ^ (r0_4 * 128 + (c0_5 + 8) % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_9), "l"(kn + (src_row_7 + 8)), "r"((_min_0 > r0_4) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_9 + 1024), "l"(kn + (src_row_7 + 8 * row_stride_6 + 8)), "r"((_min_0 > r0_4 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_9 + 2048), "l"(kn + (src_row_7 + 16 * row_stride_6 + 8)), "r"((_min_0 > r0_4 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_9 + 3072), "l"(kn + (src_row_7 + 24 * row_stride_6 + 8)), "r"((_min_0 > r0_4 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_9 + 4096), "l"(kn + (src_row_7 + 32 * row_stride_6 + 8)), "r"((_min_0 > r0_4 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_9 + 5120), "l"(kn + (src_row_7 + 40 * row_stride_6 + 8)), "r"((_min_0 > r0_4 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_9 + 6144), "l"(kn + (src_row_7 + 48 * row_stride_6 + 8)), "r"((_min_0 > r0_4 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_9 + 7168), "l"(kn + (src_row_7 + 56 * row_stride_6 + 8)), "r"((_min_0 > r0_4 + 56) ? 16 : 0));
    int r0_10 = tid / 4;
    int c0_11 = tid % 4 * 8;
    long long row_stride_12 = (long long)num_v_heads * 128;
    long long src_row_13 = (long long)(seq_tok0 + r0_10) * row_stride_12 + (long long)hv * 128 + (long long)v0 + (long long)c0_11;
    int dst0_14 = s_u_addr + (unsigned int)stage_off + (unsigned int)(c0_11 / 64 * 8192 + (r0_10 * 128 + c0_11 % 64 * 2 ^ (r0_10 * 128 + c0_11 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_14), "l"(u + src_row_13), "r"((_min_0 > r0_10) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_14 + 2048), "l"(u + (src_row_13 + 16 * row_stride_12)), "r"((_min_0 > r0_10 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_14 + 4096), "l"(u + (src_row_13 + 32 * row_stride_12)), "r"((_min_0 > r0_10 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_14 + 6144), "l"(u + (src_row_13 + 48 * row_stride_12)), "r"((_min_0 > r0_10 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    int _min_1 = ((seq_len) < (64) ? (seq_len) : (64));
    int tok = tid;
    if (tok < 64) {
        float gate_val = 0.0f;
        if (tok < _min_1) {
            gate_val = g_cs[(long long)(seq_tok0 + tok) * (long long)num_v_heads + (long long)hv];
        }
        s_g[tok] = gate_val;
    }
    int _min_2 = ((seq_len - 64) < (64) ? (seq_len - 64) : (64));
    int tok_15 = tid;
    g_next[0] = 0.0f;
    if (tok_15 < _min_2) {
        g_next[0] = g_cs[(long long)(seq_tok0 + 64 + tok_15) * (long long)num_v_heads + (long long)hv];
    }
    #pragma unroll 1
    for (int it_chunk = 0; it_chunk < num_chunks; it_chunk++) {
        int chunk = c_begin + it_chunk;
        int tok0 = seq_tok0 + it_chunk * 64;
        int _min_3 = ((seq_len - it_chunk * 64) < (64) ? (seq_len - it_chunk * 64) : (64));
        int n_valid = _min_3;
        int next_off = 40960 - stage_off;
        int next_g_off = 64 - g_off;
        int _min_4 = ((seq_len - (it_chunk + 2) * 64) < (64) ? (seq_len - (it_chunk + 2) * 64) : (64));
        int tok_0 = tid;
        g_far[0] = 0.0f;
        if (tok_0 < _min_4) {
            g_far[0] = g_cs[(long long)(tok0 + 128 + tok_0) * (long long)num_v_heads + (long long)hv];
        }
        asm volatile("cp.async.wait_group 0;");
        #pragma unroll
        for (int _lp = 0; _lp < 32; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_h[_lp*2 + 0], acc_h[_lp*2+1 + 0]));
            hpk[_lp] = *(uint32_t*)&_bf2;
        }
        __syncthreads();
        if (num_chunks > it_chunk + 1) {
            int next_tok0 = tok0 + 64;
            int _min_5 = ((seq_len - (it_chunk + 1) * 64) < (64) ? (seq_len - (it_chunk + 1) * 64) : (64));
            int next_len = _min_5;
            int r0_0 = tid / 8;
            int c0_1 = tid % 8 * 16;
            long long row_stride_2 = (long long)num_v_heads * 128;
            long long src_row_3 = (long long)(next_tok0 + r0_0) * row_stride_2 + (long long)hv * 128 + (long long)c0_1;
            int dst0_4 = s_w_addr + (unsigned int)next_off + (unsigned int)(c0_1 / 64 * 8192 + (r0_0 * 128 + c0_1 % 64 * 2 ^ (r0_0 * 128 + c0_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4), "l"(w + src_row_3), "r"((next_len > r0_0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 1024), "l"(w + (src_row_3 + 8 * row_stride_2)), "r"((next_len > r0_0 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 2048), "l"(w + (src_row_3 + 16 * row_stride_2)), "r"((next_len > r0_0 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 3072), "l"(w + (src_row_3 + 24 * row_stride_2)), "r"((next_len > r0_0 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 4096), "l"(w + (src_row_3 + 32 * row_stride_2)), "r"((next_len > r0_0 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 5120), "l"(w + (src_row_3 + 40 * row_stride_2)), "r"((next_len > r0_0 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 6144), "l"(w + (src_row_3 + 48 * row_stride_2)), "r"((next_len > r0_0 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 7168), "l"(w + (src_row_3 + 56 * row_stride_2)), "r"((next_len > r0_0 + 56) ? 16 : 0));
            int dst0_5 = s_w_addr + (unsigned int)next_off + (unsigned int)((c0_1 + 8) / 64 * 8192 + (r0_0 * 128 + (c0_1 + 8) % 64 * 2 ^ (r0_0 * 128 + (c0_1 + 8) % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5), "l"(w + (src_row_3 + 8)), "r"((next_len > r0_0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 1024), "l"(w + (src_row_3 + 8 * row_stride_2 + 8)), "r"((next_len > r0_0 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 2048), "l"(w + (src_row_3 + 16 * row_stride_2 + 8)), "r"((next_len > r0_0 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 3072), "l"(w + (src_row_3 + 24 * row_stride_2 + 8)), "r"((next_len > r0_0 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 4096), "l"(w + (src_row_3 + 32 * row_stride_2 + 8)), "r"((next_len > r0_0 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 5120), "l"(w + (src_row_3 + 40 * row_stride_2 + 8)), "r"((next_len > r0_0 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 6144), "l"(w + (src_row_3 + 48 * row_stride_2 + 8)), "r"((next_len > r0_0 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_5 + 7168), "l"(w + (src_row_3 + 56 * row_stride_2 + 8)), "r"((next_len > r0_0 + 56) ? 16 : 0));
            int r0_6 = tid / 8;
            int c0_7 = tid % 8 * 16;
            long long row_stride_8 = (long long)num_heads * 128;
            long long src_row_9 = (long long)(next_tok0 + r0_6) * row_stride_8 + (long long)hq * 128 + (long long)c0_7;
            int dst0_10 = s_kn_addr + (unsigned int)next_off + (unsigned int)(c0_7 / 64 * 8192 + (r0_6 * 128 + c0_7 % 64 * 2 ^ (r0_6 * 128 + c0_7 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_10), "l"(kn + src_row_9), "r"((next_len > r0_6) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_10 + 1024), "l"(kn + (src_row_9 + 8 * row_stride_8)), "r"((next_len > r0_6 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_10 + 2048), "l"(kn + (src_row_9 + 16 * row_stride_8)), "r"((next_len > r0_6 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_10 + 3072), "l"(kn + (src_row_9 + 24 * row_stride_8)), "r"((next_len > r0_6 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_10 + 4096), "l"(kn + (src_row_9 + 32 * row_stride_8)), "r"((next_len > r0_6 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_10 + 5120), "l"(kn + (src_row_9 + 40 * row_stride_8)), "r"((next_len > r0_6 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_10 + 6144), "l"(kn + (src_row_9 + 48 * row_stride_8)), "r"((next_len > r0_6 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_10 + 7168), "l"(kn + (src_row_9 + 56 * row_stride_8)), "r"((next_len > r0_6 + 56) ? 16 : 0));
            int dst0_11 = s_kn_addr + (unsigned int)next_off + (unsigned int)((c0_7 + 8) / 64 * 8192 + (r0_6 * 128 + (c0_7 + 8) % 64 * 2 ^ (r0_6 * 128 + (c0_7 + 8) % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_11), "l"(kn + (src_row_9 + 8)), "r"((next_len > r0_6) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_11 + 1024), "l"(kn + (src_row_9 + 8 * row_stride_8 + 8)), "r"((next_len > r0_6 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_11 + 2048), "l"(kn + (src_row_9 + 16 * row_stride_8 + 8)), "r"((next_len > r0_6 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_11 + 3072), "l"(kn + (src_row_9 + 24 * row_stride_8 + 8)), "r"((next_len > r0_6 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_11 + 4096), "l"(kn + (src_row_9 + 32 * row_stride_8 + 8)), "r"((next_len > r0_6 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_11 + 5120), "l"(kn + (src_row_9 + 40 * row_stride_8 + 8)), "r"((next_len > r0_6 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_11 + 6144), "l"(kn + (src_row_9 + 48 * row_stride_8 + 8)), "r"((next_len > r0_6 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_11 + 7168), "l"(kn + (src_row_9 + 56 * row_stride_8 + 8)), "r"((next_len > r0_6 + 56) ? 16 : 0));
            int r0_12 = tid / 4;
            int c0_13 = tid % 4 * 8;
            long long row_stride_14 = (long long)num_v_heads * 128;
            long long src_row_15 = (long long)(next_tok0 + r0_12) * row_stride_14 + (long long)hv * 128 + (long long)v0 + (long long)c0_13;
            int dst0_16 = s_u_addr + (unsigned int)next_off + (unsigned int)(c0_13 / 64 * 8192 + (r0_12 * 128 + c0_13 % 64 * 2 ^ (r0_12 * 128 + c0_13 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_16), "l"(u + src_row_15), "r"((next_len > r0_12) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_16 + 2048), "l"(u + (src_row_15 + 16 * row_stride_14)), "r"((next_len > r0_12 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_16 + 4096), "l"(u + (src_row_15 + 32 * row_stride_14)), "r"((next_len > r0_12 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_16 + 6144), "l"(u + (src_row_15 + 48 * row_stride_14)), "r"((next_len > r0_12 + 48) ? 16 : 0));
            asm volatile("cp.async.commit_group;");
        }
        float g_last = s_g[g_off + n_valid - 1];
        float _exp2_0 = approx_exp2(g_last);
        #if __CUDA_ARCH__ >= 1000
        const float2 _scale2_0 = {_exp2_0, _exp2_0};
        #pragma unroll
        for (int _ls = 0; _ls < 32; _ls++)
            mul_f32x2_inplace(&reinterpret_cast<float2*>(acc_h)[_ls], _scale2_0);
        #else
        #pragma unroll
        for (int _ls = 0; _ls < 64; _ls++) {
            acc_h[_ls] = acc_h[_ls] * _exp2_0;
        }
        #endif
        #pragma unroll
        for (int ks = 0; ks < ((0) ? 0 : 8); ks++) {
            #pragma unroll
            for (int nb3 = 0; nb3 < 4; nb3++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                    : "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)((kbase + ks * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((nb3 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kbase + ks * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((nb3 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kbase + ks * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_v + nb3 * 8)[0]), "=f"((acc_v + nb3 * 8)[1]), "=f"((acc_v + nb3 * 8)[2]), "=f"((acc_v + nb3 * 8)[3])
                    : "r"((hpk + ks * 4)[0]), "r"((hpk + ks * 4)[1]), "r"((hpk + ks * 4)[2]), "r"((hpk + ks * 4)[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[0])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[1])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[2])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_v + nb3 * 8 + 4)[0]), "=f"((acc_v + nb3 * 8 + 4)[1]), "=f"((acc_v + nb3 * 8 + 4)[2]), "=f"((acc_v + nb3 * 8 + 4)[3])
                    : "r"((hpk + ks * 4)[0]), "r"((hpk + ks * 4)[1]), "r"((hpk + ks * 4)[2]), "r"((hpk + ks * 4)[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[0])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[1])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[2])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[3])));
            }
        }
        long long tape_base = ((long long)chunk * (long long)num_v_heads + (long long)hv) * 16384 + (long long)(v0 + vwarp * 16) * 128 + (long long)kbase;
        int m_idx = lane_0 / 8;
        int r_idx = lane_0 % 8;
        int st_row = m_idx % 2 * 8 + r_idx;
        int st_nb = m_idx / 2;
        uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&hpk[0])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[1])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[2])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[3]))
            : "memory");
        int st_nb_1 = 2 + m_idx / 2;
        uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_1 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&hpk[4])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[5])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[6])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[7]))
            : "memory");
        int st_nb_2 = 4 + m_idx / 2;
        uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_2 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&hpk[8])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[9])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[10])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[11]))
            : "memory");
        int st_nb_3 = 6 + m_idx / 2;
        uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_3 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&hpk[12])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[13])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[14])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[15]))
            : "memory");
        int st_nb_4 = 8 + m_idx / 2;
        uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_4 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&hpk[16])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[17])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[18])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[19]))
            : "memory");
        int st_nb_5 = 10 + m_idx / 2;
        uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_5 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&hpk[20])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[21])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[22])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[23]))
            : "memory");
        int st_nb_6 = 12 + m_idx / 2;
        uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_6 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&hpk[24])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[25])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[26])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[27]))
            : "memory");
        int st_nb_7 = 14 + m_idx / 2;
        uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(stage_base + st_row * 256 + ((st_nb_7 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&hpk[28])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[29])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[30])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[31]))
            : "memory");
        __syncwarp();
        int ld_row_lo = lane_0 / 16;
        int ld_ch = lane_0 % 16;
        int ld_row = ld_row_lo;
        int words[4];
        {
            const int4* _ivptr_9 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row * 64 + ((ld_ch ^ ld_row % 8) << 2));
            int4 _ivld_9;
            _ivld_9 = *_ivptr_9;
            words[0 + 0] = _ivld_9.x;
            words[0 + 1] = _ivld_9.y;
            words[0 + 2] = _ivld_9.z;
            words[0 + 3] = _ivld_9.w;
        }
        {
            int4 _iv4 = make_int4(words[0 + 0], words[0 + 1], words[0 + 2], words[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_8 = 2 + ld_row_lo;
        int words_9[4];
        {
            const int4* _ivptr_10 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_8 * 64 + ((ld_ch ^ ld_row_8 % 8) << 2));
            int4 _ivld_10;
            _ivld_10 = *_ivptr_10;
            words_9[0 + 0] = _ivld_10.x;
            words_9[0 + 1] = _ivld_10.y;
            words_9[0 + 2] = _ivld_10.z;
            words_9[0 + 3] = _ivld_10.w;
        }
        {
            int4 _iv4 = make_int4(words_9[0 + 0], words_9[0 + 1], words_9[0 + 2], words_9[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_8 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_10 = 4 + ld_row_lo;
        int words_11[4];
        {
            const int4* _ivptr_11 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_10 * 64 + ((ld_ch ^ ld_row_10 % 8) << 2));
            int4 _ivld_11;
            _ivld_11 = *_ivptr_11;
            words_11[0 + 0] = _ivld_11.x;
            words_11[0 + 1] = _ivld_11.y;
            words_11[0 + 2] = _ivld_11.z;
            words_11[0 + 3] = _ivld_11.w;
        }
        {
            int4 _iv4 = make_int4(words_11[0 + 0], words_11[0 + 1], words_11[0 + 2], words_11[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_10 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_12 = 6 + ld_row_lo;
        int words_13[4];
        {
            const int4* _ivptr_12 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_12 * 64 + ((ld_ch ^ ld_row_12 % 8) << 2));
            int4 _ivld_12;
            _ivld_12 = *_ivptr_12;
            words_13[0 + 0] = _ivld_12.x;
            words_13[0 + 1] = _ivld_12.y;
            words_13[0 + 2] = _ivld_12.z;
            words_13[0 + 3] = _ivld_12.w;
        }
        {
            int4 _iv4 = make_int4(words_13[0 + 0], words_13[0 + 1], words_13[0 + 2], words_13[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_12 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_14 = 8 + ld_row_lo;
        int words_15[4];
        {
            const int4* _ivptr_13 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_14 * 64 + ((ld_ch ^ ld_row_14 % 8) << 2));
            int4 _ivld_13;
            _ivld_13 = *_ivptr_13;
            words_15[0 + 0] = _ivld_13.x;
            words_15[0 + 1] = _ivld_13.y;
            words_15[0 + 2] = _ivld_13.z;
            words_15[0 + 3] = _ivld_13.w;
        }
        {
            int4 _iv4 = make_int4(words_15[0 + 0], words_15[0 + 1], words_15[0 + 2], words_15[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_14 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_16 = 10 + ld_row_lo;
        int words_17[4];
        {
            const int4* _ivptr_14 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_16 * 64 + ((ld_ch ^ ld_row_16 % 8) << 2));
            int4 _ivld_14;
            _ivld_14 = *_ivptr_14;
            words_17[0 + 0] = _ivld_14.x;
            words_17[0 + 1] = _ivld_14.y;
            words_17[0 + 2] = _ivld_14.z;
            words_17[0 + 3] = _ivld_14.w;
        }
        {
            int4 _iv4 = make_int4(words_17[0 + 0], words_17[0 + 1], words_17[0 + 2], words_17[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_16 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_18 = 12 + ld_row_lo;
        int words_19[4];
        {
            const int4* _ivptr_15 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_18 * 64 + ((ld_ch ^ ld_row_18 % 8) << 2));
            int4 _ivld_15;
            _ivld_15 = *_ivptr_15;
            words_19[0 + 0] = _ivld_15.x;
            words_19[0 + 1] = _ivld_15.y;
            words_19[0 + 2] = _ivld_15.z;
            words_19[0 + 3] = _ivld_15.w;
        }
        {
            int4 _iv4 = make_int4(words_19[0 + 0], words_19[0 + 1], words_19[0 + 2], words_19[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_18 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_20 = 14 + ld_row_lo;
        int words_21[4];
        {
            const int4* _ivptr_16 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_20 * 64 + ((ld_ch ^ ld_row_20 % 8) << 2));
            int4 _ivld_16;
            _ivld_16 = *_ivptr_16;
            words_21[0 + 0] = _ivld_16.x;
            words_21[0 + 1] = _ivld_16.y;
            words_21[0 + 2] = _ivld_16.z;
            words_21[0 + 3] = _ivld_16.w;
        }
        {
            int4 _iv4 = make_int4(words_21[0 + 0], words_21[0 + 1], words_21[0 + 2], words_21[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_20 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        __syncwarp();
        float vn[4];
        #pragma unroll
        for (int tb = 0; tb < ((0) ? 0 : 4); tb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(u_frag[0]), "=r"(u_frag[1]), "=r"(u_frag[2]), "=r"(u_frag[3])
                : "r"(s_u_addr + (unsigned int)stage_off + (unsigned int)((vwarp * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((tb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((tb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            #pragma unroll
            for (int half_1 = 0; half_1 < 2; half_1++) {
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int tok_1 = (tb * 2 + half_1) * 8 + quad * 2 + e;
                    float decay = 0.0f;
                    if (tok_1 < n_valid) {
                        float _exp2_1 = approx_exp2(g_last - s_g[g_off + tok_1]);
                        decay = _exp2_1;
                    }
                    float u_a = ((e == 0) ? __uint_as_float(u_frag[half_1 * 2] << 16) : __uint_as_float(u_frag[half_1 * 2] & 4294901760u));
                    float u_b = ((e == 0) ? __uint_as_float(u_frag[half_1 * 2 + 1] << 16) : __uint_as_float(u_frag[half_1 * 2 + 1] & 4294901760u));
                    vn[e] = u_a - acc_v[(tb * 2 + half_1) * 4 + e];
                    vn[2 + e] = u_b - acc_v[(tb * 2 + half_1) * 4 + 2 + e];
                    acc_v[(tb * 2 + half_1) * 4 + e] = vn[e] * decay;
                    acc_v[(tb * 2 + half_1) * 4 + 2 + e] = vn[2 + e] * decay;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vn[_lp*2 + 0], vn[_lp*2+1 + 0]));
                    vw[((tb * 2 + half_1) * 2) + _lp] = *(uint32_t*)&_bf2;
                }
            }
        }
        #pragma unroll
        for (int _lp = 0; _lp < 16; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_v[_lp*2 + 0], acc_v[_lp*2+1 + 0]));
            vpk[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int ks2 = 0; ks2 < ((0) ? 0 : 4); ks2++) {
            #pragma unroll
            for (int nb5_1 = 0; nb5_1 < 8; nb5_1++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                    : "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)((kbase + nb5_1 * 16 + lane / 16 * 8) / 64 * 8192 + ((ks2 * 16 + lane % 16) * 128 + (kbase + nb5_1 * 16 + lane / 16 * 8) % 64 * 2 ^ ((ks2 * 16 + lane % 16) * 128 + (kbase + nb5_1 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_h + nb5_1 * 8)[0]), "+f"((acc_h + nb5_1 * 8)[1]), "+f"((acc_h + nb5_1 * 8)[2]), "+f"((acc_h + nb5_1 * 8)[3])
                    : "r"((vpk + ks2 * 4)[0]), "r"((vpk + ks2 * 4)[1]), "r"((vpk + ks2 * 4)[2]), "r"((vpk + ks2 * 4)[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_h + nb5_1 * 8 + 4)[0]), "+f"((acc_h + nb5_1 * 8 + 4)[1]), "+f"((acc_h + nb5_1 * 8 + 4)[2]), "+f"((acc_h + nb5_1 * 8 + 4)[3])
                    : "r"((vpk + ks2 * 4)[0]), "r"((vpk + ks2 * 4)[1]), "r"((vpk + ks2 * 4)[2]), "r"((vpk + ks2 * 4)[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
            }
        }
        if (khalf == 0) {
            long long row_base = ((long long)tok0 * (long long)num_v_heads + (long long)hv) * 128 + (long long)v0 + (long long)(vwarp * 16);
            int m_idx_0 = lane_0 / 8;
            int r_idx_1 = lane_0 % 8;
            int st_half = m_idx_0 % 2;
            int st_tok = m_idx_0 / 2 * 8 + r_idx_1;
            uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(stage_base + st_tok * 32 + ((st_half ^ st_tok >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&vw[0])), "r"(*reinterpret_cast<const uint32_t*>(&vw[1])), "r"(*reinterpret_cast<const uint32_t*>(&vw[2])), "r"(*reinterpret_cast<const uint32_t*>(&vw[3]))
                : "memory");
            int st_tok_2 = (2 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(stage_base + st_tok_2 * 32 + ((st_half ^ st_tok_2 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&vw[4])), "r"(*reinterpret_cast<const uint32_t*>(&vw[5])), "r"(*reinterpret_cast<const uint32_t*>(&vw[6])), "r"(*reinterpret_cast<const uint32_t*>(&vw[7]))
                : "memory");
            int st_tok_3 = (4 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(stage_base + st_tok_3 * 32 + ((st_half ^ st_tok_3 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&vw[8])), "r"(*reinterpret_cast<const uint32_t*>(&vw[9])), "r"(*reinterpret_cast<const uint32_t*>(&vw[10])), "r"(*reinterpret_cast<const uint32_t*>(&vw[11]))
                : "memory");
            int st_tok_4 = (6 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_20 = static_cast<uint32_t>(stage_base + st_tok_4 * 32 + ((st_half ^ st_tok_4 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_20), "r"(*reinterpret_cast<const uint32_t*>(&vw[12])), "r"(*reinterpret_cast<const uint32_t*>(&vw[13])), "r"(*reinterpret_cast<const uint32_t*>(&vw[14])), "r"(*reinterpret_cast<const uint32_t*>(&vw[15]))
                : "memory");
            __syncwarp();
            int ld_half = lane_0 % 2;
            int ld_tok = lane_0 / 2;
            int words_5[4];
            {
                const int4* _ivptr_21 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok * 8 + ((ld_half ^ ld_tok >> 2 & 1) << 2));
                int4 _ivld_21;
                _ivld_21 = *_ivptr_21;
                words_5[0 + 0] = _ivld_21.x;
                words_5[0 + 1] = _ivld_21.y;
                words_5[0 + 2] = _ivld_21.z;
                words_5[0 + 3] = _ivld_21.w;
            }
            if (ld_tok < n_valid) {
                {
                    int4 _iv4 = make_int4(words_5[0 + 0], words_5[0 + 1], words_5[0 + 2], words_5[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_6 = 16 + lane_0 / 2;
            int words_7[4];
            {
                const int4* _ivptr_22 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_6 * 8 + ((ld_half ^ ld_tok_6 >> 2 & 1) << 2));
                int4 _ivld_22;
                _ivld_22 = *_ivptr_22;
                words_7[0 + 0] = _ivld_22.x;
                words_7[0 + 1] = _ivld_22.y;
                words_7[0 + 2] = _ivld_22.z;
                words_7[0 + 3] = _ivld_22.w;
            }
            if (ld_tok_6 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_7[0 + 0], words_7[0 + 1], words_7[0 + 2], words_7[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_6 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_8 = 32 + lane_0 / 2;
            int words_10[4];
            {
                const int4* _ivptr_23 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_8 * 8 + ((ld_half ^ ld_tok_8 >> 2 & 1) << 2));
                int4 _ivld_23;
                _ivld_23 = *_ivptr_23;
                words_10[0 + 0] = _ivld_23.x;
                words_10[0 + 1] = _ivld_23.y;
                words_10[0 + 2] = _ivld_23.z;
                words_10[0 + 3] = _ivld_23.w;
            }
            if (ld_tok_8 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_10[0 + 0], words_10[0 + 1], words_10[0 + 2], words_10[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_8 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_11 = 48 + lane_0 / 2;
            int words_12[4];
            {
                const int4* _ivptr_24 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_11 * 8 + ((ld_half ^ ld_tok_11 >> 2 & 1) << 2));
                int4 _ivld_24;
                _ivld_24 = *_ivptr_24;
                words_12[0 + 0] = _ivld_24.x;
                words_12[0 + 1] = _ivld_24.y;
                words_12[0 + 2] = _ivld_24.z;
                words_12[0 + 3] = _ivld_24.w;
            }
            if (ld_tok_11 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_12[0 + 0], words_12[0 + 1], words_12[0 + 2], words_12[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_11 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            __syncwarp();
        }
        if (num_chunks > it_chunk + 1) {
            #pragma unroll
            for (int gi2 = 0; gi2 < 1; gi2++) {
                int gtok2 = gi2 * 64 + tid_2;
                if (gtok2 < 64) {
                    s_g[next_g_off + gtok2] = g_next[gi2];
                }
            }
        }
        #pragma unroll
        for (int gi3 = 0; gi3 < 1; gi3++) {
            g_next[gi3] = g_far[gi3];
        }
        stage_off = next_off;
        g_off = next_g_off;
    }
    if (store_final_state != 0) {
        if (is_last != 0) {
            #pragma unroll
            for (int nb6 = 0; nb6 < 16; nb6++) {
                int kcol2 = kbase + nb6 * 8 + quad * 2;
                final_state[out_base + (long long)kcol2 * 128 + (long long)vrow_a] = acc_h[nb6 * 4];
                final_state[out_base + (long long)(kcol2 + 1) * 128 + (long long)vrow_a] = acc_h[nb6 * 4 + 1];
                final_state[out_base + (long long)kcol2 * 128 + (long long)vrow_b] = acc_h[nb6 * 4 + 2];
                final_state[out_base + (long long)(kcol2 + 1) * 128 + (long long)vrow_b] = acc_h[nb6 * 4 + 3];
            }
        }
    }
}

} // extern "C"
