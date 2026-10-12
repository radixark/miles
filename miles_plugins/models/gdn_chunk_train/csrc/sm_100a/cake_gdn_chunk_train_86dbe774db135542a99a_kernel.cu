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
#define SMEM_S_KN_OFF 0
#define SMEM_S_KN_STAGE_BYTES 18432
#define SMEM_S_KN_STRIDE 18432
#define SMEM_S_V_OFF 18432
#define SMEM_S_V_STAGE_BYTES 18432
#define SMEM_S_V_STRIDE 18432
#define SMEM_S_DW_OFF 36864
#define SMEM_S_DW_STAGE_BYTES 18432
#define SMEM_S_DW_STRIDE 18432
#define SMEM_S_DU_OFF 55296
#define SMEM_S_DU_STAGE_BYTES 18432
#define SMEM_S_DU_STRIDE 18432
#define SMEM_S_A_OFF 73728
#define SMEM_S_A_STAGE_BYTES 9216
#define SMEM_S_A_STRIDE 9216
#define SMEM_S_DA_OFF 55296
#define SMEM_S_DA_STAGE_BYTES 9216
#define SMEM_S_DA_STRIDE 9216
#define SMEM_S_DA2_OFF 64512
#define SMEM_S_DA2_STAGE_BYTES 9216
#define SMEM_S_DA2_STRIDE 9216
#define SMEM_S_DA3_OFF 18432
#define SMEM_S_DA3_STAGE_BYTES 9216
#define SMEM_S_DA3_STRIDE 9216
#define SMEM_S_DA3B_OFF 27648
#define SMEM_S_DA3B_STAGE_BYTES 9216
#define SMEM_S_DA3B_STRIDE 9216
#define SMEM_S_G_OFF 82944
#define SMEM_S_G_STAGE_BYTES 256
#define SMEM_S_G_STRIDE 256
#define SMEM_S_B_OFF 83200
#define SMEM_S_B_STAGE_BYTES 256
#define SMEM_S_B_STRIDE 256
#define SMEM_S_GEXP_OFF 83456
#define SMEM_S_GEXP_STAGE_BYTES 256
#define SMEM_S_GEXP_STRIDE 256
#define SMEM_S_COL_OFF 83712
#define SMEM_S_COL_STAGE_BYTES 1024
#define SMEM_S_COL_STRIDE 1024
#define SMEM_S_DBP_OFF 55296
#define SMEM_S_DBP_STAGE_BYTES 512
#define SMEM_S_DBP_STRIDE 512
#define SMEM_S_DGP_OFF 55808
#define SMEM_S_DGP_STAGE_BYTES 512
#define SMEM_S_DGP_STRIDE 512
#define SMEM_TOTAL 84736
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_chunk_train_86dbe774db135542a99a(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ dw, __nv_bfloat16* __restrict__ du, float* __restrict__ g_cs, float* __restrict__ beta, __nv_bfloat16* __restrict__ dk2_out, __nv_bfloat16* __restrict__ dv_out, float* __restrict__ dbeta_out, float* __restrict__ dg2_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads)
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
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_KN_OFF);
    const int s_kn_addr = smem + SMEM_S_KN_OFF;
    __nv_bfloat16* s_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_V_OFF);
    const int s_v_addr = smem + SMEM_S_V_OFF;
    __nv_bfloat16* s_dw = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_DW_OFF);
    const int s_dw_addr = smem + SMEM_S_DW_OFF;
    __nv_bfloat16* s_du = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_DU_OFF);
    const int s_du_addr = smem + SMEM_S_DU_OFF;
    __nv_bfloat16* s_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_A_OFF);
    const int s_a_addr = smem + SMEM_S_A_OFF;
    __nv_bfloat16* s_da = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_DA_OFF);
    const int s_da_addr = smem + SMEM_S_DA_OFF;
    __nv_bfloat16* s_da2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_DA2_OFF);
    const int s_da2_addr = smem + SMEM_S_DA2_OFF;
    __nv_bfloat16* s_da3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_DA3_OFF);
    const int s_da3_addr = smem + SMEM_S_DA3_OFF;
    __nv_bfloat16* s_da3b = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_DA3B_OFF);
    const int s_da3b_addr = smem + SMEM_S_DA3B_OFF;
    float* s_g = reinterpret_cast<float*>(smem_raw + SMEM_S_G_OFF);
    const int s_g_addr = smem + SMEM_S_G_OFF;
    float* s_b = reinterpret_cast<float*>(smem_raw + SMEM_S_B_OFF);
    const int s_b_addr = smem + SMEM_S_B_OFF;
    float* s_gexp = reinterpret_cast<float*>(smem_raw + SMEM_S_GEXP_OFF);
    const int s_gexp_addr = smem + SMEM_S_GEXP_OFF;
    float* s_col = reinterpret_cast<float*>(smem_raw + SMEM_S_COL_OFF);
    const int s_col_addr = smem + SMEM_S_COL_OFF;
    float* s_dbp = reinterpret_cast<float*>(smem_raw + SMEM_S_DBP_OFF);
    const int s_dbp_addr = smem + SMEM_S_DBP_OFF;
    float* s_dgp = reinterpret_cast<float*>(smem_raw + SMEM_S_DGP_OFF);
    const int s_dgp_addr = smem + SMEM_S_DGP_OFF;

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
    int half = nh * 9216;
    int kn_h = s_kn_addr + (unsigned int)half;
    int v_h = s_v_addr + (unsigned int)half;
    int dw_h = s_dw_addr + (unsigned int)half;
    int du_h = s_du_addr + (unsigned int)half;
    int st_off = (rb * 16 + lane_0 / 8 % 2 * 8 + lane_0 % 8) * 144 + (s0 + lane_0 / 16 * 8) * 2;
    #pragma unroll
    for (int it = 0; it < 2; it++) {
        int aidx = it * 256 + tid_2;
        int a_row = aidx / 8;
        int a_col = aidx % 8 * 8;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_a_addr + (unsigned int)(a_row * 144 + a_col * 2)), "l"(A + (((long long)(tok0 + a_row) * (long long)num_v_heads + (long long)hv) * 64 + (long long)a_col)), "r"((a_row < n_valid) ? 16 : 0));
    }
    int r0 = tid / 16;
    int c0 = tid % 16 * 8;
    long long row_stride = (long long)num_v_heads * 128;
    long long src_row = (long long)(tok0 + r0) * row_stride + (long long)hv * 128 + (long long)c0;
    int dst0 = s_du_addr + (unsigned int)(c0 / 64 * 9216 + r0 * 144 + c0 % 64 * 2);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0), "l"(du + src_row), "r"((n_valid > r0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 2304), "l"(du + (src_row + 16 * row_stride)), "r"((n_valid > r0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 4608), "l"(du + (src_row + 32 * row_stride)), "r"((n_valid > r0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 6912), "l"(du + (src_row + 48 * row_stride)), "r"((n_valid > r0 + 48) ? 16 : 0));
    int r0_3 = tid / 16;
    int c0_4 = tid % 16 * 8;
    long long row_stride_5 = (long long)num_v_heads * 128;
    long long src_row_6 = (long long)(tok0 + r0_3) * row_stride_5 + (long long)hv * 128 + (long long)c0_4;
    int dst0_7 = s_v_addr + (unsigned int)(c0_4 / 64 * 9216 + r0_3 * 144 + c0_4 % 64 * 2);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7), "l"(v + src_row_6), "r"((n_valid > r0_3) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 2304), "l"(v + (src_row_6 + 16 * row_stride_5)), "r"((n_valid > r0_3 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 4608), "l"(v + (src_row_6 + 32 * row_stride_5)), "r"((n_valid > r0_3 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 6912), "l"(v + (src_row_6 + 48 * row_stride_5)), "r"((n_valid > r0_3 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    int r0_8 = tid / 16;
    int c0_9 = tid % 16 * 8;
    long long row_stride_10 = (long long)num_heads * 128;
    long long src_row_11 = (long long)(tok0 + r0_8) * row_stride_10 + (long long)hq * 128 + (long long)c0_9;
    int dst0_12 = s_kn_addr + (unsigned int)(c0_9 / 64 * 9216 + r0_8 * 144 + c0_9 % 64 * 2);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12), "l"(kn + src_row_11), "r"((n_valid > r0_8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 2304), "l"(kn + (src_row_11 + 16 * row_stride_10)), "r"((n_valid > r0_8 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 4608), "l"(kn + (src_row_11 + 32 * row_stride_10)), "r"((n_valid > r0_8 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 6912), "l"(kn + (src_row_11 + 48 * row_stride_10)), "r"((n_valid > r0_8 + 48) ? 16 : 0));
    int r0_13 = tid / 16;
    int c0_14 = tid % 16 * 8;
    long long row_stride_15 = (long long)num_v_heads * 128;
    long long src_row_16 = (long long)(tok0 + r0_13) * row_stride_15 + (long long)hv * 128 + (long long)c0_14;
    int dst0_17 = s_dw_addr + (unsigned int)(c0_14 / 64 * 9216 + r0_13 * 144 + c0_14 % 64 * 2);
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17), "l"(dw + src_row_16), "r"((n_valid > r0_13) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17 + 2304), "l"(dw + (src_row_16 + 16 * row_stride_15)), "r"((n_valid > r0_13 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17 + 4608), "l"(dw + (src_row_16 + 32 * row_stride_15)), "r"((n_valid > r0_13 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_17 + 6912), "l"(dw + (src_row_16 + 48 * row_stride_15)), "r"((n_valid > r0_13 + 48) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    if (tid_2 < 64) {
        float gate_val = 0.0f;
        float beta_val = 0.0f;
        if (tid_2 < n_valid) {
            long long gate_index = (long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv;
            gate_val = g_cs[gate_index];
            beta_val = beta[gate_index];
        }
        s_g[tid_2] = gate_val;
        s_b[tid_2] = beta_val;
        float _exp2_0 = approx_exp2(gate_val);
        s_gexp[tid_2] = _exp2_0;
    }
    asm volatile("cp.async.wait_group 1;");
    __syncthreads();
    unsigned int a_frag_k[4];
    unsigned int a_frag_mn[4];
    unsigned int b_frag_k[4];
    unsigned int b_frag_mn[4];
    float acc_s[32];
    float acc[32];
    unsigned int pair_a[1];
    unsigned int pair_b[1];
    float g_a = s_g[row_a];
    float g_b = s_g[row_b];
    float beta_a = s_b[row_a];
    float beta_b = s_b[row_b];
    float _exp2_1 = approx_exp2(g_a);
    float gexp_a = _exp2_1;
    float _exp2_2 = approx_exp2(g_b);
    float gexp_b = _exp2_2;
    float bg_a = gexp_a * beta_a;
    float bg_b = gexp_b * beta_b;
    float db_a = 0.0f;
    float db_b = 0.0f;
    float dg_a = 0.0f;
    float dg_b = 0.0f;
    long long out_a = ((long long)(tok0 + row_a) * (long long)num_v_heads + (long long)hv) * 128 + (long long)n0 + (long long)col_q;
    long long out_b = ((long long)(tok0 + row_b) * (long long)num_v_heads + (long long)hv) * 128 + (long long)n0 + (long long)col_q;
    #pragma unroll
    for (int kb4 = 0; kb4 < 4; kb4++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_mn[0]), "=r"(a_frag_mn[1]), "=r"(a_frag_mn[2]), "=r"(a_frag_mn[3])
            : "r"(s_a_addr + (unsigned int)((kb4 * 16 + 8 * (lane / 16) + lane % 8) * 144 + (rb * 16 + lane % 16 / 8 * 8) * 2))
            : "memory");
        #pragma unroll
        for (int nb5 = 0; nb5 < 4; nb5++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(du_h + ((kb4 * 16 + lane % 16) * 144 + (nb5 * 16 + lane / 16 * 8) * 2))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb5 * 8)[0]), "=f"((acc + nb5 * 8)[1]), "=f"((acc + nb5 * 8)[2]), "=f"((acc + nb5 * 8)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb4 == 0) ? 0.0f : (acc + nb5 * 8)[0])), "f"(((kb4 == 0) ? 0.0f : (acc + nb5 * 8)[1])), "f"(((kb4 == 0) ? 0.0f : (acc + nb5 * 8)[2])), "f"(((kb4 == 0) ? 0.0f : (acc + nb5 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb5 * 8 + 4)[0]), "=f"((acc + nb5 * 8 + 4)[1]), "=f"((acc + nb5 * 8 + 4)[2]), "=f"((acc + nb5 * 8 + 4)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb4 == 0) ? 0.0f : (acc + nb5 * 8 + 4)[0])), "f"(((kb4 == 0) ? 0.0f : (acc + nb5 * 8 + 4)[1])), "f"(((kb4 == 0) ? 0.0f : (acc + nb5 * 8 + 4)[2])), "f"(((kb4 == 0) ? 0.0f : (acc + nb5 * 8 + 4)[3])));
        }
    }
    #pragma unroll
    for (int nb6 = 0; nb6 < 8; nb6++) {
        int vcol = nb6 * 8 + col_q;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(v_h + (row_a * 144 + vcol * 2)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(v_h + (row_b * 144 + vcol * 2)));
        float _fma_0 = __fmaf_rn(acc[nb6 * 4], __uint_as_float(pair_a[0] << 16), db_a);
        db_a = _fma_0;
        float _fma_1 = __fmaf_rn(acc[nb6 * 4 + 2], __uint_as_float(pair_b[0] << 16), db_b);
        db_b = _fma_1;
        float _fma_2 = __fmaf_rn(acc[nb6 * 4 + 1], __uint_as_float(pair_a[0] & 4294901760u), db_a);
        db_a = _fma_2;
        float _fma_3 = __fmaf_rn(acc[nb6 * 4 + 3], __uint_as_float(pair_b[0] & 4294901760u), db_b);
        db_b = _fma_3;
        float2 _f2_0 = make_float2(acc[nb6 * 4], acc[nb6 * 4 + 1]);
        float2 _f2_1 = make_float2(beta_a, beta_a);
        float2 dv_a = mul_f32x2_noftz(_f2_0, _f2_1);
        float2 _f2_2 = make_float2(acc[nb6 * 4 + 2], acc[nb6 * 4 + 3]);
        float2 _f2_3 = make_float2(beta_b, beta_b);
        float2 dv_b = mul_f32x2_noftz(_f2_2, _f2_3);
        acc[nb6 * 4] = dv_a.x;
        acc[nb6 * 4 + 1] = dv_a.y;
        acc[nb6 * 4 + 2] = dv_b.x;
        acc[nb6 * 4 + 3] = dv_b.y;
        if (row_a < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb6 * 4 + 0], acc[nb6 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dv_out))[out_a + (long long)(nb6 * 8)]) = _pk;
            }
        }
        if (row_b < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb6 * 4 + 2 + 0], acc[nb6 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dv_out))[out_b + (long long)(nb6 * 8)]) = _pk;
            }
        }
    }
    #pragma unroll
    for (int kb2 = 0; kb2 < 8; kb2++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_du_addr + (unsigned int)(kb2 / 4 * 9216) + (unsigned int)((rb * 16 + lane % 16) * 144 + (kb2 % 4 * 16 + lane / 16 * 8) * 2))
            : "memory");
        #pragma unroll
        for (int nb2 = 0; nb2 < 2; nb2++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                : "r"(s_v_addr + (unsigned int)(kb2 / 4 * 9216) + (unsigned int)((s0 + nb2 * 16 + 8 * (lane / 16) + lane % 8) * 144 + (kb2 % 4 * 16 + lane % 16 / 8 * 8) * 2))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb2 * 8)[0]), "=f"((acc + nb2 * 8)[1]), "=f"((acc + nb2 * 8)[2]), "=f"((acc + nb2 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]), "f"(((kb2 == 0) ? 0.0f : (acc + nb2 * 8)[0])), "f"(((kb2 == 0) ? 0.0f : (acc + nb2 * 8)[1])), "f"(((kb2 == 0) ? 0.0f : (acc + nb2 * 8)[2])), "f"(((kb2 == 0) ? 0.0f : (acc + nb2 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb2 * 8 + 4)[0]), "=f"((acc + nb2 * 8 + 4)[1]), "=f"((acc + nb2 * 8 + 4)[2]), "=f"((acc + nb2 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]), "f"(((kb2 == 0) ? 0.0f : (acc + nb2 * 8 + 4)[0])), "f"(((kb2 == 0) ? 0.0f : (acc + nb2 * 8 + 4)[1])), "f"(((kb2 == 0) ? 0.0f : (acc + nb2 * 8 + 4)[2])), "f"(((kb2 == 0) ? 0.0f : (acc + nb2 * 8 + 4)[3])));
        }
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    #pragma unroll
    for (int kb = 0; kb < 8; kb++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_dw_addr + (unsigned int)(kb / 4 * 9216) + (unsigned int)((rb * 16 + lane % 16) * 144 + (kb % 4 * 16 + lane / 16 * 8) * 2))
            : "memory");
        #pragma unroll
        for (int nb = 0; nb < 2; nb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                : "r"(s_kn_addr + (unsigned int)(kb / 4 * 9216) + (unsigned int)((s0 + nb * 16 + 8 * (lane / 16) + lane % 8) * 144 + (kb % 4 * 16 + lane % 16 / 8 * 8) * 2))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb * 8)[0]), "=f"((acc_s + nb * 8)[1]), "=f"((acc_s + nb * 8)[2]), "=f"((acc_s + nb * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8)[0])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8)[1])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8)[2])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb * 8 + 4)[0]), "=f"((acc_s + nb * 8 + 4)[1]), "=f"((acc_s + nb * 8 + 4)[2]), "=f"((acc_s + nb * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8 + 4)[0])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8 + 4)[1])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8 + 4)[2])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8 + 4)[3])));
        }
    }
    float vals[8];
    unsigned int pk[4];
    float gq[2];
    float bq[2];
    #pragma unroll
    for (int p = 0; p < 2; p++) {
        #pragma unroll
        for (int q = 0; q < 2; q++) {
            int col0 = s0 + p * 16 + q * 8 + col_q;
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gq[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq[(0) + 1]))
                : "r"(s_gexp_addr + (unsigned int)(col0 * 4)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&bq[0])), "=r"(*reinterpret_cast<uint32_t*>(&bq[(0) + 1]))
                : "r"(s_b_addr + (unsigned int)(col0 * 4)));
            float2 _f2_4 = make_float2(acc_s[(2 * p + q) * 4], acc_s[(2 * p + q) * 4 + 1]);
            float2 _f2_5 = make_float2(gq[0], gq[1]);
            float2 _f2_6 = make_float2(acc[(2 * p + q) * 4], acc[(2 * p + q) * 4 + 1]);
            float2 _f2_7 = make_float2(bq[0], bq[1]);
            float2 da_a = mul_f32x2_noftz(fma_f32x2_rn_noftz(_f2_4, _f2_5, _f2_6), _f2_7);
            float2 _f2_8 = make_float2(acc_s[(2 * p + q) * 4 + 2], acc_s[(2 * p + q) * 4 + 3]);
            float2 _f2_9 = make_float2(gq[0], gq[1]);
            float2 _f2_10 = make_float2(acc[(2 * p + q) * 4 + 2], acc[(2 * p + q) * 4 + 3]);
            float2 _f2_11 = make_float2(bq[0], bq[1]);
            float2 da_b = mul_f32x2_noftz(fma_f32x2_rn_noftz(_f2_8, _f2_9, _f2_10), _f2_11);
            float val_a0 = 0.0f;
            float val_a1 = 0.0f;
            float val_b0 = 0.0f;
            float val_b1 = 0.0f;
            if (col0 < row_a) {
                if (row_a < n_valid) {
                    val_a0 = da_a.x;
                }
            }
            if (row_a > col0 + 1) {
                if (row_a < n_valid) {
                    val_a1 = da_a.y;
                }
            }
            if (col0 < row_b) {
                if (row_b < n_valid) {
                    val_b0 = da_b.x;
                }
            }
            if (row_b > col0 + 1) {
                if (row_b < n_valid) {
                    val_b1 = da_b.y;
                }
            }
            vals[q * 4] = val_a0;
            vals[q * 4 + 1] = val_a1;
            vals[q * 4 + 2] = val_b0;
            vals[q * 4 + 3] = val_b1;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
            pk[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(s_da_addr + (unsigned int)st_off + (unsigned int)(p * 32));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3]))
            : "memory");
    }
    __syncthreads();
    #pragma unroll
    for (int kb5 = 0; kb5 < 4; kb5++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_da_addr + (unsigned int)((rb * 16 + lane % 16) * 144 + (kb5 * 16 + lane / 16 * 8) * 2))
            : "memory");
        #pragma unroll
        for (int nb8 = 0; nb8 < 2; nb8++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                : "r"(s_a_addr + (unsigned int)((s0 + nb8 * 16 + 8 * (lane / 16) + lane % 8) * 144 + (kb5 * 16 + lane % 16 / 8 * 8) * 2))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb8 * 8)[0]), "=f"((acc_s + nb8 * 8)[1]), "=f"((acc_s + nb8 * 8)[2]), "=f"((acc_s + nb8 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]), "f"(((kb5 == 0) ? 0.0f : (acc_s + nb8 * 8)[0])), "f"(((kb5 == 0) ? 0.0f : (acc_s + nb8 * 8)[1])), "f"(((kb5 == 0) ? 0.0f : (acc_s + nb8 * 8)[2])), "f"(((kb5 == 0) ? 0.0f : (acc_s + nb8 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb8 * 8 + 4)[0]), "=f"((acc_s + nb8 * 8 + 4)[1]), "=f"((acc_s + nb8 * 8 + 4)[2]), "=f"((acc_s + nb8 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]), "f"(((kb5 == 0) ? 0.0f : (acc_s + nb8 * 8 + 4)[0])), "f"(((kb5 == 0) ? 0.0f : (acc_s + nb8 * 8 + 4)[1])), "f"(((kb5 == 0) ? 0.0f : (acc_s + nb8 * 8 + 4)[2])), "f"(((kb5 == 0) ? 0.0f : (acc_s + nb8 * 8 + 4)[3])));
        }
    }
    #pragma unroll
    for (int p_1 = 0; p_1 < 2; p_1++) {
        #pragma unroll
        for (int e4 = 0; e4 < 8; e4++) {
            vals[e4] = acc_s[p_1 * 8 + e4];
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
            pk[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(s_da2_addr + (unsigned int)st_off + (unsigned int)(p_1 * 32));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3]))
            : "memory");
    }
    __syncthreads();
    #pragma unroll
    for (int kb6 = 0; kb6 < 4; kb6++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_mn[0]), "=r"(a_frag_mn[1]), "=r"(a_frag_mn[2]), "=r"(a_frag_mn[3])
            : "r"(s_a_addr + (unsigned int)((kb6 * 16 + 8 * (lane / 16) + lane % 8) * 144 + (rb * 16 + lane % 16 / 8 * 8) * 2))
            : "memory");
        #pragma unroll
        for (int nb10 = 0; nb10 < 2; nb10++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_da2_addr + (unsigned int)((kb6 * 16 + lane % 16) * 144 + (s0 + nb10 * 16 + lane / 16 * 8) * 2))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb10 * 8)[0]), "=f"((acc_s + nb10 * 8)[1]), "=f"((acc_s + nb10 * 8)[2]), "=f"((acc_s + nb10 * 8)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8)[0])), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8)[1])), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8)[2])), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb10 * 8 + 4)[0]), "=f"((acc_s + nb10 * 8 + 4)[1]), "=f"((acc_s + nb10 * 8 + 4)[2]), "=f"((acc_s + nb10 * 8 + 4)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8 + 4)[0])), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8 + 4)[1])), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8 + 4)[2])), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8 + 4)[3])));
        }
    }
    #pragma unroll
    for (int kb7 = 0; kb7 < 8; kb7++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_kn_addr + (unsigned int)(kb7 / 4 * 9216) + (unsigned int)((rb * 16 + lane % 16) * 144 + (kb7 % 4 * 16 + lane / 16 * 8) * 2))
            : "memory");
        #pragma unroll
        for (int nb12 = 0; nb12 < 2; nb12++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                : "r"(s_kn_addr + (unsigned int)(kb7 / 4 * 9216) + (unsigned int)((s0 + nb12 * 16 + 8 * (lane / 16) + lane % 8) * 144 + (kb7 % 4 * 16 + lane % 16 / 8 * 8) * 2))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb12 * 8)[0]), "=f"((acc + nb12 * 8)[1]), "=f"((acc + nb12 * 8)[2]), "=f"((acc + nb12 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]), "f"(((kb7 == 0) ? 0.0f : (acc + nb12 * 8)[0])), "f"(((kb7 == 0) ? 0.0f : (acc + nb12 * 8)[1])), "f"(((kb7 == 0) ? 0.0f : (acc + nb12 * 8)[2])), "f"(((kb7 == 0) ? 0.0f : (acc + nb12 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb12 * 8 + 4)[0]), "=f"((acc + nb12 * 8 + 4)[1]), "=f"((acc + nb12 * 8 + 4)[2]), "=f"((acc + nb12 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]), "f"(((kb7 == 0) ? 0.0f : (acc + nb12 * 8 + 4)[0])), "f"(((kb7 == 0) ? 0.0f : (acc + nb12 * 8 + 4)[1])), "f"(((kb7 == 0) ? 0.0f : (acc + nb12 * 8 + 4)[2])), "f"(((kb7 == 0) ? 0.0f : (acc + nb12 * 8 + 4)[3])));
        }
    }
    float col_part[8];
    col_part[0] = 0.0f;
    col_part[1] = 0.0f;
    col_part[2] = 0.0f;
    col_part[3] = 0.0f;
    col_part[4] = 0.0f;
    col_part[5] = 0.0f;
    col_part[6] = 0.0f;
    col_part[7] = 0.0f;
    float valsb[8];
    #pragma unroll
    for (int p_2 = 0; p_2 < 2; p_2++) {
        #pragma unroll
        for (int q_1 = 0; q_1 < 2; q_1++) {
            int col0_1 = s0 + p_2 * 16 + q_1 * 8 + col_q;
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gq[0])), "=r"(*reinterpret_cast<uint32_t*>(&gq[(0) + 1]))
                : "r"(s_g_addr + (unsigned int)(col0_1 * 4)));
            float2 _f2_12 = make_float2(g_a, g_a);
            float2 _f2_13 = make_float2(-gq[0], -gq[1]);
            float2 ea = add_f32x2_noftz(_f2_12, _f2_13);
            float2 _f2_14 = make_float2(g_b, g_b);
            float2 _f2_15 = make_float2(-gq[0], -gq[1]);
            float2 eb = add_f32x2_noftz(_f2_14, _f2_15);
            float2 _f2_16 = make_float2(-acc_s[(2 * p_2 + q_1) * 4], -acc_s[(2 * p_2 + q_1) * 4 + 1]);
            float _exp2_3 = approx_exp2(ea.x);
            float _exp2_4 = approx_exp2(ea.y);
            float2 _f2_17 = make_float2(_exp2_3, _exp2_4);
            float2 va = mul_f32x2_noftz(_f2_16, _f2_17);
            float2 _f2_18 = make_float2(-acc_s[(2 * p_2 + q_1) * 4 + 2], -acc_s[(2 * p_2 + q_1) * 4 + 3]);
            float _exp2_5 = approx_exp2(eb.x);
            float _exp2_6 = approx_exp2(eb.y);
            float2 _f2_19 = make_float2(_exp2_5, _exp2_6);
            float2 vb = mul_f32x2_noftz(_f2_18, _f2_19);
            float val_a0_1 = 0.0f;
            float val_a1_1 = 0.0f;
            float val_b0_1 = 0.0f;
            float val_b1_1 = 0.0f;
            if (col0_1 < row_a) {
                if (row_a < n_valid) {
                    val_a0_1 = va.x;
                }
            }
            if (row_a > col0_1 + 1) {
                if (row_a < n_valid) {
                    val_a1_1 = va.y;
                }
            }
            if (col0_1 < row_b) {
                if (row_b < n_valid) {
                    val_b0_1 = vb.x;
                }
            }
            if (row_b > col0_1 + 1) {
                if (row_b < n_valid) {
                    val_b1_1 = vb.y;
                }
            }
            float2 _f2_20 = make_float2(val_a0_1, val_a1_1);
            float2 _f2_21 = make_float2(acc[(2 * p_2 + q_1) * 4], acc[(2 * p_2 + q_1) * 4 + 1]);
            float2 ta = mul_f32x2_noftz(_f2_20, _f2_21);
            float2 _f2_22 = make_float2(val_b0_1, val_b1_1);
            float2 _f2_23 = make_float2(acc[(2 * p_2 + q_1) * 4 + 2], acc[(2 * p_2 + q_1) * 4 + 3]);
            float2 tb = mul_f32x2_noftz(_f2_22, _f2_23);
            float ada_a0 = ta.x * beta_a;
            float ada_b0 = tb.x * beta_b;
            dg_a = dg_a + ada_a0;
            dg_b = dg_b + ada_b0;
            col_part[(2 * p_2 + q_1) * 2] = ada_a0 + ada_b0;
            float ada_a1 = ta.y * beta_a;
            float ada_b1 = tb.y * beta_b;
            dg_a = dg_a + ada_a1;
            dg_b = dg_b + ada_b1;
            col_part[(2 * p_2 + q_1) * 2 + 1] = ada_a1 + ada_b1;
            vals[q_1 * 4] = val_a0_1;
            vals[q_1 * 4 + 1] = val_a1_1;
            vals[q_1 * 4 + 2] = val_b0_1;
            vals[q_1 * 4 + 3] = val_b1_1;
            float2 _f2_24 = make_float2(val_a0_1, val_a1_1);
            float2 _f2_25 = make_float2(beta_a, beta_a);
            float2 vba = mul_f32x2_noftz(_f2_24, _f2_25);
            float2 _f2_26 = make_float2(val_b0_1, val_b1_1);
            float2 _f2_27 = make_float2(beta_b, beta_b);
            float2 vbb = mul_f32x2_noftz(_f2_26, _f2_27);
            valsb[q_1 * 4] = vba.x;
            valsb[q_1 * 4 + 1] = vba.y;
            valsb[q_1 * 4 + 2] = vbb.x;
            valsb[q_1 * 4 + 3] = vbb.y;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals[_lp*2 + 0], vals[_lp*2+1 + 0]));
            pk[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(s_da3_addr + (unsigned int)st_off + (unsigned int)(p_2 * 32));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3]))
            : "memory");
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(valsb[_lp*2 + 0], valsb[_lp*2+1 + 0]));
            pk[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(s_da3b_addr + (unsigned int)st_off + (unsigned int)(p_2 * 32));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&pk[0])), "r"(*reinterpret_cast<const uint32_t*>(&pk[1])), "r"(*reinterpret_cast<const uint32_t*>(&pk[2])), "r"(*reinterpret_cast<const uint32_t*>(&pk[3]))
            : "memory");
    }
    #pragma unroll
    for (int ci = 0; ci < 8; ci++) {
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, col_part[ci], 4);
        col_part[ci] = col_part[ci] + _shfl_xor_0;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, col_part[ci], 8);
        col_part[ci] = col_part[ci] + _shfl_xor_1;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, col_part[ci], 16);
        col_part[ci] = col_part[ci] + _shfl_xor_2;
    }
    if (lane_0 < 4) {
        #pragma unroll
        for (int nb14 = 0; nb14 < 4; nb14++) {
            #pragma unroll
            for (int e7 = 0; e7 < 2; e7++) {
                s_col[rb * 64 + s0 + nb14 * 8 + col_q + e7] = col_part[nb14 * 2 + e7];
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int kb8 = 0; kb8 < 4; kb8++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_da3_addr + (unsigned int)((rb * 16 + lane % 16) * 144 + (kb8 * 16 + lane / 16 * 8) * 2))
            : "memory");
        #pragma unroll
        for (int nb15 = 0; nb15 < 4; nb15++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(kn_h + ((kb8 * 16 + lane % 16) * 144 + (nb15 * 16 + lane / 16 * 8) * 2))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb15 * 8)[0]), "=f"((acc + nb15 * 8)[1]), "=f"((acc + nb15 * 8)[2]), "=f"((acc + nb15 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb8 == 0) ? 0.0f : (acc + nb15 * 8)[0])), "f"(((kb8 == 0) ? 0.0f : (acc + nb15 * 8)[1])), "f"(((kb8 == 0) ? 0.0f : (acc + nb15 * 8)[2])), "f"(((kb8 == 0) ? 0.0f : (acc + nb15 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb15 * 8 + 4)[0]), "=f"((acc + nb15 * 8 + 4)[1]), "=f"((acc + nb15 * 8 + 4)[2]), "=f"((acc + nb15 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb8 == 0) ? 0.0f : (acc + nb15 * 8 + 4)[0])), "f"(((kb8 == 0) ? 0.0f : (acc + nb15 * 8 + 4)[1])), "f"(((kb8 == 0) ? 0.0f : (acc + nb15 * 8 + 4)[2])), "f"(((kb8 == 0) ? 0.0f : (acc + nb15 * 8 + 4)[3])));
        }
    }
    #pragma unroll
    for (int nb16 = 0; nb16 < 8; nb16++) {
        int kcol2 = nb16 * 8 + col_q;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(kn_h + (row_a * 144 + kcol2 * 2)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(kn_h + (row_b * 144 + kcol2 * 2)));
        float _fma_4 = __fmaf_rn(acc[nb16 * 4], __uint_as_float(pair_a[0] << 16), db_a);
        db_a = _fma_4;
        float _fma_5 = __fmaf_rn(acc[nb16 * 4 + 2], __uint_as_float(pair_b[0] << 16), db_b);
        db_b = _fma_5;
        float _fma_6 = __fmaf_rn(acc[nb16 * 4 + 1], __uint_as_float(pair_a[0] & 4294901760u), db_a);
        db_a = _fma_6;
        float _fma_7 = __fmaf_rn(acc[nb16 * 4 + 3], __uint_as_float(pair_b[0] & 4294901760u), db_b);
        db_b = _fma_7;
        float2 _f2_28 = make_float2(acc[nb16 * 4], acc[nb16 * 4 + 1]);
        float2 _f2_29 = make_float2(beta_a, beta_a);
        float2 dk_a = mul_f32x2_noftz(_f2_28, _f2_29);
        float2 _f2_30 = make_float2(acc[nb16 * 4 + 2], acc[nb16 * 4 + 3]);
        float2 _f2_31 = make_float2(beta_b, beta_b);
        float2 dk_b = mul_f32x2_noftz(_f2_30, _f2_31);
        acc[nb16 * 4] = dk_a.x;
        acc[nb16 * 4 + 1] = dk_a.y;
        acc[nb16 * 4 + 2] = dk_b.x;
        acc[nb16 * 4 + 3] = dk_b.y;
    }
    #pragma unroll
    for (int kb9 = 0; kb9 < 4; kb9++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_mn[0]), "=r"(a_frag_mn[1]), "=r"(a_frag_mn[2]), "=r"(a_frag_mn[3])
            : "r"(s_da3b_addr + (unsigned int)((kb9 * 16 + 8 * (lane / 16) + lane % 8) * 144 + (rb * 16 + lane % 16 / 8 * 8) * 2))
            : "memory");
        #pragma unroll
        for (int nb17 = 0; nb17 < 4; nb17++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(kn_h + ((kb9 * 16 + lane % 16) * 144 + (nb17 * 16 + lane / 16 * 8) * 2))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc + nb17 * 8)[0]), "+f"((acc + nb17 * 8)[1]), "+f"((acc + nb17 * 8)[2]), "+f"((acc + nb17 * 8)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc + nb17 * 8 + 4)[0]), "+f"((acc + nb17 * 8 + 4)[1]), "+f"((acc + nb17 * 8 + 4)[2]), "+f"((acc + nb17 * 8 + 4)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
        }
    }
    #pragma unroll
    for (int kb3 = 0; kb3 < 4; kb3++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_mn[0]), "=r"(a_frag_mn[1]), "=r"(a_frag_mn[2]), "=r"(a_frag_mn[3])
            : "r"(s_a_addr + (unsigned int)((kb3 * 16 + 8 * (lane / 16) + lane % 8) * 144 + (rb * 16 + lane % 16 / 8 * 8) * 2))
            : "memory");
        #pragma unroll
        for (int nb3 = 0; nb3 < 4; nb3++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(dw_h + ((kb3 * 16 + lane % 16) * 144 + (nb3 * 16 + lane / 16 * 8) * 2))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb3 * 8)[0]), "=f"((acc_s + nb3 * 8)[1]), "=f"((acc_s + nb3 * 8)[2]), "=f"((acc_s + nb3 * 8)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb3 == 0) ? 0.0f : (acc_s + nb3 * 8)[0])), "f"(((kb3 == 0) ? 0.0f : (acc_s + nb3 * 8)[1])), "f"(((kb3 == 0) ? 0.0f : (acc_s + nb3 * 8)[2])), "f"(((kb3 == 0) ? 0.0f : (acc_s + nb3 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb3 * 8 + 4)[0]), "=f"((acc_s + nb3 * 8 + 4)[1]), "=f"((acc_s + nb3 * 8 + 4)[2]), "=f"((acc_s + nb3 * 8 + 4)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb3 == 0) ? 0.0f : (acc_s + nb3 * 8 + 4)[0])), "f"(((kb3 == 0) ? 0.0f : (acc_s + nb3 * 8 + 4)[1])), "f"(((kb3 == 0) ? 0.0f : (acc_s + nb3 * 8 + 4)[2])), "f"(((kb3 == 0) ? 0.0f : (acc_s + nb3 * 8 + 4)[3])));
        }
    }
    #pragma unroll
    for (int nb4 = 0; nb4 < 8; nb4++) {
        int kcol = nb4 * 8 + col_q;
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_a[0])) : "r"(kn_h + (row_a * 144 + kcol * 2)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&pair_b[0])) : "r"(kn_h + (row_b * 144 + kcol * 2)));
        float k_a0 = __uint_as_float(pair_a[0] << 16);
        float k_b0 = __uint_as_float(pair_b[0] << 16);
        float k_a1 = __uint_as_float(pair_a[0] & 4294901760u);
        float k_b1 = __uint_as_float(pair_b[0] & 4294901760u);
        float2 _f2_32 = make_float2(k_a0, k_a1);
        float2 _f2_33 = make_float2(gexp_a, gexp_a);
        float2 kg_a = mul_f32x2_noftz(_f2_32, _f2_33);
        float2 _f2_34 = make_float2(k_b0, k_b1);
        float2 _f2_35 = make_float2(gexp_b, gexp_b);
        float2 kg_b = mul_f32x2_noftz(_f2_34, _f2_35);
        float2 _f2_36 = make_float2(k_a0, k_a1);
        float2 _f2_37 = make_float2(bg_a, bg_a);
        float2 kbg_a = mul_f32x2_noftz(_f2_36, _f2_37);
        float2 _f2_38 = make_float2(k_b0, k_b1);
        float2 _f2_39 = make_float2(bg_b, bg_b);
        float2 kbg_b = mul_f32x2_noftz(_f2_38, _f2_39);
        float2 _f2_40 = make_float2(acc_s[nb4 * 4], acc_s[nb4 * 4 + 1]);
        float2 _f2_41 = make_float2(bg_a, bg_a);
        float2 _f2_42 = make_float2(acc[nb4 * 4], acc[nb4 * 4 + 1]);
        float2 dk3_a = fma_f32x2_rn_noftz(_f2_40, _f2_41, _f2_42);
        float2 _f2_43 = make_float2(acc_s[nb4 * 4 + 2], acc_s[nb4 * 4 + 3]);
        float2 _f2_44 = make_float2(bg_b, bg_b);
        float2 _f2_45 = make_float2(acc[nb4 * 4 + 2], acc[nb4 * 4 + 3]);
        float2 dk3_b = fma_f32x2_rn_noftz(_f2_43, _f2_44, _f2_45);
        float _fma_8 = __fmaf_rn(acc_s[nb4 * 4], kg_a.x, db_a);
        db_a = _fma_8;
        float _fma_9 = __fmaf_rn(acc_s[nb4 * 4 + 2], kg_b.x, db_b);
        db_b = _fma_9;
        float _fma_10 = __fmaf_rn(acc_s[nb4 * 4], kbg_a.x, dg_a);
        dg_a = _fma_10;
        float _fma_11 = __fmaf_rn(acc_s[nb4 * 4 + 2], kbg_b.x, dg_b);
        dg_b = _fma_11;
        float _fma_12 = __fmaf_rn(acc_s[nb4 * 4 + 1], kg_a.y, db_a);
        db_a = _fma_12;
        float _fma_13 = __fmaf_rn(acc_s[nb4 * 4 + 3], kg_b.y, db_b);
        db_b = _fma_13;
        float _fma_14 = __fmaf_rn(acc_s[nb4 * 4 + 1], kbg_a.y, dg_a);
        dg_a = _fma_14;
        float _fma_15 = __fmaf_rn(acc_s[nb4 * 4 + 3], kbg_b.y, dg_b);
        dg_b = _fma_15;
        acc[nb4 * 4] = dk3_a.x;
        acc[nb4 * 4 + 1] = dk3_a.y;
        acc[nb4 * 4 + 2] = dk3_b.x;
        acc[nb4 * 4 + 3] = dk3_b.y;
    }
    #pragma unroll
    for (int nb18 = 0; nb18 < 8; nb18++) {
        if (row_a < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb18 * 4 + 0], acc[nb18 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dk2_out))[out_a + (long long)(nb18 * 8)]) = _pk;
            }
        }
        if (row_b < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb18 * 4 + 2 + 0], acc[nb18 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dk2_out))[out_b + (long long)(nb18 * 8)]) = _pk;
            }
        }
    }
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, db_a, 1);
    db_a = db_a + _shfl_xor_3;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, db_a, 2);
    db_a = db_a + _shfl_xor_4;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, db_b, 1);
    db_b = db_b + _shfl_xor_5;
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, db_b, 2);
    db_b = db_b + _shfl_xor_6;
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, dg_a, 1);
    dg_a = dg_a + _shfl_xor_7;
    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, dg_a, 2);
    dg_a = dg_a + _shfl_xor_8;
    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, dg_b, 1);
    dg_b = dg_b + _shfl_xor_9;
    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, dg_b, 2);
    dg_b = dg_b + _shfl_xor_10;
    if (lane_0 % 4 == 0) {
        s_dbp[nh * 64 + row_a] = db_a;
        s_dbp[nh * 64 + row_b] = db_b;
        s_dgp[nh * 64 + row_a] = dg_a;
        s_dgp[nh * 64 + row_b] = dg_b;
    }
    __syncthreads();
    if (tid_2 < n_valid) {
        long long gate_row_index = (long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv;
        float colsum = s_col[tid_2] + s_col[64 + tid_2] + s_col[128 + tid_2] + s_col[192 + tid_2];
        float db_row = s_dbp[tid_2] + s_dbp[64 + tid_2];
        float dg_row = s_dgp[tid_2] + s_dgp[64 + tid_2];
        dbeta_out[gate_row_index] = db_row;
        dg2_out[gate_row_index] = dg_row - colsum;
    }
}

} // extern "C"
