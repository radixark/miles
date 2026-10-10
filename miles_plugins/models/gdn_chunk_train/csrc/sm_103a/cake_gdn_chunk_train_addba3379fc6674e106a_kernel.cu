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
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 16384
#define SMEM_S_V_OFF 32768
#define SMEM_S_V_STAGE_BYTES 16384
#define SMEM_S_V_STRIDE 16384
#define SMEM_S_A_OFF 16384
#define SMEM_S_A_STAGE_BYTES 8192
#define SMEM_S_A_STRIDE 8192
#define SMEM_S_LT_OFF 16384
#define SMEM_S_LT_STAGE_BYTES 16384
#define SMEM_S_LT_STRIDE 16384
#define SMEM_S_X_OFF 32768
#define SMEM_S_X_STAGE_BYTES 16384
#define SMEM_S_X_STRIDE 16384
#define SMEM_S_T_OFF 49152
#define SMEM_S_T_STAGE_BYTES 3840
#define SMEM_S_T_STRIDE 3840
#define SMEM_S_G_OFF 52992
#define SMEM_S_G_STAGE_BYTES 256
#define SMEM_S_G_STRIDE 256
#define SMEM_S_B_OFF 53248
#define SMEM_S_B_STAGE_BYTES 256
#define SMEM_S_B_STRIDE 256
#define SMEM_TOTAL 53504
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_chunk_train_addba3379fc6674e106a(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ v, float* __restrict__ g_cs, float* __restrict__ beta, __nv_bfloat16* __restrict__ A_out, __nv_bfloat16* __restrict__ w_out, __nv_bfloat16* __restrict__ u_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, int recompute)
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
    __nv_bfloat16* s_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_S_A_OFF);
    const int s_a_addr = smem + SMEM_S_A_OFF;
    float* s_lt = reinterpret_cast<float*>(smem_raw + SMEM_S_LT_OFF);
    const int s_lt_addr = smem + SMEM_S_LT_OFF;
    float* s_x = reinterpret_cast<float*>(smem_raw + SMEM_S_X_OFF);
    const int s_x_addr = smem + SMEM_S_X_OFF;
    float* s_t = reinterpret_cast<float*>(smem_raw + SMEM_S_T_OFF);
    const int s_t_addr = smem + SMEM_S_T_OFF;
    float* s_g = reinterpret_cast<float*>(smem_raw + SMEM_S_G_OFF);
    const int s_g_addr = smem + SMEM_S_G_OFF;
    float* s_b = reinterpret_cast<float*>(smem_raw + SMEM_S_B_OFF);
    const int s_b_addr = smem + SMEM_S_B_OFF;

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
    int frag_r = lane_0 / 4;
    int col_q = lane_0 % 4 * 2;
    int m0 = warp_1 / 2 * 32;
    int n0 = warp_1 % 2 * 32;
    int nn0 = warp_1 % 2 * 64;
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + tid / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)(tid % 16 * 8))), "r"((n_valid > tid / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (128 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((n_valid > (128 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (256 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((n_valid > (256 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (384 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((n_valid > (384 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (512 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((n_valid > (512 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (640 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((n_valid > (640 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (768 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((n_valid > (768 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (896 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((n_valid > (896 + tid) / 16) ? 16 : 0));
    if (recompute != 0) {
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            int aidx = it * 128 + tid_2;
            int ld_row = aidx / 8;
            int ld_col = aidx % 8 * 8;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_a_addr + (unsigned int)(ld_col / 64 * 8192 + (ld_row * 128 + ld_col % 64 * 2 ^ (ld_row * 128 + ld_col % 64 * 2 >> 7 & 7) << 4))), "l"(A_out + (((long long)(tok0 + ld_row) * (long long)num_v_heads + (long long)hv) * 64 + (long long)ld_col)), "r"((ld_row < n_valid) ? 16 : 0));
        }
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + tid / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)(tid % 16 * 8))), "r"((n_valid > tid / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (128 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((n_valid > (128 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (256 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((n_valid > (256 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (384 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((n_valid > (384 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (512 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((n_valid > (512 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (640 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((n_valid > (640 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (768 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((n_valid > (768 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (896 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((n_valid > (896 + tid) / 16) ? 16 : 0));
    }
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
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    unsigned int a_frag0[4];
    unsigned int a_frag1[4];
    unsigned int b_frag_k0[4];
    unsigned int b_frag_k1[4];
    unsigned int b_frag_mn[4];
    float acc[64];
    if (recompute == 0) {
        if (warp_1 != 1) {
            #pragma unroll
            for (int kb = 0; kb < 8; kb++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
                    : "r"(s_kn_addr + (unsigned int)((kb * 16 + lane / 16 * 8) / 64 * 8192 + ((m0 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
                    : "r"(s_kn_addr + (unsigned int)((kb * 16 + lane / 16 * 8) / 64 * 8192 + ((m0 + 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_k0[0]), "=r"(b_frag_k0[1]), "=r"(b_frag_k0[2]), "=r"(b_frag_k0[3])
                    : "r"(s_kn_addr + (unsigned int)((kb * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((n0 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((n0 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_k1[0]), "=r"(b_frag_k1[1]), "=r"(b_frag_k1[2]), "=r"(b_frag_k1[3])
                    : "r"(s_kn_addr + (unsigned int)((kb * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((n0 + 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((n0 + 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                    : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]), "f"(((kb == 0) ? 0.0f : acc[0])), "f"(((kb == 0) ? 0.0f : acc[1])), "f"(((kb == 0) ? 0.0f : acc[2])), "f"(((kb == 0) ? 0.0f : acc[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                    : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]), "f"(((kb == 0) ? 0.0f : acc[4])), "f"(((kb == 0) ? 0.0f : acc[(4) + 1])), "f"(((kb == 0) ? 0.0f : acc[(4) + 2])), "f"(((kb == 0) ? 0.0f : acc[(4) + 3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(acc[8]), "=f"(acc[(8) + 1]), "=f"(acc[(8) + 2]), "=f"(acc[(8) + 3])
                    : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]), "f"(((kb == 0) ? 0.0f : acc[8])), "f"(((kb == 0) ? 0.0f : acc[(8) + 1])), "f"(((kb == 0) ? 0.0f : acc[(8) + 2])), "f"(((kb == 0) ? 0.0f : acc[(8) + 3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(acc[12]), "=f"(acc[(12) + 1]), "=f"(acc[(12) + 2]), "=f"(acc[(12) + 3])
                    : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]), "f"(((kb == 0) ? 0.0f : acc[12])), "f"(((kb == 0) ? 0.0f : acc[(12) + 1])), "f"(((kb == 0) ? 0.0f : acc[(12) + 2])), "f"(((kb == 0) ? 0.0f : acc[(12) + 3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(acc[16]), "=f"(acc[(16) + 1]), "=f"(acc[(16) + 2]), "=f"(acc[(16) + 3])
                    : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[0]), "r"(b_frag_k0[1]), "f"(((kb == 0) ? 0.0f : acc[16])), "f"(((kb == 0) ? 0.0f : acc[(16) + 1])), "f"(((kb == 0) ? 0.0f : acc[(16) + 2])), "f"(((kb == 0) ? 0.0f : acc[(16) + 3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(acc[20]), "=f"(acc[(20) + 1]), "=f"(acc[(20) + 2]), "=f"(acc[(20) + 3])
                    : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k0[2]), "r"(b_frag_k0[(2) + 1]), "f"(((kb == 0) ? 0.0f : acc[20])), "f"(((kb == 0) ? 0.0f : acc[(20) + 1])), "f"(((kb == 0) ? 0.0f : acc[(20) + 2])), "f"(((kb == 0) ? 0.0f : acc[(20) + 3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(acc[24]), "=f"(acc[(24) + 1]), "=f"(acc[(24) + 2]), "=f"(acc[(24) + 3])
                    : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[0]), "r"(b_frag_k1[1]), "f"(((kb == 0) ? 0.0f : acc[24])), "f"(((kb == 0) ? 0.0f : acc[(24) + 1])), "f"(((kb == 0) ? 0.0f : acc[(24) + 2])), "f"(((kb == 0) ? 0.0f : acc[(24) + 3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"(acc[28]), "=f"(acc[(28) + 1]), "=f"(acc[(28) + 2]), "=f"(acc[(28) + 3])
                    : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_k1[2]), "r"(b_frag_k1[(2) + 1]), "f"(((kb == 0) ? 0.0f : acc[28])), "f"(((kb == 0) ? 0.0f : acc[(28) + 1])), "f"(((kb == 0) ? 0.0f : acc[(28) + 2])), "f"(((kb == 0) ? 0.0f : acc[(28) + 3])));
            }
            float g_row[4];
            float b_row[4];
            #pragma unroll
            for (int rr = 0; rr < 4; rr++) {
                int row_rr = m0 + rr / 2 * 16 + rr % 2 * 8 + frag_r;
                g_row[rr] = s_g[row_rr];
                b_row[rr] = s_b[row_rr];
            }
            #pragma unroll
            for (int mb = 0; mb < 2; mb++) {
                #pragma unroll
                for (int nb = 0; nb < 2; nb++) {
                    if (m0 / 16 + mb >= n0 / 16 + nb) {
                        #pragma unroll
                        for (int nh = 0; nh < 2; nh++) {
                            #pragma unroll
                            for (int e = 0; e < 2; e++) {
                                int col = n0 + nb * 16 + nh * 8 + col_q + e;
                                float g_col = s_g[col];
                                #pragma unroll
                                for (int half = 0; half < 2; half++) {
                                    int row = m0 + mb * 16 + half * 8 + frag_r;
                                    float val = 0.0f;
                                    if (col < row) {
                                        if (row < n_valid) {
                                            float _exp2_0 = approx_exp2(g_row[mb * 2 + half] - g_col);
                                            val = b_row[mb * 2 + half] * _exp2_0 * acc[(mb * 2 + nb) * 8 + nh * 4 + half * 2 + e];
                                        }
                                    }
                                    s_lt[col * 64 + (row ^ (col >> 1 & 3) << 3)] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        float zero4[4];
        zero4[0] = 0.0f;
        zero4[1] = 0.0f;
        zero4[2] = 0.0f;
        zero4[3] = 0.0f;
        #pragma unroll
        for (int zt = 0; zt < 3; zt++) {
            int zidx = zt * 128 + tid_2;
            int zblk = zidx / 64;
            int zr = zidx % 64 / 4;
            int zc = zidx % 4 * 4;
            int zbi = 0;
            int zbj = zblk + 1;
            if (zblk >= 3) {
                zbi = 1;
                zbj = zblk - 1;
            }
            if (zblk == 5) {
                zbi = 2;
                zbj = 3;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                "r"(s_x_addr + (unsigned int)(((zbi * 16 + zr) * 64 + (zbj * 16 + zc ^ (zbi * 16 + zr >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&zero4[0])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&zero4[(0) + 3])));
        }
        if (tid_2 < 64) {
            int dblk = tid_2 / 16;
            int dcol = tid_2 % 16;
            int drow0 = dblk * 16;
            float xcol[16];
            #pragma unroll
            for (int di = 0; di < 16; di++) {
                float xv = 0.0f;
                if (di == dcol) {
                    xv = 1.0f;
                }
                #pragma unroll
                for (int dj = 0; dj < di; dj++) {
                    xv = xv - s_lt[(drow0 + dj) * 64 + (drow0 + di ^ (drow0 + dj >> 1 & 3) << 3)] * xcol[dj];
                }
                xcol[di] = xv;
            }
            #pragma unroll
            for (int di2 = 0; di2 < 16; di2++) {
                s_x[(drow0 + di2) * 64 + (drow0 + dcol ^ (drow0 + di2 >> 2 & 1) << 4)] = xcol[di2];
            }
        }
        __syncthreads();
        float acc16[16];
        float lv[4];
        float xv4[4];
        float xi16[16];
        float tv16[16];
        int rg = lane_0 / 4;
        int cg = lane_0 % 4;
        if (lane_0 < 16) {
            if (warp_1 < 3) {
                acc16[0] = 0.0f;
                acc16[1] = 0.0f;
                acc16[2] = 0.0f;
                acc16[3] = 0.0f;
                acc16[4] = 0.0f;
                acc16[5] = 0.0f;
                acc16[6] = 0.0f;
                acc16[7] = 0.0f;
                acc16[8] = 0.0f;
                acc16[9] = 0.0f;
                acc16[10] = 0.0f;
                acc16[11] = 0.0f;
                acc16[12] = 0.0f;
                acc16[13] = 0.0f;
                acc16[14] = 0.0f;
                acc16[15] = 0.0f;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((warp_1 * 16 * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((warp_1 * 16 * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 >> 2 & 1) << 4)) * 4)));
                float _fma_0 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_0;
                float _fma_1 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1;
                float _fma_2 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_2;
                float _fma_3 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_3;
                float _fma_4 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_4;
                float _fma_5 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_5;
                float _fma_6 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_6;
                float _fma_7 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_7;
                float _fma_8 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_8;
                float _fma_9 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_9;
                float _fma_10 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_10;
                float _fma_11 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_11;
                float _fma_12 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_12;
                float _fma_13 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_13;
                float _fma_14 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_14;
                float _fma_15 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_15;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 1 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 1 >> 2 & 1) << 4)) * 4)));
                float _fma_16 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_16;
                float _fma_17 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_17;
                float _fma_18 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_18;
                float _fma_19 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_19;
                float _fma_20 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_20;
                float _fma_21 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_21;
                float _fma_22 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_22;
                float _fma_23 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_23;
                float _fma_24 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_24;
                float _fma_25 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_25;
                float _fma_26 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_26;
                float _fma_27 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_27;
                float _fma_28 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_28;
                float _fma_29 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_29;
                float _fma_30 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_30;
                float _fma_31 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_31;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 2 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 2 >> 2 & 1) << 4)) * 4)));
                float _fma_32 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_32;
                float _fma_33 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_33;
                float _fma_34 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_34;
                float _fma_35 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_35;
                float _fma_36 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_36;
                float _fma_37 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_37;
                float _fma_38 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_38;
                float _fma_39 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_39;
                float _fma_40 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_40;
                float _fma_41 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_41;
                float _fma_42 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_42;
                float _fma_43 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_43;
                float _fma_44 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_44;
                float _fma_45 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_45;
                float _fma_46 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_46;
                float _fma_47 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_47;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 3 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 3 >> 2 & 1) << 4)) * 4)));
                float _fma_48 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_48;
                float _fma_49 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_49;
                float _fma_50 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_50;
                float _fma_51 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_51;
                float _fma_52 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_52;
                float _fma_53 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_53;
                float _fma_54 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_54;
                float _fma_55 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_55;
                float _fma_56 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_56;
                float _fma_57 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_57;
                float _fma_58 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_58;
                float _fma_59 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_59;
                float _fma_60 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_60;
                float _fma_61 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_61;
                float _fma_62 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_62;
                float _fma_63 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_63;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 4 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 4 >> 2 & 1) << 4)) * 4)));
                float _fma_64 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_64;
                float _fma_65 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_65;
                float _fma_66 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_66;
                float _fma_67 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_67;
                float _fma_68 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_68;
                float _fma_69 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_69;
                float _fma_70 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_70;
                float _fma_71 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_71;
                float _fma_72 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_72;
                float _fma_73 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_73;
                float _fma_74 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_74;
                float _fma_75 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_75;
                float _fma_76 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_76;
                float _fma_77 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_77;
                float _fma_78 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_78;
                float _fma_79 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_79;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 5 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 5 >> 2 & 1) << 4)) * 4)));
                float _fma_80 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_80;
                float _fma_81 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_81;
                float _fma_82 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_82;
                float _fma_83 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_83;
                float _fma_84 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_84;
                float _fma_85 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_85;
                float _fma_86 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_86;
                float _fma_87 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_87;
                float _fma_88 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_88;
                float _fma_89 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_89;
                float _fma_90 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_90;
                float _fma_91 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_91;
                float _fma_92 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_92;
                float _fma_93 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_93;
                float _fma_94 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_94;
                float _fma_95 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_95;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 6 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 6 >> 2 & 1) << 4)) * 4)));
                float _fma_96 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_96;
                float _fma_97 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_97;
                float _fma_98 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_98;
                float _fma_99 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_99;
                float _fma_100 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_100;
                float _fma_101 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_101;
                float _fma_102 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_102;
                float _fma_103 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_103;
                float _fma_104 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_104;
                float _fma_105 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_105;
                float _fma_106 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_106;
                float _fma_107 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_107;
                float _fma_108 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_108;
                float _fma_109 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_109;
                float _fma_110 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_110;
                float _fma_111 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_111;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 7 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 7 >> 2 & 1) << 4)) * 4)));
                float _fma_112 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_112;
                float _fma_113 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_113;
                float _fma_114 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_114;
                float _fma_115 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_115;
                float _fma_116 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_116;
                float _fma_117 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_117;
                float _fma_118 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_118;
                float _fma_119 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_119;
                float _fma_120 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_120;
                float _fma_121 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_121;
                float _fma_122 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_122;
                float _fma_123 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_123;
                float _fma_124 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_124;
                float _fma_125 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_125;
                float _fma_126 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_126;
                float _fma_127 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_127;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 8 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 8 >> 2 & 1) << 4)) * 4)));
                float _fma_128 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_128;
                float _fma_129 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_129;
                float _fma_130 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_130;
                float _fma_131 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_131;
                float _fma_132 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_132;
                float _fma_133 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_133;
                float _fma_134 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_134;
                float _fma_135 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_135;
                float _fma_136 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_136;
                float _fma_137 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_137;
                float _fma_138 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_138;
                float _fma_139 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_139;
                float _fma_140 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_140;
                float _fma_141 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_141;
                float _fma_142 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_142;
                float _fma_143 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_143;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 9 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 9 >> 2 & 1) << 4)) * 4)));
                float _fma_144 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_144;
                float _fma_145 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_145;
                float _fma_146 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_146;
                float _fma_147 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_147;
                float _fma_148 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_148;
                float _fma_149 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_149;
                float _fma_150 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_150;
                float _fma_151 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_151;
                float _fma_152 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_152;
                float _fma_153 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_153;
                float _fma_154 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_154;
                float _fma_155 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_155;
                float _fma_156 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_156;
                float _fma_157 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_157;
                float _fma_158 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_158;
                float _fma_159 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_159;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 10 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 10 >> 2 & 1) << 4)) * 4)));
                float _fma_160 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_160;
                float _fma_161 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_161;
                float _fma_162 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_162;
                float _fma_163 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_163;
                float _fma_164 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_164;
                float _fma_165 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_165;
                float _fma_166 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_166;
                float _fma_167 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_167;
                float _fma_168 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_168;
                float _fma_169 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_169;
                float _fma_170 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_170;
                float _fma_171 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_171;
                float _fma_172 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_172;
                float _fma_173 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_173;
                float _fma_174 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_174;
                float _fma_175 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_175;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 11 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 11 >> 2 & 1) << 4)) * 4)));
                float _fma_176 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_176;
                float _fma_177 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_177;
                float _fma_178 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_178;
                float _fma_179 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_179;
                float _fma_180 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_180;
                float _fma_181 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_181;
                float _fma_182 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_182;
                float _fma_183 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_183;
                float _fma_184 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_184;
                float _fma_185 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_185;
                float _fma_186 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_186;
                float _fma_187 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_187;
                float _fma_188 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_188;
                float _fma_189 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_189;
                float _fma_190 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_190;
                float _fma_191 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_191;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 12 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 12 >> 2 & 1) << 4)) * 4)));
                float _fma_192 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_192;
                float _fma_193 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_193;
                float _fma_194 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_194;
                float _fma_195 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_195;
                float _fma_196 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_196;
                float _fma_197 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_197;
                float _fma_198 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_198;
                float _fma_199 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_199;
                float _fma_200 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_200;
                float _fma_201 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_201;
                float _fma_202 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_202;
                float _fma_203 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_203;
                float _fma_204 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_204;
                float _fma_205 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_205;
                float _fma_206 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_206;
                float _fma_207 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_207;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 13 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 13 >> 2 & 1) << 4)) * 4)));
                float _fma_208 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_208;
                float _fma_209 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_209;
                float _fma_210 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_210;
                float _fma_211 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_211;
                float _fma_212 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_212;
                float _fma_213 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_213;
                float _fma_214 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_214;
                float _fma_215 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_215;
                float _fma_216 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_216;
                float _fma_217 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_217;
                float _fma_218 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_218;
                float _fma_219 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_219;
                float _fma_220 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_220;
                float _fma_221 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_221;
                float _fma_222 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_222;
                float _fma_223 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_223;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 14 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 14 >> 2 & 1) << 4)) * 4)));
                float _fma_224 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_224;
                float _fma_225 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_225;
                float _fma_226 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_226;
                float _fma_227 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_227;
                float _fma_228 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_228;
                float _fma_229 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_229;
                float _fma_230 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_230;
                float _fma_231 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_231;
                float _fma_232 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_232;
                float _fma_233 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_233;
                float _fma_234 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_234;
                float _fma_235 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_235;
                float _fma_236 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_236;
                float _fma_237 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_237;
                float _fma_238 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_238;
                float _fma_239 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_239;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + ((warp_1 + 1) * 16 + rg * 4 ^ (warp_1 * 16 + 15 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 15 >> 2 & 1) << 4)) * 4)));
                float _fma_240 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_240;
                float _fma_241 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_241;
                float _fma_242 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_242;
                float _fma_243 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_243;
                float _fma_244 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_244;
                float _fma_245 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_245;
                float _fma_246 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_246;
                float _fma_247 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_247;
                float _fma_248 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_248;
                float _fma_249 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_249;
                float _fma_250 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_250;
                float _fma_251 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_251;
                float _fma_252 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_252;
                float _fma_253 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_253;
                float _fma_254 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_254;
                float _fma_255 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_255;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_t_addr + (unsigned int)((warp_1 * 320 + rg * 4 * 20 + cg * 4) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[0])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (rg * 4 + 1) * 20 + cg * 4) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[4])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (rg * 4 + 2) * 20 + cg * 4) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[8])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (rg * 4 + 3) * 20 + cg * 4) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[12])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 3])));
            }
        }
        __syncthreads();
        if (lane_0 < 16) {
            if (warp_1 < 3) {
                acc16[0] = 0.0f;
                acc16[1] = 0.0f;
                acc16[2] = 0.0f;
                acc16[3] = 0.0f;
                acc16[4] = 0.0f;
                acc16[5] = 0.0f;
                acc16[6] = 0.0f;
                acc16[7] = 0.0f;
                acc16[8] = 0.0f;
                acc16[9] = 0.0f;
                acc16[10] = 0.0f;
                acc16[11] = 0.0f;
                acc16[12] = 0.0f;
                acc16[13] = 0.0f;
                acc16[14] = 0.0f;
                acc16[15] = 0.0f;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4) * 64 + ((warp_1 + 1) * 16 ^ ((warp_1 + 1) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 1) * 64 + ((warp_1 + 1) * 16 ^ ((warp_1 + 1) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[8])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 2) * 64 + ((warp_1 + 1) * 16 ^ ((warp_1 + 1) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[12])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 3) * 64 + ((warp_1 + 1) * 16 ^ ((warp_1 + 1) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 20 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 40 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 60 + cg * 4) * 4)));
                float _fma_256 = __fmaf_rn(xi16[0], tv16[0], acc16[0]);
                acc16[0] = _fma_256;
                float _fma_257 = __fmaf_rn(xi16[0], tv16[1], acc16[1]);
                acc16[1] = _fma_257;
                float _fma_258 = __fmaf_rn(xi16[0], tv16[2], acc16[2]);
                acc16[2] = _fma_258;
                float _fma_259 = __fmaf_rn(xi16[0], tv16[3], acc16[3]);
                acc16[3] = _fma_259;
                float _fma_260 = __fmaf_rn(xi16[4], tv16[0], acc16[4]);
                acc16[4] = _fma_260;
                float _fma_261 = __fmaf_rn(xi16[4], tv16[1], acc16[5]);
                acc16[5] = _fma_261;
                float _fma_262 = __fmaf_rn(xi16[4], tv16[2], acc16[6]);
                acc16[6] = _fma_262;
                float _fma_263 = __fmaf_rn(xi16[4], tv16[3], acc16[7]);
                acc16[7] = _fma_263;
                float _fma_264 = __fmaf_rn(xi16[8], tv16[0], acc16[8]);
                acc16[8] = _fma_264;
                float _fma_265 = __fmaf_rn(xi16[8], tv16[1], acc16[9]);
                acc16[9] = _fma_265;
                float _fma_266 = __fmaf_rn(xi16[8], tv16[2], acc16[10]);
                acc16[10] = _fma_266;
                float _fma_267 = __fmaf_rn(xi16[8], tv16[3], acc16[11]);
                acc16[11] = _fma_267;
                float _fma_268 = __fmaf_rn(xi16[12], tv16[0], acc16[12]);
                acc16[12] = _fma_268;
                float _fma_269 = __fmaf_rn(xi16[12], tv16[1], acc16[13]);
                acc16[13] = _fma_269;
                float _fma_270 = __fmaf_rn(xi16[12], tv16[2], acc16[14]);
                acc16[14] = _fma_270;
                float _fma_271 = __fmaf_rn(xi16[12], tv16[3], acc16[15]);
                acc16[15] = _fma_271;
                float _fma_272 = __fmaf_rn(xi16[1], tv16[4], acc16[0]);
                acc16[0] = _fma_272;
                float _fma_273 = __fmaf_rn(xi16[1], tv16[5], acc16[1]);
                acc16[1] = _fma_273;
                float _fma_274 = __fmaf_rn(xi16[1], tv16[6], acc16[2]);
                acc16[2] = _fma_274;
                float _fma_275 = __fmaf_rn(xi16[1], tv16[7], acc16[3]);
                acc16[3] = _fma_275;
                float _fma_276 = __fmaf_rn(xi16[5], tv16[4], acc16[4]);
                acc16[4] = _fma_276;
                float _fma_277 = __fmaf_rn(xi16[5], tv16[5], acc16[5]);
                acc16[5] = _fma_277;
                float _fma_278 = __fmaf_rn(xi16[5], tv16[6], acc16[6]);
                acc16[6] = _fma_278;
                float _fma_279 = __fmaf_rn(xi16[5], tv16[7], acc16[7]);
                acc16[7] = _fma_279;
                float _fma_280 = __fmaf_rn(xi16[9], tv16[4], acc16[8]);
                acc16[8] = _fma_280;
                float _fma_281 = __fmaf_rn(xi16[9], tv16[5], acc16[9]);
                acc16[9] = _fma_281;
                float _fma_282 = __fmaf_rn(xi16[9], tv16[6], acc16[10]);
                acc16[10] = _fma_282;
                float _fma_283 = __fmaf_rn(xi16[9], tv16[7], acc16[11]);
                acc16[11] = _fma_283;
                float _fma_284 = __fmaf_rn(xi16[13], tv16[4], acc16[12]);
                acc16[12] = _fma_284;
                float _fma_285 = __fmaf_rn(xi16[13], tv16[5], acc16[13]);
                acc16[13] = _fma_285;
                float _fma_286 = __fmaf_rn(xi16[13], tv16[6], acc16[14]);
                acc16[14] = _fma_286;
                float _fma_287 = __fmaf_rn(xi16[13], tv16[7], acc16[15]);
                acc16[15] = _fma_287;
                float _fma_288 = __fmaf_rn(xi16[2], tv16[8], acc16[0]);
                acc16[0] = _fma_288;
                float _fma_289 = __fmaf_rn(xi16[2], tv16[9], acc16[1]);
                acc16[1] = _fma_289;
                float _fma_290 = __fmaf_rn(xi16[2], tv16[10], acc16[2]);
                acc16[2] = _fma_290;
                float _fma_291 = __fmaf_rn(xi16[2], tv16[11], acc16[3]);
                acc16[3] = _fma_291;
                float _fma_292 = __fmaf_rn(xi16[6], tv16[8], acc16[4]);
                acc16[4] = _fma_292;
                float _fma_293 = __fmaf_rn(xi16[6], tv16[9], acc16[5]);
                acc16[5] = _fma_293;
                float _fma_294 = __fmaf_rn(xi16[6], tv16[10], acc16[6]);
                acc16[6] = _fma_294;
                float _fma_295 = __fmaf_rn(xi16[6], tv16[11], acc16[7]);
                acc16[7] = _fma_295;
                float _fma_296 = __fmaf_rn(xi16[10], tv16[8], acc16[8]);
                acc16[8] = _fma_296;
                float _fma_297 = __fmaf_rn(xi16[10], tv16[9], acc16[9]);
                acc16[9] = _fma_297;
                float _fma_298 = __fmaf_rn(xi16[10], tv16[10], acc16[10]);
                acc16[10] = _fma_298;
                float _fma_299 = __fmaf_rn(xi16[10], tv16[11], acc16[11]);
                acc16[11] = _fma_299;
                float _fma_300 = __fmaf_rn(xi16[14], tv16[8], acc16[12]);
                acc16[12] = _fma_300;
                float _fma_301 = __fmaf_rn(xi16[14], tv16[9], acc16[13]);
                acc16[13] = _fma_301;
                float _fma_302 = __fmaf_rn(xi16[14], tv16[10], acc16[14]);
                acc16[14] = _fma_302;
                float _fma_303 = __fmaf_rn(xi16[14], tv16[11], acc16[15]);
                acc16[15] = _fma_303;
                float _fma_304 = __fmaf_rn(xi16[3], tv16[12], acc16[0]);
                acc16[0] = _fma_304;
                float _fma_305 = __fmaf_rn(xi16[3], tv16[13], acc16[1]);
                acc16[1] = _fma_305;
                float _fma_306 = __fmaf_rn(xi16[3], tv16[14], acc16[2]);
                acc16[2] = _fma_306;
                float _fma_307 = __fmaf_rn(xi16[3], tv16[15], acc16[3]);
                acc16[3] = _fma_307;
                float _fma_308 = __fmaf_rn(xi16[7], tv16[12], acc16[4]);
                acc16[4] = _fma_308;
                float _fma_309 = __fmaf_rn(xi16[7], tv16[13], acc16[5]);
                acc16[5] = _fma_309;
                float _fma_310 = __fmaf_rn(xi16[7], tv16[14], acc16[6]);
                acc16[6] = _fma_310;
                float _fma_311 = __fmaf_rn(xi16[7], tv16[15], acc16[7]);
                acc16[7] = _fma_311;
                float _fma_312 = __fmaf_rn(xi16[11], tv16[12], acc16[8]);
                acc16[8] = _fma_312;
                float _fma_313 = __fmaf_rn(xi16[11], tv16[13], acc16[9]);
                acc16[9] = _fma_313;
                float _fma_314 = __fmaf_rn(xi16[11], tv16[14], acc16[10]);
                acc16[10] = _fma_314;
                float _fma_315 = __fmaf_rn(xi16[11], tv16[15], acc16[11]);
                acc16[11] = _fma_315;
                float _fma_316 = __fmaf_rn(xi16[15], tv16[12], acc16[12]);
                acc16[12] = _fma_316;
                float _fma_317 = __fmaf_rn(xi16[15], tv16[13], acc16[13]);
                acc16[13] = _fma_317;
                float _fma_318 = __fmaf_rn(xi16[15], tv16[14], acc16[14]);
                acc16[14] = _fma_318;
                float _fma_319 = __fmaf_rn(xi16[15], tv16[15], acc16[15]);
                acc16[15] = _fma_319;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4) * 64 + ((warp_1 + 1) * 16 + 4 ^ ((warp_1 + 1) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 1) * 64 + ((warp_1 + 1) * 16 + 4 ^ ((warp_1 + 1) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[8])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 2) * 64 + ((warp_1 + 1) * 16 + 4 ^ ((warp_1 + 1) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[12])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 3) * 64 + ((warp_1 + 1) * 16 + 4 ^ ((warp_1 + 1) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 80 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 100 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 120 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 140 + cg * 4) * 4)));
                float _fma_320 = __fmaf_rn(xi16[0], tv16[0], acc16[0]);
                acc16[0] = _fma_320;
                float _fma_321 = __fmaf_rn(xi16[0], tv16[1], acc16[1]);
                acc16[1] = _fma_321;
                float _fma_322 = __fmaf_rn(xi16[0], tv16[2], acc16[2]);
                acc16[2] = _fma_322;
                float _fma_323 = __fmaf_rn(xi16[0], tv16[3], acc16[3]);
                acc16[3] = _fma_323;
                float _fma_324 = __fmaf_rn(xi16[4], tv16[0], acc16[4]);
                acc16[4] = _fma_324;
                float _fma_325 = __fmaf_rn(xi16[4], tv16[1], acc16[5]);
                acc16[5] = _fma_325;
                float _fma_326 = __fmaf_rn(xi16[4], tv16[2], acc16[6]);
                acc16[6] = _fma_326;
                float _fma_327 = __fmaf_rn(xi16[4], tv16[3], acc16[7]);
                acc16[7] = _fma_327;
                float _fma_328 = __fmaf_rn(xi16[8], tv16[0], acc16[8]);
                acc16[8] = _fma_328;
                float _fma_329 = __fmaf_rn(xi16[8], tv16[1], acc16[9]);
                acc16[9] = _fma_329;
                float _fma_330 = __fmaf_rn(xi16[8], tv16[2], acc16[10]);
                acc16[10] = _fma_330;
                float _fma_331 = __fmaf_rn(xi16[8], tv16[3], acc16[11]);
                acc16[11] = _fma_331;
                float _fma_332 = __fmaf_rn(xi16[12], tv16[0], acc16[12]);
                acc16[12] = _fma_332;
                float _fma_333 = __fmaf_rn(xi16[12], tv16[1], acc16[13]);
                acc16[13] = _fma_333;
                float _fma_334 = __fmaf_rn(xi16[12], tv16[2], acc16[14]);
                acc16[14] = _fma_334;
                float _fma_335 = __fmaf_rn(xi16[12], tv16[3], acc16[15]);
                acc16[15] = _fma_335;
                float _fma_336 = __fmaf_rn(xi16[1], tv16[4], acc16[0]);
                acc16[0] = _fma_336;
                float _fma_337 = __fmaf_rn(xi16[1], tv16[5], acc16[1]);
                acc16[1] = _fma_337;
                float _fma_338 = __fmaf_rn(xi16[1], tv16[6], acc16[2]);
                acc16[2] = _fma_338;
                float _fma_339 = __fmaf_rn(xi16[1], tv16[7], acc16[3]);
                acc16[3] = _fma_339;
                float _fma_340 = __fmaf_rn(xi16[5], tv16[4], acc16[4]);
                acc16[4] = _fma_340;
                float _fma_341 = __fmaf_rn(xi16[5], tv16[5], acc16[5]);
                acc16[5] = _fma_341;
                float _fma_342 = __fmaf_rn(xi16[5], tv16[6], acc16[6]);
                acc16[6] = _fma_342;
                float _fma_343 = __fmaf_rn(xi16[5], tv16[7], acc16[7]);
                acc16[7] = _fma_343;
                float _fma_344 = __fmaf_rn(xi16[9], tv16[4], acc16[8]);
                acc16[8] = _fma_344;
                float _fma_345 = __fmaf_rn(xi16[9], tv16[5], acc16[9]);
                acc16[9] = _fma_345;
                float _fma_346 = __fmaf_rn(xi16[9], tv16[6], acc16[10]);
                acc16[10] = _fma_346;
                float _fma_347 = __fmaf_rn(xi16[9], tv16[7], acc16[11]);
                acc16[11] = _fma_347;
                float _fma_348 = __fmaf_rn(xi16[13], tv16[4], acc16[12]);
                acc16[12] = _fma_348;
                float _fma_349 = __fmaf_rn(xi16[13], tv16[5], acc16[13]);
                acc16[13] = _fma_349;
                float _fma_350 = __fmaf_rn(xi16[13], tv16[6], acc16[14]);
                acc16[14] = _fma_350;
                float _fma_351 = __fmaf_rn(xi16[13], tv16[7], acc16[15]);
                acc16[15] = _fma_351;
                float _fma_352 = __fmaf_rn(xi16[2], tv16[8], acc16[0]);
                acc16[0] = _fma_352;
                float _fma_353 = __fmaf_rn(xi16[2], tv16[9], acc16[1]);
                acc16[1] = _fma_353;
                float _fma_354 = __fmaf_rn(xi16[2], tv16[10], acc16[2]);
                acc16[2] = _fma_354;
                float _fma_355 = __fmaf_rn(xi16[2], tv16[11], acc16[3]);
                acc16[3] = _fma_355;
                float _fma_356 = __fmaf_rn(xi16[6], tv16[8], acc16[4]);
                acc16[4] = _fma_356;
                float _fma_357 = __fmaf_rn(xi16[6], tv16[9], acc16[5]);
                acc16[5] = _fma_357;
                float _fma_358 = __fmaf_rn(xi16[6], tv16[10], acc16[6]);
                acc16[6] = _fma_358;
                float _fma_359 = __fmaf_rn(xi16[6], tv16[11], acc16[7]);
                acc16[7] = _fma_359;
                float _fma_360 = __fmaf_rn(xi16[10], tv16[8], acc16[8]);
                acc16[8] = _fma_360;
                float _fma_361 = __fmaf_rn(xi16[10], tv16[9], acc16[9]);
                acc16[9] = _fma_361;
                float _fma_362 = __fmaf_rn(xi16[10], tv16[10], acc16[10]);
                acc16[10] = _fma_362;
                float _fma_363 = __fmaf_rn(xi16[10], tv16[11], acc16[11]);
                acc16[11] = _fma_363;
                float _fma_364 = __fmaf_rn(xi16[14], tv16[8], acc16[12]);
                acc16[12] = _fma_364;
                float _fma_365 = __fmaf_rn(xi16[14], tv16[9], acc16[13]);
                acc16[13] = _fma_365;
                float _fma_366 = __fmaf_rn(xi16[14], tv16[10], acc16[14]);
                acc16[14] = _fma_366;
                float _fma_367 = __fmaf_rn(xi16[14], tv16[11], acc16[15]);
                acc16[15] = _fma_367;
                float _fma_368 = __fmaf_rn(xi16[3], tv16[12], acc16[0]);
                acc16[0] = _fma_368;
                float _fma_369 = __fmaf_rn(xi16[3], tv16[13], acc16[1]);
                acc16[1] = _fma_369;
                float _fma_370 = __fmaf_rn(xi16[3], tv16[14], acc16[2]);
                acc16[2] = _fma_370;
                float _fma_371 = __fmaf_rn(xi16[3], tv16[15], acc16[3]);
                acc16[3] = _fma_371;
                float _fma_372 = __fmaf_rn(xi16[7], tv16[12], acc16[4]);
                acc16[4] = _fma_372;
                float _fma_373 = __fmaf_rn(xi16[7], tv16[13], acc16[5]);
                acc16[5] = _fma_373;
                float _fma_374 = __fmaf_rn(xi16[7], tv16[14], acc16[6]);
                acc16[6] = _fma_374;
                float _fma_375 = __fmaf_rn(xi16[7], tv16[15], acc16[7]);
                acc16[7] = _fma_375;
                float _fma_376 = __fmaf_rn(xi16[11], tv16[12], acc16[8]);
                acc16[8] = _fma_376;
                float _fma_377 = __fmaf_rn(xi16[11], tv16[13], acc16[9]);
                acc16[9] = _fma_377;
                float _fma_378 = __fmaf_rn(xi16[11], tv16[14], acc16[10]);
                acc16[10] = _fma_378;
                float _fma_379 = __fmaf_rn(xi16[11], tv16[15], acc16[11]);
                acc16[11] = _fma_379;
                float _fma_380 = __fmaf_rn(xi16[15], tv16[12], acc16[12]);
                acc16[12] = _fma_380;
                float _fma_381 = __fmaf_rn(xi16[15], tv16[13], acc16[13]);
                acc16[13] = _fma_381;
                float _fma_382 = __fmaf_rn(xi16[15], tv16[14], acc16[14]);
                acc16[14] = _fma_382;
                float _fma_383 = __fmaf_rn(xi16[15], tv16[15], acc16[15]);
                acc16[15] = _fma_383;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4) * 64 + ((warp_1 + 1) * 16 + 8 ^ ((warp_1 + 1) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 1) * 64 + ((warp_1 + 1) * 16 + 8 ^ ((warp_1 + 1) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[8])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 2) * 64 + ((warp_1 + 1) * 16 + 8 ^ ((warp_1 + 1) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[12])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 3) * 64 + ((warp_1 + 1) * 16 + 8 ^ ((warp_1 + 1) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 160 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 180 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 200 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 220 + cg * 4) * 4)));
                float _fma_384 = __fmaf_rn(xi16[0], tv16[0], acc16[0]);
                acc16[0] = _fma_384;
                float _fma_385 = __fmaf_rn(xi16[0], tv16[1], acc16[1]);
                acc16[1] = _fma_385;
                float _fma_386 = __fmaf_rn(xi16[0], tv16[2], acc16[2]);
                acc16[2] = _fma_386;
                float _fma_387 = __fmaf_rn(xi16[0], tv16[3], acc16[3]);
                acc16[3] = _fma_387;
                float _fma_388 = __fmaf_rn(xi16[4], tv16[0], acc16[4]);
                acc16[4] = _fma_388;
                float _fma_389 = __fmaf_rn(xi16[4], tv16[1], acc16[5]);
                acc16[5] = _fma_389;
                float _fma_390 = __fmaf_rn(xi16[4], tv16[2], acc16[6]);
                acc16[6] = _fma_390;
                float _fma_391 = __fmaf_rn(xi16[4], tv16[3], acc16[7]);
                acc16[7] = _fma_391;
                float _fma_392 = __fmaf_rn(xi16[8], tv16[0], acc16[8]);
                acc16[8] = _fma_392;
                float _fma_393 = __fmaf_rn(xi16[8], tv16[1], acc16[9]);
                acc16[9] = _fma_393;
                float _fma_394 = __fmaf_rn(xi16[8], tv16[2], acc16[10]);
                acc16[10] = _fma_394;
                float _fma_395 = __fmaf_rn(xi16[8], tv16[3], acc16[11]);
                acc16[11] = _fma_395;
                float _fma_396 = __fmaf_rn(xi16[12], tv16[0], acc16[12]);
                acc16[12] = _fma_396;
                float _fma_397 = __fmaf_rn(xi16[12], tv16[1], acc16[13]);
                acc16[13] = _fma_397;
                float _fma_398 = __fmaf_rn(xi16[12], tv16[2], acc16[14]);
                acc16[14] = _fma_398;
                float _fma_399 = __fmaf_rn(xi16[12], tv16[3], acc16[15]);
                acc16[15] = _fma_399;
                float _fma_400 = __fmaf_rn(xi16[1], tv16[4], acc16[0]);
                acc16[0] = _fma_400;
                float _fma_401 = __fmaf_rn(xi16[1], tv16[5], acc16[1]);
                acc16[1] = _fma_401;
                float _fma_402 = __fmaf_rn(xi16[1], tv16[6], acc16[2]);
                acc16[2] = _fma_402;
                float _fma_403 = __fmaf_rn(xi16[1], tv16[7], acc16[3]);
                acc16[3] = _fma_403;
                float _fma_404 = __fmaf_rn(xi16[5], tv16[4], acc16[4]);
                acc16[4] = _fma_404;
                float _fma_405 = __fmaf_rn(xi16[5], tv16[5], acc16[5]);
                acc16[5] = _fma_405;
                float _fma_406 = __fmaf_rn(xi16[5], tv16[6], acc16[6]);
                acc16[6] = _fma_406;
                float _fma_407 = __fmaf_rn(xi16[5], tv16[7], acc16[7]);
                acc16[7] = _fma_407;
                float _fma_408 = __fmaf_rn(xi16[9], tv16[4], acc16[8]);
                acc16[8] = _fma_408;
                float _fma_409 = __fmaf_rn(xi16[9], tv16[5], acc16[9]);
                acc16[9] = _fma_409;
                float _fma_410 = __fmaf_rn(xi16[9], tv16[6], acc16[10]);
                acc16[10] = _fma_410;
                float _fma_411 = __fmaf_rn(xi16[9], tv16[7], acc16[11]);
                acc16[11] = _fma_411;
                float _fma_412 = __fmaf_rn(xi16[13], tv16[4], acc16[12]);
                acc16[12] = _fma_412;
                float _fma_413 = __fmaf_rn(xi16[13], tv16[5], acc16[13]);
                acc16[13] = _fma_413;
                float _fma_414 = __fmaf_rn(xi16[13], tv16[6], acc16[14]);
                acc16[14] = _fma_414;
                float _fma_415 = __fmaf_rn(xi16[13], tv16[7], acc16[15]);
                acc16[15] = _fma_415;
                float _fma_416 = __fmaf_rn(xi16[2], tv16[8], acc16[0]);
                acc16[0] = _fma_416;
                float _fma_417 = __fmaf_rn(xi16[2], tv16[9], acc16[1]);
                acc16[1] = _fma_417;
                float _fma_418 = __fmaf_rn(xi16[2], tv16[10], acc16[2]);
                acc16[2] = _fma_418;
                float _fma_419 = __fmaf_rn(xi16[2], tv16[11], acc16[3]);
                acc16[3] = _fma_419;
                float _fma_420 = __fmaf_rn(xi16[6], tv16[8], acc16[4]);
                acc16[4] = _fma_420;
                float _fma_421 = __fmaf_rn(xi16[6], tv16[9], acc16[5]);
                acc16[5] = _fma_421;
                float _fma_422 = __fmaf_rn(xi16[6], tv16[10], acc16[6]);
                acc16[6] = _fma_422;
                float _fma_423 = __fmaf_rn(xi16[6], tv16[11], acc16[7]);
                acc16[7] = _fma_423;
                float _fma_424 = __fmaf_rn(xi16[10], tv16[8], acc16[8]);
                acc16[8] = _fma_424;
                float _fma_425 = __fmaf_rn(xi16[10], tv16[9], acc16[9]);
                acc16[9] = _fma_425;
                float _fma_426 = __fmaf_rn(xi16[10], tv16[10], acc16[10]);
                acc16[10] = _fma_426;
                float _fma_427 = __fmaf_rn(xi16[10], tv16[11], acc16[11]);
                acc16[11] = _fma_427;
                float _fma_428 = __fmaf_rn(xi16[14], tv16[8], acc16[12]);
                acc16[12] = _fma_428;
                float _fma_429 = __fmaf_rn(xi16[14], tv16[9], acc16[13]);
                acc16[13] = _fma_429;
                float _fma_430 = __fmaf_rn(xi16[14], tv16[10], acc16[14]);
                acc16[14] = _fma_430;
                float _fma_431 = __fmaf_rn(xi16[14], tv16[11], acc16[15]);
                acc16[15] = _fma_431;
                float _fma_432 = __fmaf_rn(xi16[3], tv16[12], acc16[0]);
                acc16[0] = _fma_432;
                float _fma_433 = __fmaf_rn(xi16[3], tv16[13], acc16[1]);
                acc16[1] = _fma_433;
                float _fma_434 = __fmaf_rn(xi16[3], tv16[14], acc16[2]);
                acc16[2] = _fma_434;
                float _fma_435 = __fmaf_rn(xi16[3], tv16[15], acc16[3]);
                acc16[3] = _fma_435;
                float _fma_436 = __fmaf_rn(xi16[7], tv16[12], acc16[4]);
                acc16[4] = _fma_436;
                float _fma_437 = __fmaf_rn(xi16[7], tv16[13], acc16[5]);
                acc16[5] = _fma_437;
                float _fma_438 = __fmaf_rn(xi16[7], tv16[14], acc16[6]);
                acc16[6] = _fma_438;
                float _fma_439 = __fmaf_rn(xi16[7], tv16[15], acc16[7]);
                acc16[7] = _fma_439;
                float _fma_440 = __fmaf_rn(xi16[11], tv16[12], acc16[8]);
                acc16[8] = _fma_440;
                float _fma_441 = __fmaf_rn(xi16[11], tv16[13], acc16[9]);
                acc16[9] = _fma_441;
                float _fma_442 = __fmaf_rn(xi16[11], tv16[14], acc16[10]);
                acc16[10] = _fma_442;
                float _fma_443 = __fmaf_rn(xi16[11], tv16[15], acc16[11]);
                acc16[11] = _fma_443;
                float _fma_444 = __fmaf_rn(xi16[15], tv16[12], acc16[12]);
                acc16[12] = _fma_444;
                float _fma_445 = __fmaf_rn(xi16[15], tv16[13], acc16[13]);
                acc16[13] = _fma_445;
                float _fma_446 = __fmaf_rn(xi16[15], tv16[14], acc16[14]);
                acc16[14] = _fma_446;
                float _fma_447 = __fmaf_rn(xi16[15], tv16[15], acc16[15]);
                acc16[15] = _fma_447;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4) * 64 + ((warp_1 + 1) * 16 + 12 ^ ((warp_1 + 1) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 1) * 64 + ((warp_1 + 1) * 16 + 12 ^ ((warp_1 + 1) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[8])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 2) * 64 + ((warp_1 + 1) * 16 + 12 ^ ((warp_1 + 1) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[12])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 3) * 64 + ((warp_1 + 1) * 16 + 12 ^ ((warp_1 + 1) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 240 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 260 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 280 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 300 + cg * 4) * 4)));
                float _fma_448 = __fmaf_rn(xi16[0], tv16[0], acc16[0]);
                acc16[0] = _fma_448;
                float _fma_449 = __fmaf_rn(xi16[0], tv16[1], acc16[1]);
                acc16[1] = _fma_449;
                float _fma_450 = __fmaf_rn(xi16[0], tv16[2], acc16[2]);
                acc16[2] = _fma_450;
                float _fma_451 = __fmaf_rn(xi16[0], tv16[3], acc16[3]);
                acc16[3] = _fma_451;
                float _fma_452 = __fmaf_rn(xi16[4], tv16[0], acc16[4]);
                acc16[4] = _fma_452;
                float _fma_453 = __fmaf_rn(xi16[4], tv16[1], acc16[5]);
                acc16[5] = _fma_453;
                float _fma_454 = __fmaf_rn(xi16[4], tv16[2], acc16[6]);
                acc16[6] = _fma_454;
                float _fma_455 = __fmaf_rn(xi16[4], tv16[3], acc16[7]);
                acc16[7] = _fma_455;
                float _fma_456 = __fmaf_rn(xi16[8], tv16[0], acc16[8]);
                acc16[8] = _fma_456;
                float _fma_457 = __fmaf_rn(xi16[8], tv16[1], acc16[9]);
                acc16[9] = _fma_457;
                float _fma_458 = __fmaf_rn(xi16[8], tv16[2], acc16[10]);
                acc16[10] = _fma_458;
                float _fma_459 = __fmaf_rn(xi16[8], tv16[3], acc16[11]);
                acc16[11] = _fma_459;
                float _fma_460 = __fmaf_rn(xi16[12], tv16[0], acc16[12]);
                acc16[12] = _fma_460;
                float _fma_461 = __fmaf_rn(xi16[12], tv16[1], acc16[13]);
                acc16[13] = _fma_461;
                float _fma_462 = __fmaf_rn(xi16[12], tv16[2], acc16[14]);
                acc16[14] = _fma_462;
                float _fma_463 = __fmaf_rn(xi16[12], tv16[3], acc16[15]);
                acc16[15] = _fma_463;
                float _fma_464 = __fmaf_rn(xi16[1], tv16[4], acc16[0]);
                acc16[0] = _fma_464;
                float _fma_465 = __fmaf_rn(xi16[1], tv16[5], acc16[1]);
                acc16[1] = _fma_465;
                float _fma_466 = __fmaf_rn(xi16[1], tv16[6], acc16[2]);
                acc16[2] = _fma_466;
                float _fma_467 = __fmaf_rn(xi16[1], tv16[7], acc16[3]);
                acc16[3] = _fma_467;
                float _fma_468 = __fmaf_rn(xi16[5], tv16[4], acc16[4]);
                acc16[4] = _fma_468;
                float _fma_469 = __fmaf_rn(xi16[5], tv16[5], acc16[5]);
                acc16[5] = _fma_469;
                float _fma_470 = __fmaf_rn(xi16[5], tv16[6], acc16[6]);
                acc16[6] = _fma_470;
                float _fma_471 = __fmaf_rn(xi16[5], tv16[7], acc16[7]);
                acc16[7] = _fma_471;
                float _fma_472 = __fmaf_rn(xi16[9], tv16[4], acc16[8]);
                acc16[8] = _fma_472;
                float _fma_473 = __fmaf_rn(xi16[9], tv16[5], acc16[9]);
                acc16[9] = _fma_473;
                float _fma_474 = __fmaf_rn(xi16[9], tv16[6], acc16[10]);
                acc16[10] = _fma_474;
                float _fma_475 = __fmaf_rn(xi16[9], tv16[7], acc16[11]);
                acc16[11] = _fma_475;
                float _fma_476 = __fmaf_rn(xi16[13], tv16[4], acc16[12]);
                acc16[12] = _fma_476;
                float _fma_477 = __fmaf_rn(xi16[13], tv16[5], acc16[13]);
                acc16[13] = _fma_477;
                float _fma_478 = __fmaf_rn(xi16[13], tv16[6], acc16[14]);
                acc16[14] = _fma_478;
                float _fma_479 = __fmaf_rn(xi16[13], tv16[7], acc16[15]);
                acc16[15] = _fma_479;
                float _fma_480 = __fmaf_rn(xi16[2], tv16[8], acc16[0]);
                acc16[0] = _fma_480;
                float _fma_481 = __fmaf_rn(xi16[2], tv16[9], acc16[1]);
                acc16[1] = _fma_481;
                float _fma_482 = __fmaf_rn(xi16[2], tv16[10], acc16[2]);
                acc16[2] = _fma_482;
                float _fma_483 = __fmaf_rn(xi16[2], tv16[11], acc16[3]);
                acc16[3] = _fma_483;
                float _fma_484 = __fmaf_rn(xi16[6], tv16[8], acc16[4]);
                acc16[4] = _fma_484;
                float _fma_485 = __fmaf_rn(xi16[6], tv16[9], acc16[5]);
                acc16[5] = _fma_485;
                float _fma_486 = __fmaf_rn(xi16[6], tv16[10], acc16[6]);
                acc16[6] = _fma_486;
                float _fma_487 = __fmaf_rn(xi16[6], tv16[11], acc16[7]);
                acc16[7] = _fma_487;
                float _fma_488 = __fmaf_rn(xi16[10], tv16[8], acc16[8]);
                acc16[8] = _fma_488;
                float _fma_489 = __fmaf_rn(xi16[10], tv16[9], acc16[9]);
                acc16[9] = _fma_489;
                float _fma_490 = __fmaf_rn(xi16[10], tv16[10], acc16[10]);
                acc16[10] = _fma_490;
                float _fma_491 = __fmaf_rn(xi16[10], tv16[11], acc16[11]);
                acc16[11] = _fma_491;
                float _fma_492 = __fmaf_rn(xi16[14], tv16[8], acc16[12]);
                acc16[12] = _fma_492;
                float _fma_493 = __fmaf_rn(xi16[14], tv16[9], acc16[13]);
                acc16[13] = _fma_493;
                float _fma_494 = __fmaf_rn(xi16[14], tv16[10], acc16[14]);
                acc16[14] = _fma_494;
                float _fma_495 = __fmaf_rn(xi16[14], tv16[11], acc16[15]);
                acc16[15] = _fma_495;
                float _fma_496 = __fmaf_rn(xi16[3], tv16[12], acc16[0]);
                acc16[0] = _fma_496;
                float _fma_497 = __fmaf_rn(xi16[3], tv16[13], acc16[1]);
                acc16[1] = _fma_497;
                float _fma_498 = __fmaf_rn(xi16[3], tv16[14], acc16[2]);
                acc16[2] = _fma_498;
                float _fma_499 = __fmaf_rn(xi16[3], tv16[15], acc16[3]);
                acc16[3] = _fma_499;
                float _fma_500 = __fmaf_rn(xi16[7], tv16[12], acc16[4]);
                acc16[4] = _fma_500;
                float _fma_501 = __fmaf_rn(xi16[7], tv16[13], acc16[5]);
                acc16[5] = _fma_501;
                float _fma_502 = __fmaf_rn(xi16[7], tv16[14], acc16[6]);
                acc16[6] = _fma_502;
                float _fma_503 = __fmaf_rn(xi16[7], tv16[15], acc16[7]);
                acc16[7] = _fma_503;
                float _fma_504 = __fmaf_rn(xi16[11], tv16[12], acc16[8]);
                acc16[8] = _fma_504;
                float _fma_505 = __fmaf_rn(xi16[11], tv16[13], acc16[9]);
                acc16[9] = _fma_505;
                float _fma_506 = __fmaf_rn(xi16[11], tv16[14], acc16[10]);
                acc16[10] = _fma_506;
                float _fma_507 = __fmaf_rn(xi16[11], tv16[15], acc16[11]);
                acc16[11] = _fma_507;
                float _fma_508 = __fmaf_rn(xi16[15], tv16[12], acc16[12]);
                acc16[12] = _fma_508;
                float _fma_509 = __fmaf_rn(xi16[15], tv16[13], acc16[13]);
                acc16[13] = _fma_509;
                float _fma_510 = __fmaf_rn(xi16[15], tv16[14], acc16[14]);
                acc16[14] = _fma_510;
                float _fma_511 = __fmaf_rn(xi16[15], tv16[15], acc16[15]);
                acc16[15] = _fma_511;
                acc16[0] = -acc16[0];
                acc16[1] = -acc16[1];
                acc16[2] = -acc16[2];
                acc16[3] = -acc16[3];
                acc16[4] = -acc16[4];
                acc16[5] = -acc16[5];
                acc16[6] = -acc16[6];
                acc16[7] = -acc16[7];
                acc16[8] = -acc16[8];
                acc16[9] = -acc16[9];
                acc16[10] = -acc16[10];
                acc16[11] = -acc16[11];
                acc16[12] = -acc16[12];
                acc16[13] = -acc16[13];
                acc16[14] = -acc16[14];
                acc16[15] = -acc16[15];
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[0])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 1) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[4])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 2) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[8])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + rg * 4 + 3) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[12])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 3])));
            }
        }
        __syncthreads();
        if (lane_0 < 16) {
            if (warp_1 < 2) {
                acc16[0] = 0.0f;
                acc16[1] = 0.0f;
                acc16[2] = 0.0f;
                acc16[3] = 0.0f;
                acc16[4] = 0.0f;
                acc16[5] = 0.0f;
                acc16[6] = 0.0f;
                acc16[7] = 0.0f;
                acc16[8] = 0.0f;
                acc16[9] = 0.0f;
                acc16[10] = 0.0f;
                acc16[11] = 0.0f;
                acc16[12] = 0.0f;
                acc16[13] = 0.0f;
                acc16[14] = 0.0f;
                acc16[15] = 0.0f;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((warp_1 * 16 * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((warp_1 * 16 * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 >> 2 & 1) << 4)) * 4)));
                float _fma_512 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_512;
                float _fma_513 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_513;
                float _fma_514 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_514;
                float _fma_515 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_515;
                float _fma_516 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_516;
                float _fma_517 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_517;
                float _fma_518 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_518;
                float _fma_519 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_519;
                float _fma_520 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_520;
                float _fma_521 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_521;
                float _fma_522 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_522;
                float _fma_523 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_523;
                float _fma_524 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_524;
                float _fma_525 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_525;
                float _fma_526 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_526;
                float _fma_527 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_527;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 + 1) * 16 * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 + 1) * 16 * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 >> 2 & 1) << 4)) * 4)));
                float _fma_528 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_528;
                float _fma_529 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_529;
                float _fma_530 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_530;
                float _fma_531 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_531;
                float _fma_532 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_532;
                float _fma_533 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_533;
                float _fma_534 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_534;
                float _fma_535 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_535;
                float _fma_536 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_536;
                float _fma_537 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_537;
                float _fma_538 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_538;
                float _fma_539 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_539;
                float _fma_540 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_540;
                float _fma_541 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_541;
                float _fma_542 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_542;
                float _fma_543 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_543;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 1 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 1 >> 2 & 1) << 4)) * 4)));
                float _fma_544 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_544;
                float _fma_545 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_545;
                float _fma_546 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_546;
                float _fma_547 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_547;
                float _fma_548 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_548;
                float _fma_549 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_549;
                float _fma_550 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_550;
                float _fma_551 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_551;
                float _fma_552 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_552;
                float _fma_553 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_553;
                float _fma_554 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_554;
                float _fma_555 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_555;
                float _fma_556 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_556;
                float _fma_557 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_557;
                float _fma_558 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_558;
                float _fma_559 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_559;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 1) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 1 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 1) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 1 >> 2 & 1) << 4)) * 4)));
                float _fma_560 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_560;
                float _fma_561 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_561;
                float _fma_562 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_562;
                float _fma_563 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_563;
                float _fma_564 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_564;
                float _fma_565 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_565;
                float _fma_566 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_566;
                float _fma_567 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_567;
                float _fma_568 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_568;
                float _fma_569 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_569;
                float _fma_570 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_570;
                float _fma_571 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_571;
                float _fma_572 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_572;
                float _fma_573 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_573;
                float _fma_574 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_574;
                float _fma_575 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_575;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 2 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 2 >> 2 & 1) << 4)) * 4)));
                float _fma_576 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_576;
                float _fma_577 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_577;
                float _fma_578 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_578;
                float _fma_579 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_579;
                float _fma_580 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_580;
                float _fma_581 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_581;
                float _fma_582 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_582;
                float _fma_583 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_583;
                float _fma_584 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_584;
                float _fma_585 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_585;
                float _fma_586 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_586;
                float _fma_587 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_587;
                float _fma_588 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_588;
                float _fma_589 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_589;
                float _fma_590 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_590;
                float _fma_591 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_591;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 2) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 2 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 2) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 2 >> 2 & 1) << 4)) * 4)));
                float _fma_592 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_592;
                float _fma_593 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_593;
                float _fma_594 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_594;
                float _fma_595 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_595;
                float _fma_596 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_596;
                float _fma_597 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_597;
                float _fma_598 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_598;
                float _fma_599 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_599;
                float _fma_600 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_600;
                float _fma_601 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_601;
                float _fma_602 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_602;
                float _fma_603 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_603;
                float _fma_604 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_604;
                float _fma_605 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_605;
                float _fma_606 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_606;
                float _fma_607 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_607;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 3 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 3 >> 2 & 1) << 4)) * 4)));
                float _fma_608 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_608;
                float _fma_609 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_609;
                float _fma_610 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_610;
                float _fma_611 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_611;
                float _fma_612 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_612;
                float _fma_613 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_613;
                float _fma_614 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_614;
                float _fma_615 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_615;
                float _fma_616 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_616;
                float _fma_617 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_617;
                float _fma_618 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_618;
                float _fma_619 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_619;
                float _fma_620 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_620;
                float _fma_621 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_621;
                float _fma_622 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_622;
                float _fma_623 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_623;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 3) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 3 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 3) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 3 >> 2 & 1) << 4)) * 4)));
                float _fma_624 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_624;
                float _fma_625 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_625;
                float _fma_626 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_626;
                float _fma_627 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_627;
                float _fma_628 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_628;
                float _fma_629 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_629;
                float _fma_630 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_630;
                float _fma_631 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_631;
                float _fma_632 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_632;
                float _fma_633 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_633;
                float _fma_634 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_634;
                float _fma_635 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_635;
                float _fma_636 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_636;
                float _fma_637 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_637;
                float _fma_638 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_638;
                float _fma_639 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_639;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 4 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 4 >> 2 & 1) << 4)) * 4)));
                float _fma_640 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_640;
                float _fma_641 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_641;
                float _fma_642 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_642;
                float _fma_643 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_643;
                float _fma_644 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_644;
                float _fma_645 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_645;
                float _fma_646 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_646;
                float _fma_647 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_647;
                float _fma_648 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_648;
                float _fma_649 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_649;
                float _fma_650 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_650;
                float _fma_651 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_651;
                float _fma_652 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_652;
                float _fma_653 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_653;
                float _fma_654 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_654;
                float _fma_655 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_655;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 4) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 4 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 4) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 4 >> 2 & 1) << 4)) * 4)));
                float _fma_656 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_656;
                float _fma_657 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_657;
                float _fma_658 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_658;
                float _fma_659 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_659;
                float _fma_660 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_660;
                float _fma_661 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_661;
                float _fma_662 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_662;
                float _fma_663 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_663;
                float _fma_664 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_664;
                float _fma_665 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_665;
                float _fma_666 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_666;
                float _fma_667 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_667;
                float _fma_668 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_668;
                float _fma_669 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_669;
                float _fma_670 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_670;
                float _fma_671 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_671;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 5 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 5 >> 2 & 1) << 4)) * 4)));
                float _fma_672 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_672;
                float _fma_673 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_673;
                float _fma_674 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_674;
                float _fma_675 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_675;
                float _fma_676 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_676;
                float _fma_677 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_677;
                float _fma_678 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_678;
                float _fma_679 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_679;
                float _fma_680 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_680;
                float _fma_681 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_681;
                float _fma_682 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_682;
                float _fma_683 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_683;
                float _fma_684 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_684;
                float _fma_685 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_685;
                float _fma_686 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_686;
                float _fma_687 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_687;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 5) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 5 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 5) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 5 >> 2 & 1) << 4)) * 4)));
                float _fma_688 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_688;
                float _fma_689 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_689;
                float _fma_690 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_690;
                float _fma_691 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_691;
                float _fma_692 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_692;
                float _fma_693 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_693;
                float _fma_694 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_694;
                float _fma_695 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_695;
                float _fma_696 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_696;
                float _fma_697 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_697;
                float _fma_698 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_698;
                float _fma_699 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_699;
                float _fma_700 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_700;
                float _fma_701 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_701;
                float _fma_702 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_702;
                float _fma_703 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_703;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 6 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 6 >> 2 & 1) << 4)) * 4)));
                float _fma_704 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_704;
                float _fma_705 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_705;
                float _fma_706 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_706;
                float _fma_707 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_707;
                float _fma_708 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_708;
                float _fma_709 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_709;
                float _fma_710 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_710;
                float _fma_711 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_711;
                float _fma_712 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_712;
                float _fma_713 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_713;
                float _fma_714 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_714;
                float _fma_715 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_715;
                float _fma_716 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_716;
                float _fma_717 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_717;
                float _fma_718 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_718;
                float _fma_719 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_719;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 6) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 6 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 6) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 6 >> 2 & 1) << 4)) * 4)));
                float _fma_720 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_720;
                float _fma_721 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_721;
                float _fma_722 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_722;
                float _fma_723 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_723;
                float _fma_724 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_724;
                float _fma_725 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_725;
                float _fma_726 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_726;
                float _fma_727 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_727;
                float _fma_728 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_728;
                float _fma_729 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_729;
                float _fma_730 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_730;
                float _fma_731 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_731;
                float _fma_732 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_732;
                float _fma_733 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_733;
                float _fma_734 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_734;
                float _fma_735 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_735;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 7 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 7 >> 2 & 1) << 4)) * 4)));
                float _fma_736 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_736;
                float _fma_737 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_737;
                float _fma_738 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_738;
                float _fma_739 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_739;
                float _fma_740 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_740;
                float _fma_741 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_741;
                float _fma_742 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_742;
                float _fma_743 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_743;
                float _fma_744 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_744;
                float _fma_745 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_745;
                float _fma_746 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_746;
                float _fma_747 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_747;
                float _fma_748 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_748;
                float _fma_749 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_749;
                float _fma_750 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_750;
                float _fma_751 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_751;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 7) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 7 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 7) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 7 >> 2 & 1) << 4)) * 4)));
                float _fma_752 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_752;
                float _fma_753 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_753;
                float _fma_754 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_754;
                float _fma_755 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_755;
                float _fma_756 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_756;
                float _fma_757 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_757;
                float _fma_758 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_758;
                float _fma_759 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_759;
                float _fma_760 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_760;
                float _fma_761 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_761;
                float _fma_762 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_762;
                float _fma_763 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_763;
                float _fma_764 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_764;
                float _fma_765 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_765;
                float _fma_766 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_766;
                float _fma_767 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_767;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 8 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 8 >> 2 & 1) << 4)) * 4)));
                float _fma_768 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_768;
                float _fma_769 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_769;
                float _fma_770 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_770;
                float _fma_771 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_771;
                float _fma_772 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_772;
                float _fma_773 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_773;
                float _fma_774 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_774;
                float _fma_775 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_775;
                float _fma_776 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_776;
                float _fma_777 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_777;
                float _fma_778 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_778;
                float _fma_779 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_779;
                float _fma_780 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_780;
                float _fma_781 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_781;
                float _fma_782 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_782;
                float _fma_783 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_783;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 8) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 8 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 8) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 8 >> 2 & 1) << 4)) * 4)));
                float _fma_784 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_784;
                float _fma_785 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_785;
                float _fma_786 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_786;
                float _fma_787 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_787;
                float _fma_788 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_788;
                float _fma_789 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_789;
                float _fma_790 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_790;
                float _fma_791 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_791;
                float _fma_792 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_792;
                float _fma_793 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_793;
                float _fma_794 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_794;
                float _fma_795 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_795;
                float _fma_796 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_796;
                float _fma_797 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_797;
                float _fma_798 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_798;
                float _fma_799 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_799;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 9 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 9 >> 2 & 1) << 4)) * 4)));
                float _fma_800 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_800;
                float _fma_801 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_801;
                float _fma_802 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_802;
                float _fma_803 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_803;
                float _fma_804 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_804;
                float _fma_805 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_805;
                float _fma_806 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_806;
                float _fma_807 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_807;
                float _fma_808 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_808;
                float _fma_809 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_809;
                float _fma_810 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_810;
                float _fma_811 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_811;
                float _fma_812 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_812;
                float _fma_813 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_813;
                float _fma_814 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_814;
                float _fma_815 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_815;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 9) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 9 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 9) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 9 >> 2 & 1) << 4)) * 4)));
                float _fma_816 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_816;
                float _fma_817 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_817;
                float _fma_818 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_818;
                float _fma_819 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_819;
                float _fma_820 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_820;
                float _fma_821 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_821;
                float _fma_822 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_822;
                float _fma_823 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_823;
                float _fma_824 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_824;
                float _fma_825 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_825;
                float _fma_826 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_826;
                float _fma_827 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_827;
                float _fma_828 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_828;
                float _fma_829 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_829;
                float _fma_830 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_830;
                float _fma_831 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_831;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 10 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 10 >> 2 & 1) << 4)) * 4)));
                float _fma_832 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_832;
                float _fma_833 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_833;
                float _fma_834 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_834;
                float _fma_835 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_835;
                float _fma_836 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_836;
                float _fma_837 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_837;
                float _fma_838 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_838;
                float _fma_839 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_839;
                float _fma_840 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_840;
                float _fma_841 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_841;
                float _fma_842 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_842;
                float _fma_843 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_843;
                float _fma_844 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_844;
                float _fma_845 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_845;
                float _fma_846 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_846;
                float _fma_847 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_847;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 10) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 10 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 10) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 10 >> 2 & 1) << 4)) * 4)));
                float _fma_848 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_848;
                float _fma_849 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_849;
                float _fma_850 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_850;
                float _fma_851 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_851;
                float _fma_852 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_852;
                float _fma_853 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_853;
                float _fma_854 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_854;
                float _fma_855 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_855;
                float _fma_856 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_856;
                float _fma_857 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_857;
                float _fma_858 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_858;
                float _fma_859 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_859;
                float _fma_860 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_860;
                float _fma_861 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_861;
                float _fma_862 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_862;
                float _fma_863 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_863;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 11 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 11 >> 2 & 1) << 4)) * 4)));
                float _fma_864 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_864;
                float _fma_865 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_865;
                float _fma_866 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_866;
                float _fma_867 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_867;
                float _fma_868 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_868;
                float _fma_869 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_869;
                float _fma_870 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_870;
                float _fma_871 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_871;
                float _fma_872 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_872;
                float _fma_873 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_873;
                float _fma_874 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_874;
                float _fma_875 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_875;
                float _fma_876 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_876;
                float _fma_877 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_877;
                float _fma_878 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_878;
                float _fma_879 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_879;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 11) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 11 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 11) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 11 >> 2 & 1) << 4)) * 4)));
                float _fma_880 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_880;
                float _fma_881 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_881;
                float _fma_882 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_882;
                float _fma_883 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_883;
                float _fma_884 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_884;
                float _fma_885 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_885;
                float _fma_886 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_886;
                float _fma_887 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_887;
                float _fma_888 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_888;
                float _fma_889 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_889;
                float _fma_890 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_890;
                float _fma_891 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_891;
                float _fma_892 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_892;
                float _fma_893 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_893;
                float _fma_894 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_894;
                float _fma_895 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_895;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 12 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 12 >> 2 & 1) << 4)) * 4)));
                float _fma_896 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_896;
                float _fma_897 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_897;
                float _fma_898 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_898;
                float _fma_899 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_899;
                float _fma_900 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_900;
                float _fma_901 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_901;
                float _fma_902 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_902;
                float _fma_903 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_903;
                float _fma_904 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_904;
                float _fma_905 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_905;
                float _fma_906 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_906;
                float _fma_907 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_907;
                float _fma_908 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_908;
                float _fma_909 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_909;
                float _fma_910 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_910;
                float _fma_911 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_911;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 12) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 12 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 12) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 12 >> 2 & 1) << 4)) * 4)));
                float _fma_912 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_912;
                float _fma_913 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_913;
                float _fma_914 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_914;
                float _fma_915 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_915;
                float _fma_916 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_916;
                float _fma_917 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_917;
                float _fma_918 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_918;
                float _fma_919 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_919;
                float _fma_920 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_920;
                float _fma_921 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_921;
                float _fma_922 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_922;
                float _fma_923 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_923;
                float _fma_924 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_924;
                float _fma_925 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_925;
                float _fma_926 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_926;
                float _fma_927 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_927;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 13 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 13 >> 2 & 1) << 4)) * 4)));
                float _fma_928 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_928;
                float _fma_929 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_929;
                float _fma_930 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_930;
                float _fma_931 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_931;
                float _fma_932 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_932;
                float _fma_933 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_933;
                float _fma_934 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_934;
                float _fma_935 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_935;
                float _fma_936 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_936;
                float _fma_937 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_937;
                float _fma_938 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_938;
                float _fma_939 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_939;
                float _fma_940 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_940;
                float _fma_941 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_941;
                float _fma_942 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_942;
                float _fma_943 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_943;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 13) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 13 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 13) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 13 >> 2 & 1) << 4)) * 4)));
                float _fma_944 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_944;
                float _fma_945 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_945;
                float _fma_946 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_946;
                float _fma_947 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_947;
                float _fma_948 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_948;
                float _fma_949 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_949;
                float _fma_950 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_950;
                float _fma_951 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_951;
                float _fma_952 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_952;
                float _fma_953 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_953;
                float _fma_954 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_954;
                float _fma_955 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_955;
                float _fma_956 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_956;
                float _fma_957 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_957;
                float _fma_958 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_958;
                float _fma_959 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_959;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 14 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 14 >> 2 & 1) << 4)) * 4)));
                float _fma_960 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_960;
                float _fma_961 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_961;
                float _fma_962 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_962;
                float _fma_963 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_963;
                float _fma_964 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_964;
                float _fma_965 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_965;
                float _fma_966 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_966;
                float _fma_967 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_967;
                float _fma_968 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_968;
                float _fma_969 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_969;
                float _fma_970 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_970;
                float _fma_971 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_971;
                float _fma_972 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_972;
                float _fma_973 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_973;
                float _fma_974 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_974;
                float _fma_975 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_975;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 14) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 14 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 14) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 14 >> 2 & 1) << 4)) * 4)));
                float _fma_976 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_976;
                float _fma_977 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_977;
                float _fma_978 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_978;
                float _fma_979 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_979;
                float _fma_980 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_980;
                float _fma_981 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_981;
                float _fma_982 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_982;
                float _fma_983 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_983;
                float _fma_984 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_984;
                float _fma_985 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_985;
                float _fma_986 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_986;
                float _fma_987 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_987;
                float _fma_988 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_988;
                float _fma_989 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_989;
                float _fma_990 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_990;
                float _fma_991 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_991;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ (warp_1 * 16 + 15 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 15 >> 2 & 1) << 4)) * 4)));
                float _fma_992 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_992;
                float _fma_993 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_993;
                float _fma_994 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_994;
                float _fma_995 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_995;
                float _fma_996 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_996;
                float _fma_997 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_997;
                float _fma_998 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_998;
                float _fma_999 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_999;
                float _fma_1000 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1000;
                float _fma_1001 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1001;
                float _fma_1002 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1002;
                float _fma_1003 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1003;
                float _fma_1004 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1004;
                float _fma_1005 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1005;
                float _fma_1006 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1006;
                float _fma_1007 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1007;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 15) * 64 + ((warp_1 + 2) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 15 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 15) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 15 >> 2 & 1) << 4)) * 4)));
                float _fma_1008 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1008;
                float _fma_1009 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1009;
                float _fma_1010 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1010;
                float _fma_1011 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1011;
                float _fma_1012 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1012;
                float _fma_1013 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1013;
                float _fma_1014 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1014;
                float _fma_1015 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1015;
                float _fma_1016 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1016;
                float _fma_1017 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1017;
                float _fma_1018 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1018;
                float _fma_1019 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1019;
                float _fma_1020 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1020;
                float _fma_1021 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1021;
                float _fma_1022 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1022;
                float _fma_1023 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1023;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_t_addr + (unsigned int)((warp_1 * 320 + rg * 4 * 20 + cg * 4) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[0])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (rg * 4 + 1) * 20 + cg * 4) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[4])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (rg * 4 + 2) * 20 + cg * 4) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[8])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (rg * 4 + 3) * 20 + cg * 4) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[12])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 3])));
            }
        }
        __syncthreads();
        if (lane_0 < 16) {
            if (warp_1 < 2) {
                acc16[0] = 0.0f;
                acc16[1] = 0.0f;
                acc16[2] = 0.0f;
                acc16[3] = 0.0f;
                acc16[4] = 0.0f;
                acc16[5] = 0.0f;
                acc16[6] = 0.0f;
                acc16[7] = 0.0f;
                acc16[8] = 0.0f;
                acc16[9] = 0.0f;
                acc16[10] = 0.0f;
                acc16[11] = 0.0f;
                acc16[12] = 0.0f;
                acc16[13] = 0.0f;
                acc16[14] = 0.0f;
                acc16[15] = 0.0f;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4) * 64 + ((warp_1 + 2) * 16 ^ ((warp_1 + 2) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 1) * 64 + ((warp_1 + 2) * 16 ^ ((warp_1 + 2) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[8])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 2) * 64 + ((warp_1 + 2) * 16 ^ ((warp_1 + 2) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[12])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 3) * 64 + ((warp_1 + 2) * 16 ^ ((warp_1 + 2) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 20 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 40 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 60 + cg * 4) * 4)));
                float _fma_1024 = __fmaf_rn(xi16[0], tv16[0], acc16[0]);
                acc16[0] = _fma_1024;
                float _fma_1025 = __fmaf_rn(xi16[0], tv16[1], acc16[1]);
                acc16[1] = _fma_1025;
                float _fma_1026 = __fmaf_rn(xi16[0], tv16[2], acc16[2]);
                acc16[2] = _fma_1026;
                float _fma_1027 = __fmaf_rn(xi16[0], tv16[3], acc16[3]);
                acc16[3] = _fma_1027;
                float _fma_1028 = __fmaf_rn(xi16[4], tv16[0], acc16[4]);
                acc16[4] = _fma_1028;
                float _fma_1029 = __fmaf_rn(xi16[4], tv16[1], acc16[5]);
                acc16[5] = _fma_1029;
                float _fma_1030 = __fmaf_rn(xi16[4], tv16[2], acc16[6]);
                acc16[6] = _fma_1030;
                float _fma_1031 = __fmaf_rn(xi16[4], tv16[3], acc16[7]);
                acc16[7] = _fma_1031;
                float _fma_1032 = __fmaf_rn(xi16[8], tv16[0], acc16[8]);
                acc16[8] = _fma_1032;
                float _fma_1033 = __fmaf_rn(xi16[8], tv16[1], acc16[9]);
                acc16[9] = _fma_1033;
                float _fma_1034 = __fmaf_rn(xi16[8], tv16[2], acc16[10]);
                acc16[10] = _fma_1034;
                float _fma_1035 = __fmaf_rn(xi16[8], tv16[3], acc16[11]);
                acc16[11] = _fma_1035;
                float _fma_1036 = __fmaf_rn(xi16[12], tv16[0], acc16[12]);
                acc16[12] = _fma_1036;
                float _fma_1037 = __fmaf_rn(xi16[12], tv16[1], acc16[13]);
                acc16[13] = _fma_1037;
                float _fma_1038 = __fmaf_rn(xi16[12], tv16[2], acc16[14]);
                acc16[14] = _fma_1038;
                float _fma_1039 = __fmaf_rn(xi16[12], tv16[3], acc16[15]);
                acc16[15] = _fma_1039;
                float _fma_1040 = __fmaf_rn(xi16[1], tv16[4], acc16[0]);
                acc16[0] = _fma_1040;
                float _fma_1041 = __fmaf_rn(xi16[1], tv16[5], acc16[1]);
                acc16[1] = _fma_1041;
                float _fma_1042 = __fmaf_rn(xi16[1], tv16[6], acc16[2]);
                acc16[2] = _fma_1042;
                float _fma_1043 = __fmaf_rn(xi16[1], tv16[7], acc16[3]);
                acc16[3] = _fma_1043;
                float _fma_1044 = __fmaf_rn(xi16[5], tv16[4], acc16[4]);
                acc16[4] = _fma_1044;
                float _fma_1045 = __fmaf_rn(xi16[5], tv16[5], acc16[5]);
                acc16[5] = _fma_1045;
                float _fma_1046 = __fmaf_rn(xi16[5], tv16[6], acc16[6]);
                acc16[6] = _fma_1046;
                float _fma_1047 = __fmaf_rn(xi16[5], tv16[7], acc16[7]);
                acc16[7] = _fma_1047;
                float _fma_1048 = __fmaf_rn(xi16[9], tv16[4], acc16[8]);
                acc16[8] = _fma_1048;
                float _fma_1049 = __fmaf_rn(xi16[9], tv16[5], acc16[9]);
                acc16[9] = _fma_1049;
                float _fma_1050 = __fmaf_rn(xi16[9], tv16[6], acc16[10]);
                acc16[10] = _fma_1050;
                float _fma_1051 = __fmaf_rn(xi16[9], tv16[7], acc16[11]);
                acc16[11] = _fma_1051;
                float _fma_1052 = __fmaf_rn(xi16[13], tv16[4], acc16[12]);
                acc16[12] = _fma_1052;
                float _fma_1053 = __fmaf_rn(xi16[13], tv16[5], acc16[13]);
                acc16[13] = _fma_1053;
                float _fma_1054 = __fmaf_rn(xi16[13], tv16[6], acc16[14]);
                acc16[14] = _fma_1054;
                float _fma_1055 = __fmaf_rn(xi16[13], tv16[7], acc16[15]);
                acc16[15] = _fma_1055;
                float _fma_1056 = __fmaf_rn(xi16[2], tv16[8], acc16[0]);
                acc16[0] = _fma_1056;
                float _fma_1057 = __fmaf_rn(xi16[2], tv16[9], acc16[1]);
                acc16[1] = _fma_1057;
                float _fma_1058 = __fmaf_rn(xi16[2], tv16[10], acc16[2]);
                acc16[2] = _fma_1058;
                float _fma_1059 = __fmaf_rn(xi16[2], tv16[11], acc16[3]);
                acc16[3] = _fma_1059;
                float _fma_1060 = __fmaf_rn(xi16[6], tv16[8], acc16[4]);
                acc16[4] = _fma_1060;
                float _fma_1061 = __fmaf_rn(xi16[6], tv16[9], acc16[5]);
                acc16[5] = _fma_1061;
                float _fma_1062 = __fmaf_rn(xi16[6], tv16[10], acc16[6]);
                acc16[6] = _fma_1062;
                float _fma_1063 = __fmaf_rn(xi16[6], tv16[11], acc16[7]);
                acc16[7] = _fma_1063;
                float _fma_1064 = __fmaf_rn(xi16[10], tv16[8], acc16[8]);
                acc16[8] = _fma_1064;
                float _fma_1065 = __fmaf_rn(xi16[10], tv16[9], acc16[9]);
                acc16[9] = _fma_1065;
                float _fma_1066 = __fmaf_rn(xi16[10], tv16[10], acc16[10]);
                acc16[10] = _fma_1066;
                float _fma_1067 = __fmaf_rn(xi16[10], tv16[11], acc16[11]);
                acc16[11] = _fma_1067;
                float _fma_1068 = __fmaf_rn(xi16[14], tv16[8], acc16[12]);
                acc16[12] = _fma_1068;
                float _fma_1069 = __fmaf_rn(xi16[14], tv16[9], acc16[13]);
                acc16[13] = _fma_1069;
                float _fma_1070 = __fmaf_rn(xi16[14], tv16[10], acc16[14]);
                acc16[14] = _fma_1070;
                float _fma_1071 = __fmaf_rn(xi16[14], tv16[11], acc16[15]);
                acc16[15] = _fma_1071;
                float _fma_1072 = __fmaf_rn(xi16[3], tv16[12], acc16[0]);
                acc16[0] = _fma_1072;
                float _fma_1073 = __fmaf_rn(xi16[3], tv16[13], acc16[1]);
                acc16[1] = _fma_1073;
                float _fma_1074 = __fmaf_rn(xi16[3], tv16[14], acc16[2]);
                acc16[2] = _fma_1074;
                float _fma_1075 = __fmaf_rn(xi16[3], tv16[15], acc16[3]);
                acc16[3] = _fma_1075;
                float _fma_1076 = __fmaf_rn(xi16[7], tv16[12], acc16[4]);
                acc16[4] = _fma_1076;
                float _fma_1077 = __fmaf_rn(xi16[7], tv16[13], acc16[5]);
                acc16[5] = _fma_1077;
                float _fma_1078 = __fmaf_rn(xi16[7], tv16[14], acc16[6]);
                acc16[6] = _fma_1078;
                float _fma_1079 = __fmaf_rn(xi16[7], tv16[15], acc16[7]);
                acc16[7] = _fma_1079;
                float _fma_1080 = __fmaf_rn(xi16[11], tv16[12], acc16[8]);
                acc16[8] = _fma_1080;
                float _fma_1081 = __fmaf_rn(xi16[11], tv16[13], acc16[9]);
                acc16[9] = _fma_1081;
                float _fma_1082 = __fmaf_rn(xi16[11], tv16[14], acc16[10]);
                acc16[10] = _fma_1082;
                float _fma_1083 = __fmaf_rn(xi16[11], tv16[15], acc16[11]);
                acc16[11] = _fma_1083;
                float _fma_1084 = __fmaf_rn(xi16[15], tv16[12], acc16[12]);
                acc16[12] = _fma_1084;
                float _fma_1085 = __fmaf_rn(xi16[15], tv16[13], acc16[13]);
                acc16[13] = _fma_1085;
                float _fma_1086 = __fmaf_rn(xi16[15], tv16[14], acc16[14]);
                acc16[14] = _fma_1086;
                float _fma_1087 = __fmaf_rn(xi16[15], tv16[15], acc16[15]);
                acc16[15] = _fma_1087;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4) * 64 + ((warp_1 + 2) * 16 + 4 ^ ((warp_1 + 2) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 1) * 64 + ((warp_1 + 2) * 16 + 4 ^ ((warp_1 + 2) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[8])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 2) * 64 + ((warp_1 + 2) * 16 + 4 ^ ((warp_1 + 2) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[12])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 3) * 64 + ((warp_1 + 2) * 16 + 4 ^ ((warp_1 + 2) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 80 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 100 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 120 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 140 + cg * 4) * 4)));
                float _fma_1088 = __fmaf_rn(xi16[0], tv16[0], acc16[0]);
                acc16[0] = _fma_1088;
                float _fma_1089 = __fmaf_rn(xi16[0], tv16[1], acc16[1]);
                acc16[1] = _fma_1089;
                float _fma_1090 = __fmaf_rn(xi16[0], tv16[2], acc16[2]);
                acc16[2] = _fma_1090;
                float _fma_1091 = __fmaf_rn(xi16[0], tv16[3], acc16[3]);
                acc16[3] = _fma_1091;
                float _fma_1092 = __fmaf_rn(xi16[4], tv16[0], acc16[4]);
                acc16[4] = _fma_1092;
                float _fma_1093 = __fmaf_rn(xi16[4], tv16[1], acc16[5]);
                acc16[5] = _fma_1093;
                float _fma_1094 = __fmaf_rn(xi16[4], tv16[2], acc16[6]);
                acc16[6] = _fma_1094;
                float _fma_1095 = __fmaf_rn(xi16[4], tv16[3], acc16[7]);
                acc16[7] = _fma_1095;
                float _fma_1096 = __fmaf_rn(xi16[8], tv16[0], acc16[8]);
                acc16[8] = _fma_1096;
                float _fma_1097 = __fmaf_rn(xi16[8], tv16[1], acc16[9]);
                acc16[9] = _fma_1097;
                float _fma_1098 = __fmaf_rn(xi16[8], tv16[2], acc16[10]);
                acc16[10] = _fma_1098;
                float _fma_1099 = __fmaf_rn(xi16[8], tv16[3], acc16[11]);
                acc16[11] = _fma_1099;
                float _fma_1100 = __fmaf_rn(xi16[12], tv16[0], acc16[12]);
                acc16[12] = _fma_1100;
                float _fma_1101 = __fmaf_rn(xi16[12], tv16[1], acc16[13]);
                acc16[13] = _fma_1101;
                float _fma_1102 = __fmaf_rn(xi16[12], tv16[2], acc16[14]);
                acc16[14] = _fma_1102;
                float _fma_1103 = __fmaf_rn(xi16[12], tv16[3], acc16[15]);
                acc16[15] = _fma_1103;
                float _fma_1104 = __fmaf_rn(xi16[1], tv16[4], acc16[0]);
                acc16[0] = _fma_1104;
                float _fma_1105 = __fmaf_rn(xi16[1], tv16[5], acc16[1]);
                acc16[1] = _fma_1105;
                float _fma_1106 = __fmaf_rn(xi16[1], tv16[6], acc16[2]);
                acc16[2] = _fma_1106;
                float _fma_1107 = __fmaf_rn(xi16[1], tv16[7], acc16[3]);
                acc16[3] = _fma_1107;
                float _fma_1108 = __fmaf_rn(xi16[5], tv16[4], acc16[4]);
                acc16[4] = _fma_1108;
                float _fma_1109 = __fmaf_rn(xi16[5], tv16[5], acc16[5]);
                acc16[5] = _fma_1109;
                float _fma_1110 = __fmaf_rn(xi16[5], tv16[6], acc16[6]);
                acc16[6] = _fma_1110;
                float _fma_1111 = __fmaf_rn(xi16[5], tv16[7], acc16[7]);
                acc16[7] = _fma_1111;
                float _fma_1112 = __fmaf_rn(xi16[9], tv16[4], acc16[8]);
                acc16[8] = _fma_1112;
                float _fma_1113 = __fmaf_rn(xi16[9], tv16[5], acc16[9]);
                acc16[9] = _fma_1113;
                float _fma_1114 = __fmaf_rn(xi16[9], tv16[6], acc16[10]);
                acc16[10] = _fma_1114;
                float _fma_1115 = __fmaf_rn(xi16[9], tv16[7], acc16[11]);
                acc16[11] = _fma_1115;
                float _fma_1116 = __fmaf_rn(xi16[13], tv16[4], acc16[12]);
                acc16[12] = _fma_1116;
                float _fma_1117 = __fmaf_rn(xi16[13], tv16[5], acc16[13]);
                acc16[13] = _fma_1117;
                float _fma_1118 = __fmaf_rn(xi16[13], tv16[6], acc16[14]);
                acc16[14] = _fma_1118;
                float _fma_1119 = __fmaf_rn(xi16[13], tv16[7], acc16[15]);
                acc16[15] = _fma_1119;
                float _fma_1120 = __fmaf_rn(xi16[2], tv16[8], acc16[0]);
                acc16[0] = _fma_1120;
                float _fma_1121 = __fmaf_rn(xi16[2], tv16[9], acc16[1]);
                acc16[1] = _fma_1121;
                float _fma_1122 = __fmaf_rn(xi16[2], tv16[10], acc16[2]);
                acc16[2] = _fma_1122;
                float _fma_1123 = __fmaf_rn(xi16[2], tv16[11], acc16[3]);
                acc16[3] = _fma_1123;
                float _fma_1124 = __fmaf_rn(xi16[6], tv16[8], acc16[4]);
                acc16[4] = _fma_1124;
                float _fma_1125 = __fmaf_rn(xi16[6], tv16[9], acc16[5]);
                acc16[5] = _fma_1125;
                float _fma_1126 = __fmaf_rn(xi16[6], tv16[10], acc16[6]);
                acc16[6] = _fma_1126;
                float _fma_1127 = __fmaf_rn(xi16[6], tv16[11], acc16[7]);
                acc16[7] = _fma_1127;
                float _fma_1128 = __fmaf_rn(xi16[10], tv16[8], acc16[8]);
                acc16[8] = _fma_1128;
                float _fma_1129 = __fmaf_rn(xi16[10], tv16[9], acc16[9]);
                acc16[9] = _fma_1129;
                float _fma_1130 = __fmaf_rn(xi16[10], tv16[10], acc16[10]);
                acc16[10] = _fma_1130;
                float _fma_1131 = __fmaf_rn(xi16[10], tv16[11], acc16[11]);
                acc16[11] = _fma_1131;
                float _fma_1132 = __fmaf_rn(xi16[14], tv16[8], acc16[12]);
                acc16[12] = _fma_1132;
                float _fma_1133 = __fmaf_rn(xi16[14], tv16[9], acc16[13]);
                acc16[13] = _fma_1133;
                float _fma_1134 = __fmaf_rn(xi16[14], tv16[10], acc16[14]);
                acc16[14] = _fma_1134;
                float _fma_1135 = __fmaf_rn(xi16[14], tv16[11], acc16[15]);
                acc16[15] = _fma_1135;
                float _fma_1136 = __fmaf_rn(xi16[3], tv16[12], acc16[0]);
                acc16[0] = _fma_1136;
                float _fma_1137 = __fmaf_rn(xi16[3], tv16[13], acc16[1]);
                acc16[1] = _fma_1137;
                float _fma_1138 = __fmaf_rn(xi16[3], tv16[14], acc16[2]);
                acc16[2] = _fma_1138;
                float _fma_1139 = __fmaf_rn(xi16[3], tv16[15], acc16[3]);
                acc16[3] = _fma_1139;
                float _fma_1140 = __fmaf_rn(xi16[7], tv16[12], acc16[4]);
                acc16[4] = _fma_1140;
                float _fma_1141 = __fmaf_rn(xi16[7], tv16[13], acc16[5]);
                acc16[5] = _fma_1141;
                float _fma_1142 = __fmaf_rn(xi16[7], tv16[14], acc16[6]);
                acc16[6] = _fma_1142;
                float _fma_1143 = __fmaf_rn(xi16[7], tv16[15], acc16[7]);
                acc16[7] = _fma_1143;
                float _fma_1144 = __fmaf_rn(xi16[11], tv16[12], acc16[8]);
                acc16[8] = _fma_1144;
                float _fma_1145 = __fmaf_rn(xi16[11], tv16[13], acc16[9]);
                acc16[9] = _fma_1145;
                float _fma_1146 = __fmaf_rn(xi16[11], tv16[14], acc16[10]);
                acc16[10] = _fma_1146;
                float _fma_1147 = __fmaf_rn(xi16[11], tv16[15], acc16[11]);
                acc16[11] = _fma_1147;
                float _fma_1148 = __fmaf_rn(xi16[15], tv16[12], acc16[12]);
                acc16[12] = _fma_1148;
                float _fma_1149 = __fmaf_rn(xi16[15], tv16[13], acc16[13]);
                acc16[13] = _fma_1149;
                float _fma_1150 = __fmaf_rn(xi16[15], tv16[14], acc16[14]);
                acc16[14] = _fma_1150;
                float _fma_1151 = __fmaf_rn(xi16[15], tv16[15], acc16[15]);
                acc16[15] = _fma_1151;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4) * 64 + ((warp_1 + 2) * 16 + 8 ^ ((warp_1 + 2) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 1) * 64 + ((warp_1 + 2) * 16 + 8 ^ ((warp_1 + 2) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[8])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 2) * 64 + ((warp_1 + 2) * 16 + 8 ^ ((warp_1 + 2) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[12])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 3) * 64 + ((warp_1 + 2) * 16 + 8 ^ ((warp_1 + 2) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 160 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 180 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 200 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 220 + cg * 4) * 4)));
                float _fma_1152 = __fmaf_rn(xi16[0], tv16[0], acc16[0]);
                acc16[0] = _fma_1152;
                float _fma_1153 = __fmaf_rn(xi16[0], tv16[1], acc16[1]);
                acc16[1] = _fma_1153;
                float _fma_1154 = __fmaf_rn(xi16[0], tv16[2], acc16[2]);
                acc16[2] = _fma_1154;
                float _fma_1155 = __fmaf_rn(xi16[0], tv16[3], acc16[3]);
                acc16[3] = _fma_1155;
                float _fma_1156 = __fmaf_rn(xi16[4], tv16[0], acc16[4]);
                acc16[4] = _fma_1156;
                float _fma_1157 = __fmaf_rn(xi16[4], tv16[1], acc16[5]);
                acc16[5] = _fma_1157;
                float _fma_1158 = __fmaf_rn(xi16[4], tv16[2], acc16[6]);
                acc16[6] = _fma_1158;
                float _fma_1159 = __fmaf_rn(xi16[4], tv16[3], acc16[7]);
                acc16[7] = _fma_1159;
                float _fma_1160 = __fmaf_rn(xi16[8], tv16[0], acc16[8]);
                acc16[8] = _fma_1160;
                float _fma_1161 = __fmaf_rn(xi16[8], tv16[1], acc16[9]);
                acc16[9] = _fma_1161;
                float _fma_1162 = __fmaf_rn(xi16[8], tv16[2], acc16[10]);
                acc16[10] = _fma_1162;
                float _fma_1163 = __fmaf_rn(xi16[8], tv16[3], acc16[11]);
                acc16[11] = _fma_1163;
                float _fma_1164 = __fmaf_rn(xi16[12], tv16[0], acc16[12]);
                acc16[12] = _fma_1164;
                float _fma_1165 = __fmaf_rn(xi16[12], tv16[1], acc16[13]);
                acc16[13] = _fma_1165;
                float _fma_1166 = __fmaf_rn(xi16[12], tv16[2], acc16[14]);
                acc16[14] = _fma_1166;
                float _fma_1167 = __fmaf_rn(xi16[12], tv16[3], acc16[15]);
                acc16[15] = _fma_1167;
                float _fma_1168 = __fmaf_rn(xi16[1], tv16[4], acc16[0]);
                acc16[0] = _fma_1168;
                float _fma_1169 = __fmaf_rn(xi16[1], tv16[5], acc16[1]);
                acc16[1] = _fma_1169;
                float _fma_1170 = __fmaf_rn(xi16[1], tv16[6], acc16[2]);
                acc16[2] = _fma_1170;
                float _fma_1171 = __fmaf_rn(xi16[1], tv16[7], acc16[3]);
                acc16[3] = _fma_1171;
                float _fma_1172 = __fmaf_rn(xi16[5], tv16[4], acc16[4]);
                acc16[4] = _fma_1172;
                float _fma_1173 = __fmaf_rn(xi16[5], tv16[5], acc16[5]);
                acc16[5] = _fma_1173;
                float _fma_1174 = __fmaf_rn(xi16[5], tv16[6], acc16[6]);
                acc16[6] = _fma_1174;
                float _fma_1175 = __fmaf_rn(xi16[5], tv16[7], acc16[7]);
                acc16[7] = _fma_1175;
                float _fma_1176 = __fmaf_rn(xi16[9], tv16[4], acc16[8]);
                acc16[8] = _fma_1176;
                float _fma_1177 = __fmaf_rn(xi16[9], tv16[5], acc16[9]);
                acc16[9] = _fma_1177;
                float _fma_1178 = __fmaf_rn(xi16[9], tv16[6], acc16[10]);
                acc16[10] = _fma_1178;
                float _fma_1179 = __fmaf_rn(xi16[9], tv16[7], acc16[11]);
                acc16[11] = _fma_1179;
                float _fma_1180 = __fmaf_rn(xi16[13], tv16[4], acc16[12]);
                acc16[12] = _fma_1180;
                float _fma_1181 = __fmaf_rn(xi16[13], tv16[5], acc16[13]);
                acc16[13] = _fma_1181;
                float _fma_1182 = __fmaf_rn(xi16[13], tv16[6], acc16[14]);
                acc16[14] = _fma_1182;
                float _fma_1183 = __fmaf_rn(xi16[13], tv16[7], acc16[15]);
                acc16[15] = _fma_1183;
                float _fma_1184 = __fmaf_rn(xi16[2], tv16[8], acc16[0]);
                acc16[0] = _fma_1184;
                float _fma_1185 = __fmaf_rn(xi16[2], tv16[9], acc16[1]);
                acc16[1] = _fma_1185;
                float _fma_1186 = __fmaf_rn(xi16[2], tv16[10], acc16[2]);
                acc16[2] = _fma_1186;
                float _fma_1187 = __fmaf_rn(xi16[2], tv16[11], acc16[3]);
                acc16[3] = _fma_1187;
                float _fma_1188 = __fmaf_rn(xi16[6], tv16[8], acc16[4]);
                acc16[4] = _fma_1188;
                float _fma_1189 = __fmaf_rn(xi16[6], tv16[9], acc16[5]);
                acc16[5] = _fma_1189;
                float _fma_1190 = __fmaf_rn(xi16[6], tv16[10], acc16[6]);
                acc16[6] = _fma_1190;
                float _fma_1191 = __fmaf_rn(xi16[6], tv16[11], acc16[7]);
                acc16[7] = _fma_1191;
                float _fma_1192 = __fmaf_rn(xi16[10], tv16[8], acc16[8]);
                acc16[8] = _fma_1192;
                float _fma_1193 = __fmaf_rn(xi16[10], tv16[9], acc16[9]);
                acc16[9] = _fma_1193;
                float _fma_1194 = __fmaf_rn(xi16[10], tv16[10], acc16[10]);
                acc16[10] = _fma_1194;
                float _fma_1195 = __fmaf_rn(xi16[10], tv16[11], acc16[11]);
                acc16[11] = _fma_1195;
                float _fma_1196 = __fmaf_rn(xi16[14], tv16[8], acc16[12]);
                acc16[12] = _fma_1196;
                float _fma_1197 = __fmaf_rn(xi16[14], tv16[9], acc16[13]);
                acc16[13] = _fma_1197;
                float _fma_1198 = __fmaf_rn(xi16[14], tv16[10], acc16[14]);
                acc16[14] = _fma_1198;
                float _fma_1199 = __fmaf_rn(xi16[14], tv16[11], acc16[15]);
                acc16[15] = _fma_1199;
                float _fma_1200 = __fmaf_rn(xi16[3], tv16[12], acc16[0]);
                acc16[0] = _fma_1200;
                float _fma_1201 = __fmaf_rn(xi16[3], tv16[13], acc16[1]);
                acc16[1] = _fma_1201;
                float _fma_1202 = __fmaf_rn(xi16[3], tv16[14], acc16[2]);
                acc16[2] = _fma_1202;
                float _fma_1203 = __fmaf_rn(xi16[3], tv16[15], acc16[3]);
                acc16[3] = _fma_1203;
                float _fma_1204 = __fmaf_rn(xi16[7], tv16[12], acc16[4]);
                acc16[4] = _fma_1204;
                float _fma_1205 = __fmaf_rn(xi16[7], tv16[13], acc16[5]);
                acc16[5] = _fma_1205;
                float _fma_1206 = __fmaf_rn(xi16[7], tv16[14], acc16[6]);
                acc16[6] = _fma_1206;
                float _fma_1207 = __fmaf_rn(xi16[7], tv16[15], acc16[7]);
                acc16[7] = _fma_1207;
                float _fma_1208 = __fmaf_rn(xi16[11], tv16[12], acc16[8]);
                acc16[8] = _fma_1208;
                float _fma_1209 = __fmaf_rn(xi16[11], tv16[13], acc16[9]);
                acc16[9] = _fma_1209;
                float _fma_1210 = __fmaf_rn(xi16[11], tv16[14], acc16[10]);
                acc16[10] = _fma_1210;
                float _fma_1211 = __fmaf_rn(xi16[11], tv16[15], acc16[11]);
                acc16[11] = _fma_1211;
                float _fma_1212 = __fmaf_rn(xi16[15], tv16[12], acc16[12]);
                acc16[12] = _fma_1212;
                float _fma_1213 = __fmaf_rn(xi16[15], tv16[13], acc16[13]);
                acc16[13] = _fma_1213;
                float _fma_1214 = __fmaf_rn(xi16[15], tv16[14], acc16[14]);
                acc16[14] = _fma_1214;
                float _fma_1215 = __fmaf_rn(xi16[15], tv16[15], acc16[15]);
                acc16[15] = _fma_1215;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4) * 64 + ((warp_1 + 2) * 16 + 12 ^ ((warp_1 + 2) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 1) * 64 + ((warp_1 + 2) * 16 + 12 ^ ((warp_1 + 2) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[8])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 2) * 64 + ((warp_1 + 2) * 16 + 12 ^ ((warp_1 + 2) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[12])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 3) * 64 + ((warp_1 + 2) * 16 + 12 ^ ((warp_1 + 2) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 240 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 260 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 280 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 300 + cg * 4) * 4)));
                float _fma_1216 = __fmaf_rn(xi16[0], tv16[0], acc16[0]);
                acc16[0] = _fma_1216;
                float _fma_1217 = __fmaf_rn(xi16[0], tv16[1], acc16[1]);
                acc16[1] = _fma_1217;
                float _fma_1218 = __fmaf_rn(xi16[0], tv16[2], acc16[2]);
                acc16[2] = _fma_1218;
                float _fma_1219 = __fmaf_rn(xi16[0], tv16[3], acc16[3]);
                acc16[3] = _fma_1219;
                float _fma_1220 = __fmaf_rn(xi16[4], tv16[0], acc16[4]);
                acc16[4] = _fma_1220;
                float _fma_1221 = __fmaf_rn(xi16[4], tv16[1], acc16[5]);
                acc16[5] = _fma_1221;
                float _fma_1222 = __fmaf_rn(xi16[4], tv16[2], acc16[6]);
                acc16[6] = _fma_1222;
                float _fma_1223 = __fmaf_rn(xi16[4], tv16[3], acc16[7]);
                acc16[7] = _fma_1223;
                float _fma_1224 = __fmaf_rn(xi16[8], tv16[0], acc16[8]);
                acc16[8] = _fma_1224;
                float _fma_1225 = __fmaf_rn(xi16[8], tv16[1], acc16[9]);
                acc16[9] = _fma_1225;
                float _fma_1226 = __fmaf_rn(xi16[8], tv16[2], acc16[10]);
                acc16[10] = _fma_1226;
                float _fma_1227 = __fmaf_rn(xi16[8], tv16[3], acc16[11]);
                acc16[11] = _fma_1227;
                float _fma_1228 = __fmaf_rn(xi16[12], tv16[0], acc16[12]);
                acc16[12] = _fma_1228;
                float _fma_1229 = __fmaf_rn(xi16[12], tv16[1], acc16[13]);
                acc16[13] = _fma_1229;
                float _fma_1230 = __fmaf_rn(xi16[12], tv16[2], acc16[14]);
                acc16[14] = _fma_1230;
                float _fma_1231 = __fmaf_rn(xi16[12], tv16[3], acc16[15]);
                acc16[15] = _fma_1231;
                float _fma_1232 = __fmaf_rn(xi16[1], tv16[4], acc16[0]);
                acc16[0] = _fma_1232;
                float _fma_1233 = __fmaf_rn(xi16[1], tv16[5], acc16[1]);
                acc16[1] = _fma_1233;
                float _fma_1234 = __fmaf_rn(xi16[1], tv16[6], acc16[2]);
                acc16[2] = _fma_1234;
                float _fma_1235 = __fmaf_rn(xi16[1], tv16[7], acc16[3]);
                acc16[3] = _fma_1235;
                float _fma_1236 = __fmaf_rn(xi16[5], tv16[4], acc16[4]);
                acc16[4] = _fma_1236;
                float _fma_1237 = __fmaf_rn(xi16[5], tv16[5], acc16[5]);
                acc16[5] = _fma_1237;
                float _fma_1238 = __fmaf_rn(xi16[5], tv16[6], acc16[6]);
                acc16[6] = _fma_1238;
                float _fma_1239 = __fmaf_rn(xi16[5], tv16[7], acc16[7]);
                acc16[7] = _fma_1239;
                float _fma_1240 = __fmaf_rn(xi16[9], tv16[4], acc16[8]);
                acc16[8] = _fma_1240;
                float _fma_1241 = __fmaf_rn(xi16[9], tv16[5], acc16[9]);
                acc16[9] = _fma_1241;
                float _fma_1242 = __fmaf_rn(xi16[9], tv16[6], acc16[10]);
                acc16[10] = _fma_1242;
                float _fma_1243 = __fmaf_rn(xi16[9], tv16[7], acc16[11]);
                acc16[11] = _fma_1243;
                float _fma_1244 = __fmaf_rn(xi16[13], tv16[4], acc16[12]);
                acc16[12] = _fma_1244;
                float _fma_1245 = __fmaf_rn(xi16[13], tv16[5], acc16[13]);
                acc16[13] = _fma_1245;
                float _fma_1246 = __fmaf_rn(xi16[13], tv16[6], acc16[14]);
                acc16[14] = _fma_1246;
                float _fma_1247 = __fmaf_rn(xi16[13], tv16[7], acc16[15]);
                acc16[15] = _fma_1247;
                float _fma_1248 = __fmaf_rn(xi16[2], tv16[8], acc16[0]);
                acc16[0] = _fma_1248;
                float _fma_1249 = __fmaf_rn(xi16[2], tv16[9], acc16[1]);
                acc16[1] = _fma_1249;
                float _fma_1250 = __fmaf_rn(xi16[2], tv16[10], acc16[2]);
                acc16[2] = _fma_1250;
                float _fma_1251 = __fmaf_rn(xi16[2], tv16[11], acc16[3]);
                acc16[3] = _fma_1251;
                float _fma_1252 = __fmaf_rn(xi16[6], tv16[8], acc16[4]);
                acc16[4] = _fma_1252;
                float _fma_1253 = __fmaf_rn(xi16[6], tv16[9], acc16[5]);
                acc16[5] = _fma_1253;
                float _fma_1254 = __fmaf_rn(xi16[6], tv16[10], acc16[6]);
                acc16[6] = _fma_1254;
                float _fma_1255 = __fmaf_rn(xi16[6], tv16[11], acc16[7]);
                acc16[7] = _fma_1255;
                float _fma_1256 = __fmaf_rn(xi16[10], tv16[8], acc16[8]);
                acc16[8] = _fma_1256;
                float _fma_1257 = __fmaf_rn(xi16[10], tv16[9], acc16[9]);
                acc16[9] = _fma_1257;
                float _fma_1258 = __fmaf_rn(xi16[10], tv16[10], acc16[10]);
                acc16[10] = _fma_1258;
                float _fma_1259 = __fmaf_rn(xi16[10], tv16[11], acc16[11]);
                acc16[11] = _fma_1259;
                float _fma_1260 = __fmaf_rn(xi16[14], tv16[8], acc16[12]);
                acc16[12] = _fma_1260;
                float _fma_1261 = __fmaf_rn(xi16[14], tv16[9], acc16[13]);
                acc16[13] = _fma_1261;
                float _fma_1262 = __fmaf_rn(xi16[14], tv16[10], acc16[14]);
                acc16[14] = _fma_1262;
                float _fma_1263 = __fmaf_rn(xi16[14], tv16[11], acc16[15]);
                acc16[15] = _fma_1263;
                float _fma_1264 = __fmaf_rn(xi16[3], tv16[12], acc16[0]);
                acc16[0] = _fma_1264;
                float _fma_1265 = __fmaf_rn(xi16[3], tv16[13], acc16[1]);
                acc16[1] = _fma_1265;
                float _fma_1266 = __fmaf_rn(xi16[3], tv16[14], acc16[2]);
                acc16[2] = _fma_1266;
                float _fma_1267 = __fmaf_rn(xi16[3], tv16[15], acc16[3]);
                acc16[3] = _fma_1267;
                float _fma_1268 = __fmaf_rn(xi16[7], tv16[12], acc16[4]);
                acc16[4] = _fma_1268;
                float _fma_1269 = __fmaf_rn(xi16[7], tv16[13], acc16[5]);
                acc16[5] = _fma_1269;
                float _fma_1270 = __fmaf_rn(xi16[7], tv16[14], acc16[6]);
                acc16[6] = _fma_1270;
                float _fma_1271 = __fmaf_rn(xi16[7], tv16[15], acc16[7]);
                acc16[7] = _fma_1271;
                float _fma_1272 = __fmaf_rn(xi16[11], tv16[12], acc16[8]);
                acc16[8] = _fma_1272;
                float _fma_1273 = __fmaf_rn(xi16[11], tv16[13], acc16[9]);
                acc16[9] = _fma_1273;
                float _fma_1274 = __fmaf_rn(xi16[11], tv16[14], acc16[10]);
                acc16[10] = _fma_1274;
                float _fma_1275 = __fmaf_rn(xi16[11], tv16[15], acc16[11]);
                acc16[11] = _fma_1275;
                float _fma_1276 = __fmaf_rn(xi16[15], tv16[12], acc16[12]);
                acc16[12] = _fma_1276;
                float _fma_1277 = __fmaf_rn(xi16[15], tv16[13], acc16[13]);
                acc16[13] = _fma_1277;
                float _fma_1278 = __fmaf_rn(xi16[15], tv16[14], acc16[14]);
                acc16[14] = _fma_1278;
                float _fma_1279 = __fmaf_rn(xi16[15], tv16[15], acc16[15]);
                acc16[15] = _fma_1279;
                acc16[0] = -acc16[0];
                acc16[1] = -acc16[1];
                acc16[2] = -acc16[2];
                acc16[3] = -acc16[3];
                acc16[4] = -acc16[4];
                acc16[5] = -acc16[5];
                acc16[6] = -acc16[6];
                acc16[7] = -acc16[7];
                acc16[8] = -acc16[8];
                acc16[9] = -acc16[9];
                acc16[10] = -acc16[10];
                acc16[11] = -acc16[11];
                acc16[12] = -acc16[12];
                acc16[13] = -acc16[13];
                acc16[14] = -acc16[14];
                acc16[15] = -acc16[15];
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[0])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 1) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[4])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 2) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[8])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + rg * 4 + 3) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[12])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 3])));
            }
        }
        __syncthreads();
        if (lane_0 < 16) {
            if (warp_1 < 1) {
                acc16[0] = 0.0f;
                acc16[1] = 0.0f;
                acc16[2] = 0.0f;
                acc16[3] = 0.0f;
                acc16[4] = 0.0f;
                acc16[5] = 0.0f;
                acc16[6] = 0.0f;
                acc16[7] = 0.0f;
                acc16[8] = 0.0f;
                acc16[9] = 0.0f;
                acc16[10] = 0.0f;
                acc16[11] = 0.0f;
                acc16[12] = 0.0f;
                acc16[13] = 0.0f;
                acc16[14] = 0.0f;
                acc16[15] = 0.0f;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((warp_1 * 16 * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((warp_1 * 16 * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 >> 2 & 1) << 4)) * 4)));
                float _fma_1280 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1280;
                float _fma_1281 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1281;
                float _fma_1282 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1282;
                float _fma_1283 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1283;
                float _fma_1284 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1284;
                float _fma_1285 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1285;
                float _fma_1286 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1286;
                float _fma_1287 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1287;
                float _fma_1288 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1288;
                float _fma_1289 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1289;
                float _fma_1290 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1290;
                float _fma_1291 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1291;
                float _fma_1292 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1292;
                float _fma_1293 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1293;
                float _fma_1294 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1294;
                float _fma_1295 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1295;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 + 1) * 16 * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 + 1) * 16 * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 >> 2 & 1) << 4)) * 4)));
                float _fma_1296 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1296;
                float _fma_1297 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1297;
                float _fma_1298 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1298;
                float _fma_1299 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1299;
                float _fma_1300 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1300;
                float _fma_1301 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1301;
                float _fma_1302 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1302;
                float _fma_1303 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1303;
                float _fma_1304 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1304;
                float _fma_1305 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1305;
                float _fma_1306 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1306;
                float _fma_1307 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1307;
                float _fma_1308 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1308;
                float _fma_1309 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1309;
                float _fma_1310 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1310;
                float _fma_1311 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1311;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 + 2) * 16 * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 + 2) * 16 * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 >> 2 & 1) << 4)) * 4)));
                float _fma_1312 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1312;
                float _fma_1313 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1313;
                float _fma_1314 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1314;
                float _fma_1315 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1315;
                float _fma_1316 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1316;
                float _fma_1317 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1317;
                float _fma_1318 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1318;
                float _fma_1319 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1319;
                float _fma_1320 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1320;
                float _fma_1321 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1321;
                float _fma_1322 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1322;
                float _fma_1323 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1323;
                float _fma_1324 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1324;
                float _fma_1325 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1325;
                float _fma_1326 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1326;
                float _fma_1327 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1327;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 1 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 1) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 1 >> 2 & 1) << 4)) * 4)));
                float _fma_1328 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1328;
                float _fma_1329 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1329;
                float _fma_1330 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1330;
                float _fma_1331 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1331;
                float _fma_1332 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1332;
                float _fma_1333 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1333;
                float _fma_1334 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1334;
                float _fma_1335 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1335;
                float _fma_1336 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1336;
                float _fma_1337 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1337;
                float _fma_1338 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1338;
                float _fma_1339 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1339;
                float _fma_1340 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1340;
                float _fma_1341 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1341;
                float _fma_1342 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1342;
                float _fma_1343 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1343;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 1) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 1 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 1) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 1 >> 2 & 1) << 4)) * 4)));
                float _fma_1344 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1344;
                float _fma_1345 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1345;
                float _fma_1346 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1346;
                float _fma_1347 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1347;
                float _fma_1348 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1348;
                float _fma_1349 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1349;
                float _fma_1350 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1350;
                float _fma_1351 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1351;
                float _fma_1352 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1352;
                float _fma_1353 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1353;
                float _fma_1354 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1354;
                float _fma_1355 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1355;
                float _fma_1356 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1356;
                float _fma_1357 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1357;
                float _fma_1358 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1358;
                float _fma_1359 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1359;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 1) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 1 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 1) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 1 >> 2 & 1) << 4)) * 4)));
                float _fma_1360 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1360;
                float _fma_1361 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1361;
                float _fma_1362 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1362;
                float _fma_1363 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1363;
                float _fma_1364 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1364;
                float _fma_1365 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1365;
                float _fma_1366 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1366;
                float _fma_1367 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1367;
                float _fma_1368 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1368;
                float _fma_1369 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1369;
                float _fma_1370 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1370;
                float _fma_1371 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1371;
                float _fma_1372 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1372;
                float _fma_1373 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1373;
                float _fma_1374 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1374;
                float _fma_1375 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1375;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 2 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 2) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 2 >> 2 & 1) << 4)) * 4)));
                float _fma_1376 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1376;
                float _fma_1377 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1377;
                float _fma_1378 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1378;
                float _fma_1379 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1379;
                float _fma_1380 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1380;
                float _fma_1381 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1381;
                float _fma_1382 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1382;
                float _fma_1383 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1383;
                float _fma_1384 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1384;
                float _fma_1385 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1385;
                float _fma_1386 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1386;
                float _fma_1387 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1387;
                float _fma_1388 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1388;
                float _fma_1389 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1389;
                float _fma_1390 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1390;
                float _fma_1391 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1391;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 2) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 2 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 2) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 2 >> 2 & 1) << 4)) * 4)));
                float _fma_1392 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1392;
                float _fma_1393 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1393;
                float _fma_1394 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1394;
                float _fma_1395 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1395;
                float _fma_1396 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1396;
                float _fma_1397 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1397;
                float _fma_1398 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1398;
                float _fma_1399 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1399;
                float _fma_1400 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1400;
                float _fma_1401 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1401;
                float _fma_1402 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1402;
                float _fma_1403 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1403;
                float _fma_1404 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1404;
                float _fma_1405 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1405;
                float _fma_1406 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1406;
                float _fma_1407 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1407;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 2) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 2 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 2) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 2 >> 2 & 1) << 4)) * 4)));
                float _fma_1408 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1408;
                float _fma_1409 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1409;
                float _fma_1410 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1410;
                float _fma_1411 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1411;
                float _fma_1412 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1412;
                float _fma_1413 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1413;
                float _fma_1414 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1414;
                float _fma_1415 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1415;
                float _fma_1416 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1416;
                float _fma_1417 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1417;
                float _fma_1418 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1418;
                float _fma_1419 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1419;
                float _fma_1420 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1420;
                float _fma_1421 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1421;
                float _fma_1422 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1422;
                float _fma_1423 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1423;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 3 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 3) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 3 >> 2 & 1) << 4)) * 4)));
                float _fma_1424 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1424;
                float _fma_1425 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1425;
                float _fma_1426 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1426;
                float _fma_1427 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1427;
                float _fma_1428 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1428;
                float _fma_1429 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1429;
                float _fma_1430 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1430;
                float _fma_1431 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1431;
                float _fma_1432 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1432;
                float _fma_1433 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1433;
                float _fma_1434 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1434;
                float _fma_1435 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1435;
                float _fma_1436 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1436;
                float _fma_1437 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1437;
                float _fma_1438 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1438;
                float _fma_1439 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1439;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 3) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 3 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 3) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 3 >> 2 & 1) << 4)) * 4)));
                float _fma_1440 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1440;
                float _fma_1441 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1441;
                float _fma_1442 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1442;
                float _fma_1443 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1443;
                float _fma_1444 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1444;
                float _fma_1445 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1445;
                float _fma_1446 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1446;
                float _fma_1447 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1447;
                float _fma_1448 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1448;
                float _fma_1449 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1449;
                float _fma_1450 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1450;
                float _fma_1451 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1451;
                float _fma_1452 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1452;
                float _fma_1453 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1453;
                float _fma_1454 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1454;
                float _fma_1455 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1455;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 3) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 3 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 3) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 3 >> 2 & 1) << 4)) * 4)));
                float _fma_1456 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1456;
                float _fma_1457 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1457;
                float _fma_1458 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1458;
                float _fma_1459 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1459;
                float _fma_1460 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1460;
                float _fma_1461 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1461;
                float _fma_1462 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1462;
                float _fma_1463 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1463;
                float _fma_1464 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1464;
                float _fma_1465 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1465;
                float _fma_1466 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1466;
                float _fma_1467 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1467;
                float _fma_1468 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1468;
                float _fma_1469 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1469;
                float _fma_1470 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1470;
                float _fma_1471 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1471;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 4 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 4) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 4 >> 2 & 1) << 4)) * 4)));
                float _fma_1472 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1472;
                float _fma_1473 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1473;
                float _fma_1474 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1474;
                float _fma_1475 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1475;
                float _fma_1476 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1476;
                float _fma_1477 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1477;
                float _fma_1478 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1478;
                float _fma_1479 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1479;
                float _fma_1480 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1480;
                float _fma_1481 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1481;
                float _fma_1482 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1482;
                float _fma_1483 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1483;
                float _fma_1484 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1484;
                float _fma_1485 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1485;
                float _fma_1486 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1486;
                float _fma_1487 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1487;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 4) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 4 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 4) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 4 >> 2 & 1) << 4)) * 4)));
                float _fma_1488 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1488;
                float _fma_1489 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1489;
                float _fma_1490 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1490;
                float _fma_1491 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1491;
                float _fma_1492 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1492;
                float _fma_1493 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1493;
                float _fma_1494 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1494;
                float _fma_1495 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1495;
                float _fma_1496 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1496;
                float _fma_1497 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1497;
                float _fma_1498 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1498;
                float _fma_1499 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1499;
                float _fma_1500 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1500;
                float _fma_1501 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1501;
                float _fma_1502 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1502;
                float _fma_1503 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1503;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 4) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 4 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 4) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 4 >> 2 & 1) << 4)) * 4)));
                float _fma_1504 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1504;
                float _fma_1505 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1505;
                float _fma_1506 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1506;
                float _fma_1507 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1507;
                float _fma_1508 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1508;
                float _fma_1509 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1509;
                float _fma_1510 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1510;
                float _fma_1511 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1511;
                float _fma_1512 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1512;
                float _fma_1513 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1513;
                float _fma_1514 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1514;
                float _fma_1515 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1515;
                float _fma_1516 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1516;
                float _fma_1517 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1517;
                float _fma_1518 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1518;
                float _fma_1519 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1519;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 5 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 5) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 5 >> 2 & 1) << 4)) * 4)));
                float _fma_1520 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1520;
                float _fma_1521 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1521;
                float _fma_1522 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1522;
                float _fma_1523 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1523;
                float _fma_1524 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1524;
                float _fma_1525 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1525;
                float _fma_1526 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1526;
                float _fma_1527 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1527;
                float _fma_1528 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1528;
                float _fma_1529 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1529;
                float _fma_1530 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1530;
                float _fma_1531 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1531;
                float _fma_1532 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1532;
                float _fma_1533 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1533;
                float _fma_1534 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1534;
                float _fma_1535 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1535;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 5) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 5 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 5) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 5 >> 2 & 1) << 4)) * 4)));
                float _fma_1536 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1536;
                float _fma_1537 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1537;
                float _fma_1538 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1538;
                float _fma_1539 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1539;
                float _fma_1540 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1540;
                float _fma_1541 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1541;
                float _fma_1542 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1542;
                float _fma_1543 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1543;
                float _fma_1544 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1544;
                float _fma_1545 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1545;
                float _fma_1546 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1546;
                float _fma_1547 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1547;
                float _fma_1548 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1548;
                float _fma_1549 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1549;
                float _fma_1550 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1550;
                float _fma_1551 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1551;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 5) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 5 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 5) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 5 >> 2 & 1) << 4)) * 4)));
                float _fma_1552 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1552;
                float _fma_1553 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1553;
                float _fma_1554 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1554;
                float _fma_1555 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1555;
                float _fma_1556 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1556;
                float _fma_1557 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1557;
                float _fma_1558 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1558;
                float _fma_1559 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1559;
                float _fma_1560 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1560;
                float _fma_1561 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1561;
                float _fma_1562 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1562;
                float _fma_1563 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1563;
                float _fma_1564 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1564;
                float _fma_1565 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1565;
                float _fma_1566 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1566;
                float _fma_1567 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1567;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 6 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 6) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 6 >> 2 & 1) << 4)) * 4)));
                float _fma_1568 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1568;
                float _fma_1569 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1569;
                float _fma_1570 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1570;
                float _fma_1571 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1571;
                float _fma_1572 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1572;
                float _fma_1573 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1573;
                float _fma_1574 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1574;
                float _fma_1575 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1575;
                float _fma_1576 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1576;
                float _fma_1577 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1577;
                float _fma_1578 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1578;
                float _fma_1579 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1579;
                float _fma_1580 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1580;
                float _fma_1581 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1581;
                float _fma_1582 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1582;
                float _fma_1583 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1583;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 6) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 6 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 6) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 6 >> 2 & 1) << 4)) * 4)));
                float _fma_1584 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1584;
                float _fma_1585 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1585;
                float _fma_1586 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1586;
                float _fma_1587 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1587;
                float _fma_1588 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1588;
                float _fma_1589 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1589;
                float _fma_1590 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1590;
                float _fma_1591 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1591;
                float _fma_1592 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1592;
                float _fma_1593 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1593;
                float _fma_1594 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1594;
                float _fma_1595 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1595;
                float _fma_1596 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1596;
                float _fma_1597 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1597;
                float _fma_1598 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1598;
                float _fma_1599 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1599;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 6) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 6 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 6) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 6 >> 2 & 1) << 4)) * 4)));
                float _fma_1600 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1600;
                float _fma_1601 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1601;
                float _fma_1602 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1602;
                float _fma_1603 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1603;
                float _fma_1604 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1604;
                float _fma_1605 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1605;
                float _fma_1606 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1606;
                float _fma_1607 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1607;
                float _fma_1608 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1608;
                float _fma_1609 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1609;
                float _fma_1610 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1610;
                float _fma_1611 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1611;
                float _fma_1612 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1612;
                float _fma_1613 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1613;
                float _fma_1614 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1614;
                float _fma_1615 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1615;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 7 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 7) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 7 >> 2 & 1) << 4)) * 4)));
                float _fma_1616 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1616;
                float _fma_1617 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1617;
                float _fma_1618 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1618;
                float _fma_1619 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1619;
                float _fma_1620 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1620;
                float _fma_1621 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1621;
                float _fma_1622 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1622;
                float _fma_1623 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1623;
                float _fma_1624 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1624;
                float _fma_1625 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1625;
                float _fma_1626 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1626;
                float _fma_1627 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1627;
                float _fma_1628 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1628;
                float _fma_1629 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1629;
                float _fma_1630 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1630;
                float _fma_1631 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1631;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 7) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 7 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 7) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 7 >> 2 & 1) << 4)) * 4)));
                float _fma_1632 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1632;
                float _fma_1633 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1633;
                float _fma_1634 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1634;
                float _fma_1635 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1635;
                float _fma_1636 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1636;
                float _fma_1637 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1637;
                float _fma_1638 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1638;
                float _fma_1639 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1639;
                float _fma_1640 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1640;
                float _fma_1641 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1641;
                float _fma_1642 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1642;
                float _fma_1643 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1643;
                float _fma_1644 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1644;
                float _fma_1645 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1645;
                float _fma_1646 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1646;
                float _fma_1647 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1647;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 7) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 7 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 7) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 7 >> 2 & 1) << 4)) * 4)));
                float _fma_1648 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1648;
                float _fma_1649 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1649;
                float _fma_1650 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1650;
                float _fma_1651 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1651;
                float _fma_1652 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1652;
                float _fma_1653 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1653;
                float _fma_1654 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1654;
                float _fma_1655 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1655;
                float _fma_1656 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1656;
                float _fma_1657 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1657;
                float _fma_1658 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1658;
                float _fma_1659 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1659;
                float _fma_1660 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1660;
                float _fma_1661 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1661;
                float _fma_1662 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1662;
                float _fma_1663 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1663;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 8 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 8) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 8 >> 2 & 1) << 4)) * 4)));
                float _fma_1664 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1664;
                float _fma_1665 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1665;
                float _fma_1666 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1666;
                float _fma_1667 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1667;
                float _fma_1668 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1668;
                float _fma_1669 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1669;
                float _fma_1670 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1670;
                float _fma_1671 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1671;
                float _fma_1672 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1672;
                float _fma_1673 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1673;
                float _fma_1674 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1674;
                float _fma_1675 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1675;
                float _fma_1676 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1676;
                float _fma_1677 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1677;
                float _fma_1678 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1678;
                float _fma_1679 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1679;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 8) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 8 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 8) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 8 >> 2 & 1) << 4)) * 4)));
                float _fma_1680 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1680;
                float _fma_1681 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1681;
                float _fma_1682 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1682;
                float _fma_1683 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1683;
                float _fma_1684 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1684;
                float _fma_1685 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1685;
                float _fma_1686 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1686;
                float _fma_1687 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1687;
                float _fma_1688 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1688;
                float _fma_1689 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1689;
                float _fma_1690 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1690;
                float _fma_1691 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1691;
                float _fma_1692 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1692;
                float _fma_1693 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1693;
                float _fma_1694 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1694;
                float _fma_1695 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1695;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 8) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 8 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 8) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 8 >> 2 & 1) << 4)) * 4)));
                float _fma_1696 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1696;
                float _fma_1697 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1697;
                float _fma_1698 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1698;
                float _fma_1699 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1699;
                float _fma_1700 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1700;
                float _fma_1701 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1701;
                float _fma_1702 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1702;
                float _fma_1703 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1703;
                float _fma_1704 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1704;
                float _fma_1705 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1705;
                float _fma_1706 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1706;
                float _fma_1707 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1707;
                float _fma_1708 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1708;
                float _fma_1709 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1709;
                float _fma_1710 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1710;
                float _fma_1711 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1711;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 9 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 9) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 9 >> 2 & 1) << 4)) * 4)));
                float _fma_1712 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1712;
                float _fma_1713 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1713;
                float _fma_1714 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1714;
                float _fma_1715 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1715;
                float _fma_1716 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1716;
                float _fma_1717 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1717;
                float _fma_1718 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1718;
                float _fma_1719 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1719;
                float _fma_1720 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1720;
                float _fma_1721 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1721;
                float _fma_1722 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1722;
                float _fma_1723 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1723;
                float _fma_1724 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1724;
                float _fma_1725 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1725;
                float _fma_1726 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1726;
                float _fma_1727 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1727;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 9) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 9 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 9) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 9 >> 2 & 1) << 4)) * 4)));
                float _fma_1728 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1728;
                float _fma_1729 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1729;
                float _fma_1730 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1730;
                float _fma_1731 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1731;
                float _fma_1732 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1732;
                float _fma_1733 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1733;
                float _fma_1734 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1734;
                float _fma_1735 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1735;
                float _fma_1736 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1736;
                float _fma_1737 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1737;
                float _fma_1738 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1738;
                float _fma_1739 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1739;
                float _fma_1740 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1740;
                float _fma_1741 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1741;
                float _fma_1742 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1742;
                float _fma_1743 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1743;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 9) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 9 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 9) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 9 >> 2 & 1) << 4)) * 4)));
                float _fma_1744 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1744;
                float _fma_1745 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1745;
                float _fma_1746 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1746;
                float _fma_1747 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1747;
                float _fma_1748 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1748;
                float _fma_1749 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1749;
                float _fma_1750 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1750;
                float _fma_1751 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1751;
                float _fma_1752 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1752;
                float _fma_1753 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1753;
                float _fma_1754 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1754;
                float _fma_1755 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1755;
                float _fma_1756 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1756;
                float _fma_1757 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1757;
                float _fma_1758 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1758;
                float _fma_1759 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1759;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 10 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 10) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 10 >> 2 & 1) << 4)) * 4)));
                float _fma_1760 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1760;
                float _fma_1761 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1761;
                float _fma_1762 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1762;
                float _fma_1763 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1763;
                float _fma_1764 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1764;
                float _fma_1765 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1765;
                float _fma_1766 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1766;
                float _fma_1767 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1767;
                float _fma_1768 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1768;
                float _fma_1769 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1769;
                float _fma_1770 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1770;
                float _fma_1771 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1771;
                float _fma_1772 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1772;
                float _fma_1773 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1773;
                float _fma_1774 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1774;
                float _fma_1775 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1775;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 10) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 10 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 10) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 10 >> 2 & 1) << 4)) * 4)));
                float _fma_1776 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1776;
                float _fma_1777 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1777;
                float _fma_1778 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1778;
                float _fma_1779 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1779;
                float _fma_1780 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1780;
                float _fma_1781 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1781;
                float _fma_1782 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1782;
                float _fma_1783 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1783;
                float _fma_1784 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1784;
                float _fma_1785 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1785;
                float _fma_1786 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1786;
                float _fma_1787 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1787;
                float _fma_1788 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1788;
                float _fma_1789 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1789;
                float _fma_1790 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1790;
                float _fma_1791 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1791;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 10) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 10 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 10) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 10 >> 2 & 1) << 4)) * 4)));
                float _fma_1792 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1792;
                float _fma_1793 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1793;
                float _fma_1794 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1794;
                float _fma_1795 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1795;
                float _fma_1796 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1796;
                float _fma_1797 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1797;
                float _fma_1798 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1798;
                float _fma_1799 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1799;
                float _fma_1800 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1800;
                float _fma_1801 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1801;
                float _fma_1802 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1802;
                float _fma_1803 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1803;
                float _fma_1804 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1804;
                float _fma_1805 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1805;
                float _fma_1806 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1806;
                float _fma_1807 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1807;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 11 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 11) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 11 >> 2 & 1) << 4)) * 4)));
                float _fma_1808 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1808;
                float _fma_1809 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1809;
                float _fma_1810 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1810;
                float _fma_1811 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1811;
                float _fma_1812 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1812;
                float _fma_1813 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1813;
                float _fma_1814 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1814;
                float _fma_1815 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1815;
                float _fma_1816 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1816;
                float _fma_1817 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1817;
                float _fma_1818 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1818;
                float _fma_1819 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1819;
                float _fma_1820 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1820;
                float _fma_1821 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1821;
                float _fma_1822 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1822;
                float _fma_1823 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1823;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 11) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 11 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 11) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 11 >> 2 & 1) << 4)) * 4)));
                float _fma_1824 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1824;
                float _fma_1825 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1825;
                float _fma_1826 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1826;
                float _fma_1827 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1827;
                float _fma_1828 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1828;
                float _fma_1829 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1829;
                float _fma_1830 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1830;
                float _fma_1831 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1831;
                float _fma_1832 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1832;
                float _fma_1833 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1833;
                float _fma_1834 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1834;
                float _fma_1835 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1835;
                float _fma_1836 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1836;
                float _fma_1837 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1837;
                float _fma_1838 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1838;
                float _fma_1839 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1839;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 11) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 11 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 11) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 11 >> 2 & 1) << 4)) * 4)));
                float _fma_1840 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1840;
                float _fma_1841 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1841;
                float _fma_1842 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1842;
                float _fma_1843 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1843;
                float _fma_1844 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1844;
                float _fma_1845 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1845;
                float _fma_1846 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1846;
                float _fma_1847 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1847;
                float _fma_1848 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1848;
                float _fma_1849 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1849;
                float _fma_1850 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1850;
                float _fma_1851 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1851;
                float _fma_1852 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1852;
                float _fma_1853 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1853;
                float _fma_1854 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1854;
                float _fma_1855 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1855;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 12 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 12) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 12 >> 2 & 1) << 4)) * 4)));
                float _fma_1856 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1856;
                float _fma_1857 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1857;
                float _fma_1858 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1858;
                float _fma_1859 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1859;
                float _fma_1860 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1860;
                float _fma_1861 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1861;
                float _fma_1862 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1862;
                float _fma_1863 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1863;
                float _fma_1864 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1864;
                float _fma_1865 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1865;
                float _fma_1866 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1866;
                float _fma_1867 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1867;
                float _fma_1868 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1868;
                float _fma_1869 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1869;
                float _fma_1870 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1870;
                float _fma_1871 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1871;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 12) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 12 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 12) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 12 >> 2 & 1) << 4)) * 4)));
                float _fma_1872 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1872;
                float _fma_1873 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1873;
                float _fma_1874 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1874;
                float _fma_1875 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1875;
                float _fma_1876 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1876;
                float _fma_1877 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1877;
                float _fma_1878 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1878;
                float _fma_1879 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1879;
                float _fma_1880 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1880;
                float _fma_1881 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1881;
                float _fma_1882 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1882;
                float _fma_1883 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1883;
                float _fma_1884 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1884;
                float _fma_1885 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1885;
                float _fma_1886 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1886;
                float _fma_1887 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1887;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 12) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 12 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 12) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 12 >> 2 & 1) << 4)) * 4)));
                float _fma_1888 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1888;
                float _fma_1889 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1889;
                float _fma_1890 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1890;
                float _fma_1891 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1891;
                float _fma_1892 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1892;
                float _fma_1893 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1893;
                float _fma_1894 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1894;
                float _fma_1895 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1895;
                float _fma_1896 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1896;
                float _fma_1897 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1897;
                float _fma_1898 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1898;
                float _fma_1899 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1899;
                float _fma_1900 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1900;
                float _fma_1901 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1901;
                float _fma_1902 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1902;
                float _fma_1903 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1903;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 13 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 13) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 13 >> 2 & 1) << 4)) * 4)));
                float _fma_1904 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1904;
                float _fma_1905 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1905;
                float _fma_1906 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1906;
                float _fma_1907 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1907;
                float _fma_1908 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1908;
                float _fma_1909 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1909;
                float _fma_1910 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1910;
                float _fma_1911 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1911;
                float _fma_1912 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1912;
                float _fma_1913 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1913;
                float _fma_1914 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1914;
                float _fma_1915 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1915;
                float _fma_1916 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1916;
                float _fma_1917 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1917;
                float _fma_1918 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1918;
                float _fma_1919 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1919;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 13) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 13 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 13) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 13 >> 2 & 1) << 4)) * 4)));
                float _fma_1920 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1920;
                float _fma_1921 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1921;
                float _fma_1922 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1922;
                float _fma_1923 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1923;
                float _fma_1924 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1924;
                float _fma_1925 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1925;
                float _fma_1926 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1926;
                float _fma_1927 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1927;
                float _fma_1928 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1928;
                float _fma_1929 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1929;
                float _fma_1930 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1930;
                float _fma_1931 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1931;
                float _fma_1932 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1932;
                float _fma_1933 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1933;
                float _fma_1934 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1934;
                float _fma_1935 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1935;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 13) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 13 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 13) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 13 >> 2 & 1) << 4)) * 4)));
                float _fma_1936 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1936;
                float _fma_1937 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1937;
                float _fma_1938 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1938;
                float _fma_1939 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1939;
                float _fma_1940 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1940;
                float _fma_1941 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1941;
                float _fma_1942 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1942;
                float _fma_1943 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1943;
                float _fma_1944 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1944;
                float _fma_1945 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1945;
                float _fma_1946 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1946;
                float _fma_1947 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1947;
                float _fma_1948 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1948;
                float _fma_1949 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1949;
                float _fma_1950 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1950;
                float _fma_1951 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1951;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 14 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 14) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 14 >> 2 & 1) << 4)) * 4)));
                float _fma_1952 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1952;
                float _fma_1953 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1953;
                float _fma_1954 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1954;
                float _fma_1955 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1955;
                float _fma_1956 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1956;
                float _fma_1957 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1957;
                float _fma_1958 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1958;
                float _fma_1959 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1959;
                float _fma_1960 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1960;
                float _fma_1961 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1961;
                float _fma_1962 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1962;
                float _fma_1963 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1963;
                float _fma_1964 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1964;
                float _fma_1965 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1965;
                float _fma_1966 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1966;
                float _fma_1967 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1967;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 14) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 14 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 14) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 14 >> 2 & 1) << 4)) * 4)));
                float _fma_1968 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1968;
                float _fma_1969 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1969;
                float _fma_1970 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1970;
                float _fma_1971 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1971;
                float _fma_1972 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1972;
                float _fma_1973 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1973;
                float _fma_1974 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1974;
                float _fma_1975 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1975;
                float _fma_1976 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1976;
                float _fma_1977 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1977;
                float _fma_1978 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1978;
                float _fma_1979 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1979;
                float _fma_1980 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1980;
                float _fma_1981 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1981;
                float _fma_1982 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1982;
                float _fma_1983 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1983;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 14) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 14 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 14) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 14 >> 2 & 1) << 4)) * 4)));
                float _fma_1984 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_1984;
                float _fma_1985 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_1985;
                float _fma_1986 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_1986;
                float _fma_1987 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_1987;
                float _fma_1988 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_1988;
                float _fma_1989 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_1989;
                float _fma_1990 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_1990;
                float _fma_1991 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_1991;
                float _fma_1992 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_1992;
                float _fma_1993 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_1993;
                float _fma_1994 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_1994;
                float _fma_1995 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_1995;
                float _fma_1996 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_1996;
                float _fma_1997 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_1997;
                float _fma_1998 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_1998;
                float _fma_1999 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_1999;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ (warp_1 * 16 + 15 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)(((warp_1 * 16 + 15) * 64 + (warp_1 * 16 + cg * 4 ^ (warp_1 * 16 + 15 >> 2 & 1) << 4)) * 4)));
                float _fma_2000 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_2000;
                float _fma_2001 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_2001;
                float _fma_2002 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_2002;
                float _fma_2003 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_2003;
                float _fma_2004 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_2004;
                float _fma_2005 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_2005;
                float _fma_2006 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_2006;
                float _fma_2007 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_2007;
                float _fma_2008 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_2008;
                float _fma_2009 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_2009;
                float _fma_2010 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_2010;
                float _fma_2011 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_2011;
                float _fma_2012 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_2012;
                float _fma_2013 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_2013;
                float _fma_2014 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_2014;
                float _fma_2015 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_2015;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 1) * 16 + 15) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 1) * 16 + 15 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 1) * 16 + 15) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 1) * 16 + 15 >> 2 & 1) << 4)) * 4)));
                float _fma_2016 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_2016;
                float _fma_2017 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_2017;
                float _fma_2018 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_2018;
                float _fma_2019 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_2019;
                float _fma_2020 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_2020;
                float _fma_2021 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_2021;
                float _fma_2022 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_2022;
                float _fma_2023 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_2023;
                float _fma_2024 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_2024;
                float _fma_2025 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_2025;
                float _fma_2026 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_2026;
                float _fma_2027 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_2027;
                float _fma_2028 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_2028;
                float _fma_2029 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_2029;
                float _fma_2030 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_2030;
                float _fma_2031 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_2031;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&lv[0])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&lv[(0) + 3]))
                    : "r"(s_lt_addr + (unsigned int)((((warp_1 + 2) * 16 + 15) * 64 + ((warp_1 + 3) * 16 + rg * 4 ^ ((warp_1 + 2) * 16 + 15 >> 1 & 3) << 3)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xv4[0])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xv4[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 2) * 16 + 15) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 2) * 16 + 15 >> 2 & 1) << 4)) * 4)));
                float _fma_2032 = __fmaf_rn(lv[0], xv4[0], acc16[0]);
                acc16[0] = _fma_2032;
                float _fma_2033 = __fmaf_rn(lv[0], xv4[1], acc16[1]);
                acc16[1] = _fma_2033;
                float _fma_2034 = __fmaf_rn(lv[0], xv4[2], acc16[2]);
                acc16[2] = _fma_2034;
                float _fma_2035 = __fmaf_rn(lv[0], xv4[3], acc16[3]);
                acc16[3] = _fma_2035;
                float _fma_2036 = __fmaf_rn(lv[1], xv4[0], acc16[4]);
                acc16[4] = _fma_2036;
                float _fma_2037 = __fmaf_rn(lv[1], xv4[1], acc16[5]);
                acc16[5] = _fma_2037;
                float _fma_2038 = __fmaf_rn(lv[1], xv4[2], acc16[6]);
                acc16[6] = _fma_2038;
                float _fma_2039 = __fmaf_rn(lv[1], xv4[3], acc16[7]);
                acc16[7] = _fma_2039;
                float _fma_2040 = __fmaf_rn(lv[2], xv4[0], acc16[8]);
                acc16[8] = _fma_2040;
                float _fma_2041 = __fmaf_rn(lv[2], xv4[1], acc16[9]);
                acc16[9] = _fma_2041;
                float _fma_2042 = __fmaf_rn(lv[2], xv4[2], acc16[10]);
                acc16[10] = _fma_2042;
                float _fma_2043 = __fmaf_rn(lv[2], xv4[3], acc16[11]);
                acc16[11] = _fma_2043;
                float _fma_2044 = __fmaf_rn(lv[3], xv4[0], acc16[12]);
                acc16[12] = _fma_2044;
                float _fma_2045 = __fmaf_rn(lv[3], xv4[1], acc16[13]);
                acc16[13] = _fma_2045;
                float _fma_2046 = __fmaf_rn(lv[3], xv4[2], acc16[14]);
                acc16[14] = _fma_2046;
                float _fma_2047 = __fmaf_rn(lv[3], xv4[3], acc16[15]);
                acc16[15] = _fma_2047;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_t_addr + (unsigned int)((warp_1 * 320 + rg * 4 * 20 + cg * 4) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[0])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (rg * 4 + 1) * 20 + cg * 4) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[4])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (rg * 4 + 2) * 20 + cg * 4) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[8])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_t_addr + (unsigned int)((warp_1 * 320 + (rg * 4 + 3) * 20 + cg * 4) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[12])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 3])));
            }
        }
        __syncthreads();
        if (lane_0 < 16) {
            if (warp_1 < 1) {
                acc16[0] = 0.0f;
                acc16[1] = 0.0f;
                acc16[2] = 0.0f;
                acc16[3] = 0.0f;
                acc16[4] = 0.0f;
                acc16[5] = 0.0f;
                acc16[6] = 0.0f;
                acc16[7] = 0.0f;
                acc16[8] = 0.0f;
                acc16[9] = 0.0f;
                acc16[10] = 0.0f;
                acc16[11] = 0.0f;
                acc16[12] = 0.0f;
                acc16[13] = 0.0f;
                acc16[14] = 0.0f;
                acc16[15] = 0.0f;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4) * 64 + ((warp_1 + 3) * 16 ^ ((warp_1 + 3) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 1) * 64 + ((warp_1 + 3) * 16 ^ ((warp_1 + 3) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[8])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 2) * 64 + ((warp_1 + 3) * 16 ^ ((warp_1 + 3) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[12])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 3) * 64 + ((warp_1 + 3) * 16 ^ ((warp_1 + 3) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 20 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 40 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 60 + cg * 4) * 4)));
                float _fma_2048 = __fmaf_rn(xi16[0], tv16[0], acc16[0]);
                acc16[0] = _fma_2048;
                float _fma_2049 = __fmaf_rn(xi16[0], tv16[1], acc16[1]);
                acc16[1] = _fma_2049;
                float _fma_2050 = __fmaf_rn(xi16[0], tv16[2], acc16[2]);
                acc16[2] = _fma_2050;
                float _fma_2051 = __fmaf_rn(xi16[0], tv16[3], acc16[3]);
                acc16[3] = _fma_2051;
                float _fma_2052 = __fmaf_rn(xi16[4], tv16[0], acc16[4]);
                acc16[4] = _fma_2052;
                float _fma_2053 = __fmaf_rn(xi16[4], tv16[1], acc16[5]);
                acc16[5] = _fma_2053;
                float _fma_2054 = __fmaf_rn(xi16[4], tv16[2], acc16[6]);
                acc16[6] = _fma_2054;
                float _fma_2055 = __fmaf_rn(xi16[4], tv16[3], acc16[7]);
                acc16[7] = _fma_2055;
                float _fma_2056 = __fmaf_rn(xi16[8], tv16[0], acc16[8]);
                acc16[8] = _fma_2056;
                float _fma_2057 = __fmaf_rn(xi16[8], tv16[1], acc16[9]);
                acc16[9] = _fma_2057;
                float _fma_2058 = __fmaf_rn(xi16[8], tv16[2], acc16[10]);
                acc16[10] = _fma_2058;
                float _fma_2059 = __fmaf_rn(xi16[8], tv16[3], acc16[11]);
                acc16[11] = _fma_2059;
                float _fma_2060 = __fmaf_rn(xi16[12], tv16[0], acc16[12]);
                acc16[12] = _fma_2060;
                float _fma_2061 = __fmaf_rn(xi16[12], tv16[1], acc16[13]);
                acc16[13] = _fma_2061;
                float _fma_2062 = __fmaf_rn(xi16[12], tv16[2], acc16[14]);
                acc16[14] = _fma_2062;
                float _fma_2063 = __fmaf_rn(xi16[12], tv16[3], acc16[15]);
                acc16[15] = _fma_2063;
                float _fma_2064 = __fmaf_rn(xi16[1], tv16[4], acc16[0]);
                acc16[0] = _fma_2064;
                float _fma_2065 = __fmaf_rn(xi16[1], tv16[5], acc16[1]);
                acc16[1] = _fma_2065;
                float _fma_2066 = __fmaf_rn(xi16[1], tv16[6], acc16[2]);
                acc16[2] = _fma_2066;
                float _fma_2067 = __fmaf_rn(xi16[1], tv16[7], acc16[3]);
                acc16[3] = _fma_2067;
                float _fma_2068 = __fmaf_rn(xi16[5], tv16[4], acc16[4]);
                acc16[4] = _fma_2068;
                float _fma_2069 = __fmaf_rn(xi16[5], tv16[5], acc16[5]);
                acc16[5] = _fma_2069;
                float _fma_2070 = __fmaf_rn(xi16[5], tv16[6], acc16[6]);
                acc16[6] = _fma_2070;
                float _fma_2071 = __fmaf_rn(xi16[5], tv16[7], acc16[7]);
                acc16[7] = _fma_2071;
                float _fma_2072 = __fmaf_rn(xi16[9], tv16[4], acc16[8]);
                acc16[8] = _fma_2072;
                float _fma_2073 = __fmaf_rn(xi16[9], tv16[5], acc16[9]);
                acc16[9] = _fma_2073;
                float _fma_2074 = __fmaf_rn(xi16[9], tv16[6], acc16[10]);
                acc16[10] = _fma_2074;
                float _fma_2075 = __fmaf_rn(xi16[9], tv16[7], acc16[11]);
                acc16[11] = _fma_2075;
                float _fma_2076 = __fmaf_rn(xi16[13], tv16[4], acc16[12]);
                acc16[12] = _fma_2076;
                float _fma_2077 = __fmaf_rn(xi16[13], tv16[5], acc16[13]);
                acc16[13] = _fma_2077;
                float _fma_2078 = __fmaf_rn(xi16[13], tv16[6], acc16[14]);
                acc16[14] = _fma_2078;
                float _fma_2079 = __fmaf_rn(xi16[13], tv16[7], acc16[15]);
                acc16[15] = _fma_2079;
                float _fma_2080 = __fmaf_rn(xi16[2], tv16[8], acc16[0]);
                acc16[0] = _fma_2080;
                float _fma_2081 = __fmaf_rn(xi16[2], tv16[9], acc16[1]);
                acc16[1] = _fma_2081;
                float _fma_2082 = __fmaf_rn(xi16[2], tv16[10], acc16[2]);
                acc16[2] = _fma_2082;
                float _fma_2083 = __fmaf_rn(xi16[2], tv16[11], acc16[3]);
                acc16[3] = _fma_2083;
                float _fma_2084 = __fmaf_rn(xi16[6], tv16[8], acc16[4]);
                acc16[4] = _fma_2084;
                float _fma_2085 = __fmaf_rn(xi16[6], tv16[9], acc16[5]);
                acc16[5] = _fma_2085;
                float _fma_2086 = __fmaf_rn(xi16[6], tv16[10], acc16[6]);
                acc16[6] = _fma_2086;
                float _fma_2087 = __fmaf_rn(xi16[6], tv16[11], acc16[7]);
                acc16[7] = _fma_2087;
                float _fma_2088 = __fmaf_rn(xi16[10], tv16[8], acc16[8]);
                acc16[8] = _fma_2088;
                float _fma_2089 = __fmaf_rn(xi16[10], tv16[9], acc16[9]);
                acc16[9] = _fma_2089;
                float _fma_2090 = __fmaf_rn(xi16[10], tv16[10], acc16[10]);
                acc16[10] = _fma_2090;
                float _fma_2091 = __fmaf_rn(xi16[10], tv16[11], acc16[11]);
                acc16[11] = _fma_2091;
                float _fma_2092 = __fmaf_rn(xi16[14], tv16[8], acc16[12]);
                acc16[12] = _fma_2092;
                float _fma_2093 = __fmaf_rn(xi16[14], tv16[9], acc16[13]);
                acc16[13] = _fma_2093;
                float _fma_2094 = __fmaf_rn(xi16[14], tv16[10], acc16[14]);
                acc16[14] = _fma_2094;
                float _fma_2095 = __fmaf_rn(xi16[14], tv16[11], acc16[15]);
                acc16[15] = _fma_2095;
                float _fma_2096 = __fmaf_rn(xi16[3], tv16[12], acc16[0]);
                acc16[0] = _fma_2096;
                float _fma_2097 = __fmaf_rn(xi16[3], tv16[13], acc16[1]);
                acc16[1] = _fma_2097;
                float _fma_2098 = __fmaf_rn(xi16[3], tv16[14], acc16[2]);
                acc16[2] = _fma_2098;
                float _fma_2099 = __fmaf_rn(xi16[3], tv16[15], acc16[3]);
                acc16[3] = _fma_2099;
                float _fma_2100 = __fmaf_rn(xi16[7], tv16[12], acc16[4]);
                acc16[4] = _fma_2100;
                float _fma_2101 = __fmaf_rn(xi16[7], tv16[13], acc16[5]);
                acc16[5] = _fma_2101;
                float _fma_2102 = __fmaf_rn(xi16[7], tv16[14], acc16[6]);
                acc16[6] = _fma_2102;
                float _fma_2103 = __fmaf_rn(xi16[7], tv16[15], acc16[7]);
                acc16[7] = _fma_2103;
                float _fma_2104 = __fmaf_rn(xi16[11], tv16[12], acc16[8]);
                acc16[8] = _fma_2104;
                float _fma_2105 = __fmaf_rn(xi16[11], tv16[13], acc16[9]);
                acc16[9] = _fma_2105;
                float _fma_2106 = __fmaf_rn(xi16[11], tv16[14], acc16[10]);
                acc16[10] = _fma_2106;
                float _fma_2107 = __fmaf_rn(xi16[11], tv16[15], acc16[11]);
                acc16[11] = _fma_2107;
                float _fma_2108 = __fmaf_rn(xi16[15], tv16[12], acc16[12]);
                acc16[12] = _fma_2108;
                float _fma_2109 = __fmaf_rn(xi16[15], tv16[13], acc16[13]);
                acc16[13] = _fma_2109;
                float _fma_2110 = __fmaf_rn(xi16[15], tv16[14], acc16[14]);
                acc16[14] = _fma_2110;
                float _fma_2111 = __fmaf_rn(xi16[15], tv16[15], acc16[15]);
                acc16[15] = _fma_2111;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4) * 64 + ((warp_1 + 3) * 16 + 4 ^ ((warp_1 + 3) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 1) * 64 + ((warp_1 + 3) * 16 + 4 ^ ((warp_1 + 3) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[8])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 2) * 64 + ((warp_1 + 3) * 16 + 4 ^ ((warp_1 + 3) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[12])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 3) * 64 + ((warp_1 + 3) * 16 + 4 ^ ((warp_1 + 3) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 80 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 100 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 120 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 140 + cg * 4) * 4)));
                float _fma_2112 = __fmaf_rn(xi16[0], tv16[0], acc16[0]);
                acc16[0] = _fma_2112;
                float _fma_2113 = __fmaf_rn(xi16[0], tv16[1], acc16[1]);
                acc16[1] = _fma_2113;
                float _fma_2114 = __fmaf_rn(xi16[0], tv16[2], acc16[2]);
                acc16[2] = _fma_2114;
                float _fma_2115 = __fmaf_rn(xi16[0], tv16[3], acc16[3]);
                acc16[3] = _fma_2115;
                float _fma_2116 = __fmaf_rn(xi16[4], tv16[0], acc16[4]);
                acc16[4] = _fma_2116;
                float _fma_2117 = __fmaf_rn(xi16[4], tv16[1], acc16[5]);
                acc16[5] = _fma_2117;
                float _fma_2118 = __fmaf_rn(xi16[4], tv16[2], acc16[6]);
                acc16[6] = _fma_2118;
                float _fma_2119 = __fmaf_rn(xi16[4], tv16[3], acc16[7]);
                acc16[7] = _fma_2119;
                float _fma_2120 = __fmaf_rn(xi16[8], tv16[0], acc16[8]);
                acc16[8] = _fma_2120;
                float _fma_2121 = __fmaf_rn(xi16[8], tv16[1], acc16[9]);
                acc16[9] = _fma_2121;
                float _fma_2122 = __fmaf_rn(xi16[8], tv16[2], acc16[10]);
                acc16[10] = _fma_2122;
                float _fma_2123 = __fmaf_rn(xi16[8], tv16[3], acc16[11]);
                acc16[11] = _fma_2123;
                float _fma_2124 = __fmaf_rn(xi16[12], tv16[0], acc16[12]);
                acc16[12] = _fma_2124;
                float _fma_2125 = __fmaf_rn(xi16[12], tv16[1], acc16[13]);
                acc16[13] = _fma_2125;
                float _fma_2126 = __fmaf_rn(xi16[12], tv16[2], acc16[14]);
                acc16[14] = _fma_2126;
                float _fma_2127 = __fmaf_rn(xi16[12], tv16[3], acc16[15]);
                acc16[15] = _fma_2127;
                float _fma_2128 = __fmaf_rn(xi16[1], tv16[4], acc16[0]);
                acc16[0] = _fma_2128;
                float _fma_2129 = __fmaf_rn(xi16[1], tv16[5], acc16[1]);
                acc16[1] = _fma_2129;
                float _fma_2130 = __fmaf_rn(xi16[1], tv16[6], acc16[2]);
                acc16[2] = _fma_2130;
                float _fma_2131 = __fmaf_rn(xi16[1], tv16[7], acc16[3]);
                acc16[3] = _fma_2131;
                float _fma_2132 = __fmaf_rn(xi16[5], tv16[4], acc16[4]);
                acc16[4] = _fma_2132;
                float _fma_2133 = __fmaf_rn(xi16[5], tv16[5], acc16[5]);
                acc16[5] = _fma_2133;
                float _fma_2134 = __fmaf_rn(xi16[5], tv16[6], acc16[6]);
                acc16[6] = _fma_2134;
                float _fma_2135 = __fmaf_rn(xi16[5], tv16[7], acc16[7]);
                acc16[7] = _fma_2135;
                float _fma_2136 = __fmaf_rn(xi16[9], tv16[4], acc16[8]);
                acc16[8] = _fma_2136;
                float _fma_2137 = __fmaf_rn(xi16[9], tv16[5], acc16[9]);
                acc16[9] = _fma_2137;
                float _fma_2138 = __fmaf_rn(xi16[9], tv16[6], acc16[10]);
                acc16[10] = _fma_2138;
                float _fma_2139 = __fmaf_rn(xi16[9], tv16[7], acc16[11]);
                acc16[11] = _fma_2139;
                float _fma_2140 = __fmaf_rn(xi16[13], tv16[4], acc16[12]);
                acc16[12] = _fma_2140;
                float _fma_2141 = __fmaf_rn(xi16[13], tv16[5], acc16[13]);
                acc16[13] = _fma_2141;
                float _fma_2142 = __fmaf_rn(xi16[13], tv16[6], acc16[14]);
                acc16[14] = _fma_2142;
                float _fma_2143 = __fmaf_rn(xi16[13], tv16[7], acc16[15]);
                acc16[15] = _fma_2143;
                float _fma_2144 = __fmaf_rn(xi16[2], tv16[8], acc16[0]);
                acc16[0] = _fma_2144;
                float _fma_2145 = __fmaf_rn(xi16[2], tv16[9], acc16[1]);
                acc16[1] = _fma_2145;
                float _fma_2146 = __fmaf_rn(xi16[2], tv16[10], acc16[2]);
                acc16[2] = _fma_2146;
                float _fma_2147 = __fmaf_rn(xi16[2], tv16[11], acc16[3]);
                acc16[3] = _fma_2147;
                float _fma_2148 = __fmaf_rn(xi16[6], tv16[8], acc16[4]);
                acc16[4] = _fma_2148;
                float _fma_2149 = __fmaf_rn(xi16[6], tv16[9], acc16[5]);
                acc16[5] = _fma_2149;
                float _fma_2150 = __fmaf_rn(xi16[6], tv16[10], acc16[6]);
                acc16[6] = _fma_2150;
                float _fma_2151 = __fmaf_rn(xi16[6], tv16[11], acc16[7]);
                acc16[7] = _fma_2151;
                float _fma_2152 = __fmaf_rn(xi16[10], tv16[8], acc16[8]);
                acc16[8] = _fma_2152;
                float _fma_2153 = __fmaf_rn(xi16[10], tv16[9], acc16[9]);
                acc16[9] = _fma_2153;
                float _fma_2154 = __fmaf_rn(xi16[10], tv16[10], acc16[10]);
                acc16[10] = _fma_2154;
                float _fma_2155 = __fmaf_rn(xi16[10], tv16[11], acc16[11]);
                acc16[11] = _fma_2155;
                float _fma_2156 = __fmaf_rn(xi16[14], tv16[8], acc16[12]);
                acc16[12] = _fma_2156;
                float _fma_2157 = __fmaf_rn(xi16[14], tv16[9], acc16[13]);
                acc16[13] = _fma_2157;
                float _fma_2158 = __fmaf_rn(xi16[14], tv16[10], acc16[14]);
                acc16[14] = _fma_2158;
                float _fma_2159 = __fmaf_rn(xi16[14], tv16[11], acc16[15]);
                acc16[15] = _fma_2159;
                float _fma_2160 = __fmaf_rn(xi16[3], tv16[12], acc16[0]);
                acc16[0] = _fma_2160;
                float _fma_2161 = __fmaf_rn(xi16[3], tv16[13], acc16[1]);
                acc16[1] = _fma_2161;
                float _fma_2162 = __fmaf_rn(xi16[3], tv16[14], acc16[2]);
                acc16[2] = _fma_2162;
                float _fma_2163 = __fmaf_rn(xi16[3], tv16[15], acc16[3]);
                acc16[3] = _fma_2163;
                float _fma_2164 = __fmaf_rn(xi16[7], tv16[12], acc16[4]);
                acc16[4] = _fma_2164;
                float _fma_2165 = __fmaf_rn(xi16[7], tv16[13], acc16[5]);
                acc16[5] = _fma_2165;
                float _fma_2166 = __fmaf_rn(xi16[7], tv16[14], acc16[6]);
                acc16[6] = _fma_2166;
                float _fma_2167 = __fmaf_rn(xi16[7], tv16[15], acc16[7]);
                acc16[7] = _fma_2167;
                float _fma_2168 = __fmaf_rn(xi16[11], tv16[12], acc16[8]);
                acc16[8] = _fma_2168;
                float _fma_2169 = __fmaf_rn(xi16[11], tv16[13], acc16[9]);
                acc16[9] = _fma_2169;
                float _fma_2170 = __fmaf_rn(xi16[11], tv16[14], acc16[10]);
                acc16[10] = _fma_2170;
                float _fma_2171 = __fmaf_rn(xi16[11], tv16[15], acc16[11]);
                acc16[11] = _fma_2171;
                float _fma_2172 = __fmaf_rn(xi16[15], tv16[12], acc16[12]);
                acc16[12] = _fma_2172;
                float _fma_2173 = __fmaf_rn(xi16[15], tv16[13], acc16[13]);
                acc16[13] = _fma_2173;
                float _fma_2174 = __fmaf_rn(xi16[15], tv16[14], acc16[14]);
                acc16[14] = _fma_2174;
                float _fma_2175 = __fmaf_rn(xi16[15], tv16[15], acc16[15]);
                acc16[15] = _fma_2175;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4) * 64 + ((warp_1 + 3) * 16 + 8 ^ ((warp_1 + 3) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 1) * 64 + ((warp_1 + 3) * 16 + 8 ^ ((warp_1 + 3) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[8])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 2) * 64 + ((warp_1 + 3) * 16 + 8 ^ ((warp_1 + 3) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[12])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 3) * 64 + ((warp_1 + 3) * 16 + 8 ^ ((warp_1 + 3) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 160 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 180 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 200 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 220 + cg * 4) * 4)));
                float _fma_2176 = __fmaf_rn(xi16[0], tv16[0], acc16[0]);
                acc16[0] = _fma_2176;
                float _fma_2177 = __fmaf_rn(xi16[0], tv16[1], acc16[1]);
                acc16[1] = _fma_2177;
                float _fma_2178 = __fmaf_rn(xi16[0], tv16[2], acc16[2]);
                acc16[2] = _fma_2178;
                float _fma_2179 = __fmaf_rn(xi16[0], tv16[3], acc16[3]);
                acc16[3] = _fma_2179;
                float _fma_2180 = __fmaf_rn(xi16[4], tv16[0], acc16[4]);
                acc16[4] = _fma_2180;
                float _fma_2181 = __fmaf_rn(xi16[4], tv16[1], acc16[5]);
                acc16[5] = _fma_2181;
                float _fma_2182 = __fmaf_rn(xi16[4], tv16[2], acc16[6]);
                acc16[6] = _fma_2182;
                float _fma_2183 = __fmaf_rn(xi16[4], tv16[3], acc16[7]);
                acc16[7] = _fma_2183;
                float _fma_2184 = __fmaf_rn(xi16[8], tv16[0], acc16[8]);
                acc16[8] = _fma_2184;
                float _fma_2185 = __fmaf_rn(xi16[8], tv16[1], acc16[9]);
                acc16[9] = _fma_2185;
                float _fma_2186 = __fmaf_rn(xi16[8], tv16[2], acc16[10]);
                acc16[10] = _fma_2186;
                float _fma_2187 = __fmaf_rn(xi16[8], tv16[3], acc16[11]);
                acc16[11] = _fma_2187;
                float _fma_2188 = __fmaf_rn(xi16[12], tv16[0], acc16[12]);
                acc16[12] = _fma_2188;
                float _fma_2189 = __fmaf_rn(xi16[12], tv16[1], acc16[13]);
                acc16[13] = _fma_2189;
                float _fma_2190 = __fmaf_rn(xi16[12], tv16[2], acc16[14]);
                acc16[14] = _fma_2190;
                float _fma_2191 = __fmaf_rn(xi16[12], tv16[3], acc16[15]);
                acc16[15] = _fma_2191;
                float _fma_2192 = __fmaf_rn(xi16[1], tv16[4], acc16[0]);
                acc16[0] = _fma_2192;
                float _fma_2193 = __fmaf_rn(xi16[1], tv16[5], acc16[1]);
                acc16[1] = _fma_2193;
                float _fma_2194 = __fmaf_rn(xi16[1], tv16[6], acc16[2]);
                acc16[2] = _fma_2194;
                float _fma_2195 = __fmaf_rn(xi16[1], tv16[7], acc16[3]);
                acc16[3] = _fma_2195;
                float _fma_2196 = __fmaf_rn(xi16[5], tv16[4], acc16[4]);
                acc16[4] = _fma_2196;
                float _fma_2197 = __fmaf_rn(xi16[5], tv16[5], acc16[5]);
                acc16[5] = _fma_2197;
                float _fma_2198 = __fmaf_rn(xi16[5], tv16[6], acc16[6]);
                acc16[6] = _fma_2198;
                float _fma_2199 = __fmaf_rn(xi16[5], tv16[7], acc16[7]);
                acc16[7] = _fma_2199;
                float _fma_2200 = __fmaf_rn(xi16[9], tv16[4], acc16[8]);
                acc16[8] = _fma_2200;
                float _fma_2201 = __fmaf_rn(xi16[9], tv16[5], acc16[9]);
                acc16[9] = _fma_2201;
                float _fma_2202 = __fmaf_rn(xi16[9], tv16[6], acc16[10]);
                acc16[10] = _fma_2202;
                float _fma_2203 = __fmaf_rn(xi16[9], tv16[7], acc16[11]);
                acc16[11] = _fma_2203;
                float _fma_2204 = __fmaf_rn(xi16[13], tv16[4], acc16[12]);
                acc16[12] = _fma_2204;
                float _fma_2205 = __fmaf_rn(xi16[13], tv16[5], acc16[13]);
                acc16[13] = _fma_2205;
                float _fma_2206 = __fmaf_rn(xi16[13], tv16[6], acc16[14]);
                acc16[14] = _fma_2206;
                float _fma_2207 = __fmaf_rn(xi16[13], tv16[7], acc16[15]);
                acc16[15] = _fma_2207;
                float _fma_2208 = __fmaf_rn(xi16[2], tv16[8], acc16[0]);
                acc16[0] = _fma_2208;
                float _fma_2209 = __fmaf_rn(xi16[2], tv16[9], acc16[1]);
                acc16[1] = _fma_2209;
                float _fma_2210 = __fmaf_rn(xi16[2], tv16[10], acc16[2]);
                acc16[2] = _fma_2210;
                float _fma_2211 = __fmaf_rn(xi16[2], tv16[11], acc16[3]);
                acc16[3] = _fma_2211;
                float _fma_2212 = __fmaf_rn(xi16[6], tv16[8], acc16[4]);
                acc16[4] = _fma_2212;
                float _fma_2213 = __fmaf_rn(xi16[6], tv16[9], acc16[5]);
                acc16[5] = _fma_2213;
                float _fma_2214 = __fmaf_rn(xi16[6], tv16[10], acc16[6]);
                acc16[6] = _fma_2214;
                float _fma_2215 = __fmaf_rn(xi16[6], tv16[11], acc16[7]);
                acc16[7] = _fma_2215;
                float _fma_2216 = __fmaf_rn(xi16[10], tv16[8], acc16[8]);
                acc16[8] = _fma_2216;
                float _fma_2217 = __fmaf_rn(xi16[10], tv16[9], acc16[9]);
                acc16[9] = _fma_2217;
                float _fma_2218 = __fmaf_rn(xi16[10], tv16[10], acc16[10]);
                acc16[10] = _fma_2218;
                float _fma_2219 = __fmaf_rn(xi16[10], tv16[11], acc16[11]);
                acc16[11] = _fma_2219;
                float _fma_2220 = __fmaf_rn(xi16[14], tv16[8], acc16[12]);
                acc16[12] = _fma_2220;
                float _fma_2221 = __fmaf_rn(xi16[14], tv16[9], acc16[13]);
                acc16[13] = _fma_2221;
                float _fma_2222 = __fmaf_rn(xi16[14], tv16[10], acc16[14]);
                acc16[14] = _fma_2222;
                float _fma_2223 = __fmaf_rn(xi16[14], tv16[11], acc16[15]);
                acc16[15] = _fma_2223;
                float _fma_2224 = __fmaf_rn(xi16[3], tv16[12], acc16[0]);
                acc16[0] = _fma_2224;
                float _fma_2225 = __fmaf_rn(xi16[3], tv16[13], acc16[1]);
                acc16[1] = _fma_2225;
                float _fma_2226 = __fmaf_rn(xi16[3], tv16[14], acc16[2]);
                acc16[2] = _fma_2226;
                float _fma_2227 = __fmaf_rn(xi16[3], tv16[15], acc16[3]);
                acc16[3] = _fma_2227;
                float _fma_2228 = __fmaf_rn(xi16[7], tv16[12], acc16[4]);
                acc16[4] = _fma_2228;
                float _fma_2229 = __fmaf_rn(xi16[7], tv16[13], acc16[5]);
                acc16[5] = _fma_2229;
                float _fma_2230 = __fmaf_rn(xi16[7], tv16[14], acc16[6]);
                acc16[6] = _fma_2230;
                float _fma_2231 = __fmaf_rn(xi16[7], tv16[15], acc16[7]);
                acc16[7] = _fma_2231;
                float _fma_2232 = __fmaf_rn(xi16[11], tv16[12], acc16[8]);
                acc16[8] = _fma_2232;
                float _fma_2233 = __fmaf_rn(xi16[11], tv16[13], acc16[9]);
                acc16[9] = _fma_2233;
                float _fma_2234 = __fmaf_rn(xi16[11], tv16[14], acc16[10]);
                acc16[10] = _fma_2234;
                float _fma_2235 = __fmaf_rn(xi16[11], tv16[15], acc16[11]);
                acc16[11] = _fma_2235;
                float _fma_2236 = __fmaf_rn(xi16[15], tv16[12], acc16[12]);
                acc16[12] = _fma_2236;
                float _fma_2237 = __fmaf_rn(xi16[15], tv16[13], acc16[13]);
                acc16[13] = _fma_2237;
                float _fma_2238 = __fmaf_rn(xi16[15], tv16[14], acc16[14]);
                acc16[14] = _fma_2238;
                float _fma_2239 = __fmaf_rn(xi16[15], tv16[15], acc16[15]);
                acc16[15] = _fma_2239;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[0])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(0) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4) * 64 + ((warp_1 + 3) * 16 + 12 ^ ((warp_1 + 3) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[4])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(4) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 1) * 64 + ((warp_1 + 3) * 16 + 12 ^ ((warp_1 + 3) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[8])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(8) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 2) * 64 + ((warp_1 + 3) * 16 + 12 ^ ((warp_1 + 3) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&xi16[12])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&xi16[(12) + 3]))
                    : "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 3) * 64 + ((warp_1 + 3) * 16 + 12 ^ ((warp_1 + 3) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[0])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(0) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 240 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[4])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(4) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 260 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[8])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(8) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 280 + cg * 4) * 4)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&tv16[12])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tv16[(12) + 3]))
                    : "r"(s_t_addr + (unsigned int)((warp_1 * 320 + 300 + cg * 4) * 4)));
                float _fma_2240 = __fmaf_rn(xi16[0], tv16[0], acc16[0]);
                acc16[0] = _fma_2240;
                float _fma_2241 = __fmaf_rn(xi16[0], tv16[1], acc16[1]);
                acc16[1] = _fma_2241;
                float _fma_2242 = __fmaf_rn(xi16[0], tv16[2], acc16[2]);
                acc16[2] = _fma_2242;
                float _fma_2243 = __fmaf_rn(xi16[0], tv16[3], acc16[3]);
                acc16[3] = _fma_2243;
                float _fma_2244 = __fmaf_rn(xi16[4], tv16[0], acc16[4]);
                acc16[4] = _fma_2244;
                float _fma_2245 = __fmaf_rn(xi16[4], tv16[1], acc16[5]);
                acc16[5] = _fma_2245;
                float _fma_2246 = __fmaf_rn(xi16[4], tv16[2], acc16[6]);
                acc16[6] = _fma_2246;
                float _fma_2247 = __fmaf_rn(xi16[4], tv16[3], acc16[7]);
                acc16[7] = _fma_2247;
                float _fma_2248 = __fmaf_rn(xi16[8], tv16[0], acc16[8]);
                acc16[8] = _fma_2248;
                float _fma_2249 = __fmaf_rn(xi16[8], tv16[1], acc16[9]);
                acc16[9] = _fma_2249;
                float _fma_2250 = __fmaf_rn(xi16[8], tv16[2], acc16[10]);
                acc16[10] = _fma_2250;
                float _fma_2251 = __fmaf_rn(xi16[8], tv16[3], acc16[11]);
                acc16[11] = _fma_2251;
                float _fma_2252 = __fmaf_rn(xi16[12], tv16[0], acc16[12]);
                acc16[12] = _fma_2252;
                float _fma_2253 = __fmaf_rn(xi16[12], tv16[1], acc16[13]);
                acc16[13] = _fma_2253;
                float _fma_2254 = __fmaf_rn(xi16[12], tv16[2], acc16[14]);
                acc16[14] = _fma_2254;
                float _fma_2255 = __fmaf_rn(xi16[12], tv16[3], acc16[15]);
                acc16[15] = _fma_2255;
                float _fma_2256 = __fmaf_rn(xi16[1], tv16[4], acc16[0]);
                acc16[0] = _fma_2256;
                float _fma_2257 = __fmaf_rn(xi16[1], tv16[5], acc16[1]);
                acc16[1] = _fma_2257;
                float _fma_2258 = __fmaf_rn(xi16[1], tv16[6], acc16[2]);
                acc16[2] = _fma_2258;
                float _fma_2259 = __fmaf_rn(xi16[1], tv16[7], acc16[3]);
                acc16[3] = _fma_2259;
                float _fma_2260 = __fmaf_rn(xi16[5], tv16[4], acc16[4]);
                acc16[4] = _fma_2260;
                float _fma_2261 = __fmaf_rn(xi16[5], tv16[5], acc16[5]);
                acc16[5] = _fma_2261;
                float _fma_2262 = __fmaf_rn(xi16[5], tv16[6], acc16[6]);
                acc16[6] = _fma_2262;
                float _fma_2263 = __fmaf_rn(xi16[5], tv16[7], acc16[7]);
                acc16[7] = _fma_2263;
                float _fma_2264 = __fmaf_rn(xi16[9], tv16[4], acc16[8]);
                acc16[8] = _fma_2264;
                float _fma_2265 = __fmaf_rn(xi16[9], tv16[5], acc16[9]);
                acc16[9] = _fma_2265;
                float _fma_2266 = __fmaf_rn(xi16[9], tv16[6], acc16[10]);
                acc16[10] = _fma_2266;
                float _fma_2267 = __fmaf_rn(xi16[9], tv16[7], acc16[11]);
                acc16[11] = _fma_2267;
                float _fma_2268 = __fmaf_rn(xi16[13], tv16[4], acc16[12]);
                acc16[12] = _fma_2268;
                float _fma_2269 = __fmaf_rn(xi16[13], tv16[5], acc16[13]);
                acc16[13] = _fma_2269;
                float _fma_2270 = __fmaf_rn(xi16[13], tv16[6], acc16[14]);
                acc16[14] = _fma_2270;
                float _fma_2271 = __fmaf_rn(xi16[13], tv16[7], acc16[15]);
                acc16[15] = _fma_2271;
                float _fma_2272 = __fmaf_rn(xi16[2], tv16[8], acc16[0]);
                acc16[0] = _fma_2272;
                float _fma_2273 = __fmaf_rn(xi16[2], tv16[9], acc16[1]);
                acc16[1] = _fma_2273;
                float _fma_2274 = __fmaf_rn(xi16[2], tv16[10], acc16[2]);
                acc16[2] = _fma_2274;
                float _fma_2275 = __fmaf_rn(xi16[2], tv16[11], acc16[3]);
                acc16[3] = _fma_2275;
                float _fma_2276 = __fmaf_rn(xi16[6], tv16[8], acc16[4]);
                acc16[4] = _fma_2276;
                float _fma_2277 = __fmaf_rn(xi16[6], tv16[9], acc16[5]);
                acc16[5] = _fma_2277;
                float _fma_2278 = __fmaf_rn(xi16[6], tv16[10], acc16[6]);
                acc16[6] = _fma_2278;
                float _fma_2279 = __fmaf_rn(xi16[6], tv16[11], acc16[7]);
                acc16[7] = _fma_2279;
                float _fma_2280 = __fmaf_rn(xi16[10], tv16[8], acc16[8]);
                acc16[8] = _fma_2280;
                float _fma_2281 = __fmaf_rn(xi16[10], tv16[9], acc16[9]);
                acc16[9] = _fma_2281;
                float _fma_2282 = __fmaf_rn(xi16[10], tv16[10], acc16[10]);
                acc16[10] = _fma_2282;
                float _fma_2283 = __fmaf_rn(xi16[10], tv16[11], acc16[11]);
                acc16[11] = _fma_2283;
                float _fma_2284 = __fmaf_rn(xi16[14], tv16[8], acc16[12]);
                acc16[12] = _fma_2284;
                float _fma_2285 = __fmaf_rn(xi16[14], tv16[9], acc16[13]);
                acc16[13] = _fma_2285;
                float _fma_2286 = __fmaf_rn(xi16[14], tv16[10], acc16[14]);
                acc16[14] = _fma_2286;
                float _fma_2287 = __fmaf_rn(xi16[14], tv16[11], acc16[15]);
                acc16[15] = _fma_2287;
                float _fma_2288 = __fmaf_rn(xi16[3], tv16[12], acc16[0]);
                acc16[0] = _fma_2288;
                float _fma_2289 = __fmaf_rn(xi16[3], tv16[13], acc16[1]);
                acc16[1] = _fma_2289;
                float _fma_2290 = __fmaf_rn(xi16[3], tv16[14], acc16[2]);
                acc16[2] = _fma_2290;
                float _fma_2291 = __fmaf_rn(xi16[3], tv16[15], acc16[3]);
                acc16[3] = _fma_2291;
                float _fma_2292 = __fmaf_rn(xi16[7], tv16[12], acc16[4]);
                acc16[4] = _fma_2292;
                float _fma_2293 = __fmaf_rn(xi16[7], tv16[13], acc16[5]);
                acc16[5] = _fma_2293;
                float _fma_2294 = __fmaf_rn(xi16[7], tv16[14], acc16[6]);
                acc16[6] = _fma_2294;
                float _fma_2295 = __fmaf_rn(xi16[7], tv16[15], acc16[7]);
                acc16[7] = _fma_2295;
                float _fma_2296 = __fmaf_rn(xi16[11], tv16[12], acc16[8]);
                acc16[8] = _fma_2296;
                float _fma_2297 = __fmaf_rn(xi16[11], tv16[13], acc16[9]);
                acc16[9] = _fma_2297;
                float _fma_2298 = __fmaf_rn(xi16[11], tv16[14], acc16[10]);
                acc16[10] = _fma_2298;
                float _fma_2299 = __fmaf_rn(xi16[11], tv16[15], acc16[11]);
                acc16[11] = _fma_2299;
                float _fma_2300 = __fmaf_rn(xi16[15], tv16[12], acc16[12]);
                acc16[12] = _fma_2300;
                float _fma_2301 = __fmaf_rn(xi16[15], tv16[13], acc16[13]);
                acc16[13] = _fma_2301;
                float _fma_2302 = __fmaf_rn(xi16[15], tv16[14], acc16[14]);
                acc16[14] = _fma_2302;
                float _fma_2303 = __fmaf_rn(xi16[15], tv16[15], acc16[15]);
                acc16[15] = _fma_2303;
                acc16[0] = -acc16[0];
                acc16[1] = -acc16[1];
                acc16[2] = -acc16[2];
                acc16[3] = -acc16[3];
                acc16[4] = -acc16[4];
                acc16[5] = -acc16[5];
                acc16[6] = -acc16[6];
                acc16[7] = -acc16[7];
                acc16[8] = -acc16[8];
                acc16[9] = -acc16[9];
                acc16[10] = -acc16[10];
                acc16[11] = -acc16[11];
                acc16[12] = -acc16[12];
                acc16[13] = -acc16[13];
                acc16[14] = -acc16[14];
                acc16[15] = -acc16[15];
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 3) * 16 + rg * 4 >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[0])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(0) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 1) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 3) * 16 + rg * 4 + 1 >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[4])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(4) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 2) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 3) * 16 + rg * 4 + 2 >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[8])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(8) + 3])));
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                    "r"(s_x_addr + (unsigned int)((((warp_1 + 3) * 16 + rg * 4 + 3) * 64 + (warp_1 * 16 + cg * 4 ^ ((warp_1 + 3) * 16 + rg * 4 + 3 >> 2 & 1) << 4)) * 4)), "r"(*reinterpret_cast<uint32_t*>(&acc16[12])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 1])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 2])), "r"(*reinterpret_cast<uint32_t*>(&acc16[(12) + 3])));
            }
        }
        __syncthreads();
        float quad[4];
        unsigned int pk2[2];
        #pragma unroll
        for (int t = 0; t < 8; t++) {
            int pidx = t * 128 + tid_2;
            int a_row = pidx / 16;
            int a_col = pidx % 16 * 4;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&quad[0])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&quad[(0) + 3]))
                : "r"(s_x_addr + (unsigned int)((a_row * 64 + (a_col ^ (a_row >> 2 & 1) << 4)) * 4)));
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(quad[_lp*2 + 0], quad[_lp*2+1 + 0]));
                pk2[_lp] = *(uint32_t*)&_bf2;
            }
            asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"(s_a_addr + (unsigned int)(a_col / 64 * 8192 + (a_row * 128 + a_col % 64 * 2 ^ (a_row * 128 + a_col % 64 * 2 >> 7 & 7) << 4))), "r"(pk2[0]), "r"(pk2[1]) : "memory");
            if (a_row < n_valid) {
                long long a_index = ((long long)(tok0 + a_row) * (long long)num_v_heads + (long long)hv) * 64 + (long long)a_col;
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(quad[0 + 0], quad[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(quad[0 + 2], quad[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(A_out))[a_index]) = _pk2;
                }
            }
        }
        __syncthreads();
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + tid / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)(tid % 16 * 8))), "r"((n_valid > tid / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (128 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((n_valid > (128 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (256 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((n_valid > (256 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (384 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((n_valid > (384 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (512 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((n_valid > (512 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (640 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((n_valid > (640 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (768 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((n_valid > (768 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_v_addr + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v + (((long long)(tok0 + (896 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((n_valid > (896 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.commit_group;");
    }
    unsigned int words[4];
    float vals8[8];
    unsigned int pk4[4];
    float _exp2_1 = approx_exp2(s_g[tid_2 / 16]);
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_kn_addr + (unsigned int)(tid_2 % 16 * 8 / 64 * 8192 + (tid_2 / 16 * 128 + tid_2 % 16 * 8 % 64 * 2 ^ (tid_2 / 16 * 128 + tid_2 % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * (s_b[tid_2 / 16] * _exp2_1);
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[tid_2 / 16] * _exp2_1);
    vals8[2] = __uint_as_float(words[1] << 16) * (s_b[tid_2 / 16] * _exp2_1);
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[tid_2 / 16] * _exp2_1);
    vals8[4] = __uint_as_float(words[2] << 16) * (s_b[tid_2 / 16] * _exp2_1);
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[tid_2 / 16] * _exp2_1);
    vals8[6] = __uint_as_float(words[3] << 16) * (s_b[tid_2 / 16] * _exp2_1);
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[tid_2 / 16] * _exp2_1);
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)(tid_2 % 16 * 8 / 64 * 8192 + (tid_2 / 16 * 128 + tid_2 % 16 * 8 % 64 * 2 ^ (tid_2 / 16 * 128 + tid_2 % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    float _exp2_2 = approx_exp2(s_g[(128 + tid_2) / 16]);
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_kn_addr + (unsigned int)((128 + tid_2) % 16 * 8 / 64 * 8192 + ((128 + tid_2) / 16 * 128 + (128 + tid_2) % 16 * 8 % 64 * 2 ^ ((128 + tid_2) / 16 * 128 + (128 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * (s_b[(128 + tid_2) / 16] * _exp2_2);
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[(128 + tid_2) / 16] * _exp2_2);
    vals8[2] = __uint_as_float(words[1] << 16) * (s_b[(128 + tid_2) / 16] * _exp2_2);
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[(128 + tid_2) / 16] * _exp2_2);
    vals8[4] = __uint_as_float(words[2] << 16) * (s_b[(128 + tid_2) / 16] * _exp2_2);
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[(128 + tid_2) / 16] * _exp2_2);
    vals8[6] = __uint_as_float(words[3] << 16) * (s_b[(128 + tid_2) / 16] * _exp2_2);
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[(128 + tid_2) / 16] * _exp2_2);
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)((128 + tid_2) % 16 * 8 / 64 * 8192 + ((128 + tid_2) / 16 * 128 + (128 + tid_2) % 16 * 8 % 64 * 2 ^ ((128 + tid_2) / 16 * 128 + (128 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    float _exp2_3 = approx_exp2(s_g[(256 + tid_2) / 16]);
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_kn_addr + (unsigned int)((256 + tid_2) % 16 * 8 / 64 * 8192 + ((256 + tid_2) / 16 * 128 + (256 + tid_2) % 16 * 8 % 64 * 2 ^ ((256 + tid_2) / 16 * 128 + (256 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * (s_b[(256 + tid_2) / 16] * _exp2_3);
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[(256 + tid_2) / 16] * _exp2_3);
    vals8[2] = __uint_as_float(words[1] << 16) * (s_b[(256 + tid_2) / 16] * _exp2_3);
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[(256 + tid_2) / 16] * _exp2_3);
    vals8[4] = __uint_as_float(words[2] << 16) * (s_b[(256 + tid_2) / 16] * _exp2_3);
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[(256 + tid_2) / 16] * _exp2_3);
    vals8[6] = __uint_as_float(words[3] << 16) * (s_b[(256 + tid_2) / 16] * _exp2_3);
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[(256 + tid_2) / 16] * _exp2_3);
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)((256 + tid_2) % 16 * 8 / 64 * 8192 + ((256 + tid_2) / 16 * 128 + (256 + tid_2) % 16 * 8 % 64 * 2 ^ ((256 + tid_2) / 16 * 128 + (256 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    float _exp2_4 = approx_exp2(s_g[(384 + tid_2) / 16]);
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_kn_addr + (unsigned int)((384 + tid_2) % 16 * 8 / 64 * 8192 + ((384 + tid_2) / 16 * 128 + (384 + tid_2) % 16 * 8 % 64 * 2 ^ ((384 + tid_2) / 16 * 128 + (384 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * (s_b[(384 + tid_2) / 16] * _exp2_4);
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[(384 + tid_2) / 16] * _exp2_4);
    vals8[2] = __uint_as_float(words[1] << 16) * (s_b[(384 + tid_2) / 16] * _exp2_4);
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[(384 + tid_2) / 16] * _exp2_4);
    vals8[4] = __uint_as_float(words[2] << 16) * (s_b[(384 + tid_2) / 16] * _exp2_4);
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[(384 + tid_2) / 16] * _exp2_4);
    vals8[6] = __uint_as_float(words[3] << 16) * (s_b[(384 + tid_2) / 16] * _exp2_4);
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[(384 + tid_2) / 16] * _exp2_4);
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)((384 + tid_2) % 16 * 8 / 64 * 8192 + ((384 + tid_2) / 16 * 128 + (384 + tid_2) % 16 * 8 % 64 * 2 ^ ((384 + tid_2) / 16 * 128 + (384 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    float _exp2_5 = approx_exp2(s_g[(512 + tid_2) / 16]);
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_kn_addr + (unsigned int)((512 + tid_2) % 16 * 8 / 64 * 8192 + ((512 + tid_2) / 16 * 128 + (512 + tid_2) % 16 * 8 % 64 * 2 ^ ((512 + tid_2) / 16 * 128 + (512 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * (s_b[(512 + tid_2) / 16] * _exp2_5);
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[(512 + tid_2) / 16] * _exp2_5);
    vals8[2] = __uint_as_float(words[1] << 16) * (s_b[(512 + tid_2) / 16] * _exp2_5);
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[(512 + tid_2) / 16] * _exp2_5);
    vals8[4] = __uint_as_float(words[2] << 16) * (s_b[(512 + tid_2) / 16] * _exp2_5);
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[(512 + tid_2) / 16] * _exp2_5);
    vals8[6] = __uint_as_float(words[3] << 16) * (s_b[(512 + tid_2) / 16] * _exp2_5);
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[(512 + tid_2) / 16] * _exp2_5);
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)((512 + tid_2) % 16 * 8 / 64 * 8192 + ((512 + tid_2) / 16 * 128 + (512 + tid_2) % 16 * 8 % 64 * 2 ^ ((512 + tid_2) / 16 * 128 + (512 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    float _exp2_6 = approx_exp2(s_g[(640 + tid_2) / 16]);
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_kn_addr + (unsigned int)((640 + tid_2) % 16 * 8 / 64 * 8192 + ((640 + tid_2) / 16 * 128 + (640 + tid_2) % 16 * 8 % 64 * 2 ^ ((640 + tid_2) / 16 * 128 + (640 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * (s_b[(640 + tid_2) / 16] * _exp2_6);
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[(640 + tid_2) / 16] * _exp2_6);
    vals8[2] = __uint_as_float(words[1] << 16) * (s_b[(640 + tid_2) / 16] * _exp2_6);
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[(640 + tid_2) / 16] * _exp2_6);
    vals8[4] = __uint_as_float(words[2] << 16) * (s_b[(640 + tid_2) / 16] * _exp2_6);
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[(640 + tid_2) / 16] * _exp2_6);
    vals8[6] = __uint_as_float(words[3] << 16) * (s_b[(640 + tid_2) / 16] * _exp2_6);
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[(640 + tid_2) / 16] * _exp2_6);
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)((640 + tid_2) % 16 * 8 / 64 * 8192 + ((640 + tid_2) / 16 * 128 + (640 + tid_2) % 16 * 8 % 64 * 2 ^ ((640 + tid_2) / 16 * 128 + (640 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    float _exp2_7 = approx_exp2(s_g[(768 + tid_2) / 16]);
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_kn_addr + (unsigned int)((768 + tid_2) % 16 * 8 / 64 * 8192 + ((768 + tid_2) / 16 * 128 + (768 + tid_2) % 16 * 8 % 64 * 2 ^ ((768 + tid_2) / 16 * 128 + (768 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * (s_b[(768 + tid_2) / 16] * _exp2_7);
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[(768 + tid_2) / 16] * _exp2_7);
    vals8[2] = __uint_as_float(words[1] << 16) * (s_b[(768 + tid_2) / 16] * _exp2_7);
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[(768 + tid_2) / 16] * _exp2_7);
    vals8[4] = __uint_as_float(words[2] << 16) * (s_b[(768 + tid_2) / 16] * _exp2_7);
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[(768 + tid_2) / 16] * _exp2_7);
    vals8[6] = __uint_as_float(words[3] << 16) * (s_b[(768 + tid_2) / 16] * _exp2_7);
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[(768 + tid_2) / 16] * _exp2_7);
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)((768 + tid_2) % 16 * 8 / 64 * 8192 + ((768 + tid_2) / 16 * 128 + (768 + tid_2) % 16 * 8 % 64 * 2 ^ ((768 + tid_2) / 16 * 128 + (768 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    float _exp2_8 = approx_exp2(s_g[(896 + tid_2) / 16]);
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_kn_addr + (unsigned int)((896 + tid_2) % 16 * 8 / 64 * 8192 + ((896 + tid_2) / 16 * 128 + (896 + tid_2) % 16 * 8 % 64 * 2 ^ ((896 + tid_2) / 16 * 128 + (896 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * (s_b[(896 + tid_2) / 16] * _exp2_8);
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * (s_b[(896 + tid_2) / 16] * _exp2_8);
    vals8[2] = __uint_as_float(words[1] << 16) * (s_b[(896 + tid_2) / 16] * _exp2_8);
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * (s_b[(896 + tid_2) / 16] * _exp2_8);
    vals8[4] = __uint_as_float(words[2] << 16) * (s_b[(896 + tid_2) / 16] * _exp2_8);
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * (s_b[(896 + tid_2) / 16] * _exp2_8);
    vals8[6] = __uint_as_float(words[3] << 16) * (s_b[(896 + tid_2) / 16] * _exp2_8);
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * (s_b[(896 + tid_2) / 16] * _exp2_8);
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_kn_addr + (unsigned int)((896 + tid_2) % 16 * 8 / 64 * 8192 + ((896 + tid_2) / 16 * 128 + (896 + tid_2) % 16 * 8 % 64 * 2 ^ ((896 + tid_2) / 16 * 128 + (896 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    __syncthreads();
    #pragma unroll
    for (int kb2 = 0; kb2 < 4; kb2++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
            : "r"(s_a_addr + (unsigned int)((kb2 * 16 + lane / 16 * 8) / 64 * 8192 + ((m0 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
            : "r"(s_a_addr + (unsigned int)((kb2 * 16 + lane / 16 * 8) / 64 * 8192 + ((m0 + 16 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + 16 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb3 = 0; nb3 < 4; nb3++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_kn_addr + (unsigned int)((nn0 + nb3 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb2 * 16 + lane % 16) * 128 + (nn0 + nb3 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb2 * 16 + lane % 16) * 128 + (nn0 + nb3 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb3 * 8)[0]), "=f"((acc + nb3 * 8)[1]), "=f"((acc + nb3 * 8)[2]), "=f"((acc + nb3 * 8)[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[0])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[1])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[2])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb3 * 8 + 4)[0]), "=f"((acc + nb3 * 8 + 4)[1]), "=f"((acc + nb3 * 8 + 4)[2]), "=f"((acc + nb3 * 8 + 4)[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[0])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[1])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[2])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + 32 + nb3 * 8)[0]), "=f"((acc + 32 + nb3 * 8)[1]), "=f"((acc + 32 + nb3 * 8)[2]), "=f"((acc + 32 + nb3 * 8)[3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb2 == 0) ? 0.0f : (acc + 32 + nb3 * 8)[0])), "f"(((kb2 == 0) ? 0.0f : (acc + 32 + nb3 * 8)[1])), "f"(((kb2 == 0) ? 0.0f : (acc + 32 + nb3 * 8)[2])), "f"(((kb2 == 0) ? 0.0f : (acc + 32 + nb3 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + 32 + nb3 * 8 + 4)[0]), "=f"((acc + 32 + nb3 * 8 + 4)[1]), "=f"((acc + 32 + nb3 * 8 + 4)[2]), "=f"((acc + 32 + nb3 * 8 + 4)[3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb2 == 0) ? 0.0f : (acc + 32 + nb3 * 8 + 4)[0])), "f"(((kb2 == 0) ? 0.0f : (acc + 32 + nb3 * 8 + 4)[1])), "f"(((kb2 == 0) ? 0.0f : (acc + 32 + nb3 * 8 + 4)[2])), "f"(((kb2 == 0) ? 0.0f : (acc + 32 + nb3 * 8 + 4)[3])));
        }
    }
    #pragma unroll
    for (int mb4 = 0; mb4 < 2; mb4++) {
        #pragma unroll
        for (int half4 = 0; half4 < 2; half4++) {
            int row_w = m0 + mb4 * 16 + half4 * 8 + frag_r;
            long long out_w = ((long long)(tok0 + row_w) * (long long)num_v_heads + (long long)hv) * 128 + (long long)nn0 + (long long)col_q;
            if (row_w < n_valid) {
                #pragma unroll
                for (int nb4 = 0; nb4 < 4; nb4++) {
                    #pragma unroll
                    for (int nh4 = 0; nh4 < 2; nh4++) {
                        {
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[mb4 * 32 + nb4 * 8 + nh4 * 4 + half4 * 2 + 0], acc[mb4 * 32 + nb4 * 8 + nh4 * 4 + half4 * 2 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(w_out))[out_w + (long long)(nb4 * 16) + (long long)(nh4 * 8)]) = _pk;
                        }
                    }
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)(tid_2 % 16 * 8 / 64 * 8192 + (tid_2 / 16 * 128 + tid_2 % 16 * 8 % 64 * 2 ^ (tid_2 / 16 * 128 + tid_2 % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[tid_2 / 16];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[tid_2 / 16];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[tid_2 / 16];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[tid_2 / 16];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[tid_2 / 16];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[tid_2 / 16];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[tid_2 / 16];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[tid_2 / 16];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)(tid_2 % 16 * 8 / 64 * 8192 + (tid_2 / 16 * 128 + tid_2 % 16 * 8 % 64 * 2 ^ (tid_2 / 16 * 128 + tid_2 % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)((128 + tid_2) % 16 * 8 / 64 * 8192 + ((128 + tid_2) / 16 * 128 + (128 + tid_2) % 16 * 8 % 64 * 2 ^ ((128 + tid_2) / 16 * 128 + (128 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[(128 + tid_2) / 16];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[(128 + tid_2) / 16];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[(128 + tid_2) / 16];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[(128 + tid_2) / 16];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[(128 + tid_2) / 16];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[(128 + tid_2) / 16];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[(128 + tid_2) / 16];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[(128 + tid_2) / 16];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)((128 + tid_2) % 16 * 8 / 64 * 8192 + ((128 + tid_2) / 16 * 128 + (128 + tid_2) % 16 * 8 % 64 * 2 ^ ((128 + tid_2) / 16 * 128 + (128 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)((256 + tid_2) % 16 * 8 / 64 * 8192 + ((256 + tid_2) / 16 * 128 + (256 + tid_2) % 16 * 8 % 64 * 2 ^ ((256 + tid_2) / 16 * 128 + (256 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[(256 + tid_2) / 16];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[(256 + tid_2) / 16];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[(256 + tid_2) / 16];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[(256 + tid_2) / 16];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[(256 + tid_2) / 16];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[(256 + tid_2) / 16];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[(256 + tid_2) / 16];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[(256 + tid_2) / 16];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)((256 + tid_2) % 16 * 8 / 64 * 8192 + ((256 + tid_2) / 16 * 128 + (256 + tid_2) % 16 * 8 % 64 * 2 ^ ((256 + tid_2) / 16 * 128 + (256 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)((384 + tid_2) % 16 * 8 / 64 * 8192 + ((384 + tid_2) / 16 * 128 + (384 + tid_2) % 16 * 8 % 64 * 2 ^ ((384 + tid_2) / 16 * 128 + (384 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[(384 + tid_2) / 16];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[(384 + tid_2) / 16];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[(384 + tid_2) / 16];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[(384 + tid_2) / 16];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[(384 + tid_2) / 16];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[(384 + tid_2) / 16];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[(384 + tid_2) / 16];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[(384 + tid_2) / 16];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)((384 + tid_2) % 16 * 8 / 64 * 8192 + ((384 + tid_2) / 16 * 128 + (384 + tid_2) % 16 * 8 % 64 * 2 ^ ((384 + tid_2) / 16 * 128 + (384 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)((512 + tid_2) % 16 * 8 / 64 * 8192 + ((512 + tid_2) / 16 * 128 + (512 + tid_2) % 16 * 8 % 64 * 2 ^ ((512 + tid_2) / 16 * 128 + (512 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[(512 + tid_2) / 16];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[(512 + tid_2) / 16];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[(512 + tid_2) / 16];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[(512 + tid_2) / 16];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[(512 + tid_2) / 16];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[(512 + tid_2) / 16];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[(512 + tid_2) / 16];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[(512 + tid_2) / 16];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)((512 + tid_2) % 16 * 8 / 64 * 8192 + ((512 + tid_2) / 16 * 128 + (512 + tid_2) % 16 * 8 % 64 * 2 ^ ((512 + tid_2) / 16 * 128 + (512 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)((640 + tid_2) % 16 * 8 / 64 * 8192 + ((640 + tid_2) / 16 * 128 + (640 + tid_2) % 16 * 8 % 64 * 2 ^ ((640 + tid_2) / 16 * 128 + (640 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[(640 + tid_2) / 16];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[(640 + tid_2) / 16];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[(640 + tid_2) / 16];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[(640 + tid_2) / 16];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[(640 + tid_2) / 16];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[(640 + tid_2) / 16];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[(640 + tid_2) / 16];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[(640 + tid_2) / 16];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)((640 + tid_2) % 16 * 8 / 64 * 8192 + ((640 + tid_2) / 16 * 128 + (640 + tid_2) % 16 * 8 % 64 * 2 ^ ((640 + tid_2) / 16 * 128 + (640 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)((768 + tid_2) % 16 * 8 / 64 * 8192 + ((768 + tid_2) / 16 * 128 + (768 + tid_2) % 16 * 8 % 64 * 2 ^ ((768 + tid_2) / 16 * 128 + (768 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[(768 + tid_2) / 16];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[(768 + tid_2) / 16];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[(768 + tid_2) / 16];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[(768 + tid_2) / 16];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[(768 + tid_2) / 16];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[(768 + tid_2) / 16];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[(768 + tid_2) / 16];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[(768 + tid_2) / 16];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)((768 + tid_2) % 16 * 8 / 64 * 8192 + ((768 + tid_2) / 16 * 128 + (768 + tid_2) % 16 * 8 % 64 * 2 ^ ((768 + tid_2) / 16 * 128 + (768 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
        : "r"(s_v_addr + (unsigned int)((896 + tid_2) % 16 * 8 / 64 * 8192 + ((896 + tid_2) / 16 * 128 + (896 + tid_2) % 16 * 8 % 64 * 2 ^ ((896 + tid_2) / 16 * 128 + (896 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))));
    vals8[0] = __uint_as_float(words[0] << 16) * s_b[(896 + tid_2) / 16];
    vals8[1] = __uint_as_float(words[0] & 4294901760u) * s_b[(896 + tid_2) / 16];
    vals8[2] = __uint_as_float(words[1] << 16) * s_b[(896 + tid_2) / 16];
    vals8[3] = __uint_as_float(words[1] & 4294901760u) * s_b[(896 + tid_2) / 16];
    vals8[4] = __uint_as_float(words[2] << 16) * s_b[(896 + tid_2) / 16];
    vals8[5] = __uint_as_float(words[2] & 4294901760u) * s_b[(896 + tid_2) / 16];
    vals8[6] = __uint_as_float(words[3] << 16) * s_b[(896 + tid_2) / 16];
    vals8[7] = __uint_as_float(words[3] & 4294901760u) * s_b[(896 + tid_2) / 16];
    #pragma unroll
    for (int _lp = 0; _lp < 4; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vals8[_lp*2 + 0], vals8[_lp*2+1 + 0]));
        pk4[_lp] = *(uint32_t*)&_bf2;
    }
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(s_v_addr + (unsigned int)((896 + tid_2) % 16 * 8 / 64 * 8192 + ((896 + tid_2) / 16 * 128 + (896 + tid_2) % 16 * 8 % 64 * 2 ^ ((896 + tid_2) / 16 * 128 + (896 + tid_2) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(pk4[0]), "r"(pk4[1]), "r"(pk4[2]), "r"(pk4[3]) : "memory");
    __syncthreads();
    #pragma unroll
    for (int kb3 = 0; kb3 < 4; kb3++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag0[0]), "=r"(a_frag0[1]), "=r"(a_frag0[2]), "=r"(a_frag0[3])
            : "r"(s_a_addr + (unsigned int)((kb3 * 16 + lane / 16 * 8) / 64 * 8192 + ((m0 + lane % 16) * 128 + (kb3 * 16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + lane % 16) * 128 + (kb3 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag1[0]), "=r"(a_frag1[1]), "=r"(a_frag1[2]), "=r"(a_frag1[3])
            : "r"(s_a_addr + (unsigned int)((kb3 * 16 + lane / 16 * 8) / 64 * 8192 + ((m0 + 16 + lane % 16) * 128 + (kb3 * 16 + lane / 16 * 8) % 64 * 2 ^ ((m0 + 16 + lane % 16) * 128 + (kb3 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb5 = 0; nb5 < 4; nb5++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_v_addr + (unsigned int)((nn0 + nb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb3 * 16 + lane % 16) * 128 + (nn0 + nb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb3 * 16 + lane % 16) * 128 + (nn0 + nb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb5 * 8)[0]), "=f"((acc + nb5 * 8)[1]), "=f"((acc + nb5 * 8)[2]), "=f"((acc + nb5 * 8)[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8)[0])), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8)[1])), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8)[2])), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb5 * 8 + 4)[0]), "=f"((acc + nb5 * 8 + 4)[1]), "=f"((acc + nb5 * 8 + 4)[2]), "=f"((acc + nb5 * 8 + 4)[3])
                : "r"(a_frag0[0]), "r"(a_frag0[1]), "r"(a_frag0[2]), "r"(a_frag0[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8 + 4)[0])), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8 + 4)[1])), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8 + 4)[2])), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8 + 4)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + 32 + nb5 * 8)[0]), "=f"((acc + 32 + nb5 * 8)[1]), "=f"((acc + 32 + nb5 * 8)[2]), "=f"((acc + 32 + nb5 * 8)[3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb3 == 0) ? 0.0f : (acc + 32 + nb5 * 8)[0])), "f"(((kb3 == 0) ? 0.0f : (acc + 32 + nb5 * 8)[1])), "f"(((kb3 == 0) ? 0.0f : (acc + 32 + nb5 * 8)[2])), "f"(((kb3 == 0) ? 0.0f : (acc + 32 + nb5 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + 32 + nb5 * 8 + 4)[0]), "=f"((acc + 32 + nb5 * 8 + 4)[1]), "=f"((acc + 32 + nb5 * 8 + 4)[2]), "=f"((acc + 32 + nb5 * 8 + 4)[3])
                : "r"(a_frag1[0]), "r"(a_frag1[1]), "r"(a_frag1[2]), "r"(a_frag1[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb3 == 0) ? 0.0f : (acc + 32 + nb5 * 8 + 4)[0])), "f"(((kb3 == 0) ? 0.0f : (acc + 32 + nb5 * 8 + 4)[1])), "f"(((kb3 == 0) ? 0.0f : (acc + 32 + nb5 * 8 + 4)[2])), "f"(((kb3 == 0) ? 0.0f : (acc + 32 + nb5 * 8 + 4)[3])));
        }
    }
    #pragma unroll
    for (int mb6 = 0; mb6 < 2; mb6++) {
        #pragma unroll
        for (int half6 = 0; half6 < 2; half6++) {
            int row_u = m0 + mb6 * 16 + half6 * 8 + frag_r;
            long long out_u = ((long long)(tok0 + row_u) * (long long)num_v_heads + (long long)hv) * 128 + (long long)nn0 + (long long)col_q;
            if (row_u < n_valid) {
                #pragma unroll
                for (int nb6 = 0; nb6 < 4; nb6++) {
                    #pragma unroll
                    for (int nh6 = 0; nh6 < 2; nh6++) {
                        {
                            __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[mb6 * 32 + nb6 * 8 + nh6 * 4 + half6 * 2 + 0], acc[mb6 * 32 + nb6 * 8 + nh6 * 4 + half6 * 2 + 1]);
                            *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(u_out))[out_u + (long long)(nb6 * 16) + (long long)(nh6 * 8)]) = _pk;
                        }
                    }
                }
            }
        }
    }
}

} // extern "C"
