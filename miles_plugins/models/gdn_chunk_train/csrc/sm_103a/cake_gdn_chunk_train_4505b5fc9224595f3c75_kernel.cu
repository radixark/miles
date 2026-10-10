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
#define SMEM_S_KN_OFF 0
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 16384
#define SMEM_S_V_OFF 16384
#define SMEM_S_V_STAGE_BYTES 16384
#define SMEM_S_V_STRIDE 16384
#define SMEM_S_KBG_OFF 32768
#define SMEM_S_KBG_STAGE_BYTES 16384
#define SMEM_S_KBG_STRIDE 16384
#define SMEM_S_A_OFF 49152
#define SMEM_S_A_STAGE_BYTES 8192
#define SMEM_S_A_STRIDE 8192
#define SMEM_S_L_OFF 57344
#define SMEM_S_L_STAGE_BYTES 16384
#define SMEM_S_L_STRIDE 16384
#define SMEM_S_X_OFF 73728
#define SMEM_S_X_STAGE_BYTES 16384
#define SMEM_S_X_STRIDE 16384
#define SMEM_S_T_OFF 90112
#define SMEM_S_T_STAGE_BYTES 3072
#define SMEM_S_T_STRIDE 3072
#define SMEM_S_G_OFF 93184
#define SMEM_S_G_STAGE_BYTES 256
#define SMEM_S_G_STRIDE 256
#define SMEM_S_B_OFF 93440
#define SMEM_S_B_STAGE_BYTES 256
#define SMEM_S_B_STRIDE 256
#define SMEM_TOTAL 93696
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_gdn_chunk_train_4505b5fc9224595f3c75(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ v, float* __restrict__ g_cs, float* __restrict__ beta, __nv_bfloat16* __restrict__ A_out, __nv_bfloat16* __restrict__ w_out, __nv_bfloat16* __restrict__ u_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, int recompute)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_kn_addr = smem + 0;
    __nv_bfloat16* s_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_v_addr = smem + 16384;
    __nv_bfloat16* s_kbg = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_kbg_addr = smem + 32768;
    __nv_bfloat16* s_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int s_a_addr = smem + 49152;
    float* s_l = reinterpret_cast<float*>(smem_raw + 57344);
    const int s_l_addr = smem + 57344;
    float* s_x = reinterpret_cast<float*>(smem_raw + 73728);
    const int s_x_addr = smem + 73728;
    float* s_t = reinterpret_cast<float*>(smem_raw + 90112);
    const int s_t_addr = smem + 90112;
    float* s_g = reinterpret_cast<float*>(smem_raw + 93184);
    const int s_g_addr = smem + 93184;
    float* s_b = reinterpret_cast<float*>(smem_raw + 93440);
    const int s_b_addr = smem + 93440;

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
    int row_a = warp_1 * 16 + lane_0 / 4;
    int row_b = row_a + 8;
    int col_q = lane_0 % 4 * 2;
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
    if (recompute != 0) {
        #pragma unroll
        for (int it = 0; it < 4; it++) {
            int aidx = it * 128 + tid_2;
            int ld_row = aidx / 8;
            int ld_col = aidx % 8 * 8;
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_a_addr + (unsigned int)(ld_col / 64 * 8192 + (ld_row * 128 + ld_col % 64 * 2 ^ (ld_row * 128 + ld_col % 64 * 2 >> 7 & 7) << 4))), "l"(A_out + (((long long)(tok0 + ld_row) * (long long)num_v_heads + (long long)hv) * 64 + (long long)ld_col)), "r"((ld_row < n_valid) ? 16 : 0));
        }
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
    unsigned int a_frag[4];
    unsigned int b_frag_k[4];
    unsigned int b_frag_mn[4];
    float acc[64];
    if (recompute == 0) {
        #pragma unroll
        for (int kb = 0; kb < 8; kb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(s_kn_addr + (unsigned int)((kb * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            #pragma unroll
            for (int nb2 = 0; nb2 < 4; nb2++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                    : "r"(s_kn_addr + (unsigned int)((kb * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((nb2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((nb2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc + nb2 * 8)[0]), "=f"((acc + nb2 * 8)[1]), "=f"((acc + nb2 * 8)[2]), "=f"((acc + nb2 * 8)[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8)[0])), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8)[1])), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8)[2])), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8)[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc + nb2 * 8 + 4)[0]), "=f"((acc + nb2 * 8 + 4)[1]), "=f"((acc + nb2 * 8 + 4)[2]), "=f"((acc + nb2 * 8 + 4)[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8 + 4)[0])), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8 + 4)[1])), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8 + 4)[2])), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8 + 4)[3])));
            }
        }
        float g_a = s_g[row_a];
        float g_b = s_g[row_b];
        float beta_a = s_b[row_a];
        float beta_b = s_b[row_b];
        #pragma unroll
        for (int nb = 0; nb < 8; nb++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int col = nb * 8 + col_q + e;
                float g_col = s_g[col];
                float val_a = 0.0f;
                float val_b = 0.0f;
                if (col < row_a) {
                    if (row_a < n_valid) {
                        float _exp2_0 = approx_exp2(g_a - g_col);
                        val_a = beta_a * _exp2_0 * acc[nb * 4 + e];
                    }
                }
                if (col < row_b) {
                    if (row_b < n_valid) {
                        float _exp2_1 = approx_exp2(g_b - g_col);
                        val_b = beta_b * _exp2_1 * acc[nb * 4 + 2 + e];
                    }
                }
                s_l[row_a * 64 + col] = val_a;
                s_l[row_b * 64 + col] = val_b;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int zt = 0; zt < 12; zt++) {
            int zidx = zt * 128 + tid_2;
            int zblk = zidx / 256;
            int zr = zidx % 256 / 16;
            int zc = zidx % 16;
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
            s_x[(zbi * 16 + zr) * 64 + zbj * 16 + zc] = 0.0f;
        }
        if (tid_2 < 64) {
            int dblk = tid_2 / 16;
            int dcol = tid_2 % 16;
            int drow0 = dblk * 16;
            #pragma unroll
            for (int di = 0; di < 16; di++) {
                float xv = 0.0f;
                if (di == dcol) {
                    xv = 1.0f;
                }
                #pragma unroll
                for (int dj = 0; dj < di; dj++) {
                    xv = xv - s_l[(drow0 + di) * 64 + drow0 + dj] * s_x[(drow0 + dj) * 64 + drow0 + dcol];
                }
                s_x[(drow0 + di) * 64 + drow0 + dcol] = xv;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int l1 = 0; l1 < 6; l1++) {
            int e1 = l1 * 128 + tid_2;
            int b1 = e1 / 256;
            int r1 = e1 % 256 / 16;
            int c1 = e1 % 16;
            float t1 = 0.0f;
            #pragma unroll
            for (int k1 = 0; k1 < 16; k1++) {
                float _fma_0 = __fmaf_rn(s_l[((b1 + 1) * 16 + r1) * 64 + b1 * 16 + k1], s_x[(b1 * 16 + k1) * 64 + b1 * 16 + c1], t1);
                t1 = _fma_0;
            }
            s_t[b1 * 256 + r1 * 16 + c1] = t1;
        }
        __syncthreads();
        #pragma unroll
        for (int l1b = 0; l1b < 6; l1b++) {
            int e1b = l1b * 128 + tid_2;
            int b1b = e1b / 256;
            int r1b = e1b % 256 / 16;
            int c1b = e1b % 16;
            float v1 = 0.0f;
            #pragma unroll
            for (int k1b = 0; k1b < 16; k1b++) {
                float _fma_1 = __fmaf_rn(s_x[((b1b + 1) * 16 + r1b) * 64 + (b1b + 1) * 16 + k1b], s_t[b1b * 256 + k1b * 16 + c1b], v1);
                v1 = _fma_1;
            }
            s_x[((b1b + 1) * 16 + r1b) * 64 + b1b * 16 + c1b] = -v1;
        }
        __syncthreads();
        #pragma unroll
        for (int l2 = 0; l2 < 4; l2++) {
            int e2 = l2 * 128 + tid_2;
            int b2 = e2 / 256;
            int r2 = e2 % 256 / 16;
            int c2 = e2 % 16;
            float t2 = 0.0f;
            #pragma unroll
            for (int k2 = 0; k2 < 16; k2++) {
                float _fma_2 = __fmaf_rn(s_l[((b2 + 2) * 16 + r2) * 64 + b2 * 16 + k2], s_x[(b2 * 16 + k2) * 64 + b2 * 16 + c2], t2);
                t2 = _fma_2;
                float _fma_3 = __fmaf_rn(s_l[((b2 + 2) * 16 + r2) * 64 + (b2 + 1) * 16 + k2], s_x[((b2 + 1) * 16 + k2) * 64 + b2 * 16 + c2], t2);
                t2 = _fma_3;
            }
            s_t[b2 * 256 + r2 * 16 + c2] = t2;
        }
        __syncthreads();
        #pragma unroll
        for (int l2b = 0; l2b < 4; l2b++) {
            int e2b = l2b * 128 + tid_2;
            int b2b = e2b / 256;
            int r2b = e2b % 256 / 16;
            int c2b = e2b % 16;
            float v2 = 0.0f;
            #pragma unroll
            for (int k2b = 0; k2b < 16; k2b++) {
                float _fma_4 = __fmaf_rn(s_x[((b2b + 2) * 16 + r2b) * 64 + (b2b + 2) * 16 + k2b], s_t[b2b * 256 + k2b * 16 + c2b], v2);
                v2 = _fma_4;
            }
            s_x[((b2b + 2) * 16 + r2b) * 64 + b2b * 16 + c2b] = -v2;
        }
        __syncthreads();
        #pragma unroll
        for (int l3 = 0; l3 < 2; l3++) {
            int e3 = l3 * 128 + tid_2;
            int r3 = e3 / 16;
            int c3 = e3 % 16;
            float t3 = 0.0f;
            #pragma unroll
            for (int k3 = 0; k3 < 16; k3++) {
                float _fma_5 = __fmaf_rn(s_l[(48 + r3) * 64 + k3], s_x[k3 * 64 + c3], t3);
                t3 = _fma_5;
                float _fma_6 = __fmaf_rn(s_l[(48 + r3) * 64 + 16 + k3], s_x[(16 + k3) * 64 + c3], t3);
                t3 = _fma_6;
                float _fma_7 = __fmaf_rn(s_l[(48 + r3) * 64 + 32 + k3], s_x[(32 + k3) * 64 + c3], t3);
                t3 = _fma_7;
            }
            s_t[r3 * 16 + c3] = t3;
        }
        __syncthreads();
        #pragma unroll
        for (int l3b = 0; l3b < 2; l3b++) {
            int e3b = l3b * 128 + tid_2;
            int r3b = e3b / 16;
            int c3b = e3b % 16;
            float v3 = 0.0f;
            #pragma unroll
            for (int k3b = 0; k3b < 16; k3b++) {
                float _fma_8 = __fmaf_rn(s_x[(48 + r3b) * 64 + 48 + k3b], s_t[k3b * 16 + c3b], v3);
                v3 = _fma_8;
            }
            s_x[(48 + r3b) * 64 + c3b] = -v3;
        }
        __syncthreads();
        float pair[2];
        #pragma unroll
        for (int t = 0; t < 16; t++) {
            int pidx = t * 128 + tid_2;
            int a_row = pidx / 32;
            int a_col = pidx % 32 * 2;
            pair[0] = s_x[a_row * 64 + a_col];
            pair[1] = s_x[a_row * 64 + a_col + 1];
            {
                __nv_bfloat16 _bval_0 = __float2bfloat16_rn(pair[0]);
                uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                uint32_t _addr_0 = static_cast<uint32_t>((s_a_addr + (unsigned int)(a_row * 128 + a_col * 2 ^ (a_row * 128 + a_col * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
            }
            {
                __nv_bfloat16 _bval_1 = __float2bfloat16_rn(pair[1]);
                uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                uint32_t _addr_1 = static_cast<uint32_t>((s_a_addr + (unsigned int)(a_row * 128 + (a_col * 2 + 2) ^ (a_row * 128 + (a_col * 2 + 2) >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
            }
            if (a_row < n_valid) {
                long long a_index = ((long long)(tok0 + a_row) * (long long)num_v_heads + (long long)hv) * 64 + (long long)a_col;
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(pair[0 + 0], pair[0 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(A_out))[a_index]) = _pk;
                }
            }
        }
    }
    #pragma unroll
    for (int t2_1 = 0; t2_1 < 32; t2_1++) {
        int s_row = t2_1 * 2 + tid_2 / 64;
        int s_col = tid_2 % 64 * 2;
        float _exp2_2 = approx_exp2(s_g[s_row]);
        float scale_k = s_b[s_row] * _exp2_2;
        float scale_v = s_b[s_row];
        #pragma unroll
        for (int e2_1 = 0; e2_1 < 2; e2_1++) {
            int elem_off = (s_col + e2_1) / 64 * 8192 + (s_row * 128 + (s_col + e2_1) % 64 * 2 ^ (s_row * 128 + (s_col + e2_1) % 64 * 2 >> 7 & 7) << 4);
            float k_val = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kn) + elem_off)[0];
            {
                __nv_bfloat16 _bval_2 = __float2bfloat16_rn(k_val * scale_k);
                uint16_t _bits_2 = *(uint16_t*)&_bval_2;
                uint32_t _addr_2 = static_cast<uint32_t>(s_kbg_addr + (unsigned int)elem_off);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_2), "h"(_bits_2) : "memory");
            }
            float v_val = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_v) + elem_off)[0];
            {
                __nv_bfloat16 _bval_3 = __float2bfloat16_rn(v_val * scale_v);
                uint16_t _bits_3 = *(uint16_t*)&_bval_3;
                uint32_t _addr_3 = static_cast<uint32_t>(s_v_addr + (unsigned int)elem_off);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_3), "h"(_bits_3) : "memory");
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int kb2 = 0; kb2 < 4; kb2++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(s_a_addr + (unsigned int)((kb2 * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb3 = 0; nb3 < 8; nb3++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_kbg_addr + (unsigned int)((nb3 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb2 * 16 + lane % 16) * 128 + (nb3 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb2 * 16 + lane % 16) * 128 + (nb3 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb3 * 8)[0]), "=f"((acc + nb3 * 8)[1]), "=f"((acc + nb3 * 8)[2]), "=f"((acc + nb3 * 8)[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[0])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[1])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[2])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb3 * 8 + 4)[0]), "=f"((acc + nb3 * 8 + 4)[1]), "=f"((acc + nb3 * 8 + 4)[2]), "=f"((acc + nb3 * 8 + 4)[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[0])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[1])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[2])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[3])));
        }
    }
    long long out_a = ((long long)(tok0 + row_a) * (long long)num_v_heads + (long long)hv) * 128 + (long long)col_q;
    long long out_b = ((long long)(tok0 + row_b) * (long long)num_v_heads + (long long)hv) * 128 + (long long)col_q;
    #pragma unroll
    for (int nb4 = 0; nb4 < 16; nb4++) {
        if (row_a < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb4 * 4 + 0], acc[nb4 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(w_out))[out_a + (long long)(nb4 * 8)]) = _pk;
            }
        }
        if (row_b < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb4 * 4 + 2 + 0], acc[nb4 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(w_out))[out_b + (long long)(nb4 * 8)]) = _pk;
            }
        }
    }
    #pragma unroll
    for (int kb3 = 0; kb3 < 4; kb3++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(s_a_addr + (unsigned int)((kb3 * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb3 * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb3 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb5 = 0; nb5 < 8; nb5++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_v_addr + (unsigned int)((nb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb3 * 16 + lane % 16) * 128 + (nb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb3 * 16 + lane % 16) * 128 + (nb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb5 * 8)[0]), "=f"((acc + nb5 * 8)[1]), "=f"((acc + nb5 * 8)[2]), "=f"((acc + nb5 * 8)[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8)[0])), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8)[1])), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8)[2])), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb5 * 8 + 4)[0]), "=f"((acc + nb5 * 8 + 4)[1]), "=f"((acc + nb5 * 8 + 4)[2]), "=f"((acc + nb5 * 8 + 4)[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8 + 4)[0])), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8 + 4)[1])), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8 + 4)[2])), "f"(((kb3 == 0) ? 0.0f : (acc + nb5 * 8 + 4)[3])));
        }
    }
    #pragma unroll
    for (int nb6 = 0; nb6 < 16; nb6++) {
        if (row_a < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb6 * 4 + 0], acc[nb6 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(u_out))[out_a + (long long)(nb6 * 8)]) = _pk;
            }
        }
        if (row_b < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb6 * 4 + 2 + 0], acc[nb6 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(u_out))[out_b + (long long)(nb6 * 8)]) = _pk;
            }
        }
    }
}

} // extern "C"
