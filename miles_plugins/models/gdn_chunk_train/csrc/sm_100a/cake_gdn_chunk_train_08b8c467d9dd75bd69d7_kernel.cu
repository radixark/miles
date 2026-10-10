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
#define SMEM_S_VB_OFF 32768
#define SMEM_S_VB_STAGE_BYTES 16384
#define SMEM_S_VB_STRIDE 16384
#define SMEM_S_KB_OFF 65536
#define SMEM_S_KB_STAGE_BYTES 16384
#define SMEM_S_KB_STRIDE 16384
#define SMEM_S_KBG_OFF 49152
#define SMEM_S_KBG_STAGE_BYTES 16384
#define SMEM_S_KBG_STRIDE 16384
#define SMEM_S_DW_OFF 65536
#define SMEM_S_DW_STAGE_BYTES 16384
#define SMEM_S_DW_STRIDE 16384
#define SMEM_S_DU_OFF 81920
#define SMEM_S_DU_STAGE_BYTES 16384
#define SMEM_S_DU_STRIDE 16384
#define SMEM_S_A_OFF 98304
#define SMEM_S_A_STAGE_BYTES 8192
#define SMEM_S_A_STRIDE 8192
#define SMEM_S_DA_OFF 32768
#define SMEM_S_DA_STAGE_BYTES 8192
#define SMEM_S_DA_STRIDE 8192
#define SMEM_S_DA2_OFF 81920
#define SMEM_S_DA2_STAGE_BYTES 8192
#define SMEM_S_DA2_STRIDE 8192
#define SMEM_S_G_OFF 106496
#define SMEM_S_G_STAGE_BYTES 256
#define SMEM_S_G_STRIDE 256
#define SMEM_S_B_OFF 106752
#define SMEM_S_B_STAGE_BYTES 256
#define SMEM_S_B_STRIDE 256
#define SMEM_S_COL_OFF 107008
#define SMEM_S_COL_STAGE_BYTES 1024
#define SMEM_S_COL_STRIDE 1024
#define SMEM_TOTAL 108032
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_gdn_chunk_train_08b8c467d9dd75bd69d7(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ A, __nv_bfloat16* __restrict__ dw, __nv_bfloat16* __restrict__ du, float* __restrict__ g_cs, float* __restrict__ beta, __nv_bfloat16* __restrict__ dk2_out, __nv_bfloat16* __restrict__ dv_out, float* __restrict__ dbeta_out, float* __restrict__ dg2_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads)
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
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_kn_addr = smem + 0;
    __nv_bfloat16* s_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_v_addr = smem + 16384;
    __nv_bfloat16* s_vb = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_vb_addr = smem + 32768;
    __nv_bfloat16* s_kb = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_kb_addr = smem + 65536;
    __nv_bfloat16* s_kbg = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int s_kbg_addr = smem + 49152;
    __nv_bfloat16* s_dw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_dw_addr = smem + 65536;
    __nv_bfloat16* s_du = reinterpret_cast<__nv_bfloat16*>(smem_raw + 81920);
    const int s_du_addr = smem + 81920;
    __nv_bfloat16* s_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 98304);
    const int s_a_addr = smem + 98304;
    __nv_bfloat16* s_da = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_da_addr = smem + 32768;
    __nv_bfloat16* s_da2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 81920);
    const int s_da2_addr = smem + 81920;
    float* s_g = reinterpret_cast<float*>(smem_raw + 106496);
    const int s_g_addr = smem + 106496;
    float* s_b = reinterpret_cast<float*>(smem_raw + 106752);
    const int s_b_addr = smem + 106752;
    float* s_col = reinterpret_cast<float*>(smem_raw + 107008);
    const int s_col_addr = smem + 107008;

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
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dw_addr + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dw + (((long long)(tok0 + tid / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)(tid % 16 * 8))), "r"((n_valid > tid / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dw_addr + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dw + (((long long)(tok0 + (128 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((n_valid > (128 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dw_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dw + (((long long)(tok0 + (256 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((n_valid > (256 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dw_addr + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dw + (((long long)(tok0 + (384 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((n_valid > (384 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dw_addr + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dw + (((long long)(tok0 + (512 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((n_valid > (512 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dw_addr + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dw + (((long long)(tok0 + (640 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((n_valid > (640 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dw_addr + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dw + (((long long)(tok0 + (768 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((n_valid > (768 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dw_addr + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dw + (((long long)(tok0 + (896 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((n_valid > (896 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_du_addr + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(du + (((long long)(tok0 + tid / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)(tid % 16 * 8))), "r"((n_valid > tid / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_du_addr + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(du + (((long long)(tok0 + (128 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((n_valid > (128 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_du_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(du + (((long long)(tok0 + (256 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((n_valid > (256 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_du_addr + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(du + (((long long)(tok0 + (384 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((n_valid > (384 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_du_addr + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(du + (((long long)(tok0 + (512 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((n_valid > (512 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_du_addr + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(du + (((long long)(tok0 + (640 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((n_valid > (640 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_du_addr + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(du + (((long long)(tok0 + (768 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((n_valid > (768 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_du_addr + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(du + (((long long)(tok0 + (896 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((n_valid > (896 + tid) / 16) ? 16 : 0));
    #pragma unroll
    for (int it = 0; it < 4; it++) {
        int aidx = it * 128 + tid_2;
        int a_row = aidx / 8;
        int a_col = aidx % 8 * 8;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_a_addr + (unsigned int)(a_col / 64 * 8192 + (a_row * 128 + a_col % 64 * 2 ^ (a_row * 128 + a_col % 64 * 2 >> 7 & 7) << 4))), "l"(A + (((long long)(tok0 + a_row) * (long long)num_v_heads + (long long)hv) * 64 + (long long)a_col)), "r"((a_row < n_valid) ? 16 : 0));
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
    #pragma unroll
    for (int t2 = 0; t2 < 32; t2++) {
        int s_row = t2 * 2 + tid_2 / 64;
        int s_col_e = tid_2 % 64 * 2;
        float _exp2_0 = approx_exp2(s_g[s_row]);
        float scale_kg = s_b[s_row] * _exp2_0;
        float scale_b = s_b[s_row];
        #pragma unroll
        for (int e0 = 0; e0 < 2; e0++) {
            int elem_off = (s_col_e + e0) / 64 * 8192 + (s_row * 128 + (s_col_e + e0) % 64 * 2 ^ (s_row * 128 + (s_col_e + e0) % 64 * 2 >> 7 & 7) << 4);
            float k_val = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kn) + elem_off)[0];
            {
                __nv_bfloat16 _bval_0 = __float2bfloat16_rn(k_val * scale_kg);
                uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                uint32_t _addr_0 = static_cast<uint32_t>(s_kbg_addr + (unsigned int)elem_off);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
            }
            float v_val = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_v) + elem_off)[0];
            {
                __nv_bfloat16 _bval_1 = __float2bfloat16_rn(v_val * scale_b);
                uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                uint32_t _addr_1 = static_cast<uint32_t>(s_vb_addr + (unsigned int)elem_off);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
            }
        }
    }
    __syncthreads();
    unsigned int a_frag_k[4];
    unsigned int a_frag_mn[4];
    unsigned int b_frag_k[4];
    unsigned int b_frag_mn[4];
    float acc_s[32];
    float acc[64];
    float pair[2];
    float g_a = s_g[row_a];
    float g_b = s_g[row_b];
    float beta_a = s_b[row_a];
    float beta_b = s_b[row_b];
    float _exp2_1 = approx_exp2(g_a);
    float gexp_a = _exp2_1;
    float _exp2_2 = approx_exp2(g_b);
    float gexp_b = _exp2_2;
    float db_a = 0.0f;
    float db_b = 0.0f;
    float dg_a = 0.0f;
    float dg_b = 0.0f;
    long long out_a = ((long long)(tok0 + row_a) * (long long)num_v_heads + (long long)hv) * 128 + (long long)col_q;
    long long out_b = ((long long)(tok0 + row_b) * (long long)num_v_heads + (long long)hv) * 128 + (long long)col_q;
    #pragma unroll
    for (int kb = 0; kb < 8; kb++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_dw_addr + (unsigned int)((kb * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                : "r"(s_kbg_addr + (unsigned int)((kb * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((nb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((nb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
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
    for (int kb2 = 0; kb2 < 8; kb2++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_du_addr + (unsigned int)((kb2 * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb2 = 0; nb2 < 4; nb2++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                : "r"(s_vb_addr + (unsigned int)((kb2 * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((nb2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb2 * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((nb2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb2 * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc_s + nb2 * 8)[0]), "+f"((acc_s + nb2 * 8)[1]), "+f"((acc_s + nb2 * 8)[2]), "+f"((acc_s + nb2 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc_s + nb2 * 8 + 4)[0]), "+f"((acc_s + nb2 * 8 + 4)[1]), "+f"((acc_s + nb2 * 8 + 4)[2]), "+f"((acc_s + nb2 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]));
        }
    }
    #pragma unroll
    for (int kb3 = 0; kb3 < 4; kb3++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_mn[0]), "=r"(a_frag_mn[1]), "=r"(a_frag_mn[2]), "=r"(a_frag_mn[3])
            : "r"(s_a_addr + (unsigned int)((warp_1 * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((kb3 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (warp_1 * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((kb3 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (warp_1 * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb3 = 0; nb3 < 8; nb3++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_dw_addr + (unsigned int)((nb3 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb3 * 16 + lane % 16) * 128 + (nb3 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb3 * 16 + lane % 16) * 128 + (nb3 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb3 * 8)[0]), "=f"((acc + nb3 * 8)[1]), "=f"((acc + nb3 * 8)[2]), "=f"((acc + nb3 * 8)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb3 == 0) ? 0.0f : (acc + nb3 * 8)[0])), "f"(((kb3 == 0) ? 0.0f : (acc + nb3 * 8)[1])), "f"(((kb3 == 0) ? 0.0f : (acc + nb3 * 8)[2])), "f"(((kb3 == 0) ? 0.0f : (acc + nb3 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb3 * 8 + 4)[0]), "=f"((acc + nb3 * 8 + 4)[1]), "=f"((acc + nb3 * 8 + 4)[2]), "=f"((acc + nb3 * 8 + 4)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb3 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[0])), "f"(((kb3 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[1])), "f"(((kb3 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[2])), "f"(((kb3 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[3])));
        }
    }
    #pragma unroll
    for (int nb4 = 0; nb4 < 16; nb4++) {
        #pragma unroll
        for (int e = 0; e < 2; e++) {
            int kcol = nb4 * 8 + col_q + e;
            float k_a = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kn) + (kcol / 64 * 8192 + (row_a * 128 + kcol % 64 * 2 ^ (row_a * 128 + kcol % 64 * 2 >> 7 & 7) << 4)))[0];
            float k_b = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kn) + (kcol / 64 * 8192 + (row_b * 128 + kcol % 64 * 2 ^ (row_b * 128 + kcol % 64 * 2 >> 7 & 7) << 4)))[0];
            float kbg_a = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kbg) + (kcol / 64 * 8192 + (row_a * 128 + kcol % 64 * 2 ^ (row_a * 128 + kcol % 64 * 2 >> 7 & 7) << 4)))[0];
            float kbg_b = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kbg) + (kcol / 64 * 8192 + (row_b * 128 + kcol % 64 * 2 ^ (row_b * 128 + kcol % 64 * 2 >> 7 & 7) << 4)))[0];
            float _fma_0 = __fmaf_rn(acc[nb4 * 4 + e], k_a * gexp_a, db_a);
            db_a = _fma_0;
            float _fma_1 = __fmaf_rn(acc[nb4 * 4 + 2 + e], k_b * gexp_b, db_b);
            db_b = _fma_1;
            float _fma_2 = __fmaf_rn(acc[nb4 * 4 + e], kbg_a, dg_a);
            dg_a = _fma_2;
            float _fma_3 = __fmaf_rn(acc[nb4 * 4 + 2 + e], kbg_b, dg_b);
            dg_b = _fma_3;
            acc[nb4 * 4 + e] = acc[nb4 * 4 + e] * (gexp_a * beta_a);
            acc[nb4 * 4 + 2 + e] = acc[nb4 * 4 + 2 + e] * (gexp_b * beta_b);
        }
        if (row_a < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb4 * 4 + 0], acc[nb4 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dk2_out))[out_a + (long long)(nb4 * 8)]) = _pk;
            }
        }
        if (row_b < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb4 * 4 + 2 + 0], acc[nb4 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dk2_out))[out_b + (long long)(nb4 * 8)]) = _pk;
            }
        }
    }
    #pragma unroll
    for (int kb4 = 0; kb4 < 4; kb4++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_mn[0]), "=r"(a_frag_mn[1]), "=r"(a_frag_mn[2]), "=r"(a_frag_mn[3])
            : "r"(s_a_addr + (unsigned int)((warp_1 * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((kb4 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (warp_1 * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((kb4 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (warp_1 * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb5 = 0; nb5 < 8; nb5++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_du_addr + (unsigned int)((nb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb4 * 16 + lane % 16) * 128 + (nb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb4 * 16 + lane % 16) * 128 + (nb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
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
    for (int nb6 = 0; nb6 < 16; nb6++) {
        #pragma unroll
        for (int e2 = 0; e2 < 2; e2++) {
            int vcol = nb6 * 8 + col_q + e2;
            float _fma_4 = __fmaf_rn(acc[nb6 * 4 + e2], (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_v) + (vcol / 64 * 8192 + (row_a * 128 + vcol % 64 * 2 ^ (row_a * 128 + vcol % 64 * 2 >> 7 & 7) << 4)))[0], db_a);
            db_a = _fma_4;
            float _fma_5 = __fmaf_rn(acc[nb6 * 4 + 2 + e2], (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_v) + (vcol / 64 * 8192 + (row_b * 128 + vcol % 64 * 2 ^ (row_b * 128 + vcol % 64 * 2 >> 7 & 7) << 4)))[0], db_b);
            db_b = _fma_5;
            acc[nb6 * 4 + e2] = acc[nb6 * 4 + e2] * beta_a;
            acc[nb6 * 4 + 2 + e2] = acc[nb6 * 4 + 2 + e2] * beta_b;
        }
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
    __syncthreads();
    #pragma unroll
    for (int nb7 = 0; nb7 < 8; nb7++) {
        #pragma unroll
        for (int e3 = 0; e3 < 2; e3++) {
            int col = nb7 * 8 + col_q + e3;
            float val_a = 0.0f;
            float val_b = 0.0f;
            if (col < row_a) {
                if (row_a < n_valid) {
                    val_a = acc_s[nb7 * 4 + e3];
                }
            }
            if (col < row_b) {
                if (row_b < n_valid) {
                    val_b = acc_s[nb7 * 4 + 2 + e3];
                }
            }
            {
                __nv_bfloat16 _bval_2 = __float2bfloat16_rn(val_a);
                uint16_t _bits_2 = *(uint16_t*)&_bval_2;
                uint32_t _addr_2 = static_cast<uint32_t>((s_da_addr + (unsigned int)(row_a * 128 + col * 2 ^ (row_a * 128 + col * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_2), "h"(_bits_2) : "memory");
            }
            {
                __nv_bfloat16 _bval_3 = __float2bfloat16_rn(val_b);
                uint16_t _bits_3 = *(uint16_t*)&_bval_3;
                uint32_t _addr_3 = static_cast<uint32_t>((s_da_addr + (unsigned int)(row_b * 128 + col * 2 ^ (row_b * 128 + col * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_3), "h"(_bits_3) : "memory");
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int kb5 = 0; kb5 < 4; kb5++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_da_addr + (unsigned int)((kb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb8 = 0; nb8 < 4; nb8++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                : "r"(s_a_addr + (unsigned int)((kb5 * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((nb8 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb5 * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((nb8 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb5 * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
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
    for (int nb9 = 0; nb9 < 8; nb9++) {
        #pragma unroll
        for (int e4 = 0; e4 < 2; e4++) {
            int col2 = nb9 * 8 + col_q + e4;
            {
                __nv_bfloat16 _bval_4 = __float2bfloat16_rn(acc_s[nb9 * 4 + e4]);
                uint16_t _bits_4 = *(uint16_t*)&_bval_4;
                uint32_t _addr_4 = static_cast<uint32_t>((s_da2_addr + (unsigned int)(row_a * 128 + col2 * 2 ^ (row_a * 128 + col2 * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_4), "h"(_bits_4) : "memory");
            }
            {
                __nv_bfloat16 _bval_5 = __float2bfloat16_rn(acc_s[nb9 * 4 + 2 + e4]);
                uint16_t _bits_5 = *(uint16_t*)&_bval_5;
                uint32_t _addr_5 = static_cast<uint32_t>((s_da2_addr + (unsigned int)(row_b * 128 + col2 * 2 ^ (row_b * 128 + col2 * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_5), "h"(_bits_5) : "memory");
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int kb6 = 0; kb6 < 4; kb6++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_mn[0]), "=r"(a_frag_mn[1]), "=r"(a_frag_mn[2]), "=r"(a_frag_mn[3])
            : "r"(s_a_addr + (unsigned int)((warp_1 * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((kb6 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (warp_1 * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((kb6 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (warp_1 * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb10 = 0; nb10 < 4; nb10++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_da2_addr + (unsigned int)((nb10 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb6 * 16 + lane % 16) * 128 + (nb10 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb6 * 16 + lane % 16) * 128 + (nb10 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb10 * 8)[0]), "=f"((acc_s + nb10 * 8)[1]), "=f"((acc_s + nb10 * 8)[2]), "=f"((acc_s + nb10 * 8)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8)[0])), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8)[1])), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8)[2])), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb10 * 8 + 4)[0]), "=f"((acc_s + nb10 * 8 + 4)[1]), "=f"((acc_s + nb10 * 8 + 4)[2]), "=f"((acc_s + nb10 * 8 + 4)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8 + 4)[0])), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8 + 4)[1])), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8 + 4)[2])), "f"(((kb6 == 0) ? 0.0f : (acc_s + nb10 * 8 + 4)[3])));
        }
    }
    __syncthreads();
    #pragma unroll
    for (int nb11 = 0; nb11 < 8; nb11++) {
        #pragma unroll
        for (int e5 = 0; e5 < 2; e5++) {
            int col3 = nb11 * 8 + col_q + e5;
            float g_col = s_g[col3];
            float val_a2 = 0.0f;
            float val_b2 = 0.0f;
            if (col3 < row_a) {
                if (row_a < n_valid) {
                    float _exp2_3 = approx_exp2(g_a - g_col);
                    val_a2 = (-acc_s[nb11 * 4 + e5]) * _exp2_3;
                }
            }
            if (col3 < row_b) {
                if (row_b < n_valid) {
                    float _exp2_4 = approx_exp2(g_b - g_col);
                    val_b2 = (-acc_s[nb11 * 4 + 2 + e5]) * _exp2_4;
                }
            }
            {
                __nv_bfloat16 _bval_6 = __float2bfloat16_rn(val_a2);
                uint16_t _bits_6 = *(uint16_t*)&_bval_6;
                uint32_t _addr_6 = static_cast<uint32_t>((s_da_addr + (unsigned int)(row_a * 128 + col3 * 2 ^ (row_a * 128 + col3 * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_6), "h"(_bits_6) : "memory");
            }
            {
                __nv_bfloat16 _bval_7 = __float2bfloat16_rn(val_b2);
                uint16_t _bits_7 = *(uint16_t*)&_bval_7;
                uint32_t _addr_7 = static_cast<uint32_t>((s_da_addr + (unsigned int)(row_b * 128 + col3 * 2 ^ (row_b * 128 + col3 * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_7), "h"(_bits_7) : "memory");
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int t3 = 0; t3 < 32; t3++) {
        int kb_row = t3 * 2 + tid_2 / 64;
        int kb_col = tid_2 % 64 * 2;
        float kb_scale = s_b[kb_row];
        #pragma unroll
        for (int e9 = 0; e9 < 2; e9++) {
            int kb_off = (kb_col + e9) / 64 * 8192 + (kb_row * 128 + (kb_col + e9) % 64 * 2 ^ (kb_row * 128 + (kb_col + e9) % 64 * 2 >> 7 & 7) << 4);
            {
                __nv_bfloat16 _bval_8 = __float2bfloat16_rn((float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kn) + kb_off)[0] * kb_scale);
                uint16_t _bits_8 = *(uint16_t*)&_bval_8;
                uint32_t _addr_8 = static_cast<uint32_t>(s_kb_addr + (unsigned int)kb_off);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_8), "h"(_bits_8) : "memory");
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int kb7 = 0; kb7 < 8; kb7++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_kn_addr + (unsigned int)((kb7 * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb7 * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb7 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb12 = 0; nb12 < 4; nb12++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                : "r"(s_kn_addr + (unsigned int)((kb7 * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((nb12 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb7 * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((nb12 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb7 * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb12 * 8)[0]), "=f"((acc_s + nb12 * 8)[1]), "=f"((acc_s + nb12 * 8)[2]), "=f"((acc_s + nb12 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]), "f"(((kb7 == 0) ? 0.0f : (acc_s + nb12 * 8)[0])), "f"(((kb7 == 0) ? 0.0f : (acc_s + nb12 * 8)[1])), "f"(((kb7 == 0) ? 0.0f : (acc_s + nb12 * 8)[2])), "f"(((kb7 == 0) ? 0.0f : (acc_s + nb12 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb12 * 8 + 4)[0]), "=f"((acc_s + nb12 * 8 + 4)[1]), "=f"((acc_s + nb12 * 8 + 4)[2]), "=f"((acc_s + nb12 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]), "f"(((kb7 == 0) ? 0.0f : (acc_s + nb12 * 8 + 4)[0])), "f"(((kb7 == 0) ? 0.0f : (acc_s + nb12 * 8 + 4)[1])), "f"(((kb7 == 0) ? 0.0f : (acc_s + nb12 * 8 + 4)[2])), "f"(((kb7 == 0) ? 0.0f : (acc_s + nb12 * 8 + 4)[3])));
        }
    }
    float col_part[16];
    col_part[0] = 0.0f;
    col_part[1] = 0.0f;
    col_part[2] = 0.0f;
    col_part[3] = 0.0f;
    col_part[4] = 0.0f;
    col_part[5] = 0.0f;
    col_part[6] = 0.0f;
    col_part[7] = 0.0f;
    col_part[8] = 0.0f;
    col_part[9] = 0.0f;
    col_part[10] = 0.0f;
    col_part[11] = 0.0f;
    col_part[12] = 0.0f;
    col_part[13] = 0.0f;
    col_part[14] = 0.0f;
    col_part[15] = 0.0f;
    #pragma unroll
    for (int nb13 = 0; nb13 < 8; nb13++) {
        #pragma unroll
        for (int e6 = 0; e6 < 2; e6++) {
            int col4 = nb13 * 8 + col_q + e6;
            float ada_a = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_da) + (col4 / 64 * 8192 + (row_a * 128 + col4 % 64 * 2 ^ (row_a * 128 + col4 % 64 * 2 >> 7 & 7) << 4)))[0] * acc_s[nb13 * 4 + e6] * beta_a;
            float ada_b = (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_da) + (col4 / 64 * 8192 + (row_b * 128 + col4 % 64 * 2 ^ (row_b * 128 + col4 % 64 * 2 >> 7 & 7) << 4)))[0] * acc_s[nb13 * 4 + 2 + e6] * beta_b;
            dg_a = dg_a + ada_a;
            dg_b = dg_b + ada_b;
            col_part[nb13 * 2 + e6] = ada_a + ada_b;
        }
    }
    #pragma unroll
    for (int ci = 0; ci < 16; ci++) {
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, col_part[ci], 4);
        col_part[ci] = col_part[ci] + _shfl_xor_0;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, col_part[ci], 8);
        col_part[ci] = col_part[ci] + _shfl_xor_1;
        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, col_part[ci], 16);
        col_part[ci] = col_part[ci] + _shfl_xor_2;
    }
    if (lane_0 < 4) {
        #pragma unroll
        for (int nb14 = 0; nb14 < 8; nb14++) {
            #pragma unroll
            for (int e7 = 0; e7 < 2; e7++) {
                s_col[warp_1 * 64 + nb14 * 8 + col_q + e7] = col_part[nb14 * 2 + e7];
            }
        }
    }
    #pragma unroll
    for (int kb8 = 0; kb8 < 4; kb8++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_da_addr + (unsigned int)((kb8 * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb8 * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb8 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb15 = 0; nb15 < 8; nb15++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_kn_addr + (unsigned int)((nb15 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb8 * 16 + lane % 16) * 128 + (nb15 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb8 * 16 + lane % 16) * 128 + (nb15 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
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
    for (int nb16 = 0; nb16 < 16; nb16++) {
        #pragma unroll
        for (int e8 = 0; e8 < 2; e8++) {
            int kcol2 = nb16 * 8 + col_q + e8;
            float _fma_6 = __fmaf_rn(acc[nb16 * 4 + e8], (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kn) + (kcol2 / 64 * 8192 + (row_a * 128 + kcol2 % 64 * 2 ^ (row_a * 128 + kcol2 % 64 * 2 >> 7 & 7) << 4)))[0], db_a);
            db_a = _fma_6;
            float _fma_7 = __fmaf_rn(acc[nb16 * 4 + 2 + e8], (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kn) + (kcol2 / 64 * 8192 + (row_b * 128 + kcol2 % 64 * 2 ^ (row_b * 128 + kcol2 % 64 * 2 >> 7 & 7) << 4)))[0], db_b);
            db_b = _fma_7;
            acc[nb16 * 4 + e8] = acc[nb16 * 4 + e8] * beta_a;
            acc[nb16 * 4 + 2 + e8] = acc[nb16 * 4 + 2 + e8] * beta_b;
        }
    }
    #pragma unroll
    for (int kb9 = 0; kb9 < 4; kb9++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_mn[0]), "=r"(a_frag_mn[1]), "=r"(a_frag_mn[2]), "=r"(a_frag_mn[3])
            : "r"(s_da_addr + (unsigned int)((warp_1 * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((kb9 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (warp_1 * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((kb9 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (warp_1 * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb17 = 0; nb17 < 8; nb17++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_kb_addr + (unsigned int)((nb17 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb9 * 16 + lane % 16) * 128 + (nb17 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb9 * 16 + lane % 16) * 128 + (nb17 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
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
    for (int nb18 = 0; nb18 < 16; nb18++) {
        if (row_a < n_valid) {
            {
                uint32_t _bf16x2_bits_9;
                _bf16x2_bits_9 = *reinterpret_cast<const uint32_t*>(dk2_out + out_a + (long long)(nb18 * 8));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&pair[0])[0]), "=f"((&pair[0])[1])
                    : "r"(_bf16x2_bits_9));
            }
            pair[0] = pair[0] + acc[nb18 * 4];
            pair[1] = pair[1] + acc[nb18 * 4 + 1];
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(pair[0 + 0], pair[0 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dk2_out))[out_a + (long long)(nb18 * 8)]) = _pk;
            }
        }
        if (row_b < n_valid) {
            {
                uint32_t _bf16x2_bits_10;
                _bf16x2_bits_10 = *reinterpret_cast<const uint32_t*>(dk2_out + out_b + (long long)(nb18 * 8));
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&pair[0])[0]), "=f"((&pair[0])[1])
                    : "r"(_bf16x2_bits_10));
            }
            pair[0] = pair[0] + acc[nb18 * 4 + 2];
            pair[1] = pair[1] + acc[nb18 * 4 + 3];
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(pair[0 + 0], pair[0 + 1]);
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
    __syncthreads();
    if (lane_0 % 4 == 0) {
        long long gate_a_index = (long long)(tok0 + row_a) * (long long)num_v_heads + (long long)hv;
        long long gate_b_index = (long long)(tok0 + row_b) * (long long)num_v_heads + (long long)hv;
        float colsum_a = s_col[row_a] + s_col[64 + row_a] + s_col[128 + row_a] + s_col[192 + row_a];
        float colsum_b = s_col[row_b] + s_col[64 + row_b] + s_col[128 + row_b] + s_col[192 + row_b];
        if (row_a < n_valid) {
            dbeta_out[gate_a_index] = db_a;
            dg2_out[gate_a_index] = dg_a - colsum_a;
        }
        if (row_b < n_valid) {
            dbeta_out[gate_b_index] = db_b;
            dg2_out[gate_b_index] = dg_b - colsum_b;
        }
    }
}

} // extern "C"
