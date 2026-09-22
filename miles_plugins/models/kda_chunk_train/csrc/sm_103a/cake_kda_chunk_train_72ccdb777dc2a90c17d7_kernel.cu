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
#define TMEM_NCOLS 448
#define TMEM_ACC_DQ_OFFSET 0
#define TMEM_ACC_DK_OFFSET 128
#define TMEM_ACC_DW_OFFSET 256
#define TMEM_ACC_DA_OFFSET 384
#define TMEM_ACC_DVB_OFFSET 0
#define TMEM_ACC_DKGB_OFFSET 128
#define TMEM_ACC_DA1_OFFSET 256
#define TMEM_ACC_DA2_OFFSET 320
#define NUM_RING_PIPE_STAGES 2
#define NUM_ONE_STAGES 1
#define SMEM_SMEM_V0_OFF 1024
#define SMEM_SMEM_V0_STAGE_BYTES 8192
#define SMEM_SMEM_V0_STRIDE 57344
#define SMEM_SMEM_V1_OFF 17408
#define SMEM_SMEM_V1_STAGE_BYTES 8192
#define SMEM_SMEM_V1_STRIDE 57344
#define SMEM_SMEM_V2_OFF 33792
#define SMEM_SMEM_V2_STAGE_BYTES 8192
#define SMEM_SMEM_V2_STRIDE 57344
#define SMEM_SMEM_V3_OFF 1024
#define SMEM_SMEM_V3_STAGE_BYTES 16384
#define SMEM_SMEM_V3_STRIDE 57344
#define SMEM_SMEM_V4_OFF 17408
#define SMEM_SMEM_V4_STAGE_BYTES 16384
#define SMEM_SMEM_V4_STRIDE 57344
#define SMEM_SMEM_V5_OFF 33792
#define SMEM_SMEM_V5_STAGE_BYTES 16384
#define SMEM_SMEM_V5_STRIDE 57344
#define SMEM_DV2_MN_OFF 33792
#define SMEM_DV2_MN_STAGE_BYTES 16384
#define SMEM_DV2_MN_STRIDE 57344
#define SMEM_AKK_K_OFF 50176
#define SMEM_AKK_K_STAGE_BYTES 8192
#define SMEM_AKK_K_STRIDE 57344
#define SMEM_AKK_MN_OFF 50176
#define SMEM_AKK_MN_STAGE_BYTES 8192
#define SMEM_AKK_MN_STRIDE 57344
#define SMEM_DAB_K_OFF 33792
#define SMEM_DAB_K_STAGE_BYTES 8192
#define SMEM_DAB_K_STRIDE 57344
#define SMEM_DA1B_MN_OFF 41984
#define SMEM_DA1B_MN_STAGE_BYTES 8192
#define SMEM_DA1B_MN_STRIDE 57344
#define SMEM_DA1B_W_OFF 41984
#define SMEM_DA1B_W_STAGE_BYTES 8192
#define SMEM_DA1B_W_STRIDE 57344
#define SMEM_RING_F32_OFF 1024
#define SMEM_RING_F32_STAGE_BYTES 114688
#define SMEM_RING_F32_STRIDE 114688
#define SMEM_V_P_OFF 115712
#define SMEM_V_P_STAGE_BYTES 8192
#define SMEM_V_P_STRIDE 8192
#define SMEM_V_K_OFF 115712
#define SMEM_V_K_STAGE_BYTES 16384
#define SMEM_V_K_STRIDE 16384
#define SMEM_H_P_OFF 132096
#define SMEM_H_P_STAGE_BYTES 16384
#define SMEM_H_P_STRIDE 16384
#define SMEM_DH_P_OFF 164864
#define SMEM_DH_P_STAGE_BYTES 16384
#define SMEM_DH_P_STRIDE 16384
#define SMEM_H_K_OFF 132096
#define SMEM_H_K_STAGE_BYTES 32768
#define SMEM_H_K_STRIDE 32768
#define SMEM_DH_K_OFF 164864
#define SMEM_DH_K_STAGE_BYTES 32768
#define SMEM_DH_K_STRIDE 32768
#define SMEM_DWB_K_OFF 197632
#define SMEM_DWB_K_STAGE_BYTES 16384
#define SMEM_DWB_K_STRIDE 16384
#define SMEM_DWB_MN_OFF 197632
#define SMEM_DWB_MN_STAGE_BYTES 16384
#define SMEM_DWB_MN_STRIDE 16384
#define SMEM_KGB_K_OFF 214016
#define SMEM_KGB_K_STAGE_BYTES 16384
#define SMEM_KGB_K_STRIDE 16384
#define SMEM_DGK_S_OFF 230400
#define SMEM_DGK_S_STAGE_BYTES 512
#define SMEM_DGK_S_STRIDE 512
#define SMEM_DBP_S_OFF 230912
#define SMEM_DBP_S_STAGE_BYTES 1024
#define SMEM_DBP_S_STRIDE 1024
#define SMEM_TOTAL 231936
#define THREADS 576

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
}


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

extern "C" {

__global__ __launch_bounds__(576, 1) void
kernel_cake_kda_chunk_train_72ccdb777dc2a90c17d7(CakeTensorMap const* do_tma, CakeTensorMap const* vn_tma, CakeTensorMap const* dv2_tma, CakeTensorMap const* v_tma, CakeTensorMap const* h_tma, CakeTensorMap const* dh_tma, CakeTensorMap const* akk_tma, __nv_bfloat16* __restrict__ h_ptr, __nv_bfloat16* __restrict__ dh_ptr, __nv_bfloat16* __restrict__ k_ptr, __nv_bfloat16* __restrict__ q_ptr, __nv_bfloat16* __restrict__ v_ptr, float* __restrict__ gk, float* __restrict__ beta, float* __restrict__ dq_out, float* __restrict__ dk_out, float* __restrict__ dg_out, float* __restrict__ db_out, float* __restrict__ dAkk_out, __nv_bfloat16* __restrict__ dv_out, int num_heads, int num_items, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define tiles_full_addr (mbar_base + 0)
    #define tiles_empty_addr (mbar_base + 16)
    #define hdv_full_addr (mbar_base + 32)
    #define hdv_empty_addr (mbar_base + 40)
    #define a_ready_addr (mbar_base + 48)
    #define a_drained_addr (mbar_base + 56)
    #define b_ready_addr (mbar_base + 64)
    #define b_drained_addr (mbar_base + 72)
    #define c_ready_addr (mbar_base + 80)
    #define c_drained_addr (mbar_base + 88)
    #define d_ready_addr (mbar_base + 96)
    #define d_drained_addr (mbar_base + 104)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(do_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(vn_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(dv2_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(v_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(h_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(dh_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(akk_tma)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_v0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v0_addr = smem + 1024;
    __nv_bfloat16* smem_v1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_v1_addr = smem + 17408;
    __nv_bfloat16* smem_v2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_v2_addr = smem + 33792;
    __nv_bfloat16* smem_v3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v3_addr = smem + 1024;
    __nv_bfloat16* smem_v4 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_v4_addr = smem + 17408;
    __nv_bfloat16* smem_v5 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_v5_addr = smem + 33792;
    __nv_bfloat16* dv2_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int dv2_mn_addr = smem + 33792;
    __nv_bfloat16* akk_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int akk_k_addr = smem + 50176;
    __nv_bfloat16* akk_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int akk_mn_addr = smem + 50176;
    __nv_bfloat16* dab_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int dab_k_addr = smem + 33792;
    __nv_bfloat16* da1b_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int da1b_mn_addr = smem + 41984;
    __nv_bfloat16* da1b_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int da1b_w_addr = smem + 41984;
    float* ring_f32 = reinterpret_cast<float*>(smem_raw + 1024);
    const int ring_f32_addr = smem + 1024;
    __nv_bfloat16* v_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int v_p_addr = smem + 115712;
    __nv_bfloat16* v_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 115712);
    const int v_k_addr = smem + 115712;
    __nv_bfloat16* h_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int h_p_addr = smem + 132096;
    __nv_bfloat16* dh_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int dh_p_addr = smem + 164864;
    __nv_bfloat16* h_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int h_k_addr = smem + 132096;
    __nv_bfloat16* dh_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int dh_k_addr = smem + 164864;
    __nv_bfloat16* dwb_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int dwb_k_addr = smem + 197632;
    __nv_bfloat16* dwb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int dwb_mn_addr = smem + 197632;
    __nv_bfloat16* kgb_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 214016);
    const int kgb_k_addr = smem + 214016;
    float* dgk_s = reinterpret_cast<float*>(smem_raw + 230400);
    const int dgk_s_addr = smem + 230400;
    float* dbp_s = reinterpret_cast<float*>(smem_raw + 230912);
    const int dbp_s_addr = smem + 230912;

    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 14 barriers)
    // Mbarriers at smem_raw[0..112)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring_pipe' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // tiles_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'one' ---
            // hdv_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // hdv_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // a_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // a_drained: 1 barriers, init_count=512
            mbarrier_init(smem + 56, 512);
            // b_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // b_drained: 1 barriers, init_count=512
            mbarrier_init(smem + 72, 512);
            // c_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // c_drained: 1 barriers, init_count=512
            mbarrier_init(smem + 88, 512);
            // d_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // d_drained: 1 barriers, init_count=512
            mbarrier_init(smem + 104, 512);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 448 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 112);
    if (warp == 0) {
        int _tmem_hold = smem + 112;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_dq = taddr;
    const int tmem_acc_dk = taddr + 128;
    const int tmem_acc_dw = taddr + 256;
    const int tmem_acc_da = taddr + 384;
    const int tmem_acc_dvb = taddr;
    const int tmem_acc_dkgb = taddr + 128;
    const int tmem_acc_da1 = taddr + 256;
    const int tmem_acc_da2 = taddr + 320;

    // ---- Role: compute ----
    if (warp <= 15) {
        { // compute_main
            int wq = warp % 4;
            int cblk = warp / 4;
            int lane_half = lane >> 4;
            int t_row = wq * 16 + (lane & 15);
            int row_base = wq * 32 << 16;
            int tid_1 = warp * 32 + lane;
            int it_i = 0;
            unsigned int _phase_a_ready_0 = 0;
            unsigned int _phase_b_ready_0 = 0;
            unsigned int _phase_c_ready_0 = 0;
            unsigned int _phase_d_ready_0 = 0;
            #pragma unroll 1
            for (unsigned int it = bid; it < num_items; it += num_bids) {
                int chunk = it / (unsigned int)num_heads;
                int head = it % (unsigned int)num_heads;
                int row0 = chunk * 64;
                int hrow0 = (chunk * num_heads + head) * 128;
                int slot = it_i & 1;
                int kbase = slot * 14336;
                int grow = row0 + t_row;
                int rbase = (grow * num_heads + head) * 128;
                int gn_base = ((row0 + 64 - 1) * num_heads + head) * 128;
                float beta_t = beta[grow * num_heads + head];
                int ext_row = chunk_bos[chunk] + t_row;
                int clen = chunk_len[chunk];
                float dgk_inter = 0.0f;
                if (tid_1 < 128) {
                    int hrow = (hrow0 + tid_1) * 128;
                    #pragma unroll
                    for (int seg = 0; seg < 8; seg++) {
                        float hv[16];
                        float dhv[16];
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(h_ptr + hrow + seg * 16);
                            uint4 _vld_0[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                _vld_0[_blk] = _vptr_0[_blk];
                                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&hv[0 + _blk * 8 + _pair * 2])[0]), "=f"((&hv[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_0[_pair]));
                                }
                            }
                        }
                        {
                            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(dh_ptr + hrow + seg * 16);
                            uint4 _vld_1[2];
                            #pragma unroll
                            for (int _blk = 0; _blk < 2; _blk++) {
                                _vld_1[_blk] = _vptr_1[_blk];
                                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                                #pragma unroll
                                for (int _pair = 0; _pair < 4; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&dhv[0 + _blk * 8 + _pair * 2])[0]), "=f"((&dhv[0 + _blk * 8 + _pair * 2])[1])
                                        : "r"(_vpairs_1[_pair]));
                                }
                            }
                        }
                        #pragma unroll
                        for (int j = 0; j < 16; j++) {
                            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(hv[j] * dhv[j]);
                            float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                            dgk_inter = dgk_inter + _cvt_f32_0;
                        }
                    }
                    float _exp2_0 = approx_exp2(gk[gn_base + tid_1]);
                    dgk_inter = dgk_inter * _exp2_0;
                }
                mbarrier_wait(a_ready_addr, _phase_a_ready_0);
                _phase_a_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int _one_a = 0; _one_a < 1; _one_a++) {
                    int blk = cblk;
                    int col = blk * 32 + lane_half * 16;
                    float _tmem_load_0[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                        : "r"(taddr + (unsigned int)row_base + (unsigned int)(blk * 32)));
                    float _tmem_load_1[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                        : "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(blk * 32)));
                    float _tmem_load_2[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                        : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(blk * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float gkv[16];
                    float gnv[16];
                    float kv[16];
                    {
                        unsigned _ldv8_2_0;
                        unsigned _ldv8_2_1;
                        unsigned _ldv8_2_2;
                        unsigned _ldv8_2_3;
                        unsigned _ldv8_2_4;
                        unsigned _ldv8_2_5;
                        unsigned _ldv8_2_6;
                        unsigned _ldv8_2_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_2_0), "=r"(_ldv8_2_1), "=r"(_ldv8_2_2), "=r"(_ldv8_2_3), "=r"(_ldv8_2_4), "=r"(_ldv8_2_5), "=r"(_ldv8_2_6), "=r"(_ldv8_2_7) : "l"((const void*)(gk + (rbase + col))) : "memory");
                        gkv[0 + 0] = __uint_as_float(_ldv8_2_0);
                        gkv[0 + 1] = __uint_as_float(_ldv8_2_1);
                        gkv[0 + 2] = __uint_as_float(_ldv8_2_2);
                        gkv[0 + 3] = __uint_as_float(_ldv8_2_3);
                        gkv[0 + 4] = __uint_as_float(_ldv8_2_4);
                        gkv[0 + 5] = __uint_as_float(_ldv8_2_5);
                        gkv[0 + 6] = __uint_as_float(_ldv8_2_6);
                        gkv[0 + 7] = __uint_as_float(_ldv8_2_7);
                    }
                    {
                        unsigned _ldv8_3_0;
                        unsigned _ldv8_3_1;
                        unsigned _ldv8_3_2;
                        unsigned _ldv8_3_3;
                        unsigned _ldv8_3_4;
                        unsigned _ldv8_3_5;
                        unsigned _ldv8_3_6;
                        unsigned _ldv8_3_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(gk + (rbase + col + 8))) : "memory");
                        gkv[8 + 0] = __uint_as_float(_ldv8_3_0);
                        gkv[8 + 1] = __uint_as_float(_ldv8_3_1);
                        gkv[8 + 2] = __uint_as_float(_ldv8_3_2);
                        gkv[8 + 3] = __uint_as_float(_ldv8_3_3);
                        gkv[8 + 4] = __uint_as_float(_ldv8_3_4);
                        gkv[8 + 5] = __uint_as_float(_ldv8_3_5);
                        gkv[8 + 6] = __uint_as_float(_ldv8_3_6);
                        gkv[8 + 7] = __uint_as_float(_ldv8_3_7);
                    }
                    {
                        unsigned _ldv8_4_0;
                        unsigned _ldv8_4_1;
                        unsigned _ldv8_4_2;
                        unsigned _ldv8_4_3;
                        unsigned _ldv8_4_4;
                        unsigned _ldv8_4_5;
                        unsigned _ldv8_4_6;
                        unsigned _ldv8_4_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3), "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7) : "l"((const void*)(gk + (gn_base + col))) : "memory");
                        gnv[0 + 0] = __uint_as_float(_ldv8_4_0);
                        gnv[0 + 1] = __uint_as_float(_ldv8_4_1);
                        gnv[0 + 2] = __uint_as_float(_ldv8_4_2);
                        gnv[0 + 3] = __uint_as_float(_ldv8_4_3);
                        gnv[0 + 4] = __uint_as_float(_ldv8_4_4);
                        gnv[0 + 5] = __uint_as_float(_ldv8_4_5);
                        gnv[0 + 6] = __uint_as_float(_ldv8_4_6);
                        gnv[0 + 7] = __uint_as_float(_ldv8_4_7);
                    }
                    {
                        unsigned _ldv8_5_0;
                        unsigned _ldv8_5_1;
                        unsigned _ldv8_5_2;
                        unsigned _ldv8_5_3;
                        unsigned _ldv8_5_4;
                        unsigned _ldv8_5_5;
                        unsigned _ldv8_5_6;
                        unsigned _ldv8_5_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_5_0), "=r"(_ldv8_5_1), "=r"(_ldv8_5_2), "=r"(_ldv8_5_3), "=r"(_ldv8_5_4), "=r"(_ldv8_5_5), "=r"(_ldv8_5_6), "=r"(_ldv8_5_7) : "l"((const void*)(gk + (gn_base + col + 8))) : "memory");
                        gnv[8 + 0] = __uint_as_float(_ldv8_5_0);
                        gnv[8 + 1] = __uint_as_float(_ldv8_5_1);
                        gnv[8 + 2] = __uint_as_float(_ldv8_5_2);
                        gnv[8 + 3] = __uint_as_float(_ldv8_5_3);
                        gnv[8 + 4] = __uint_as_float(_ldv8_5_4);
                        gnv[8 + 5] = __uint_as_float(_ldv8_5_5);
                        gnv[8 + 6] = __uint_as_float(_ldv8_5_6);
                        gnv[8 + 7] = __uint_as_float(_ldv8_5_7);
                    }
                    {
                        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(k_ptr + rbase + col);
                        uint4 _vld_6[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_6[_blk] = _vptr_6[_blk];
                            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&kv[0 + _blk * 8 + _pair * 2])[0]), "=f"((&kv[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_6[_pair]));
                            }
                        }
                    }
                    float dqv[16];
                    float dkv[16];
                    float kdkv[16];
                    float dwbv[16];
                    float kgv[16];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 16; j_1++) {
                        float _exp2_1 = approx_exp2(gkv[j_1]);
                        float e_g = _exp2_1;
                        dqv[j_1] = _tmem_load_0[j_1] * e_g * scale;
                        float _exp2_2 = approx_exp2(gnv[j_1] - gkv[j_1]);
                        dkv[j_1] = _tmem_load_1[j_1] * _exp2_2;
                        kdkv[j_1] = kv[j_1] * dkv[j_1];
                        __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_tmem_load_2[j_1]);
                        float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                        dwbv[j_1] = -_cvt_f32_1;
                        kgv[j_1] = kv[j_1] * e_g;
                    }
                    float dqv_lo[8];
                    float dqv_hi[8];
                    #pragma unroll
                    for (int jj = 0; jj < 8; jj++) {
                        dqv_lo[jj] = dqv[jj];
                        dqv_hi[jj] = dqv[jj + 8];
                    }
                    {
                        unsigned _stv8_7_0 = __float_as_uint(dqv_lo[0 + 0]);
                        unsigned _stv8_7_1 = __float_as_uint(dqv_lo[0 + 1]);
                        unsigned _stv8_7_2 = __float_as_uint(dqv_lo[0 + 2]);
                        unsigned _stv8_7_3 = __float_as_uint(dqv_lo[0 + 3]);
                        unsigned _stv8_7_4 = __float_as_uint(dqv_lo[0 + 4]);
                        unsigned _stv8_7_5 = __float_as_uint(dqv_lo[0 + 5]);
                        unsigned _stv8_7_6 = __float_as_uint(dqv_lo[0 + 6]);
                        unsigned _stv8_7_7 = __float_as_uint(dqv_lo[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dq_out + (rbase + col) + (0))), "r"(_stv8_7_0), "r"(_stv8_7_1), "r"(_stv8_7_2), "r"(_stv8_7_3), "r"(_stv8_7_4), "r"(_stv8_7_5), "r"(_stv8_7_6), "r"(_stv8_7_7) : "memory");
                    }
                    {
                        unsigned _stv8_8_0 = __float_as_uint(dqv_hi[0 + 0]);
                        unsigned _stv8_8_1 = __float_as_uint(dqv_hi[0 + 1]);
                        unsigned _stv8_8_2 = __float_as_uint(dqv_hi[0 + 2]);
                        unsigned _stv8_8_3 = __float_as_uint(dqv_hi[0 + 3]);
                        unsigned _stv8_8_4 = __float_as_uint(dqv_hi[0 + 4]);
                        unsigned _stv8_8_5 = __float_as_uint(dqv_hi[0 + 5]);
                        unsigned _stv8_8_6 = __float_as_uint(dqv_hi[0 + 6]);
                        unsigned _stv8_8_7 = __float_as_uint(dqv_hi[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dq_out + (rbase + col + 8) + (0))), "r"(_stv8_8_0), "r"(_stv8_8_1), "r"(_stv8_8_2), "r"(_stv8_8_3), "r"(_stv8_8_4), "r"(_stv8_8_5), "r"(_stv8_8_6), "r"(_stv8_8_7) : "memory");
                    }
                    float dkv_lo[8];
                    float dkv_hi[8];
                    #pragma unroll
                    for (int jj_1 = 0; jj_1 < 8; jj_1++) {
                        dkv_lo[jj_1] = dkv[jj_1];
                        dkv_hi[jj_1] = dkv[jj_1 + 8];
                    }
                    {
                        unsigned _stv8_9_0 = __float_as_uint(dkv_lo[0 + 0]);
                        unsigned _stv8_9_1 = __float_as_uint(dkv_lo[0 + 1]);
                        unsigned _stv8_9_2 = __float_as_uint(dkv_lo[0 + 2]);
                        unsigned _stv8_9_3 = __float_as_uint(dkv_lo[0 + 3]);
                        unsigned _stv8_9_4 = __float_as_uint(dkv_lo[0 + 4]);
                        unsigned _stv8_9_5 = __float_as_uint(dkv_lo[0 + 5]);
                        unsigned _stv8_9_6 = __float_as_uint(dkv_lo[0 + 6]);
                        unsigned _stv8_9_7 = __float_as_uint(dkv_lo[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dk_out + (rbase + col) + (0))), "r"(_stv8_9_0), "r"(_stv8_9_1), "r"(_stv8_9_2), "r"(_stv8_9_3), "r"(_stv8_9_4), "r"(_stv8_9_5), "r"(_stv8_9_6), "r"(_stv8_9_7) : "memory");
                    }
                    {
                        unsigned _stv8_10_0 = __float_as_uint(dkv_hi[0 + 0]);
                        unsigned _stv8_10_1 = __float_as_uint(dkv_hi[0 + 1]);
                        unsigned _stv8_10_2 = __float_as_uint(dkv_hi[0 + 2]);
                        unsigned _stv8_10_3 = __float_as_uint(dkv_hi[0 + 3]);
                        unsigned _stv8_10_4 = __float_as_uint(dkv_hi[0 + 4]);
                        unsigned _stv8_10_5 = __float_as_uint(dkv_hi[0 + 5]);
                        unsigned _stv8_10_6 = __float_as_uint(dkv_hi[0 + 6]);
                        unsigned _stv8_10_7 = __float_as_uint(dkv_hi[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dk_out + (rbase + col + 8) + (0))), "r"(_stv8_10_0), "r"(_stv8_10_1), "r"(_stv8_10_2), "r"(_stv8_10_3), "r"(_stv8_10_4), "r"(_stv8_10_5), "r"(_stv8_10_6), "r"(_stv8_10_7) : "memory");
                    }
                    int kchunk = col >> 2;
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ring_f32_addr + (unsigned int)((kbase + t_row * 128) * 4) + (unsigned int)((kchunk + q ^ t_row & 7) << 4)), "f"(kdkv[q * 4]), "f"(kdkv[q * 4 + 1]), "f"(kdkv[q * 4 + 2]), "f"(kdkv[q * 4 + 3]) : "memory");
                    }
                    uint32_t dwbv_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dwbv[_lp*2 + 0], dwbv[_lp*2+1 + 0]));
                        dwbv_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t kgv_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kgv[_lp*2 + 0], kgv[_lp*2+1 + 0]));
                        kgv_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_1 = 0; q_1 < 2; q_1++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dwb_k_addr + (unsigned int)((col + q_1 * 8) / 64 * 8192 + t_row * 128 + (col + q_1 * 8) % 64 * 2 ^ ((col + q_1 * 8) / 64 * 8192 + t_row * 128 + (col + q_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(dwbv_bf16[q_1 * 4]), "r"(dwbv_bf16[q_1 * 4 + 1]), "r"(dwbv_bf16[q_1 * 4 + 2]), "r"(dwbv_bf16[q_1 * 4 + 3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((kgb_k_addr + (unsigned int)((col + q_1 * 8) / 64 * 8192 + t_row * 128 + (col + q_1 * 8) % 64 * 2 ^ ((col + q_1 * 8) / 64 * 8192 + t_row * 128 + (col + q_1 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(kgv_bf16[q_1 * 4]), "r"(kgv_bf16[q_1 * 4 + 1]), "r"(kgv_bf16[q_1 * 4 + 2]), "r"(kgv_bf16[q_1 * 4 + 3]) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(a_drained_addr);
                asm volatile("barrier.sync 8, 512;" ::: "memory");
                if (tid_1 < 128) {
                    float colsum = 0.0f;
                    #pragma unroll 8
                    for (int t = 0; t < 64; t++) {
                        colsum = colsum + ring_f32[kbase + t * 128 + (((tid_1 >> 2 ^ t & 7) << 2) + (tid_1 & 3))];
                    }
                    dgk_s[tid_1] = dgk_inter + colsum;
                }
                asm volatile("barrier.sync 8, 512;" ::: "memory");
                mbarrier_wait(b_ready_addr, _phase_b_ready_0);
                _phase_b_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float db_acc = 0.0f;
                #pragma unroll
                for (int _one_b = 0; _one_b < 1; _one_b++) {
                    int blk_1 = cblk;
                    int col_1 = blk_1 * 32 + lane_half * 16;
                    float _tmem_load_3[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                        : "r"(taddr + (unsigned int)row_base + (unsigned int)(blk_1 * 32)));
                    float _tmem_load_4[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                        : "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(blk_1 * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    float gkv2[16];
                    float kv2[16];
                    float qv2[16];
                    float vv2[16];
                    float dq2[16];
                    float dk2[16];
                    {
                        unsigned _ldv8_11_0;
                        unsigned _ldv8_11_1;
                        unsigned _ldv8_11_2;
                        unsigned _ldv8_11_3;
                        unsigned _ldv8_11_4;
                        unsigned _ldv8_11_5;
                        unsigned _ldv8_11_6;
                        unsigned _ldv8_11_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_11_0), "=r"(_ldv8_11_1), "=r"(_ldv8_11_2), "=r"(_ldv8_11_3), "=r"(_ldv8_11_4), "=r"(_ldv8_11_5), "=r"(_ldv8_11_6), "=r"(_ldv8_11_7) : "l"((const void*)(gk + (rbase + col_1))) : "memory");
                        gkv2[0 + 0] = __uint_as_float(_ldv8_11_0);
                        gkv2[0 + 1] = __uint_as_float(_ldv8_11_1);
                        gkv2[0 + 2] = __uint_as_float(_ldv8_11_2);
                        gkv2[0 + 3] = __uint_as_float(_ldv8_11_3);
                        gkv2[0 + 4] = __uint_as_float(_ldv8_11_4);
                        gkv2[0 + 5] = __uint_as_float(_ldv8_11_5);
                        gkv2[0 + 6] = __uint_as_float(_ldv8_11_6);
                        gkv2[0 + 7] = __uint_as_float(_ldv8_11_7);
                    }
                    {
                        unsigned _ldv8_12_0;
                        unsigned _ldv8_12_1;
                        unsigned _ldv8_12_2;
                        unsigned _ldv8_12_3;
                        unsigned _ldv8_12_4;
                        unsigned _ldv8_12_5;
                        unsigned _ldv8_12_6;
                        unsigned _ldv8_12_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_12_0), "=r"(_ldv8_12_1), "=r"(_ldv8_12_2), "=r"(_ldv8_12_3), "=r"(_ldv8_12_4), "=r"(_ldv8_12_5), "=r"(_ldv8_12_6), "=r"(_ldv8_12_7) : "l"((const void*)(gk + (rbase + col_1 + 8))) : "memory");
                        gkv2[8 + 0] = __uint_as_float(_ldv8_12_0);
                        gkv2[8 + 1] = __uint_as_float(_ldv8_12_1);
                        gkv2[8 + 2] = __uint_as_float(_ldv8_12_2);
                        gkv2[8 + 3] = __uint_as_float(_ldv8_12_3);
                        gkv2[8 + 4] = __uint_as_float(_ldv8_12_4);
                        gkv2[8 + 5] = __uint_as_float(_ldv8_12_5);
                        gkv2[8 + 6] = __uint_as_float(_ldv8_12_6);
                        gkv2[8 + 7] = __uint_as_float(_ldv8_12_7);
                    }
                    {
                        const uint4* _vptr_13 = reinterpret_cast<const uint4*>(k_ptr + rbase + col_1);
                        uint4 _vld_13[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_13[_blk] = _vptr_13[_blk];
                            uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&kv2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&kv2[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_13[_pair]));
                            }
                        }
                    }
                    {
                        const uint4* _vptr_14 = reinterpret_cast<const uint4*>(q_ptr + rbase + col_1);
                        uint4 _vld_14[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_14[_blk] = _vptr_14[_blk];
                            uint32_t* _vpairs_14 = reinterpret_cast<uint32_t*>(&_vld_14[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&qv2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&qv2[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_14[_pair]));
                            }
                        }
                    }
                    {
                        const uint4* _vptr_15 = reinterpret_cast<const uint4*>(v_ptr + rbase + col_1);
                        uint4 _vld_15[2];
                        #pragma unroll
                        for (int _blk = 0; _blk < 2; _blk++) {
                            _vld_15[_blk] = _vptr_15[_blk];
                            uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&vv2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&vv2[0 + _blk * 8 + _pair * 2])[1])
                                    : "r"(_vpairs_15[_pair]));
                            }
                        }
                    }
                    {
                        unsigned _ldv8_16_0;
                        unsigned _ldv8_16_1;
                        unsigned _ldv8_16_2;
                        unsigned _ldv8_16_3;
                        unsigned _ldv8_16_4;
                        unsigned _ldv8_16_5;
                        unsigned _ldv8_16_6;
                        unsigned _ldv8_16_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_16_0), "=r"(_ldv8_16_1), "=r"(_ldv8_16_2), "=r"(_ldv8_16_3), "=r"(_ldv8_16_4), "=r"(_ldv8_16_5), "=r"(_ldv8_16_6), "=r"(_ldv8_16_7) : "l"((const void*)(dq_out + (rbase + col_1))) : "memory");
                        dq2[0 + 0] = __uint_as_float(_ldv8_16_0);
                        dq2[0 + 1] = __uint_as_float(_ldv8_16_1);
                        dq2[0 + 2] = __uint_as_float(_ldv8_16_2);
                        dq2[0 + 3] = __uint_as_float(_ldv8_16_3);
                        dq2[0 + 4] = __uint_as_float(_ldv8_16_4);
                        dq2[0 + 5] = __uint_as_float(_ldv8_16_5);
                        dq2[0 + 6] = __uint_as_float(_ldv8_16_6);
                        dq2[0 + 7] = __uint_as_float(_ldv8_16_7);
                    }
                    {
                        unsigned _ldv8_17_0;
                        unsigned _ldv8_17_1;
                        unsigned _ldv8_17_2;
                        unsigned _ldv8_17_3;
                        unsigned _ldv8_17_4;
                        unsigned _ldv8_17_5;
                        unsigned _ldv8_17_6;
                        unsigned _ldv8_17_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_17_0), "=r"(_ldv8_17_1), "=r"(_ldv8_17_2), "=r"(_ldv8_17_3), "=r"(_ldv8_17_4), "=r"(_ldv8_17_5), "=r"(_ldv8_17_6), "=r"(_ldv8_17_7) : "l"((const void*)(dq_out + (rbase + col_1 + 8))) : "memory");
                        dq2[8 + 0] = __uint_as_float(_ldv8_17_0);
                        dq2[8 + 1] = __uint_as_float(_ldv8_17_1);
                        dq2[8 + 2] = __uint_as_float(_ldv8_17_2);
                        dq2[8 + 3] = __uint_as_float(_ldv8_17_3);
                        dq2[8 + 4] = __uint_as_float(_ldv8_17_4);
                        dq2[8 + 5] = __uint_as_float(_ldv8_17_5);
                        dq2[8 + 6] = __uint_as_float(_ldv8_17_6);
                        dq2[8 + 7] = __uint_as_float(_ldv8_17_7);
                    }
                    {
                        unsigned _ldv8_18_0;
                        unsigned _ldv8_18_1;
                        unsigned _ldv8_18_2;
                        unsigned _ldv8_18_3;
                        unsigned _ldv8_18_4;
                        unsigned _ldv8_18_5;
                        unsigned _ldv8_18_6;
                        unsigned _ldv8_18_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_18_0), "=r"(_ldv8_18_1), "=r"(_ldv8_18_2), "=r"(_ldv8_18_3), "=r"(_ldv8_18_4), "=r"(_ldv8_18_5), "=r"(_ldv8_18_6), "=r"(_ldv8_18_7) : "l"((const void*)(dk_out + (rbase + col_1))) : "memory");
                        dk2[0 + 0] = __uint_as_float(_ldv8_18_0);
                        dk2[0 + 1] = __uint_as_float(_ldv8_18_1);
                        dk2[0 + 2] = __uint_as_float(_ldv8_18_2);
                        dk2[0 + 3] = __uint_as_float(_ldv8_18_3);
                        dk2[0 + 4] = __uint_as_float(_ldv8_18_4);
                        dk2[0 + 5] = __uint_as_float(_ldv8_18_5);
                        dk2[0 + 6] = __uint_as_float(_ldv8_18_6);
                        dk2[0 + 7] = __uint_as_float(_ldv8_18_7);
                    }
                    {
                        unsigned _ldv8_19_0;
                        unsigned _ldv8_19_1;
                        unsigned _ldv8_19_2;
                        unsigned _ldv8_19_3;
                        unsigned _ldv8_19_4;
                        unsigned _ldv8_19_5;
                        unsigned _ldv8_19_6;
                        unsigned _ldv8_19_7;
                        asm volatile(
                            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(_ldv8_19_0), "=r"(_ldv8_19_1), "=r"(_ldv8_19_2), "=r"(_ldv8_19_3), "=r"(_ldv8_19_4), "=r"(_ldv8_19_5), "=r"(_ldv8_19_6), "=r"(_ldv8_19_7) : "l"((const void*)(dk_out + (rbase + col_1 + 8))) : "memory");
                        dk2[8 + 0] = __uint_as_float(_ldv8_19_0);
                        dk2[8 + 1] = __uint_as_float(_ldv8_19_1);
                        dk2[8 + 2] = __uint_as_float(_ldv8_19_2);
                        dk2[8 + 3] = __uint_as_float(_ldv8_19_3);
                        dk2[8 + 4] = __uint_as_float(_ldv8_19_4);
                        dk2[8 + 5] = __uint_as_float(_ldv8_19_5);
                        dk2[8 + 6] = __uint_as_float(_ldv8_19_6);
                        dk2[8 + 7] = __uint_as_float(_ldv8_19_7);
                    }
                    float dvo[16];
                    float dgv[16];
                    float dkf[16];
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 16; j_2++) {
                        float _exp2_3 = approx_exp2(gkv2[j_2]);
                        float e_g2 = _exp2_3;
                        float kgf = kv2[j_2] * e_g2;
                        dvo[j_2] = _tmem_load_3[j_2] * beta_t;
                        db_acc = db_acc + _tmem_load_3[j_2] * vv2[j_2];
                        db_acc = db_acc + _tmem_load_4[j_2] * kgf;
                        float dgj = qv2[j_2] * dq2[j_2] - kv2[j_2] * dk2[j_2] + kgf * _tmem_load_4[j_2] * beta_t;
                        if (t_row == 63) {
                            dgj = dgj + dgk_s[col_1 + j_2];
                        }
                        dgv[j_2] = dgj;
                        dkf[j_2] = dk2[j_2] + _tmem_load_4[j_2] * e_g2 * beta_t;
                    }
                    if (t_row < clen) {
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(dvo[0 + 0], dvo[0 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(dvo[0 + 2], dvo[0 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(dvo[0 + 4], dvo[0 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(dvo[0 + 6], dvo[0 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(dvo[0 + 8], dvo[0 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(dvo[0 + 10], dvo[0 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(dvo[0 + 12], dvo[0 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(dvo[0 + 14], dvo[0 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(dv_out + ((ext_row * num_heads + head) * 128 + col_1)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                    }
                    float dgv_lo[8];
                    float dgv_hi[8];
                    #pragma unroll
                    for (int jj_2 = 0; jj_2 < 8; jj_2++) {
                        dgv_lo[jj_2] = dgv[jj_2];
                        dgv_hi[jj_2] = dgv[jj_2 + 8];
                    }
                    {
                        unsigned _stv8_20_0 = __float_as_uint(dgv_lo[0 + 0]);
                        unsigned _stv8_20_1 = __float_as_uint(dgv_lo[0 + 1]);
                        unsigned _stv8_20_2 = __float_as_uint(dgv_lo[0 + 2]);
                        unsigned _stv8_20_3 = __float_as_uint(dgv_lo[0 + 3]);
                        unsigned _stv8_20_4 = __float_as_uint(dgv_lo[0 + 4]);
                        unsigned _stv8_20_5 = __float_as_uint(dgv_lo[0 + 5]);
                        unsigned _stv8_20_6 = __float_as_uint(dgv_lo[0 + 6]);
                        unsigned _stv8_20_7 = __float_as_uint(dgv_lo[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dg_out + (rbase + col_1) + (0))), "r"(_stv8_20_0), "r"(_stv8_20_1), "r"(_stv8_20_2), "r"(_stv8_20_3), "r"(_stv8_20_4), "r"(_stv8_20_5), "r"(_stv8_20_6), "r"(_stv8_20_7) : "memory");
                    }
                    {
                        unsigned _stv8_21_0 = __float_as_uint(dgv_hi[0 + 0]);
                        unsigned _stv8_21_1 = __float_as_uint(dgv_hi[0 + 1]);
                        unsigned _stv8_21_2 = __float_as_uint(dgv_hi[0 + 2]);
                        unsigned _stv8_21_3 = __float_as_uint(dgv_hi[0 + 3]);
                        unsigned _stv8_21_4 = __float_as_uint(dgv_hi[0 + 4]);
                        unsigned _stv8_21_5 = __float_as_uint(dgv_hi[0 + 5]);
                        unsigned _stv8_21_6 = __float_as_uint(dgv_hi[0 + 6]);
                        unsigned _stv8_21_7 = __float_as_uint(dgv_hi[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dg_out + (rbase + col_1 + 8) + (0))), "r"(_stv8_21_0), "r"(_stv8_21_1), "r"(_stv8_21_2), "r"(_stv8_21_3), "r"(_stv8_21_4), "r"(_stv8_21_5), "r"(_stv8_21_6), "r"(_stv8_21_7) : "memory");
                    }
                    float dkf_lo[8];
                    float dkf_hi[8];
                    #pragma unroll
                    for (int jj_3 = 0; jj_3 < 8; jj_3++) {
                        dkf_lo[jj_3] = dkf[jj_3];
                        dkf_hi[jj_3] = dkf[jj_3 + 8];
                    }
                    {
                        unsigned _stv8_22_0 = __float_as_uint(dkf_lo[0 + 0]);
                        unsigned _stv8_22_1 = __float_as_uint(dkf_lo[0 + 1]);
                        unsigned _stv8_22_2 = __float_as_uint(dkf_lo[0 + 2]);
                        unsigned _stv8_22_3 = __float_as_uint(dkf_lo[0 + 3]);
                        unsigned _stv8_22_4 = __float_as_uint(dkf_lo[0 + 4]);
                        unsigned _stv8_22_5 = __float_as_uint(dkf_lo[0 + 5]);
                        unsigned _stv8_22_6 = __float_as_uint(dkf_lo[0 + 6]);
                        unsigned _stv8_22_7 = __float_as_uint(dkf_lo[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dk_out + (rbase + col_1) + (0))), "r"(_stv8_22_0), "r"(_stv8_22_1), "r"(_stv8_22_2), "r"(_stv8_22_3), "r"(_stv8_22_4), "r"(_stv8_22_5), "r"(_stv8_22_6), "r"(_stv8_22_7) : "memory");
                    }
                    {
                        unsigned _stv8_23_0 = __float_as_uint(dkf_hi[0 + 0]);
                        unsigned _stv8_23_1 = __float_as_uint(dkf_hi[0 + 1]);
                        unsigned _stv8_23_2 = __float_as_uint(dkf_hi[0 + 2]);
                        unsigned _stv8_23_3 = __float_as_uint(dkf_hi[0 + 3]);
                        unsigned _stv8_23_4 = __float_as_uint(dkf_hi[0 + 4]);
                        unsigned _stv8_23_5 = __float_as_uint(dkf_hi[0 + 5]);
                        unsigned _stv8_23_6 = __float_as_uint(dkf_hi[0 + 6]);
                        unsigned _stv8_23_7 = __float_as_uint(dkf_hi[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dk_out + (rbase + col_1 + 8) + (0))), "r"(_stv8_23_0), "r"(_stv8_23_1), "r"(_stv8_23_2), "r"(_stv8_23_3), "r"(_stv8_23_4), "r"(_stv8_23_5), "r"(_stv8_23_6), "r"(_stv8_23_7) : "memory");
                    }
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, db_acc, 16);
                float db_other = _shfl_xor_0;
                float db_row = db_acc + db_other;
                if (lane_half == 0) {
                    dbp_s[cblk * 64 + t_row] = db_row;
                }
                asm volatile("barrier.sync 8, 512;" ::: "memory");
                if (cblk == 0) {
                    if (lane_half == 0) {
                        float db_sum = dbp_s[t_row];
                        db_sum = db_sum + dbp_s[64 + t_row];
                        db_sum = db_sum + dbp_s[128 + t_row];
                        db_sum = db_sum + dbp_s[192 + t_row];
                        db_out[grow * num_heads + head] = db_sum;
                    }
                }
                if (cblk < 2) {
                    int blk_2 = cblk;
                    int col_2 = blk_2 * 32 + lane_half * 16;
                    float _tmem_load_5[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                        : "r"(taddr + 384 + (unsigned int)row_base + (unsigned int)(blk_2 * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float dab[16];
                    #pragma unroll
                    for (int j_3 = 0; j_3 < 16; j_3++) {
                        int s_col = col_2 + j_3;
                        float val = _tmem_load_5[j_3] * beta[(row0 + s_col) * num_heads + head];
                        if (s_col >= t_row) {
                            val = 0.0f;
                        }
                        dab[j_3] = val;
                    }
                    uint32_t dab_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dab[_lp*2 + 0], dab[_lp*2+1 + 0]));
                        dab_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_2 = 0; q_2 < 2; q_2++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dab_k_addr + (unsigned int)(slot * 57344) + (unsigned int)((col_2 + q_2 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_2 * 8) % 64 * 2 ^ ((col_2 + q_2 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_2 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(dab_bf16[q_2 * 4]), "r"(dab_bf16[q_2 * 4 + 1]), "r"(dab_bf16[q_2 * 4 + 2]), "r"(dab_bf16[q_2 * 4 + 3]) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(b_drained_addr);
                mbarrier_wait(c_ready_addr, _phase_c_ready_0);
                _phase_c_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (cblk < 2) {
                    int blk_3 = cblk;
                    int col_3 = blk_3 * 32 + lane_half * 16;
                    float _tmem_load_6[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                        : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(blk_3 * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    uint32_t _tmem_load_6_bf16[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                        _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int q_3 = 0; q_3 < 2; q_3++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((da1b_w_addr + (unsigned int)(slot * 57344) + (unsigned int)((col_3 + q_3 * 8) / 64 * 8192 + t_row * 128 + (col_3 + q_3 * 8) % 64 * 2 ^ ((col_3 + q_3 * 8) / 64 * 8192 + t_row * 128 + (col_3 + q_3 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[q_3 * 4]), "r"(_tmem_load_6_bf16[q_3 * 4 + 1]), "r"(_tmem_load_6_bf16[q_3 * 4 + 2]), "r"(_tmem_load_6_bf16[q_3 * 4 + 3]) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(c_drained_addr);
                mbarrier_wait(d_ready_addr, _phase_d_ready_0);
                _phase_d_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int abase = (grow * num_heads + head) * 64;
                if (cblk < 2) {
                    int blk_4 = cblk;
                    int col_4 = blk_4 * 32 + lane_half * 16;
                    float _tmem_load_7[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                        : "r"(taddr + 320 + (unsigned int)row_base + (unsigned int)(blk_4 * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float outv[16];
                    #pragma unroll
                    for (int j_4 = 0; j_4 < 16; j_4++) {
                        int s_col2 = col_4 + j_4;
                        float v2 = -_tmem_load_7[j_4];
                        if (s_col2 >= t_row) {
                            v2 = 0.0f;
                        }
                        outv[j_4] = v2;
                    }
                    float outv_lo[8];
                    float outv_hi[8];
                    #pragma unroll
                    for (int jj_4 = 0; jj_4 < 8; jj_4++) {
                        outv_lo[jj_4] = outv[jj_4];
                        outv_hi[jj_4] = outv[jj_4 + 8];
                    }
                    {
                        unsigned _stv8_24_0 = __float_as_uint(outv_lo[0 + 0]);
                        unsigned _stv8_24_1 = __float_as_uint(outv_lo[0 + 1]);
                        unsigned _stv8_24_2 = __float_as_uint(outv_lo[0 + 2]);
                        unsigned _stv8_24_3 = __float_as_uint(outv_lo[0 + 3]);
                        unsigned _stv8_24_4 = __float_as_uint(outv_lo[0 + 4]);
                        unsigned _stv8_24_5 = __float_as_uint(outv_lo[0 + 5]);
                        unsigned _stv8_24_6 = __float_as_uint(outv_lo[0 + 6]);
                        unsigned _stv8_24_7 = __float_as_uint(outv_lo[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dAkk_out + (abase + col_4) + (0))), "r"(_stv8_24_0), "r"(_stv8_24_1), "r"(_stv8_24_2), "r"(_stv8_24_3), "r"(_stv8_24_4), "r"(_stv8_24_5), "r"(_stv8_24_6), "r"(_stv8_24_7) : "memory");
                    }
                    {
                        unsigned _stv8_25_0 = __float_as_uint(outv_hi[0 + 0]);
                        unsigned _stv8_25_1 = __float_as_uint(outv_hi[0 + 1]);
                        unsigned _stv8_25_2 = __float_as_uint(outv_hi[0 + 2]);
                        unsigned _stv8_25_3 = __float_as_uint(outv_hi[0 + 3]);
                        unsigned _stv8_25_4 = __float_as_uint(outv_hi[0 + 4]);
                        unsigned _stv8_25_5 = __float_as_uint(outv_hi[0 + 5]);
                        unsigned _stv8_25_6 = __float_as_uint(outv_hi[0 + 6]);
                        unsigned _stv8_25_7 = __float_as_uint(outv_hi[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(dAkk_out + (abase + col_4 + 8) + (0))), "r"(_stv8_25_0), "r"(_stv8_25_1), "r"(_stv8_25_2), "r"(_stv8_25_3), "r"(_stv8_25_4), "r"(_stv8_25_5), "r"(_stv8_25_6), "r"(_stv8_25_7) : "memory");
                    }
                }
                mbarrier_arrive(d_drained_addr);
                it_i = it_i + 1;
            }
            asm volatile("barrier.sync 8, 512;" ::: "memory");
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 16) {
        { // load_main
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            unsigned int _phase_hdv_empty_0 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int it_1 = bid; it_1 < num_items; it_1 += num_bids) {
                    int chunk_1 = it_1 / (unsigned int)num_heads;
                    int head_1 = it_1 % (unsigned int)num_heads;
                    int row0_1 = chunk_1 * 64;
                    int hrow0_1 = (chunk_1 * num_heads + head_1) * 128;
                    mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 57344);
                    #pragma unroll
                    for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                        int c0 = seg_1 * 64;
                        tma_3d_gmem2smem(smem_v0_addr + stage * 57344 + (unsigned int)(seg_1 * 8192), do_tma, c0, head_1, row0_1, tiles_full_addr + (stage) * 8);
                        tma_3d_gmem2smem(smem_v1_addr + stage * 57344 + (unsigned int)(seg_1 * 8192), vn_tma, c0, head_1, row0_1, tiles_full_addr + (stage) * 8);
                        tma_3d_gmem2smem(smem_v2_addr + stage * 57344 + (unsigned int)(seg_1 * 8192), dv2_tma, c0, head_1, row0_1, tiles_full_addr + (stage) * 8);
                    }
                    tma_3d_gmem2smem(akk_k_addr + stage * 57344, akk_tma, 0, head_1, row0_1, tiles_full_addr + (stage) * 8);
                    mbarrier_wait(hdv_empty_addr, _phase_hdv_empty_0);
                    _phase_hdv_empty_0 ^= 1;
                    mbarrier_arrive_expect_tx(hdv_full_addr, 81920);
                    #pragma unroll
                    for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                        int c1 = seg_2 * 64;
                        tma_3d_gmem2smem(v_p_addr + (unsigned int)(seg_2 * 8192), v_tma, c1, head_1, row0_1, hdv_full_addr);
                        tma_2d_gmem2smem(h_p_addr + (unsigned int)(seg_2 * 16384), h_tma, c1, hrow0_1, hdv_full_addr);
                        tma_2d_gmem2smem(dh_p_addr + (unsigned int)(seg_2 * 16384), dh_tma, c1, hrow0_1, hdv_full_addr);
                    }
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 17) {
        { // mma_main
            unsigned int stage_1 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_hdv_full_0 = 0;
            unsigned int _phase_d_drained_0 = 1;
            unsigned int _phase_a_drained_0 = 0;
            unsigned int _phase_b_drained_0 = 0;
            unsigned int _phase_c_drained_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (unsigned int it_2 = bid; it_2 < num_items; it_2 += num_bids) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full);
                    mbarrier_wait(hdv_full_addr, _phase_hdv_full_0);
                    _phase_hdv_full_0 ^= 1;
                    mbarrier_wait(d_drained_addr, _phase_d_drained_0);
                    _phase_d_drained_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = (((smem_v3_addr) >> 4) & 0x3FFF) + (stage_1) * 3584;
                    int _mma_b_lo_0 = ((h_k_addr) >> 4) & 0x3FFF;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 69207184;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc_dq), "r"(0));
                    int _mma_a_lo_1 = (((smem_v4_addr) >> 4) & 0x3FFF) + (stage_1) * 3584;
                    int _mma_b_lo_1 = ((dh_k_addr) >> 4) & 0x3FFF;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 69207184;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc_dk), "r"(0));
                    int _mma_a_lo_2 = (((smem_v5_addr) >> 4) & 0x3FFF) + (stage_1) * 3584;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 69207184;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_0), "r"(tmem_acc_dw), "r"(0));
                    int _mma_a_lo_3 = (((smem_v5_addr) >> 4) & 0x3FFF) + (stage_1) * 3584;
                    int _mma_b_lo_3 = ((v_k_addr) >> 4) & 0x3FFF;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_acc_da), "r"(0));
                    tcgen05_commit(a_ready_addr);
                    tcgen05_commit(hdv_empty_addr);
                    mbarrier_wait(a_drained_addr, _phase_a_drained_0);
                    _phase_a_drained_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_4 = ((((akk_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 3584;
                    int _mma_b_lo_4 = ((((dv2_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 3584;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 69305488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_acc_dvb), "r"(0));
                    int _mma_a_lo_5 = ((((akk_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 3584;
                    int _mma_b_lo_5 = (((dwb_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 69305488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"(tmem_acc_dkgb), "r"(0));
                    int _mma_a_lo_6 = ((dwb_k_addr) >> 4) & 0x3FFF;
                    int _mma_b_lo_6 = ((kgb_k_addr) >> 4) & 0x3FFF;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_acc_da), "r"(1));
                    tcgen05_commit(b_ready_addr);
                    mbarrier_wait(b_drained_addr, _phase_b_drained_0);
                    _phase_b_drained_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_7 = (((dab_k_addr) >> 4) & 0x3FFF) + (stage_1) * 3584;
                    int _mma_b_lo_7 = (((akk_k_addr) >> 4) & 0x3FFF) + (stage_1) * 3584;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_acc_da1), "r"(0));
                    tcgen05_commit(c_ready_addr);
                    mbarrier_wait(c_drained_addr, _phase_c_drained_0);
                    _phase_c_drained_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_8 = ((((akk_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 3584;
                    int _mma_b_lo_8 = ((((da1b_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 3584;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68256912;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"(tmem_acc_da2), "r"(0));
                    tcgen05_commit(d_ready_addr);
                    tcgen05_commit(tiles_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 2) { stage_1 = 0; _phase_tiles_full ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
