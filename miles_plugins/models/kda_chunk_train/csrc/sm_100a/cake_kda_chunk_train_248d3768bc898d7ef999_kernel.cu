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
#define TMEM_NCOLS 512
#define TMEM_ACC_DQ_OFFSET 0
#define TMEM_ACC_DK_OFFSET 128
#define TMEM_ACC_DW_OFFSET 256
#define TMEM_ACC_DA_OFFSET 384
#define TMEM_ACC_DVB_OFFSET 128
#define TMEM_ACC_DKGB_OFFSET 256
#define TMEM_ACC_DA1_OFFSET 448
#define TMEM_ACC_DA2_OFFSET 448
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
#define SMEM_GK_R_OFF 1024
#define SMEM_GK_R_STAGE_BYTES 32768
#define SMEM_GK_R_STRIDE 57344
#define SMEM_SMEM_V23_OFF 33792
#define SMEM_SMEM_V23_STAGE_BYTES 16384
#define SMEM_SMEM_V23_STRIDE 57344
#define SMEM_Q_P_OFF 181248
#define SMEM_Q_P_STAGE_BYTES 8192
#define SMEM_Q_P_STRIDE 8192
#define SMEM_Q_S_OFF 181248
#define SMEM_Q_S_STAGE_BYTES 16384
#define SMEM_Q_S_STRIDE 16384
#define SMEM_DG_STG_OFF 197632
#define SMEM_DG_STG_STAGE_BYTES 32768
#define SMEM_DG_STG_STRIDE 32768
#define SMEM_DK_STG_OFF 1024
#define SMEM_DK_STG_STAGE_BYTES 32768
#define SMEM_DK_STG_STRIDE 57344
#define SMEM_BETA_S_OFF 231936
#define SMEM_BETA_S_STAGE_BYTES 256
#define SMEM_BETA_S_STRIDE 256
#define SMEM_DGK_S_OFF 230400
#define SMEM_DGK_S_STAGE_BYTES 512
#define SMEM_DGK_S_STRIDE 512
#define SMEM_DBP_S_OFF 230912
#define SMEM_DBP_S_STAGE_BYTES 1024
#define SMEM_DBP_S_STRIDE 1024
#define SMEM_TOTAL 232192
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 1

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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



__device__ __forceinline__ void mbarrier_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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



__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_chunk_train_248d3768bc898d7ef999(const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap vn_tma, const __grid_constant__ CUtensorMap dv2_tma, const __grid_constant__ CUtensorMap v_tma, const __grid_constant__ CUtensorMap h_tma, const __grid_constant__ CUtensorMap dh_tma, const __grid_constant__ CUtensorMap akk_tma, const __grid_constant__ CUtensorMap gk_tma, const __grid_constant__ CUtensorMap k_tma, const __grid_constant__ CUtensorMap q_tma, float* __restrict__ beta, float* __restrict__ dq_out, const __grid_constant__ CUtensorMap dk_tma, const __grid_constant__ CUtensorMap dg_tma, float* __restrict__ db_out, float* __restrict__ dAkk_out, __nv_bfloat16* __restrict__ dv_out, int num_heads, int num_items, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float scale)
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

    const int mbar_base = smem;
    #define tiles_full_addr (mbar_base + 0)
    #define tiles_empty_addr (mbar_base + 16)
    #define hdv_full_addr (mbar_base + 32)
    #define hd_empty_addr (mbar_base + 40)
    #define hd_read_done_addr (mbar_base + 48)
    #define ops_empty_addr (mbar_base + 56)
    #define ops_full_addr (mbar_base + 64)
    #define a_ready_addr (mbar_base + 72)
    #define a_drained_addr (mbar_base + 80)
    #define b_ready_addr (mbar_base + 88)
    #define b_drained_addr (mbar_base + 96)
    #define c_ready_addr (mbar_base + 104)
    #define c_drained_addr (mbar_base + 112)
    #define d_ready_addr (mbar_base + 120)
    #define d_drained_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_v0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V0_OFF);
    const int smem_v0_addr = smem + SMEM_SMEM_V0_OFF;
    __nv_bfloat16* smem_v1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V1_OFF);
    const int smem_v1_addr = smem + SMEM_SMEM_V1_OFF;
    __nv_bfloat16* smem_v2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V2_OFF);
    const int smem_v2_addr = smem + SMEM_SMEM_V2_OFF;
    __nv_bfloat16* smem_v3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V3_OFF);
    const int smem_v3_addr = smem + SMEM_SMEM_V3_OFF;
    __nv_bfloat16* smem_v4 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V4_OFF);
    const int smem_v4_addr = smem + SMEM_SMEM_V4_OFF;
    __nv_bfloat16* smem_v5 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V5_OFF);
    const int smem_v5_addr = smem + SMEM_SMEM_V5_OFF;
    __nv_bfloat16* dv2_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DV2_MN_OFF);
    const int dv2_mn_addr = smem + SMEM_DV2_MN_OFF;
    __nv_bfloat16* akk_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_AKK_K_OFF);
    const int akk_k_addr = smem + SMEM_AKK_K_OFF;
    __nv_bfloat16* akk_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_AKK_MN_OFF);
    const int akk_mn_addr = smem + SMEM_AKK_MN_OFF;
    __nv_bfloat16* dab_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DAB_K_OFF);
    const int dab_k_addr = smem + SMEM_DAB_K_OFF;
    __nv_bfloat16* da1b_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DA1B_MN_OFF);
    const int da1b_mn_addr = smem + SMEM_DA1B_MN_OFF;
    __nv_bfloat16* da1b_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DA1B_W_OFF);
    const int da1b_w_addr = smem + SMEM_DA1B_W_OFF;
    float* ring_f32 = reinterpret_cast<float*>(smem_raw + SMEM_RING_F32_OFF);
    const int ring_f32_addr = smem + SMEM_RING_F32_OFF;
    __nv_bfloat16* v_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_V_P_OFF);
    const int v_p_addr = smem + SMEM_V_P_OFF;
    __nv_bfloat16* v_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_V_K_OFF);
    const int v_k_addr = smem + SMEM_V_K_OFF;
    __nv_bfloat16* h_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_H_P_OFF);
    const int h_p_addr = smem + SMEM_H_P_OFF;
    __nv_bfloat16* dh_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DH_P_OFF);
    const int dh_p_addr = smem + SMEM_DH_P_OFF;
    __nv_bfloat16* h_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_H_K_OFF);
    const int h_k_addr = smem + SMEM_H_K_OFF;
    __nv_bfloat16* dh_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DH_K_OFF);
    const int dh_k_addr = smem + SMEM_DH_K_OFF;
    __nv_bfloat16* dwb_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DWB_K_OFF);
    const int dwb_k_addr = smem + SMEM_DWB_K_OFF;
    __nv_bfloat16* dwb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DWB_MN_OFF);
    const int dwb_mn_addr = smem + SMEM_DWB_MN_OFF;
    __nv_bfloat16* kgb_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_KGB_K_OFF);
    const int kgb_k_addr = smem + SMEM_KGB_K_OFF;
    float* gk_r = reinterpret_cast<float*>(smem_raw + SMEM_GK_R_OFF);
    const int gk_r_addr = smem + SMEM_GK_R_OFF;
    __nv_bfloat16* smem_v23 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V23_OFF);
    const int smem_v23_addr = smem + SMEM_SMEM_V23_OFF;
    __nv_bfloat16* q_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_Q_P_OFF);
    const int q_p_addr = smem + SMEM_Q_P_OFF;
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_Q_S_OFF);
    const int q_s_addr = smem + SMEM_Q_S_OFF;
    float* dg_stg = reinterpret_cast<float*>(smem_raw + SMEM_DG_STG_OFF);
    const int dg_stg_addr = smem + SMEM_DG_STG_OFF;
    float* dk_stg = reinterpret_cast<float*>(smem_raw + SMEM_DK_STG_OFF);
    const int dk_stg_addr = smem + SMEM_DK_STG_OFF;
    float* beta_s = reinterpret_cast<float*>(smem_raw + SMEM_BETA_S_OFF);
    const int beta_s_addr = smem + SMEM_BETA_S_OFF;
    float* dgk_s = reinterpret_cast<float*>(smem_raw + SMEM_DGK_S_OFF);
    const int dgk_s_addr = smem + SMEM_DGK_S_OFF;
    float* dbp_s = reinterpret_cast<float*>(smem_raw + SMEM_DBP_S_OFF);
    const int dbp_s_addr = smem + SMEM_DBP_S_OFF;

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 17 barriers)
    // Mbarriers at smem_raw[0..136)

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
            // hd_empty: 1 barriers, init_count=512
            mbarrier_init(smem + 40, 512);
            // hd_read_done: 1 barriers, init_count=512
            mbarrier_init(smem + 48, 512);
            // ops_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // ops_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // a_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // a_drained: 1 barriers, init_count=512
            mbarrier_init(smem + 80, 512);
            // b_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            // b_drained: 1 barriers, init_count=512
            mbarrier_init(smem + 96, 512);
            // c_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // c_drained: 1 barriers, init_count=512
            mbarrier_init(smem + 112, 512);
            // d_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // d_drained: 1 barriers, init_count=512
            mbarrier_init(smem + 128, 512);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 136);
    if (warp == 0) {
        int _tmem_hold = smem + 136;
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
    const int tmem_acc_dvb = taddr + 128;
    const int tmem_acc_dkgb = taddr + 256;
    const int tmem_acc_da1 = taddr + 448;
    const int tmem_acc_da2 = taddr + 448;

    // === Task calls (dependency order) ===
    int wq = warp % 4;
    int cblk = warp / 4;
    int lane_half = lane >> 4;
    int t_row = wq * 16 + (lane & 15);
    int row_base = wq * 32 << 16;
    int tid_1 = warp * 32 + lane;
    int it_i = 0;
    unsigned int stage = 0;
    unsigned int pstage = 0;
    unsigned int _phase_tiles_empty = 1;
    unsigned int _phase_hd_empty_0 = 1;
    if (warp == 15) {
        if (elect_sync()) {
            int first = bid;
            if (first < num_items) {
                int chunk = first / num_heads;
                int head = first % num_heads;
                int row0 = chunk * 64;
                int hrow0 = (chunk * num_heads + head) * 128;
                mbarrier_wait(tiles_empty_addr + (pstage) * 8, _phase_tiles_empty);
                mbarrier_arrive_expect_tx(tiles_full_addr + (pstage) * 8, 57344);
                #pragma unroll
                for (int seg = 0; seg < 2; seg++) {
                    int c0 = seg * 64;
                    tma_3d_gmem2smem(smem_v0_addr + pstage * 57344 + (unsigned int)(seg * 8192), (&do_tma), c0, head, row0, tiles_full_addr + (pstage) * 8);
                    tma_3d_gmem2smem(smem_v1_addr + pstage * 57344 + (unsigned int)(seg * 8192), (&vn_tma), c0, head, row0, tiles_full_addr + (pstage) * 8);
                    tma_3d_gmem2smem(smem_v2_addr + pstage * 57344 + (unsigned int)(seg * 8192), (&dv2_tma), c0, head, row0, tiles_full_addr + (pstage) * 8);
                }
                tma_3d_gmem2smem(akk_k_addr + pstage * 57344, (&akk_tma), 0, head, row0, tiles_full_addr + (pstage) * 8);
                pstage += 1;
                if (pstage == 2) { pstage = 0; _phase_tiles_empty ^= 1; }
                mbarrier_wait(hd_empty_addr, _phase_hd_empty_0);
                _phase_hd_empty_0 ^= 1;
                mbarrier_arrive_expect_tx(hdv_full_addr, 81920);
                #pragma unroll
                for (int seg_1 = 0; seg_1 < 2; seg_1++) {
                    int c1a = seg_1 * 64;
                    tma_2d_gmem2smem(h_p_addr + (unsigned int)(seg_1 * 16384), (&h_tma), c1a, hrow0, hdv_full_addr);
                }
                tma_2d_gmem2smem(dh_p_addr, (&dh_tma), 0, hrow0, hdv_full_addr);
                #pragma unroll
                for (int seg_2 = 0; seg_2 < 2; seg_2++) {
                    int c1b = seg_2 * 64;
                    tma_3d_gmem2smem(v_p_addr + (unsigned int)(seg_2 * 8192), (&v_tma), c1b, head, row0, hdv_full_addr);
                }
                tma_2d_gmem2smem(dh_p_addr + 16384, (&dh_tma), 64, hrow0, hdv_full_addr);
                int oslot_f = 1;
                mbarrier_arrive_expect_tx(ops_full_addr, 49152);
                #pragma unroll
                for (int sl = 0; sl < 4; sl++) {
                    int c3 = sl * 32;
                    tma_3d_gmem2smem(gk_r_addr + (unsigned int)(oslot_f * 57344) + (unsigned int)(sl * 8192), (&gk_tma), c3, head, row0, ops_full_addr);
                }
                #pragma unroll
                for (int seg_3 = 0; seg_3 < 2; seg_3++) {
                    int c4 = seg_3 * 64;
                    tma_3d_gmem2smem(smem_v23_addr + (unsigned int)(oslot_f * 57344) + (unsigned int)(seg_3 * 8192), (&k_tma), c4, head, row0, ops_full_addr);
                }
            }
        }
    }
    unsigned int _phase_tiles_full = 0;
    unsigned int _phase_hdv_full_0 = 0;
    unsigned int _phase_d_drained_0 = 1;
    unsigned int _phase_ops_empty_0 = 0;
    unsigned int _phase_hd_read_done_0 = 0;
    unsigned int _phase_a_ready_0 = 0;
    unsigned int _phase_ops_full_0 = 0;
    unsigned int _phase_a_drained_0 = 0;
    unsigned int _phase_b_ready_0 = 0;
    unsigned int _phase_b_drained_0 = 0;
    unsigned int _phase_c_ready_0 = 0;
    unsigned int _phase_c_drained_0 = 0;
    unsigned int _phase_d_ready_0 = 0;
    #pragma unroll 1
    for (unsigned int it = bid; it < num_items; it += num_bids) {
        int chunk_1 = it / (unsigned int)num_heads;
        int head_1 = it % (unsigned int)num_heads;
        int row0_1 = chunk_1 * 64;
        int hrow0_1 = (chunk_1 * num_heads + head_1) * 128;
        int nxt = it + (unsigned int)num_bids;
        int slot = it_i & 1;
        int oslot = slot ^ 1;
        int kbase = slot * 14336;
        int grow = row0_1 + t_row;
        int rbase = (grow * num_heads + head_1) * 128;
        int gn_base = ((row0_1 + 64 - 1) * num_heads + head_1) * 128;
        float beta_t = beta[grow * num_heads + head_1];
        int ext_row = chunk_bos[chunk_1] + t_row;
        int clen = chunk_len[chunk_1];
        if (tid_1 < 64) {
            beta_s[tid_1] = beta[(row0_1 + tid_1) * num_heads + head_1];
        }
        asm volatile("barrier.sync 8, 512;" ::: "memory");
        if (warp == 15) {
            if (elect_sync()) {
                mbarrier_wait(tiles_full_addr + (stage) * 8, _phase_tiles_full);
                mbarrier_wait(hdv_full_addr, _phase_hdv_full_0);
                mbarrier_wait(d_drained_addr, _phase_d_drained_0);
                _phase_d_drained_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = (((smem_v3_addr) >> 4) & 0x3FFF) + (stage) * 3584;
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
                int _mma_a_lo_1 = (((smem_v4_addr) >> 4) & 0x3FFF) + (stage) * 3584;
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
                int _mma_a_lo_2 = (((smem_v5_addr) >> 4) & 0x3FFF) + (stage) * 3584;
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
                int _mma_a_lo_3 = (((smem_v5_addr) >> 4) & 0x3FFF) + (stage) * 3584;
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
                tcgen05_commit(ops_empty_addr);
            }
        }
        float dgk_inter = 0.0f;
        mbarrier_wait(hdv_full_addr, _phase_hdv_full_0);
        _phase_hdv_full_0 ^= 1;
        if (tid_1 < 128) {
            #pragma unroll
            for (int seg_4 = 0; seg_4 < 8; seg_4++) {
                unsigned int hraw[8];
                unsigned int dhraw[8];
                #pragma unroll
                for (int q = 0; q < 2; q++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&hraw[q * 4])), "=r"(*reinterpret_cast<uint32_t*>(&hraw[(q * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&hraw[(q * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&hraw[(q * 4) + 3]))
                        : "r"((h_k_addr + (unsigned int)((seg_4 * 16 + q * 8) / 64 * 16384 + tid_1 * 128 + (seg_4 * 16 + q * 8) % 64 * 2 ^ ((seg_4 * 16 + q * 8) / 64 * 16384 + tid_1 * 128 + (seg_4 * 16 + q * 8) % 64 * 2 >> 7 & 7) << 4))));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&dhraw[q * 4])), "=r"(*reinterpret_cast<uint32_t*>(&dhraw[(q * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dhraw[(q * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dhraw[(q * 4) + 3]))
                        : "r"((dh_k_addr + (unsigned int)((seg_4 * 16 + q * 8) / 64 * 16384 + tid_1 * 128 + (seg_4 * 16 + q * 8) % 64 * 2 ^ ((seg_4 * 16 + q * 8) / 64 * 16384 + tid_1 * 128 + (seg_4 * 16 + q * 8) % 64 * 2 >> 7 & 7) << 4))));
                }
                float hraw_f32[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&hraw_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hraw[_pair]) << 16);
                    (&hraw_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hraw[_pair]) & 0xffff0000u);
                }
                float dhraw_f32[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&dhraw_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(dhraw[_pair]) << 16);
                    (&dhraw_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(dhraw[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(hraw_f32[j] * dhraw_f32[j]);
                    float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                    dgk_inter = dgk_inter + _cvt_f32_0;
                }
            }
        }
        mbarrier_arrive(hd_read_done_addr);
        if (warp == 15) {
            if (elect_sync()) {
                mbarrier_wait(ops_empty_addr, _phase_ops_empty_0);
                _phase_ops_empty_0 ^= 1;
                mbarrier_wait(hd_read_done_addr, _phase_hd_read_done_0);
                _phase_hd_read_done_0 ^= 1;
                mbarrier_expect_tx(b_ready_addr, 16384);
                #pragma unroll
                for (int seg_5 = 0; seg_5 < 2; seg_5++) {
                    int c7 = seg_5 * 64;
                    tma_3d_gmem2smem(q_p_addr + (unsigned int)(seg_5 * 8192), (&q_tma), c7, head_1, row0_1, b_ready_addr);
                }
                if (nxt < num_items) {
                    int chunk_0 = nxt / num_heads;
                    int head_1_1 = nxt % num_heads;
                    int row0_2 = chunk_0 * 64;
                    int hrow0_3 = (chunk_0 * num_heads + head_1_1) * 128;
                    mbarrier_arrive_expect_tx(hdv_full_addr, 81920);
                    #pragma unroll
                    for (int seg_6 = 0; seg_6 < 2; seg_6++) {
                        int c1a_1 = seg_6 * 64;
                        tma_2d_gmem2smem(h_p_addr + (unsigned int)(seg_6 * 16384), (&h_tma), c1a_1, hrow0_3, hdv_full_addr);
                    }
                    tma_2d_gmem2smem(dh_p_addr, (&dh_tma), 0, hrow0_3, hdv_full_addr);
                }
            }
        }
        mbarrier_wait(a_ready_addr, _phase_a_ready_0);
        _phase_a_ready_0 ^= 1;
        mbarrier_wait(ops_full_addr, _phase_ops_full_0);
        _phase_ops_full_0 ^= 1;
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
            unsigned int kraw[8];
            #pragma unroll
            for (int q_1 = 0; q_1 < 4; q_1++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gkv[q_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&gkv[(q_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gkv[(q_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gkv[(q_1 * 4) + 3]))
                    : "r"((gk_r_addr + (unsigned int)(oslot * 57344) + (unsigned int)((col + q_1 * 4) / 32 * 8192 + t_row * 128 + (col + q_1 * 4) % 32 * 4 ^ ((col + q_1 * 4) / 32 * 8192 + t_row * 128 + (col + q_1 * 4) % 32 * 4 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gnv[q_1 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&gnv[(q_1 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gnv[(q_1 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gnv[(q_1 * 4) + 3]))
                    : "r"((gk_r_addr + (unsigned int)(oslot * 57344) + (unsigned int)((col + q_1 * 4) / 32 * 8192 + 8064 + (col + q_1 * 4) % 32 * 4 ^ ((col + q_1 * 4) / 32 * 8192 + 8064 + (col + q_1 * 4) % 32 * 4 >> 7 & 7) << 4))));
            }
            #pragma unroll
            for (int q_2 = 0; q_2 < 2; q_2++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw[q_2 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&kraw[(q_2 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw[(q_2 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw[(q_2 * 4) + 3]))
                    : "r"((smem_v23_addr + (unsigned int)(oslot * 57344) + (unsigned int)((col + q_2 * 8) / 64 * 8192 + t_row * 128 + (col + q_2 * 8) % 64 * 2 ^ ((col + q_2 * 8) / 64 * 8192 + t_row * 128 + (col + q_2 * 8) % 64 * 2 >> 7 & 7) << 4))));
            }
            float kraw_f32[16];
            #pragma unroll
            for (int _pair = 0; _pair < 8; _pair++) {
                (&kraw_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kraw[_pair]) << 16);
                (&kraw_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kraw[_pair]) & 0xffff0000u);
            }
            float dqv[16];
            float dkv[16];
            float dwbv[16];
            float kgv[16];
            #pragma unroll
            for (int j_1 = 0; j_1 < 16; j_1++) {
                float _exp2_0 = approx_exp2(gkv[j_1]);
                float e_g = _exp2_0;
                dqv[j_1] = _tmem_load_0[j_1] * e_g * scale;
                float _exp2_1 = approx_exp2(gnv[j_1] - gkv[j_1]);
                dkv[j_1] = _tmem_load_1[j_1] * _exp2_1;
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_tmem_load_2[j_1]);
                float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                dwbv[j_1] = -_cvt_f32_1;
                kgv[j_1] = kraw_f32[j_1] * e_g;
            }
            float dqv_lo[8];
            float dqv_hi[8];
            #pragma unroll
            for (int jj = 0; jj < 8; jj++) {
                dqv_lo[jj] = dqv[jj];
                dqv_hi[jj] = dqv[jj + 8];
            }
            {
                unsigned _stv8_0_0 = __float_as_uint(dqv_lo[0 + 0]);
                unsigned _stv8_0_1 = __float_as_uint(dqv_lo[0 + 1]);
                unsigned _stv8_0_2 = __float_as_uint(dqv_lo[0 + 2]);
                unsigned _stv8_0_3 = __float_as_uint(dqv_lo[0 + 3]);
                unsigned _stv8_0_4 = __float_as_uint(dqv_lo[0 + 4]);
                unsigned _stv8_0_5 = __float_as_uint(dqv_lo[0 + 5]);
                unsigned _stv8_0_6 = __float_as_uint(dqv_lo[0 + 6]);
                unsigned _stv8_0_7 = __float_as_uint(dqv_lo[0 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(dq_out + (rbase + col) + (0))), "r"(_stv8_0_0), "r"(_stv8_0_1), "r"(_stv8_0_2), "r"(_stv8_0_3), "r"(_stv8_0_4), "r"(_stv8_0_5), "r"(_stv8_0_6), "r"(_stv8_0_7) : "memory");
            }
            {
                unsigned _stv8_1_0 = __float_as_uint(dqv_hi[0 + 0]);
                unsigned _stv8_1_1 = __float_as_uint(dqv_hi[0 + 1]);
                unsigned _stv8_1_2 = __float_as_uint(dqv_hi[0 + 2]);
                unsigned _stv8_1_3 = __float_as_uint(dqv_hi[0 + 3]);
                unsigned _stv8_1_4 = __float_as_uint(dqv_hi[0 + 4]);
                unsigned _stv8_1_5 = __float_as_uint(dqv_hi[0 + 5]);
                unsigned _stv8_1_6 = __float_as_uint(dqv_hi[0 + 6]);
                unsigned _stv8_1_7 = __float_as_uint(dqv_hi[0 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(dq_out + (rbase + col + 8) + (0))), "r"(_stv8_1_0), "r"(_stv8_1_1), "r"(_stv8_1_2), "r"(_stv8_1_3), "r"(_stv8_1_4), "r"(_stv8_1_5), "r"(_stv8_1_6), "r"(_stv8_1_7) : "memory");
            }
            int kchunk = col >> 2;
            #pragma unroll
            for (int q_3 = 0; q_3 < 4; q_3++) {
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(ring_f32_addr + (unsigned int)((kbase + t_row * 128) * 4) + (unsigned int)((kchunk + q_3 ^ t_row & 7) << 4)), "f"(dkv[q_3 * 4]), "f"(dkv[q_3 * 4 + 1]), "f"(dkv[q_3 * 4 + 2]), "f"(dkv[q_3 * 4 + 3]) : "memory");
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
            for (int q_4 = 0; q_4 < 2; q_4++) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dwb_k_addr + (unsigned int)((col + q_4 * 8) / 64 * 8192 + t_row * 128 + (col + q_4 * 8) % 64 * 2 ^ ((col + q_4 * 8) / 64 * 8192 + t_row * 128 + (col + q_4 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(dwbv_bf16[q_4 * 4]), "r"(dwbv_bf16[q_4 * 4 + 1]), "r"(dwbv_bf16[q_4 * 4 + 2]), "r"(dwbv_bf16[q_4 * 4 + 3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((kgb_k_addr + (unsigned int)((col + q_4 * 8) / 64 * 8192 + t_row * 128 + (col + q_4 * 8) % 64 * 2 ^ ((col + q_4 * 8) / 64 * 8192 + t_row * 128 + (col + q_4 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(kgv_bf16[q_4 * 4]), "r"(kgv_bf16[q_4 * 4 + 1]), "r"(kgv_bf16[q_4 * 4 + 2]), "r"(kgv_bf16[q_4 * 4 + 3]) : "memory");
            }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(a_drained_addr);
        asm volatile("barrier.sync 8, 512;" ::: "memory");
        if (warp == 15) {
            if (elect_sync()) {
                mbarrier_wait(a_drained_addr, _phase_a_drained_0);
                _phase_a_drained_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_4 = ((((akk_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage) * 3584;
                int _mma_b_lo_4 = ((((dv2_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage) * 3584;
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
                int _mma_a_lo_5 = ((((akk_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage) * 3584;
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
            }
        }
        if (tid_1 < 128) {
            float colsum = 0.0f;
            unsigned int kshift = (tid_1 & 1 ^ 1) * 16;
            #pragma unroll 4
            for (int t = 0; t < 64; t += 2) {
                unsigned int kw0[1];
                unsigned int kw1[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw0[0])) : "r"((smem_v23_addr + (unsigned int)(oslot * 57344) + (unsigned int)((tid_1 & -2) / 64 * 8192 + t * 128 + (tid_1 & -2) % 64 * 2 ^ ((tid_1 & -2) / 64 * 8192 + t * 128 + (tid_1 & -2) % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw1[0])) : "r"((smem_v23_addr + (unsigned int)(oslot * 57344) + (unsigned int)((tid_1 & -2) / 64 * 8192 + (t + 1) * 128 + (tid_1 & -2) % 64 * 2 ^ ((tid_1 & -2) / 64 * 8192 + (t + 1) * 128 + (tid_1 & -2) % 64 * 2 >> 7 & 7) << 4))));
                unsigned int kb0 = kw0[0] << kshift & 4294901760u;
                unsigned int kb1 = kw1[0] << kshift & 4294901760u;
                float dk0 = ring_f32[kbase + t * 128 + (((tid_1 >> 2 ^ t & 7) << 2) + (tid_1 & 3))];
                float dk1 = ring_f32[kbase + (t + 1) * 128 + (((tid_1 >> 2 ^ t + 1 & 7) << 2) + (tid_1 & 3))];
                float2 _f2_0 = make_float2(__uint_as_float(kb0), __uint_as_float(kb1));
                float2 _f2_1 = make_float2(dk0, dk1);
                float2 _mul_f32x2_0;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_1));
                colsum = colsum + _mul_f32x2_0.x;
                colsum = colsum + _mul_f32x2_0.y;
            }
            float gn1[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&gn1[0])) : "r"((gk_r_addr + (unsigned int)(oslot * 57344) + (unsigned int)(tid_1 / 32 * 8192 + 8064 + tid_1 % 32 * 4 ^ (tid_1 / 32 * 8192 + 8064 + tid_1 % 32 * 4 >> 7 & 7) << 4))));
            float _exp2_2 = approx_exp2(gn1[0]);
            dgk_inter = dgk_inter * _exp2_2;
            dgk_s[tid_1] = dgk_inter + colsum;
        }
        asm volatile("barrier.sync 8, 512;" ::: "memory");
        mbarrier_wait(b_ready_addr, _phase_b_ready_0);
        _phase_b_ready_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        float db_acc = 0.0f;
        int colb = cblk * 32 + lane_half * 16;
        float dkf[16];
        #pragma unroll
        for (int _one_b = 0; _one_b < 1; _one_b++) {
            int blk_1 = cblk;
            int col_1 = blk_1 * 32 + lane_half * 16;
            float _tmem_load_3[16];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                : "r"(taddr + 128 + (unsigned int)row_base + (unsigned int)(blk_1 * 32)));
            float _tmem_load_4[16];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                : "r"(taddr + 256 + (unsigned int)row_base + (unsigned int)(blk_1 * 32)));
            float _tmem_load_5[16];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                : "r"(taddr + (unsigned int)row_base + (unsigned int)(blk_1 * 32)));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            float gkv2[16];
            unsigned int kraw2[8];
            float dk2[16];
            #pragma unroll
            for (int q_5 = 0; q_5 < 4; q_5++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gkv2[q_5 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&gkv2[(q_5 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gkv2[(q_5 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gkv2[(q_5 * 4) + 3]))
                    : "r"((gk_r_addr + (unsigned int)(oslot * 57344) + (unsigned int)((col_1 + q_5 * 4) / 32 * 8192 + t_row * 128 + (col_1 + q_5 * 4) % 32 * 4 ^ ((col_1 + q_5 * 4) / 32 * 8192 + t_row * 128 + (col_1 + q_5 * 4) % 32 * 4 >> 7 & 7) << 4))));
            }
            #pragma unroll
            for (int q_6 = 0; q_6 < 2; q_6++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&kraw2[q_6 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&kraw2[(q_6 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&kraw2[(q_6 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&kraw2[(q_6 * 4) + 3]))
                    : "r"((smem_v23_addr + (unsigned int)(oslot * 57344) + (unsigned int)((col_1 + q_6 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_6 * 8) % 64 * 2 ^ ((col_1 + q_6 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_6 * 8) % 64 * 2 >> 7 & 7) << 4))));
            }
            float kraw2_f32[16];
            #pragma unroll
            for (int _pair = 0; _pair < 8; _pair++) {
                (&kraw2_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kraw2[_pair]) << 16);
                (&kraw2_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kraw2[_pair]) & 0xffff0000u);
            }
            unsigned int qraw2[8];
            unsigned int vraw2[8];
            #pragma unroll
            for (int q_7 = 0; q_7 < 2; q_7++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&qraw2[q_7 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&qraw2[(q_7 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&qraw2[(q_7 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&qraw2[(q_7 * 4) + 3]))
                    : "r"((q_s_addr + (unsigned int)((col_1 + q_7 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_7 * 8) % 64 * 2 ^ ((col_1 + q_7 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_7 * 8) % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&vraw2[q_7 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&vraw2[(q_7 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vraw2[(q_7 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vraw2[(q_7 * 4) + 3]))
                    : "r"((v_k_addr + (unsigned int)((col_1 + q_7 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_7 * 8) % 64 * 2 ^ ((col_1 + q_7 * 8) / 64 * 8192 + t_row * 128 + (col_1 + q_7 * 8) % 64 * 2 >> 7 & 7) << 4))));
            }
            float qraw2_f32[16];
            #pragma unroll
            for (int _pair = 0; _pair < 8; _pair++) {
                (&qraw2_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qraw2[_pair]) << 16);
                (&qraw2_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qraw2[_pair]) & 0xffff0000u);
            }
            float vraw2_f32[16];
            #pragma unroll
            for (int _pair = 0; _pair < 8; _pair++) {
                (&vraw2_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(vraw2[_pair]) << 16);
                (&vraw2_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(vraw2[_pair]) & 0xffff0000u);
            }
            int kchunk2 = col_1 >> 2;
            #pragma unroll
            for (int q_8 = 0; q_8 < 4; q_8++) {
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dk2[q_8 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&dk2[(q_8 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dk2[(q_8 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dk2[(q_8 * 4) + 3]))
                    : "r"(ring_f32_addr + (unsigned int)((kbase + t_row * 128) * 4) + (unsigned int)((kchunk2 + q_8 ^ t_row & 7) << 4)));
            }
            float dvo[16];
            float dgv[16];
            #pragma unroll
            for (int j_2 = 0; j_2 < 16; j_2++) {
                float _exp2_3 = approx_exp2(gkv2[j_2]);
                float e_g2 = _exp2_3;
                float kgf = kraw2_f32[j_2] * e_g2;
                float dq2j = _tmem_load_5[j_2] * e_g2 * scale;
                dvo[j_2] = _tmem_load_3[j_2] * beta_t;
                db_acc = db_acc + _tmem_load_3[j_2] * vraw2_f32[j_2];
                db_acc = db_acc + _tmem_load_4[j_2] * kgf;
                float dgj = qraw2_f32[j_2] * dq2j - kraw2_f32[j_2] * dk2[j_2] + kgf * _tmem_load_4[j_2] * beta_t;
                if (t_row == 63) {
                    dgj = dgj + dgk_s[col_1 + j_2];
                }
                dgv[j_2] = dgj;
                dkf[j_2] = dk2[j_2] + _tmem_load_4[j_2] * e_g2 * beta_t;
            }
            #pragma unroll
            for (int q_9 = 0; q_9 < 4; q_9++) {
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((dg_stg_addr + (unsigned int)((col_1 + q_9 * 4) / 32 * 8192 + t_row * 128 + (col_1 + q_9 * 4) % 32 * 4 ^ ((col_1 + q_9 * 4) / 32 * 8192 + t_row * 128 + (col_1 + q_9 * 4) % 32 * 4 >> 7 & 7) << 4))), "f"(dgv[q_9 * 4]), "f"(dgv[q_9 * 4 + 1]), "f"(dgv[q_9 * 4 + 2]), "f"(dgv[q_9 * 4 + 3]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(hd_empty_addr);
            if (warp == 15) {
                if (elect_sync()) {
                    mbarrier_wait(hd_empty_addr, _phase_hd_empty_0);
                    _phase_hd_empty_0 ^= 1;
                    if (nxt < num_items) {
                        int chunk_0_1 = nxt / num_heads;
                        int head_1_2 = nxt % num_heads;
                        int row0_2_1 = chunk_0_1 * 64;
                        int hrow0_3_1 = (chunk_0_1 * num_heads + head_1_2) * 128;
                        #pragma unroll
                        for (int seg_7 = 0; seg_7 < 2; seg_7++) {
                            int c1b_1 = seg_7 * 64;
                            tma_3d_gmem2smem(v_p_addr + (unsigned int)(seg_7 * 8192), (&v_tma), c1b_1, head_1_2, row0_2_1, hdv_full_addr);
                        }
                        tma_2d_gmem2smem(dh_p_addr + 16384, (&dh_tma), 64, hrow0_3_1, hdv_full_addr);
                        mbarrier_wait(tiles_empty_addr + (pstage) * 8, _phase_tiles_empty);
                        mbarrier_arrive_expect_tx(tiles_full_addr + (pstage) * 8, 57344);
                        #pragma unroll
                        for (int seg_8 = 0; seg_8 < 2; seg_8++) {
                            int c0_1 = seg_8 * 64;
                            tma_3d_gmem2smem(smem_v0_addr + pstage * 57344 + (unsigned int)(seg_8 * 8192), (&do_tma), c0_1, head_1_2, row0_2_1, tiles_full_addr + (pstage) * 8);
                            tma_3d_gmem2smem(smem_v1_addr + pstage * 57344 + (unsigned int)(seg_8 * 8192), (&vn_tma), c0_1, head_1_2, row0_2_1, tiles_full_addr + (pstage) * 8);
                            tma_3d_gmem2smem(smem_v2_addr + pstage * 57344 + (unsigned int)(seg_8 * 8192), (&dv2_tma), c0_1, head_1_2, row0_2_1, tiles_full_addr + (pstage) * 8);
                        }
                        tma_3d_gmem2smem(akk_k_addr + pstage * 57344, (&akk_tma), 0, head_1_2, row0_2_1, tiles_full_addr + (pstage) * 8);
                        pstage += 1;
                        if (pstage == 2) { pstage = 0; _phase_tiles_empty ^= 1; }
                    }
                    #pragma unroll
                    for (int sl_1 = 0; sl_1 < 4; sl_1++) {
                        int c5 = sl_1 * 32;
                        tma_store_3d((&dg_tma), c5, head_1, row0_1, dg_stg_addr + (unsigned int)(sl_1 * 8192));
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
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
                            :: "l"((void*)(&((__nv_bfloat16*)(dv_out + ((ext_row * num_heads + head_1) * 128 + col_1)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
            }
        }
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, db_acc, 16);
        float db_other = _shfl_xor_0;
        float db_row = db_acc + db_other;
        if (lane_half == 0) {
            dbp_s[cblk * 64 + t_row] = db_row;
        }
        asm volatile("barrier.sync 8, 512;" ::: "memory");
        #pragma unroll
        for (int q_10 = 0; q_10 < 4; q_10++) {
            asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"((dk_stg_addr + (unsigned int)(slot * 57344) + (unsigned int)((colb + q_10 * 4) / 32 * 8192 + t_row * 128 + (colb + q_10 * 4) % 32 * 4 ^ ((colb + q_10 * 4) / 32 * 8192 + t_row * 128 + (colb + q_10 * 4) % 32 * 4 >> 7 & 7) << 4))), "f"(dkf[q_10 * 4]), "f"(dkf[q_10 * 4 + 1]), "f"(dkf[q_10 * 4 + 2]), "f"(dkf[q_10 * 4 + 3]) : "memory");
        }
        if (cblk == 0) {
            if (lane_half == 0) {
                float db_sum = dbp_s[t_row];
                db_sum = db_sum + dbp_s[64 + t_row];
                db_sum = db_sum + dbp_s[128 + t_row];
                db_sum = db_sum + dbp_s[192 + t_row];
                db_out[grow * num_heads + head_1] = db_sum;
            }
        }
        if (cblk < 2) {
            int blk_2 = cblk;
            int col_2 = blk_2 * 32 + lane_half * 16;
            float _tmem_load_6[16];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                : "r"(taddr + 384 + (unsigned int)row_base + (unsigned int)(blk_2 * 32)));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            float dab[16];
            #pragma unroll
            for (int j_3 = 0; j_3 < 16; j_3++) {
                int s_col = col_2 + j_3;
                float val = _tmem_load_6[j_3] * beta_s[s_col];
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
            for (int q_11 = 0; q_11 < 2; q_11++) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dab_k_addr + (unsigned int)(slot * 57344) + (unsigned int)((col_2 + q_11 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_11 * 8) % 64 * 2 ^ ((col_2 + q_11 * 8) / 64 * 8192 + t_row * 128 + (col_2 + q_11 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(dab_bf16[q_11 * 4]), "r"(dab_bf16[q_11 * 4 + 1]), "r"(dab_bf16[q_11 * 4 + 2]), "r"(dab_bf16[q_11 * 4 + 3]) : "memory");
            }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(b_drained_addr);
        if (warp == 15) {
            if (elect_sync()) {
                mbarrier_wait(b_drained_addr, _phase_b_drained_0);
                _phase_b_drained_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_7 = (((dab_k_addr) >> 4) & 0x3FFF) + (stage) * 3584;
                int _mma_b_lo_7 = (((akk_k_addr) >> 4) & 0x3FFF) + (stage) * 3584;
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
                #pragma unroll
                for (int sl_2 = 0; sl_2 < 4; sl_2++) {
                    int c6 = sl_2 * 32;
                    tma_store_3d((&dk_tma), c6, head_1, row0_1, dk_stg_addr + (unsigned int)(slot * 57344) + (unsigned int)(sl_2 * 8192));
                }
                asm volatile("cp.async.bulk.commit_group;");
            }
        }
        mbarrier_wait(c_ready_addr, _phase_c_ready_0);
        _phase_c_ready_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (cblk < 2) {
            int blk_3 = cblk;
            int col_3 = blk_3 * 32 + lane_half * 16;
            float _tmem_load_7[16];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                : "r"(taddr + 448 + (unsigned int)row_base + (unsigned int)(blk_3 * 32)));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            uint32_t _tmem_load_7_bf16[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                _tmem_load_7_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int q_12 = 0; q_12 < 2; q_12++) {
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((da1b_w_addr + (unsigned int)(slot * 57344) + (unsigned int)((col_3 + q_12 * 8) / 64 * 8192 + t_row * 128 + (col_3 + q_12 * 8) % 64 * 2 ^ ((col_3 + q_12 * 8) / 64 * 8192 + t_row * 128 + (col_3 + q_12 * 8) % 64 * 2 >> 7 & 7) << 4))), "r"(_tmem_load_7_bf16[q_12 * 4]), "r"(_tmem_load_7_bf16[q_12 * 4 + 1]), "r"(_tmem_load_7_bf16[q_12 * 4 + 2]), "r"(_tmem_load_7_bf16[q_12 * 4 + 3]) : "memory");
            }
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbarrier_arrive(c_drained_addr);
        if (warp == 15) {
            if (elect_sync()) {
                mbarrier_wait(c_drained_addr, _phase_c_drained_0);
                _phase_c_drained_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_8 = ((((akk_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage) * 3584;
                int _mma_b_lo_8 = ((((da1b_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage) * 3584;
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
                tcgen05_commit(tiles_empty_addr + (stage) * 8);
            }
        }
        mbarrier_wait(d_ready_addr, _phase_d_ready_0);
        _phase_d_ready_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        if (warp == 15) {
            if (elect_sync()) {
                if (nxt < num_items) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    int chunk_0_2 = nxt / num_heads;
                    int head_1_3 = nxt % num_heads;
                    int row0_2_2 = chunk_0_2 * 64;
                    int hrow0_3_2 = (chunk_0_2 * num_heads + head_1_3) * 128;
                    mbarrier_arrive_expect_tx(ops_full_addr, 49152);
                    #pragma unroll
                    for (int sl_3 = 0; sl_3 < 4; sl_3++) {
                        int c3_1 = sl_3 * 32;
                        tma_3d_gmem2smem(gk_r_addr + (unsigned int)(slot * 57344) + (unsigned int)(sl_3 * 8192), (&gk_tma), c3_1, head_1_3, row0_2_2, ops_full_addr);
                    }
                    #pragma unroll
                    for (int seg_9 = 0; seg_9 < 2; seg_9++) {
                        int c4_1 = seg_9 * 64;
                        tma_3d_gmem2smem(smem_v23_addr + (unsigned int)(slot * 57344) + (unsigned int)(seg_9 * 8192), (&k_tma), c4_1, head_1_3, row0_2_2, ops_full_addr);
                    }
                }
            }
        }
        int abase = (grow * num_heads + head_1) * 64;
        if (cblk < 2) {
            int blk_4 = cblk;
            int col_4 = blk_4 * 32 + lane_half * 16;
            float _tmem_load_8[16];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                : "r"(taddr + 448 + (unsigned int)row_base + (unsigned int)(blk_4 * 32)));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            float outv[16];
            #pragma unroll
            for (int j_4 = 0; j_4 < 16; j_4++) {
                int s_col2 = col_4 + j_4;
                float v2 = -_tmem_load_8[j_4];
                if (s_col2 >= t_row) {
                    v2 = 0.0f;
                }
                outv[j_4] = v2;
            }
            float outv_lo[8];
            float outv_hi[8];
            #pragma unroll
            for (int jj_1 = 0; jj_1 < 8; jj_1++) {
                outv_lo[jj_1] = outv[jj_1];
                outv_hi[jj_1] = outv[jj_1 + 8];
            }
            {
                unsigned _stv8_2_0 = __float_as_uint(outv_lo[0 + 0]);
                unsigned _stv8_2_1 = __float_as_uint(outv_lo[0 + 1]);
                unsigned _stv8_2_2 = __float_as_uint(outv_lo[0 + 2]);
                unsigned _stv8_2_3 = __float_as_uint(outv_lo[0 + 3]);
                unsigned _stv8_2_4 = __float_as_uint(outv_lo[0 + 4]);
                unsigned _stv8_2_5 = __float_as_uint(outv_lo[0 + 5]);
                unsigned _stv8_2_6 = __float_as_uint(outv_lo[0 + 6]);
                unsigned _stv8_2_7 = __float_as_uint(outv_lo[0 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(dAkk_out + (abase + col_4) + (0))), "r"(_stv8_2_0), "r"(_stv8_2_1), "r"(_stv8_2_2), "r"(_stv8_2_3), "r"(_stv8_2_4), "r"(_stv8_2_5), "r"(_stv8_2_6), "r"(_stv8_2_7) : "memory");
            }
            {
                unsigned _stv8_3_0 = __float_as_uint(outv_hi[0 + 0]);
                unsigned _stv8_3_1 = __float_as_uint(outv_hi[0 + 1]);
                unsigned _stv8_3_2 = __float_as_uint(outv_hi[0 + 2]);
                unsigned _stv8_3_3 = __float_as_uint(outv_hi[0 + 3]);
                unsigned _stv8_3_4 = __float_as_uint(outv_hi[0 + 4]);
                unsigned _stv8_3_5 = __float_as_uint(outv_hi[0 + 5]);
                unsigned _stv8_3_6 = __float_as_uint(outv_hi[0 + 6]);
                unsigned _stv8_3_7 = __float_as_uint(outv_hi[0 + 7]);
                asm volatile(
                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "l"((void*)(dAkk_out + (abase + col_4 + 8) + (0))), "r"(_stv8_3_0), "r"(_stv8_3_1), "r"(_stv8_3_2), "r"(_stv8_3_3), "r"(_stv8_3_4), "r"(_stv8_3_5), "r"(_stv8_3_6), "r"(_stv8_3_7) : "memory");
            }
        }
        mbarrier_arrive(d_drained_addr);
        it_i = it_i + 1;
        stage += 1;
        if (stage == 2) { stage = 0; _phase_tiles_full ^= 1; }
    }
    if (warp == 15) {
        if (elect_sync()) {
            asm volatile("cp.async.bulk.wait_group 0;");
        }
    }
    asm volatile("barrier.sync 8, 512;" ::: "memory");
    if (warp == 0) {
        int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
