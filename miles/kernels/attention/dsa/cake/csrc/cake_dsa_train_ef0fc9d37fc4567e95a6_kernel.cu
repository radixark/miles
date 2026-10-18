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
#define TMEM_NCOLS 256
#define TMEM_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_Q_K_OFF 1024
#define SMEM_Q_K_STAGE_BYTES 18432
#define SMEM_Q_K_STRIDE 18432
#define SMEM_Q_MN_OFF 1024
#define SMEM_Q_MN_STAGE_BYTES 4096
#define SMEM_Q_MN_STRIDE 4096
#define SMEM_QR_MN_OFF 17408
#define SMEM_QR_MN_STAGE_BYTES 2048
#define SMEM_QR_MN_STRIDE 2048
#define SMEM_DO_K_OFF 19456
#define SMEM_DO_K_STAGE_BYTES 16384
#define SMEM_DO_K_STRIDE 16384
#define SMEM_DO_MN_OFF 19456
#define SMEM_DO_MN_STAGE_BYTES 4096
#define SMEM_DO_MN_STRIDE 4096
#define SMEM_K_K_OFF 35840
#define SMEM_K_K_STAGE_BYTES 73728
#define SMEM_K_K_STRIDE 81920
#define SMEM_V_K_OFF 35840
#define SMEM_V_K_STAGE_BYTES 65536
#define SMEM_V_K_STRIDE 81920
#define SMEM_K_MN0_OFF 35840
#define SMEM_K_MN0_STAGE_BYTES 16384
#define SMEM_K_MN0_STRIDE 81920
#define SMEM_K_MN1_OFF 52224
#define SMEM_K_MN1_STAGE_BYTES 16384
#define SMEM_K_MN1_STRIDE 81920
#define SMEM_K_MN2_OFF 68608
#define SMEM_K_MN2_STAGE_BYTES 16384
#define SMEM_K_MN2_STRIDE 81920
#define SMEM_K_MN3_OFF 84992
#define SMEM_K_MN3_STAGE_BYTES 16384
#define SMEM_K_MN3_STRIDE 81920
#define SMEM_KR_MN_OFF 101376
#define SMEM_KR_MN_STAGE_BYTES 8192
#define SMEM_KR_MN_STRIDE 81920
#define SMEM_K_BLK_OFF 35840
#define SMEM_K_BLK_STAGE_BYTES 8192
#define SMEM_K_BLK_STRIDE 8192
#define SMEM_P_K_OFF 199680
#define SMEM_P_K_STAGE_BYTES 2048
#define SMEM_P_K_STRIDE 2048
#define SMEM_DS_K_OFF 207872
#define SMEM_DS_K_STAGE_BYTES 2048
#define SMEM_DS_K_STRIDE 2048
#define SMEM_DS_HK_OFF 216064
#define SMEM_DS_HK_STAGE_BYTES 1024
#define SMEM_DS_HK_STRIDE 1024
#define SMEM_ZERO_QDO_OFF 1024
#define SMEM_ZERO_QDO_STAGE_BYTES 34816
#define SMEM_ZERO_QDO_STRIDE 34816
#define SMEM_ZERO_PDS_OFF 199680
#define SMEM_ZERO_PDS_STAGE_BYTES 16384
#define SMEM_ZERO_PDS_STRIDE 16384
#define SMEM_STATS_OFF 217088
#define SMEM_STATS_STAGE_BYTES 256
#define SMEM_STATS_STRIDE 256
#define SMEM_TILE_IDX_OFF 217344
#define SMEM_TILE_IDX_STAGE_BYTES 640
#define SMEM_TILE_IDX_STRIDE 640
#define SMEM_VALIDITY8_OFF 217344
#define SMEM_VALIDITY8_STAGE_BYTES 640
#define SMEM_VALIDITY8_STRIDE 640
#define SMEM_VALIDITY32_OFF 217344
#define SMEM_VALIDITY32_STAGE_BYTES 640
#define SMEM_VALIDITY32_STRIDE 640
#define SMEM_BLOCKS_WORD_OFF 217984
#define SMEM_BLOCKS_WORD_STAGE_BYTES 4
#define SMEM_BLOCKS_WORD_STRIDE 4
#define SMEM_TOTAL 218112
#define THREADS 640
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





__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_train_ef0fc9d37fc4567e95a6(const __grid_constant__ CUtensorMap q_latent, const __grid_constant__ CUtensorMap q_rope, const __grid_constant__ CUtensorMap dout, const __grid_constant__ CUtensorMap dq_latent, const __grid_constant__ CUtensorMap dq_rope, const __grid_constant__ CUtensorMap kv_latent, const __grid_constant__ CUtensorMap k_rope, float* __restrict__ lse, float* __restrict__ delta, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ o_lo, int* __restrict__ indices, int* __restrict__ topk_length, float* __restrict__ dkv_f32, float* __restrict__ dkr_f32, int dkv_stride, int dkr_stride, int dkr_col0, int* __restrict__ dkv_dst_map, int dkv_has_map, int num_queries, int num_kv, int topk, int idx_stride, int indices_offset, int has_topk_length, int token_base, int token_step, float scale_log2, float sm_scale, int pass_lo, int pass_hi, int dq_mode, float* __restrict__ dq_partial, int* __restrict__ key_scratch, int* __restrict__ pass_counts)
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
    #define qdo_full_addr (mbar_base + 0)
    #define pads_ready_addr (mbar_base + 8)
    #define stats_full_addr (mbar_base + 16)
    #define blocks_ready_addr (mbar_base + 24)
    #define dq_done_addr (mbar_base + 32)
    #define idx_full_addr (mbar_base + 40)
    #define idx_free_addr (mbar_base + 56)
    #define k_full_addr (mbar_base + 72)
    #define k_free_addr (mbar_base + 88)
    #define s_full_addr (mbar_base + 104)
    #define s_free_addr (mbar_base + 112)
    #define dp_full_addr (mbar_base + 120)
    #define dp_free_addr (mbar_base + 128)
    #define p_full_addr (mbar_base + 136)
    #define p_free_addr (mbar_base + 144)
    #define ds_full_addr (mbar_base + 152)
    #define ds_free_addr (mbar_base + 160)
    #define dkr_full_addr (mbar_base + 168)
    #define dkr_drained_addr (mbar_base + 176)
    #define dkv_a_full_addr (mbar_base + 184)
    #define dkv_a_drained_addr (mbar_base + 192)
    #define dkv_b_full_addr (mbar_base + 200)
    #define dkv_b_drained_addr (mbar_base + 208)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* q_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_Q_K_OFF);
    const int q_k_addr = smem + SMEM_Q_K_OFF;
    __nv_bfloat16* q_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_Q_MN_OFF);
    const int q_mn_addr = smem + SMEM_Q_MN_OFF;
    __nv_bfloat16* qr_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_QR_MN_OFF);
    const int qr_mn_addr = smem + SMEM_QR_MN_OFF;
    __nv_bfloat16* do_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DO_K_OFF);
    const int do_k_addr = smem + SMEM_DO_K_OFF;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DO_MN_OFF);
    const int do_mn_addr = smem + SMEM_DO_MN_OFF;
    __nv_bfloat16* k_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_K_K_OFF);
    const int k_k_addr = smem + SMEM_K_K_OFF;
    __nv_bfloat16* v_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_V_K_OFF);
    const int v_k_addr = smem + SMEM_V_K_OFF;
    __nv_bfloat16* k_mn0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_K_MN0_OFF);
    const int k_mn0_addr = smem + SMEM_K_MN0_OFF;
    __nv_bfloat16* k_mn1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_K_MN1_OFF);
    const int k_mn1_addr = smem + SMEM_K_MN1_OFF;
    __nv_bfloat16* k_mn2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_K_MN2_OFF);
    const int k_mn2_addr = smem + SMEM_K_MN2_OFF;
    __nv_bfloat16* k_mn3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_K_MN3_OFF);
    const int k_mn3_addr = smem + SMEM_K_MN3_OFF;
    __nv_bfloat16* kr_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_KR_MN_OFF);
    const int kr_mn_addr = smem + SMEM_KR_MN_OFF;
    __nv_bfloat16* k_blk = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_K_BLK_OFF);
    const int k_blk_addr = smem + SMEM_K_BLK_OFF;
    __nv_bfloat16* p_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_P_K_OFF);
    const int p_k_addr = smem + SMEM_P_K_OFF;
    __nv_bfloat16* ds_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DS_K_OFF);
    const int ds_k_addr = smem + SMEM_DS_K_OFF;
    __nv_bfloat16* ds_hk = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DS_HK_OFF);
    const int ds_hk_addr = smem + SMEM_DS_HK_OFF;
    unsigned int* zero_qdo = reinterpret_cast<unsigned int*>(smem_raw + SMEM_ZERO_QDO_OFF);
    const int zero_qdo_addr = smem + SMEM_ZERO_QDO_OFF;
    unsigned int* zero_pds = reinterpret_cast<unsigned int*>(smem_raw + SMEM_ZERO_PDS_OFF);
    const int zero_pds_addr = smem + SMEM_ZERO_PDS_OFF;
    float* stats = reinterpret_cast<float*>(smem_raw + SMEM_STATS_OFF);
    const int stats_addr = smem + SMEM_STATS_OFF;
    int* tile_idx = reinterpret_cast<int*>(smem_raw + SMEM_TILE_IDX_OFF);
    const int tile_idx_addr = smem + SMEM_TILE_IDX_OFF;
    uint8_t* validity8 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_VALIDITY8_OFF);
    const int validity8_addr = smem + SMEM_VALIDITY8_OFF;
    unsigned int* validity32 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_VALIDITY32_OFF);
    const int validity32_addr = smem + SMEM_VALIDITY32_OFF;
    int* blocks_word = reinterpret_cast<int*>(smem_raw + SMEM_BLOCKS_WORD_OFF);
    const int blocks_word_addr = smem + SMEM_BLOCKS_WORD_OFF;
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_latent))) : "memory"); }
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_rope))) : "memory"); }
    if (warp == 17 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dout))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&kv_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&k_rope))) : "memory"); }
    if (warp == 4 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dq_latent))) : "memory"); }
    if (warp == 4 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&dq_rope))) : "memory"); }

    // Mbarrier init (23 pipeline groups, 0 ordered-sequence groups, 27 barriers)
    // Mbarriers at smem_raw[0..216)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // qdo_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // pads_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 8, 128);
            // stats_full: 1 barriers, init_count=32
            mbarrier_init(smem + 16, 32);
            // blocks_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // dq_done: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // idx_full: 2 barriers, init_count=8
            mbarrier_init(smem + 40, 8);
            mbarrier_init(smem + 48, 8);
            // idx_free: 2 barriers, init_count=16
            mbarrier_init(smem + 56, 16);
            mbarrier_init(smem + 64, 16);
            // k_full: 2 barriers, init_count=4
            mbarrier_init(smem + 72, 4);
            mbarrier_init(smem + 80, 4);
            // k_free: 2 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            // s_free: 1 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            // dp_full: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // dp_free: 1 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            // p_full: 1 barriers, init_count=128
            mbarrier_init(smem + 136, 128);
            // p_free: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // ds_full: 1 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            // ds_free: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // dkr_full: 1 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            // dkr_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 176, 256);
            // dkv_a_full: 1 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            // dkv_a_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 192, 256);
            // dkv_b_full: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            // dkv_b_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 208, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 216);
    if (warp == 4) {
        int _tmem_hold = smem + 216;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;

    // ---- Role: gather ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // gather_main
            int pw = warp;
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks = blocks_word[0];
            if (elect_sync()) {
                #pragma unroll 1
                for (int i = 0; i < blocks; i++) {
                    int slot = i % 2;
                    int ks = i % 2;
                    mbarrier_wait(idx_full_addr + (slot) * 8, i / 2 & 1);
                    int rows[16];
                    rows[0] = tile_idx[slot * 80 + pw * 4];
                    rows[1] = tile_idx[slot * 80 + pw * 4 + 1];
                    rows[2] = tile_idx[slot * 80 + pw * 4 + 2];
                    rows[3] = tile_idx[slot * 80 + pw * 4 + 3];
                    rows[4] = tile_idx[slot * 80 + 16 + pw * 4];
                    rows[5] = tile_idx[slot * 80 + 16 + pw * 4 + 1];
                    rows[6] = tile_idx[slot * 80 + 16 + pw * 4 + 2];
                    rows[7] = tile_idx[slot * 80 + 16 + pw * 4 + 3];
                    rows[8] = tile_idx[slot * 80 + 32 + pw * 4];
                    rows[9] = tile_idx[slot * 80 + 32 + pw * 4 + 1];
                    rows[10] = tile_idx[slot * 80 + 32 + pw * 4 + 2];
                    rows[11] = tile_idx[slot * 80 + 32 + pw * 4 + 3];
                    rows[12] = tile_idx[slot * 80 + 48 + pw * 4];
                    rows[13] = tile_idx[slot * 80 + 48 + pw * 4 + 1];
                    rows[14] = tile_idx[slot * 80 + 48 + pw * 4 + 2];
                    rows[15] = tile_idx[slot * 80 + 48 + pw * 4 + 3];
                    mbarrier_arrive(idx_free_addr + (slot) * 8);
                    mbarrier_wait(k_free_addr + (ks) * 8, i / 2 & 1 ^ 1);
                    mbarrier_arrive_expect_tx(k_full_addr + (ks) * 8, 18432);
                    unsigned int kbase = k_blk_addr + (unsigned int)(ks * 81920) + (unsigned int)(pw * 512);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 0 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 2048), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 0 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 4096), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 0 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 6144), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 0 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 8192), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 64 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 8192 + 2048), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 64 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 8192 + 4096), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 64 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 8192 + 6144), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 64 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 16384), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 128 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 16384 + 2048), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 128 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 16384 + 4096), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 128 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 16384 + 6144), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 128 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 24576), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 192 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 24576 + 2048), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 192 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 24576 + 4096), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 192 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 24576 + 6144), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 192 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 32768), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 256 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 32768 + 2048), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 256 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 32768 + 4096), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 256 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 32768 + 6144), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 256 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 40960), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 320 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 40960 + 2048), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 320 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 40960 + 4096), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 320 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 40960 + 6144), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 320 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 49152), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 384 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 49152 + 2048), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 384 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 49152 + 4096), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 384 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 49152 + 6144), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 384 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 57344), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 448 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 57344 + 2048), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 448 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 57344 + 4096), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 448 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 57344 + 6144), "l"(((1) ? ((&kv_latent)) : ((&k_rope)))), "r"(((1) ? 448 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 65536), "l"(((0) ? ((&kv_latent)) : ((&k_rope)))), "r"(((0) ? 512 : 0)), "r"(rows[0]), "r"(rows[1]), "r"(rows[2]), "r"(rows[3]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 65536 + 2048), "l"(((0) ? ((&kv_latent)) : ((&k_rope)))), "r"(((0) ? 512 : 0)), "r"(rows[4]), "r"(rows[5]), "r"(rows[6]), "r"(rows[7]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 65536 + 4096), "l"(((0) ? ((&kv_latent)) : ((&k_rope)))), "r"(((0) ? 512 : 0)), "r"(rows[8]), "r"(rows[9]), "r"(rows[10]), "r"(rows[11]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(kbase + 65536 + 6144), "l"(((0) ? ((&kv_latent)) : ((&k_rope)))), "r"(((0) ? 512 : 0)), "r"(rows[12]), "r"(rows[13]), "r"(rows[14]), "r"(rows[15]), "r"(k_full_addr + (ks) * 8), "l"(0x14F0000000000000ULL) : "memory");
                }
            }
        }
    }
    // ---- Role: compute ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 128;");
        { // compute_main
            int token = token_base + token_step * blockIdx.x;
            int lane_0 = lane;
            int w = warp - 4;
            int tid_1 = w * 32 + lane_0;
            zero_qdo[256 + tid_1 * 2] = 0;
            zero_qdo[256 + tid_1 * 2 + 1] = 0;
            zero_qdo[768 + tid_1 * 2] = 0;
            zero_qdo[768 + tid_1 * 2 + 1] = 0;
            zero_qdo[1280 + tid_1 * 2] = 0;
            zero_qdo[1280 + tid_1 * 2 + 1] = 0;
            zero_qdo[1792 + tid_1 * 2] = 0;
            zero_qdo[1792 + tid_1 * 2 + 1] = 0;
            zero_qdo[2304 + tid_1 * 2] = 0;
            zero_qdo[2304 + tid_1 * 2 + 1] = 0;
            zero_qdo[2816 + tid_1 * 2] = 0;
            zero_qdo[2816 + tid_1 * 2 + 1] = 0;
            zero_qdo[3328 + tid_1 * 2] = 0;
            zero_qdo[3328 + tid_1 * 2 + 1] = 0;
            zero_qdo[3840 + tid_1 * 2] = 0;
            zero_qdo[3840 + tid_1 * 2 + 1] = 0;
            zero_qdo[4352 + tid_1 * 2] = 0;
            zero_qdo[4352 + tid_1 * 2 + 1] = 0;
            zero_qdo[4864 + tid_1 * 2] = 0;
            zero_qdo[4864 + tid_1 * 2 + 1] = 0;
            zero_qdo[5376 + tid_1 * 2] = 0;
            zero_qdo[5376 + tid_1 * 2 + 1] = 0;
            zero_qdo[5888 + tid_1 * 2] = 0;
            zero_qdo[5888 + tid_1 * 2 + 1] = 0;
            zero_qdo[6400 + tid_1 * 2] = 0;
            zero_qdo[6400 + tid_1 * 2 + 1] = 0;
            zero_qdo[6912 + tid_1 * 2] = 0;
            zero_qdo[6912 + tid_1 * 2 + 1] = 0;
            zero_qdo[7424 + tid_1 * 2] = 0;
            zero_qdo[7424 + tid_1 * 2 + 1] = 0;
            zero_qdo[7936 + tid_1 * 2] = 0;
            zero_qdo[7936 + tid_1 * 2 + 1] = 0;
            zero_qdo[8448 + tid_1 * 2] = 0;
            zero_qdo[8448 + tid_1 * 2 + 1] = 0;
            zero_pds[tid_1] = 0;
            zero_pds[tid_1 + 128] = 0;
            zero_pds[tid_1 + 256] = 0;
            zero_pds[tid_1 + 384] = 0;
            zero_pds[tid_1 + 512] = 0;
            zero_pds[tid_1 + 640] = 0;
            zero_pds[tid_1 + 768] = 0;
            zero_pds[tid_1 + 896] = 0;
            zero_pds[tid_1 + 1024] = 0;
            zero_pds[tid_1 + 1152] = 0;
            zero_pds[tid_1 + 1280] = 0;
            zero_pds[tid_1 + 1408] = 0;
            zero_pds[tid_1 + 1536] = 0;
            zero_pds[tid_1 + 1664] = 0;
            zero_pds[tid_1 + 1792] = 0;
            zero_pds[tid_1 + 1920] = 0;
            zero_pds[tid_1 + 2048] = 0;
            zero_pds[tid_1 + 2176] = 0;
            zero_pds[tid_1 + 2304] = 0;
            zero_pds[tid_1 + 2432] = 0;
            zero_pds[tid_1 + 2560] = 0;
            zero_pds[tid_1 + 2688] = 0;
            zero_pds[tid_1 + 2816] = 0;
            zero_pds[tid_1 + 2944] = 0;
            zero_pds[tid_1 + 3072] = 0;
            zero_pds[tid_1 + 3200] = 0;
            zero_pds[tid_1 + 3328] = 0;
            zero_pds[tid_1 + 3456] = 0;
            zero_pds[tid_1 + 3584] = 0;
            zero_pds[tid_1 + 3712] = 0;
            zero_pds[tid_1 + 3840] = 0;
            zero_pds[tid_1 + 3968] = 0;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(pads_ready_addr);
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_1 = blocks_word[0];
            mbarrier_wait(stats_full_addr, 0);
            int h0 = 2 * (lane_0 % 4);
            float lsev[2];
            float dlv[2];
            lsev[0] = stats[h0];
            dlv[0] = stats[32 + h0];
            lsev[1] = stats[h0 + 1];
            dlv[1] = stats[32 + h0 + 1];
            int kl = lane_0 / 4;
            int bit_a = 16 * (w % 2) + kl;
            int j16 = lane_0 % 16;
            int pk_offa[1];
            int dsh_offa[1];
            int qrow_a[1];
            int hb = 0;
            pk_offa[0] = (16 * w + j16) * 128 + (hb ^ j16 % 8) * 16;
            dsh_offa[0] = (8 * hb + j16 % 8) * 128 + (2 * w + j16 / 8 ^ j16 % 8) * 16;
            qrow_a[0] = (8 * hb + j16 % 8) * 128;
            #pragma unroll 1
            for (int i_1 = 0; i_1 < blocks_1; i_1++) {
                unsigned int par = i_1 & 1;
                int slot_1 = i_1 % 2;
                mbarrier_wait(idx_full_addr + (slot_1) * 8, i_1 / 2 & 1);
                unsigned int vw = validity32[slot_1 * 80 + 64 + w / 2];
                if (lane_0 == 0) {
                    mbarrier_arrive(idx_free_addr + (slot_1) * 8);
                }
                mbarrier_wait(s_full_addr, par);
                float s[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&s[0])), "=r"(*reinterpret_cast<uint32_t*>(&s[1])), "=r"(*reinterpret_cast<uint32_t*>(&s[2])), "=r"(*reinterpret_cast<uint32_t*>(&s[3]))
                    : "r"(tmem_tmem));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float p[4];
                float _fma_0 = __fmaf_rn(s[0], scale_log2, -lsev[0]);
                float _exp2_0 = approx_exp2(_fma_0);
                p[0] = _exp2_0;
                float _fma_1 = __fmaf_rn(s[1], scale_log2, -lsev[1]);
                float _exp2_1 = approx_exp2(_fma_1);
                p[1] = _exp2_1;
                float _fma_2 = __fmaf_rn(s[2], scale_log2, -lsev[0]);
                float _exp2_2 = approx_exp2(_fma_2);
                p[2] = _exp2_2;
                float _fma_3 = __fmaf_rn(s[3], scale_log2, -lsev[1]);
                float _exp2_3 = approx_exp2(_fma_3);
                p[3] = _exp2_3;
                if ((vw >> (unsigned int)bit_a & 1) == 0) {
                    p[0] = 0.0f;
                    p[1] = 0.0f;
                }
                if ((vw >> (unsigned int)(bit_a + 8) & 1) == 0) {
                    p[2] = 0.0f;
                    p[3] = 0.0f;
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(s_free_addr);
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                mbarrier_wait(p_free_addr, par ^ 1);
                uint32_t p_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p[_lp*2 + 0], p[_lp*2+1 + 0]));
                    p_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(p_k_addr + (unsigned int)pk_offa[0]);
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&p_bf16[1]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(p_full_addr);
                mbarrier_wait(dp_full_addr, par);
                float dp[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&dp[0])), "=r"(*reinterpret_cast<uint32_t*>(&dp[1])), "=r"(*reinterpret_cast<uint32_t*>(&dp[2])), "=r"(*reinterpret_cast<uint32_t*>(&dp[3]))
                    : "r"(tmem_tmem + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float ds[4];
                ds[0] = p[0] * (dp[0] - dlv[0]) * sm_scale;
                ds[1] = p[1] * (dp[1] - dlv[1]) * sm_scale;
                ds[2] = p[2] * (dp[2] - dlv[0]) * sm_scale;
                ds[3] = p[3] * (dp[3] - dlv[1]) * sm_scale;
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dp_free_addr);
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                mbarrier_wait(ds_free_addr, par ^ 1);
                uint32_t ds_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ds[_lp*2 + 0], ds[_lp*2+1 + 0]));
                    ds_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(ds_k_addr + (unsigned int)pk_offa[0]);
                asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[1]))
                    : "memory");
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(ds_hk_addr + (unsigned int)dsh_offa[0]);
                asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&ds_bf16[1]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(ds_full_addr);
            }
            mbarrier_wait(dq_done_addr, 0);
            float dq[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq[3]))
                : "r"(tmem_tmem + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_bf16[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq[_lp*2 + 0], dq[_lp*2+1 + 0]));
                dq_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int qchunk = (4 * (w % 2) + j16 / 8 ^ j16 % 8) * 16;
            uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(k_blk_addr + (unsigned int)(w / 2 * 1024) + (unsigned int)qrow_a[0] + qchunk);
            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_bf16[1]))
                : "memory");
            float dq_1[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_1[3]))
                : "r"(tmem_tmem + 32 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_1_bf16[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_1[_lp*2 + 0], dq_1[_lp*2+1 + 0]));
                dq_1_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int qchunk_2 = (4 * (w % 2) + 2 + j16 / 8 ^ j16 % 8) * 16;
            uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(k_blk_addr + (unsigned int)(w / 2 * 1024) + (unsigned int)qrow_a[0] + qchunk_2);
            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&dq_1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_1_bf16[1]))
                : "memory");
            float dq_3[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_3[3]))
                : "r"(tmem_tmem + 32 + 8));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_3_bf16[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_3[_lp*2 + 0], dq_3[_lp*2+1 + 0]));
                dq_3_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int qchunk_4 = (4 * (w % 2) + j16 / 8 ^ j16 % 8) * 16;
            uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(k_blk_addr + (unsigned int)((2 + w / 2) * 1024) + (unsigned int)qrow_a[0] + qchunk_4);
            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&dq_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_3_bf16[1]))
                : "memory");
            float dq_5[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_5[3]))
                : "r"(tmem_tmem + 32 + 8 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_5_bf16[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_5[_lp*2 + 0], dq_5[_lp*2+1 + 0]));
                dq_5_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int qchunk_6 = (4 * (w % 2) + 2 + j16 / 8 ^ j16 % 8) * 16;
            uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(k_blk_addr + (unsigned int)((2 + w / 2) * 1024) + (unsigned int)qrow_a[0] + qchunk_6);
            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&dq_5_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_5_bf16[1]))
                : "memory");
            float dq_7[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_7[3]))
                : "r"(tmem_tmem + 32 + 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_7_bf16[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_7[_lp*2 + 0], dq_7[_lp*2+1 + 0]));
                dq_7_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int qchunk_8 = (4 * (w % 2) + j16 / 8 ^ j16 % 8) * 16;
            uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(k_blk_addr + (unsigned int)((4 + w / 2) * 1024) + (unsigned int)qrow_a[0] + qchunk_8);
            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&dq_7_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_7_bf16[1]))
                : "memory");
            float dq_9[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_9[3]))
                : "r"(tmem_tmem + 32 + 16 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_9_bf16[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_9[_lp*2 + 0], dq_9[_lp*2+1 + 0]));
                dq_9_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int qchunk_10 = (4 * (w % 2) + 2 + j16 / 8 ^ j16 % 8) * 16;
            uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(k_blk_addr + (unsigned int)((4 + w / 2) * 1024) + (unsigned int)qrow_a[0] + qchunk_10);
            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&dq_9_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_9_bf16[1]))
                : "memory");
            float dq_11[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_11[3]))
                : "r"(tmem_tmem + 32 + 24));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_11_bf16[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_11[_lp*2 + 0], dq_11[_lp*2+1 + 0]));
                dq_11_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int qchunk_12 = (4 * (w % 2) + j16 / 8 ^ j16 % 8) * 16;
            uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(k_blk_addr + (unsigned int)((6 + w / 2) * 1024) + (unsigned int)qrow_a[0] + qchunk_12);
            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&dq_11_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_11_bf16[1]))
                : "memory");
            float dq_13[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dq_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&dq_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&dq_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&dq_13[3]))
                : "r"(tmem_tmem + 32 + 24 + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dq_13_bf16[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dq_13[_lp*2 + 0], dq_13[_lp*2+1 + 0]));
                dq_13_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int qchunk_14 = (4 * (w % 2) + 2 + j16 / 8 ^ j16 % 8) * 16;
            uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(k_blk_addr + (unsigned int)((6 + w / 2) * 1024) + (unsigned int)qrow_a[0] + qchunk_14);
            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&dq_13_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dq_13_bf16[1]))
                : "memory");
            float dqr[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&dqr[0])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[1])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[2])), "=r"(*reinterpret_cast<uint32_t*>(&dqr[3]))
                : "r"(tmem_tmem + 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            uint32_t dqr_bf16[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dqr[_lp*2 + 0], dqr[_lp*2+1 + 0]));
                dqr_bf16[_lp] = *(uint32_t*)&_bf2;
            }
            unsigned int rchunk = (2 * w + j16 / 8 ^ j16 % 8) * 16;
            uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(k_blk_addr + 8192 + (unsigned int)qrow_a[0] + rchunk);
            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&dqr_bf16[1]))
                : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (w == 0) {
                if (elect_sync()) {
                    tma_store_4d((&dq_latent), 0, 0, 0, token, k_blk_addr);
                    tma_store_4d((&dq_latent), 0, 0, 1, token, k_blk_addr + 1024);
                    tma_store_4d((&dq_latent), 0, 0, 2, token, k_blk_addr + 2048);
                    tma_store_4d((&dq_latent), 0, 0, 3, token, k_blk_addr + 3072);
                    tma_store_4d((&dq_latent), 0, 0, 4, token, k_blk_addr + 4096);
                    tma_store_4d((&dq_latent), 0, 0, 5, token, k_blk_addr + 5120);
                    tma_store_4d((&dq_latent), 0, 0, 6, token, k_blk_addr + 6144);
                    tma_store_4d((&dq_latent), 0, 0, 7, token, k_blk_addr + 7168);
                    tma_store_4d((&dq_rope), 0, 0, 0, token, k_blk_addr + 8192);
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
            }
        }
    }
    // ---- Role: reduce ----
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // reduce_main
            int lane_0_1 = lane;
            int w2 = (warp - 8) % 4;
            int wg = (warp - 8) / 4;
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_2 = blocks_word[0];
            int npos = 32 * w2 + 8 * (lane_0_1 >> 2 & 3) + 4 * (lane_0_1 >> 4);
            int nrpos = 16 * w2 + (lane_0_1 >> 2 & 6) + 8 * (lane_0_1 >> 2 & 1);
            int mtoken = token_base + token_step * blockIdx.x;
            long long mrow0 = (long long)mtoken * (long long)topk;
            float pa[32];
            float pb[32];
            float pr[16];
            #pragma unroll 1
            for (int i_2 = 0; i_2 < blocks_2; i_2++) {
                unsigned int par_1 = i_2 & 1;
                int slot_2 = i_2 % 2;
                mbarrier_wait(idx_full_addr + (slot_2) * 8, i_2 / 2 & 1);
                int keys[8];
                keys[0] = tile_idx[slot_2 * 80 + wg * 32 + 2 * (lane_0_1 % 4)];
                keys[1] = tile_idx[slot_2 * 80 + wg * 32 + 2 * (lane_0_1 % 4) + 1];
                keys[2] = tile_idx[slot_2 * 80 + wg * 32 + 8 + 2 * (lane_0_1 % 4)];
                keys[3] = tile_idx[slot_2 * 80 + wg * 32 + 8 + 2 * (lane_0_1 % 4) + 1];
                keys[4] = tile_idx[slot_2 * 80 + wg * 32 + 16 + 2 * (lane_0_1 % 4)];
                keys[5] = tile_idx[slot_2 * 80 + wg * 32 + 16 + 2 * (lane_0_1 % 4) + 1];
                keys[6] = tile_idx[slot_2 * 80 + wg * 32 + 24 + 2 * (lane_0_1 % 4)];
                keys[7] = tile_idx[slot_2 * 80 + wg * 32 + 24 + 2 * (lane_0_1 % 4) + 1];
                if (lane_0_1 == 0) {
                    mbarrier_arrive(idx_free_addr + (slot_2) * 8);
                }
                int rows_1[8];
                int mslot0 = (blocks_2 - 1 - i_2) * 64 + wg * 32 + 2 * (lane_0_1 % 4);
                rows_1[0] = mslot0;
                rows_1[1] = mslot0 + 1;
                rows_1[2] = mslot0 + 8;
                rows_1[3] = mslot0 + 8 + 1;
                rows_1[4] = mslot0 + 16;
                rows_1[5] = mslot0 + 16 + 1;
                rows_1[6] = mslot0 + 24;
                rows_1[7] = mslot0 + 24 + 1;
                mbarrier_wait(dkv_a_full_addr, par_1);
                float a0[32];
                float a1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a0[0])), "=r"(*reinterpret_cast<uint32_t*>(&a0[1])), "=r"(*reinterpret_cast<uint32_t*>(&a0[2])), "=r"(*reinterpret_cast<uint32_t*>(&a0[3])), "=r"(*reinterpret_cast<uint32_t*>(&a0[4])), "=r"(*reinterpret_cast<uint32_t*>(&a0[5])), "=r"(*reinterpret_cast<uint32_t*>(&a0[6])), "=r"(*reinterpret_cast<uint32_t*>(&a0[7])), "=r"(*reinterpret_cast<uint32_t*>(&a0[8])), "=r"(*reinterpret_cast<uint32_t*>(&a0[9])), "=r"(*reinterpret_cast<uint32_t*>(&a0[10])), "=r"(*reinterpret_cast<uint32_t*>(&a0[11])), "=r"(*reinterpret_cast<uint32_t*>(&a0[12])), "=r"(*reinterpret_cast<uint32_t*>(&a0[13])), "=r"(*reinterpret_cast<uint32_t*>(&a0[14])), "=r"(*reinterpret_cast<uint32_t*>(&a0[15]))
                    : "r"(tmem_tmem + 64 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[15]))
                    : "r"(tmem_tmem + 64 + wg * 32 + 1048576));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a1[0])), "=r"(*reinterpret_cast<uint32_t*>(&a1[1])), "=r"(*reinterpret_cast<uint32_t*>(&a1[2])), "=r"(*reinterpret_cast<uint32_t*>(&a1[3])), "=r"(*reinterpret_cast<uint32_t*>(&a1[4])), "=r"(*reinterpret_cast<uint32_t*>(&a1[5])), "=r"(*reinterpret_cast<uint32_t*>(&a1[6])), "=r"(*reinterpret_cast<uint32_t*>(&a1[7])), "=r"(*reinterpret_cast<uint32_t*>(&a1[8])), "=r"(*reinterpret_cast<uint32_t*>(&a1[9])), "=r"(*reinterpret_cast<uint32_t*>(&a1[10])), "=r"(*reinterpret_cast<uint32_t*>(&a1[11])), "=r"(*reinterpret_cast<uint32_t*>(&a1[12])), "=r"(*reinterpret_cast<uint32_t*>(&a1[13])), "=r"(*reinterpret_cast<uint32_t*>(&a1[14])), "=r"(*reinterpret_cast<uint32_t*>(&a1[15]))
                    : "r"(tmem_tmem + 128 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[15]))
                    : "r"(tmem_tmem + 128 + wg * 32 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dkv_a_drained_addr);
                float rk[16];
                mbarrier_wait(dkr_full_addr, par_1);
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&rk[0])), "=r"(*reinterpret_cast<uint32_t*>(&rk[1])), "=r"(*reinterpret_cast<uint32_t*>(&rk[2])), "=r"(*reinterpret_cast<uint32_t*>(&rk[3])), "=r"(*reinterpret_cast<uint32_t*>(&rk[4])), "=r"(*reinterpret_cast<uint32_t*>(&rk[5])), "=r"(*reinterpret_cast<uint32_t*>(&rk[6])), "=r"(*reinterpret_cast<uint32_t*>(&rk[7])), "=r"(*reinterpret_cast<uint32_t*>(&rk[8])), "=r"(*reinterpret_cast<uint32_t*>(&rk[9])), "=r"(*reinterpret_cast<uint32_t*>(&rk[10])), "=r"(*reinterpret_cast<uint32_t*>(&rk[11])), "=r"(*reinterpret_cast<uint32_t*>(&rk[12])), "=r"(*reinterpret_cast<uint32_t*>(&rk[13])), "=r"(*reinterpret_cast<uint32_t*>(&rk[14])), "=r"(*reinterpret_cast<uint32_t*>(&rk[15]))
                    : "r"(tmem_tmem + 192 + wg * 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dkr_drained_addr);
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[0] : a0[2]), 4);
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[16] : a0[18]), 4);
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_0 : a0[0]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_1 : a0[16])), 8);
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[2] : _shfl_xor_0) : (((lane_0_1 >> 2 & 1) != 0) ? a0[18] : _shfl_xor_1)), 8);
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[0] : a1[2]), 4);
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[16] : a1[18]), 4);
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_4 : a1[0]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_5 : a1[16])), 8);
                float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[2] : _shfl_xor_4) : (((lane_0_1 >> 2 & 1) != 0) ? a1[18] : _shfl_xor_5)), 8);
                pa[0] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_2 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_0 : a0[0]));
                pb[0] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_6 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_4 : a1[0]));
                pa[1] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_3 : (((lane_0_1 >> 2 & 1) != 0) ? a0[2] : _shfl_xor_0));
                pb[1] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_7 : (((lane_0_1 >> 2 & 1) != 0) ? a1[2] : _shfl_xor_4));
                pa[2] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_1 : a0[16]) : _shfl_xor_2);
                pb[2] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_5 : a1[16]) : _shfl_xor_6);
                pa[3] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[18] : _shfl_xor_1) : _shfl_xor_3);
                pb[3] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[18] : _shfl_xor_5) : _shfl_xor_7);
                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[0] : rk[2]), 4);
                pr[0] = (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_8 : rk[0]);
                pr[1] = (((lane_0_1 >> 2 & 1) != 0) ? rk[2] : _shfl_xor_8);
                int mkey = keys[0];
                if (mkey >= 0) {
                    long long mbase = (mrow0 + (long long)rows_1[0]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[0 + 0], pa[0 + 1], pa[0 + 2], pa[0 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[0 + 0], pb[0 + 1], pb[0 + 2], pb[0 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase + 128) = _v4;
                    }
                    long long mrbase = (mrow0 + (long long)rows_1[0]) * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos;
                    {
                        float2 _v2 = make_float2(pr[0 + 0], pr[0 + 1]);
                        *reinterpret_cast<float2*>(dkr_f32 + mrbase) = _v2;
                    }
                }
                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[1] : a0[3]), 4);
                float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[17] : a0[19]), 4);
                float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_9 : a0[1]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_10 : a0[17])), 8);
                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[3] : _shfl_xor_9) : (((lane_0_1 >> 2 & 1) != 0) ? a0[19] : _shfl_xor_10)), 8);
                float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[1] : a1[3]), 4);
                float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[17] : a1[19]), 4);
                float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_13 : a1[1]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_14 : a1[17])), 8);
                float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[3] : _shfl_xor_13) : (((lane_0_1 >> 2 & 1) != 0) ? a1[19] : _shfl_xor_14)), 8);
                pa[4] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_11 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_9 : a0[1]));
                pb[4] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_15 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_13 : a1[1]));
                pa[5] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_12 : (((lane_0_1 >> 2 & 1) != 0) ? a0[3] : _shfl_xor_9));
                pb[5] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_16 : (((lane_0_1 >> 2 & 1) != 0) ? a1[3] : _shfl_xor_13));
                pa[6] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_10 : a0[17]) : _shfl_xor_11);
                pb[6] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_14 : a1[17]) : _shfl_xor_15);
                pa[7] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[19] : _shfl_xor_10) : _shfl_xor_12);
                pb[7] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[19] : _shfl_xor_14) : _shfl_xor_16);
                float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[1] : rk[3]), 4);
                pr[2] = (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_17 : rk[1]);
                pr[3] = (((lane_0_1 >> 2 & 1) != 0) ? rk[3] : _shfl_xor_17);
                int mkey_0 = keys[1];
                if (mkey_0 >= 0) {
                    long long mbase_1 = (mrow0 + (long long)rows_1[1]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[4 + 0], pa[4 + 1], pa[4 + 2], pa[4 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_1) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[4 + 0], pb[4 + 1], pb[4 + 2], pb[4 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_1 + 128) = _v4;
                    }
                    long long mrbase_1 = (mrow0 + (long long)rows_1[1]) * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos;
                    {
                        float2 _v2 = make_float2(pr[2 + 0], pr[2 + 1]);
                        *reinterpret_cast<float2*>(dkr_f32 + mrbase_1) = _v2;
                    }
                }
                float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[4] : a0[6]), 4);
                float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[20] : a0[22]), 4);
                float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_18 : a0[4]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_19 : a0[20])), 8);
                float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[6] : _shfl_xor_18) : (((lane_0_1 >> 2 & 1) != 0) ? a0[22] : _shfl_xor_19)), 8);
                float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[4] : a1[6]), 4);
                float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[20] : a1[22]), 4);
                float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_22 : a1[4]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_23 : a1[20])), 8);
                float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[6] : _shfl_xor_22) : (((lane_0_1 >> 2 & 1) != 0) ? a1[22] : _shfl_xor_23)), 8);
                pa[8] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_20 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_18 : a0[4]));
                pb[8] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_24 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_22 : a1[4]));
                pa[9] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_21 : (((lane_0_1 >> 2 & 1) != 0) ? a0[6] : _shfl_xor_18));
                pb[9] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_25 : (((lane_0_1 >> 2 & 1) != 0) ? a1[6] : _shfl_xor_22));
                pa[10] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_19 : a0[20]) : _shfl_xor_20);
                pb[10] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_23 : a1[20]) : _shfl_xor_24);
                pa[11] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[22] : _shfl_xor_19) : _shfl_xor_21);
                pb[11] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[22] : _shfl_xor_23) : _shfl_xor_25);
                float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[4] : rk[6]), 4);
                pr[4] = (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_26 : rk[4]);
                pr[5] = (((lane_0_1 >> 2 & 1) != 0) ? rk[6] : _shfl_xor_26);
                int mkey_1 = keys[2];
                if (mkey_1 >= 0) {
                    long long mbase_2 = (mrow0 + (long long)rows_1[2]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[8 + 0], pa[8 + 1], pa[8 + 2], pa[8 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_2) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[8 + 0], pb[8 + 1], pb[8 + 2], pb[8 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_2 + 128) = _v4;
                    }
                    long long mrbase_2 = (mrow0 + (long long)rows_1[2]) * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos;
                    {
                        float2 _v2 = make_float2(pr[4 + 0], pr[4 + 1]);
                        *reinterpret_cast<float2*>(dkr_f32 + mrbase_2) = _v2;
                    }
                }
                float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[5] : a0[7]), 4);
                float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[21] : a0[23]), 4);
                float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_27 : a0[5]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_28 : a0[21])), 8);
                float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[7] : _shfl_xor_27) : (((lane_0_1 >> 2 & 1) != 0) ? a0[23] : _shfl_xor_28)), 8);
                float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[5] : a1[7]), 4);
                float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[21] : a1[23]), 4);
                float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_31 : a1[5]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_32 : a1[21])), 8);
                float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[7] : _shfl_xor_31) : (((lane_0_1 >> 2 & 1) != 0) ? a1[23] : _shfl_xor_32)), 8);
                pa[12] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_29 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_27 : a0[5]));
                pb[12] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_33 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_31 : a1[5]));
                pa[13] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_30 : (((lane_0_1 >> 2 & 1) != 0) ? a0[7] : _shfl_xor_27));
                pb[13] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_34 : (((lane_0_1 >> 2 & 1) != 0) ? a1[7] : _shfl_xor_31));
                pa[14] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_28 : a0[21]) : _shfl_xor_29);
                pb[14] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_32 : a1[21]) : _shfl_xor_33);
                pa[15] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[23] : _shfl_xor_28) : _shfl_xor_30);
                pb[15] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[23] : _shfl_xor_32) : _shfl_xor_34);
                float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[5] : rk[7]), 4);
                pr[6] = (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_35 : rk[5]);
                pr[7] = (((lane_0_1 >> 2 & 1) != 0) ? rk[7] : _shfl_xor_35);
                int mkey_2 = keys[3];
                if (mkey_2 >= 0) {
                    long long mbase_3 = (mrow0 + (long long)rows_1[3]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[12 + 0], pa[12 + 1], pa[12 + 2], pa[12 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_3) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[12 + 0], pb[12 + 1], pb[12 + 2], pb[12 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_3 + 128) = _v4;
                    }
                    long long mrbase_3 = (mrow0 + (long long)rows_1[3]) * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos;
                    {
                        float2 _v2 = make_float2(pr[6 + 0], pr[6 + 1]);
                        *reinterpret_cast<float2*>(dkr_f32 + mrbase_3) = _v2;
                    }
                }
                float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[8] : a0[10]), 4);
                float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[24] : a0[26]), 4);
                float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_36 : a0[8]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_37 : a0[24])), 8);
                float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[10] : _shfl_xor_36) : (((lane_0_1 >> 2 & 1) != 0) ? a0[26] : _shfl_xor_37)), 8);
                float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[8] : a1[10]), 4);
                float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[24] : a1[26]), 4);
                float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_40 : a1[8]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_41 : a1[24])), 8);
                float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[10] : _shfl_xor_40) : (((lane_0_1 >> 2 & 1) != 0) ? a1[26] : _shfl_xor_41)), 8);
                pa[16] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_38 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_36 : a0[8]));
                pb[16] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_42 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_40 : a1[8]));
                pa[17] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_39 : (((lane_0_1 >> 2 & 1) != 0) ? a0[10] : _shfl_xor_36));
                pb[17] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_43 : (((lane_0_1 >> 2 & 1) != 0) ? a1[10] : _shfl_xor_40));
                pa[18] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_37 : a0[24]) : _shfl_xor_38);
                pb[18] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_41 : a1[24]) : _shfl_xor_42);
                pa[19] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[26] : _shfl_xor_37) : _shfl_xor_39);
                pb[19] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[26] : _shfl_xor_41) : _shfl_xor_43);
                float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[8] : rk[10]), 4);
                pr[8] = (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_44 : rk[8]);
                pr[9] = (((lane_0_1 >> 2 & 1) != 0) ? rk[10] : _shfl_xor_44);
                int mkey_3 = keys[4];
                if (mkey_3 >= 0) {
                    long long mbase_4 = (mrow0 + (long long)rows_1[4]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[16 + 0], pa[16 + 1], pa[16 + 2], pa[16 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_4) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[16 + 0], pb[16 + 1], pb[16 + 2], pb[16 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_4 + 128) = _v4;
                    }
                    long long mrbase_4 = (mrow0 + (long long)rows_1[4]) * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos;
                    {
                        float2 _v2 = make_float2(pr[8 + 0], pr[8 + 1]);
                        *reinterpret_cast<float2*>(dkr_f32 + mrbase_4) = _v2;
                    }
                }
                float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[9] : a0[11]), 4);
                float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[25] : a0[27]), 4);
                float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_45 : a0[9]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_46 : a0[25])), 8);
                float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[11] : _shfl_xor_45) : (((lane_0_1 >> 2 & 1) != 0) ? a0[27] : _shfl_xor_46)), 8);
                float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[9] : a1[11]), 4);
                float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[25] : a1[27]), 4);
                float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_49 : a1[9]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_50 : a1[25])), 8);
                float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[11] : _shfl_xor_49) : (((lane_0_1 >> 2 & 1) != 0) ? a1[27] : _shfl_xor_50)), 8);
                pa[20] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_47 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_45 : a0[9]));
                pb[20] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_51 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_49 : a1[9]));
                pa[21] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_48 : (((lane_0_1 >> 2 & 1) != 0) ? a0[11] : _shfl_xor_45));
                pb[21] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_52 : (((lane_0_1 >> 2 & 1) != 0) ? a1[11] : _shfl_xor_49));
                pa[22] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_46 : a0[25]) : _shfl_xor_47);
                pb[22] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_50 : a1[25]) : _shfl_xor_51);
                pa[23] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[27] : _shfl_xor_46) : _shfl_xor_48);
                pb[23] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[27] : _shfl_xor_50) : _shfl_xor_52);
                float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[9] : rk[11]), 4);
                pr[10] = (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_53 : rk[9]);
                pr[11] = (((lane_0_1 >> 2 & 1) != 0) ? rk[11] : _shfl_xor_53);
                int mkey_4 = keys[5];
                if (mkey_4 >= 0) {
                    long long mbase_5 = (mrow0 + (long long)rows_1[5]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[20 + 0], pa[20 + 1], pa[20 + 2], pa[20 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_5) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[20 + 0], pb[20 + 1], pb[20 + 2], pb[20 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_5 + 128) = _v4;
                    }
                    long long mrbase_5 = (mrow0 + (long long)rows_1[5]) * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos;
                    {
                        float2 _v2 = make_float2(pr[10 + 0], pr[10 + 1]);
                        *reinterpret_cast<float2*>(dkr_f32 + mrbase_5) = _v2;
                    }
                }
                float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[12] : a0[14]), 4);
                float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[28] : a0[30]), 4);
                float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_54 : a0[12]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_55 : a0[28])), 8);
                float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[14] : _shfl_xor_54) : (((lane_0_1 >> 2 & 1) != 0) ? a0[30] : _shfl_xor_55)), 8);
                float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[12] : a1[14]), 4);
                float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[28] : a1[30]), 4);
                float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_58 : a1[12]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_59 : a1[28])), 8);
                float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[14] : _shfl_xor_58) : (((lane_0_1 >> 2 & 1) != 0) ? a1[30] : _shfl_xor_59)), 8);
                pa[24] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_56 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_54 : a0[12]));
                pb[24] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_60 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_58 : a1[12]));
                pa[25] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_57 : (((lane_0_1 >> 2 & 1) != 0) ? a0[14] : _shfl_xor_54));
                pb[25] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_61 : (((lane_0_1 >> 2 & 1) != 0) ? a1[14] : _shfl_xor_58));
                pa[26] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_55 : a0[28]) : _shfl_xor_56);
                pb[26] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_59 : a1[28]) : _shfl_xor_60);
                pa[27] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[30] : _shfl_xor_55) : _shfl_xor_57);
                pb[27] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[30] : _shfl_xor_59) : _shfl_xor_61);
                float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[12] : rk[14]), 4);
                pr[12] = (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_62 : rk[12]);
                pr[13] = (((lane_0_1 >> 2 & 1) != 0) ? rk[14] : _shfl_xor_62);
                int mkey_5 = keys[6];
                if (mkey_5 >= 0) {
                    long long mbase_6 = (mrow0 + (long long)rows_1[6]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[24 + 0], pa[24 + 1], pa[24 + 2], pa[24 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_6) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[24 + 0], pb[24 + 1], pb[24 + 2], pb[24 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_6 + 128) = _v4;
                    }
                    long long mrbase_6 = (mrow0 + (long long)rows_1[6]) * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos;
                    {
                        float2 _v2 = make_float2(pr[12 + 0], pr[12 + 1]);
                        *reinterpret_cast<float2*>(dkr_f32 + mrbase_6) = _v2;
                    }
                }
                float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[13] : a0[15]), 4);
                float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[29] : a0[31]), 4);
                float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_63 : a0[13]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_64 : a0[29])), 8);
                float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[15] : _shfl_xor_63) : (((lane_0_1 >> 2 & 1) != 0) ? a0[31] : _shfl_xor_64)), 8);
                float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[13] : a1[15]), 4);
                float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[29] : a1[31]), 4);
                float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_67 : a1[13]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_68 : a1[29])), 8);
                float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[15] : _shfl_xor_67) : (((lane_0_1 >> 2 & 1) != 0) ? a1[31] : _shfl_xor_68)), 8);
                pa[28] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_65 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_63 : a0[13]));
                pb[28] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_69 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_67 : a1[13]));
                pa[29] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_66 : (((lane_0_1 >> 2 & 1) != 0) ? a0[15] : _shfl_xor_63));
                pb[29] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_70 : (((lane_0_1 >> 2 & 1) != 0) ? a1[15] : _shfl_xor_67));
                pa[30] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_64 : a0[29]) : _shfl_xor_65);
                pb[30] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_68 : a1[29]) : _shfl_xor_69);
                pa[31] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[31] : _shfl_xor_64) : _shfl_xor_66);
                pb[31] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[31] : _shfl_xor_68) : _shfl_xor_70);
                float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? rk[13] : rk[15]), 4);
                pr[14] = (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_71 : rk[13]);
                pr[15] = (((lane_0_1 >> 2 & 1) != 0) ? rk[15] : _shfl_xor_71);
                int mkey_6 = keys[7];
                if (mkey_6 >= 0) {
                    long long mbase_7 = (mrow0 + (long long)rows_1[7]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[28 + 0], pa[28 + 1], pa[28 + 2], pa[28 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_7) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[28 + 0], pb[28 + 1], pb[28 + 2], pb[28 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase_7 + 128) = _v4;
                    }
                    long long mrbase_7 = (mrow0 + (long long)rows_1[7]) * (long long)dkr_stride + (long long)dkr_col0 + (long long)nrpos;
                    {
                        float2 _v2 = make_float2(pr[14 + 0], pr[14 + 1]);
                        *reinterpret_cast<float2*>(dkr_f32 + mrbase_7) = _v2;
                    }
                }
                mbarrier_wait(dkv_b_full_addr, par_1);
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a0[0])), "=r"(*reinterpret_cast<uint32_t*>(&a0[1])), "=r"(*reinterpret_cast<uint32_t*>(&a0[2])), "=r"(*reinterpret_cast<uint32_t*>(&a0[3])), "=r"(*reinterpret_cast<uint32_t*>(&a0[4])), "=r"(*reinterpret_cast<uint32_t*>(&a0[5])), "=r"(*reinterpret_cast<uint32_t*>(&a0[6])), "=r"(*reinterpret_cast<uint32_t*>(&a0[7])), "=r"(*reinterpret_cast<uint32_t*>(&a0[8])), "=r"(*reinterpret_cast<uint32_t*>(&a0[9])), "=r"(*reinterpret_cast<uint32_t*>(&a0[10])), "=r"(*reinterpret_cast<uint32_t*>(&a0[11])), "=r"(*reinterpret_cast<uint32_t*>(&a0[12])), "=r"(*reinterpret_cast<uint32_t*>(&a0[13])), "=r"(*reinterpret_cast<uint32_t*>(&a0[14])), "=r"(*reinterpret_cast<uint32_t*>(&a0[15]))
                    : "r"(tmem_tmem + 64 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((a0 + 16))[15]))
                    : "r"(tmem_tmem + 64 + wg * 32 + 1048576));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&a1[0])), "=r"(*reinterpret_cast<uint32_t*>(&a1[1])), "=r"(*reinterpret_cast<uint32_t*>(&a1[2])), "=r"(*reinterpret_cast<uint32_t*>(&a1[3])), "=r"(*reinterpret_cast<uint32_t*>(&a1[4])), "=r"(*reinterpret_cast<uint32_t*>(&a1[5])), "=r"(*reinterpret_cast<uint32_t*>(&a1[6])), "=r"(*reinterpret_cast<uint32_t*>(&a1[7])), "=r"(*reinterpret_cast<uint32_t*>(&a1[8])), "=r"(*reinterpret_cast<uint32_t*>(&a1[9])), "=r"(*reinterpret_cast<uint32_t*>(&a1[10])), "=r"(*reinterpret_cast<uint32_t*>(&a1[11])), "=r"(*reinterpret_cast<uint32_t*>(&a1[12])), "=r"(*reinterpret_cast<uint32_t*>(&a1[13])), "=r"(*reinterpret_cast<uint32_t*>(&a1[14])), "=r"(*reinterpret_cast<uint32_t*>(&a1[15]))
                    : "r"(tmem_tmem + 128 + wg * 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[0])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[1])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[2])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[3])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[4])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[5])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[6])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[7])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[8])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[9])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[10])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[11])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[12])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[13])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[14])), "=r"(*reinterpret_cast<uint32_t*>(&((a1 + 16))[15]))
                    : "r"(tmem_tmem + 128 + wg * 32 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(dkv_b_drained_addr);
                float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[0] : a0[2]), 4);
                float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[16] : a0[18]), 4);
                float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_72 : a0[0]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_73 : a0[16])), 8);
                float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[2] : _shfl_xor_72) : (((lane_0_1 >> 2 & 1) != 0) ? a0[18] : _shfl_xor_73)), 8);
                float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[0] : a1[2]), 4);
                float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[16] : a1[18]), 4);
                float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_76 : a1[0]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_77 : a1[16])), 8);
                float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[2] : _shfl_xor_76) : (((lane_0_1 >> 2 & 1) != 0) ? a1[18] : _shfl_xor_77)), 8);
                pa[0] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_74 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_72 : a0[0]));
                pb[0] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_78 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_76 : a1[0]));
                pa[1] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_75 : (((lane_0_1 >> 2 & 1) != 0) ? a0[2] : _shfl_xor_72));
                pb[1] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_79 : (((lane_0_1 >> 2 & 1) != 0) ? a1[2] : _shfl_xor_76));
                pa[2] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_73 : a0[16]) : _shfl_xor_74);
                pb[2] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_77 : a1[16]) : _shfl_xor_78);
                pa[3] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[18] : _shfl_xor_73) : _shfl_xor_75);
                pb[3] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[18] : _shfl_xor_77) : _shfl_xor_79);
                int mkey2 = keys[0];
                if (mkey2 >= 0) {
                    long long mbase2 = (mrow0 + (long long)rows_1[0]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[0 + 0], pa[0 + 1], pa[0 + 2], pa[0 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2 + 256) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[0 + 0], pb[0 + 1], pb[0 + 2], pb[0 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2 + 384) = _v4;
                    }
                }
                float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[1] : a0[3]), 4);
                float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[17] : a0[19]), 4);
                float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_80 : a0[1]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_81 : a0[17])), 8);
                float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[3] : _shfl_xor_80) : (((lane_0_1 >> 2 & 1) != 0) ? a0[19] : _shfl_xor_81)), 8);
                float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[1] : a1[3]), 4);
                float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[17] : a1[19]), 4);
                float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_84 : a1[1]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_85 : a1[17])), 8);
                float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[3] : _shfl_xor_84) : (((lane_0_1 >> 2 & 1) != 0) ? a1[19] : _shfl_xor_85)), 8);
                pa[4] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_82 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_80 : a0[1]));
                pb[4] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_86 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_84 : a1[1]));
                pa[5] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_83 : (((lane_0_1 >> 2 & 1) != 0) ? a0[3] : _shfl_xor_80));
                pb[5] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_87 : (((lane_0_1 >> 2 & 1) != 0) ? a1[3] : _shfl_xor_84));
                pa[6] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_81 : a0[17]) : _shfl_xor_82);
                pb[6] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_85 : a1[17]) : _shfl_xor_86);
                pa[7] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[19] : _shfl_xor_81) : _shfl_xor_83);
                pb[7] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[19] : _shfl_xor_85) : _shfl_xor_87);
                int mkey2_7 = keys[1];
                if (mkey2_7 >= 0) {
                    long long mbase2_1 = (mrow0 + (long long)rows_1[1]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[4 + 0], pa[4 + 1], pa[4 + 2], pa[4 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_1 + 256) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[4 + 0], pb[4 + 1], pb[4 + 2], pb[4 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_1 + 384) = _v4;
                    }
                }
                float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[4] : a0[6]), 4);
                float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[20] : a0[22]), 4);
                float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_88 : a0[4]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_89 : a0[20])), 8);
                float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[6] : _shfl_xor_88) : (((lane_0_1 >> 2 & 1) != 0) ? a0[22] : _shfl_xor_89)), 8);
                float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[4] : a1[6]), 4);
                float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[20] : a1[22]), 4);
                float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_92 : a1[4]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_93 : a1[20])), 8);
                float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[6] : _shfl_xor_92) : (((lane_0_1 >> 2 & 1) != 0) ? a1[22] : _shfl_xor_93)), 8);
                pa[8] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_90 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_88 : a0[4]));
                pb[8] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_94 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_92 : a1[4]));
                pa[9] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_91 : (((lane_0_1 >> 2 & 1) != 0) ? a0[6] : _shfl_xor_88));
                pb[9] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_95 : (((lane_0_1 >> 2 & 1) != 0) ? a1[6] : _shfl_xor_92));
                pa[10] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_89 : a0[20]) : _shfl_xor_90);
                pb[10] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_93 : a1[20]) : _shfl_xor_94);
                pa[11] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[22] : _shfl_xor_89) : _shfl_xor_91);
                pb[11] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[22] : _shfl_xor_93) : _shfl_xor_95);
                int mkey2_8 = keys[2];
                if (mkey2_8 >= 0) {
                    long long mbase2_2 = (mrow0 + (long long)rows_1[2]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[8 + 0], pa[8 + 1], pa[8 + 2], pa[8 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_2 + 256) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[8 + 0], pb[8 + 1], pb[8 + 2], pb[8 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_2 + 384) = _v4;
                    }
                }
                float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[5] : a0[7]), 4);
                float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[21] : a0[23]), 4);
                float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_96 : a0[5]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_97 : a0[21])), 8);
                float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[7] : _shfl_xor_96) : (((lane_0_1 >> 2 & 1) != 0) ? a0[23] : _shfl_xor_97)), 8);
                float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[5] : a1[7]), 4);
                float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[21] : a1[23]), 4);
                float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_100 : a1[5]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_101 : a1[21])), 8);
                float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[7] : _shfl_xor_100) : (((lane_0_1 >> 2 & 1) != 0) ? a1[23] : _shfl_xor_101)), 8);
                pa[12] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_98 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_96 : a0[5]));
                pb[12] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_102 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_100 : a1[5]));
                pa[13] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_99 : (((lane_0_1 >> 2 & 1) != 0) ? a0[7] : _shfl_xor_96));
                pb[13] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_103 : (((lane_0_1 >> 2 & 1) != 0) ? a1[7] : _shfl_xor_100));
                pa[14] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_97 : a0[21]) : _shfl_xor_98);
                pb[14] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_101 : a1[21]) : _shfl_xor_102);
                pa[15] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[23] : _shfl_xor_97) : _shfl_xor_99);
                pb[15] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[23] : _shfl_xor_101) : _shfl_xor_103);
                int mkey2_9 = keys[3];
                if (mkey2_9 >= 0) {
                    long long mbase2_3 = (mrow0 + (long long)rows_1[3]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[12 + 0], pa[12 + 1], pa[12 + 2], pa[12 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_3 + 256) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[12 + 0], pb[12 + 1], pb[12 + 2], pb[12 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_3 + 384) = _v4;
                    }
                }
                float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[8] : a0[10]), 4);
                float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[24] : a0[26]), 4);
                float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_104 : a0[8]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_105 : a0[24])), 8);
                float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[10] : _shfl_xor_104) : (((lane_0_1 >> 2 & 1) != 0) ? a0[26] : _shfl_xor_105)), 8);
                float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[8] : a1[10]), 4);
                float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[24] : a1[26]), 4);
                float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_108 : a1[8]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_109 : a1[24])), 8);
                float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[10] : _shfl_xor_108) : (((lane_0_1 >> 2 & 1) != 0) ? a1[26] : _shfl_xor_109)), 8);
                pa[16] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_106 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_104 : a0[8]));
                pb[16] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_110 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_108 : a1[8]));
                pa[17] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_107 : (((lane_0_1 >> 2 & 1) != 0) ? a0[10] : _shfl_xor_104));
                pb[17] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_111 : (((lane_0_1 >> 2 & 1) != 0) ? a1[10] : _shfl_xor_108));
                pa[18] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_105 : a0[24]) : _shfl_xor_106);
                pb[18] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_109 : a1[24]) : _shfl_xor_110);
                pa[19] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[26] : _shfl_xor_105) : _shfl_xor_107);
                pb[19] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[26] : _shfl_xor_109) : _shfl_xor_111);
                int mkey2_10 = keys[4];
                if (mkey2_10 >= 0) {
                    long long mbase2_4 = (mrow0 + (long long)rows_1[4]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[16 + 0], pa[16 + 1], pa[16 + 2], pa[16 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_4 + 256) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[16 + 0], pb[16 + 1], pb[16 + 2], pb[16 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_4 + 384) = _v4;
                    }
                }
                float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[9] : a0[11]), 4);
                float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[25] : a0[27]), 4);
                float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_112 : a0[9]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_113 : a0[25])), 8);
                float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[11] : _shfl_xor_112) : (((lane_0_1 >> 2 & 1) != 0) ? a0[27] : _shfl_xor_113)), 8);
                float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[9] : a1[11]), 4);
                float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[25] : a1[27]), 4);
                float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_116 : a1[9]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_117 : a1[25])), 8);
                float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[11] : _shfl_xor_116) : (((lane_0_1 >> 2 & 1) != 0) ? a1[27] : _shfl_xor_117)), 8);
                pa[20] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_114 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_112 : a0[9]));
                pb[20] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_118 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_116 : a1[9]));
                pa[21] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_115 : (((lane_0_1 >> 2 & 1) != 0) ? a0[11] : _shfl_xor_112));
                pb[21] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_119 : (((lane_0_1 >> 2 & 1) != 0) ? a1[11] : _shfl_xor_116));
                pa[22] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_113 : a0[25]) : _shfl_xor_114);
                pb[22] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_117 : a1[25]) : _shfl_xor_118);
                pa[23] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[27] : _shfl_xor_113) : _shfl_xor_115);
                pb[23] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[27] : _shfl_xor_117) : _shfl_xor_119);
                int mkey2_11 = keys[5];
                if (mkey2_11 >= 0) {
                    long long mbase2_5 = (mrow0 + (long long)rows_1[5]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[20 + 0], pa[20 + 1], pa[20 + 2], pa[20 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_5 + 256) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[20 + 0], pb[20 + 1], pb[20 + 2], pb[20 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_5 + 384) = _v4;
                    }
                }
                float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[12] : a0[14]), 4);
                float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[28] : a0[30]), 4);
                float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_120 : a0[12]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_121 : a0[28])), 8);
                float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[14] : _shfl_xor_120) : (((lane_0_1 >> 2 & 1) != 0) ? a0[30] : _shfl_xor_121)), 8);
                float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[12] : a1[14]), 4);
                float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[28] : a1[30]), 4);
                float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_124 : a1[12]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_125 : a1[28])), 8);
                float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[14] : _shfl_xor_124) : (((lane_0_1 >> 2 & 1) != 0) ? a1[30] : _shfl_xor_125)), 8);
                pa[24] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_122 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_120 : a0[12]));
                pb[24] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_126 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_124 : a1[12]));
                pa[25] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_123 : (((lane_0_1 >> 2 & 1) != 0) ? a0[14] : _shfl_xor_120));
                pb[25] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_127 : (((lane_0_1 >> 2 & 1) != 0) ? a1[14] : _shfl_xor_124));
                pa[26] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_121 : a0[28]) : _shfl_xor_122);
                pb[26] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_125 : a1[28]) : _shfl_xor_126);
                pa[27] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[30] : _shfl_xor_121) : _shfl_xor_123);
                pb[27] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[30] : _shfl_xor_125) : _shfl_xor_127);
                int mkey2_12 = keys[6];
                if (mkey2_12 >= 0) {
                    long long mbase2_6 = (mrow0 + (long long)rows_1[6]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[24 + 0], pa[24 + 1], pa[24 + 2], pa[24 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_6 + 256) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[24 + 0], pb[24 + 1], pb[24 + 2], pb[24 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_6 + 384) = _v4;
                    }
                }
                float _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[13] : a0[15]), 4);
                float _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a0[29] : a0[31]), 4);
                float _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_128 : a0[13]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_129 : a0[29])), 8);
                float _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[15] : _shfl_xor_128) : (((lane_0_1 >> 2 & 1) != 0) ? a0[31] : _shfl_xor_129)), 8);
                float _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[13] : a1[15]), 4);
                float _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 2 & 1) != 0) ? a1[29] : a1[31]), 4);
                float _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_132 : a1[13]) : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_133 : a1[29])), 8);
                float _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[15] : _shfl_xor_132) : (((lane_0_1 >> 2 & 1) != 0) ? a1[31] : _shfl_xor_133)), 8);
                pa[28] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_130 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_128 : a0[13]));
                pb[28] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_134 : (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_132 : a1[13]));
                pa[29] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_131 : (((lane_0_1 >> 2 & 1) != 0) ? a0[15] : _shfl_xor_128));
                pb[29] = (((lane_0_1 >> 3 & 1) != 0) ? _shfl_xor_135 : (((lane_0_1 >> 2 & 1) != 0) ? a1[15] : _shfl_xor_132));
                pa[30] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_129 : a0[29]) : _shfl_xor_130);
                pb[30] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? _shfl_xor_133 : a1[29]) : _shfl_xor_134);
                pa[31] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a0[31] : _shfl_xor_129) : _shfl_xor_131);
                pb[31] = (((lane_0_1 >> 3 & 1) != 0) ? (((lane_0_1 >> 2 & 1) != 0) ? a1[31] : _shfl_xor_133) : _shfl_xor_135);
                int mkey2_13 = keys[7];
                if (mkey2_13 >= 0) {
                    long long mbase2_7 = (mrow0 + (long long)rows_1[7]) * (long long)dkv_stride + (long long)npos;
                    {
                        float4 _v4 = make_float4(pa[28 + 0], pa[28 + 1], pa[28 + 2], pa[28 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_7 + 256) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(pb[28 + 0], pb[28 + 1], pb[28 + 2], pb[28 + 3]);
                        *reinterpret_cast<float4*>(dkv_f32 + mbase2_7 + 384) = _v4;
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 16) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // mma_main
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_3 = blocks_word[0];
            mbarrier_wait(qdo_full_addr, 0);
            mbarrier_wait(pads_ready_addr, 0);
            if (elect_sync()) {
                #pragma unroll 1
                for (int i_3 = 0; i_3 < blocks_3; i_3++) {
                    unsigned int par_2 = i_3 & 1;
                    int ks_1 = i_3 % 2;
                    if (i_3 == 0) {
                        mbarrier_wait(k_full_addr, 0);
                        int _mma_a_lo_0 = (((k_k_addr) >> 4) & 0x3FFF) + (0) * 5120;
                        int _mma_b_lo_0 = ((q_k_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 67372176;\n\t"
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
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_tmem), "r"(0));
                        tcgen05_commit(s_full_addr);
                    }
                    mbarrier_wait(dp_free_addr, par_2 ^ 1);
                    int _mma_a_lo_1 = (((v_k_addr) >> 4) & 0x3FFF) + (ks_1) * 5120;
                    int _mma_b_lo_1 = ((do_k_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 67372176;\n\t"
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
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem + (1048576))), "r"(0));
                    tcgen05_commit(dp_full_addr);
                    mbarrier_wait(p_full_addr, par_2);
                    if (i_3 > 0) {
                        mbarrier_wait(dkv_b_drained_addr, par_2 ^ 1);
                    }
                    int _mma_a_lo_2 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x800000) + (0) * 256;
                    int _mma_b_lo_2 = (((p_k_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem + (64))), "r"(0));
                    int _mma_a_lo_3 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x800000) + (1) * 256;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_2), "r"((tmem_tmem + (128))), "r"(0));
                    mbarrier_wait(ds_full_addr, par_2);
                    int _mma_a_lo_4 = ((((k_mn0_addr) >> 4) & 0x3FFF) | 0x2000000) + (ks_1) * 5120;
                    int _mma_b_lo_4 = (((ds_hk_addr) >> 4) & 0x3FFF) | 0x400000;
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
                    "mov.b32 id, 134382736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_tmem + (32))), "r"(((i_3 == 0) ? 0 : 1)));
                    int _mma_a_lo_5 = ((((k_mn1_addr) >> 4) & 0x3FFF) | 0x2000000) + (ks_1) * 5120;
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
                    "mov.b32 id, 134382736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_4), "r"((tmem_tmem + (40))), "r"(((i_3 == 0) ? 0 : 1)));
                    int _mma_a_lo_6 = ((((k_mn2_addr) >> 4) & 0x3FFF) | 0x2000000) + (ks_1) * 5120;
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
                    "mov.b32 id, 134382736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_4), "r"((tmem_tmem + (48))), "r"(((i_3 == 0) ? 0 : 1)));
                    int _mma_a_lo_7 = ((((k_mn3_addr) >> 4) & 0x3FFF) | 0x2000000) + (ks_1) * 5120;
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
                    "mov.b32 id, 134382736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_4), "r"((tmem_tmem + (56))), "r"(((i_3 == 0) ? 0 : 1)));
                    int _mma_a_lo_8 = ((((kr_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (ks_1) * 5120;
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
                    "mov.b32 id, 67273872;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_4), "r"((tmem_tmem + (16))), "r"(((i_3 == 0) ? 0 : 1)));
                    tcgen05_commit(k_free_addr + (ks_1) * 8);
                    int _mma_a_lo_9 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x800000) + (0) * 256;
                    int _mma_b_lo_9 = (((ds_k_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"((tmem_tmem + (64))), "r"(1));
                    int _mma_a_lo_10 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x800000) + (1) * 256;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_9), "r"((tmem_tmem + (128))), "r"(1));
                    tcgen05_commit(dkv_a_full_addr);
                    mbarrier_wait(dkr_drained_addr, par_2 ^ 1);
                    int _mma_a_lo_11 = (((qr_mn_addr) >> 4) & 0x3FFF) | 0x800000;
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
                    "mov.b32 id, 68191376;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_9), "r"((tmem_tmem + (192))), "r"(0));
                    tcgen05_commit(dkr_full_addr);
                    mbarrier_wait(dkv_a_drained_addr, par_2);
                    int _mma_a_lo_12 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x800000) + (2) * 256;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_12), "r"(_mma_b_lo_2), "r"((tmem_tmem + (64))), "r"(0));
                    int _mma_a_lo_13 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x800000) + (3) * 256;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_13), "r"(_mma_b_lo_2), "r"((tmem_tmem + (128))), "r"(0));
                    tcgen05_commit(p_free_addr);
                    int _mma_a_lo_14 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x800000) + (2) * 256;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_14), "r"(_mma_b_lo_9), "r"((tmem_tmem + (64))), "r"(1));
                    int _mma_a_lo_15 = ((((q_mn_addr) >> 4) & 0x3FFF) | 0x800000) + (3) * 256;
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
                    "mov.b32 id, 135300240;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_15), "r"(_mma_b_lo_9), "r"((tmem_tmem + (128))), "r"(1));
                    tcgen05_commit(dkv_b_full_addr);
                    tcgen05_commit(ds_free_addr);
                    if (blocks_3 > i_3 + 1) {
                        int nks = (i_3 + 1) % 2;
                        mbarrier_wait(k_full_addr + (nks) * 8, (i_3 + 1) / 2 & 1);
                        mbarrier_wait(s_free_addr, par_2);
                        int _mma_a_lo_16 = (((k_k_addr) >> 4) & 0x3FFF) + (nks) * 5120;
                        int _mma_b_lo_16 = ((q_k_addr) >> 4) & 0x3FFF;
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
                    "mov.b32 id, 67372176;\n\t"
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
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 122;\n\t"
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
                    :: "r"(_mma_a_lo_16), "r"(_mma_b_lo_16), "r"(tmem_tmem), "r"(0));
                        tcgen05_commit(s_full_addr);
                    }
                }
                tcgen05_commit(dq_done_addr);
            }
        }
    }
    // ---- Role: load ----
    if (warp == 17) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // load_main
            int token_1 = token_base + token_step * blockIdx.x;
            int lane_0_2 = lane;
            if (elect_sync()) {
                tma_4d_gmem2smem(q_k_addr, (&q_latent), 0, 0, 0, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr, (&dout), 0, 0, 0, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 2048, (&q_latent), 0, 0, 1, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 2048, (&dout), 0, 0, 1, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 4096, (&q_latent), 0, 0, 2, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 4096, (&dout), 0, 0, 2, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 6144, (&q_latent), 0, 0, 3, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 6144, (&dout), 0, 0, 3, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 8192, (&q_latent), 0, 0, 4, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 8192, (&dout), 0, 0, 4, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 10240, (&q_latent), 0, 0, 5, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 10240, (&dout), 0, 0, 5, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 12288, (&q_latent), 0, 0, 6, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 12288, (&dout), 0, 0, 6, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 14336, (&q_latent), 0, 0, 7, token_1, qdo_full_addr);
                tma_4d_gmem2smem(do_k_addr + 14336, (&dout), 0, 0, 7, token_1, qdo_full_addr);
                tma_4d_gmem2smem(q_k_addr + 16384, (&q_rope), 0, 0, 0, token_1, qdo_full_addr);
                mbarrier_arrive_expect_tx(qdo_full_addr, 17408);
            }
            if (lane_0_2 < 8) {
                stats[lane_0_2] = lse[token_1 * 8 + lane_0_2];
                stats[32 + lane_0_2] = delta[token_1 * 8 + lane_0_2];
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(stats_full_addr);
        }
    }
    // ---- Role: metadata ----
    if (warp == 18) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // metadata_main
            int token_2 = token_base + token_step * blockIdx.x;
            int lane_0_3 = lane;
            int active = topk;
            if (has_topk_length != 0) {
                int _max_0 = ((topk_length[token_2]) > (0) ? (topk_length[token_2]) : (0));
                int _min_0 = ((_max_0) < (topk) ? (_max_0) : (topk));
                active = _min_0;
            }
            long long row_base = (long long)indices_offset + (long long)token_2 * (long long)idx_stride;
            int aligned4 = (int)(((idx_stride | indices_offset) & 3) == 0);
            int last_valid = -1;
            for (int blk = (active + 127) / 128 - 1; blk >= 0; blk--) {
                int pos = blk * 128 + lane_0_3 * 4;
                int best = -1;
                if (pos < active) {
                    if (aligned4 != 0) {
                        int _vec_load_0[4];
                        {
                            const int4* _ivptr_0 = reinterpret_cast<const int4*>(indices + (row_base + (long long)pos) + 0);
                            int4 _ivld_0;
                            _ivld_0 = *_ivptr_0;
                            _vec_load_0[0 + 0] = _ivld_0.x;
                            _vec_load_0[0 + 1] = _ivld_0.y;
                            _vec_load_0[0 + 2] = _ivld_0.z;
                            _vec_load_0[0 + 3] = _ivld_0.w;
                        }
                        if (_vec_load_0[0] >= 0 && _vec_load_0[0] < num_kv && active > pos) {
                            best = pos;
                        }
                        if (_vec_load_0[1] >= 0 && _vec_load_0[1] < num_kv && active > pos + 1) {
                            best = pos + 1;
                        }
                        if (_vec_load_0[2] >= 0 && _vec_load_0[2] < num_kv && active > pos + 2) {
                            best = pos + 2;
                        }
                        if (_vec_load_0[3] >= 0 && _vec_load_0[3] < num_kv && active > pos + 3) {
                            best = pos + 3;
                        }
                    } else {
                        int sv = indices[row_base + (long long)pos];
                        if (sv >= 0 && sv < num_kv && active > pos) {
                            best = pos;
                        }
                        int sv_0 = indices[row_base + (long long)(pos + 1)];
                        if (sv_0 >= 0 && sv_0 < num_kv && active > pos + 1) {
                            best = pos + 1;
                        }
                        int sv_1 = indices[row_base + (long long)(pos + 2)];
                        if (sv_1 >= 0 && sv_1 < num_kv && active > pos + 2) {
                            best = pos + 2;
                        }
                        int sv_2 = indices[row_base + (long long)(pos + 3)];
                        if (sv_2 >= 0 && sv_2 < num_kv && active > pos + 3) {
                            best = pos + 3;
                        }
                    }
                }
                int _warp_redux_i32_0;
                asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_i32_0) : "r"(best));
                best = _warp_redux_i32_0;
                if (best >= 0) {
                    last_valid = best;
                    break;
                }
            }
            int _max_1 = (((last_valid + 64) / 64) > (1) ? ((last_valid + 64) / 64) : (1));
            int blocks_4 = _max_1;
            if (lane_0_3 == 0) {
                blocks_word[0] = blocks_4;
                mbarrier_arrive(blocks_ready_addr);
            }
            if (lane_0_3 < 8) {
                #pragma unroll 1
                for (int i_4 = 0; i_4 < blocks_4; i_4++) {
                    int n = blocks_4 - 1 - i_4;
                    int slot_3 = i_4 % 2;
                    unsigned int mask = 0;
                    int position = n * 64 + lane_0_3 * 8;
                    int values[8];
                    int whole_tile = (int)(active >= n * 64 + 64 && ((idx_stride | indices_offset) & 7) == 0);
                    if (whole_tile != 0) {
                        int _vec_load_1[8];
                        {
                            uint32_t _iv_1_0;
                            uint32_t _iv_1_1;
                            uint32_t _iv_1_2;
                            uint32_t _iv_1_3;
                            uint32_t _iv_1_4;
                            uint32_t _iv_1_5;
                            uint32_t _iv_1_6;
                            uint32_t _iv_1_7;
                            asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(indices + (row_base + (long long)position) + (0))) : "memory");
                            _vec_load_1[0 + 0] = (int32_t)_iv_1_0;
                            _vec_load_1[0 + 1] = (int32_t)_iv_1_1;
                            _vec_load_1[0 + 2] = (int32_t)_iv_1_2;
                            _vec_load_1[0 + 3] = (int32_t)_iv_1_3;
                            _vec_load_1[0 + 4] = (int32_t)_iv_1_4;
                            _vec_load_1[0 + 5] = (int32_t)_iv_1_5;
                            _vec_load_1[0 + 6] = (int32_t)_iv_1_6;
                            _vec_load_1[0 + 7] = (int32_t)_iv_1_7;
                        }
                        values[0] = _vec_load_1[0];
                        values[1] = _vec_load_1[1];
                        values[2] = _vec_load_1[2];
                        values[3] = _vec_load_1[3];
                        values[4] = _vec_load_1[4];
                        values[5] = _vec_load_1[5];
                        values[6] = _vec_load_1[6];
                        values[7] = _vec_load_1[7];
                    } else {
                        int _min_1 = ((position) < (topk - 1) ? (position) : (topk - 1));
                        int clamped = _min_1;
                        int value = indices[row_base + (long long)clamped];
                        if (active <= position) {
                            value = -1;
                        }
                        values[0] = value;
                        int _min_2 = ((position + 1) < (topk - 1) ? (position + 1) : (topk - 1));
                        int clamped_0 = _min_2;
                        int value_1 = indices[row_base + (long long)clamped_0];
                        if (active <= position + 1) {
                            value_1 = -1;
                        }
                        values[1] = value_1;
                        int _min_3 = ((position + 2) < (topk - 1) ? (position + 2) : (topk - 1));
                        int clamped_2 = _min_3;
                        int value_3 = indices[row_base + (long long)clamped_2];
                        if (active <= position + 2) {
                            value_3 = -1;
                        }
                        values[2] = value_3;
                        int _min_4 = ((position + 3) < (topk - 1) ? (position + 3) : (topk - 1));
                        int clamped_4 = _min_4;
                        int value_5 = indices[row_base + (long long)clamped_4];
                        if (active <= position + 3) {
                            value_5 = -1;
                        }
                        values[3] = value_5;
                        int _min_5 = ((position + 4) < (topk - 1) ? (position + 4) : (topk - 1));
                        int clamped_6 = _min_5;
                        int value_7 = indices[row_base + (long long)clamped_6];
                        if (active <= position + 4) {
                            value_7 = -1;
                        }
                        values[4] = value_7;
                        int _min_6 = ((position + 5) < (topk - 1) ? (position + 5) : (topk - 1));
                        int clamped_8 = _min_6;
                        int value_9 = indices[row_base + (long long)clamped_8];
                        if (active <= position + 5) {
                            value_9 = -1;
                        }
                        values[5] = value_9;
                        int _min_7 = ((position + 6) < (topk - 1) ? (position + 6) : (topk - 1));
                        int clamped_10 = _min_7;
                        int value_11 = indices[row_base + (long long)clamped_10];
                        if (active <= position + 6) {
                            value_11 = -1;
                        }
                        values[6] = value_11;
                        int _min_8 = ((position + 7) < (topk - 1) ? (position + 7) : (topk - 1));
                        int clamped_12 = _min_8;
                        int value_13 = indices[row_base + (long long)clamped_12];
                        if (active <= position + 7) {
                            value_13 = -1;
                        }
                        values[7] = value_13;
                    }
                    mbarrier_wait(idx_free_addr + (slot_3) * 8, i_4 / 2 & 1 ^ 1);
                    int ok = (int)(values[0] >= 0 && values[0] < num_kv);
                    if (ok != 0) {
                        mask = mask | 1;
                    } else {
                        values[0] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8] = values[0];
                    int ok_0 = (int)(values[1] >= 0 && values[1] < num_kv);
                    if (ok_0 != 0) {
                        mask = mask | 2;
                    } else {
                        values[1] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 1] = values[1];
                    int ok_1 = (int)(values[2] >= 0 && values[2] < num_kv);
                    if (ok_1 != 0) {
                        mask = mask | 4;
                    } else {
                        values[2] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 2] = values[2];
                    int ok_2 = (int)(values[3] >= 0 && values[3] < num_kv);
                    if (ok_2 != 0) {
                        mask = mask | 8;
                    } else {
                        values[3] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 3] = values[3];
                    int ok_3 = (int)(values[4] >= 0 && values[4] < num_kv);
                    if (ok_3 != 0) {
                        mask = mask | 16;
                    } else {
                        values[4] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 4] = values[4];
                    int ok_4 = (int)(values[5] >= 0 && values[5] < num_kv);
                    if (ok_4 != 0) {
                        mask = mask | 32;
                    } else {
                        values[5] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 5] = values[5];
                    int ok_5 = (int)(values[6] >= 0 && values[6] < num_kv);
                    if (ok_5 != 0) {
                        mask = mask | 64;
                    } else {
                        values[6] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 6] = values[6];
                    int ok_6 = (int)(values[7] >= 0 && values[7] < num_kv);
                    if (ok_6 != 0) {
                        mask = mask | 128;
                    } else {
                        values[7] = -1;
                    }
                    tile_idx[slot_3 * 80 + lane_0_3 * 8 + 7] = values[7];
                    validity8[slot_3 * 320 + 256 + lane_0_3] = mask;
                    mbarrier_arrive(idx_full_addr + (slot_3) * 8);
                }
            }
        }
    }
    // ---- Role: spare ----
    if (warp == 19) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        // idle — no tasks assigned
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 4) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
