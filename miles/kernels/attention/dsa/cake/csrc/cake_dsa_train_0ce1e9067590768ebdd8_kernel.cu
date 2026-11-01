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
#define TMEM_TMEM_OFFSET 0
#define NUM_MAIN_STAGES 1
#define SMEM_Q_K_OFF 1024
#define SMEM_Q_K_STAGE_BYTES 16384
#define SMEM_Q_K_STRIDE 16384
#define SMEM_Q_MN_OFF 1024
#define SMEM_Q_MN_STAGE_BYTES 16384
#define SMEM_Q_MN_STRIDE 16384
#define SMEM_Q_RAW_OFF 1024
#define SMEM_Q_RAW_STAGE_BYTES 16384
#define SMEM_Q_RAW_STRIDE 16384
#define SMEM_K_K_OFF 17408
#define SMEM_K_K_STAGE_BYTES 32768
#define SMEM_K_K_STRIDE 32768
#define SMEM_K_MN_OFF 17408
#define SMEM_K_MN_STAGE_BYTES 32768
#define SMEM_K_MN_STRIDE 32768
#define SMEM_K_BLK_OFF 17408
#define SMEM_K_BLK_STAGE_BYTES 16384
#define SMEM_K_BLK_STRIDE 16384
#define SMEM_GT_K_OFF 115712
#define SMEM_GT_K_STAGE_BYTES 16384
#define SMEM_GT_K_STRIDE 16384
#define SMEM_GT_MN_OFF 115712
#define SMEM_GT_MN_STAGE_BYTES 16384
#define SMEM_GT_MN_STRIDE 16384
#define SMEM_GT_RAW_OFF 115712
#define SMEM_GT_RAW_STAGE_BYTES 65536
#define SMEM_GT_RAW_STRIDE 65536
#define SMEM_S_IDX_OFF 181248
#define SMEM_S_IDX_STAGE_BYTES 3072
#define SMEM_S_IDX_STRIDE 3072
#define SMEM_S_GRAD_OFF 184320
#define SMEM_S_GRAD_STAGE_BYTES 3072
#define SMEM_S_GRAD_STRIDE 3072
#define SMEM_S_DW_OFF 187392
#define SMEM_S_DW_STAGE_BYTES 1024
#define SMEM_S_DW_STRIDE 1024
#define SMEM_BLOCKS_WORD_OFF 188416
#define SMEM_BLOCKS_WORD_STAGE_BYTES 4
#define SMEM_BLOCKS_WORD_STRIDE 4
#define SMEM_TOTAL 188544
#define THREADS 256
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



__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}






__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_train_0ce1e9067590768ebdd8(__nv_bfloat16* __restrict__ index_q, const __grid_constant__ CUtensorMap index_k, float* __restrict__ weights, int* __restrict__ topk_indices, float* __restrict__ grad_scores, __nv_bfloat16* __restrict__ d_index_q, float* __restrict__ d_weights, float* __restrict__ partial, int heads, int topk)
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
    #define blocks_ready_addr (mbar_base + 0)
    #define q_full_addr (mbar_base + 8)
    #define idx_full_addr (mbar_base + 16)
    #define idx_free_addr (mbar_base + 64)
    #define k_full_addr (mbar_base + 112)
    #define k_free_addr (mbar_base + 136)
    #define l_full_addr (mbar_base + 160)
    #define l_free_addr (mbar_base + 176)
    #define g_full_addr (mbar_base + 192)
    #define g_free_addr (mbar_base + 208)
    #define dk_full_addr (mbar_base + 224)
    #define dk_free_addr (mbar_base + 240)
    #define dq_done_addr (mbar_base + 256)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* q_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_Q_K_OFF);
    const int q_k_addr = smem + SMEM_Q_K_OFF;
    __nv_bfloat16* q_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_Q_MN_OFF);
    const int q_mn_addr = smem + SMEM_Q_MN_OFF;
    __nv_bfloat16* q_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_Q_RAW_OFF);
    const int q_raw_addr = smem + SMEM_Q_RAW_OFF;
    __nv_bfloat16* k_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_K_K_OFF);
    const int k_k_addr = smem + SMEM_K_K_OFF;
    __nv_bfloat16* k_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_K_MN_OFF);
    const int k_mn_addr = smem + SMEM_K_MN_OFF;
    __nv_bfloat16* k_blk = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_K_BLK_OFF);
    const int k_blk_addr = smem + SMEM_K_BLK_OFF;
    __nv_bfloat16* gt_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_GT_K_OFF);
    const int gt_k_addr = smem + SMEM_GT_K_OFF;
    __nv_bfloat16* gt_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_GT_MN_OFF);
    const int gt_mn_addr = smem + SMEM_GT_MN_OFF;
    __nv_bfloat16* gt_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_GT_RAW_OFF);
    const int gt_raw_addr = smem + SMEM_GT_RAW_OFF;
    int* s_idx = reinterpret_cast<int*>(smem_raw + SMEM_S_IDX_OFF);
    const int s_idx_addr = smem + SMEM_S_IDX_OFF;
    float* s_grad = reinterpret_cast<float*>(smem_raw + SMEM_S_GRAD_OFF);
    const int s_grad_addr = smem + SMEM_S_GRAD_OFF;
    float* s_dw = reinterpret_cast<float*>(smem_raw + SMEM_S_DW_OFF);
    const int s_dw_addr = smem + SMEM_S_DW_OFF;
    int* blocks_word = reinterpret_cast<int*>(smem_raw + SMEM_BLOCKS_WORD_OFF);
    const int blocks_word_addr = smem + SMEM_BLOCKS_WORD_OFF;
    if (warp == 4 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&index_k))) : "memory"); }

    // Mbarrier init (13 pipeline groups, 0 ordered-sequence groups, 33 barriers)
    // Mbarriers at smem_raw[0..264)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // blocks_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_full: 1 barriers, init_count=128
            mbarrier_init(smem + 8, 128);
            // idx_full: 6 barriers, init_count=32
            mbarrier_init(smem + 16, 32);
            mbarrier_init(smem + 24, 32);
            mbarrier_init(smem + 32, 32);
            mbarrier_init(smem + 40, 32);
            mbarrier_init(smem + 48, 32);
            mbarrier_init(smem + 56, 32);
            // idx_free: 6 barriers, init_count=130
            mbarrier_init(smem + 64, 130);
            mbarrier_init(smem + 72, 130);
            mbarrier_init(smem + 80, 130);
            mbarrier_init(smem + 88, 130);
            mbarrier_init(smem + 96, 130);
            mbarrier_init(smem + 104, 130);
            // k_full: 3 barriers, init_count=2
            mbarrier_init(smem + 112, 2);
            mbarrier_init(smem + 120, 2);
            mbarrier_init(smem + 128, 2);
            // k_free: 3 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // l_full: 2 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // l_free: 2 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            // g_full: 2 barriers, init_count=128
            mbarrier_init(smem + 192, 128);
            mbarrier_init(smem + 200, 128);
            // g_free: 2 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // dk_full: 2 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // dk_free: 2 barriers, init_count=128
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 128);
            // dq_done: 1 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 264);
    if (warp == 6) {
        int _tmem_hold = smem + 264;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;

    // ---- Role: math ----
    if (warp <= 3) {
        { // math_main
            int row_c = blockIdx.x;
            int lane_c = lane;
            int warp_c = warp;
            int tid_1 = warp_c * 32 + lane_c;
            int lane_base = warp_c * 32 << 16;
            long long q_row0 = (long long)row_c * 64;
            int c = tid_1;
            int qh = c / 16;
            int qd = c % 16 * 8;
            unsigned int _vec_load_3[4];
            {
                uint4 _uv4_0 = *reinterpret_cast<const uint4*>(index_q + ((q_row0 + (long long)qh) * 128 + (long long)qd) + 0);
                _vec_load_3[0 + 0] = _uv4_0.x;
                _vec_load_3[0 + 1] = _uv4_0.y;
                _vec_load_3[0 + 2] = _uv4_0.z;
                _vec_load_3[0 + 3] = _uv4_0.w;
            }
            int qoff = qd / 64 * 8192 + qh * 128 + (qd % 64 * 2 ^ qh % 8 << 4);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_raw_addr + (unsigned int)qoff), "r"(_vec_load_3[0]), "r"(_vec_load_3[1]), "r"(_vec_load_3[2]), "r"(_vec_load_3[3]) : "memory");
            int c_0 = 128 + tid_1;
            int qh_1 = c_0 / 16;
            int qd_2 = c_0 % 16 * 8;
            unsigned int _vec_load_4[4];
            {
                uint4 _uv4_1 = *reinterpret_cast<const uint4*>(index_q + ((q_row0 + (long long)qh_1) * 128 + (long long)qd_2) + 0);
                _vec_load_4[0 + 0] = _uv4_1.x;
                _vec_load_4[0 + 1] = _uv4_1.y;
                _vec_load_4[0 + 2] = _uv4_1.z;
                _vec_load_4[0 + 3] = _uv4_1.w;
            }
            int qoff_3 = qd_2 / 64 * 8192 + qh_1 * 128 + (qd_2 % 64 * 2 ^ qh_1 % 8 << 4);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_raw_addr + (unsigned int)qoff_3), "r"(_vec_load_4[0]), "r"(_vec_load_4[1]), "r"(_vec_load_4[2]), "r"(_vec_load_4[3]) : "memory");
            int c_4 = 256 + tid_1;
            int qh_5 = c_4 / 16;
            int qd_6 = c_4 % 16 * 8;
            unsigned int _vec_load_5[4];
            {
                uint4 _uv4_2 = *reinterpret_cast<const uint4*>(index_q + ((q_row0 + (long long)qh_5) * 128 + (long long)qd_6) + 0);
                _vec_load_5[0 + 0] = _uv4_2.x;
                _vec_load_5[0 + 1] = _uv4_2.y;
                _vec_load_5[0 + 2] = _uv4_2.z;
                _vec_load_5[0 + 3] = _uv4_2.w;
            }
            int qoff_7 = qd_6 / 64 * 8192 + qh_5 * 128 + (qd_6 % 64 * 2 ^ qh_5 % 8 << 4);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_raw_addr + (unsigned int)qoff_7), "r"(_vec_load_5[0]), "r"(_vec_load_5[1]), "r"(_vec_load_5[2]), "r"(_vec_load_5[3]) : "memory");
            int c_8 = 384 + tid_1;
            int qh_9 = c_8 / 16;
            int qd_10 = c_8 % 16 * 8;
            unsigned int _vec_load_6[4];
            {
                uint4 _uv4_3 = *reinterpret_cast<const uint4*>(index_q + ((q_row0 + (long long)qh_9) * 128 + (long long)qd_10) + 0);
                _vec_load_6[0 + 0] = _uv4_3.x;
                _vec_load_6[0 + 1] = _uv4_3.y;
                _vec_load_6[0 + 2] = _uv4_3.z;
                _vec_load_6[0 + 3] = _uv4_3.w;
            }
            int qoff_11 = qd_10 / 64 * 8192 + qh_9 * 128 + (qd_10 % 64 * 2 ^ qh_9 % 8 << 4);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_raw_addr + (unsigned int)qoff_11), "r"(_vec_load_6[0]), "r"(_vec_load_6[1]), "r"(_vec_load_6[2]), "r"(_vec_load_6[3]) : "memory");
            int c_12 = 512 + tid_1;
            int qh_13 = c_12 / 16;
            int qd_14 = c_12 % 16 * 8;
            unsigned int _vec_load_7[4];
            {
                uint4 _uv4_4 = *reinterpret_cast<const uint4*>(index_q + ((q_row0 + (long long)qh_13) * 128 + (long long)qd_14) + 0);
                _vec_load_7[0 + 0] = _uv4_4.x;
                _vec_load_7[0 + 1] = _uv4_4.y;
                _vec_load_7[0 + 2] = _uv4_4.z;
                _vec_load_7[0 + 3] = _uv4_4.w;
            }
            int qoff_15 = qd_14 / 64 * 8192 + qh_13 * 128 + (qd_14 % 64 * 2 ^ qh_13 % 8 << 4);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_raw_addr + (unsigned int)qoff_15), "r"(_vec_load_7[0]), "r"(_vec_load_7[1]), "r"(_vec_load_7[2]), "r"(_vec_load_7[3]) : "memory");
            int c_16 = 640 + tid_1;
            int qh_17 = c_16 / 16;
            int qd_18 = c_16 % 16 * 8;
            unsigned int _vec_load_8[4];
            {
                uint4 _uv4_5 = *reinterpret_cast<const uint4*>(index_q + ((q_row0 + (long long)qh_17) * 128 + (long long)qd_18) + 0);
                _vec_load_8[0 + 0] = _uv4_5.x;
                _vec_load_8[0 + 1] = _uv4_5.y;
                _vec_load_8[0 + 2] = _uv4_5.z;
                _vec_load_8[0 + 3] = _uv4_5.w;
            }
            int qoff_19 = qd_18 / 64 * 8192 + qh_17 * 128 + (qd_18 % 64 * 2 ^ qh_17 % 8 << 4);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_raw_addr + (unsigned int)qoff_19), "r"(_vec_load_8[0]), "r"(_vec_load_8[1]), "r"(_vec_load_8[2]), "r"(_vec_load_8[3]) : "memory");
            int c_20 = 768 + tid_1;
            int qh_21 = c_20 / 16;
            int qd_22 = c_20 % 16 * 8;
            unsigned int _vec_load_9[4];
            {
                uint4 _uv4_6 = *reinterpret_cast<const uint4*>(index_q + ((q_row0 + (long long)qh_21) * 128 + (long long)qd_22) + 0);
                _vec_load_9[0 + 0] = _uv4_6.x;
                _vec_load_9[0 + 1] = _uv4_6.y;
                _vec_load_9[0 + 2] = _uv4_6.z;
                _vec_load_9[0 + 3] = _uv4_6.w;
            }
            int qoff_23 = qd_22 / 64 * 8192 + qh_21 * 128 + (qd_22 % 64 * 2 ^ qh_21 % 8 << 4);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_raw_addr + (unsigned int)qoff_23), "r"(_vec_load_9[0]), "r"(_vec_load_9[1]), "r"(_vec_load_9[2]), "r"(_vec_load_9[3]) : "memory");
            int c_24 = 896 + tid_1;
            int qh_25 = c_24 / 16;
            int qd_26 = c_24 % 16 * 8;
            unsigned int _vec_load_10[4];
            {
                uint4 _uv4_7 = *reinterpret_cast<const uint4*>(index_q + ((q_row0 + (long long)qh_25) * 128 + (long long)qd_26) + 0);
                _vec_load_10[0 + 0] = _uv4_7.x;
                _vec_load_10[0 + 1] = _uv4_7.y;
                _vec_load_10[0 + 2] = _uv4_7.z;
                _vec_load_10[0 + 3] = _uv4_7.w;
            }
            int qoff_27 = qd_26 / 64 * 8192 + qh_25 * 128 + (qd_26 % 64 * 2 ^ qh_25 % 8 << 4);
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_raw_addr + (unsigned int)qoff_27), "r"(_vec_load_10[0]), "r"(_vec_load_10[1]), "r"(_vec_load_10[2]), "r"(_vec_load_10[3]) : "memory");
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(q_full_addr);
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_c = blocks_word[0];
            float dw[64];
            dw[0] = 0.0f;
            dw[1] = 0.0f;
            dw[2] = 0.0f;
            dw[3] = 0.0f;
            dw[4] = 0.0f;
            dw[5] = 0.0f;
            dw[6] = 0.0f;
            dw[7] = 0.0f;
            dw[8] = 0.0f;
            dw[9] = 0.0f;
            dw[10] = 0.0f;
            dw[11] = 0.0f;
            dw[12] = 0.0f;
            dw[13] = 0.0f;
            dw[14] = 0.0f;
            dw[15] = 0.0f;
            dw[16] = 0.0f;
            dw[17] = 0.0f;
            dw[18] = 0.0f;
            dw[19] = 0.0f;
            dw[20] = 0.0f;
            dw[21] = 0.0f;
            dw[22] = 0.0f;
            dw[23] = 0.0f;
            dw[24] = 0.0f;
            dw[25] = 0.0f;
            dw[26] = 0.0f;
            dw[27] = 0.0f;
            dw[28] = 0.0f;
            dw[29] = 0.0f;
            dw[30] = 0.0f;
            dw[31] = 0.0f;
            dw[32] = 0.0f;
            dw[33] = 0.0f;
            dw[34] = 0.0f;
            dw[35] = 0.0f;
            dw[36] = 0.0f;
            dw[37] = 0.0f;
            dw[38] = 0.0f;
            dw[39] = 0.0f;
            dw[40] = 0.0f;
            dw[41] = 0.0f;
            dw[42] = 0.0f;
            dw[43] = 0.0f;
            dw[44] = 0.0f;
            dw[45] = 0.0f;
            dw[46] = 0.0f;
            dw[47] = 0.0f;
            dw[48] = 0.0f;
            dw[49] = 0.0f;
            dw[50] = 0.0f;
            dw[51] = 0.0f;
            dw[52] = 0.0f;
            dw[53] = 0.0f;
            dw[54] = 0.0f;
            dw[55] = 0.0f;
            dw[56] = 0.0f;
            dw[57] = 0.0f;
            dw[58] = 0.0f;
            dw[59] = 0.0f;
            dw[60] = 0.0f;
            dw[61] = 0.0f;
            dw[62] = 0.0f;
            dw[63] = 0.0f;
            int gpan = tid_1 / 64 * 8192;
            int gcol = tid_1 % 64 * 2;
            int gx[8];
            gx[0] = gpan + (gcol ^ 0);
            gx[1] = gpan + (gcol ^ 16);
            gx[2] = gpan + (gcol ^ 32);
            gx[3] = gpan + (gcol ^ 48);
            gx[4] = gpan + (gcol ^ 64);
            gx[5] = gpan + (gcol ^ 80);
            gx[6] = gpan + (gcol ^ 96);
            gx[7] = gpan + (gcol ^ 112);
            int idx_prev = -1;
            #pragma unroll 1
            for (int i = 0; i < blocks_c; i++) {
                int par_c = i & 1;
                int use_c = i / 2 & 1;
                int slot_c = i % 6;
                mbarrier_wait(idx_full_addr + (slot_c) * 8, i / 6 & 1);
                int idx_t = s_idx[slot_c * 128 + tid_1];
                float grad_t = s_grad[slot_c * 128 + tid_1];
                mbarrier_arrive(idx_free_addr + (slot_c) * 8);
                mbarrier_wait(l_full_addr + (par_c) * 8, use_c);
                mbarrier_wait(g_free_addr + (par_c) * 8, use_c ^ 1);
                int gbase = 2 * par_c * 16384;
                float lg[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(lg[0]), "=f"(lg[1]), "=f"(lg[2]), "=f"(lg[3]), "=f"(lg[4]), "=f"(lg[5]), "=f"(lg[6]), "=f"(lg[7]), "=f"(lg[8]), "=f"(lg[9]), "=f"(lg[10]), "=f"(lg[11]), "=f"(lg[12]), "=f"(lg[13]), "=f"(lg[14]), "=f"(lg[15]), "=f"(lg[16]), "=f"(lg[17]), "=f"(lg[18]), "=f"(lg[19]), "=f"(lg[20]), "=f"(lg[21]), "=f"(lg[22]), "=f"(lg[23]), "=f"(lg[24]), "=f"(lg[25]), "=f"(lg[26]), "=f"(lg[27]), "=f"(lg[28]), "=f"(lg[29]), "=f"(lg[30]), "=f"(lg[31])
                    : "r"(tmem_tmem + par_c * 64 + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float g[32];
                float w_h = weights[q_row0];
                float _max_1 = max_noftz(lg[0], 0.0f);
                float _fma_0 = __fmaf_rn(grad_t, _max_1, dw[0]);
                dw[0] = _fma_0;
                g[0] = ((lg[0] > 0.0f) ? grad_t * w_h : 0.0f);
                float w_h_0 = weights[q_row0 + 1];
                float _max_2 = max_noftz(lg[1], 0.0f);
                float _fma_1 = __fmaf_rn(grad_t, _max_2, dw[1]);
                dw[1] = _fma_1;
                g[1] = ((lg[1] > 0.0f) ? grad_t * w_h_0 : 0.0f);
                float w_h_1 = weights[q_row0 + 2];
                float _max_3 = max_noftz(lg[2], 0.0f);
                float _fma_2 = __fmaf_rn(grad_t, _max_3, dw[2]);
                dw[2] = _fma_2;
                g[2] = ((lg[2] > 0.0f) ? grad_t * w_h_1 : 0.0f);
                float w_h_2 = weights[q_row0 + 3];
                float _max_4 = max_noftz(lg[3], 0.0f);
                float _fma_3 = __fmaf_rn(grad_t, _max_4, dw[3]);
                dw[3] = _fma_3;
                g[3] = ((lg[3] > 0.0f) ? grad_t * w_h_2 : 0.0f);
                float w_h_3 = weights[q_row0 + 4];
                float _max_5 = max_noftz(lg[4], 0.0f);
                float _fma_4 = __fmaf_rn(grad_t, _max_5, dw[4]);
                dw[4] = _fma_4;
                g[4] = ((lg[4] > 0.0f) ? grad_t * w_h_3 : 0.0f);
                float w_h_4 = weights[q_row0 + 5];
                float _max_6 = max_noftz(lg[5], 0.0f);
                float _fma_5 = __fmaf_rn(grad_t, _max_6, dw[5]);
                dw[5] = _fma_5;
                g[5] = ((lg[5] > 0.0f) ? grad_t * w_h_4 : 0.0f);
                float w_h_5 = weights[q_row0 + 6];
                float _max_7 = max_noftz(lg[6], 0.0f);
                float _fma_6 = __fmaf_rn(grad_t, _max_7, dw[6]);
                dw[6] = _fma_6;
                g[6] = ((lg[6] > 0.0f) ? grad_t * w_h_5 : 0.0f);
                float w_h_6 = weights[q_row0 + 7];
                float _max_8 = max_noftz(lg[7], 0.0f);
                float _fma_7 = __fmaf_rn(grad_t, _max_8, dw[7]);
                dw[7] = _fma_7;
                g[7] = ((lg[7] > 0.0f) ? grad_t * w_h_6 : 0.0f);
                float w_h_7 = weights[q_row0 + 8];
                float _max_9 = max_noftz(lg[8], 0.0f);
                float _fma_8 = __fmaf_rn(grad_t, _max_9, dw[8]);
                dw[8] = _fma_8;
                g[8] = ((lg[8] > 0.0f) ? grad_t * w_h_7 : 0.0f);
                float w_h_8 = weights[q_row0 + 9];
                float _max_10 = max_noftz(lg[9], 0.0f);
                float _fma_9 = __fmaf_rn(grad_t, _max_10, dw[9]);
                dw[9] = _fma_9;
                g[9] = ((lg[9] > 0.0f) ? grad_t * w_h_8 : 0.0f);
                float w_h_9 = weights[q_row0 + 10];
                float _max_11 = max_noftz(lg[10], 0.0f);
                float _fma_10 = __fmaf_rn(grad_t, _max_11, dw[10]);
                dw[10] = _fma_10;
                g[10] = ((lg[10] > 0.0f) ? grad_t * w_h_9 : 0.0f);
                float w_h_10 = weights[q_row0 + 11];
                float _max_12 = max_noftz(lg[11], 0.0f);
                float _fma_11 = __fmaf_rn(grad_t, _max_12, dw[11]);
                dw[11] = _fma_11;
                g[11] = ((lg[11] > 0.0f) ? grad_t * w_h_10 : 0.0f);
                float w_h_11 = weights[q_row0 + 12];
                float _max_13 = max_noftz(lg[12], 0.0f);
                float _fma_12 = __fmaf_rn(grad_t, _max_13, dw[12]);
                dw[12] = _fma_12;
                g[12] = ((lg[12] > 0.0f) ? grad_t * w_h_11 : 0.0f);
                float w_h_12 = weights[q_row0 + 13];
                float _max_14 = max_noftz(lg[13], 0.0f);
                float _fma_13 = __fmaf_rn(grad_t, _max_14, dw[13]);
                dw[13] = _fma_13;
                g[13] = ((lg[13] > 0.0f) ? grad_t * w_h_12 : 0.0f);
                float w_h_13 = weights[q_row0 + 14];
                float _max_15 = max_noftz(lg[14], 0.0f);
                float _fma_14 = __fmaf_rn(grad_t, _max_15, dw[14]);
                dw[14] = _fma_14;
                g[14] = ((lg[14] > 0.0f) ? grad_t * w_h_13 : 0.0f);
                float w_h_14 = weights[q_row0 + 15];
                float _max_16 = max_noftz(lg[15], 0.0f);
                float _fma_15 = __fmaf_rn(grad_t, _max_16, dw[15]);
                dw[15] = _fma_15;
                g[15] = ((lg[15] > 0.0f) ? grad_t * w_h_14 : 0.0f);
                float w_h_15 = weights[q_row0 + 16];
                float _max_17 = max_noftz(lg[16], 0.0f);
                float _fma_16 = __fmaf_rn(grad_t, _max_17, dw[16]);
                dw[16] = _fma_16;
                g[16] = ((lg[16] > 0.0f) ? grad_t * w_h_15 : 0.0f);
                float w_h_16 = weights[q_row0 + 17];
                float _max_18 = max_noftz(lg[17], 0.0f);
                float _fma_17 = __fmaf_rn(grad_t, _max_18, dw[17]);
                dw[17] = _fma_17;
                g[17] = ((lg[17] > 0.0f) ? grad_t * w_h_16 : 0.0f);
                float w_h_17 = weights[q_row0 + 18];
                float _max_19 = max_noftz(lg[18], 0.0f);
                float _fma_18 = __fmaf_rn(grad_t, _max_19, dw[18]);
                dw[18] = _fma_18;
                g[18] = ((lg[18] > 0.0f) ? grad_t * w_h_17 : 0.0f);
                float w_h_18 = weights[q_row0 + 19];
                float _max_20 = max_noftz(lg[19], 0.0f);
                float _fma_19 = __fmaf_rn(grad_t, _max_20, dw[19]);
                dw[19] = _fma_19;
                g[19] = ((lg[19] > 0.0f) ? grad_t * w_h_18 : 0.0f);
                float w_h_19 = weights[q_row0 + 20];
                float _max_21 = max_noftz(lg[20], 0.0f);
                float _fma_20 = __fmaf_rn(grad_t, _max_21, dw[20]);
                dw[20] = _fma_20;
                g[20] = ((lg[20] > 0.0f) ? grad_t * w_h_19 : 0.0f);
                float w_h_20 = weights[q_row0 + 21];
                float _max_22 = max_noftz(lg[21], 0.0f);
                float _fma_21 = __fmaf_rn(grad_t, _max_22, dw[21]);
                dw[21] = _fma_21;
                g[21] = ((lg[21] > 0.0f) ? grad_t * w_h_20 : 0.0f);
                float w_h_21 = weights[q_row0 + 22];
                float _max_23 = max_noftz(lg[22], 0.0f);
                float _fma_22 = __fmaf_rn(grad_t, _max_23, dw[22]);
                dw[22] = _fma_22;
                g[22] = ((lg[22] > 0.0f) ? grad_t * w_h_21 : 0.0f);
                float w_h_22 = weights[q_row0 + 23];
                float _max_24 = max_noftz(lg[23], 0.0f);
                float _fma_23 = __fmaf_rn(grad_t, _max_24, dw[23]);
                dw[23] = _fma_23;
                g[23] = ((lg[23] > 0.0f) ? grad_t * w_h_22 : 0.0f);
                float w_h_23 = weights[q_row0 + 24];
                float _max_25 = max_noftz(lg[24], 0.0f);
                float _fma_24 = __fmaf_rn(grad_t, _max_25, dw[24]);
                dw[24] = _fma_24;
                g[24] = ((lg[24] > 0.0f) ? grad_t * w_h_23 : 0.0f);
                float w_h_24 = weights[q_row0 + 25];
                float _max_26 = max_noftz(lg[25], 0.0f);
                float _fma_25 = __fmaf_rn(grad_t, _max_26, dw[25]);
                dw[25] = _fma_25;
                g[25] = ((lg[25] > 0.0f) ? grad_t * w_h_24 : 0.0f);
                float w_h_25 = weights[q_row0 + 26];
                float _max_27 = max_noftz(lg[26], 0.0f);
                float _fma_26 = __fmaf_rn(grad_t, _max_27, dw[26]);
                dw[26] = _fma_26;
                g[26] = ((lg[26] > 0.0f) ? grad_t * w_h_25 : 0.0f);
                float w_h_26 = weights[q_row0 + 27];
                float _max_28 = max_noftz(lg[27], 0.0f);
                float _fma_27 = __fmaf_rn(grad_t, _max_28, dw[27]);
                dw[27] = _fma_27;
                g[27] = ((lg[27] > 0.0f) ? grad_t * w_h_26 : 0.0f);
                float w_h_27 = weights[q_row0 + 28];
                float _max_29 = max_noftz(lg[28], 0.0f);
                float _fma_28 = __fmaf_rn(grad_t, _max_29, dw[28]);
                dw[28] = _fma_28;
                g[28] = ((lg[28] > 0.0f) ? grad_t * w_h_27 : 0.0f);
                float w_h_28 = weights[q_row0 + 29];
                float _max_30 = max_noftz(lg[29], 0.0f);
                float _fma_29 = __fmaf_rn(grad_t, _max_30, dw[29]);
                dw[29] = _fma_29;
                g[29] = ((lg[29] > 0.0f) ? grad_t * w_h_28 : 0.0f);
                float w_h_29 = weights[q_row0 + 30];
                float _max_31 = max_noftz(lg[30], 0.0f);
                float _fma_30 = __fmaf_rn(grad_t, _max_31, dw[30]);
                dw[30] = _fma_30;
                g[30] = ((lg[30] > 0.0f) ? grad_t * w_h_29 : 0.0f);
                float w_h_30 = weights[q_row0 + 31];
                float _max_32 = max_noftz(lg[31], 0.0f);
                float _fma_31 = __fmaf_rn(grad_t, _max_32, dw[31]);
                dw[31] = _fma_31;
                g[31] = ((lg[31] > 0.0f) ? grad_t * w_h_30 : 0.0f);
                uint32_t g_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(g[_lp*2 + 0], g[_lp*2+1 + 0]));
                    g_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float lo[32];
                float _cvt_f32_bf16_0;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_0) : "h"((uint16_t)(g_bf16[0] & 65535)));
                float hi_e = _cvt_f32_bf16_0;
                float _cvt_f32_bf16_1;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_1) : "h"((uint16_t)(g_bf16[0] >> 16)));
                float hi_o = _cvt_f32_bf16_1;
                lo[0] = g[0] - hi_e;
                lo[1] = g[1] - hi_o;
                float _cvt_f32_bf16_2;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_2) : "h"((uint16_t)(g_bf16[1] & 65535)));
                float hi_e_31 = _cvt_f32_bf16_2;
                float _cvt_f32_bf16_3;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_3) : "h"((uint16_t)(g_bf16[1] >> 16)));
                float hi_o_32 = _cvt_f32_bf16_3;
                lo[2] = g[2] - hi_e_31;
                lo[3] = g[3] - hi_o_32;
                float _cvt_f32_bf16_4;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_4) : "h"((uint16_t)(g_bf16[2] & 65535)));
                float hi_e_33 = _cvt_f32_bf16_4;
                float _cvt_f32_bf16_5;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_5) : "h"((uint16_t)(g_bf16[2] >> 16)));
                float hi_o_34 = _cvt_f32_bf16_5;
                lo[4] = g[4] - hi_e_33;
                lo[5] = g[5] - hi_o_34;
                float _cvt_f32_bf16_6;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_6) : "h"((uint16_t)(g_bf16[3] & 65535)));
                float hi_e_35 = _cvt_f32_bf16_6;
                float _cvt_f32_bf16_7;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_7) : "h"((uint16_t)(g_bf16[3] >> 16)));
                float hi_o_36 = _cvt_f32_bf16_7;
                lo[6] = g[6] - hi_e_35;
                lo[7] = g[7] - hi_o_36;
                float _cvt_f32_bf16_8;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_8) : "h"((uint16_t)(g_bf16[4] & 65535)));
                float hi_e_37 = _cvt_f32_bf16_8;
                float _cvt_f32_bf16_9;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_9) : "h"((uint16_t)(g_bf16[4] >> 16)));
                float hi_o_38 = _cvt_f32_bf16_9;
                lo[8] = g[8] - hi_e_37;
                lo[9] = g[9] - hi_o_38;
                float _cvt_f32_bf16_10;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_10) : "h"((uint16_t)(g_bf16[5] & 65535)));
                float hi_e_39 = _cvt_f32_bf16_10;
                float _cvt_f32_bf16_11;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_11) : "h"((uint16_t)(g_bf16[5] >> 16)));
                float hi_o_40 = _cvt_f32_bf16_11;
                lo[10] = g[10] - hi_e_39;
                lo[11] = g[11] - hi_o_40;
                float _cvt_f32_bf16_12;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_12) : "h"((uint16_t)(g_bf16[6] & 65535)));
                float hi_e_41 = _cvt_f32_bf16_12;
                float _cvt_f32_bf16_13;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_13) : "h"((uint16_t)(g_bf16[6] >> 16)));
                float hi_o_42 = _cvt_f32_bf16_13;
                lo[12] = g[12] - hi_e_41;
                lo[13] = g[13] - hi_o_42;
                float _cvt_f32_bf16_14;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_14) : "h"((uint16_t)(g_bf16[7] & 65535)));
                float hi_e_43 = _cvt_f32_bf16_14;
                float _cvt_f32_bf16_15;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_15) : "h"((uint16_t)(g_bf16[7] >> 16)));
                float hi_o_44 = _cvt_f32_bf16_15;
                lo[14] = g[14] - hi_e_43;
                lo[15] = g[15] - hi_o_44;
                float _cvt_f32_bf16_16;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_16) : "h"((uint16_t)(g_bf16[8] & 65535)));
                float hi_e_45 = _cvt_f32_bf16_16;
                float _cvt_f32_bf16_17;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_17) : "h"((uint16_t)(g_bf16[8] >> 16)));
                float hi_o_46 = _cvt_f32_bf16_17;
                lo[16] = g[16] - hi_e_45;
                lo[17] = g[17] - hi_o_46;
                float _cvt_f32_bf16_18;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_18) : "h"((uint16_t)(g_bf16[9] & 65535)));
                float hi_e_47 = _cvt_f32_bf16_18;
                float _cvt_f32_bf16_19;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_19) : "h"((uint16_t)(g_bf16[9] >> 16)));
                float hi_o_48 = _cvt_f32_bf16_19;
                lo[18] = g[18] - hi_e_47;
                lo[19] = g[19] - hi_o_48;
                float _cvt_f32_bf16_20;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_20) : "h"((uint16_t)(g_bf16[10] & 65535)));
                float hi_e_49 = _cvt_f32_bf16_20;
                float _cvt_f32_bf16_21;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_21) : "h"((uint16_t)(g_bf16[10] >> 16)));
                float hi_o_50 = _cvt_f32_bf16_21;
                lo[20] = g[20] - hi_e_49;
                lo[21] = g[21] - hi_o_50;
                float _cvt_f32_bf16_22;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_22) : "h"((uint16_t)(g_bf16[11] & 65535)));
                float hi_e_51 = _cvt_f32_bf16_22;
                float _cvt_f32_bf16_23;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_23) : "h"((uint16_t)(g_bf16[11] >> 16)));
                float hi_o_52 = _cvt_f32_bf16_23;
                lo[22] = g[22] - hi_e_51;
                lo[23] = g[23] - hi_o_52;
                float _cvt_f32_bf16_24;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_24) : "h"((uint16_t)(g_bf16[12] & 65535)));
                float hi_e_53 = _cvt_f32_bf16_24;
                float _cvt_f32_bf16_25;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_25) : "h"((uint16_t)(g_bf16[12] >> 16)));
                float hi_o_54 = _cvt_f32_bf16_25;
                lo[24] = g[24] - hi_e_53;
                lo[25] = g[25] - hi_o_54;
                float _cvt_f32_bf16_26;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_26) : "h"((uint16_t)(g_bf16[13] & 65535)));
                float hi_e_55 = _cvt_f32_bf16_26;
                float _cvt_f32_bf16_27;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_27) : "h"((uint16_t)(g_bf16[13] >> 16)));
                float hi_o_56 = _cvt_f32_bf16_27;
                lo[26] = g[26] - hi_e_55;
                lo[27] = g[27] - hi_o_56;
                float _cvt_f32_bf16_28;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_28) : "h"((uint16_t)(g_bf16[14] & 65535)));
                float hi_e_57 = _cvt_f32_bf16_28;
                float _cvt_f32_bf16_29;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_29) : "h"((uint16_t)(g_bf16[14] >> 16)));
                float hi_o_58 = _cvt_f32_bf16_29;
                lo[28] = g[28] - hi_e_57;
                lo[29] = g[29] - hi_o_58;
                float _cvt_f32_bf16_30;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_30) : "h"((uint16_t)(g_bf16[15] & 65535)));
                float hi_e_59 = _cvt_f32_bf16_30;
                float _cvt_f32_bf16_31;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_31) : "h"((uint16_t)(g_bf16[15] >> 16)));
                float hi_o_60 = _cvt_f32_bf16_31;
                lo[30] = g[30] - hi_e_59;
                lo[31] = g[31] - hi_o_60;
                {
                    __nv_bfloat16 _bval_8 = __float2bfloat16_rn(g[0]);
                    uint16_t _bits_8 = *(uint16_t*)&_bval_8;
                    uint32_t _addr_8 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_8), "h"(_bits_8) : "memory");
                }
                {
                    __nv_bfloat16 _bval_9 = __float2bfloat16_rn(lo[0]);
                    uint16_t _bits_9 = *(uint16_t*)&_bval_9;
                    uint32_t _addr_9 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_9), "h"(_bits_9) : "memory");
                }
                {
                    __nv_bfloat16 _bval_10 = __float2bfloat16_rn(g[1]);
                    uint16_t _bits_10 = *(uint16_t*)&_bval_10;
                    uint32_t _addr_10 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 128 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_10), "h"(_bits_10) : "memory");
                }
                {
                    __nv_bfloat16 _bval_11 = __float2bfloat16_rn(lo[1]);
                    uint16_t _bits_11 = *(uint16_t*)&_bval_11;
                    uint32_t _addr_11 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 128 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_11), "h"(_bits_11) : "memory");
                }
                {
                    __nv_bfloat16 _bval_12 = __float2bfloat16_rn(g[2]);
                    uint16_t _bits_12 = *(uint16_t*)&_bval_12;
                    uint32_t _addr_12 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 256 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_12), "h"(_bits_12) : "memory");
                }
                {
                    __nv_bfloat16 _bval_13 = __float2bfloat16_rn(lo[2]);
                    uint16_t _bits_13 = *(uint16_t*)&_bval_13;
                    uint32_t _addr_13 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 256 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_13), "h"(_bits_13) : "memory");
                }
                {
                    __nv_bfloat16 _bval_14 = __float2bfloat16_rn(g[3]);
                    uint16_t _bits_14 = *(uint16_t*)&_bval_14;
                    uint32_t _addr_14 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 384 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_14), "h"(_bits_14) : "memory");
                }
                {
                    __nv_bfloat16 _bval_15 = __float2bfloat16_rn(lo[3]);
                    uint16_t _bits_15 = *(uint16_t*)&_bval_15;
                    uint32_t _addr_15 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 384 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_15), "h"(_bits_15) : "memory");
                }
                {
                    __nv_bfloat16 _bval_16 = __float2bfloat16_rn(g[4]);
                    uint16_t _bits_16 = *(uint16_t*)&_bval_16;
                    uint32_t _addr_16 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 512 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_16), "h"(_bits_16) : "memory");
                }
                {
                    __nv_bfloat16 _bval_17 = __float2bfloat16_rn(lo[4]);
                    uint16_t _bits_17 = *(uint16_t*)&_bval_17;
                    uint32_t _addr_17 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 512 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_17), "h"(_bits_17) : "memory");
                }
                {
                    __nv_bfloat16 _bval_18 = __float2bfloat16_rn(g[5]);
                    uint16_t _bits_18 = *(uint16_t*)&_bval_18;
                    uint32_t _addr_18 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 640 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_18), "h"(_bits_18) : "memory");
                }
                {
                    __nv_bfloat16 _bval_19 = __float2bfloat16_rn(lo[5]);
                    uint16_t _bits_19 = *(uint16_t*)&_bval_19;
                    uint32_t _addr_19 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 640 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_19), "h"(_bits_19) : "memory");
                }
                {
                    __nv_bfloat16 _bval_20 = __float2bfloat16_rn(g[6]);
                    uint16_t _bits_20 = *(uint16_t*)&_bval_20;
                    uint32_t _addr_20 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 768 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_20), "h"(_bits_20) : "memory");
                }
                {
                    __nv_bfloat16 _bval_21 = __float2bfloat16_rn(lo[6]);
                    uint16_t _bits_21 = *(uint16_t*)&_bval_21;
                    uint32_t _addr_21 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 768 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_21), "h"(_bits_21) : "memory");
                }
                {
                    __nv_bfloat16 _bval_22 = __float2bfloat16_rn(g[7]);
                    uint16_t _bits_22 = *(uint16_t*)&_bval_22;
                    uint32_t _addr_22 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 896 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_22), "h"(_bits_22) : "memory");
                }
                {
                    __nv_bfloat16 _bval_23 = __float2bfloat16_rn(lo[7]);
                    uint16_t _bits_23 = *(uint16_t*)&_bval_23;
                    uint32_t _addr_23 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 896 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_23), "h"(_bits_23) : "memory");
                }
                {
                    __nv_bfloat16 _bval_24 = __float2bfloat16_rn(g[8]);
                    uint16_t _bits_24 = *(uint16_t*)&_bval_24;
                    uint32_t _addr_24 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 1024 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_24), "h"(_bits_24) : "memory");
                }
                {
                    __nv_bfloat16 _bval_25 = __float2bfloat16_rn(lo[8]);
                    uint16_t _bits_25 = *(uint16_t*)&_bval_25;
                    uint32_t _addr_25 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 1024 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_25), "h"(_bits_25) : "memory");
                }
                {
                    __nv_bfloat16 _bval_26 = __float2bfloat16_rn(g[9]);
                    uint16_t _bits_26 = *(uint16_t*)&_bval_26;
                    uint32_t _addr_26 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 1152 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_26), "h"(_bits_26) : "memory");
                }
                {
                    __nv_bfloat16 _bval_27 = __float2bfloat16_rn(lo[9]);
                    uint16_t _bits_27 = *(uint16_t*)&_bval_27;
                    uint32_t _addr_27 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 1152 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_27), "h"(_bits_27) : "memory");
                }
                {
                    __nv_bfloat16 _bval_28 = __float2bfloat16_rn(g[10]);
                    uint16_t _bits_28 = *(uint16_t*)&_bval_28;
                    uint32_t _addr_28 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 1280 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_28), "h"(_bits_28) : "memory");
                }
                {
                    __nv_bfloat16 _bval_29 = __float2bfloat16_rn(lo[10]);
                    uint16_t _bits_29 = *(uint16_t*)&_bval_29;
                    uint32_t _addr_29 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 1280 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_29), "h"(_bits_29) : "memory");
                }
                {
                    __nv_bfloat16 _bval_30 = __float2bfloat16_rn(g[11]);
                    uint16_t _bits_30 = *(uint16_t*)&_bval_30;
                    uint32_t _addr_30 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 1408 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_30), "h"(_bits_30) : "memory");
                }
                {
                    __nv_bfloat16 _bval_31 = __float2bfloat16_rn(lo[11]);
                    uint16_t _bits_31 = *(uint16_t*)&_bval_31;
                    uint32_t _addr_31 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 1408 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_31), "h"(_bits_31) : "memory");
                }
                {
                    __nv_bfloat16 _bval_32 = __float2bfloat16_rn(g[12]);
                    uint16_t _bits_32 = *(uint16_t*)&_bval_32;
                    uint32_t _addr_32 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 1536 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_32), "h"(_bits_32) : "memory");
                }
                {
                    __nv_bfloat16 _bval_33 = __float2bfloat16_rn(lo[12]);
                    uint16_t _bits_33 = *(uint16_t*)&_bval_33;
                    uint32_t _addr_33 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 1536 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_33), "h"(_bits_33) : "memory");
                }
                {
                    __nv_bfloat16 _bval_34 = __float2bfloat16_rn(g[13]);
                    uint16_t _bits_34 = *(uint16_t*)&_bval_34;
                    uint32_t _addr_34 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 1664 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_34), "h"(_bits_34) : "memory");
                }
                {
                    __nv_bfloat16 _bval_35 = __float2bfloat16_rn(lo[13]);
                    uint16_t _bits_35 = *(uint16_t*)&_bval_35;
                    uint32_t _addr_35 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 1664 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_35), "h"(_bits_35) : "memory");
                }
                {
                    __nv_bfloat16 _bval_36 = __float2bfloat16_rn(g[14]);
                    uint16_t _bits_36 = *(uint16_t*)&_bval_36;
                    uint32_t _addr_36 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 1792 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_36), "h"(_bits_36) : "memory");
                }
                {
                    __nv_bfloat16 _bval_37 = __float2bfloat16_rn(lo[14]);
                    uint16_t _bits_37 = *(uint16_t*)&_bval_37;
                    uint32_t _addr_37 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 1792 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_37), "h"(_bits_37) : "memory");
                }
                {
                    __nv_bfloat16 _bval_38 = __float2bfloat16_rn(g[15]);
                    uint16_t _bits_38 = *(uint16_t*)&_bval_38;
                    uint32_t _addr_38 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 1920 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_38), "h"(_bits_38) : "memory");
                }
                {
                    __nv_bfloat16 _bval_39 = __float2bfloat16_rn(lo[15]);
                    uint16_t _bits_39 = *(uint16_t*)&_bval_39;
                    uint32_t _addr_39 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 1920 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_39), "h"(_bits_39) : "memory");
                }
                {
                    __nv_bfloat16 _bval_40 = __float2bfloat16_rn(g[16]);
                    uint16_t _bits_40 = *(uint16_t*)&_bval_40;
                    uint32_t _addr_40 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 2048 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_40), "h"(_bits_40) : "memory");
                }
                {
                    __nv_bfloat16 _bval_41 = __float2bfloat16_rn(lo[16]);
                    uint16_t _bits_41 = *(uint16_t*)&_bval_41;
                    uint32_t _addr_41 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 2048 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_41), "h"(_bits_41) : "memory");
                }
                {
                    __nv_bfloat16 _bval_42 = __float2bfloat16_rn(g[17]);
                    uint16_t _bits_42 = *(uint16_t*)&_bval_42;
                    uint32_t _addr_42 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 2176 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_42), "h"(_bits_42) : "memory");
                }
                {
                    __nv_bfloat16 _bval_43 = __float2bfloat16_rn(lo[17]);
                    uint16_t _bits_43 = *(uint16_t*)&_bval_43;
                    uint32_t _addr_43 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 2176 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_43), "h"(_bits_43) : "memory");
                }
                {
                    __nv_bfloat16 _bval_44 = __float2bfloat16_rn(g[18]);
                    uint16_t _bits_44 = *(uint16_t*)&_bval_44;
                    uint32_t _addr_44 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 2304 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_44), "h"(_bits_44) : "memory");
                }
                {
                    __nv_bfloat16 _bval_45 = __float2bfloat16_rn(lo[18]);
                    uint16_t _bits_45 = *(uint16_t*)&_bval_45;
                    uint32_t _addr_45 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 2304 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_45), "h"(_bits_45) : "memory");
                }
                {
                    __nv_bfloat16 _bval_46 = __float2bfloat16_rn(g[19]);
                    uint16_t _bits_46 = *(uint16_t*)&_bval_46;
                    uint32_t _addr_46 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 2432 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_46), "h"(_bits_46) : "memory");
                }
                {
                    __nv_bfloat16 _bval_47 = __float2bfloat16_rn(lo[19]);
                    uint16_t _bits_47 = *(uint16_t*)&_bval_47;
                    uint32_t _addr_47 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 2432 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_47), "h"(_bits_47) : "memory");
                }
                {
                    __nv_bfloat16 _bval_48 = __float2bfloat16_rn(g[20]);
                    uint16_t _bits_48 = *(uint16_t*)&_bval_48;
                    uint32_t _addr_48 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 2560 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_48), "h"(_bits_48) : "memory");
                }
                {
                    __nv_bfloat16 _bval_49 = __float2bfloat16_rn(lo[20]);
                    uint16_t _bits_49 = *(uint16_t*)&_bval_49;
                    uint32_t _addr_49 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 2560 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_49), "h"(_bits_49) : "memory");
                }
                {
                    __nv_bfloat16 _bval_50 = __float2bfloat16_rn(g[21]);
                    uint16_t _bits_50 = *(uint16_t*)&_bval_50;
                    uint32_t _addr_50 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 2688 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_50), "h"(_bits_50) : "memory");
                }
                {
                    __nv_bfloat16 _bval_51 = __float2bfloat16_rn(lo[21]);
                    uint16_t _bits_51 = *(uint16_t*)&_bval_51;
                    uint32_t _addr_51 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 2688 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_51), "h"(_bits_51) : "memory");
                }
                {
                    __nv_bfloat16 _bval_52 = __float2bfloat16_rn(g[22]);
                    uint16_t _bits_52 = *(uint16_t*)&_bval_52;
                    uint32_t _addr_52 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 2816 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_52), "h"(_bits_52) : "memory");
                }
                {
                    __nv_bfloat16 _bval_53 = __float2bfloat16_rn(lo[22]);
                    uint16_t _bits_53 = *(uint16_t*)&_bval_53;
                    uint32_t _addr_53 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 2816 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_53), "h"(_bits_53) : "memory");
                }
                {
                    __nv_bfloat16 _bval_54 = __float2bfloat16_rn(g[23]);
                    uint16_t _bits_54 = *(uint16_t*)&_bval_54;
                    uint32_t _addr_54 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 2944 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_54), "h"(_bits_54) : "memory");
                }
                {
                    __nv_bfloat16 _bval_55 = __float2bfloat16_rn(lo[23]);
                    uint16_t _bits_55 = *(uint16_t*)&_bval_55;
                    uint32_t _addr_55 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 2944 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_55), "h"(_bits_55) : "memory");
                }
                {
                    __nv_bfloat16 _bval_56 = __float2bfloat16_rn(g[24]);
                    uint16_t _bits_56 = *(uint16_t*)&_bval_56;
                    uint32_t _addr_56 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 3072 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_56), "h"(_bits_56) : "memory");
                }
                {
                    __nv_bfloat16 _bval_57 = __float2bfloat16_rn(lo[24]);
                    uint16_t _bits_57 = *(uint16_t*)&_bval_57;
                    uint32_t _addr_57 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 3072 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_57), "h"(_bits_57) : "memory");
                }
                {
                    __nv_bfloat16 _bval_58 = __float2bfloat16_rn(g[25]);
                    uint16_t _bits_58 = *(uint16_t*)&_bval_58;
                    uint32_t _addr_58 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 3200 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_58), "h"(_bits_58) : "memory");
                }
                {
                    __nv_bfloat16 _bval_59 = __float2bfloat16_rn(lo[25]);
                    uint16_t _bits_59 = *(uint16_t*)&_bval_59;
                    uint32_t _addr_59 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 3200 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_59), "h"(_bits_59) : "memory");
                }
                {
                    __nv_bfloat16 _bval_60 = __float2bfloat16_rn(g[26]);
                    uint16_t _bits_60 = *(uint16_t*)&_bval_60;
                    uint32_t _addr_60 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 3328 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_60), "h"(_bits_60) : "memory");
                }
                {
                    __nv_bfloat16 _bval_61 = __float2bfloat16_rn(lo[26]);
                    uint16_t _bits_61 = *(uint16_t*)&_bval_61;
                    uint32_t _addr_61 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 3328 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_61), "h"(_bits_61) : "memory");
                }
                {
                    __nv_bfloat16 _bval_62 = __float2bfloat16_rn(g[27]);
                    uint16_t _bits_62 = *(uint16_t*)&_bval_62;
                    uint32_t _addr_62 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 3456 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_62), "h"(_bits_62) : "memory");
                }
                {
                    __nv_bfloat16 _bval_63 = __float2bfloat16_rn(lo[27]);
                    uint16_t _bits_63 = *(uint16_t*)&_bval_63;
                    uint32_t _addr_63 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 3456 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_63), "h"(_bits_63) : "memory");
                }
                {
                    __nv_bfloat16 _bval_64 = __float2bfloat16_rn(g[28]);
                    uint16_t _bits_64 = *(uint16_t*)&_bval_64;
                    uint32_t _addr_64 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 3584 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_64), "h"(_bits_64) : "memory");
                }
                {
                    __nv_bfloat16 _bval_65 = __float2bfloat16_rn(lo[28]);
                    uint16_t _bits_65 = *(uint16_t*)&_bval_65;
                    uint32_t _addr_65 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 3584 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_65), "h"(_bits_65) : "memory");
                }
                {
                    __nv_bfloat16 _bval_66 = __float2bfloat16_rn(g[29]);
                    uint16_t _bits_66 = *(uint16_t*)&_bval_66;
                    uint32_t _addr_66 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 3712 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_66), "h"(_bits_66) : "memory");
                }
                {
                    __nv_bfloat16 _bval_67 = __float2bfloat16_rn(lo[29]);
                    uint16_t _bits_67 = *(uint16_t*)&_bval_67;
                    uint32_t _addr_67 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 3712 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_67), "h"(_bits_67) : "memory");
                }
                {
                    __nv_bfloat16 _bval_68 = __float2bfloat16_rn(g[30]);
                    uint16_t _bits_68 = *(uint16_t*)&_bval_68;
                    uint32_t _addr_68 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 3840 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_68), "h"(_bits_68) : "memory");
                }
                {
                    __nv_bfloat16 _bval_69 = __float2bfloat16_rn(lo[30]);
                    uint16_t _bits_69 = *(uint16_t*)&_bval_69;
                    uint32_t _addr_69 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 3840 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_69), "h"(_bits_69) : "memory");
                }
                {
                    __nv_bfloat16 _bval_70 = __float2bfloat16_rn(g[31]);
                    uint16_t _bits_70 = *(uint16_t*)&_bval_70;
                    uint32_t _addr_70 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 3968 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_70), "h"(_bits_70) : "memory");
                }
                {
                    __nv_bfloat16 _bval_71 = __float2bfloat16_rn(lo[31]);
                    uint16_t _bits_71 = *(uint16_t*)&_bval_71;
                    uint32_t _addr_71 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 3968 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_71), "h"(_bits_71) : "memory");
                }
                float lg_61[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(lg_61[0]), "=f"(lg_61[1]), "=f"(lg_61[2]), "=f"(lg_61[3]), "=f"(lg_61[4]), "=f"(lg_61[5]), "=f"(lg_61[6]), "=f"(lg_61[7]), "=f"(lg_61[8]), "=f"(lg_61[9]), "=f"(lg_61[10]), "=f"(lg_61[11]), "=f"(lg_61[12]), "=f"(lg_61[13]), "=f"(lg_61[14]), "=f"(lg_61[15]), "=f"(lg_61[16]), "=f"(lg_61[17]), "=f"(lg_61[18]), "=f"(lg_61[19]), "=f"(lg_61[20]), "=f"(lg_61[21]), "=f"(lg_61[22]), "=f"(lg_61[23]), "=f"(lg_61[24]), "=f"(lg_61[25]), "=f"(lg_61[26]), "=f"(lg_61[27]), "=f"(lg_61[28]), "=f"(lg_61[29]), "=f"(lg_61[30]), "=f"(lg_61[31])
                    : "r"(tmem_tmem + par_c * 64 + 32 + lane_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                float g_62[32];
                float w_h_63 = weights[q_row0 + 32];
                float _max_33 = max_noftz(lg_61[0], 0.0f);
                float _fma_32 = __fmaf_rn(grad_t, _max_33, dw[32]);
                dw[32] = _fma_32;
                g_62[0] = ((lg_61[0] > 0.0f) ? grad_t * w_h_63 : 0.0f);
                float w_h_64 = weights[q_row0 + 33];
                float _max_34 = max_noftz(lg_61[1], 0.0f);
                float _fma_33 = __fmaf_rn(grad_t, _max_34, dw[33]);
                dw[33] = _fma_33;
                g_62[1] = ((lg_61[1] > 0.0f) ? grad_t * w_h_64 : 0.0f);
                float w_h_65 = weights[q_row0 + 34];
                float _max_35 = max_noftz(lg_61[2], 0.0f);
                float _fma_34 = __fmaf_rn(grad_t, _max_35, dw[34]);
                dw[34] = _fma_34;
                g_62[2] = ((lg_61[2] > 0.0f) ? grad_t * w_h_65 : 0.0f);
                float w_h_66 = weights[q_row0 + 35];
                float _max_36 = max_noftz(lg_61[3], 0.0f);
                float _fma_35 = __fmaf_rn(grad_t, _max_36, dw[35]);
                dw[35] = _fma_35;
                g_62[3] = ((lg_61[3] > 0.0f) ? grad_t * w_h_66 : 0.0f);
                float w_h_67 = weights[q_row0 + 36];
                float _max_37 = max_noftz(lg_61[4], 0.0f);
                float _fma_36 = __fmaf_rn(grad_t, _max_37, dw[36]);
                dw[36] = _fma_36;
                g_62[4] = ((lg_61[4] > 0.0f) ? grad_t * w_h_67 : 0.0f);
                float w_h_68 = weights[q_row0 + 37];
                float _max_38 = max_noftz(lg_61[5], 0.0f);
                float _fma_37 = __fmaf_rn(grad_t, _max_38, dw[37]);
                dw[37] = _fma_37;
                g_62[5] = ((lg_61[5] > 0.0f) ? grad_t * w_h_68 : 0.0f);
                float w_h_69 = weights[q_row0 + 38];
                float _max_39 = max_noftz(lg_61[6], 0.0f);
                float _fma_38 = __fmaf_rn(grad_t, _max_39, dw[38]);
                dw[38] = _fma_38;
                g_62[6] = ((lg_61[6] > 0.0f) ? grad_t * w_h_69 : 0.0f);
                float w_h_70 = weights[q_row0 + 39];
                float _max_40 = max_noftz(lg_61[7], 0.0f);
                float _fma_39 = __fmaf_rn(grad_t, _max_40, dw[39]);
                dw[39] = _fma_39;
                g_62[7] = ((lg_61[7] > 0.0f) ? grad_t * w_h_70 : 0.0f);
                float w_h_71 = weights[q_row0 + 40];
                float _max_41 = max_noftz(lg_61[8], 0.0f);
                float _fma_40 = __fmaf_rn(grad_t, _max_41, dw[40]);
                dw[40] = _fma_40;
                g_62[8] = ((lg_61[8] > 0.0f) ? grad_t * w_h_71 : 0.0f);
                float w_h_72 = weights[q_row0 + 41];
                float _max_42 = max_noftz(lg_61[9], 0.0f);
                float _fma_41 = __fmaf_rn(grad_t, _max_42, dw[41]);
                dw[41] = _fma_41;
                g_62[9] = ((lg_61[9] > 0.0f) ? grad_t * w_h_72 : 0.0f);
                float w_h_73 = weights[q_row0 + 42];
                float _max_43 = max_noftz(lg_61[10], 0.0f);
                float _fma_42 = __fmaf_rn(grad_t, _max_43, dw[42]);
                dw[42] = _fma_42;
                g_62[10] = ((lg_61[10] > 0.0f) ? grad_t * w_h_73 : 0.0f);
                float w_h_74 = weights[q_row0 + 43];
                float _max_44 = max_noftz(lg_61[11], 0.0f);
                float _fma_43 = __fmaf_rn(grad_t, _max_44, dw[43]);
                dw[43] = _fma_43;
                g_62[11] = ((lg_61[11] > 0.0f) ? grad_t * w_h_74 : 0.0f);
                float w_h_75 = weights[q_row0 + 44];
                float _max_45 = max_noftz(lg_61[12], 0.0f);
                float _fma_44 = __fmaf_rn(grad_t, _max_45, dw[44]);
                dw[44] = _fma_44;
                g_62[12] = ((lg_61[12] > 0.0f) ? grad_t * w_h_75 : 0.0f);
                float w_h_76 = weights[q_row0 + 45];
                float _max_46 = max_noftz(lg_61[13], 0.0f);
                float _fma_45 = __fmaf_rn(grad_t, _max_46, dw[45]);
                dw[45] = _fma_45;
                g_62[13] = ((lg_61[13] > 0.0f) ? grad_t * w_h_76 : 0.0f);
                float w_h_77 = weights[q_row0 + 46];
                float _max_47 = max_noftz(lg_61[14], 0.0f);
                float _fma_46 = __fmaf_rn(grad_t, _max_47, dw[46]);
                dw[46] = _fma_46;
                g_62[14] = ((lg_61[14] > 0.0f) ? grad_t * w_h_77 : 0.0f);
                float w_h_78 = weights[q_row0 + 47];
                float _max_48 = max_noftz(lg_61[15], 0.0f);
                float _fma_47 = __fmaf_rn(grad_t, _max_48, dw[47]);
                dw[47] = _fma_47;
                g_62[15] = ((lg_61[15] > 0.0f) ? grad_t * w_h_78 : 0.0f);
                float w_h_79 = weights[q_row0 + 48];
                float _max_49 = max_noftz(lg_61[16], 0.0f);
                float _fma_48 = __fmaf_rn(grad_t, _max_49, dw[48]);
                dw[48] = _fma_48;
                g_62[16] = ((lg_61[16] > 0.0f) ? grad_t * w_h_79 : 0.0f);
                float w_h_80 = weights[q_row0 + 49];
                float _max_50 = max_noftz(lg_61[17], 0.0f);
                float _fma_49 = __fmaf_rn(grad_t, _max_50, dw[49]);
                dw[49] = _fma_49;
                g_62[17] = ((lg_61[17] > 0.0f) ? grad_t * w_h_80 : 0.0f);
                float w_h_81 = weights[q_row0 + 50];
                float _max_51 = max_noftz(lg_61[18], 0.0f);
                float _fma_50 = __fmaf_rn(grad_t, _max_51, dw[50]);
                dw[50] = _fma_50;
                g_62[18] = ((lg_61[18] > 0.0f) ? grad_t * w_h_81 : 0.0f);
                float w_h_82 = weights[q_row0 + 51];
                float _max_52 = max_noftz(lg_61[19], 0.0f);
                float _fma_51 = __fmaf_rn(grad_t, _max_52, dw[51]);
                dw[51] = _fma_51;
                g_62[19] = ((lg_61[19] > 0.0f) ? grad_t * w_h_82 : 0.0f);
                float w_h_83 = weights[q_row0 + 52];
                float _max_53 = max_noftz(lg_61[20], 0.0f);
                float _fma_52 = __fmaf_rn(grad_t, _max_53, dw[52]);
                dw[52] = _fma_52;
                g_62[20] = ((lg_61[20] > 0.0f) ? grad_t * w_h_83 : 0.0f);
                float w_h_84 = weights[q_row0 + 53];
                float _max_54 = max_noftz(lg_61[21], 0.0f);
                float _fma_53 = __fmaf_rn(grad_t, _max_54, dw[53]);
                dw[53] = _fma_53;
                g_62[21] = ((lg_61[21] > 0.0f) ? grad_t * w_h_84 : 0.0f);
                float w_h_85 = weights[q_row0 + 54];
                float _max_55 = max_noftz(lg_61[22], 0.0f);
                float _fma_54 = __fmaf_rn(grad_t, _max_55, dw[54]);
                dw[54] = _fma_54;
                g_62[22] = ((lg_61[22] > 0.0f) ? grad_t * w_h_85 : 0.0f);
                float w_h_86 = weights[q_row0 + 55];
                float _max_56 = max_noftz(lg_61[23], 0.0f);
                float _fma_55 = __fmaf_rn(grad_t, _max_56, dw[55]);
                dw[55] = _fma_55;
                g_62[23] = ((lg_61[23] > 0.0f) ? grad_t * w_h_86 : 0.0f);
                float w_h_87 = weights[q_row0 + 56];
                float _max_57 = max_noftz(lg_61[24], 0.0f);
                float _fma_56 = __fmaf_rn(grad_t, _max_57, dw[56]);
                dw[56] = _fma_56;
                g_62[24] = ((lg_61[24] > 0.0f) ? grad_t * w_h_87 : 0.0f);
                float w_h_88 = weights[q_row0 + 57];
                float _max_58 = max_noftz(lg_61[25], 0.0f);
                float _fma_57 = __fmaf_rn(grad_t, _max_58, dw[57]);
                dw[57] = _fma_57;
                g_62[25] = ((lg_61[25] > 0.0f) ? grad_t * w_h_88 : 0.0f);
                float w_h_89 = weights[q_row0 + 58];
                float _max_59 = max_noftz(lg_61[26], 0.0f);
                float _fma_58 = __fmaf_rn(grad_t, _max_59, dw[58]);
                dw[58] = _fma_58;
                g_62[26] = ((lg_61[26] > 0.0f) ? grad_t * w_h_89 : 0.0f);
                float w_h_90 = weights[q_row0 + 59];
                float _max_60 = max_noftz(lg_61[27], 0.0f);
                float _fma_59 = __fmaf_rn(grad_t, _max_60, dw[59]);
                dw[59] = _fma_59;
                g_62[27] = ((lg_61[27] > 0.0f) ? grad_t * w_h_90 : 0.0f);
                float w_h_91 = weights[q_row0 + 60];
                float _max_61 = max_noftz(lg_61[28], 0.0f);
                float _fma_60 = __fmaf_rn(grad_t, _max_61, dw[60]);
                dw[60] = _fma_60;
                g_62[28] = ((lg_61[28] > 0.0f) ? grad_t * w_h_91 : 0.0f);
                float w_h_92 = weights[q_row0 + 61];
                float _max_62 = max_noftz(lg_61[29], 0.0f);
                float _fma_61 = __fmaf_rn(grad_t, _max_62, dw[61]);
                dw[61] = _fma_61;
                g_62[29] = ((lg_61[29] > 0.0f) ? grad_t * w_h_92 : 0.0f);
                float w_h_93 = weights[q_row0 + 62];
                float _max_63 = max_noftz(lg_61[30], 0.0f);
                float _fma_62 = __fmaf_rn(grad_t, _max_63, dw[62]);
                dw[62] = _fma_62;
                g_62[30] = ((lg_61[30] > 0.0f) ? grad_t * w_h_93 : 0.0f);
                float w_h_94 = weights[q_row0 + 63];
                float _max_64 = max_noftz(lg_61[31], 0.0f);
                float _fma_63 = __fmaf_rn(grad_t, _max_64, dw[63]);
                dw[63] = _fma_63;
                g_62[31] = ((lg_61[31] > 0.0f) ? grad_t * w_h_94 : 0.0f);
                uint32_t g_62_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(g_62[_lp*2 + 0], g_62[_lp*2+1 + 0]));
                    g_62_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                float lo_95[32];
                float _cvt_f32_bf16_32;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_32) : "h"((uint16_t)(g_62_bf16[0] & 65535)));
                float hi_e_96 = _cvt_f32_bf16_32;
                float _cvt_f32_bf16_33;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_33) : "h"((uint16_t)(g_62_bf16[0] >> 16)));
                float hi_o_97 = _cvt_f32_bf16_33;
                lo_95[0] = g_62[0] - hi_e_96;
                lo_95[1] = g_62[1] - hi_o_97;
                float _cvt_f32_bf16_34;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_34) : "h"((uint16_t)(g_62_bf16[1] & 65535)));
                float hi_e_98 = _cvt_f32_bf16_34;
                float _cvt_f32_bf16_35;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_35) : "h"((uint16_t)(g_62_bf16[1] >> 16)));
                float hi_o_99 = _cvt_f32_bf16_35;
                lo_95[2] = g_62[2] - hi_e_98;
                lo_95[3] = g_62[3] - hi_o_99;
                float _cvt_f32_bf16_36;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_36) : "h"((uint16_t)(g_62_bf16[2] & 65535)));
                float hi_e_100 = _cvt_f32_bf16_36;
                float _cvt_f32_bf16_37;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_37) : "h"((uint16_t)(g_62_bf16[2] >> 16)));
                float hi_o_101 = _cvt_f32_bf16_37;
                lo_95[4] = g_62[4] - hi_e_100;
                lo_95[5] = g_62[5] - hi_o_101;
                float _cvt_f32_bf16_38;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_38) : "h"((uint16_t)(g_62_bf16[3] & 65535)));
                float hi_e_102 = _cvt_f32_bf16_38;
                float _cvt_f32_bf16_39;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_39) : "h"((uint16_t)(g_62_bf16[3] >> 16)));
                float hi_o_103 = _cvt_f32_bf16_39;
                lo_95[6] = g_62[6] - hi_e_102;
                lo_95[7] = g_62[7] - hi_o_103;
                float _cvt_f32_bf16_40;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_40) : "h"((uint16_t)(g_62_bf16[4] & 65535)));
                float hi_e_104 = _cvt_f32_bf16_40;
                float _cvt_f32_bf16_41;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_41) : "h"((uint16_t)(g_62_bf16[4] >> 16)));
                float hi_o_105 = _cvt_f32_bf16_41;
                lo_95[8] = g_62[8] - hi_e_104;
                lo_95[9] = g_62[9] - hi_o_105;
                float _cvt_f32_bf16_42;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_42) : "h"((uint16_t)(g_62_bf16[5] & 65535)));
                float hi_e_106 = _cvt_f32_bf16_42;
                float _cvt_f32_bf16_43;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_43) : "h"((uint16_t)(g_62_bf16[5] >> 16)));
                float hi_o_107 = _cvt_f32_bf16_43;
                lo_95[10] = g_62[10] - hi_e_106;
                lo_95[11] = g_62[11] - hi_o_107;
                float _cvt_f32_bf16_44;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_44) : "h"((uint16_t)(g_62_bf16[6] & 65535)));
                float hi_e_108 = _cvt_f32_bf16_44;
                float _cvt_f32_bf16_45;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_45) : "h"((uint16_t)(g_62_bf16[6] >> 16)));
                float hi_o_109 = _cvt_f32_bf16_45;
                lo_95[12] = g_62[12] - hi_e_108;
                lo_95[13] = g_62[13] - hi_o_109;
                float _cvt_f32_bf16_46;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_46) : "h"((uint16_t)(g_62_bf16[7] & 65535)));
                float hi_e_110 = _cvt_f32_bf16_46;
                float _cvt_f32_bf16_47;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_47) : "h"((uint16_t)(g_62_bf16[7] >> 16)));
                float hi_o_111 = _cvt_f32_bf16_47;
                lo_95[14] = g_62[14] - hi_e_110;
                lo_95[15] = g_62[15] - hi_o_111;
                float _cvt_f32_bf16_48;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_48) : "h"((uint16_t)(g_62_bf16[8] & 65535)));
                float hi_e_112 = _cvt_f32_bf16_48;
                float _cvt_f32_bf16_49;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_49) : "h"((uint16_t)(g_62_bf16[8] >> 16)));
                float hi_o_113 = _cvt_f32_bf16_49;
                lo_95[16] = g_62[16] - hi_e_112;
                lo_95[17] = g_62[17] - hi_o_113;
                float _cvt_f32_bf16_50;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_50) : "h"((uint16_t)(g_62_bf16[9] & 65535)));
                float hi_e_114 = _cvt_f32_bf16_50;
                float _cvt_f32_bf16_51;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_51) : "h"((uint16_t)(g_62_bf16[9] >> 16)));
                float hi_o_115 = _cvt_f32_bf16_51;
                lo_95[18] = g_62[18] - hi_e_114;
                lo_95[19] = g_62[19] - hi_o_115;
                float _cvt_f32_bf16_52;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_52) : "h"((uint16_t)(g_62_bf16[10] & 65535)));
                float hi_e_116 = _cvt_f32_bf16_52;
                float _cvt_f32_bf16_53;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_53) : "h"((uint16_t)(g_62_bf16[10] >> 16)));
                float hi_o_117 = _cvt_f32_bf16_53;
                lo_95[20] = g_62[20] - hi_e_116;
                lo_95[21] = g_62[21] - hi_o_117;
                float _cvt_f32_bf16_54;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_54) : "h"((uint16_t)(g_62_bf16[11] & 65535)));
                float hi_e_118 = _cvt_f32_bf16_54;
                float _cvt_f32_bf16_55;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_55) : "h"((uint16_t)(g_62_bf16[11] >> 16)));
                float hi_o_119 = _cvt_f32_bf16_55;
                lo_95[22] = g_62[22] - hi_e_118;
                lo_95[23] = g_62[23] - hi_o_119;
                float _cvt_f32_bf16_56;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_56) : "h"((uint16_t)(g_62_bf16[12] & 65535)));
                float hi_e_120 = _cvt_f32_bf16_56;
                float _cvt_f32_bf16_57;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_57) : "h"((uint16_t)(g_62_bf16[12] >> 16)));
                float hi_o_121 = _cvt_f32_bf16_57;
                lo_95[24] = g_62[24] - hi_e_120;
                lo_95[25] = g_62[25] - hi_o_121;
                float _cvt_f32_bf16_58;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_58) : "h"((uint16_t)(g_62_bf16[13] & 65535)));
                float hi_e_122 = _cvt_f32_bf16_58;
                float _cvt_f32_bf16_59;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_59) : "h"((uint16_t)(g_62_bf16[13] >> 16)));
                float hi_o_123 = _cvt_f32_bf16_59;
                lo_95[26] = g_62[26] - hi_e_122;
                lo_95[27] = g_62[27] - hi_o_123;
                float _cvt_f32_bf16_60;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_60) : "h"((uint16_t)(g_62_bf16[14] & 65535)));
                float hi_e_124 = _cvt_f32_bf16_60;
                float _cvt_f32_bf16_61;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_61) : "h"((uint16_t)(g_62_bf16[14] >> 16)));
                float hi_o_125 = _cvt_f32_bf16_61;
                lo_95[28] = g_62[28] - hi_e_124;
                lo_95[29] = g_62[29] - hi_o_125;
                float _cvt_f32_bf16_62;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_62) : "h"((uint16_t)(g_62_bf16[15] & 65535)));
                float hi_e_126 = _cvt_f32_bf16_62;
                float _cvt_f32_bf16_63;
                asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_63) : "h"((uint16_t)(g_62_bf16[15] >> 16)));
                float hi_o_127 = _cvt_f32_bf16_63;
                lo_95[30] = g_62[30] - hi_e_126;
                lo_95[31] = g_62[31] - hi_o_127;
                {
                    __nv_bfloat16 _bval_72 = __float2bfloat16_rn(g_62[0]);
                    uint16_t _bits_72 = *(uint16_t*)&_bval_72;
                    uint32_t _addr_72 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 4096 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_72), "h"(_bits_72) : "memory");
                }
                {
                    __nv_bfloat16 _bval_73 = __float2bfloat16_rn(lo_95[0]);
                    uint16_t _bits_73 = *(uint16_t*)&_bval_73;
                    uint32_t _addr_73 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 4096 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_73), "h"(_bits_73) : "memory");
                }
                {
                    __nv_bfloat16 _bval_74 = __float2bfloat16_rn(g_62[1]);
                    uint16_t _bits_74 = *(uint16_t*)&_bval_74;
                    uint32_t _addr_74 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 4224 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_74), "h"(_bits_74) : "memory");
                }
                {
                    __nv_bfloat16 _bval_75 = __float2bfloat16_rn(lo_95[1]);
                    uint16_t _bits_75 = *(uint16_t*)&_bval_75;
                    uint32_t _addr_75 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 4224 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_75), "h"(_bits_75) : "memory");
                }
                {
                    __nv_bfloat16 _bval_76 = __float2bfloat16_rn(g_62[2]);
                    uint16_t _bits_76 = *(uint16_t*)&_bval_76;
                    uint32_t _addr_76 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 4352 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_76), "h"(_bits_76) : "memory");
                }
                {
                    __nv_bfloat16 _bval_77 = __float2bfloat16_rn(lo_95[2]);
                    uint16_t _bits_77 = *(uint16_t*)&_bval_77;
                    uint32_t _addr_77 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 4352 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_77), "h"(_bits_77) : "memory");
                }
                {
                    __nv_bfloat16 _bval_78 = __float2bfloat16_rn(g_62[3]);
                    uint16_t _bits_78 = *(uint16_t*)&_bval_78;
                    uint32_t _addr_78 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 4480 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_78), "h"(_bits_78) : "memory");
                }
                {
                    __nv_bfloat16 _bval_79 = __float2bfloat16_rn(lo_95[3]);
                    uint16_t _bits_79 = *(uint16_t*)&_bval_79;
                    uint32_t _addr_79 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 4480 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_79), "h"(_bits_79) : "memory");
                }
                {
                    __nv_bfloat16 _bval_80 = __float2bfloat16_rn(g_62[4]);
                    uint16_t _bits_80 = *(uint16_t*)&_bval_80;
                    uint32_t _addr_80 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 4608 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_80), "h"(_bits_80) : "memory");
                }
                {
                    __nv_bfloat16 _bval_81 = __float2bfloat16_rn(lo_95[4]);
                    uint16_t _bits_81 = *(uint16_t*)&_bval_81;
                    uint32_t _addr_81 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 4608 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_81), "h"(_bits_81) : "memory");
                }
                {
                    __nv_bfloat16 _bval_82 = __float2bfloat16_rn(g_62[5]);
                    uint16_t _bits_82 = *(uint16_t*)&_bval_82;
                    uint32_t _addr_82 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 4736 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_82), "h"(_bits_82) : "memory");
                }
                {
                    __nv_bfloat16 _bval_83 = __float2bfloat16_rn(lo_95[5]);
                    uint16_t _bits_83 = *(uint16_t*)&_bval_83;
                    uint32_t _addr_83 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 4736 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_83), "h"(_bits_83) : "memory");
                }
                {
                    __nv_bfloat16 _bval_84 = __float2bfloat16_rn(g_62[6]);
                    uint16_t _bits_84 = *(uint16_t*)&_bval_84;
                    uint32_t _addr_84 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 4864 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_84), "h"(_bits_84) : "memory");
                }
                {
                    __nv_bfloat16 _bval_85 = __float2bfloat16_rn(lo_95[6]);
                    uint16_t _bits_85 = *(uint16_t*)&_bval_85;
                    uint32_t _addr_85 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 4864 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_85), "h"(_bits_85) : "memory");
                }
                {
                    __nv_bfloat16 _bval_86 = __float2bfloat16_rn(g_62[7]);
                    uint16_t _bits_86 = *(uint16_t*)&_bval_86;
                    uint32_t _addr_86 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 4992 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_86), "h"(_bits_86) : "memory");
                }
                {
                    __nv_bfloat16 _bval_87 = __float2bfloat16_rn(lo_95[7]);
                    uint16_t _bits_87 = *(uint16_t*)&_bval_87;
                    uint32_t _addr_87 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 4992 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_87), "h"(_bits_87) : "memory");
                }
                {
                    __nv_bfloat16 _bval_88 = __float2bfloat16_rn(g_62[8]);
                    uint16_t _bits_88 = *(uint16_t*)&_bval_88;
                    uint32_t _addr_88 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 5120 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_88), "h"(_bits_88) : "memory");
                }
                {
                    __nv_bfloat16 _bval_89 = __float2bfloat16_rn(lo_95[8]);
                    uint16_t _bits_89 = *(uint16_t*)&_bval_89;
                    uint32_t _addr_89 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 5120 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_89), "h"(_bits_89) : "memory");
                }
                {
                    __nv_bfloat16 _bval_90 = __float2bfloat16_rn(g_62[9]);
                    uint16_t _bits_90 = *(uint16_t*)&_bval_90;
                    uint32_t _addr_90 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 5248 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_90), "h"(_bits_90) : "memory");
                }
                {
                    __nv_bfloat16 _bval_91 = __float2bfloat16_rn(lo_95[9]);
                    uint16_t _bits_91 = *(uint16_t*)&_bval_91;
                    uint32_t _addr_91 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 5248 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_91), "h"(_bits_91) : "memory");
                }
                {
                    __nv_bfloat16 _bval_92 = __float2bfloat16_rn(g_62[10]);
                    uint16_t _bits_92 = *(uint16_t*)&_bval_92;
                    uint32_t _addr_92 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 5376 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_92), "h"(_bits_92) : "memory");
                }
                {
                    __nv_bfloat16 _bval_93 = __float2bfloat16_rn(lo_95[10]);
                    uint16_t _bits_93 = *(uint16_t*)&_bval_93;
                    uint32_t _addr_93 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 5376 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_93), "h"(_bits_93) : "memory");
                }
                {
                    __nv_bfloat16 _bval_94 = __float2bfloat16_rn(g_62[11]);
                    uint16_t _bits_94 = *(uint16_t*)&_bval_94;
                    uint32_t _addr_94 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 5504 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_94), "h"(_bits_94) : "memory");
                }
                {
                    __nv_bfloat16 _bval_95 = __float2bfloat16_rn(lo_95[11]);
                    uint16_t _bits_95 = *(uint16_t*)&_bval_95;
                    uint32_t _addr_95 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 5504 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_95), "h"(_bits_95) : "memory");
                }
                {
                    __nv_bfloat16 _bval_96 = __float2bfloat16_rn(g_62[12]);
                    uint16_t _bits_96 = *(uint16_t*)&_bval_96;
                    uint32_t _addr_96 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 5632 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_96), "h"(_bits_96) : "memory");
                }
                {
                    __nv_bfloat16 _bval_97 = __float2bfloat16_rn(lo_95[12]);
                    uint16_t _bits_97 = *(uint16_t*)&_bval_97;
                    uint32_t _addr_97 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 5632 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_97), "h"(_bits_97) : "memory");
                }
                {
                    __nv_bfloat16 _bval_98 = __float2bfloat16_rn(g_62[13]);
                    uint16_t _bits_98 = *(uint16_t*)&_bval_98;
                    uint32_t _addr_98 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 5760 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_98), "h"(_bits_98) : "memory");
                }
                {
                    __nv_bfloat16 _bval_99 = __float2bfloat16_rn(lo_95[13]);
                    uint16_t _bits_99 = *(uint16_t*)&_bval_99;
                    uint32_t _addr_99 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 5760 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_99), "h"(_bits_99) : "memory");
                }
                {
                    __nv_bfloat16 _bval_100 = __float2bfloat16_rn(g_62[14]);
                    uint16_t _bits_100 = *(uint16_t*)&_bval_100;
                    uint32_t _addr_100 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 5888 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_100), "h"(_bits_100) : "memory");
                }
                {
                    __nv_bfloat16 _bval_101 = __float2bfloat16_rn(lo_95[14]);
                    uint16_t _bits_101 = *(uint16_t*)&_bval_101;
                    uint32_t _addr_101 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 5888 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_101), "h"(_bits_101) : "memory");
                }
                {
                    __nv_bfloat16 _bval_102 = __float2bfloat16_rn(g_62[15]);
                    uint16_t _bits_102 = *(uint16_t*)&_bval_102;
                    uint32_t _addr_102 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 6016 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_102), "h"(_bits_102) : "memory");
                }
                {
                    __nv_bfloat16 _bval_103 = __float2bfloat16_rn(lo_95[15]);
                    uint16_t _bits_103 = *(uint16_t*)&_bval_103;
                    uint32_t _addr_103 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 6016 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_103), "h"(_bits_103) : "memory");
                }
                {
                    __nv_bfloat16 _bval_104 = __float2bfloat16_rn(g_62[16]);
                    uint16_t _bits_104 = *(uint16_t*)&_bval_104;
                    uint32_t _addr_104 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 6144 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_104), "h"(_bits_104) : "memory");
                }
                {
                    __nv_bfloat16 _bval_105 = __float2bfloat16_rn(lo_95[16]);
                    uint16_t _bits_105 = *(uint16_t*)&_bval_105;
                    uint32_t _addr_105 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 6144 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_105), "h"(_bits_105) : "memory");
                }
                {
                    __nv_bfloat16 _bval_106 = __float2bfloat16_rn(g_62[17]);
                    uint16_t _bits_106 = *(uint16_t*)&_bval_106;
                    uint32_t _addr_106 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 6272 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_106), "h"(_bits_106) : "memory");
                }
                {
                    __nv_bfloat16 _bval_107 = __float2bfloat16_rn(lo_95[17]);
                    uint16_t _bits_107 = *(uint16_t*)&_bval_107;
                    uint32_t _addr_107 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 6272 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_107), "h"(_bits_107) : "memory");
                }
                {
                    __nv_bfloat16 _bval_108 = __float2bfloat16_rn(g_62[18]);
                    uint16_t _bits_108 = *(uint16_t*)&_bval_108;
                    uint32_t _addr_108 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 6400 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_108), "h"(_bits_108) : "memory");
                }
                {
                    __nv_bfloat16 _bval_109 = __float2bfloat16_rn(lo_95[18]);
                    uint16_t _bits_109 = *(uint16_t*)&_bval_109;
                    uint32_t _addr_109 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 6400 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_109), "h"(_bits_109) : "memory");
                }
                {
                    __nv_bfloat16 _bval_110 = __float2bfloat16_rn(g_62[19]);
                    uint16_t _bits_110 = *(uint16_t*)&_bval_110;
                    uint32_t _addr_110 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 6528 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_110), "h"(_bits_110) : "memory");
                }
                {
                    __nv_bfloat16 _bval_111 = __float2bfloat16_rn(lo_95[19]);
                    uint16_t _bits_111 = *(uint16_t*)&_bval_111;
                    uint32_t _addr_111 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 6528 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_111), "h"(_bits_111) : "memory");
                }
                {
                    __nv_bfloat16 _bval_112 = __float2bfloat16_rn(g_62[20]);
                    uint16_t _bits_112 = *(uint16_t*)&_bval_112;
                    uint32_t _addr_112 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 6656 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_112), "h"(_bits_112) : "memory");
                }
                {
                    __nv_bfloat16 _bval_113 = __float2bfloat16_rn(lo_95[20]);
                    uint16_t _bits_113 = *(uint16_t*)&_bval_113;
                    uint32_t _addr_113 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 6656 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_113), "h"(_bits_113) : "memory");
                }
                {
                    __nv_bfloat16 _bval_114 = __float2bfloat16_rn(g_62[21]);
                    uint16_t _bits_114 = *(uint16_t*)&_bval_114;
                    uint32_t _addr_114 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 6784 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_114), "h"(_bits_114) : "memory");
                }
                {
                    __nv_bfloat16 _bval_115 = __float2bfloat16_rn(lo_95[21]);
                    uint16_t _bits_115 = *(uint16_t*)&_bval_115;
                    uint32_t _addr_115 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 6784 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_115), "h"(_bits_115) : "memory");
                }
                {
                    __nv_bfloat16 _bval_116 = __float2bfloat16_rn(g_62[22]);
                    uint16_t _bits_116 = *(uint16_t*)&_bval_116;
                    uint32_t _addr_116 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 6912 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_116), "h"(_bits_116) : "memory");
                }
                {
                    __nv_bfloat16 _bval_117 = __float2bfloat16_rn(lo_95[22]);
                    uint16_t _bits_117 = *(uint16_t*)&_bval_117;
                    uint32_t _addr_117 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 6912 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_117), "h"(_bits_117) : "memory");
                }
                {
                    __nv_bfloat16 _bval_118 = __float2bfloat16_rn(g_62[23]);
                    uint16_t _bits_118 = *(uint16_t*)&_bval_118;
                    uint32_t _addr_118 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 7040 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_118), "h"(_bits_118) : "memory");
                }
                {
                    __nv_bfloat16 _bval_119 = __float2bfloat16_rn(lo_95[23]);
                    uint16_t _bits_119 = *(uint16_t*)&_bval_119;
                    uint32_t _addr_119 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 7040 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_119), "h"(_bits_119) : "memory");
                }
                {
                    __nv_bfloat16 _bval_120 = __float2bfloat16_rn(g_62[24]);
                    uint16_t _bits_120 = *(uint16_t*)&_bval_120;
                    uint32_t _addr_120 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 7168 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_120), "h"(_bits_120) : "memory");
                }
                {
                    __nv_bfloat16 _bval_121 = __float2bfloat16_rn(lo_95[24]);
                    uint16_t _bits_121 = *(uint16_t*)&_bval_121;
                    uint32_t _addr_121 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 7168 + gx[0]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_121), "h"(_bits_121) : "memory");
                }
                {
                    __nv_bfloat16 _bval_122 = __float2bfloat16_rn(g_62[25]);
                    uint16_t _bits_122 = *(uint16_t*)&_bval_122;
                    uint32_t _addr_122 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 7296 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_122), "h"(_bits_122) : "memory");
                }
                {
                    __nv_bfloat16 _bval_123 = __float2bfloat16_rn(lo_95[25]);
                    uint16_t _bits_123 = *(uint16_t*)&_bval_123;
                    uint32_t _addr_123 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 7296 + gx[1]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_123), "h"(_bits_123) : "memory");
                }
                {
                    __nv_bfloat16 _bval_124 = __float2bfloat16_rn(g_62[26]);
                    uint16_t _bits_124 = *(uint16_t*)&_bval_124;
                    uint32_t _addr_124 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 7424 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_124), "h"(_bits_124) : "memory");
                }
                {
                    __nv_bfloat16 _bval_125 = __float2bfloat16_rn(lo_95[26]);
                    uint16_t _bits_125 = *(uint16_t*)&_bval_125;
                    uint32_t _addr_125 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 7424 + gx[2]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_125), "h"(_bits_125) : "memory");
                }
                {
                    __nv_bfloat16 _bval_126 = __float2bfloat16_rn(g_62[27]);
                    uint16_t _bits_126 = *(uint16_t*)&_bval_126;
                    uint32_t _addr_126 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 7552 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_126), "h"(_bits_126) : "memory");
                }
                {
                    __nv_bfloat16 _bval_127 = __float2bfloat16_rn(lo_95[27]);
                    uint16_t _bits_127 = *(uint16_t*)&_bval_127;
                    uint32_t _addr_127 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 7552 + gx[3]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_127), "h"(_bits_127) : "memory");
                }
                {
                    __nv_bfloat16 _bval_128 = __float2bfloat16_rn(g_62[28]);
                    uint16_t _bits_128 = *(uint16_t*)&_bval_128;
                    uint32_t _addr_128 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 7680 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_128), "h"(_bits_128) : "memory");
                }
                {
                    __nv_bfloat16 _bval_129 = __float2bfloat16_rn(lo_95[28]);
                    uint16_t _bits_129 = *(uint16_t*)&_bval_129;
                    uint32_t _addr_129 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 7680 + gx[4]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_129), "h"(_bits_129) : "memory");
                }
                {
                    __nv_bfloat16 _bval_130 = __float2bfloat16_rn(g_62[29]);
                    uint16_t _bits_130 = *(uint16_t*)&_bval_130;
                    uint32_t _addr_130 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 7808 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_130), "h"(_bits_130) : "memory");
                }
                {
                    __nv_bfloat16 _bval_131 = __float2bfloat16_rn(lo_95[29]);
                    uint16_t _bits_131 = *(uint16_t*)&_bval_131;
                    uint32_t _addr_131 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 7808 + gx[5]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_131), "h"(_bits_131) : "memory");
                }
                {
                    __nv_bfloat16 _bval_132 = __float2bfloat16_rn(g_62[30]);
                    uint16_t _bits_132 = *(uint16_t*)&_bval_132;
                    uint32_t _addr_132 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 7936 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_132), "h"(_bits_132) : "memory");
                }
                {
                    __nv_bfloat16 _bval_133 = __float2bfloat16_rn(lo_95[30]);
                    uint16_t _bits_133 = *(uint16_t*)&_bval_133;
                    uint32_t _addr_133 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 7936 + gx[6]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_133), "h"(_bits_133) : "memory");
                }
                {
                    __nv_bfloat16 _bval_134 = __float2bfloat16_rn(g_62[31]);
                    uint16_t _bits_134 = *(uint16_t*)&_bval_134;
                    uint32_t _addr_134 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 8064 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_134), "h"(_bits_134) : "memory");
                }
                {
                    __nv_bfloat16 _bval_135 = __float2bfloat16_rn(lo_95[31]);
                    uint16_t _bits_135 = *(uint16_t*)&_bval_135;
                    uint32_t _addr_135 = static_cast<uint32_t>(gt_raw_addr + (unsigned int)(gbase + 16384 + 8064 + gx[7]));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_135), "h"(_bits_135) : "memory");
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(l_free_addr + (par_c) * 8);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(g_full_addr + (par_c) * 8);
                if (i > 0) {
                    int pprev = i - 1 & 1;
                    int puse = (i - 1) / 2 & 1;
                    mbarrier_wait(dk_full_addr + (pprev) * 8, puse);
                    long long prow = ((long long)row_c * (long long)topk + (long long)((i - 1) * 128 + tid_1)) * 128;
                    float x[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3]), "=f"(x[4]), "=f"(x[5]), "=f"(x[6]), "=f"(x[7]), "=f"(x[8]), "=f"(x[9]), "=f"(x[10]), "=f"(x[11]), "=f"(x[12]), "=f"(x[13]), "=f"(x[14]), "=f"(x[15]), "=f"(x[16]), "=f"(x[17]), "=f"(x[18]), "=f"(x[19]), "=f"(x[20]), "=f"(x[21]), "=f"(x[22]), "=f"(x[23]), "=f"(x[24]), "=f"(x[25]), "=f"(x[26]), "=f"(x[27]), "=f"(x[28]), "=f"(x[29]), "=f"(x[30]), "=f"(x[31])
                        : "r"(tmem_tmem + 256 + pprev * 128 + lane_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (idx_prev >= 0) {
                        {
                            float4 _v4 = make_float4(x[0 + 0], x[0 + 1], x[0 + 2], x[0 + 3]);
                            *reinterpret_cast<float4*>(partial + prow) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x[4 + 0], x[4 + 1], x[4 + 2], x[4 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 4) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x[8 + 0], x[8 + 1], x[8 + 2], x[8 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 8) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x[12 + 0], x[12 + 1], x[12 + 2], x[12 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 12) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x[16 + 0], x[16 + 1], x[16 + 2], x[16 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 16) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x[20 + 0], x[20 + 1], x[20 + 2], x[20 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 20) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x[24 + 0], x[24 + 1], x[24 + 2], x[24 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 24) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x[28 + 0], x[28 + 1], x[28 + 2], x[28 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 28) = _v4;
                        }
                    }
                    float x_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(x_0[0]), "=f"(x_0[1]), "=f"(x_0[2]), "=f"(x_0[3]), "=f"(x_0[4]), "=f"(x_0[5]), "=f"(x_0[6]), "=f"(x_0[7]), "=f"(x_0[8]), "=f"(x_0[9]), "=f"(x_0[10]), "=f"(x_0[11]), "=f"(x_0[12]), "=f"(x_0[13]), "=f"(x_0[14]), "=f"(x_0[15]), "=f"(x_0[16]), "=f"(x_0[17]), "=f"(x_0[18]), "=f"(x_0[19]), "=f"(x_0[20]), "=f"(x_0[21]), "=f"(x_0[22]), "=f"(x_0[23]), "=f"(x_0[24]), "=f"(x_0[25]), "=f"(x_0[26]), "=f"(x_0[27]), "=f"(x_0[28]), "=f"(x_0[29]), "=f"(x_0[30]), "=f"(x_0[31])
                        : "r"(tmem_tmem + 256 + pprev * 128 + 32 + lane_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (idx_prev >= 0) {
                        {
                            float4 _v4 = make_float4(x_0[0 + 0], x_0[0 + 1], x_0[0 + 2], x_0[0 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 32) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_0[4 + 0], x_0[4 + 1], x_0[4 + 2], x_0[4 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 32 + 4) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_0[8 + 0], x_0[8 + 1], x_0[8 + 2], x_0[8 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 32 + 8) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_0[12 + 0], x_0[12 + 1], x_0[12 + 2], x_0[12 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 32 + 12) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_0[16 + 0], x_0[16 + 1], x_0[16 + 2], x_0[16 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 32 + 16) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_0[20 + 0], x_0[20 + 1], x_0[20 + 2], x_0[20 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 32 + 20) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_0[24 + 0], x_0[24 + 1], x_0[24 + 2], x_0[24 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 32 + 24) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_0[28 + 0], x_0[28 + 1], x_0[28 + 2], x_0[28 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 32 + 28) = _v4;
                        }
                    }
                    float x_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(x_1[0]), "=f"(x_1[1]), "=f"(x_1[2]), "=f"(x_1[3]), "=f"(x_1[4]), "=f"(x_1[5]), "=f"(x_1[6]), "=f"(x_1[7]), "=f"(x_1[8]), "=f"(x_1[9]), "=f"(x_1[10]), "=f"(x_1[11]), "=f"(x_1[12]), "=f"(x_1[13]), "=f"(x_1[14]), "=f"(x_1[15]), "=f"(x_1[16]), "=f"(x_1[17]), "=f"(x_1[18]), "=f"(x_1[19]), "=f"(x_1[20]), "=f"(x_1[21]), "=f"(x_1[22]), "=f"(x_1[23]), "=f"(x_1[24]), "=f"(x_1[25]), "=f"(x_1[26]), "=f"(x_1[27]), "=f"(x_1[28]), "=f"(x_1[29]), "=f"(x_1[30]), "=f"(x_1[31])
                        : "r"(tmem_tmem + 256 + pprev * 128 + 64 + lane_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (idx_prev >= 0) {
                        {
                            float4 _v4 = make_float4(x_1[0 + 0], x_1[0 + 1], x_1[0 + 2], x_1[0 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 64) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_1[4 + 0], x_1[4 + 1], x_1[4 + 2], x_1[4 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 64 + 4) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_1[8 + 0], x_1[8 + 1], x_1[8 + 2], x_1[8 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 64 + 8) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_1[12 + 0], x_1[12 + 1], x_1[12 + 2], x_1[12 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 64 + 12) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_1[16 + 0], x_1[16 + 1], x_1[16 + 2], x_1[16 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 64 + 16) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_1[20 + 0], x_1[20 + 1], x_1[20 + 2], x_1[20 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 64 + 20) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_1[24 + 0], x_1[24 + 1], x_1[24 + 2], x_1[24 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 64 + 24) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_1[28 + 0], x_1[28 + 1], x_1[28 + 2], x_1[28 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 64 + 28) = _v4;
                        }
                    }
                    float x_2[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(x_2[0]), "=f"(x_2[1]), "=f"(x_2[2]), "=f"(x_2[3]), "=f"(x_2[4]), "=f"(x_2[5]), "=f"(x_2[6]), "=f"(x_2[7]), "=f"(x_2[8]), "=f"(x_2[9]), "=f"(x_2[10]), "=f"(x_2[11]), "=f"(x_2[12]), "=f"(x_2[13]), "=f"(x_2[14]), "=f"(x_2[15]), "=f"(x_2[16]), "=f"(x_2[17]), "=f"(x_2[18]), "=f"(x_2[19]), "=f"(x_2[20]), "=f"(x_2[21]), "=f"(x_2[22]), "=f"(x_2[23]), "=f"(x_2[24]), "=f"(x_2[25]), "=f"(x_2[26]), "=f"(x_2[27]), "=f"(x_2[28]), "=f"(x_2[29]), "=f"(x_2[30]), "=f"(x_2[31])
                        : "r"(tmem_tmem + 256 + pprev * 128 + 96 + lane_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (idx_prev >= 0) {
                        {
                            float4 _v4 = make_float4(x_2[0 + 0], x_2[0 + 1], x_2[0 + 2], x_2[0 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 96) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_2[4 + 0], x_2[4 + 1], x_2[4 + 2], x_2[4 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 96 + 4) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_2[8 + 0], x_2[8 + 1], x_2[8 + 2], x_2[8 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 96 + 8) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_2[12 + 0], x_2[12 + 1], x_2[12 + 2], x_2[12 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 96 + 12) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_2[16 + 0], x_2[16 + 1], x_2[16 + 2], x_2[16 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 96 + 16) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_2[20 + 0], x_2[20 + 1], x_2[20 + 2], x_2[20 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 96 + 20) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_2[24 + 0], x_2[24 + 1], x_2[24 + 2], x_2[24 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 96 + 24) = _v4;
                        }
                        {
                            float4 _v4 = make_float4(x_2[28 + 0], x_2[28 + 1], x_2[28 + 2], x_2[28 + 3]);
                            *reinterpret_cast<float4*>(partial + prow + 96 + 28) = _v4;
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(dk_free_addr + (pprev) * 8);
                }
                idx_prev = idx_t;
            }
            int lpar = blocks_c - 1 & 1;
            int luse = (blocks_c - 1) / 2 & 1;
            mbarrier_wait(dk_full_addr + (lpar) * 8, luse);
            long long lrow = ((long long)row_c * (long long)topk + (long long)((blocks_c - 1) * 128 + tid_1)) * 128;
            float xl[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(xl[0]), "=f"(xl[1]), "=f"(xl[2]), "=f"(xl[3]), "=f"(xl[4]), "=f"(xl[5]), "=f"(xl[6]), "=f"(xl[7]), "=f"(xl[8]), "=f"(xl[9]), "=f"(xl[10]), "=f"(xl[11]), "=f"(xl[12]), "=f"(xl[13]), "=f"(xl[14]), "=f"(xl[15]), "=f"(xl[16]), "=f"(xl[17]), "=f"(xl[18]), "=f"(xl[19]), "=f"(xl[20]), "=f"(xl[21]), "=f"(xl[22]), "=f"(xl[23]), "=f"(xl[24]), "=f"(xl[25]), "=f"(xl[26]), "=f"(xl[27]), "=f"(xl[28]), "=f"(xl[29]), "=f"(xl[30]), "=f"(xl[31])
                : "r"(tmem_tmem + 256 + lpar * 128 + lane_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            if (idx_prev >= 0) {
                {
                    float4 _v4 = make_float4(xl[0 + 0], xl[0 + 1], xl[0 + 2], xl[0 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl[4 + 0], xl[4 + 1], xl[4 + 2], xl[4 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 4) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl[8 + 0], xl[8 + 1], xl[8 + 2], xl[8 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 8) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl[12 + 0], xl[12 + 1], xl[12 + 2], xl[12 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 12) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl[16 + 0], xl[16 + 1], xl[16 + 2], xl[16 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 16) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl[20 + 0], xl[20 + 1], xl[20 + 2], xl[20 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 20) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl[24 + 0], xl[24 + 1], xl[24 + 2], xl[24 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 24) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl[28 + 0], xl[28 + 1], xl[28 + 2], xl[28 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 28) = _v4;
                }
            }
            float xl_28[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(xl_28[0]), "=f"(xl_28[1]), "=f"(xl_28[2]), "=f"(xl_28[3]), "=f"(xl_28[4]), "=f"(xl_28[5]), "=f"(xl_28[6]), "=f"(xl_28[7]), "=f"(xl_28[8]), "=f"(xl_28[9]), "=f"(xl_28[10]), "=f"(xl_28[11]), "=f"(xl_28[12]), "=f"(xl_28[13]), "=f"(xl_28[14]), "=f"(xl_28[15]), "=f"(xl_28[16]), "=f"(xl_28[17]), "=f"(xl_28[18]), "=f"(xl_28[19]), "=f"(xl_28[20]), "=f"(xl_28[21]), "=f"(xl_28[22]), "=f"(xl_28[23]), "=f"(xl_28[24]), "=f"(xl_28[25]), "=f"(xl_28[26]), "=f"(xl_28[27]), "=f"(xl_28[28]), "=f"(xl_28[29]), "=f"(xl_28[30]), "=f"(xl_28[31])
                : "r"(tmem_tmem + 256 + lpar * 128 + 32 + lane_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            if (idx_prev >= 0) {
                {
                    float4 _v4 = make_float4(xl_28[0 + 0], xl_28[0 + 1], xl_28[0 + 2], xl_28[0 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 32) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_28[4 + 0], xl_28[4 + 1], xl_28[4 + 2], xl_28[4 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 32 + 4) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_28[8 + 0], xl_28[8 + 1], xl_28[8 + 2], xl_28[8 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 32 + 8) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_28[12 + 0], xl_28[12 + 1], xl_28[12 + 2], xl_28[12 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 32 + 12) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_28[16 + 0], xl_28[16 + 1], xl_28[16 + 2], xl_28[16 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 32 + 16) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_28[20 + 0], xl_28[20 + 1], xl_28[20 + 2], xl_28[20 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 32 + 20) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_28[24 + 0], xl_28[24 + 1], xl_28[24 + 2], xl_28[24 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 32 + 24) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_28[28 + 0], xl_28[28 + 1], xl_28[28 + 2], xl_28[28 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 32 + 28) = _v4;
                }
            }
            float xl_29[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(xl_29[0]), "=f"(xl_29[1]), "=f"(xl_29[2]), "=f"(xl_29[3]), "=f"(xl_29[4]), "=f"(xl_29[5]), "=f"(xl_29[6]), "=f"(xl_29[7]), "=f"(xl_29[8]), "=f"(xl_29[9]), "=f"(xl_29[10]), "=f"(xl_29[11]), "=f"(xl_29[12]), "=f"(xl_29[13]), "=f"(xl_29[14]), "=f"(xl_29[15]), "=f"(xl_29[16]), "=f"(xl_29[17]), "=f"(xl_29[18]), "=f"(xl_29[19]), "=f"(xl_29[20]), "=f"(xl_29[21]), "=f"(xl_29[22]), "=f"(xl_29[23]), "=f"(xl_29[24]), "=f"(xl_29[25]), "=f"(xl_29[26]), "=f"(xl_29[27]), "=f"(xl_29[28]), "=f"(xl_29[29]), "=f"(xl_29[30]), "=f"(xl_29[31])
                : "r"(tmem_tmem + 256 + lpar * 128 + 64 + lane_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            if (idx_prev >= 0) {
                {
                    float4 _v4 = make_float4(xl_29[0 + 0], xl_29[0 + 1], xl_29[0 + 2], xl_29[0 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 64) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_29[4 + 0], xl_29[4 + 1], xl_29[4 + 2], xl_29[4 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 64 + 4) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_29[8 + 0], xl_29[8 + 1], xl_29[8 + 2], xl_29[8 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 64 + 8) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_29[12 + 0], xl_29[12 + 1], xl_29[12 + 2], xl_29[12 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 64 + 12) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_29[16 + 0], xl_29[16 + 1], xl_29[16 + 2], xl_29[16 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 64 + 16) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_29[20 + 0], xl_29[20 + 1], xl_29[20 + 2], xl_29[20 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 64 + 20) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_29[24 + 0], xl_29[24 + 1], xl_29[24 + 2], xl_29[24 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 64 + 24) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_29[28 + 0], xl_29[28 + 1], xl_29[28 + 2], xl_29[28 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 64 + 28) = _v4;
                }
            }
            float xl_30[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(xl_30[0]), "=f"(xl_30[1]), "=f"(xl_30[2]), "=f"(xl_30[3]), "=f"(xl_30[4]), "=f"(xl_30[5]), "=f"(xl_30[6]), "=f"(xl_30[7]), "=f"(xl_30[8]), "=f"(xl_30[9]), "=f"(xl_30[10]), "=f"(xl_30[11]), "=f"(xl_30[12]), "=f"(xl_30[13]), "=f"(xl_30[14]), "=f"(xl_30[15]), "=f"(xl_30[16]), "=f"(xl_30[17]), "=f"(xl_30[18]), "=f"(xl_30[19]), "=f"(xl_30[20]), "=f"(xl_30[21]), "=f"(xl_30[22]), "=f"(xl_30[23]), "=f"(xl_30[24]), "=f"(xl_30[25]), "=f"(xl_30[26]), "=f"(xl_30[27]), "=f"(xl_30[28]), "=f"(xl_30[29]), "=f"(xl_30[30]), "=f"(xl_30[31])
                : "r"(tmem_tmem + 256 + lpar * 128 + 96 + lane_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            if (idx_prev >= 0) {
                {
                    float4 _v4 = make_float4(xl_30[0 + 0], xl_30[0 + 1], xl_30[0 + 2], xl_30[0 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 96) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_30[4 + 0], xl_30[4 + 1], xl_30[4 + 2], xl_30[4 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 96 + 4) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_30[8 + 0], xl_30[8 + 1], xl_30[8 + 2], xl_30[8 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 96 + 8) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_30[12 + 0], xl_30[12 + 1], xl_30[12 + 2], xl_30[12 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 96 + 12) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_30[16 + 0], xl_30[16 + 1], xl_30[16 + 2], xl_30[16 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 96 + 16) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_30[20 + 0], xl_30[20 + 1], xl_30[20 + 2], xl_30[20 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 96 + 20) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_30[24 + 0], xl_30[24 + 1], xl_30[24 + 2], xl_30[24 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 96 + 24) = _v4;
                }
                {
                    float4 _v4 = make_float4(xl_30[28 + 0], xl_30[28 + 1], xl_30[28 + 2], xl_30[28 + 3]);
                    *reinterpret_cast<float4*>(partial + lrow + 96 + 28) = _v4;
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(dk_free_addr + (lpar) * 8);
            mbarrier_wait(dq_done_addr, 0);
            float dq[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(dq[0]), "=f"(dq[1]), "=f"(dq[2]), "=f"(dq[3]), "=f"(dq[4]), "=f"(dq[5]), "=f"(dq[6]), "=f"(dq[7]), "=f"(dq[8]), "=f"(dq[9]), "=f"(dq[10]), "=f"(dq[11]), "=f"(dq[12]), "=f"(dq[13]), "=f"(dq[14]), "=f"(dq[15]), "=f"(dq[16]), "=f"(dq[17]), "=f"(dq[18]), "=f"(dq[19]), "=f"(dq[20]), "=f"(dq[21]), "=f"(dq[22]), "=f"(dq[23]), "=f"(dq[24]), "=f"(dq[25]), "=f"(dq[26]), "=f"(dq[27]), "=f"(dq[28]), "=f"(dq[29]), "=f"(dq[30]), "=f"(dq[31])
                : "r"(tmem_tmem + 128 + lane_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            {
                __nv_bfloat16 _bval_136 = __float2bfloat16_rn(dq[0]);
                uint16_t _bits_136 = *(uint16_t*)&_bval_136;
                uint32_t _addr_136 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_136), "h"(_bits_136) : "memory");
            }
            {
                __nv_bfloat16 _bval_137 = __float2bfloat16_rn(dq[1]);
                uint16_t _bits_137 = *(uint16_t*)&_bval_137;
                uint32_t _addr_137 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(256 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_137), "h"(_bits_137) : "memory");
            }
            {
                __nv_bfloat16 _bval_138 = __float2bfloat16_rn(dq[2]);
                uint16_t _bits_138 = *(uint16_t*)&_bval_138;
                uint32_t _addr_138 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(512 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_138), "h"(_bits_138) : "memory");
            }
            {
                __nv_bfloat16 _bval_139 = __float2bfloat16_rn(dq[3]);
                uint16_t _bits_139 = *(uint16_t*)&_bval_139;
                uint32_t _addr_139 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(768 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_139), "h"(_bits_139) : "memory");
            }
            {
                __nv_bfloat16 _bval_140 = __float2bfloat16_rn(dq[4]);
                uint16_t _bits_140 = *(uint16_t*)&_bval_140;
                uint32_t _addr_140 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(1024 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_140), "h"(_bits_140) : "memory");
            }
            {
                __nv_bfloat16 _bval_141 = __float2bfloat16_rn(dq[5]);
                uint16_t _bits_141 = *(uint16_t*)&_bval_141;
                uint32_t _addr_141 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(1280 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_141), "h"(_bits_141) : "memory");
            }
            {
                __nv_bfloat16 _bval_142 = __float2bfloat16_rn(dq[6]);
                uint16_t _bits_142 = *(uint16_t*)&_bval_142;
                uint32_t _addr_142 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(1536 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_142), "h"(_bits_142) : "memory");
            }
            {
                __nv_bfloat16 _bval_143 = __float2bfloat16_rn(dq[7]);
                uint16_t _bits_143 = *(uint16_t*)&_bval_143;
                uint32_t _addr_143 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(1792 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_143), "h"(_bits_143) : "memory");
            }
            {
                __nv_bfloat16 _bval_144 = __float2bfloat16_rn(dq[8]);
                uint16_t _bits_144 = *(uint16_t*)&_bval_144;
                uint32_t _addr_144 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(2048 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_144), "h"(_bits_144) : "memory");
            }
            {
                __nv_bfloat16 _bval_145 = __float2bfloat16_rn(dq[9]);
                uint16_t _bits_145 = *(uint16_t*)&_bval_145;
                uint32_t _addr_145 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(2304 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_145), "h"(_bits_145) : "memory");
            }
            {
                __nv_bfloat16 _bval_146 = __float2bfloat16_rn(dq[10]);
                uint16_t _bits_146 = *(uint16_t*)&_bval_146;
                uint32_t _addr_146 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(2560 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_146), "h"(_bits_146) : "memory");
            }
            {
                __nv_bfloat16 _bval_147 = __float2bfloat16_rn(dq[11]);
                uint16_t _bits_147 = *(uint16_t*)&_bval_147;
                uint32_t _addr_147 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(2816 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_147), "h"(_bits_147) : "memory");
            }
            {
                __nv_bfloat16 _bval_148 = __float2bfloat16_rn(dq[12]);
                uint16_t _bits_148 = *(uint16_t*)&_bval_148;
                uint32_t _addr_148 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(3072 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_148), "h"(_bits_148) : "memory");
            }
            {
                __nv_bfloat16 _bval_149 = __float2bfloat16_rn(dq[13]);
                uint16_t _bits_149 = *(uint16_t*)&_bval_149;
                uint32_t _addr_149 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(3328 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_149), "h"(_bits_149) : "memory");
            }
            {
                __nv_bfloat16 _bval_150 = __float2bfloat16_rn(dq[14]);
                uint16_t _bits_150 = *(uint16_t*)&_bval_150;
                uint32_t _addr_150 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(3584 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_150), "h"(_bits_150) : "memory");
            }
            {
                __nv_bfloat16 _bval_151 = __float2bfloat16_rn(dq[15]);
                uint16_t _bits_151 = *(uint16_t*)&_bval_151;
                uint32_t _addr_151 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(3840 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_151), "h"(_bits_151) : "memory");
            }
            {
                __nv_bfloat16 _bval_152 = __float2bfloat16_rn(dq[16]);
                uint16_t _bits_152 = *(uint16_t*)&_bval_152;
                uint32_t _addr_152 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(4096 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_152), "h"(_bits_152) : "memory");
            }
            {
                __nv_bfloat16 _bval_153 = __float2bfloat16_rn(dq[17]);
                uint16_t _bits_153 = *(uint16_t*)&_bval_153;
                uint32_t _addr_153 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(4352 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_153), "h"(_bits_153) : "memory");
            }
            {
                __nv_bfloat16 _bval_154 = __float2bfloat16_rn(dq[18]);
                uint16_t _bits_154 = *(uint16_t*)&_bval_154;
                uint32_t _addr_154 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(4608 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_154), "h"(_bits_154) : "memory");
            }
            {
                __nv_bfloat16 _bval_155 = __float2bfloat16_rn(dq[19]);
                uint16_t _bits_155 = *(uint16_t*)&_bval_155;
                uint32_t _addr_155 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(4864 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_155), "h"(_bits_155) : "memory");
            }
            {
                __nv_bfloat16 _bval_156 = __float2bfloat16_rn(dq[20]);
                uint16_t _bits_156 = *(uint16_t*)&_bval_156;
                uint32_t _addr_156 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(5120 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_156), "h"(_bits_156) : "memory");
            }
            {
                __nv_bfloat16 _bval_157 = __float2bfloat16_rn(dq[21]);
                uint16_t _bits_157 = *(uint16_t*)&_bval_157;
                uint32_t _addr_157 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(5376 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_157), "h"(_bits_157) : "memory");
            }
            {
                __nv_bfloat16 _bval_158 = __float2bfloat16_rn(dq[22]);
                uint16_t _bits_158 = *(uint16_t*)&_bval_158;
                uint32_t _addr_158 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(5632 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_158), "h"(_bits_158) : "memory");
            }
            {
                __nv_bfloat16 _bval_159 = __float2bfloat16_rn(dq[23]);
                uint16_t _bits_159 = *(uint16_t*)&_bval_159;
                uint32_t _addr_159 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(5888 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_159), "h"(_bits_159) : "memory");
            }
            {
                __nv_bfloat16 _bval_160 = __float2bfloat16_rn(dq[24]);
                uint16_t _bits_160 = *(uint16_t*)&_bval_160;
                uint32_t _addr_160 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(6144 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_160), "h"(_bits_160) : "memory");
            }
            {
                __nv_bfloat16 _bval_161 = __float2bfloat16_rn(dq[25]);
                uint16_t _bits_161 = *(uint16_t*)&_bval_161;
                uint32_t _addr_161 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(6400 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_161), "h"(_bits_161) : "memory");
            }
            {
                __nv_bfloat16 _bval_162 = __float2bfloat16_rn(dq[26]);
                uint16_t _bits_162 = *(uint16_t*)&_bval_162;
                uint32_t _addr_162 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(6656 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_162), "h"(_bits_162) : "memory");
            }
            {
                __nv_bfloat16 _bval_163 = __float2bfloat16_rn(dq[27]);
                uint16_t _bits_163 = *(uint16_t*)&_bval_163;
                uint32_t _addr_163 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(6912 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_163), "h"(_bits_163) : "memory");
            }
            {
                __nv_bfloat16 _bval_164 = __float2bfloat16_rn(dq[28]);
                uint16_t _bits_164 = *(uint16_t*)&_bval_164;
                uint32_t _addr_164 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(7168 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_164), "h"(_bits_164) : "memory");
            }
            {
                __nv_bfloat16 _bval_165 = __float2bfloat16_rn(dq[29]);
                uint16_t _bits_165 = *(uint16_t*)&_bval_165;
                uint32_t _addr_165 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(7424 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_165), "h"(_bits_165) : "memory");
            }
            {
                __nv_bfloat16 _bval_166 = __float2bfloat16_rn(dq[30]);
                uint16_t _bits_166 = *(uint16_t*)&_bval_166;
                uint32_t _addr_166 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(7680 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_166), "h"(_bits_166) : "memory");
            }
            {
                __nv_bfloat16 _bval_167 = __float2bfloat16_rn(dq[31]);
                uint16_t _bits_167 = *(uint16_t*)&_bval_167;
                uint32_t _addr_167 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(7936 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_167), "h"(_bits_167) : "memory");
            }
            float dq_31[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(dq_31[0]), "=f"(dq_31[1]), "=f"(dq_31[2]), "=f"(dq_31[3]), "=f"(dq_31[4]), "=f"(dq_31[5]), "=f"(dq_31[6]), "=f"(dq_31[7]), "=f"(dq_31[8]), "=f"(dq_31[9]), "=f"(dq_31[10]), "=f"(dq_31[11]), "=f"(dq_31[12]), "=f"(dq_31[13]), "=f"(dq_31[14]), "=f"(dq_31[15]), "=f"(dq_31[16]), "=f"(dq_31[17]), "=f"(dq_31[18]), "=f"(dq_31[19]), "=f"(dq_31[20]), "=f"(dq_31[21]), "=f"(dq_31[22]), "=f"(dq_31[23]), "=f"(dq_31[24]), "=f"(dq_31[25]), "=f"(dq_31[26]), "=f"(dq_31[27]), "=f"(dq_31[28]), "=f"(dq_31[29]), "=f"(dq_31[30]), "=f"(dq_31[31])
                : "r"(tmem_tmem + 128 + 32 + lane_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            {
                __nv_bfloat16 _bval_168 = __float2bfloat16_rn(dq_31[0]);
                uint16_t _bits_168 = *(uint16_t*)&_bval_168;
                uint32_t _addr_168 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(8192 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_168), "h"(_bits_168) : "memory");
            }
            {
                __nv_bfloat16 _bval_169 = __float2bfloat16_rn(dq_31[1]);
                uint16_t _bits_169 = *(uint16_t*)&_bval_169;
                uint32_t _addr_169 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(8448 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_169), "h"(_bits_169) : "memory");
            }
            {
                __nv_bfloat16 _bval_170 = __float2bfloat16_rn(dq_31[2]);
                uint16_t _bits_170 = *(uint16_t*)&_bval_170;
                uint32_t _addr_170 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(8704 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_170), "h"(_bits_170) : "memory");
            }
            {
                __nv_bfloat16 _bval_171 = __float2bfloat16_rn(dq_31[3]);
                uint16_t _bits_171 = *(uint16_t*)&_bval_171;
                uint32_t _addr_171 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(8960 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_171), "h"(_bits_171) : "memory");
            }
            {
                __nv_bfloat16 _bval_172 = __float2bfloat16_rn(dq_31[4]);
                uint16_t _bits_172 = *(uint16_t*)&_bval_172;
                uint32_t _addr_172 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(9216 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_172), "h"(_bits_172) : "memory");
            }
            {
                __nv_bfloat16 _bval_173 = __float2bfloat16_rn(dq_31[5]);
                uint16_t _bits_173 = *(uint16_t*)&_bval_173;
                uint32_t _addr_173 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(9472 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_173), "h"(_bits_173) : "memory");
            }
            {
                __nv_bfloat16 _bval_174 = __float2bfloat16_rn(dq_31[6]);
                uint16_t _bits_174 = *(uint16_t*)&_bval_174;
                uint32_t _addr_174 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(9728 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_174), "h"(_bits_174) : "memory");
            }
            {
                __nv_bfloat16 _bval_175 = __float2bfloat16_rn(dq_31[7]);
                uint16_t _bits_175 = *(uint16_t*)&_bval_175;
                uint32_t _addr_175 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(9984 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_175), "h"(_bits_175) : "memory");
            }
            {
                __nv_bfloat16 _bval_176 = __float2bfloat16_rn(dq_31[8]);
                uint16_t _bits_176 = *(uint16_t*)&_bval_176;
                uint32_t _addr_176 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(10240 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_176), "h"(_bits_176) : "memory");
            }
            {
                __nv_bfloat16 _bval_177 = __float2bfloat16_rn(dq_31[9]);
                uint16_t _bits_177 = *(uint16_t*)&_bval_177;
                uint32_t _addr_177 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(10496 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_177), "h"(_bits_177) : "memory");
            }
            {
                __nv_bfloat16 _bval_178 = __float2bfloat16_rn(dq_31[10]);
                uint16_t _bits_178 = *(uint16_t*)&_bval_178;
                uint32_t _addr_178 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(10752 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_178), "h"(_bits_178) : "memory");
            }
            {
                __nv_bfloat16 _bval_179 = __float2bfloat16_rn(dq_31[11]);
                uint16_t _bits_179 = *(uint16_t*)&_bval_179;
                uint32_t _addr_179 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(11008 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_179), "h"(_bits_179) : "memory");
            }
            {
                __nv_bfloat16 _bval_180 = __float2bfloat16_rn(dq_31[12]);
                uint16_t _bits_180 = *(uint16_t*)&_bval_180;
                uint32_t _addr_180 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(11264 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_180), "h"(_bits_180) : "memory");
            }
            {
                __nv_bfloat16 _bval_181 = __float2bfloat16_rn(dq_31[13]);
                uint16_t _bits_181 = *(uint16_t*)&_bval_181;
                uint32_t _addr_181 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(11520 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_181), "h"(_bits_181) : "memory");
            }
            {
                __nv_bfloat16 _bval_182 = __float2bfloat16_rn(dq_31[14]);
                uint16_t _bits_182 = *(uint16_t*)&_bval_182;
                uint32_t _addr_182 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(11776 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_182), "h"(_bits_182) : "memory");
            }
            {
                __nv_bfloat16 _bval_183 = __float2bfloat16_rn(dq_31[15]);
                uint16_t _bits_183 = *(uint16_t*)&_bval_183;
                uint32_t _addr_183 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(12032 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_183), "h"(_bits_183) : "memory");
            }
            {
                __nv_bfloat16 _bval_184 = __float2bfloat16_rn(dq_31[16]);
                uint16_t _bits_184 = *(uint16_t*)&_bval_184;
                uint32_t _addr_184 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(12288 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_184), "h"(_bits_184) : "memory");
            }
            {
                __nv_bfloat16 _bval_185 = __float2bfloat16_rn(dq_31[17]);
                uint16_t _bits_185 = *(uint16_t*)&_bval_185;
                uint32_t _addr_185 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(12544 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_185), "h"(_bits_185) : "memory");
            }
            {
                __nv_bfloat16 _bval_186 = __float2bfloat16_rn(dq_31[18]);
                uint16_t _bits_186 = *(uint16_t*)&_bval_186;
                uint32_t _addr_186 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(12800 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_186), "h"(_bits_186) : "memory");
            }
            {
                __nv_bfloat16 _bval_187 = __float2bfloat16_rn(dq_31[19]);
                uint16_t _bits_187 = *(uint16_t*)&_bval_187;
                uint32_t _addr_187 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(13056 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_187), "h"(_bits_187) : "memory");
            }
            {
                __nv_bfloat16 _bval_188 = __float2bfloat16_rn(dq_31[20]);
                uint16_t _bits_188 = *(uint16_t*)&_bval_188;
                uint32_t _addr_188 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(13312 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_188), "h"(_bits_188) : "memory");
            }
            {
                __nv_bfloat16 _bval_189 = __float2bfloat16_rn(dq_31[21]);
                uint16_t _bits_189 = *(uint16_t*)&_bval_189;
                uint32_t _addr_189 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(13568 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_189), "h"(_bits_189) : "memory");
            }
            {
                __nv_bfloat16 _bval_190 = __float2bfloat16_rn(dq_31[22]);
                uint16_t _bits_190 = *(uint16_t*)&_bval_190;
                uint32_t _addr_190 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(13824 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_190), "h"(_bits_190) : "memory");
            }
            {
                __nv_bfloat16 _bval_191 = __float2bfloat16_rn(dq_31[23]);
                uint16_t _bits_191 = *(uint16_t*)&_bval_191;
                uint32_t _addr_191 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(14080 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_191), "h"(_bits_191) : "memory");
            }
            {
                __nv_bfloat16 _bval_192 = __float2bfloat16_rn(dq_31[24]);
                uint16_t _bits_192 = *(uint16_t*)&_bval_192;
                uint32_t _addr_192 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(14336 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_192), "h"(_bits_192) : "memory");
            }
            {
                __nv_bfloat16 _bval_193 = __float2bfloat16_rn(dq_31[25]);
                uint16_t _bits_193 = *(uint16_t*)&_bval_193;
                uint32_t _addr_193 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(14592 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_193), "h"(_bits_193) : "memory");
            }
            {
                __nv_bfloat16 _bval_194 = __float2bfloat16_rn(dq_31[26]);
                uint16_t _bits_194 = *(uint16_t*)&_bval_194;
                uint32_t _addr_194 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(14848 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_194), "h"(_bits_194) : "memory");
            }
            {
                __nv_bfloat16 _bval_195 = __float2bfloat16_rn(dq_31[27]);
                uint16_t _bits_195 = *(uint16_t*)&_bval_195;
                uint32_t _addr_195 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(15104 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_195), "h"(_bits_195) : "memory");
            }
            {
                __nv_bfloat16 _bval_196 = __float2bfloat16_rn(dq_31[28]);
                uint16_t _bits_196 = *(uint16_t*)&_bval_196;
                uint32_t _addr_196 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(15360 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_196), "h"(_bits_196) : "memory");
            }
            {
                __nv_bfloat16 _bval_197 = __float2bfloat16_rn(dq_31[29]);
                uint16_t _bits_197 = *(uint16_t*)&_bval_197;
                uint32_t _addr_197 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(15616 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_197), "h"(_bits_197) : "memory");
            }
            {
                __nv_bfloat16 _bval_198 = __float2bfloat16_rn(dq_31[30]);
                uint16_t _bits_198 = *(uint16_t*)&_bval_198;
                uint32_t _addr_198 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(15872 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_198), "h"(_bits_198) : "memory");
            }
            {
                __nv_bfloat16 _bval_199 = __float2bfloat16_rn(dq_31[31]);
                uint16_t _bits_199 = *(uint16_t*)&_bval_199;
                uint32_t _addr_199 = static_cast<uint32_t>(q_raw_addr + (unsigned int)(16128 + tid_1 * 2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_199), "h"(_bits_199) : "memory");
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            int c2 = tid_1;
            int oh = c2 / 16;
            int od = c2 % 16 * 8;
            float _q_raw_reg_0[8];
            {
                uint32_t _smem_addr_200 = q_raw_addr + (oh * 128 + od) * 2;
                uint32_t _bf16x2_200[4];
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_bf16x2_200[0]), "=r"(_bf16x2_200[1]), "=r"(_bf16x2_200[2]), "=r"(_bf16x2_200[3]) : "r"(_smem_addr_200) : "memory");
                (&_q_raw_reg_0[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_200[0]) << 16);
                (&_q_raw_reg_0[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_200[0]) & 0xffff0000u);
                (&_q_raw_reg_0[2])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_200[1]) << 16);
                (&_q_raw_reg_0[2])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_200[1]) & 0xffff0000u);
                (&_q_raw_reg_0[4])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_200[2]) << 16);
                (&_q_raw_reg_0[4])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_200[2]) & 0xffff0000u);
                (&_q_raw_reg_0[6])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_200[3]) << 16);
                (&_q_raw_reg_0[6])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_200[3]) & 0xffff0000u);
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(_q_raw_reg_0[0 + 0], _q_raw_reg_0[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(_q_raw_reg_0[0 + 2], _q_raw_reg_0[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(_q_raw_reg_0[0 + 4], _q_raw_reg_0[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(_q_raw_reg_0[0 + 6], _q_raw_reg_0[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(d_index_q))[(q_row0 + (long long)oh) * 128 + (long long)od + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            int c2_32 = 128 + tid_1;
            int oh_33 = c2_32 / 16;
            int od_34 = c2_32 % 16 * 8;
            float _q_raw_reg_1[8];
            {
                uint32_t _smem_addr_201 = q_raw_addr + (oh_33 * 128 + od_34) * 2;
                uint32_t _bf16x2_201[4];
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_bf16x2_201[0]), "=r"(_bf16x2_201[1]), "=r"(_bf16x2_201[2]), "=r"(_bf16x2_201[3]) : "r"(_smem_addr_201) : "memory");
                (&_q_raw_reg_1[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_201[0]) << 16);
                (&_q_raw_reg_1[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_201[0]) & 0xffff0000u);
                (&_q_raw_reg_1[2])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_201[1]) << 16);
                (&_q_raw_reg_1[2])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_201[1]) & 0xffff0000u);
                (&_q_raw_reg_1[4])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_201[2]) << 16);
                (&_q_raw_reg_1[4])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_201[2]) & 0xffff0000u);
                (&_q_raw_reg_1[6])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_201[3]) << 16);
                (&_q_raw_reg_1[6])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_201[3]) & 0xffff0000u);
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(_q_raw_reg_1[0 + 0], _q_raw_reg_1[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(_q_raw_reg_1[0 + 2], _q_raw_reg_1[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(_q_raw_reg_1[0 + 4], _q_raw_reg_1[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(_q_raw_reg_1[0 + 6], _q_raw_reg_1[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(d_index_q))[(q_row0 + (long long)oh_33) * 128 + (long long)od_34 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            int c2_35 = 256 + tid_1;
            int oh_36 = c2_35 / 16;
            int od_37 = c2_35 % 16 * 8;
            float _q_raw_reg_2[8];
            {
                uint32_t _smem_addr_202 = q_raw_addr + (oh_36 * 128 + od_37) * 2;
                uint32_t _bf16x2_202[4];
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_bf16x2_202[0]), "=r"(_bf16x2_202[1]), "=r"(_bf16x2_202[2]), "=r"(_bf16x2_202[3]) : "r"(_smem_addr_202) : "memory");
                (&_q_raw_reg_2[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_202[0]) << 16);
                (&_q_raw_reg_2[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_202[0]) & 0xffff0000u);
                (&_q_raw_reg_2[2])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_202[1]) << 16);
                (&_q_raw_reg_2[2])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_202[1]) & 0xffff0000u);
                (&_q_raw_reg_2[4])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_202[2]) << 16);
                (&_q_raw_reg_2[4])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_202[2]) & 0xffff0000u);
                (&_q_raw_reg_2[6])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_202[3]) << 16);
                (&_q_raw_reg_2[6])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_202[3]) & 0xffff0000u);
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(_q_raw_reg_2[0 + 0], _q_raw_reg_2[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(_q_raw_reg_2[0 + 2], _q_raw_reg_2[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(_q_raw_reg_2[0 + 4], _q_raw_reg_2[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(_q_raw_reg_2[0 + 6], _q_raw_reg_2[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(d_index_q))[(q_row0 + (long long)oh_36) * 128 + (long long)od_37 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            int c2_38 = 384 + tid_1;
            int oh_39 = c2_38 / 16;
            int od_40 = c2_38 % 16 * 8;
            float _q_raw_reg_3[8];
            {
                uint32_t _smem_addr_203 = q_raw_addr + (oh_39 * 128 + od_40) * 2;
                uint32_t _bf16x2_203[4];
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_bf16x2_203[0]), "=r"(_bf16x2_203[1]), "=r"(_bf16x2_203[2]), "=r"(_bf16x2_203[3]) : "r"(_smem_addr_203) : "memory");
                (&_q_raw_reg_3[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_203[0]) << 16);
                (&_q_raw_reg_3[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_203[0]) & 0xffff0000u);
                (&_q_raw_reg_3[2])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_203[1]) << 16);
                (&_q_raw_reg_3[2])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_203[1]) & 0xffff0000u);
                (&_q_raw_reg_3[4])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_203[2]) << 16);
                (&_q_raw_reg_3[4])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_203[2]) & 0xffff0000u);
                (&_q_raw_reg_3[6])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_203[3]) << 16);
                (&_q_raw_reg_3[6])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_203[3]) & 0xffff0000u);
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(_q_raw_reg_3[0 + 0], _q_raw_reg_3[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(_q_raw_reg_3[0 + 2], _q_raw_reg_3[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(_q_raw_reg_3[0 + 4], _q_raw_reg_3[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(_q_raw_reg_3[0 + 6], _q_raw_reg_3[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(d_index_q))[(q_row0 + (long long)oh_39) * 128 + (long long)od_40 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            int c2_41 = 512 + tid_1;
            int oh_42 = c2_41 / 16;
            int od_43 = c2_41 % 16 * 8;
            float _q_raw_reg_4[8];
            {
                uint32_t _smem_addr_204 = q_raw_addr + (oh_42 * 128 + od_43) * 2;
                uint32_t _bf16x2_204[4];
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_bf16x2_204[0]), "=r"(_bf16x2_204[1]), "=r"(_bf16x2_204[2]), "=r"(_bf16x2_204[3]) : "r"(_smem_addr_204) : "memory");
                (&_q_raw_reg_4[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_204[0]) << 16);
                (&_q_raw_reg_4[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_204[0]) & 0xffff0000u);
                (&_q_raw_reg_4[2])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_204[1]) << 16);
                (&_q_raw_reg_4[2])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_204[1]) & 0xffff0000u);
                (&_q_raw_reg_4[4])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_204[2]) << 16);
                (&_q_raw_reg_4[4])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_204[2]) & 0xffff0000u);
                (&_q_raw_reg_4[6])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_204[3]) << 16);
                (&_q_raw_reg_4[6])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_204[3]) & 0xffff0000u);
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(_q_raw_reg_4[0 + 0], _q_raw_reg_4[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(_q_raw_reg_4[0 + 2], _q_raw_reg_4[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(_q_raw_reg_4[0 + 4], _q_raw_reg_4[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(_q_raw_reg_4[0 + 6], _q_raw_reg_4[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(d_index_q))[(q_row0 + (long long)oh_42) * 128 + (long long)od_43 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            int c2_44 = 640 + tid_1;
            int oh_45 = c2_44 / 16;
            int od_46 = c2_44 % 16 * 8;
            float _q_raw_reg_5[8];
            {
                uint32_t _smem_addr_205 = q_raw_addr + (oh_45 * 128 + od_46) * 2;
                uint32_t _bf16x2_205[4];
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_bf16x2_205[0]), "=r"(_bf16x2_205[1]), "=r"(_bf16x2_205[2]), "=r"(_bf16x2_205[3]) : "r"(_smem_addr_205) : "memory");
                (&_q_raw_reg_5[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_205[0]) << 16);
                (&_q_raw_reg_5[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_205[0]) & 0xffff0000u);
                (&_q_raw_reg_5[2])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_205[1]) << 16);
                (&_q_raw_reg_5[2])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_205[1]) & 0xffff0000u);
                (&_q_raw_reg_5[4])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_205[2]) << 16);
                (&_q_raw_reg_5[4])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_205[2]) & 0xffff0000u);
                (&_q_raw_reg_5[6])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_205[3]) << 16);
                (&_q_raw_reg_5[6])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_205[3]) & 0xffff0000u);
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(_q_raw_reg_5[0 + 0], _q_raw_reg_5[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(_q_raw_reg_5[0 + 2], _q_raw_reg_5[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(_q_raw_reg_5[0 + 4], _q_raw_reg_5[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(_q_raw_reg_5[0 + 6], _q_raw_reg_5[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(d_index_q))[(q_row0 + (long long)oh_45) * 128 + (long long)od_46 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            int c2_47 = 768 + tid_1;
            int oh_48 = c2_47 / 16;
            int od_49 = c2_47 % 16 * 8;
            float _q_raw_reg_6[8];
            {
                uint32_t _smem_addr_206 = q_raw_addr + (oh_48 * 128 + od_49) * 2;
                uint32_t _bf16x2_206[4];
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_bf16x2_206[0]), "=r"(_bf16x2_206[1]), "=r"(_bf16x2_206[2]), "=r"(_bf16x2_206[3]) : "r"(_smem_addr_206) : "memory");
                (&_q_raw_reg_6[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_206[0]) << 16);
                (&_q_raw_reg_6[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_206[0]) & 0xffff0000u);
                (&_q_raw_reg_6[2])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_206[1]) << 16);
                (&_q_raw_reg_6[2])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_206[1]) & 0xffff0000u);
                (&_q_raw_reg_6[4])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_206[2]) << 16);
                (&_q_raw_reg_6[4])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_206[2]) & 0xffff0000u);
                (&_q_raw_reg_6[6])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_206[3]) << 16);
                (&_q_raw_reg_6[6])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_206[3]) & 0xffff0000u);
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(_q_raw_reg_6[0 + 0], _q_raw_reg_6[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(_q_raw_reg_6[0 + 2], _q_raw_reg_6[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(_q_raw_reg_6[0 + 4], _q_raw_reg_6[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(_q_raw_reg_6[0 + 6], _q_raw_reg_6[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(d_index_q))[(q_row0 + (long long)oh_48) * 128 + (long long)od_49 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            int c2_50 = 896 + tid_1;
            int oh_51 = c2_50 / 16;
            int od_52 = c2_50 % 16 * 8;
            float _q_raw_reg_7[8];
            {
                uint32_t _smem_addr_207 = q_raw_addr + (oh_51 * 128 + od_52) * 2;
                uint32_t _bf16x2_207[4];
                asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                    : "=r"(_bf16x2_207[0]), "=r"(_bf16x2_207[1]), "=r"(_bf16x2_207[2]), "=r"(_bf16x2_207[3]) : "r"(_smem_addr_207) : "memory");
                (&_q_raw_reg_7[0])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_207[0]) << 16);
                (&_q_raw_reg_7[0])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_207[0]) & 0xffff0000u);
                (&_q_raw_reg_7[2])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_207[1]) << 16);
                (&_q_raw_reg_7[2])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_207[1]) & 0xffff0000u);
                (&_q_raw_reg_7[4])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_207[2]) << 16);
                (&_q_raw_reg_7[4])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_207[2]) & 0xffff0000u);
                (&_q_raw_reg_7[6])[0] = __uint_as_float(static_cast<uint32_t>(_bf16x2_207[3]) << 16);
                (&_q_raw_reg_7[6])[1] = __uint_as_float(static_cast<uint32_t>(_bf16x2_207[3]) & 0xffff0000u);
            }
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(_q_raw_reg_7[0 + 0], _q_raw_reg_7[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(_q_raw_reg_7[0 + 2], _q_raw_reg_7[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(_q_raw_reg_7[0 + 4], _q_raw_reg_7[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(_q_raw_reg_7[0 + 6], _q_raw_reg_7[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(d_index_q))[(q_row0 + (long long)oh_51) * 128 + (long long)od_52 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, dw[0], 1);
            dw[0] = dw[0] + _shfl_xor_0;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, dw[0], 2);
            dw[0] = dw[0] + _shfl_xor_1;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, dw[0], 4);
            dw[0] = dw[0] + _shfl_xor_2;
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, dw[0], 8);
            dw[0] = dw[0] + _shfl_xor_3;
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, dw[0], 16);
            dw[0] = dw[0] + _shfl_xor_4;
            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, dw[1], 1);
            dw[1] = dw[1] + _shfl_xor_5;
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, dw[1], 2);
            dw[1] = dw[1] + _shfl_xor_6;
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, dw[1], 4);
            dw[1] = dw[1] + _shfl_xor_7;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, dw[1], 8);
            dw[1] = dw[1] + _shfl_xor_8;
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, dw[1], 16);
            dw[1] = dw[1] + _shfl_xor_9;
            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, dw[2], 1);
            dw[2] = dw[2] + _shfl_xor_10;
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, dw[2], 2);
            dw[2] = dw[2] + _shfl_xor_11;
            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, dw[2], 4);
            dw[2] = dw[2] + _shfl_xor_12;
            float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, dw[2], 8);
            dw[2] = dw[2] + _shfl_xor_13;
            float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, dw[2], 16);
            dw[2] = dw[2] + _shfl_xor_14;
            float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, dw[3], 1);
            dw[3] = dw[3] + _shfl_xor_15;
            float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, dw[3], 2);
            dw[3] = dw[3] + _shfl_xor_16;
            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, dw[3], 4);
            dw[3] = dw[3] + _shfl_xor_17;
            float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, dw[3], 8);
            dw[3] = dw[3] + _shfl_xor_18;
            float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, dw[3], 16);
            dw[3] = dw[3] + _shfl_xor_19;
            float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, dw[4], 1);
            dw[4] = dw[4] + _shfl_xor_20;
            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, dw[4], 2);
            dw[4] = dw[4] + _shfl_xor_21;
            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, dw[4], 4);
            dw[4] = dw[4] + _shfl_xor_22;
            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, dw[4], 8);
            dw[4] = dw[4] + _shfl_xor_23;
            float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, dw[4], 16);
            dw[4] = dw[4] + _shfl_xor_24;
            float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, dw[5], 1);
            dw[5] = dw[5] + _shfl_xor_25;
            float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, dw[5], 2);
            dw[5] = dw[5] + _shfl_xor_26;
            float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, dw[5], 4);
            dw[5] = dw[5] + _shfl_xor_27;
            float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, dw[5], 8);
            dw[5] = dw[5] + _shfl_xor_28;
            float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, dw[5], 16);
            dw[5] = dw[5] + _shfl_xor_29;
            float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, dw[6], 1);
            dw[6] = dw[6] + _shfl_xor_30;
            float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, dw[6], 2);
            dw[6] = dw[6] + _shfl_xor_31;
            float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, dw[6], 4);
            dw[6] = dw[6] + _shfl_xor_32;
            float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, dw[6], 8);
            dw[6] = dw[6] + _shfl_xor_33;
            float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, dw[6], 16);
            dw[6] = dw[6] + _shfl_xor_34;
            float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, dw[7], 1);
            dw[7] = dw[7] + _shfl_xor_35;
            float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, dw[7], 2);
            dw[7] = dw[7] + _shfl_xor_36;
            float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, dw[7], 4);
            dw[7] = dw[7] + _shfl_xor_37;
            float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, dw[7], 8);
            dw[7] = dw[7] + _shfl_xor_38;
            float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, dw[7], 16);
            dw[7] = dw[7] + _shfl_xor_39;
            float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, dw[8], 1);
            dw[8] = dw[8] + _shfl_xor_40;
            float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, dw[8], 2);
            dw[8] = dw[8] + _shfl_xor_41;
            float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, dw[8], 4);
            dw[8] = dw[8] + _shfl_xor_42;
            float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, dw[8], 8);
            dw[8] = dw[8] + _shfl_xor_43;
            float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, dw[8], 16);
            dw[8] = dw[8] + _shfl_xor_44;
            float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, dw[9], 1);
            dw[9] = dw[9] + _shfl_xor_45;
            float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, dw[9], 2);
            dw[9] = dw[9] + _shfl_xor_46;
            float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, dw[9], 4);
            dw[9] = dw[9] + _shfl_xor_47;
            float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, dw[9], 8);
            dw[9] = dw[9] + _shfl_xor_48;
            float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, dw[9], 16);
            dw[9] = dw[9] + _shfl_xor_49;
            float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, dw[10], 1);
            dw[10] = dw[10] + _shfl_xor_50;
            float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, dw[10], 2);
            dw[10] = dw[10] + _shfl_xor_51;
            float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, dw[10], 4);
            dw[10] = dw[10] + _shfl_xor_52;
            float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, dw[10], 8);
            dw[10] = dw[10] + _shfl_xor_53;
            float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, dw[10], 16);
            dw[10] = dw[10] + _shfl_xor_54;
            float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, dw[11], 1);
            dw[11] = dw[11] + _shfl_xor_55;
            float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, dw[11], 2);
            dw[11] = dw[11] + _shfl_xor_56;
            float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, dw[11], 4);
            dw[11] = dw[11] + _shfl_xor_57;
            float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, dw[11], 8);
            dw[11] = dw[11] + _shfl_xor_58;
            float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, dw[11], 16);
            dw[11] = dw[11] + _shfl_xor_59;
            float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, dw[12], 1);
            dw[12] = dw[12] + _shfl_xor_60;
            float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, dw[12], 2);
            dw[12] = dw[12] + _shfl_xor_61;
            float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, dw[12], 4);
            dw[12] = dw[12] + _shfl_xor_62;
            float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, dw[12], 8);
            dw[12] = dw[12] + _shfl_xor_63;
            float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, dw[12], 16);
            dw[12] = dw[12] + _shfl_xor_64;
            float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, dw[13], 1);
            dw[13] = dw[13] + _shfl_xor_65;
            float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, dw[13], 2);
            dw[13] = dw[13] + _shfl_xor_66;
            float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, dw[13], 4);
            dw[13] = dw[13] + _shfl_xor_67;
            float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, dw[13], 8);
            dw[13] = dw[13] + _shfl_xor_68;
            float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, dw[13], 16);
            dw[13] = dw[13] + _shfl_xor_69;
            float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, dw[14], 1);
            dw[14] = dw[14] + _shfl_xor_70;
            float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, dw[14], 2);
            dw[14] = dw[14] + _shfl_xor_71;
            float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, dw[14], 4);
            dw[14] = dw[14] + _shfl_xor_72;
            float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, dw[14], 8);
            dw[14] = dw[14] + _shfl_xor_73;
            float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, dw[14], 16);
            dw[14] = dw[14] + _shfl_xor_74;
            float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, dw[15], 1);
            dw[15] = dw[15] + _shfl_xor_75;
            float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, dw[15], 2);
            dw[15] = dw[15] + _shfl_xor_76;
            float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, dw[15], 4);
            dw[15] = dw[15] + _shfl_xor_77;
            float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, dw[15], 8);
            dw[15] = dw[15] + _shfl_xor_78;
            float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, dw[15], 16);
            dw[15] = dw[15] + _shfl_xor_79;
            float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, dw[16], 1);
            dw[16] = dw[16] + _shfl_xor_80;
            float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, dw[16], 2);
            dw[16] = dw[16] + _shfl_xor_81;
            float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, dw[16], 4);
            dw[16] = dw[16] + _shfl_xor_82;
            float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, dw[16], 8);
            dw[16] = dw[16] + _shfl_xor_83;
            float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, dw[16], 16);
            dw[16] = dw[16] + _shfl_xor_84;
            float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, dw[17], 1);
            dw[17] = dw[17] + _shfl_xor_85;
            float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, dw[17], 2);
            dw[17] = dw[17] + _shfl_xor_86;
            float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, dw[17], 4);
            dw[17] = dw[17] + _shfl_xor_87;
            float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, dw[17], 8);
            dw[17] = dw[17] + _shfl_xor_88;
            float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, dw[17], 16);
            dw[17] = dw[17] + _shfl_xor_89;
            float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, dw[18], 1);
            dw[18] = dw[18] + _shfl_xor_90;
            float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, dw[18], 2);
            dw[18] = dw[18] + _shfl_xor_91;
            float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, dw[18], 4);
            dw[18] = dw[18] + _shfl_xor_92;
            float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, dw[18], 8);
            dw[18] = dw[18] + _shfl_xor_93;
            float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, dw[18], 16);
            dw[18] = dw[18] + _shfl_xor_94;
            float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, dw[19], 1);
            dw[19] = dw[19] + _shfl_xor_95;
            float _shfl_xor_96 = __shfl_xor_sync(0xFFFFFFFF, dw[19], 2);
            dw[19] = dw[19] + _shfl_xor_96;
            float _shfl_xor_97 = __shfl_xor_sync(0xFFFFFFFF, dw[19], 4);
            dw[19] = dw[19] + _shfl_xor_97;
            float _shfl_xor_98 = __shfl_xor_sync(0xFFFFFFFF, dw[19], 8);
            dw[19] = dw[19] + _shfl_xor_98;
            float _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, dw[19], 16);
            dw[19] = dw[19] + _shfl_xor_99;
            float _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, dw[20], 1);
            dw[20] = dw[20] + _shfl_xor_100;
            float _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, dw[20], 2);
            dw[20] = dw[20] + _shfl_xor_101;
            float _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, dw[20], 4);
            dw[20] = dw[20] + _shfl_xor_102;
            float _shfl_xor_103 = __shfl_xor_sync(0xFFFFFFFF, dw[20], 8);
            dw[20] = dw[20] + _shfl_xor_103;
            float _shfl_xor_104 = __shfl_xor_sync(0xFFFFFFFF, dw[20], 16);
            dw[20] = dw[20] + _shfl_xor_104;
            float _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, dw[21], 1);
            dw[21] = dw[21] + _shfl_xor_105;
            float _shfl_xor_106 = __shfl_xor_sync(0xFFFFFFFF, dw[21], 2);
            dw[21] = dw[21] + _shfl_xor_106;
            float _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, dw[21], 4);
            dw[21] = dw[21] + _shfl_xor_107;
            float _shfl_xor_108 = __shfl_xor_sync(0xFFFFFFFF, dw[21], 8);
            dw[21] = dw[21] + _shfl_xor_108;
            float _shfl_xor_109 = __shfl_xor_sync(0xFFFFFFFF, dw[21], 16);
            dw[21] = dw[21] + _shfl_xor_109;
            float _shfl_xor_110 = __shfl_xor_sync(0xFFFFFFFF, dw[22], 1);
            dw[22] = dw[22] + _shfl_xor_110;
            float _shfl_xor_111 = __shfl_xor_sync(0xFFFFFFFF, dw[22], 2);
            dw[22] = dw[22] + _shfl_xor_111;
            float _shfl_xor_112 = __shfl_xor_sync(0xFFFFFFFF, dw[22], 4);
            dw[22] = dw[22] + _shfl_xor_112;
            float _shfl_xor_113 = __shfl_xor_sync(0xFFFFFFFF, dw[22], 8);
            dw[22] = dw[22] + _shfl_xor_113;
            float _shfl_xor_114 = __shfl_xor_sync(0xFFFFFFFF, dw[22], 16);
            dw[22] = dw[22] + _shfl_xor_114;
            float _shfl_xor_115 = __shfl_xor_sync(0xFFFFFFFF, dw[23], 1);
            dw[23] = dw[23] + _shfl_xor_115;
            float _shfl_xor_116 = __shfl_xor_sync(0xFFFFFFFF, dw[23], 2);
            dw[23] = dw[23] + _shfl_xor_116;
            float _shfl_xor_117 = __shfl_xor_sync(0xFFFFFFFF, dw[23], 4);
            dw[23] = dw[23] + _shfl_xor_117;
            float _shfl_xor_118 = __shfl_xor_sync(0xFFFFFFFF, dw[23], 8);
            dw[23] = dw[23] + _shfl_xor_118;
            float _shfl_xor_119 = __shfl_xor_sync(0xFFFFFFFF, dw[23], 16);
            dw[23] = dw[23] + _shfl_xor_119;
            float _shfl_xor_120 = __shfl_xor_sync(0xFFFFFFFF, dw[24], 1);
            dw[24] = dw[24] + _shfl_xor_120;
            float _shfl_xor_121 = __shfl_xor_sync(0xFFFFFFFF, dw[24], 2);
            dw[24] = dw[24] + _shfl_xor_121;
            float _shfl_xor_122 = __shfl_xor_sync(0xFFFFFFFF, dw[24], 4);
            dw[24] = dw[24] + _shfl_xor_122;
            float _shfl_xor_123 = __shfl_xor_sync(0xFFFFFFFF, dw[24], 8);
            dw[24] = dw[24] + _shfl_xor_123;
            float _shfl_xor_124 = __shfl_xor_sync(0xFFFFFFFF, dw[24], 16);
            dw[24] = dw[24] + _shfl_xor_124;
            float _shfl_xor_125 = __shfl_xor_sync(0xFFFFFFFF, dw[25], 1);
            dw[25] = dw[25] + _shfl_xor_125;
            float _shfl_xor_126 = __shfl_xor_sync(0xFFFFFFFF, dw[25], 2);
            dw[25] = dw[25] + _shfl_xor_126;
            float _shfl_xor_127 = __shfl_xor_sync(0xFFFFFFFF, dw[25], 4);
            dw[25] = dw[25] + _shfl_xor_127;
            float _shfl_xor_128 = __shfl_xor_sync(0xFFFFFFFF, dw[25], 8);
            dw[25] = dw[25] + _shfl_xor_128;
            float _shfl_xor_129 = __shfl_xor_sync(0xFFFFFFFF, dw[25], 16);
            dw[25] = dw[25] + _shfl_xor_129;
            float _shfl_xor_130 = __shfl_xor_sync(0xFFFFFFFF, dw[26], 1);
            dw[26] = dw[26] + _shfl_xor_130;
            float _shfl_xor_131 = __shfl_xor_sync(0xFFFFFFFF, dw[26], 2);
            dw[26] = dw[26] + _shfl_xor_131;
            float _shfl_xor_132 = __shfl_xor_sync(0xFFFFFFFF, dw[26], 4);
            dw[26] = dw[26] + _shfl_xor_132;
            float _shfl_xor_133 = __shfl_xor_sync(0xFFFFFFFF, dw[26], 8);
            dw[26] = dw[26] + _shfl_xor_133;
            float _shfl_xor_134 = __shfl_xor_sync(0xFFFFFFFF, dw[26], 16);
            dw[26] = dw[26] + _shfl_xor_134;
            float _shfl_xor_135 = __shfl_xor_sync(0xFFFFFFFF, dw[27], 1);
            dw[27] = dw[27] + _shfl_xor_135;
            float _shfl_xor_136 = __shfl_xor_sync(0xFFFFFFFF, dw[27], 2);
            dw[27] = dw[27] + _shfl_xor_136;
            float _shfl_xor_137 = __shfl_xor_sync(0xFFFFFFFF, dw[27], 4);
            dw[27] = dw[27] + _shfl_xor_137;
            float _shfl_xor_138 = __shfl_xor_sync(0xFFFFFFFF, dw[27], 8);
            dw[27] = dw[27] + _shfl_xor_138;
            float _shfl_xor_139 = __shfl_xor_sync(0xFFFFFFFF, dw[27], 16);
            dw[27] = dw[27] + _shfl_xor_139;
            float _shfl_xor_140 = __shfl_xor_sync(0xFFFFFFFF, dw[28], 1);
            dw[28] = dw[28] + _shfl_xor_140;
            float _shfl_xor_141 = __shfl_xor_sync(0xFFFFFFFF, dw[28], 2);
            dw[28] = dw[28] + _shfl_xor_141;
            float _shfl_xor_142 = __shfl_xor_sync(0xFFFFFFFF, dw[28], 4);
            dw[28] = dw[28] + _shfl_xor_142;
            float _shfl_xor_143 = __shfl_xor_sync(0xFFFFFFFF, dw[28], 8);
            dw[28] = dw[28] + _shfl_xor_143;
            float _shfl_xor_144 = __shfl_xor_sync(0xFFFFFFFF, dw[28], 16);
            dw[28] = dw[28] + _shfl_xor_144;
            float _shfl_xor_145 = __shfl_xor_sync(0xFFFFFFFF, dw[29], 1);
            dw[29] = dw[29] + _shfl_xor_145;
            float _shfl_xor_146 = __shfl_xor_sync(0xFFFFFFFF, dw[29], 2);
            dw[29] = dw[29] + _shfl_xor_146;
            float _shfl_xor_147 = __shfl_xor_sync(0xFFFFFFFF, dw[29], 4);
            dw[29] = dw[29] + _shfl_xor_147;
            float _shfl_xor_148 = __shfl_xor_sync(0xFFFFFFFF, dw[29], 8);
            dw[29] = dw[29] + _shfl_xor_148;
            float _shfl_xor_149 = __shfl_xor_sync(0xFFFFFFFF, dw[29], 16);
            dw[29] = dw[29] + _shfl_xor_149;
            float _shfl_xor_150 = __shfl_xor_sync(0xFFFFFFFF, dw[30], 1);
            dw[30] = dw[30] + _shfl_xor_150;
            float _shfl_xor_151 = __shfl_xor_sync(0xFFFFFFFF, dw[30], 2);
            dw[30] = dw[30] + _shfl_xor_151;
            float _shfl_xor_152 = __shfl_xor_sync(0xFFFFFFFF, dw[30], 4);
            dw[30] = dw[30] + _shfl_xor_152;
            float _shfl_xor_153 = __shfl_xor_sync(0xFFFFFFFF, dw[30], 8);
            dw[30] = dw[30] + _shfl_xor_153;
            float _shfl_xor_154 = __shfl_xor_sync(0xFFFFFFFF, dw[30], 16);
            dw[30] = dw[30] + _shfl_xor_154;
            float _shfl_xor_155 = __shfl_xor_sync(0xFFFFFFFF, dw[31], 1);
            dw[31] = dw[31] + _shfl_xor_155;
            float _shfl_xor_156 = __shfl_xor_sync(0xFFFFFFFF, dw[31], 2);
            dw[31] = dw[31] + _shfl_xor_156;
            float _shfl_xor_157 = __shfl_xor_sync(0xFFFFFFFF, dw[31], 4);
            dw[31] = dw[31] + _shfl_xor_157;
            float _shfl_xor_158 = __shfl_xor_sync(0xFFFFFFFF, dw[31], 8);
            dw[31] = dw[31] + _shfl_xor_158;
            float _shfl_xor_159 = __shfl_xor_sync(0xFFFFFFFF, dw[31], 16);
            dw[31] = dw[31] + _shfl_xor_159;
            float _shfl_xor_160 = __shfl_xor_sync(0xFFFFFFFF, dw[32], 1);
            dw[32] = dw[32] + _shfl_xor_160;
            float _shfl_xor_161 = __shfl_xor_sync(0xFFFFFFFF, dw[32], 2);
            dw[32] = dw[32] + _shfl_xor_161;
            float _shfl_xor_162 = __shfl_xor_sync(0xFFFFFFFF, dw[32], 4);
            dw[32] = dw[32] + _shfl_xor_162;
            float _shfl_xor_163 = __shfl_xor_sync(0xFFFFFFFF, dw[32], 8);
            dw[32] = dw[32] + _shfl_xor_163;
            float _shfl_xor_164 = __shfl_xor_sync(0xFFFFFFFF, dw[32], 16);
            dw[32] = dw[32] + _shfl_xor_164;
            float _shfl_xor_165 = __shfl_xor_sync(0xFFFFFFFF, dw[33], 1);
            dw[33] = dw[33] + _shfl_xor_165;
            float _shfl_xor_166 = __shfl_xor_sync(0xFFFFFFFF, dw[33], 2);
            dw[33] = dw[33] + _shfl_xor_166;
            float _shfl_xor_167 = __shfl_xor_sync(0xFFFFFFFF, dw[33], 4);
            dw[33] = dw[33] + _shfl_xor_167;
            float _shfl_xor_168 = __shfl_xor_sync(0xFFFFFFFF, dw[33], 8);
            dw[33] = dw[33] + _shfl_xor_168;
            float _shfl_xor_169 = __shfl_xor_sync(0xFFFFFFFF, dw[33], 16);
            dw[33] = dw[33] + _shfl_xor_169;
            float _shfl_xor_170 = __shfl_xor_sync(0xFFFFFFFF, dw[34], 1);
            dw[34] = dw[34] + _shfl_xor_170;
            float _shfl_xor_171 = __shfl_xor_sync(0xFFFFFFFF, dw[34], 2);
            dw[34] = dw[34] + _shfl_xor_171;
            float _shfl_xor_172 = __shfl_xor_sync(0xFFFFFFFF, dw[34], 4);
            dw[34] = dw[34] + _shfl_xor_172;
            float _shfl_xor_173 = __shfl_xor_sync(0xFFFFFFFF, dw[34], 8);
            dw[34] = dw[34] + _shfl_xor_173;
            float _shfl_xor_174 = __shfl_xor_sync(0xFFFFFFFF, dw[34], 16);
            dw[34] = dw[34] + _shfl_xor_174;
            float _shfl_xor_175 = __shfl_xor_sync(0xFFFFFFFF, dw[35], 1);
            dw[35] = dw[35] + _shfl_xor_175;
            float _shfl_xor_176 = __shfl_xor_sync(0xFFFFFFFF, dw[35], 2);
            dw[35] = dw[35] + _shfl_xor_176;
            float _shfl_xor_177 = __shfl_xor_sync(0xFFFFFFFF, dw[35], 4);
            dw[35] = dw[35] + _shfl_xor_177;
            float _shfl_xor_178 = __shfl_xor_sync(0xFFFFFFFF, dw[35], 8);
            dw[35] = dw[35] + _shfl_xor_178;
            float _shfl_xor_179 = __shfl_xor_sync(0xFFFFFFFF, dw[35], 16);
            dw[35] = dw[35] + _shfl_xor_179;
            float _shfl_xor_180 = __shfl_xor_sync(0xFFFFFFFF, dw[36], 1);
            dw[36] = dw[36] + _shfl_xor_180;
            float _shfl_xor_181 = __shfl_xor_sync(0xFFFFFFFF, dw[36], 2);
            dw[36] = dw[36] + _shfl_xor_181;
            float _shfl_xor_182 = __shfl_xor_sync(0xFFFFFFFF, dw[36], 4);
            dw[36] = dw[36] + _shfl_xor_182;
            float _shfl_xor_183 = __shfl_xor_sync(0xFFFFFFFF, dw[36], 8);
            dw[36] = dw[36] + _shfl_xor_183;
            float _shfl_xor_184 = __shfl_xor_sync(0xFFFFFFFF, dw[36], 16);
            dw[36] = dw[36] + _shfl_xor_184;
            float _shfl_xor_185 = __shfl_xor_sync(0xFFFFFFFF, dw[37], 1);
            dw[37] = dw[37] + _shfl_xor_185;
            float _shfl_xor_186 = __shfl_xor_sync(0xFFFFFFFF, dw[37], 2);
            dw[37] = dw[37] + _shfl_xor_186;
            float _shfl_xor_187 = __shfl_xor_sync(0xFFFFFFFF, dw[37], 4);
            dw[37] = dw[37] + _shfl_xor_187;
            float _shfl_xor_188 = __shfl_xor_sync(0xFFFFFFFF, dw[37], 8);
            dw[37] = dw[37] + _shfl_xor_188;
            float _shfl_xor_189 = __shfl_xor_sync(0xFFFFFFFF, dw[37], 16);
            dw[37] = dw[37] + _shfl_xor_189;
            float _shfl_xor_190 = __shfl_xor_sync(0xFFFFFFFF, dw[38], 1);
            dw[38] = dw[38] + _shfl_xor_190;
            float _shfl_xor_191 = __shfl_xor_sync(0xFFFFFFFF, dw[38], 2);
            dw[38] = dw[38] + _shfl_xor_191;
            float _shfl_xor_192 = __shfl_xor_sync(0xFFFFFFFF, dw[38], 4);
            dw[38] = dw[38] + _shfl_xor_192;
            float _shfl_xor_193 = __shfl_xor_sync(0xFFFFFFFF, dw[38], 8);
            dw[38] = dw[38] + _shfl_xor_193;
            float _shfl_xor_194 = __shfl_xor_sync(0xFFFFFFFF, dw[38], 16);
            dw[38] = dw[38] + _shfl_xor_194;
            float _shfl_xor_195 = __shfl_xor_sync(0xFFFFFFFF, dw[39], 1);
            dw[39] = dw[39] + _shfl_xor_195;
            float _shfl_xor_196 = __shfl_xor_sync(0xFFFFFFFF, dw[39], 2);
            dw[39] = dw[39] + _shfl_xor_196;
            float _shfl_xor_197 = __shfl_xor_sync(0xFFFFFFFF, dw[39], 4);
            dw[39] = dw[39] + _shfl_xor_197;
            float _shfl_xor_198 = __shfl_xor_sync(0xFFFFFFFF, dw[39], 8);
            dw[39] = dw[39] + _shfl_xor_198;
            float _shfl_xor_199 = __shfl_xor_sync(0xFFFFFFFF, dw[39], 16);
            dw[39] = dw[39] + _shfl_xor_199;
            float _shfl_xor_200 = __shfl_xor_sync(0xFFFFFFFF, dw[40], 1);
            dw[40] = dw[40] + _shfl_xor_200;
            float _shfl_xor_201 = __shfl_xor_sync(0xFFFFFFFF, dw[40], 2);
            dw[40] = dw[40] + _shfl_xor_201;
            float _shfl_xor_202 = __shfl_xor_sync(0xFFFFFFFF, dw[40], 4);
            dw[40] = dw[40] + _shfl_xor_202;
            float _shfl_xor_203 = __shfl_xor_sync(0xFFFFFFFF, dw[40], 8);
            dw[40] = dw[40] + _shfl_xor_203;
            float _shfl_xor_204 = __shfl_xor_sync(0xFFFFFFFF, dw[40], 16);
            dw[40] = dw[40] + _shfl_xor_204;
            float _shfl_xor_205 = __shfl_xor_sync(0xFFFFFFFF, dw[41], 1);
            dw[41] = dw[41] + _shfl_xor_205;
            float _shfl_xor_206 = __shfl_xor_sync(0xFFFFFFFF, dw[41], 2);
            dw[41] = dw[41] + _shfl_xor_206;
            float _shfl_xor_207 = __shfl_xor_sync(0xFFFFFFFF, dw[41], 4);
            dw[41] = dw[41] + _shfl_xor_207;
            float _shfl_xor_208 = __shfl_xor_sync(0xFFFFFFFF, dw[41], 8);
            dw[41] = dw[41] + _shfl_xor_208;
            float _shfl_xor_209 = __shfl_xor_sync(0xFFFFFFFF, dw[41], 16);
            dw[41] = dw[41] + _shfl_xor_209;
            float _shfl_xor_210 = __shfl_xor_sync(0xFFFFFFFF, dw[42], 1);
            dw[42] = dw[42] + _shfl_xor_210;
            float _shfl_xor_211 = __shfl_xor_sync(0xFFFFFFFF, dw[42], 2);
            dw[42] = dw[42] + _shfl_xor_211;
            float _shfl_xor_212 = __shfl_xor_sync(0xFFFFFFFF, dw[42], 4);
            dw[42] = dw[42] + _shfl_xor_212;
            float _shfl_xor_213 = __shfl_xor_sync(0xFFFFFFFF, dw[42], 8);
            dw[42] = dw[42] + _shfl_xor_213;
            float _shfl_xor_214 = __shfl_xor_sync(0xFFFFFFFF, dw[42], 16);
            dw[42] = dw[42] + _shfl_xor_214;
            float _shfl_xor_215 = __shfl_xor_sync(0xFFFFFFFF, dw[43], 1);
            dw[43] = dw[43] + _shfl_xor_215;
            float _shfl_xor_216 = __shfl_xor_sync(0xFFFFFFFF, dw[43], 2);
            dw[43] = dw[43] + _shfl_xor_216;
            float _shfl_xor_217 = __shfl_xor_sync(0xFFFFFFFF, dw[43], 4);
            dw[43] = dw[43] + _shfl_xor_217;
            float _shfl_xor_218 = __shfl_xor_sync(0xFFFFFFFF, dw[43], 8);
            dw[43] = dw[43] + _shfl_xor_218;
            float _shfl_xor_219 = __shfl_xor_sync(0xFFFFFFFF, dw[43], 16);
            dw[43] = dw[43] + _shfl_xor_219;
            float _shfl_xor_220 = __shfl_xor_sync(0xFFFFFFFF, dw[44], 1);
            dw[44] = dw[44] + _shfl_xor_220;
            float _shfl_xor_221 = __shfl_xor_sync(0xFFFFFFFF, dw[44], 2);
            dw[44] = dw[44] + _shfl_xor_221;
            float _shfl_xor_222 = __shfl_xor_sync(0xFFFFFFFF, dw[44], 4);
            dw[44] = dw[44] + _shfl_xor_222;
            float _shfl_xor_223 = __shfl_xor_sync(0xFFFFFFFF, dw[44], 8);
            dw[44] = dw[44] + _shfl_xor_223;
            float _shfl_xor_224 = __shfl_xor_sync(0xFFFFFFFF, dw[44], 16);
            dw[44] = dw[44] + _shfl_xor_224;
            float _shfl_xor_225 = __shfl_xor_sync(0xFFFFFFFF, dw[45], 1);
            dw[45] = dw[45] + _shfl_xor_225;
            float _shfl_xor_226 = __shfl_xor_sync(0xFFFFFFFF, dw[45], 2);
            dw[45] = dw[45] + _shfl_xor_226;
            float _shfl_xor_227 = __shfl_xor_sync(0xFFFFFFFF, dw[45], 4);
            dw[45] = dw[45] + _shfl_xor_227;
            float _shfl_xor_228 = __shfl_xor_sync(0xFFFFFFFF, dw[45], 8);
            dw[45] = dw[45] + _shfl_xor_228;
            float _shfl_xor_229 = __shfl_xor_sync(0xFFFFFFFF, dw[45], 16);
            dw[45] = dw[45] + _shfl_xor_229;
            float _shfl_xor_230 = __shfl_xor_sync(0xFFFFFFFF, dw[46], 1);
            dw[46] = dw[46] + _shfl_xor_230;
            float _shfl_xor_231 = __shfl_xor_sync(0xFFFFFFFF, dw[46], 2);
            dw[46] = dw[46] + _shfl_xor_231;
            float _shfl_xor_232 = __shfl_xor_sync(0xFFFFFFFF, dw[46], 4);
            dw[46] = dw[46] + _shfl_xor_232;
            float _shfl_xor_233 = __shfl_xor_sync(0xFFFFFFFF, dw[46], 8);
            dw[46] = dw[46] + _shfl_xor_233;
            float _shfl_xor_234 = __shfl_xor_sync(0xFFFFFFFF, dw[46], 16);
            dw[46] = dw[46] + _shfl_xor_234;
            float _shfl_xor_235 = __shfl_xor_sync(0xFFFFFFFF, dw[47], 1);
            dw[47] = dw[47] + _shfl_xor_235;
            float _shfl_xor_236 = __shfl_xor_sync(0xFFFFFFFF, dw[47], 2);
            dw[47] = dw[47] + _shfl_xor_236;
            float _shfl_xor_237 = __shfl_xor_sync(0xFFFFFFFF, dw[47], 4);
            dw[47] = dw[47] + _shfl_xor_237;
            float _shfl_xor_238 = __shfl_xor_sync(0xFFFFFFFF, dw[47], 8);
            dw[47] = dw[47] + _shfl_xor_238;
            float _shfl_xor_239 = __shfl_xor_sync(0xFFFFFFFF, dw[47], 16);
            dw[47] = dw[47] + _shfl_xor_239;
            float _shfl_xor_240 = __shfl_xor_sync(0xFFFFFFFF, dw[48], 1);
            dw[48] = dw[48] + _shfl_xor_240;
            float _shfl_xor_241 = __shfl_xor_sync(0xFFFFFFFF, dw[48], 2);
            dw[48] = dw[48] + _shfl_xor_241;
            float _shfl_xor_242 = __shfl_xor_sync(0xFFFFFFFF, dw[48], 4);
            dw[48] = dw[48] + _shfl_xor_242;
            float _shfl_xor_243 = __shfl_xor_sync(0xFFFFFFFF, dw[48], 8);
            dw[48] = dw[48] + _shfl_xor_243;
            float _shfl_xor_244 = __shfl_xor_sync(0xFFFFFFFF, dw[48], 16);
            dw[48] = dw[48] + _shfl_xor_244;
            float _shfl_xor_245 = __shfl_xor_sync(0xFFFFFFFF, dw[49], 1);
            dw[49] = dw[49] + _shfl_xor_245;
            float _shfl_xor_246 = __shfl_xor_sync(0xFFFFFFFF, dw[49], 2);
            dw[49] = dw[49] + _shfl_xor_246;
            float _shfl_xor_247 = __shfl_xor_sync(0xFFFFFFFF, dw[49], 4);
            dw[49] = dw[49] + _shfl_xor_247;
            float _shfl_xor_248 = __shfl_xor_sync(0xFFFFFFFF, dw[49], 8);
            dw[49] = dw[49] + _shfl_xor_248;
            float _shfl_xor_249 = __shfl_xor_sync(0xFFFFFFFF, dw[49], 16);
            dw[49] = dw[49] + _shfl_xor_249;
            float _shfl_xor_250 = __shfl_xor_sync(0xFFFFFFFF, dw[50], 1);
            dw[50] = dw[50] + _shfl_xor_250;
            float _shfl_xor_251 = __shfl_xor_sync(0xFFFFFFFF, dw[50], 2);
            dw[50] = dw[50] + _shfl_xor_251;
            float _shfl_xor_252 = __shfl_xor_sync(0xFFFFFFFF, dw[50], 4);
            dw[50] = dw[50] + _shfl_xor_252;
            float _shfl_xor_253 = __shfl_xor_sync(0xFFFFFFFF, dw[50], 8);
            dw[50] = dw[50] + _shfl_xor_253;
            float _shfl_xor_254 = __shfl_xor_sync(0xFFFFFFFF, dw[50], 16);
            dw[50] = dw[50] + _shfl_xor_254;
            float _shfl_xor_255 = __shfl_xor_sync(0xFFFFFFFF, dw[51], 1);
            dw[51] = dw[51] + _shfl_xor_255;
            float _shfl_xor_256 = __shfl_xor_sync(0xFFFFFFFF, dw[51], 2);
            dw[51] = dw[51] + _shfl_xor_256;
            float _shfl_xor_257 = __shfl_xor_sync(0xFFFFFFFF, dw[51], 4);
            dw[51] = dw[51] + _shfl_xor_257;
            float _shfl_xor_258 = __shfl_xor_sync(0xFFFFFFFF, dw[51], 8);
            dw[51] = dw[51] + _shfl_xor_258;
            float _shfl_xor_259 = __shfl_xor_sync(0xFFFFFFFF, dw[51], 16);
            dw[51] = dw[51] + _shfl_xor_259;
            float _shfl_xor_260 = __shfl_xor_sync(0xFFFFFFFF, dw[52], 1);
            dw[52] = dw[52] + _shfl_xor_260;
            float _shfl_xor_261 = __shfl_xor_sync(0xFFFFFFFF, dw[52], 2);
            dw[52] = dw[52] + _shfl_xor_261;
            float _shfl_xor_262 = __shfl_xor_sync(0xFFFFFFFF, dw[52], 4);
            dw[52] = dw[52] + _shfl_xor_262;
            float _shfl_xor_263 = __shfl_xor_sync(0xFFFFFFFF, dw[52], 8);
            dw[52] = dw[52] + _shfl_xor_263;
            float _shfl_xor_264 = __shfl_xor_sync(0xFFFFFFFF, dw[52], 16);
            dw[52] = dw[52] + _shfl_xor_264;
            float _shfl_xor_265 = __shfl_xor_sync(0xFFFFFFFF, dw[53], 1);
            dw[53] = dw[53] + _shfl_xor_265;
            float _shfl_xor_266 = __shfl_xor_sync(0xFFFFFFFF, dw[53], 2);
            dw[53] = dw[53] + _shfl_xor_266;
            float _shfl_xor_267 = __shfl_xor_sync(0xFFFFFFFF, dw[53], 4);
            dw[53] = dw[53] + _shfl_xor_267;
            float _shfl_xor_268 = __shfl_xor_sync(0xFFFFFFFF, dw[53], 8);
            dw[53] = dw[53] + _shfl_xor_268;
            float _shfl_xor_269 = __shfl_xor_sync(0xFFFFFFFF, dw[53], 16);
            dw[53] = dw[53] + _shfl_xor_269;
            float _shfl_xor_270 = __shfl_xor_sync(0xFFFFFFFF, dw[54], 1);
            dw[54] = dw[54] + _shfl_xor_270;
            float _shfl_xor_271 = __shfl_xor_sync(0xFFFFFFFF, dw[54], 2);
            dw[54] = dw[54] + _shfl_xor_271;
            float _shfl_xor_272 = __shfl_xor_sync(0xFFFFFFFF, dw[54], 4);
            dw[54] = dw[54] + _shfl_xor_272;
            float _shfl_xor_273 = __shfl_xor_sync(0xFFFFFFFF, dw[54], 8);
            dw[54] = dw[54] + _shfl_xor_273;
            float _shfl_xor_274 = __shfl_xor_sync(0xFFFFFFFF, dw[54], 16);
            dw[54] = dw[54] + _shfl_xor_274;
            float _shfl_xor_275 = __shfl_xor_sync(0xFFFFFFFF, dw[55], 1);
            dw[55] = dw[55] + _shfl_xor_275;
            float _shfl_xor_276 = __shfl_xor_sync(0xFFFFFFFF, dw[55], 2);
            dw[55] = dw[55] + _shfl_xor_276;
            float _shfl_xor_277 = __shfl_xor_sync(0xFFFFFFFF, dw[55], 4);
            dw[55] = dw[55] + _shfl_xor_277;
            float _shfl_xor_278 = __shfl_xor_sync(0xFFFFFFFF, dw[55], 8);
            dw[55] = dw[55] + _shfl_xor_278;
            float _shfl_xor_279 = __shfl_xor_sync(0xFFFFFFFF, dw[55], 16);
            dw[55] = dw[55] + _shfl_xor_279;
            float _shfl_xor_280 = __shfl_xor_sync(0xFFFFFFFF, dw[56], 1);
            dw[56] = dw[56] + _shfl_xor_280;
            float _shfl_xor_281 = __shfl_xor_sync(0xFFFFFFFF, dw[56], 2);
            dw[56] = dw[56] + _shfl_xor_281;
            float _shfl_xor_282 = __shfl_xor_sync(0xFFFFFFFF, dw[56], 4);
            dw[56] = dw[56] + _shfl_xor_282;
            float _shfl_xor_283 = __shfl_xor_sync(0xFFFFFFFF, dw[56], 8);
            dw[56] = dw[56] + _shfl_xor_283;
            float _shfl_xor_284 = __shfl_xor_sync(0xFFFFFFFF, dw[56], 16);
            dw[56] = dw[56] + _shfl_xor_284;
            float _shfl_xor_285 = __shfl_xor_sync(0xFFFFFFFF, dw[57], 1);
            dw[57] = dw[57] + _shfl_xor_285;
            float _shfl_xor_286 = __shfl_xor_sync(0xFFFFFFFF, dw[57], 2);
            dw[57] = dw[57] + _shfl_xor_286;
            float _shfl_xor_287 = __shfl_xor_sync(0xFFFFFFFF, dw[57], 4);
            dw[57] = dw[57] + _shfl_xor_287;
            float _shfl_xor_288 = __shfl_xor_sync(0xFFFFFFFF, dw[57], 8);
            dw[57] = dw[57] + _shfl_xor_288;
            float _shfl_xor_289 = __shfl_xor_sync(0xFFFFFFFF, dw[57], 16);
            dw[57] = dw[57] + _shfl_xor_289;
            float _shfl_xor_290 = __shfl_xor_sync(0xFFFFFFFF, dw[58], 1);
            dw[58] = dw[58] + _shfl_xor_290;
            float _shfl_xor_291 = __shfl_xor_sync(0xFFFFFFFF, dw[58], 2);
            dw[58] = dw[58] + _shfl_xor_291;
            float _shfl_xor_292 = __shfl_xor_sync(0xFFFFFFFF, dw[58], 4);
            dw[58] = dw[58] + _shfl_xor_292;
            float _shfl_xor_293 = __shfl_xor_sync(0xFFFFFFFF, dw[58], 8);
            dw[58] = dw[58] + _shfl_xor_293;
            float _shfl_xor_294 = __shfl_xor_sync(0xFFFFFFFF, dw[58], 16);
            dw[58] = dw[58] + _shfl_xor_294;
            float _shfl_xor_295 = __shfl_xor_sync(0xFFFFFFFF, dw[59], 1);
            dw[59] = dw[59] + _shfl_xor_295;
            float _shfl_xor_296 = __shfl_xor_sync(0xFFFFFFFF, dw[59], 2);
            dw[59] = dw[59] + _shfl_xor_296;
            float _shfl_xor_297 = __shfl_xor_sync(0xFFFFFFFF, dw[59], 4);
            dw[59] = dw[59] + _shfl_xor_297;
            float _shfl_xor_298 = __shfl_xor_sync(0xFFFFFFFF, dw[59], 8);
            dw[59] = dw[59] + _shfl_xor_298;
            float _shfl_xor_299 = __shfl_xor_sync(0xFFFFFFFF, dw[59], 16);
            dw[59] = dw[59] + _shfl_xor_299;
            float _shfl_xor_300 = __shfl_xor_sync(0xFFFFFFFF, dw[60], 1);
            dw[60] = dw[60] + _shfl_xor_300;
            float _shfl_xor_301 = __shfl_xor_sync(0xFFFFFFFF, dw[60], 2);
            dw[60] = dw[60] + _shfl_xor_301;
            float _shfl_xor_302 = __shfl_xor_sync(0xFFFFFFFF, dw[60], 4);
            dw[60] = dw[60] + _shfl_xor_302;
            float _shfl_xor_303 = __shfl_xor_sync(0xFFFFFFFF, dw[60], 8);
            dw[60] = dw[60] + _shfl_xor_303;
            float _shfl_xor_304 = __shfl_xor_sync(0xFFFFFFFF, dw[60], 16);
            dw[60] = dw[60] + _shfl_xor_304;
            float _shfl_xor_305 = __shfl_xor_sync(0xFFFFFFFF, dw[61], 1);
            dw[61] = dw[61] + _shfl_xor_305;
            float _shfl_xor_306 = __shfl_xor_sync(0xFFFFFFFF, dw[61], 2);
            dw[61] = dw[61] + _shfl_xor_306;
            float _shfl_xor_307 = __shfl_xor_sync(0xFFFFFFFF, dw[61], 4);
            dw[61] = dw[61] + _shfl_xor_307;
            float _shfl_xor_308 = __shfl_xor_sync(0xFFFFFFFF, dw[61], 8);
            dw[61] = dw[61] + _shfl_xor_308;
            float _shfl_xor_309 = __shfl_xor_sync(0xFFFFFFFF, dw[61], 16);
            dw[61] = dw[61] + _shfl_xor_309;
            float _shfl_xor_310 = __shfl_xor_sync(0xFFFFFFFF, dw[62], 1);
            dw[62] = dw[62] + _shfl_xor_310;
            float _shfl_xor_311 = __shfl_xor_sync(0xFFFFFFFF, dw[62], 2);
            dw[62] = dw[62] + _shfl_xor_311;
            float _shfl_xor_312 = __shfl_xor_sync(0xFFFFFFFF, dw[62], 4);
            dw[62] = dw[62] + _shfl_xor_312;
            float _shfl_xor_313 = __shfl_xor_sync(0xFFFFFFFF, dw[62], 8);
            dw[62] = dw[62] + _shfl_xor_313;
            float _shfl_xor_314 = __shfl_xor_sync(0xFFFFFFFF, dw[62], 16);
            dw[62] = dw[62] + _shfl_xor_314;
            float _shfl_xor_315 = __shfl_xor_sync(0xFFFFFFFF, dw[63], 1);
            dw[63] = dw[63] + _shfl_xor_315;
            float _shfl_xor_316 = __shfl_xor_sync(0xFFFFFFFF, dw[63], 2);
            dw[63] = dw[63] + _shfl_xor_316;
            float _shfl_xor_317 = __shfl_xor_sync(0xFFFFFFFF, dw[63], 4);
            dw[63] = dw[63] + _shfl_xor_317;
            float _shfl_xor_318 = __shfl_xor_sync(0xFFFFFFFF, dw[63], 8);
            dw[63] = dw[63] + _shfl_xor_318;
            float _shfl_xor_319 = __shfl_xor_sync(0xFFFFFFFF, dw[63], 16);
            dw[63] = dw[63] + _shfl_xor_319;
            if (lane_c == 0) {
                s_dw[warp_c * 64] = dw[0];
                s_dw[warp_c * 64 + 1] = dw[1];
                s_dw[warp_c * 64 + 2] = dw[2];
                s_dw[warp_c * 64 + 3] = dw[3];
                s_dw[warp_c * 64 + 4] = dw[4];
                s_dw[warp_c * 64 + 5] = dw[5];
                s_dw[warp_c * 64 + 6] = dw[6];
                s_dw[warp_c * 64 + 7] = dw[7];
                s_dw[warp_c * 64 + 8] = dw[8];
                s_dw[warp_c * 64 + 9] = dw[9];
                s_dw[warp_c * 64 + 10] = dw[10];
                s_dw[warp_c * 64 + 11] = dw[11];
                s_dw[warp_c * 64 + 12] = dw[12];
                s_dw[warp_c * 64 + 13] = dw[13];
                s_dw[warp_c * 64 + 14] = dw[14];
                s_dw[warp_c * 64 + 15] = dw[15];
                s_dw[warp_c * 64 + 16] = dw[16];
                s_dw[warp_c * 64 + 17] = dw[17];
                s_dw[warp_c * 64 + 18] = dw[18];
                s_dw[warp_c * 64 + 19] = dw[19];
                s_dw[warp_c * 64 + 20] = dw[20];
                s_dw[warp_c * 64 + 21] = dw[21];
                s_dw[warp_c * 64 + 22] = dw[22];
                s_dw[warp_c * 64 + 23] = dw[23];
                s_dw[warp_c * 64 + 24] = dw[24];
                s_dw[warp_c * 64 + 25] = dw[25];
                s_dw[warp_c * 64 + 26] = dw[26];
                s_dw[warp_c * 64 + 27] = dw[27];
                s_dw[warp_c * 64 + 28] = dw[28];
                s_dw[warp_c * 64 + 29] = dw[29];
                s_dw[warp_c * 64 + 30] = dw[30];
                s_dw[warp_c * 64 + 31] = dw[31];
                s_dw[warp_c * 64 + 32] = dw[32];
                s_dw[warp_c * 64 + 33] = dw[33];
                s_dw[warp_c * 64 + 34] = dw[34];
                s_dw[warp_c * 64 + 35] = dw[35];
                s_dw[warp_c * 64 + 36] = dw[36];
                s_dw[warp_c * 64 + 37] = dw[37];
                s_dw[warp_c * 64 + 38] = dw[38];
                s_dw[warp_c * 64 + 39] = dw[39];
                s_dw[warp_c * 64 + 40] = dw[40];
                s_dw[warp_c * 64 + 41] = dw[41];
                s_dw[warp_c * 64 + 42] = dw[42];
                s_dw[warp_c * 64 + 43] = dw[43];
                s_dw[warp_c * 64 + 44] = dw[44];
                s_dw[warp_c * 64 + 45] = dw[45];
                s_dw[warp_c * 64 + 46] = dw[46];
                s_dw[warp_c * 64 + 47] = dw[47];
                s_dw[warp_c * 64 + 48] = dw[48];
                s_dw[warp_c * 64 + 49] = dw[49];
                s_dw[warp_c * 64 + 50] = dw[50];
                s_dw[warp_c * 64 + 51] = dw[51];
                s_dw[warp_c * 64 + 52] = dw[52];
                s_dw[warp_c * 64 + 53] = dw[53];
                s_dw[warp_c * 64 + 54] = dw[54];
                s_dw[warp_c * 64 + 55] = dw[55];
                s_dw[warp_c * 64 + 56] = dw[56];
                s_dw[warp_c * 64 + 57] = dw[57];
                s_dw[warp_c * 64 + 58] = dw[58];
                s_dw[warp_c * 64 + 59] = dw[59];
                s_dw[warp_c * 64 + 60] = dw[60];
                s_dw[warp_c * 64 + 61] = dw[61];
                s_dw[warp_c * 64 + 62] = dw[62];
                s_dw[warp_c * 64 + 63] = dw[63];
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (tid_1 < 64) {
                float total = s_dw[tid_1];
                total = total + s_dw[64 + tid_1];
                total = total + s_dw[128 + tid_1];
                total = total + s_dw[192 + tid_1];
                d_weights[q_row0 + (long long)tid_1] = total;
            }
        }
    }
    // ---- Role: gather ----
    if (warp >= 4 && warp <= 5) {
        { // gather_main
            int pw = warp - 4;
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_g = blocks_word[0];
            if (elect_sync()) {
                #pragma unroll 1
                for (int i_1 = 0; i_1 < blocks_g; i_1++) {
                    int slot_g = i_1 % 6;
                    int st = i_1 % 3;
                    mbarrier_wait(idx_full_addr + (slot_g) * 8, i_1 / 6 & 1);
                    mbarrier_wait(k_free_addr + (st) * 8, i_1 / 3 & 1 ^ 1);
                    mbarrier_arrive_expect_tx(k_full_addr + (st) * 8, 16384);
                    int kq = slot_g * 128 + pw * 64;
                    int r0 = s_idx[kq];
                    int r1 = s_idx[kq + 1];
                    int r2 = s_idx[kq + 2];
                    int r3 = s_idx[kq + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192)), "l"((&index_k)), "r"(0), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192)), "l"((&index_k)), "r"(64), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_0 = slot_g * 128 + pw * 64 + 4;
                    int r0_1 = s_idx[kq_0];
                    int r1_2 = s_idx[kq_0 + 1];
                    int r2_3 = s_idx[kq_0 + 2];
                    int r3_4 = s_idx[kq_0 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 512), "l"((&index_k)), "r"(0), "r"(r0_1), "r"(r1_2), "r"(r2_3), "r"(r3_4), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 512), "l"((&index_k)), "r"(64), "r"(r0_1), "r"(r1_2), "r"(r2_3), "r"(r3_4), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_5 = slot_g * 128 + pw * 64 + 8;
                    int r0_6 = s_idx[kq_5];
                    int r1_7 = s_idx[kq_5 + 1];
                    int r2_8 = s_idx[kq_5 + 2];
                    int r3_9 = s_idx[kq_5 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 1024), "l"((&index_k)), "r"(0), "r"(r0_6), "r"(r1_7), "r"(r2_8), "r"(r3_9), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 1024), "l"((&index_k)), "r"(64), "r"(r0_6), "r"(r1_7), "r"(r2_8), "r"(r3_9), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_10 = slot_g * 128 + pw * 64 + 12;
                    int r0_11 = s_idx[kq_10];
                    int r1_12 = s_idx[kq_10 + 1];
                    int r2_13 = s_idx[kq_10 + 2];
                    int r3_14 = s_idx[kq_10 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 1536), "l"((&index_k)), "r"(0), "r"(r0_11), "r"(r1_12), "r"(r2_13), "r"(r3_14), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 1536), "l"((&index_k)), "r"(64), "r"(r0_11), "r"(r1_12), "r"(r2_13), "r"(r3_14), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_15 = slot_g * 128 + pw * 64 + 16;
                    int r0_16 = s_idx[kq_15];
                    int r1_17 = s_idx[kq_15 + 1];
                    int r2_18 = s_idx[kq_15 + 2];
                    int r3_19 = s_idx[kq_15 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 2048), "l"((&index_k)), "r"(0), "r"(r0_16), "r"(r1_17), "r"(r2_18), "r"(r3_19), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 2048), "l"((&index_k)), "r"(64), "r"(r0_16), "r"(r1_17), "r"(r2_18), "r"(r3_19), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_20 = slot_g * 128 + pw * 64 + 20;
                    int r0_21 = s_idx[kq_20];
                    int r1_22 = s_idx[kq_20 + 1];
                    int r2_23 = s_idx[kq_20 + 2];
                    int r3_24 = s_idx[kq_20 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 2560), "l"((&index_k)), "r"(0), "r"(r0_21), "r"(r1_22), "r"(r2_23), "r"(r3_24), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 2560), "l"((&index_k)), "r"(64), "r"(r0_21), "r"(r1_22), "r"(r2_23), "r"(r3_24), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_25 = slot_g * 128 + pw * 64 + 24;
                    int r0_26 = s_idx[kq_25];
                    int r1_27 = s_idx[kq_25 + 1];
                    int r2_28 = s_idx[kq_25 + 2];
                    int r3_29 = s_idx[kq_25 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 3072), "l"((&index_k)), "r"(0), "r"(r0_26), "r"(r1_27), "r"(r2_28), "r"(r3_29), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 3072), "l"((&index_k)), "r"(64), "r"(r0_26), "r"(r1_27), "r"(r2_28), "r"(r3_29), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_30 = slot_g * 128 + pw * 64 + 28;
                    int r0_31 = s_idx[kq_30];
                    int r1_32 = s_idx[kq_30 + 1];
                    int r2_33 = s_idx[kq_30 + 2];
                    int r3_34 = s_idx[kq_30 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 3584), "l"((&index_k)), "r"(0), "r"(r0_31), "r"(r1_32), "r"(r2_33), "r"(r3_34), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 3584), "l"((&index_k)), "r"(64), "r"(r0_31), "r"(r1_32), "r"(r2_33), "r"(r3_34), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_35 = slot_g * 128 + pw * 64 + 32;
                    int r0_36 = s_idx[kq_35];
                    int r1_37 = s_idx[kq_35 + 1];
                    int r2_38 = s_idx[kq_35 + 2];
                    int r3_39 = s_idx[kq_35 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 4096), "l"((&index_k)), "r"(0), "r"(r0_36), "r"(r1_37), "r"(r2_38), "r"(r3_39), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 4096), "l"((&index_k)), "r"(64), "r"(r0_36), "r"(r1_37), "r"(r2_38), "r"(r3_39), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_40 = slot_g * 128 + pw * 64 + 36;
                    int r0_41 = s_idx[kq_40];
                    int r1_42 = s_idx[kq_40 + 1];
                    int r2_43 = s_idx[kq_40 + 2];
                    int r3_44 = s_idx[kq_40 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 4608), "l"((&index_k)), "r"(0), "r"(r0_41), "r"(r1_42), "r"(r2_43), "r"(r3_44), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 4608), "l"((&index_k)), "r"(64), "r"(r0_41), "r"(r1_42), "r"(r2_43), "r"(r3_44), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_45 = slot_g * 128 + pw * 64 + 40;
                    int r0_46 = s_idx[kq_45];
                    int r1_47 = s_idx[kq_45 + 1];
                    int r2_48 = s_idx[kq_45 + 2];
                    int r3_49 = s_idx[kq_45 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 5120), "l"((&index_k)), "r"(0), "r"(r0_46), "r"(r1_47), "r"(r2_48), "r"(r3_49), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 5120), "l"((&index_k)), "r"(64), "r"(r0_46), "r"(r1_47), "r"(r2_48), "r"(r3_49), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_50 = slot_g * 128 + pw * 64 + 44;
                    int r0_51 = s_idx[kq_50];
                    int r1_52 = s_idx[kq_50 + 1];
                    int r2_53 = s_idx[kq_50 + 2];
                    int r3_54 = s_idx[kq_50 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 5632), "l"((&index_k)), "r"(0), "r"(r0_51), "r"(r1_52), "r"(r2_53), "r"(r3_54), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 5632), "l"((&index_k)), "r"(64), "r"(r0_51), "r"(r1_52), "r"(r2_53), "r"(r3_54), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_55 = slot_g * 128 + pw * 64 + 48;
                    int r0_56 = s_idx[kq_55];
                    int r1_57 = s_idx[kq_55 + 1];
                    int r2_58 = s_idx[kq_55 + 2];
                    int r3_59 = s_idx[kq_55 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 6144), "l"((&index_k)), "r"(0), "r"(r0_56), "r"(r1_57), "r"(r2_58), "r"(r3_59), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 6144), "l"((&index_k)), "r"(64), "r"(r0_56), "r"(r1_57), "r"(r2_58), "r"(r3_59), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_60 = slot_g * 128 + pw * 64 + 52;
                    int r0_61 = s_idx[kq_60];
                    int r1_62 = s_idx[kq_60 + 1];
                    int r2_63 = s_idx[kq_60 + 2];
                    int r3_64 = s_idx[kq_60 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 6656), "l"((&index_k)), "r"(0), "r"(r0_61), "r"(r1_62), "r"(r2_63), "r"(r3_64), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 6656), "l"((&index_k)), "r"(64), "r"(r0_61), "r"(r1_62), "r"(r2_63), "r"(r3_64), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_65 = slot_g * 128 + pw * 64 + 56;
                    int r0_66 = s_idx[kq_65];
                    int r1_67 = s_idx[kq_65 + 1];
                    int r2_68 = s_idx[kq_65 + 2];
                    int r3_69 = s_idx[kq_65 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 7168), "l"((&index_k)), "r"(0), "r"(r0_66), "r"(r1_67), "r"(r2_68), "r"(r3_69), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 7168), "l"((&index_k)), "r"(64), "r"(r0_66), "r"(r1_67), "r"(r2_68), "r"(r3_69), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    int kq_70 = slot_g * 128 + pw * 64 + 60;
                    int r0_71 = s_idx[kq_70];
                    int r1_72 = s_idx[kq_70 + 1];
                    int r2_73 = s_idx[kq_70 + 2];
                    int r3_74 = s_idx[kq_70 + 3];
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)(2 * st * 16384) + (unsigned int)(pw * 8192) + 7680), "l"((&index_k)), "r"(0), "r"(r0_71), "r"(r1_72), "r"(r2_73), "r"(r3_74), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                        ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                        :: "r"(k_blk_addr + (unsigned int)((2 * st + 1) * 16384) + (unsigned int)(pw * 8192) + 7680), "l"((&index_k)), "r"(64), "r"(r0_71), "r"(r1_72), "r"(r2_73), "r"(r3_74), "r"(k_full_addr + (st) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    mbarrier_arrive(idx_free_addr + (slot_g) * 8);
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 6) {
        { // mma_main
            mbarrier_wait(blocks_ready_addr, 0);
            int blocks_m = blocks_word[0];
            mbarrier_wait(q_full_addr, 0);
            if (elect_sync()) {
                #pragma unroll 1
                for (int i_2 = 0; i_2 < blocks_m; i_2++) {
                    int par = i_2 & 1;
                    int use = i_2 / 2 & 1;
                    int sm = i_2 % 3;
                    mbarrier_wait(k_full_addr + (sm) * 8, i_2 / 3 & 1);
                    mbarrier_wait(l_free_addr + (par) * 8, use ^ 1);
                    int _mma_a_lo_0 = (((k_k_addr) >> 4) & 0x3FFF) + (sm) * 2048;
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
                    "mov.b32 id, 135267472;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem + (par * 64))), "r"(0));
                    tcgen05_commit(l_full_addr + (par) * 8);
                    mbarrier_wait(g_full_addr + (par) * 8, use);
                    int _mma_a_lo_1 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (sm) * 2048;
                    int _mma_b_lo_1 = ((((gt_k_addr) >> 4) & 0x3FFF) | 0x2000000) + (2 * par) * 1024;
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
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem + (128))), "r"(((i_2 == 0) ? 0 : 1)));
                    int _mma_a_lo_2 = ((((k_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (sm) * 2048;
                    int _mma_b_lo_2 = ((((gt_k_addr) >> 4) & 0x3FFF) | 0x2000000) + (2 * par + 1) * 1024;
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
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
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
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem + (128))), "r"(1));
                    tcgen05_commit(k_free_addr + (sm) * 8);
                    mbarrier_wait(dk_free_addr + (par) * 8, use ^ 1);
                    int _mma_a_lo_3 = ((((gt_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2 * par) * 1024;
                    int _mma_b_lo_3 = (((q_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 id, 136414352;\n\t"
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_tmem + (256 + par * 128))), "r"(0));
                    int _mma_a_lo_4 = ((((gt_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (2 * par + 1) * 1024;
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
                    "mov.b32 id, 136414352;\n\t"
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_3), "r"((tmem_tmem + (256 + par * 128))), "r"(1));
                    tcgen05_commit(g_free_addr + (par) * 8);
                    tcgen05_commit(dk_full_addr + (par) * 8);
                }
                tcgen05_commit(dq_done_addr);
            }
        }
    }
    // ---- Role: metadata ----
    if (warp == 7) {
        { // metadata_main
            int row = blockIdx.x;
            int lane_0 = lane;
            long long row_base = (long long)row * (long long)topk;
            int last_valid = -1;
            for (int blk = (topk + 128 - 1) / 128 - 1; blk >= 0; blk--) {
                int pos = blk * 128 + lane_0 * 4;
                int best = -1;
                if (pos < topk) {
                    int _vec_load_0[4];
                    {
                        const int4* _ivptr_0 = reinterpret_cast<const int4*>(topk_indices + (row_base + (long long)pos) + 0);
                        int4 _ivld_0;
                        _ivld_0 = *_ivptr_0;
                        _vec_load_0[0 + 0] = _ivld_0.x;
                        _vec_load_0[0 + 1] = _ivld_0.y;
                        _vec_load_0[0 + 2] = _ivld_0.z;
                        _vec_load_0[0 + 3] = _ivld_0.w;
                    }
                    if (_vec_load_0[0] >= 0) {
                        best = pos;
                    }
                    if (_vec_load_0[1] >= 0) {
                        best = pos + 1;
                    }
                    if (_vec_load_0[2] >= 0) {
                        best = pos + 2;
                    }
                    if (_vec_load_0[3] >= 0) {
                        best = pos + 3;
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
            int _max_0 = (((last_valid + 128) / 128) > (1) ? ((last_valid + 128) / 128) : (1));
            int blocks = _max_0;
            if (lane_0 == 0) {
                blocks_word[0] = blocks;
                mbarrier_arrive(blocks_ready_addr);
            }
            #pragma unroll 1
            for (int i_3 = 0; i_3 < blocks; i_3++) {
                int slot = i_3 % 6;
                int pos2 = i_3 * 128 + lane_0 * 4;
                int vals[4];
                float grads[4];
                if (pos2 < topk) {
                    int _vec_load_1[4];
                    {
                        const int4* _ivptr_1 = reinterpret_cast<const int4*>(topk_indices + (row_base + (long long)pos2) + 0);
                        int4 _ivld_1;
                        _ivld_1 = *_ivptr_1;
                        _vec_load_1[0 + 0] = _ivld_1.x;
                        _vec_load_1[0 + 1] = _ivld_1.y;
                        _vec_load_1[0 + 2] = _ivld_1.z;
                        _vec_load_1[0 + 3] = _ivld_1.w;
                    }
                    float _vec_load_2[4];
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(grad_scores + (row_base + (long long)pos2) + 0);
                        _vec_load_2[0 + 0] = _v4.x;
                        _vec_load_2[0 + 1] = _v4.y;
                        _vec_load_2[0 + 2] = _v4.z;
                        _vec_load_2[0 + 3] = _v4.w;
                    }
                    vals[0] = _vec_load_1[0];
                    grads[0] = _vec_load_2[0];
                    vals[1] = _vec_load_1[1];
                    grads[1] = _vec_load_2[1];
                    vals[2] = _vec_load_1[2];
                    grads[2] = _vec_load_2[2];
                    vals[3] = _vec_load_1[3];
                    grads[3] = _vec_load_2[3];
                } else {
                    vals[0] = -1;
                    grads[0] = 0.0f;
                    vals[1] = -1;
                    grads[1] = 0.0f;
                    vals[2] = -1;
                    grads[2] = 0.0f;
                    vals[3] = -1;
                    grads[3] = 0.0f;
                }
                mbarrier_wait(idx_free_addr + (slot) * 8, i_3 / 6 & 1 ^ 1);
                int valid = (int)(vals[0] >= 0);
                if (valid == 0) {
                    vals[0] = -1;
                    grads[0] = 0.0f;
                }
                s_idx[slot * 128 + lane_0 * 4] = vals[0];
                s_grad[slot * 128 + lane_0 * 4] = grads[0];
                int valid_0 = (int)(vals[1] >= 0);
                if (valid_0 == 0) {
                    vals[1] = -1;
                    grads[1] = 0.0f;
                }
                s_idx[slot * 128 + lane_0 * 4 + 1] = vals[1];
                s_grad[slot * 128 + lane_0 * 4 + 1] = grads[1];
                int valid_1 = (int)(vals[2] >= 0);
                if (valid_1 == 0) {
                    vals[2] = -1;
                    grads[2] = 0.0f;
                }
                s_idx[slot * 128 + lane_0 * 4 + 2] = vals[2];
                s_grad[slot * 128 + lane_0 * 4 + 2] = grads[2];
                int valid_2 = (int)(vals[3] >= 0);
                if (valid_2 == 0) {
                    vals[3] = -1;
                    grads[3] = 0.0f;
                }
                s_idx[slot * 128 + lane_0 * 4 + 3] = vals[3];
                s_grad[slot * 128 + lane_0 * 4 + 3] = grads[3];
                mbarrier_arrive(idx_full_addr + (slot) * 8);
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 6) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
