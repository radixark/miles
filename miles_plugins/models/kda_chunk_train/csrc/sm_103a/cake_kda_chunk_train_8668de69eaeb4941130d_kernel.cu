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
#define TMEM_ACC_U_OFFSET 0
#define TMEM_ACC_W_OFFSET 256
#define NUM_RING_STAGES 2
#define NUM_ACCS_STAGES 2
#define NUM_ONE_STAGES 1
#define SMEM_AKK_BLK0_OFF 1024
#define SMEM_AKK_BLK0_STAGE_BYTES 8192
#define SMEM_AKK_BLK0_STRIDE 98304
#define SMEM_AKK_BLK3_OFF 25600
#define SMEM_AKK_BLK3_STAGE_BYTES 8192
#define SMEM_AKK_BLK3_STRIDE 98304
#define SMEM_VB_S0_OFF 33792
#define SMEM_VB_S0_STAGE_BYTES 16384
#define SMEM_VB_S0_STRIDE 98304
#define SMEM_VB_S1_OFF 50176
#define SMEM_VB_S1_STAGE_BYTES 16384
#define SMEM_VB_S1_STRIDE 98304
#define SMEM_KB_S0_OFF 66560
#define SMEM_KB_S0_STAGE_BYTES 16384
#define SMEM_KB_S0_STRIDE 98304
#define SMEM_KB_S1_OFF 82944
#define SMEM_KB_S1_STAGE_BYTES 16384
#define SMEM_KB_S1_STRIDE 98304
#define SMEM_AKK_KMAJ_OFF 1024
#define SMEM_AKK_KMAJ_STAGE_BYTES 32768
#define SMEM_AKK_KMAJ_STRIDE 98304
#define SMEM_VB_MN_OFF 33792
#define SMEM_VB_MN_STAGE_BYTES 32768
#define SMEM_VB_MN_STRIDE 98304
#define SMEM_KB_MN_OFF 66560
#define SMEM_KB_MN_STAGE_BYTES 32768
#define SMEM_KB_MN_STRIDE 98304
#define SMEM_STG_OFF 197632
#define SMEM_STG_STAGE_BYTES 16384
#define SMEM_STG_STRIDE 16384
#define SMEM_TOTAL 230400
#define THREADS 192
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




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
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


extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_chunk_train_8668de69eaeb4941130d(const __grid_constant__ CUtensorMap akk_tma, const __grid_constant__ CUtensorMap vb_tma, const __grid_constant__ CUtensorMap kb_tma, const __grid_constant__ CUtensorMap u_tma, const __grid_constant__ CUtensorMap w_tma, int num_heads, int num_pairs)
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
    #define acc_full_addr (mbar_base + 32)
    #define acc_empty_addr (mbar_base + 48)
    #define zero_done_addr (mbar_base + 64)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* akk_blk0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_AKK_BLK0_OFF);
    const int akk_blk0_addr = smem + SMEM_AKK_BLK0_OFF;
    __nv_bfloat16* akk_blk3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_AKK_BLK3_OFF);
    const int akk_blk3_addr = smem + SMEM_AKK_BLK3_OFF;
    __nv_bfloat16* vb_s0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_VB_S0_OFF);
    const int vb_s0_addr = smem + SMEM_VB_S0_OFF;
    __nv_bfloat16* vb_s1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_VB_S1_OFF);
    const int vb_s1_addr = smem + SMEM_VB_S1_OFF;
    __nv_bfloat16* kb_s0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_KB_S0_OFF);
    const int kb_s0_addr = smem + SMEM_KB_S0_OFF;
    __nv_bfloat16* kb_s1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_KB_S1_OFF);
    const int kb_s1_addr = smem + SMEM_KB_S1_OFF;
    __nv_bfloat16* akk_kmaj = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_AKK_KMAJ_OFF);
    const int akk_kmaj_addr = smem + SMEM_AKK_KMAJ_OFF;
    __nv_bfloat16* vb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_VB_MN_OFF);
    const int vb_mn_addr = smem + SMEM_VB_MN_OFF;
    __nv_bfloat16* kb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_KB_MN_OFF);
    const int kb_mn_addr = smem + SMEM_KB_MN_OFF;
    __nv_bfloat16* stg = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_STG_OFF);
    const int stg_addr = smem + SMEM_STG_OFF;

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 9 barriers)
    // Mbarriers at smem_raw[0..72)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            // tiles_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // tiles_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'accs' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // acc_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            mbarrier_init(smem + 56, 128);
            // --- pipeline 'one' ---
            // zero_done: 1 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 72);
    if (warp == 0) {
        int _tmem_hold = smem + 72;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_u = taddr;
    const int tmem_acc_w = taddr + 256;

    // ---- Role: epilogue ----
    if (warp <= 3) {
        { // epilogue_main
            unsigned int epi_buf = 0;
            int num_tiles = num_pairs * num_heads;
            const int epi_warp = warp;
            const int epi_tid = epi_warp * 32 + lane;
            unsigned int zeros[4];
            zeros[0] = 0;
            zeros[1] = 0;
            zeros[2] = 0;
            zeros[3] = 0;
            #pragma unroll
            for (int s = 0; s < 2; s++) {
                #pragma unroll
                for (int blk = 1; blk < 3; blk++) {
                    int base = akk_kmaj_addr + (unsigned int)(s * 98304) + (unsigned int)(blk * 8192);
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(base + (j * 128 + epi_tid) * 16), "r"(zeros[0]), "r"(zeros[1]), "r"(zeros[2]), "r"(zeros[3]) : "memory");
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(zero_done_addr);
            const int row_base = epi_warp * 32;
            const int my_row = epi_tid;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (int tile = bid; tile < num_tiles; tile += num_bids) {
                int rtile = num_tiles - 1 - tile;
                int pair = rtile % num_pairs;
                int head = rtile / num_pairs;
                int row0 = pair * 128;
                mbarrier_wait(acc_full_addr + (epi_buf) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int tm_u = taddr + (unsigned int)(row_base << 16) + epi_buf * 128;
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], tm_u);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_0_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 ^ (my_row * 128 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 16 ^ (my_row * 128 + 16 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[4]), "r"(_tmem_load_0_bf16[5]), "r"(_tmem_load_0_bf16[6]), "r"(_tmem_load_0_bf16[7]) : "memory");
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], tm_u + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_1_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 32 ^ (my_row * 128 + 32 >> 7 & 7) << 4))), "r"(_tmem_load_1_bf16[0]), "r"(_tmem_load_1_bf16[1]), "r"(_tmem_load_1_bf16[2]), "r"(_tmem_load_1_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 48 ^ (my_row * 128 + 48 >> 7 & 7) << 4))), "r"(_tmem_load_1_bf16[4]), "r"(_tmem_load_1_bf16[5]), "r"(_tmem_load_1_bf16[6]), "r"(_tmem_load_1_bf16[7]) : "memory");
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], tm_u + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_2_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                    _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 64 ^ (my_row * 128 + 64 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[0]), "r"(_tmem_load_2_bf16[1]), "r"(_tmem_load_2_bf16[2]), "r"(_tmem_load_2_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 80 ^ (my_row * 128 + 80 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[4]), "r"(_tmem_load_2_bf16[5]), "r"(_tmem_load_2_bf16[6]), "r"(_tmem_load_2_bf16[7]) : "memory");
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], tm_u + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_3_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 96 ^ (my_row * 128 + 96 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[0]), "r"(_tmem_load_3_bf16[1]), "r"(_tmem_load_3_bf16[2]), "r"(_tmem_load_3_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 112 ^ (my_row * 128 + 112 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[4]), "r"(_tmem_load_3_bf16[5]), "r"(_tmem_load_3_bf16[6]), "r"(_tmem_load_3_bf16[7]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        tma_store_3d((&u_tma), 0, head, row0, stg_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                float _tmem_load_4[16];
                tmem_ld_x16(&_tmem_load_4[0], tm_u + 64);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_4_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                    _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 ^ (my_row * 128 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[0]), "r"(_tmem_load_4_bf16[1]), "r"(_tmem_load_4_bf16[2]), "r"(_tmem_load_4_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 16 ^ (my_row * 128 + 16 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[4]), "r"(_tmem_load_4_bf16[5]), "r"(_tmem_load_4_bf16[6]), "r"(_tmem_load_4_bf16[7]) : "memory");
                float _tmem_load_5[16];
                tmem_ld_x16(&_tmem_load_5[0], tm_u + 64 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_5_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                    _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 32 ^ (my_row * 128 + 32 >> 7 & 7) << 4))), "r"(_tmem_load_5_bf16[0]), "r"(_tmem_load_5_bf16[1]), "r"(_tmem_load_5_bf16[2]), "r"(_tmem_load_5_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 48 ^ (my_row * 128 + 48 >> 7 & 7) << 4))), "r"(_tmem_load_5_bf16[4]), "r"(_tmem_load_5_bf16[5]), "r"(_tmem_load_5_bf16[6]), "r"(_tmem_load_5_bf16[7]) : "memory");
                float _tmem_load_6[16];
                tmem_ld_x16(&_tmem_load_6[0], tm_u + 64 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_6_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                    _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 64 ^ (my_row * 128 + 64 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[0]), "r"(_tmem_load_6_bf16[1]), "r"(_tmem_load_6_bf16[2]), "r"(_tmem_load_6_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 80 ^ (my_row * 128 + 80 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[4]), "r"(_tmem_load_6_bf16[5]), "r"(_tmem_load_6_bf16[6]), "r"(_tmem_load_6_bf16[7]) : "memory");
                float _tmem_load_7[16];
                tmem_ld_x16(&_tmem_load_7[0], tm_u + 64 + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_7_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                    _tmem_load_7_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 96 ^ (my_row * 128 + 96 >> 7 & 7) << 4))), "r"(_tmem_load_7_bf16[0]), "r"(_tmem_load_7_bf16[1]), "r"(_tmem_load_7_bf16[2]), "r"(_tmem_load_7_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 112 ^ (my_row * 128 + 112 >> 7 & 7) << 4))), "r"(_tmem_load_7_bf16[4]), "r"(_tmem_load_7_bf16[5]), "r"(_tmem_load_7_bf16[6]), "r"(_tmem_load_7_bf16[7]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        tma_store_3d((&u_tma), 64, head, row0, stg_addr + 16384);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                float _tmem_load_8[16];
                tmem_ld_x16(&_tmem_load_8[0], tm_u + 256);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_8_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                    _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 ^ (my_row * 128 >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[0]), "r"(_tmem_load_8_bf16[1]), "r"(_tmem_load_8_bf16[2]), "r"(_tmem_load_8_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 16 ^ (my_row * 128 + 16 >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[4]), "r"(_tmem_load_8_bf16[5]), "r"(_tmem_load_8_bf16[6]), "r"(_tmem_load_8_bf16[7]) : "memory");
                float _tmem_load_9[16];
                tmem_ld_x16(&_tmem_load_9[0], tm_u + 256 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_9_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 0], _tmem_load_9[_lp*2+1 + 0]));
                    _tmem_load_9_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 32 ^ (my_row * 128 + 32 >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[0]), "r"(_tmem_load_9_bf16[1]), "r"(_tmem_load_9_bf16[2]), "r"(_tmem_load_9_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 48 ^ (my_row * 128 + 48 >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[4]), "r"(_tmem_load_9_bf16[5]), "r"(_tmem_load_9_bf16[6]), "r"(_tmem_load_9_bf16[7]) : "memory");
                float _tmem_load_10[16];
                tmem_ld_x16(&_tmem_load_10[0], tm_u + 256 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_10_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 0], _tmem_load_10[_lp*2+1 + 0]));
                    _tmem_load_10_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 64 ^ (my_row * 128 + 64 >> 7 & 7) << 4))), "r"(_tmem_load_10_bf16[0]), "r"(_tmem_load_10_bf16[1]), "r"(_tmem_load_10_bf16[2]), "r"(_tmem_load_10_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 80 ^ (my_row * 128 + 80 >> 7 & 7) << 4))), "r"(_tmem_load_10_bf16[4]), "r"(_tmem_load_10_bf16[5]), "r"(_tmem_load_10_bf16[6]), "r"(_tmem_load_10_bf16[7]) : "memory");
                float _tmem_load_11[16];
                tmem_ld_x16(&_tmem_load_11[0], tm_u + 256 + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_11_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                    _tmem_load_11_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 96 ^ (my_row * 128 + 96 >> 7 & 7) << 4))), "r"(_tmem_load_11_bf16[0]), "r"(_tmem_load_11_bf16[1]), "r"(_tmem_load_11_bf16[2]), "r"(_tmem_load_11_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + (unsigned int)(my_row * 128 + 112 ^ (my_row * 128 + 112 >> 7 & 7) << 4))), "r"(_tmem_load_11_bf16[4]), "r"(_tmem_load_11_bf16[5]), "r"(_tmem_load_11_bf16[6]), "r"(_tmem_load_11_bf16[7]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        tma_store_3d((&w_tma), 0, head, row0, stg_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                float _tmem_load_12[16];
                tmem_ld_x16(&_tmem_load_12[0], tm_u + 320);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_12_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 0], _tmem_load_12[_lp*2+1 + 0]));
                    _tmem_load_12_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 ^ (my_row * 128 >> 7 & 7) << 4))), "r"(_tmem_load_12_bf16[0]), "r"(_tmem_load_12_bf16[1]), "r"(_tmem_load_12_bf16[2]), "r"(_tmem_load_12_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 16 ^ (my_row * 128 + 16 >> 7 & 7) << 4))), "r"(_tmem_load_12_bf16[4]), "r"(_tmem_load_12_bf16[5]), "r"(_tmem_load_12_bf16[6]), "r"(_tmem_load_12_bf16[7]) : "memory");
                float _tmem_load_13[16];
                tmem_ld_x16(&_tmem_load_13[0], tm_u + 320 + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_13_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 0], _tmem_load_13[_lp*2+1 + 0]));
                    _tmem_load_13_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 32 ^ (my_row * 128 + 32 >> 7 & 7) << 4))), "r"(_tmem_load_13_bf16[0]), "r"(_tmem_load_13_bf16[1]), "r"(_tmem_load_13_bf16[2]), "r"(_tmem_load_13_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 48 ^ (my_row * 128 + 48 >> 7 & 7) << 4))), "r"(_tmem_load_13_bf16[4]), "r"(_tmem_load_13_bf16[5]), "r"(_tmem_load_13_bf16[6]), "r"(_tmem_load_13_bf16[7]) : "memory");
                float _tmem_load_14[16];
                tmem_ld_x16(&_tmem_load_14[0], tm_u + 320 + 32);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_14_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 0], _tmem_load_14[_lp*2+1 + 0]));
                    _tmem_load_14_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 64 ^ (my_row * 128 + 64 >> 7 & 7) << 4))), "r"(_tmem_load_14_bf16[0]), "r"(_tmem_load_14_bf16[1]), "r"(_tmem_load_14_bf16[2]), "r"(_tmem_load_14_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 80 ^ (my_row * 128 + 80 >> 7 & 7) << 4))), "r"(_tmem_load_14_bf16[4]), "r"(_tmem_load_14_bf16[5]), "r"(_tmem_load_14_bf16[6]), "r"(_tmem_load_14_bf16[7]) : "memory");
                float _tmem_load_15[16];
                tmem_ld_x16(&_tmem_load_15[0], tm_u + 320 + 48);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t _tmem_load_15_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_15[_lp*2 + 0], _tmem_load_15[_lp*2+1 + 0]));
                    _tmem_load_15_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 96 ^ (my_row * 128 + 96 >> 7 & 7) << 4))), "r"(_tmem_load_15_bf16[0]), "r"(_tmem_load_15_bf16[1]), "r"(_tmem_load_15_bf16[2]), "r"(_tmem_load_15_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((stg_addr + 16384 + (unsigned int)(my_row * 128 + 112 ^ (my_row * 128 + 112 >> 7 & 7) << 4))), "r"(_tmem_load_15_bf16[4]), "r"(_tmem_load_15_bf16[5]), "r"(_tmem_load_15_bf16[6]), "r"(_tmem_load_15_bf16[7]) : "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(acc_empty_addr + (epi_buf) * 8);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        tma_store_3d((&w_tma), 64, head, row0, stg_addr + 16384);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                epi_buf += 1;
                if (epi_buf == 2) { epi_buf = 0; _phase_acc_full ^= 1; }
            }
            asm volatile("cp.async.bulk.wait_group.read 0;");
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            unsigned int ld_stage = 0;
            int num_tiles_1 = num_pairs * num_heads;
            unsigned int _phase_tiles_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int tile_1 = bid; tile_1 < num_tiles_1; tile_1 += num_bids) {
                    int rtile_1 = num_tiles_1 - 1 - tile_1;
                    int pair_1 = rtile_1 % num_pairs;
                    int head_1 = rtile_1 / num_pairs;
                    int row0_1 = pair_1 * 128;
                    mbarrier_wait(tiles_empty_addr + (ld_stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (ld_stage) * 8, 81920);
                    tma_3d_gmem2smem(vb_s0_addr + ld_stage * 98304, (&vb_tma), 0, head_1, row0_1, tiles_full_addr + (ld_stage) * 8);
                    tma_3d_gmem2smem(kb_s0_addr + ld_stage * 98304, (&kb_tma), 0, head_1, row0_1, tiles_full_addr + (ld_stage) * 8);
                    tma_3d_gmem2smem(vb_s1_addr + ld_stage * 98304, (&vb_tma), 64, head_1, row0_1, tiles_full_addr + (ld_stage) * 8);
                    tma_3d_gmem2smem(kb_s1_addr + ld_stage * 98304, (&kb_tma), 64, head_1, row0_1, tiles_full_addr + (ld_stage) * 8);
                    tma_3d_gmem2smem(akk_blk0_addr + ld_stage * 98304, (&akk_tma), 0, head_1, row0_1, tiles_full_addr + (ld_stage) * 8);
                    tma_3d_gmem2smem(akk_blk3_addr + ld_stage * 98304, (&akk_tma), 0, head_1, row0_1 + 64, tiles_full_addr + (ld_stage) * 8);
                    ld_stage += 1;
                    if (ld_stage == 2) { ld_stage = 0; _phase_tiles_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int mma_buf = 0;
            int num_tiles_2 = num_pairs * num_heads;
            unsigned int _phase_zero_done_0 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_acc_empty = 1;
            if (elect_sync()) {
                mbarrier_wait(zero_done_addr, _phase_zero_done_0);
                _phase_zero_done_0 ^= 1;
                #pragma unroll 1
                for (int tile_2 = bid; tile_2 < num_tiles_2; tile_2 += num_bids) {
                    mbarrier_wait(tiles_full_addr + (mma_stage) * 8, _phase_tiles_full);
                    mbarrier_wait(acc_empty_addr + (mma_buf) * 8, _phase_acc_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = (((akk_kmaj_addr) >> 4) & 0x3FFF) + (mma_stage) * 6144;
                    int _mma_b_lo_0 = ((((vb_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_stage) * 6144;
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_acc_u + (mma_buf * 128))), "r"(0));
                    int _mma_a_lo_1 = (((akk_kmaj_addr) >> 4) & 0x3FFF) + (mma_stage) * 6144;
                    int _mma_b_lo_1 = ((((kb_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_stage) * 6144;
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_acc_w + (mma_buf * 128))), "r"(0));
                    tcgen05_commit(tiles_empty_addr + (mma_stage) * 8);
                    tcgen05_commit(acc_full_addr + (mma_buf) * 8);
                    mma_stage += 1;
                    if (mma_stage == 2) { mma_stage = 0; _phase_tiles_full ^= 1; }
                    mma_buf += 1;
                    if (mma_buf == 2) { mma_buf = 0; _phase_acc_empty ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
