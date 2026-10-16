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
#define TMEM_NCOLS 96
#define TMEM_ACC_DH_OFFSET 0
#define TMEM_ACC_DV_OFFSET 32
#define TMEM_ACC_P_OFFSET 64
#define NUM_RING_STAGES 2
#define NUM_ONE_STAGES 1
#define SMEM_KG_PANEL_OFF 1024
#define SMEM_KG_PANEL_STAGE_BYTES 8192
#define SMEM_KG_PANEL_STRIDE 8192
#define SMEM_QG_PANEL_OFF 33792
#define SMEM_QG_PANEL_STAGE_BYTES 8192
#define SMEM_QG_PANEL_STRIDE 8192
#define SMEM_W_PANEL_OFF 66560
#define SMEM_W_PANEL_STAGE_BYTES 8192
#define SMEM_W_PANEL_STRIDE 8192
#define SMEM_DO_PANEL_OFF 99328
#define SMEM_DO_PANEL_STAGE_BYTES 4096
#define SMEM_DO_PANEL_STRIDE 4096
#define SMEM_DV1_PANEL_OFF 107520
#define SMEM_DV1_PANEL_STAGE_BYTES 4096
#define SMEM_DV1_PANEL_STRIDE 4096
#define SMEM_KG_OP_OFF 1024
#define SMEM_KG_OP_STAGE_BYTES 16384
#define SMEM_KG_OP_STRIDE 16384
#define SMEM_QG_MN_OFF 33792
#define SMEM_QG_MN_STAGE_BYTES 16384
#define SMEM_QG_MN_STRIDE 16384
#define SMEM_W_MN_OFF 66560
#define SMEM_W_MN_STAGE_BYTES 16384
#define SMEM_W_MN_STRIDE 16384
#define SMEM_DO_MN_OFF 99328
#define SMEM_DO_MN_STAGE_BYTES 4096
#define SMEM_DO_MN_STRIDE 4096
#define SMEM_DHS_OFF 115712
#define SMEM_DHS_STAGE_BYTES 8192
#define SMEM_DHS_STRIDE 8192
#define SMEM_DHS_MN_OFF 115712
#define SMEM_DHS_MN_STAGE_BYTES 8192
#define SMEM_DHS_MN_STRIDE 8192
#define SMEM_BDV_OFF 123904
#define SMEM_BDV_STAGE_BYTES 4096
#define SMEM_BDV_STRIDE 4096
#define SMEM_BDV_MN_OFF 123904
#define SMEM_BDV_MN_STAGE_BYTES 4096
#define SMEM_BDV_MN_STRIDE 4096
#define SMEM_TOTAL 128000
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


__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
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
kernel_cake_kda_chunk_train_ab8fa57cfd3ea6694431(const __grid_constant__ CUtensorMap kg_tma, const __grid_constant__ CUtensorMap qg_tma, const __grid_constant__ CUtensorMap w_tma, const __grid_constant__ CUtensorMap do_tma, const __grid_constant__ CUtensorMap dv1, float* __restrict__ gk, __nv_bfloat16* __restrict__ dh_out, __nv_bfloat16* __restrict__ dv2, int num_heads, int* __restrict__ seq_chunk_start, float scale, float* __restrict__ dht, int use_dht, float* __restrict__ dh0_out, int store_dh0, int* __restrict__ seq_order)
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
    #define dhs_ready_addr (mbar_base + 32)
    #define dv_ready_addr (mbar_base + 40)
    #define bdv_ready_addr (mbar_base + 48)
    #define dh_ready_addr (mbar_base + 56)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* kg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_KG_PANEL_OFF);
    const int kg_panel_addr = smem + SMEM_KG_PANEL_OFF;
    __nv_bfloat16* qg_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_QG_PANEL_OFF);
    const int qg_panel_addr = smem + SMEM_QG_PANEL_OFF;
    __nv_bfloat16* w_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_W_PANEL_OFF);
    const int w_panel_addr = smem + SMEM_W_PANEL_OFF;
    __nv_bfloat16* do_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DO_PANEL_OFF);
    const int do_panel_addr = smem + SMEM_DO_PANEL_OFF;
    __nv_bfloat16* dv1_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DV1_PANEL_OFF);
    const int dv1_panel_addr = smem + SMEM_DV1_PANEL_OFF;
    __nv_bfloat16* kg_op = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_KG_OP_OFF);
    const int kg_op_addr = smem + SMEM_KG_OP_OFF;
    __nv_bfloat16* qg_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_QG_MN_OFF);
    const int qg_mn_addr = smem + SMEM_QG_MN_OFF;
    __nv_bfloat16* w_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_W_MN_OFF);
    const int w_mn_addr = smem + SMEM_W_MN_OFF;
    __nv_bfloat16* do_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DO_MN_OFF);
    const int do_mn_addr = smem + SMEM_DO_MN_OFF;
    __nv_bfloat16* dhs = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DHS_OFF);
    const int dhs_addr = smem + SMEM_DHS_OFF;
    __nv_bfloat16* dhs_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_DHS_MN_OFF);
    const int dhs_mn_addr = smem + SMEM_DHS_MN_OFF;
    __nv_bfloat16* bdv = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_BDV_OFF);
    const int bdv_addr = smem + SMEM_BDV_OFF;
    __nv_bfloat16* bdv_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_BDV_MN_OFF);
    const int bdv_mn_addr = smem + SMEM_BDV_MN_OFF;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

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
            // --- pipeline 'one' ---
            // dhs_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            // dv_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // bdv_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // dh_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 64);
    if (warp == 0) {
        int _tmem_hold = smem + 64;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_acc_dh = taddr;
    const int tmem_acc_dv = taddr + 32;
    const int tmem_acc_p = taddr + 64;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int half = blockIdx.x % 4;
            int head = blockIdx.x / 4 % num_heads;
            int rank = blockIdx.x / (4 * num_heads);
            int seq = seq_order[rank];
            int c0 = seq_chunk_start[seq];
            int nc = seq_chunk_start[seq + 1] - c0;
            int row0 = c0 * 64;
            int vbase = half * 32;
            int warp_in_wg = warp % 4;
            int k_row = warp_in_wg * 32 + lane;
            int m128_row_base = warp_in_wg * 32 << 16;
            int lane_half = lane >> 4;
            int t_row = warp_in_wg * 16 + (lane & 15);
            int m64_row_base = warp_in_wg * 32 << 16;
            unsigned int cstage = 0;
            if (use_dht != 0) {
                int in_slot = seq_order[blockIdx.x / (4 * num_heads)] * num_heads + blockIdx.x / 4 % num_heads;
                int in_base = (in_slot * 128 + k_row) * 128 + vbase;
                #pragma unroll
                for (int blk = 0; blk < 2; blk++) {
                    float init16[16];
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(dht + (in_base + blk * 16 + q * 4) + 0);
                            init16[q * 4 + 0] = _v4.x;
                            init16[q * 4 + 1] = _v4.y;
                            init16[q * 4 + 2] = _v4.z;
                            init16[q * 4 + 3] = _v4.w;
                        }
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + (unsigned int)(blk * 16), init16);
                }
            } else {
                #pragma unroll
                for (int blk_1 = 0; blk_1 < 2; blk_1++) {
                    float zero_dh[16];
                    zero_dh[0] = 0.0f;
                    zero_dh[1] = 0.0f;
                    zero_dh[2] = 0.0f;
                    zero_dh[3] = 0.0f;
                    zero_dh[4] = 0.0f;
                    zero_dh[5] = 0.0f;
                    zero_dh[6] = 0.0f;
                    zero_dh[7] = 0.0f;
                    zero_dh[8] = 0.0f;
                    zero_dh[9] = 0.0f;
                    zero_dh[10] = 0.0f;
                    zero_dh[11] = 0.0f;
                    zero_dh[12] = 0.0f;
                    zero_dh[13] = 0.0f;
                    zero_dh[14] = 0.0f;
                    zero_dh[15] = 0.0f;
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + (unsigned int)(blk_1 * 16), zero_dh);
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            int zero_i = 0;
            float gkn = 0.0f;
            int _max_0 = ((nc - 1) > (zero_i) ? (nc - 1) : (zero_i));
            int c_first = _max_0;
            int gi_first = ((row0 + c_first * 64 + 64 - 1) * num_heads + head) * 128 + k_row;
            gkn = gk[gi_first];
            unsigned int _phase_dv_ready_0 = 0;
            unsigned int _phase_tiles_full = 0;
            unsigned int _phase_dh_ready_0 = 0;
            #pragma unroll 1
            for (int rc = 0; rc < nc; rc++) {
                int c = nc - 1 - rc;
                float gkv = gkn;
                int _max_1 = ((c - 1) > (zero_i) ? (c - 1) : (zero_i));
                int c_next = _max_1;
                int gi_next = ((row0 + c_next * 64 + 64 - 1) * num_heads + head) * 128 + k_row;
                gkn = gk[gi_next];
                int dh_base = (((c0 + c) * num_heads + head) * 128 + k_row) * 128 + vbase;
                float dha[16];
                float dhb[16];
                float dhc[16];
                float dhd[16];
                tmem_ld_x16(&dha[0], taddr + (unsigned int)m128_row_base);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                tmem_ld_x16(&dhb[0], taddr + (unsigned int)m128_row_base + 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                uint32_t dha_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dha[_lp*2 + 0], dha[_lp*2+1 + 0]));
                    dha_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_1 = 0; q_1 < 2; q_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)(q_1 * 8 / 32 * 8192 + k_row * 64 + q_1 * 8 % 32 * 2 ^ (q_1 * 8 / 32 * 8192 + k_row * 64 + q_1 * 8 % 32 * 2 >> 7 & 3) << 4))), "r"(dha_bf16[q_1 * 4]), "r"(dha_bf16[q_1 * 4 + 1]), "r"(dha_bf16[q_1 * 4 + 2]), "r"(dha_bf16[q_1 * 4 + 3]) : "memory");
                }
                uint32_t dhb_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dhb[_lp*2 + 0], dhb[_lp*2+1 + 0]));
                    dhb_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_2 = 0; q_2 < 2; q_2++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((dhs_addr + (unsigned int)((16 + q_2 * 8) / 32 * 8192 + k_row * 64 + (16 + q_2 * 8) % 32 * 2 ^ ((16 + q_2 * 8) / 32 * 8192 + k_row * 64 + (16 + q_2 * 8) % 32 * 2 >> 7 & 3) << 4))), "r"(dhb_bf16[q_2 * 4]), "r"(dhb_bf16[q_2 * 4 + 1]), "r"(dhb_bf16[q_2 * 4 + 2]), "r"(dhb_bf16[q_2 * 4 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(dhs_ready_addr);
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(dha[0 + 0], dha[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(dha[0 + 2], dha[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(dha[0 + 4], dha[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(dha[0 + 6], dha[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(dha[0 + 8], dha[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(dha[0 + 10], dha[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(dha[0 + 12], dha[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(dha[0 + 14], dha[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(dh_out + dh_base))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(dhb[0 + 0], dhb[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(dhb[0 + 2], dhb[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(dhb[0 + 4], dhb[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(dhb[0 + 6], dhb[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(dhb[0 + 8], dhb[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(dhb[0 + 10], dhb[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(dhb[0 + 12], dhb[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(dhb[0 + 14], dhb[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(dh_out + (dh_base + 16)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                mbarrier_wait(dv_ready_addr, _phase_dv_ready_0);
                _phase_dv_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(tiles_full_addr + (cstage) * 8, _phase_tiles_full);
                int trow = row0 + c * 64 + t_row;
                int t_base = (trow * num_heads + head) * 128 + vbase;
                float bdva[16];
                float bdvb[16];
                float _tmem_load_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                    : "r"(taddr + 32 + (unsigned int)m64_row_base));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int col = lane_half * 16;
                unsigned int d1raw[8];
                #pragma unroll
                for (int q_3 = 0; q_3 < 2; q_3++) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&d1raw[q_3 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&d1raw[(q_3 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&d1raw[(q_3 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&d1raw[(q_3 * 4) + 3]))
                        : "r"((dv1_panel_addr + cstage * 4096 + (unsigned int)((col + q_3 * 8) / 32 * 4096 + t_row * 64 + (col + q_3 * 8) % 32 * 2 ^ ((col + q_3 * 8) / 32 * 4096 + t_row * 64 + (col + q_3 * 8) % 32 * 2 >> 7 & 3) << 4))));
                }
                float d1raw_f32[16];
                #pragma unroll
                for (int _pair = 0; _pair < 8; _pair++) {
                    (&d1raw_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(d1raw[_pair]) << 16);
                    (&d1raw_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(d1raw[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    bdva[j] = _tmem_load_0[j] + d1raw_f32[j];
                }
                uint32_t bdva_bf16[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bdva[_lp*2 + 0], bdva[_lp*2+1 + 0]));
                    bdva_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int q_4 = 0; q_4 < 2; q_4++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((bdv_addr + (unsigned int)((col + q_4 * 8) / 32 * 4096 + t_row * 64 + (col + q_4 * 8) % 32 * 2 ^ ((col + q_4 * 8) / 32 * 4096 + t_row * 64 + (col + q_4 * 8) % 32 * 2 >> 7 & 3) << 4))), "r"(bdva_bf16[q_4 * 4]), "r"(bdva_bf16[q_4 * 4 + 1]), "r"(bdva_bf16[q_4 * 4 + 2]), "r"(bdva_bf16[q_4 * 4 + 3]) : "memory");
                }
                float _exp2_0 = approx_exp2(gkv);
                float decay = _exp2_0;
                #pragma unroll
                for (int blk_2 = 0; blk_2 < 2; blk_2++) {
                    float _tmem_load_1[16];
                    tmem_ld_x16(&_tmem_load_1[0], taddr + (unsigned int)m128_row_base + (unsigned int)(blk_2 * 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float sc[16];
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 16; j_1++) {
                        sc[j_1] = _tmem_load_1[j_1] * decay;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + (unsigned int)(blk_2 * 16), sc);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(bdv_ready_addr);
                {
                    {
                        __nv_bfloat162 _pk0 = __floats2bfloat162_rn(bdva[0 + 0], bdva[0 + 1]);
                        unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                        __nv_bfloat162 _pk1 = __floats2bfloat162_rn(bdva[0 + 2], bdva[0 + 3]);
                        unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                        __nv_bfloat162 _pk2 = __floats2bfloat162_rn(bdva[0 + 4], bdva[0 + 5]);
                        unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                        __nv_bfloat162 _pk3 = __floats2bfloat162_rn(bdva[0 + 6], bdva[0 + 7]);
                        unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                        __nv_bfloat162 _pk4 = __floats2bfloat162_rn(bdva[0 + 8], bdva[0 + 9]);
                        unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                        __nv_bfloat162 _pk5 = __floats2bfloat162_rn(bdva[0 + 10], bdva[0 + 11]);
                        unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                        __nv_bfloat162 _pk6 = __floats2bfloat162_rn(bdva[0 + 12], bdva[0 + 13]);
                        unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                        __nv_bfloat162 _pk7 = __floats2bfloat162_rn(bdva[0 + 14], bdva[0 + 15]);
                        unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(&((__nv_bfloat16*)(dv2 + (t_base + lane_half * 16)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                    }
                }
                mbarrier_wait(dh_ready_addr, _phase_dh_ready_0);
                _phase_dh_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int blk_3 = 0; blk_3 < 2; blk_3++) {
                    float _tmem_load_2[16];
                    tmem_ld_x16(&_tmem_load_2[0], taddr + (unsigned int)m128_row_base + (unsigned int)(blk_3 * 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float _tmem_load_3[16];
                    tmem_ld_x16(&_tmem_load_3[0], taddr + 64 + (unsigned int)m128_row_base + (unsigned int)(blk_3 * 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float upd[16];
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 16; j_2++) {
                        float _fma_0 = __fmaf_rn(_tmem_load_3[j_2], scale, _tmem_load_2[j_2]);
                        upd[j_2] = _fma_0;
                    }
                    tmem_st_x16_f32(taddr + (unsigned int)m128_row_base + (unsigned int)(blk_3 * 16), upd);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                cstage += 1;
                if (cstage == 2) { cstage = 0; _phase_tiles_full ^= 1; }
            }
            if (store_dh0 != 0) {
                int out_slot = seq_order[blockIdx.x / (4 * num_heads)] * num_heads + blockIdx.x / 4 % num_heads;
                int out_base = (out_slot * 128 + k_row) * 128 + vbase;
                #pragma unroll
                for (int blk_4 = 0; blk_4 < 2; blk_4++) {
                    float _tmem_load_4[16];
                    tmem_ld_x16(&_tmem_load_4[0], taddr + (unsigned int)m128_row_base + (unsigned int)(blk_4 * 16));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int q_5 = 0; q_5 < 4; q_5++) {
                        {
                            float4 _v4 = make_float4(_tmem_load_4[q_5 * 4 + 0], _tmem_load_4[q_5 * 4 + 1], _tmem_load_4[q_5 * 4 + 2], _tmem_load_4[q_5 * 4 + 3]);
                            *reinterpret_cast<float4*>((dh0_out + (out_base + blk_4 * 16 + q_5 * 4)) + 0) = _v4;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(128));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int half_1 = blockIdx.x % 4;
            int head_1 = blockIdx.x / 4 % num_heads;
            int rank_1 = blockIdx.x / (4 * num_heads);
            int seq_1 = seq_order[rank_1];
            int c0_1 = seq_chunk_start[seq_1];
            int nc_1 = seq_chunk_start[seq_1 + 1] - c0_1;
            int row0_1 = c0_1 * 64;
            int n_seq = gridDim.x / (4 * num_heads);
            int nc_total = seq_chunk_start[n_seq];
            if (nc_total % 2 != 0) {
                if (rank_1 == 0) {
                    float zero16[16];
                    zero16[0] = 0.0f;
                    zero16[1] = 0.0f;
                    zero16[2] = 0.0f;
                    zero16[3] = 0.0f;
                    zero16[4] = 0.0f;
                    zero16[5] = 0.0f;
                    zero16[6] = 0.0f;
                    zero16[7] = 0.0f;
                    zero16[8] = 0.0f;
                    zero16[9] = 0.0f;
                    zero16[10] = 0.0f;
                    zero16[11] = 0.0f;
                    zero16[12] = 0.0f;
                    zero16[13] = 0.0f;
                    zero16[14] = 0.0f;
                    zero16[15] = 0.0f;
                    int prow = lane / 2;
                    int pcol = lane % 2 * 16;
                    #pragma unroll
                    for (int p = 0; p < 4; p++) {
                        int pidx = ((nc_total * 64 + p * 16 + prow) * num_heads + head_1) * 128 + half_1 * 32 + pcol;
                        {
                            {
                                __nv_bfloat162 _pk0 = __floats2bfloat162_rn(zero16[0 + 0], zero16[0 + 1]);
                                unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                __nv_bfloat162 _pk1 = __floats2bfloat162_rn(zero16[0 + 2], zero16[0 + 3]);
                                unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                __nv_bfloat162 _pk2 = __floats2bfloat162_rn(zero16[0 + 4], zero16[0 + 5]);
                                unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                __nv_bfloat162 _pk3 = __floats2bfloat162_rn(zero16[0 + 6], zero16[0 + 7]);
                                unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                __nv_bfloat162 _pk4 = __floats2bfloat162_rn(zero16[0 + 8], zero16[0 + 9]);
                                unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                __nv_bfloat162 _pk5 = __floats2bfloat162_rn(zero16[0 + 10], zero16[0 + 11]);
                                unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                __nv_bfloat162 _pk6 = __floats2bfloat162_rn(zero16[0 + 12], zero16[0 + 13]);
                                unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                __nv_bfloat162 _pk7 = __floats2bfloat162_rn(zero16[0 + 14], zero16[0 + 15]);
                                unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(&((__nv_bfloat16*)(dv2 + pidx))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                            }
                        }
                    }
                }
            }
            unsigned int stage = 0;
            unsigned int _phase_tiles_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_1 = 0; rc_1 < nc_1; rc_1++) {
                    int c_1 = nc_1 - 1 - rc_1;
                    mbarrier_wait(tiles_empty_addr + (stage) * 8, _phase_tiles_empty);
                    mbarrier_arrive_expect_tx(tiles_full_addr + (stage) * 8, 57344);
                    int trow_1 = row0_1 + c_1 * 64;
                    #pragma unroll
                    for (int seg = 0; seg < 2; seg++) {
                        tma_3d_gmem2smem(kg_panel_addr + (stage * 2 + (unsigned int)seg) * 8192, (&kg_tma), seg * 64, head_1, trow_1, tiles_full_addr + (stage) * 8);
                        tma_3d_gmem2smem(qg_panel_addr + (stage * 2 + (unsigned int)seg) * 8192, (&qg_tma), seg * 64, head_1, trow_1, tiles_full_addr + (stage) * 8);
                        tma_3d_gmem2smem(w_panel_addr + (stage * 2 + (unsigned int)seg) * 8192, (&w_tma), seg * 64, head_1, trow_1, tiles_full_addr + (stage) * 8);
                    }
                    tma_3d_gmem2smem(do_panel_addr + stage * 4096, (&do_tma), half_1 * 32, head_1, trow_1, tiles_full_addr + (stage) * 8);
                    tma_3d_gmem2smem(dv1_panel_addr + stage * 4096, (&dv1), half_1 * 32, head_1, trow_1, tiles_full_addr + (stage) * 8);
                    stage += 1;
                    if (stage == 2) { stage = 0; _phase_tiles_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            int half_2 = blockIdx.x % 4;
            int head_2 = blockIdx.x / 4 % num_heads;
            int rank_2 = blockIdx.x / (4 * num_heads);
            int seq_2 = seq_order[rank_2];
            int c0_2 = seq_chunk_start[seq_2];
            int nc_2 = seq_chunk_start[seq_2 + 1] - c0_2;
            int row0_2 = c0_2 * 64;
            unsigned int stage_1 = 0;
            unsigned int _phase_tiles_full_1 = 0;
            unsigned int _phase_dhs_ready_0 = 0;
            unsigned int _phase_bdv_ready_0 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rc_2 = 0; rc_2 < nc_2; rc_2++) {
                    mbarrier_wait(tiles_full_addr + (stage_1) * 8, _phase_tiles_full_1);
                    mbarrier_wait(dhs_ready_addr, _phase_dhs_ready_0);
                    _phase_dhs_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = (((kg_op_addr) >> 4) & 0x3FFF) + (stage_1) * 1024;
                    int _mma_b_lo_0 = (((dhs_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
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
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 67699856;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc_dv), "r"(0));
                    tcgen05_commit(dv_ready_addr);
                    mbarrier_wait(bdv_ready_addr, _phase_bdv_ready_0);
                    _phase_bdv_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_1 = ((((qg_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int _mma_b_lo_1 = ((((do_mn_addr) >> 4) & 0x3FFF) | 0x1000000) + (stage_1) * 256;
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
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_acc_p), "r"(0));
                    int _mma_a_lo_2 = ((((w_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (stage_1) * 1024;
                    int _mma_b_lo_2 = (((bdv_mn_addr) >> 4) & 0x3FFF) | 0x1000000;
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
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, %4;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_acc_dh), "r"(1), "r"(((uint32_t)(134841488) | ((uint32_t)(1) << 13) | ((uint32_t)(0) << 14))));
                    tcgen05_commit(dh_ready_addr);
                    tcgen05_commit(tiles_empty_addr + (stage_1) * 8);
                    stage_1 += 1;
                    if (stage_1 == 2) { stage_1 = 0; _phase_tiles_full_1 ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
