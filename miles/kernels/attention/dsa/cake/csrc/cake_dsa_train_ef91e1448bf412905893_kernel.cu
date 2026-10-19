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
#define TMEM_NCOLS 128
#define TMEM_TMEM_OFFSET 0
#define NUM_CLC_PIPELINE_STAGES 1
#define SMEM_KA_OFF 1024
#define SMEM_KA_STAGE_BYTES 16384
#define SMEM_KA_STRIDE 16384
#define SMEM_VT_OFF 1024
#define SMEM_VT_STAGE_BYTES 32768
#define SMEM_VT_STRIDE 32768
#define SMEM_QT_OFF 197632
#define SMEM_QT_STAGE_BYTES 16384
#define SMEM_QT_STRIDE 16384
#define SMEM_PT_OFF 214016
#define SMEM_PT_STAGE_BYTES 4096
#define SMEM_PT_STRIDE 4096
#define SMEM_VALIDITY_OFF 218112
#define SMEM_VALIDITY_STAGE_BYTES 48
#define SMEM_VALIDITY_STRIDE 48
#define SMEM_VALIDITY32_OFF 218112
#define SMEM_VALIDITY32_STAGE_BYTES 48
#define SMEM_VALIDITY32_STRIDE 48
#define SMEM_MAXIMA_OFF 218176
#define SMEM_MAXIMA_STAGE_BYTES 512
#define SMEM_MAXIMA_STRIDE 512
#define SMEM_SUMS_OFF 218688
#define SMEM_SUMS_STAGE_BYTES 256
#define SMEM_SUMS_STRIDE 256
#define SMEM_CLC_PAYLOAD_OFF 218944
#define SMEM_CLC_PAYLOAD_STAGE_BYTES 16
#define SMEM_CLC_PAYLOAD_STRIDE 16
#define SMEM_LEN_CELLS_OFF 218960
#define SMEM_LEN_CELLS_STAGE_BYTES 8
#define SMEM_LEN_CELLS_STRIDE 8
#define SMEM_TOTAL 219008
#define THREADS 384
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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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
kernel_cake_dsa_train_ef91e1448bf412905893(const __grid_constant__ CUtensorMap q_latent, const __grid_constant__ CUtensorMap q_rope, const __grid_constant__ CUtensorMap kv_latent, __nv_bfloat16* __restrict__ k_rope, const __grid_constant__ CUtensorMap out, __nv_bfloat16* __restrict__ o_lo, float* __restrict__ lse, int* __restrict__ indices, int* __restrict__ topk_length, float* __restrict__ sink, int num_queries, int num_kv, int topk, int idx_stride, int indices_offset, int k_rope_stride, int k_rope_offset, int has_topk_length, int derive_length, float scale_log2, int heads)
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
    #define q_full_addr (mbar_base + 0)
    #define q_free_addr (mbar_base + 8)
    #define clc_full_addr (mbar_base + 16)
    #define clc_empty_addr (mbar_base + 24)
    #define len_ready_addr (mbar_base + 32)
    #define qk_done_addr (mbar_base + 48)
    #define p_free_addr (mbar_base + 64)
    #define so_ready_addr (mbar_base + 80)
    #define sv_done_addr (mbar_base + 88)
    #define kv_ready_addr (mbar_base + 136)
    #define valid_ready_addr (mbar_base + 232)
    #define valid_free_addr (mbar_base + 256)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* ka = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_KA_OFF);
    const int ka_addr = smem + SMEM_KA_OFF;
    __nv_bfloat16* vt = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_VT_OFF);
    const int vt_addr = smem + SMEM_VT_OFF;
    __nv_bfloat16* qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_QT_OFF);
    const int qt_addr = smem + SMEM_QT_OFF;
    __nv_bfloat16* pt = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_PT_OFF);
    const int pt_addr = smem + SMEM_PT_OFF;
    uint8_t* validity = reinterpret_cast<uint8_t*>(smem_raw + SMEM_VALIDITY_OFF);
    const int validity_addr = smem + SMEM_VALIDITY_OFF;
    unsigned int* validity32 = reinterpret_cast<unsigned int*>(smem_raw + SMEM_VALIDITY32_OFF);
    const int validity32_addr = smem + SMEM_VALIDITY32_OFF;
    float* maxima = reinterpret_cast<float*>(smem_raw + SMEM_MAXIMA_OFF);
    const int maxima_addr = smem + SMEM_MAXIMA_OFF;
    float* sums = reinterpret_cast<float*>(smem_raw + SMEM_SUMS_OFF);
    const int sums_addr = smem + SMEM_SUMS_OFF;
    unsigned int* clc_payload = reinterpret_cast<unsigned int*>(smem_raw + SMEM_CLC_PAYLOAD_OFF);
    const int clc_payload_addr = smem + SMEM_CLC_PAYLOAD_OFF;
    int* len_cells = reinterpret_cast<int*>(smem_raw + SMEM_LEN_CELLS_OFF);
    const int len_cells_addr = smem + SMEM_LEN_CELLS_OFF;
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&kv_latent))) : "memory"); }
    if (warp == 0 && lane == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }

    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 35 barriers)
    // Mbarriers at smem_raw[0..280)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'clc_pipeline' ---
            // clc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // clc_empty: 1 barriers, init_count=346
            mbarrier_init(smem + 24, 346);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_free: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // len_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // qk_done: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // p_free: 2 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            mbarrier_init(smem + 72, 128);
            // so_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 80, 128);
            // sv_done: 6 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // kv_ready: 12 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // valid_ready: 3 barriers, init_count=8
            mbarrier_init(smem + 232, 8);
            mbarrier_init(smem + 240, 8);
            mbarrier_init(smem + 248, 8);
            // valid_free: 3 barriers, init_count=128
            mbarrier_init(smem + 256, 128);
            mbarrier_init(smem + 264, 128);
            mbarrier_init(smem + 272, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 280);
    if (warp == 2) {
        int _tmem_hold = smem + 280;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;

    // ---- Role: softmax ----
    if (warp <= 3) {
        { // softmax_main
            int query = blockIdx.x;
            unsigned int outer_phase = 0;
            int tcur = 0;
            int w = warp;
            int kb = 2 * (lane % 4);
            int r8 = lane % 8;
            int m4 = lane / 8;
            int pt_atom = w / 2;
            int pt_chunk = 4 * (w % 2) + m4 ^ r8;
            unsigned int pt_addr_0 = pt_addr + (unsigned int)(pt_atom * 2048) + (unsigned int)(r8 * 128) + (unsigned int)(pt_chunk << 4);
            float s0[4];
            float s1[4];
            float p0[4];
            float p1[4];
            float s0_1[4];
            float s1_2[4];
            float p0_3[4];
            float p1_4[4];
            #pragma unroll 1
            for (int outer = 0; outer < num_queries; outer++) {
                int active = topk;
                if (derive_length != 0) {
                    int len_slot = outer & 1;
                    unsigned int len_phase = outer >> 1 & 1;
                    mbarrier_wait(len_ready_addr + (len_slot) * 8, len_phase);
                    int _max_0 = ((len_cells[len_slot]) > (0) ? (len_cells[len_slot]) : (0));
                    int _min_0 = ((_max_0) < (topk) ? (_max_0) : (topk));
                    active = _min_0;
                } else if (has_topk_length != 0) {
                    int _max_1 = ((topk_length[query]) > (0) ? (topk_length[query]) : (0));
                    int _min_1 = ((_max_1) < (topk) ? (_max_1) : (topk));
                    active = _min_1;
                }
                int active_0 = active;
                int _max_2 = (((active_0 + 128 - 1) / 128) > (1) ? ((active_0 + 128 - 1) / 128) : (1));
                int blocks = _max_2;
                float mrow[4];
                float lrow[4];
                float real[4];
                mrow[0] = -1e+30f;
                lrow[0] = 0.0f;
                real[0] = -CAKE_INF;
                mrow[1] = -1e+30f;
                lrow[1] = 0.0f;
                real[1] = -CAKE_INF;
                mrow[2] = -1e+30f;
                lrow[2] = 0.0f;
                real[2] = -CAKE_INF;
                mrow[3] = -1e+30f;
                lrow[3] = 0.0f;
                real[3] = -CAKE_INF;
                #pragma unroll 1
                for (int block = 0; block < blocks; block++) {
                    int buf = tcur & 1;
                    unsigned int bphase = tcur >> 1 & 1;
                    int vstage = tcur % 3;
                    unsigned int vphase = tcur / 3 & 1;
                    int par = tcur & 1;
                    mbarrier_wait(qk_done_addr + (buf) * 8, bphase);
                    mbarrier_wait(valid_ready_addr + (vstage) * 8, vphase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&s0[0])), "=r"(*reinterpret_cast<uint32_t*>(&s0[1])), "=r"(*reinterpret_cast<uint32_t*>(&s0[2])), "=r"(*reinterpret_cast<uint32_t*>(&s0[3]))
                        : "r"(tmem_tmem + buf * 16));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&s1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s1[1])), "=r"(*reinterpret_cast<uint32_t*>(&s1[2])), "=r"(*reinterpret_cast<uint32_t*>(&s1[3]))
                        : "r"(tmem_tmem + buf * 16 + 1048576));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&s0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&s0_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&s0_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&s0_1[3]))
                        : "r"(tmem_tmem + buf * 16 + 8));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&s1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&s1_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&s1_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&s1_2[3]))
                        : "r"(tmem_tmem + buf * 16 + 8 + 1048576));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_free_addr + (buf) * 8);
                    unsigned int vmask = validity32[vstage * 4 + w];
                    int kbit = lane / 4;
                    if ((vmask >> (unsigned int)kbit & 1) == 0) {
                        s0[0] = -CAKE_INF;
                        s0[1] = -CAKE_INF;
                        s0_1[0] = -CAKE_INF;
                        s0_1[1] = -CAKE_INF;
                    }
                    if ((vmask >> (unsigned int)(kbit + 8) & 1) == 0) {
                        s0[2] = -CAKE_INF;
                        s0[3] = -CAKE_INF;
                        s0_1[2] = -CAKE_INF;
                        s0_1[3] = -CAKE_INF;
                    }
                    if ((vmask >> (unsigned int)(kbit + 16) & 1) == 0) {
                        s1[0] = -CAKE_INF;
                        s1[1] = -CAKE_INF;
                        s1_2[0] = -CAKE_INF;
                        s1_2[1] = -CAKE_INF;
                    }
                    if ((vmask >> (unsigned int)(kbit + 24) & 1) == 0) {
                        s1[2] = -CAKE_INF;
                        s1[3] = -CAKE_INF;
                        s1_2[2] = -CAKE_INF;
                        s1_2[3] = -CAKE_INF;
                    }
                    float lmax[4];
                    float _max_3 = max_noftz(s0[0], s0[2]);
                    float _max_4 = max_noftz(s1[0], s1[2]);
                    float _max_5 = max_noftz(_max_3, _max_4);
                    lmax[0] = _max_5;
                    float _max_6 = max_noftz(s0[1], s0[3]);
                    float _max_7 = max_noftz(s1[1], s1[3]);
                    float _max_8 = max_noftz(_max_6, _max_7);
                    lmax[1] = _max_8;
                    float _max_9 = max_noftz(s0_1[0], s0_1[2]);
                    float _max_10 = max_noftz(s1_2[0], s1_2[2]);
                    float _max_11 = max_noftz(_max_9, _max_10);
                    lmax[2] = _max_11;
                    float _max_12 = max_noftz(s0_1[1], s0_1[3]);
                    float _max_13 = max_noftz(s1_2[1], s1_2[3]);
                    float _max_14 = max_noftz(_max_12, _max_13);
                    lmax[3] = _max_14;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, lmax[0], 4);
                    float _max_15 = max_noftz(lmax[0], _shfl_xor_0);
                    lmax[0] = _max_15;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, lmax[1], 4);
                    float _max_16 = max_noftz(lmax[1], _shfl_xor_1);
                    lmax[1] = _max_16;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, lmax[2], 4);
                    float _max_17 = max_noftz(lmax[2], _shfl_xor_2);
                    lmax[2] = _max_17;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, lmax[3], 4);
                    float _max_18 = max_noftz(lmax[3], _shfl_xor_3);
                    lmax[3] = _max_18;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, lmax[0], 8);
                    float _max_19 = max_noftz(lmax[0], _shfl_xor_4);
                    lmax[0] = _max_19;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, lmax[1], 8);
                    float _max_20 = max_noftz(lmax[1], _shfl_xor_5);
                    lmax[1] = _max_20;
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, lmax[2], 8);
                    float _max_21 = max_noftz(lmax[2], _shfl_xor_6);
                    lmax[2] = _max_21;
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, lmax[3], 8);
                    float _max_22 = max_noftz(lmax[3], _shfl_xor_7);
                    lmax[3] = _max_22;
                    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, lmax[0], 16);
                    float _max_23 = max_noftz(lmax[0], _shfl_xor_8);
                    lmax[0] = _max_23;
                    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, lmax[1], 16);
                    float _max_24 = max_noftz(lmax[1], _shfl_xor_9);
                    lmax[1] = _max_24;
                    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, lmax[2], 16);
                    float _max_25 = max_noftz(lmax[2], _shfl_xor_10);
                    lmax[2] = _max_25;
                    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, lmax[3], 16);
                    float _max_26 = max_noftz(lmax[3], _shfl_xor_11);
                    lmax[3] = _max_26;
                    if (lane < 4) {
                        maxima[par * 64 + w * 16 + kb] = lmax[0];
                        maxima[par * 64 + w * 16 + kb + 1] = lmax[1];
                        maxima[par * 64 + w * 16 + 8 + kb] = lmax[2];
                        maxima[par * 64 + w * 16 + 8 + kb + 1] = lmax[3];
                    }
                    mbarrier_arrive(valid_free_addr + (vstage) * 8);
                    asm volatile("barrier.sync 0, 128;" ::: "memory");
                    float tmax[4];
                    tmax[0] = -CAKE_INF;
                    tmax[1] = -CAKE_INF;
                    tmax[2] = -CAKE_INF;
                    tmax[3] = -CAKE_INF;
                    float _maxima_reg_0[2];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(maxima);
                        #pragma unroll
                        for (int _lr = 0; _lr < 2; _lr++)
                            _maxima_reg_0[_lr] = _smem_ptr[(par * 64 + kb) + _lr];
                    }
                    float _max_27 = max_noftz(tmax[0], _maxima_reg_0[0]);
                    tmax[0] = _max_27;
                    float _max_28 = max_noftz(tmax[1], _maxima_reg_0[1]);
                    tmax[1] = _max_28;
                    float _maxima_reg_1[2];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(maxima);
                        #pragma unroll
                        for (int _lr = 0; _lr < 2; _lr++)
                            _maxima_reg_1[_lr] = _smem_ptr[(par * 64 + 8 + kb) + _lr];
                    }
                    float _max_29 = max_noftz(tmax[2], _maxima_reg_1[0]);
                    tmax[2] = _max_29;
                    float _max_30 = max_noftz(tmax[3], _maxima_reg_1[1]);
                    tmax[3] = _max_30;
                    float _maxima_reg_2[2];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(maxima);
                        #pragma unroll
                        for (int _lr = 0; _lr < 2; _lr++)
                            _maxima_reg_2[_lr] = _smem_ptr[(par * 64 + 16 + kb) + _lr];
                    }
                    float _max_31 = max_noftz(tmax[0], _maxima_reg_2[0]);
                    tmax[0] = _max_31;
                    float _max_32 = max_noftz(tmax[1], _maxima_reg_2[1]);
                    tmax[1] = _max_32;
                    float _maxima_reg_3[2];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(maxima);
                        #pragma unroll
                        for (int _lr = 0; _lr < 2; _lr++)
                            _maxima_reg_3[_lr] = _smem_ptr[(par * 64 + 16 + 8 + kb) + _lr];
                    }
                    float _max_33 = max_noftz(tmax[2], _maxima_reg_3[0]);
                    tmax[2] = _max_33;
                    float _max_34 = max_noftz(tmax[3], _maxima_reg_3[1]);
                    tmax[3] = _max_34;
                    float _maxima_reg_4[2];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(maxima);
                        #pragma unroll
                        for (int _lr = 0; _lr < 2; _lr++)
                            _maxima_reg_4[_lr] = _smem_ptr[(par * 64 + 32 + kb) + _lr];
                    }
                    float _max_35 = max_noftz(tmax[0], _maxima_reg_4[0]);
                    tmax[0] = _max_35;
                    float _max_36 = max_noftz(tmax[1], _maxima_reg_4[1]);
                    tmax[1] = _max_36;
                    float _maxima_reg_5[2];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(maxima);
                        #pragma unroll
                        for (int _lr = 0; _lr < 2; _lr++)
                            _maxima_reg_5[_lr] = _smem_ptr[(par * 64 + 32 + 8 + kb) + _lr];
                    }
                    float _max_37 = max_noftz(tmax[2], _maxima_reg_5[0]);
                    tmax[2] = _max_37;
                    float _max_38 = max_noftz(tmax[3], _maxima_reg_5[1]);
                    tmax[3] = _max_38;
                    float _maxima_reg_6[2];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(maxima);
                        #pragma unroll
                        for (int _lr = 0; _lr < 2; _lr++)
                            _maxima_reg_6[_lr] = _smem_ptr[(par * 64 + 48 + kb) + _lr];
                    }
                    float _max_39 = max_noftz(tmax[0], _maxima_reg_6[0]);
                    tmax[0] = _max_39;
                    float _max_40 = max_noftz(tmax[1], _maxima_reg_6[1]);
                    tmax[1] = _max_40;
                    float _maxima_reg_7[2];
                    {
                        const float* _smem_ptr = reinterpret_cast<const float*>(maxima);
                        #pragma unroll
                        for (int _lr = 0; _lr < 2; _lr++)
                            _maxima_reg_7[_lr] = _smem_ptr[(par * 64 + 48 + 8 + kb) + _lr];
                    }
                    float _max_41 = max_noftz(tmax[2], _maxima_reg_7[0]);
                    tmax[2] = _max_41;
                    float _max_42 = max_noftz(tmax[3], _maxima_reg_7[1]);
                    tmax[3] = _max_42;
                    float fac[4];
                    float cur = tmax[0] * scale_log2;
                    float _max_43 = max_noftz(real[0], cur);
                    real[0] = _max_43;
                    float _max_44 = max_noftz(cur, mrow[0]);
                    float new_ = _max_44;
                    float _exp2_0 = approx_exp2(mrow[0] - new_);
                    fac[0] = _exp2_0;
                    mrow[0] = new_;
                    float cur_0 = tmax[1] * scale_log2;
                    float _max_45 = max_noftz(real[1], cur_0);
                    real[1] = _max_45;
                    float _max_46 = max_noftz(cur_0, mrow[1]);
                    float new_1 = _max_46;
                    float _exp2_1 = approx_exp2(mrow[1] - new_1);
                    fac[1] = _exp2_1;
                    mrow[1] = new_1;
                    float cur_2 = tmax[2] * scale_log2;
                    float _max_47 = max_noftz(real[2], cur_2);
                    real[2] = _max_47;
                    float _max_48 = max_noftz(cur_2, mrow[2]);
                    float new_3 = _max_48;
                    float _exp2_2 = approx_exp2(mrow[2] - new_3);
                    fac[2] = _exp2_2;
                    mrow[2] = new_3;
                    float cur_4 = tmax[3] * scale_log2;
                    float _max_49 = max_noftz(real[3], cur_4);
                    real[3] = _max_49;
                    float _max_50 = max_noftz(cur_4, mrow[3]);
                    float new_5 = _max_50;
                    float _exp2_3 = approx_exp2(mrow[3] - new_5);
                    fac[3] = _exp2_3;
                    mrow[3] = new_5;
                    float _fma_0 = __fmaf_rn(s0[0], scale_log2, -mrow[0]);
                    float _exp2_4 = approx_exp2(_fma_0);
                    p0[0] = _exp2_4;
                    float _fma_1 = __fmaf_rn(s0[1], scale_log2, -mrow[1]);
                    float _exp2_5 = approx_exp2(_fma_1);
                    p0[1] = _exp2_5;
                    float _fma_2 = __fmaf_rn(s0[2], scale_log2, -mrow[0]);
                    float _exp2_6 = approx_exp2(_fma_2);
                    p0[2] = _exp2_6;
                    float _fma_3 = __fmaf_rn(s0[3], scale_log2, -mrow[1]);
                    float _exp2_7 = approx_exp2(_fma_3);
                    p0[3] = _exp2_7;
                    float _fma_4 = __fmaf_rn(s1[0], scale_log2, -mrow[0]);
                    float _exp2_8 = approx_exp2(_fma_4);
                    p1[0] = _exp2_8;
                    float _fma_5 = __fmaf_rn(s1[1], scale_log2, -mrow[1]);
                    float _exp2_9 = approx_exp2(_fma_5);
                    p1[1] = _exp2_9;
                    float _fma_6 = __fmaf_rn(s1[2], scale_log2, -mrow[0]);
                    float _exp2_10 = approx_exp2(_fma_6);
                    p1[2] = _exp2_10;
                    float _fma_7 = __fmaf_rn(s1[3], scale_log2, -mrow[1]);
                    float _exp2_11 = approx_exp2(_fma_7);
                    p1[3] = _exp2_11;
                    float sum0 = p0[0] + p0[2] + (p1[0] + p1[2]);
                    float sum1 = p0[1] + p0[3] + (p1[1] + p1[3]);
                    float _fma_8 = __fmaf_rn(lrow[0], fac[0], sum0);
                    lrow[0] = _fma_8;
                    float _fma_9 = __fmaf_rn(lrow[1], fac[1], sum1);
                    lrow[1] = _fma_9;
                    float _fma_10 = __fmaf_rn(s0_1[0], scale_log2, -mrow[2]);
                    float _exp2_12 = approx_exp2(_fma_10);
                    p0_3[0] = _exp2_12;
                    float _fma_11 = __fmaf_rn(s0_1[1], scale_log2, -mrow[3]);
                    float _exp2_13 = approx_exp2(_fma_11);
                    p0_3[1] = _exp2_13;
                    float _fma_12 = __fmaf_rn(s0_1[2], scale_log2, -mrow[2]);
                    float _exp2_14 = approx_exp2(_fma_12);
                    p0_3[2] = _exp2_14;
                    float _fma_13 = __fmaf_rn(s0_1[3], scale_log2, -mrow[3]);
                    float _exp2_15 = approx_exp2(_fma_13);
                    p0_3[3] = _exp2_15;
                    float _fma_14 = __fmaf_rn(s1_2[0], scale_log2, -mrow[2]);
                    float _exp2_16 = approx_exp2(_fma_14);
                    p1_4[0] = _exp2_16;
                    float _fma_15 = __fmaf_rn(s1_2[1], scale_log2, -mrow[3]);
                    float _exp2_17 = approx_exp2(_fma_15);
                    p1_4[1] = _exp2_17;
                    float _fma_16 = __fmaf_rn(s1_2[2], scale_log2, -mrow[2]);
                    float _exp2_18 = approx_exp2(_fma_16);
                    p1_4[2] = _exp2_18;
                    float _fma_17 = __fmaf_rn(s1_2[3], scale_log2, -mrow[3]);
                    float _exp2_19 = approx_exp2(_fma_17);
                    p1_4[3] = _exp2_19;
                    float sum0_6 = p0_3[0] + p0_3[2] + (p1_4[0] + p1_4[2]);
                    float sum1_7 = p0_3[1] + p0_3[3] + (p1_4[1] + p1_4[3]);
                    float _fma_18 = __fmaf_rn(lrow[2], fac[2], sum0_6);
                    lrow[2] = _fma_18;
                    float _fma_19 = __fmaf_rn(lrow[3], fac[3], sum1_7);
                    lrow[3] = _fma_19;
                    if (block > 0) {
                        int prev_pair = (4 * (tcur - 1) + 3) % 6;
                        unsigned int prev_phase = (4 * (tcur - 1) + 3) / 6 & 1;
                        mbarrier_wait(sv_done_addr + (prev_pair) * 8, prev_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _vote_0 = __any_sync(0xFFFFFFFF, fac[0] != 1.0f || fac[1] != 1.0f || fac[2] != 1.0f || fac[3] != 1.0f);
                        int rescale = _vote_0;
                        if (rescale != 0) {
                            float ov[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov[3]))
                                : "r"(tmem_tmem + 32));
                            float ov_0[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_0[3]))
                                : "r"(tmem_tmem + 32 + 1048576));
                            float ov_1[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_1[3]))
                                : "r"(tmem_tmem + 32 + 8));
                            float ov_2[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_2[3]))
                                : "r"(tmem_tmem + 32 + 8 + 1048576));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            ov[0] = ov[0] * fac[0];
                            ov[1] = ov[1] * fac[1];
                            ov[2] = ov[2] * fac[0];
                            ov[3] = ov[3] * fac[1];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32), "r"(*reinterpret_cast<const uint32_t*>(&ov[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov[3])));
                            ov_0[0] = ov_0[0] * fac[0];
                            ov_0[1] = ov_0[1] * fac[1];
                            ov_0[2] = ov_0[2] * fac[0];
                            ov_0[3] = ov_0[3] * fac[1];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&ov_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_0[3])));
                            ov_1[0] = ov_1[0] * fac[2];
                            ov_1[1] = ov_1[1] * fac[3];
                            ov_1[2] = ov_1[2] * fac[2];
                            ov_1[3] = ov_1[3] * fac[3];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 8), "r"(*reinterpret_cast<const uint32_t*>(&ov_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_1[3])));
                            ov_2[0] = ov_2[0] * fac[2];
                            ov_2[1] = ov_2[1] * fac[3];
                            ov_2[2] = ov_2[2] * fac[2];
                            ov_2[3] = ov_2[3] * fac[3];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 8 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&ov_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_2[3])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            float ov_3[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_3[3]))
                                : "r"(tmem_tmem + 32 + 16));
                            float ov_4[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_4[3]))
                                : "r"(tmem_tmem + 32 + 16 + 1048576));
                            float ov_5[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_5[3]))
                                : "r"(tmem_tmem + 32 + 16 + 8));
                            float ov_6[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_6[3]))
                                : "r"(tmem_tmem + 32 + 16 + 8 + 1048576));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            ov_3[0] = ov_3[0] * fac[0];
                            ov_3[1] = ov_3[1] * fac[1];
                            ov_3[2] = ov_3[2] * fac[0];
                            ov_3[3] = ov_3[3] * fac[1];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 16), "r"(*reinterpret_cast<const uint32_t*>(&ov_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_3[3])));
                            ov_4[0] = ov_4[0] * fac[0];
                            ov_4[1] = ov_4[1] * fac[1];
                            ov_4[2] = ov_4[2] * fac[0];
                            ov_4[3] = ov_4[3] * fac[1];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 16 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&ov_4[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_4[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_4[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_4[3])));
                            ov_5[0] = ov_5[0] * fac[2];
                            ov_5[1] = ov_5[1] * fac[3];
                            ov_5[2] = ov_5[2] * fac[2];
                            ov_5[3] = ov_5[3] * fac[3];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 16 + 8), "r"(*reinterpret_cast<const uint32_t*>(&ov_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_5[3])));
                            ov_6[0] = ov_6[0] * fac[2];
                            ov_6[1] = ov_6[1] * fac[3];
                            ov_6[2] = ov_6[2] * fac[2];
                            ov_6[3] = ov_6[3] * fac[3];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 16 + 8 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&ov_6[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_6[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_6[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_6[3])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            float ov_7[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_7[3]))
                                : "r"(tmem_tmem + 32 + 32));
                            float ov_8[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_8[3]))
                                : "r"(tmem_tmem + 32 + 32 + 1048576));
                            float ov_9[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_9[3]))
                                : "r"(tmem_tmem + 32 + 32 + 8));
                            float ov_10[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_10[3]))
                                : "r"(tmem_tmem + 32 + 32 + 8 + 1048576));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            ov_7[0] = ov_7[0] * fac[0];
                            ov_7[1] = ov_7[1] * fac[1];
                            ov_7[2] = ov_7[2] * fac[0];
                            ov_7[3] = ov_7[3] * fac[1];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 32), "r"(*reinterpret_cast<const uint32_t*>(&ov_7[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_7[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_7[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_7[3])));
                            ov_8[0] = ov_8[0] * fac[0];
                            ov_8[1] = ov_8[1] * fac[1];
                            ov_8[2] = ov_8[2] * fac[0];
                            ov_8[3] = ov_8[3] * fac[1];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 32 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&ov_8[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_8[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_8[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_8[3])));
                            ov_9[0] = ov_9[0] * fac[2];
                            ov_9[1] = ov_9[1] * fac[3];
                            ov_9[2] = ov_9[2] * fac[2];
                            ov_9[3] = ov_9[3] * fac[3];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 32 + 8), "r"(*reinterpret_cast<const uint32_t*>(&ov_9[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_9[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_9[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_9[3])));
                            ov_10[0] = ov_10[0] * fac[2];
                            ov_10[1] = ov_10[1] * fac[3];
                            ov_10[2] = ov_10[2] * fac[2];
                            ov_10[3] = ov_10[3] * fac[3];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 32 + 8 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&ov_10[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_10[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_10[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_10[3])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            float ov_11[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_11[3]))
                                : "r"(tmem_tmem + 32 + 48));
                            float ov_12[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_12[3]))
                                : "r"(tmem_tmem + 32 + 48 + 1048576));
                            float ov_13[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_13[3]))
                                : "r"(tmem_tmem + 32 + 48 + 8));
                            float ov_14[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                                " {%0, %1, %2, %3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&ov_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&ov_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&ov_14[3]))
                                : "r"(tmem_tmem + 32 + 48 + 8 + 1048576));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            ov_11[0] = ov_11[0] * fac[0];
                            ov_11[1] = ov_11[1] * fac[1];
                            ov_11[2] = ov_11[2] * fac[0];
                            ov_11[3] = ov_11[3] * fac[1];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 48), "r"(*reinterpret_cast<const uint32_t*>(&ov_11[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_11[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_11[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_11[3])));
                            ov_12[0] = ov_12[0] * fac[0];
                            ov_12[1] = ov_12[1] * fac[1];
                            ov_12[2] = ov_12[2] * fac[0];
                            ov_12[3] = ov_12[3] * fac[1];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 48 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&ov_12[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_12[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_12[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_12[3])));
                            ov_13[0] = ov_13[0] * fac[2];
                            ov_13[1] = ov_13[1] * fac[3];
                            ov_13[2] = ov_13[2] * fac[2];
                            ov_13[3] = ov_13[3] * fac[3];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 48 + 8), "r"(*reinterpret_cast<const uint32_t*>(&ov_13[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_13[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_13[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_13[3])));
                            ov_14[0] = ov_14[0] * fac[2];
                            ov_14[1] = ov_14[1] * fac[3];
                            ov_14[2] = ov_14[2] * fac[2];
                            ov_14[3] = ov_14[3] * fac[3];
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x1.b32"
                                " [%0], {%1, %2, %3, %4};"
                                :: "r"(tmem_tmem + 32 + 48 + 8 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&ov_14[0])), "r"(*reinterpret_cast<const uint32_t*>(&ov_14[1])), "r"(*reinterpret_cast<const uint32_t*>(&ov_14[2])), "r"(*reinterpret_cast<const uint32_t*>(&ov_14[3])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                        }
                    }
                    uint32_t p0_bf16[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p0[_lp*2 + 0], p0[_lp*2+1 + 0]));
                        p0_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t p1_bf16[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p1[_lp*2 + 0], p1[_lp*2+1 + 0]));
                        p1_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_0 = static_cast<uint32_t>(pt_addr_0);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&p0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&p0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&p1_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&p1_bf16[1]))
                        : "memory");
                    uint32_t p0_3_bf16[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p0_3[_lp*2 + 0], p0_3[_lp*2+1 + 0]));
                        p0_3_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t p1_4_bf16[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p1_4[_lp*2 + 0], p1_4[_lp*2+1 + 0]));
                        p1_4_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(pt_addr_0 + 1024);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&p0_3_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&p0_3_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&p1_4_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&p1_4_bf16[1]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(so_ready_addr);
                    tcur = tcur + 1;
                }
                if (real[0] == -CAKE_INF) {
                    lrow[0] = 0.0f;
                    mrow[0] = -CAKE_INF;
                }
                if (real[1] == -CAKE_INF) {
                    lrow[1] = 0.0f;
                    mrow[1] = -CAKE_INF;
                }
                if (real[2] == -CAKE_INF) {
                    lrow[2] = 0.0f;
                    mrow[2] = -CAKE_INF;
                }
                if (real[3] == -CAKE_INF) {
                    lrow[3] = 0.0f;
                    mrow[3] = -CAKE_INF;
                }
                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, lrow[0], 4);
                lrow[0] = lrow[0] + _shfl_xor_12;
                float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, lrow[1], 4);
                lrow[1] = lrow[1] + _shfl_xor_13;
                float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, lrow[2], 4);
                lrow[2] = lrow[2] + _shfl_xor_14;
                float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, lrow[3], 4);
                lrow[3] = lrow[3] + _shfl_xor_15;
                float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, lrow[0], 8);
                lrow[0] = lrow[0] + _shfl_xor_16;
                float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, lrow[1], 8);
                lrow[1] = lrow[1] + _shfl_xor_17;
                float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, lrow[2], 8);
                lrow[2] = lrow[2] + _shfl_xor_18;
                float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, lrow[3], 8);
                lrow[3] = lrow[3] + _shfl_xor_19;
                float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, lrow[0], 16);
                lrow[0] = lrow[0] + _shfl_xor_20;
                float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, lrow[1], 16);
                lrow[1] = lrow[1] + _shfl_xor_21;
                float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, lrow[2], 16);
                lrow[2] = lrow[2] + _shfl_xor_22;
                float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, lrow[3], 16);
                lrow[3] = lrow[3] + _shfl_xor_23;
                if (lane < 4) {
                    sums[w * 16 + kb] = lrow[0];
                    sums[w * 16 + kb + 1] = lrow[1];
                    sums[w * 16 + 8 + kb] = lrow[2];
                    sums[w * 16 + 8 + kb + 1] = lrow[3];
                }
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                float lsum[4];
                lsum[0] = 0.0f;
                lsum[1] = 0.0f;
                lsum[2] = 0.0f;
                lsum[3] = 0.0f;
                float _sums_reg_0[2];
                {
                    const float* _smem_ptr = reinterpret_cast<const float*>(sums);
                    #pragma unroll
                    for (int _lr = 0; _lr < 2; _lr++)
                        _sums_reg_0[_lr] = _smem_ptr[(kb) + _lr];
                }
                lsum[0] = lsum[0] + _sums_reg_0[0];
                lsum[1] = lsum[1] + _sums_reg_0[1];
                float _sums_reg_1[2];
                {
                    const float* _smem_ptr = reinterpret_cast<const float*>(sums);
                    #pragma unroll
                    for (int _lr = 0; _lr < 2; _lr++)
                        _sums_reg_1[_lr] = _smem_ptr[(8 + kb) + _lr];
                }
                lsum[2] = lsum[2] + _sums_reg_1[0];
                lsum[3] = lsum[3] + _sums_reg_1[1];
                float _sums_reg_2[2];
                {
                    const float* _smem_ptr = reinterpret_cast<const float*>(sums);
                    #pragma unroll
                    for (int _lr = 0; _lr < 2; _lr++)
                        _sums_reg_2[_lr] = _smem_ptr[(16 + kb) + _lr];
                }
                lsum[0] = lsum[0] + _sums_reg_2[0];
                lsum[1] = lsum[1] + _sums_reg_2[1];
                float _sums_reg_3[2];
                {
                    const float* _smem_ptr = reinterpret_cast<const float*>(sums);
                    #pragma unroll
                    for (int _lr = 0; _lr < 2; _lr++)
                        _sums_reg_3[_lr] = _smem_ptr[(24 + kb) + _lr];
                }
                lsum[2] = lsum[2] + _sums_reg_3[0];
                lsum[3] = lsum[3] + _sums_reg_3[1];
                float _sums_reg_4[2];
                {
                    const float* _smem_ptr = reinterpret_cast<const float*>(sums);
                    #pragma unroll
                    for (int _lr = 0; _lr < 2; _lr++)
                        _sums_reg_4[_lr] = _smem_ptr[(32 + kb) + _lr];
                }
                lsum[0] = lsum[0] + _sums_reg_4[0];
                lsum[1] = lsum[1] + _sums_reg_4[1];
                float _sums_reg_5[2];
                {
                    const float* _smem_ptr = reinterpret_cast<const float*>(sums);
                    #pragma unroll
                    for (int _lr = 0; _lr < 2; _lr++)
                        _sums_reg_5[_lr] = _smem_ptr[(40 + kb) + _lr];
                }
                lsum[2] = lsum[2] + _sums_reg_5[0];
                lsum[3] = lsum[3] + _sums_reg_5[1];
                float _sums_reg_6[2];
                {
                    const float* _smem_ptr = reinterpret_cast<const float*>(sums);
                    #pragma unroll
                    for (int _lr = 0; _lr < 2; _lr++)
                        _sums_reg_6[_lr] = _smem_ptr[(48 + kb) + _lr];
                }
                lsum[0] = lsum[0] + _sums_reg_6[0];
                lsum[1] = lsum[1] + _sums_reg_6[1];
                float _sums_reg_7[2];
                {
                    const float* _smem_ptr = reinterpret_cast<const float*>(sums);
                    #pragma unroll
                    for (int _lr = 0; _lr < 2; _lr++)
                        _sums_reg_7[_lr] = _smem_ptr[(56 + kb) + _lr];
                }
                lsum[2] = lsum[2] + _sums_reg_7[0];
                lsum[3] = lsum[3] + _sums_reg_7[1];
                float osc[4];
                int head = kb;
                float sink_v = -CAKE_INF;
                if (head < heads) {
                    sink_v = sink[head];
                }
                float sl = sink_v * 1.4426950408889634f;
                float meff = mrow[0];
                if (real[0] == -CAKE_INF) {
                    meff = -1073741824.0f * scale_log2;
                }
                float _max_51 = max_noftz(meff, sl);
                float mfin = _max_51;
                float _exp2_20 = approx_exp2(meff - mfin);
                float fin = _exp2_20;
                float _exp2_21 = approx_exp2(sl - mfin);
                float _max_52 = max_noftz(lsum[0] * fin + _exp2_21, 1e-30f);
                float lfin = _max_52;
                if (w == 0 && lane < 4) {
                    if (head < heads) {
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(lfin));
                        lse[query * heads + head] = _log2_0 + mfin;
                    }
                }
                osc[0] = fin / lfin;
                int head_1 = kb + 1;
                float sink_v_2 = -CAKE_INF;
                if (head_1 < heads) {
                    sink_v_2 = sink[head_1];
                }
                float sl_3 = sink_v_2 * 1.4426950408889634f;
                float meff_4 = mrow[1];
                if (real[1] == -CAKE_INF) {
                    meff_4 = -1073741824.0f * scale_log2;
                }
                float _max_53 = max_noftz(meff_4, sl_3);
                float mfin_5 = _max_53;
                float _exp2_22 = approx_exp2(meff_4 - mfin_5);
                float fin_6 = _exp2_22;
                float _exp2_23 = approx_exp2(sl_3 - mfin_5);
                float _max_54 = max_noftz(lsum[1] * fin_6 + _exp2_23, 1e-30f);
                float lfin_7 = _max_54;
                if (w == 0 && lane < 4) {
                    if (head_1 < heads) {
                        float _log2_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(lfin_7));
                        lse[query * heads + head_1] = _log2_1 + mfin_5;
                    }
                }
                osc[1] = fin_6 / lfin_7;
                int head_8 = 8 + kb;
                float sink_v_9 = -CAKE_INF;
                if (head_8 < heads) {
                    sink_v_9 = sink[head_8];
                }
                float sl_10 = sink_v_9 * 1.4426950408889634f;
                float meff_11 = mrow[2];
                if (real[2] == -CAKE_INF) {
                    meff_11 = -1073741824.0f * scale_log2;
                }
                float _max_55 = max_noftz(meff_11, sl_10);
                float mfin_12 = _max_55;
                float _exp2_24 = approx_exp2(meff_11 - mfin_12);
                float fin_13 = _exp2_24;
                float _exp2_25 = approx_exp2(sl_10 - mfin_12);
                float _max_56 = max_noftz(lsum[2] * fin_13 + _exp2_25, 1e-30f);
                float lfin_14 = _max_56;
                if (w == 0 && lane < 4) {
                    if (head_8 < heads) {
                        float _log2_2;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(lfin_14));
                        lse[query * heads + head_8] = _log2_2 + mfin_12;
                    }
                }
                osc[2] = fin_13 / lfin_14;
                int head_15 = 8 + kb + 1;
                float sink_v_16 = -CAKE_INF;
                if (head_15 < heads) {
                    sink_v_16 = sink[head_15];
                }
                float sl_17 = sink_v_16 * 1.4426950408889634f;
                float meff_18 = mrow[3];
                if (real[3] == -CAKE_INF) {
                    meff_18 = -1073741824.0f * scale_log2;
                }
                float _max_57 = max_noftz(meff_18, sl_17);
                float mfin_19 = _max_57;
                float _exp2_26 = approx_exp2(meff_18 - mfin_19);
                float fin_20 = _exp2_26;
                float _exp2_27 = approx_exp2(sl_17 - mfin_19);
                float _max_58 = max_noftz(lsum[3] * fin_20 + _exp2_27, 1e-30f);
                float lfin_21 = _max_58;
                if (w == 0 && lane < 4) {
                    if (head_15 < heads) {
                        float _log2_3;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_3) : "f"(lfin_21));
                        lse[query * heads + head_15] = _log2_3 + mfin_19;
                    }
                }
                osc[3] = fin_20 / lfin_21;
                int last_pair = (4 * (tcur - 1) + 3) % 6;
                unsigned int last_phase = (4 * (tcur - 1) + 3) / 6 & 1;
                mbarrier_wait(sv_done_addr + (last_pair) * 8, last_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float of[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of[0])), "=r"(*reinterpret_cast<uint32_t*>(&of[1])), "=r"(*reinterpret_cast<uint32_t*>(&of[2])), "=r"(*reinterpret_cast<uint32_t*>(&of[3]))
                    : "r"(tmem_tmem + 32));
                float of_22[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_22[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_22[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_22[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_22[3]))
                    : "r"(tmem_tmem + 32 + 1048576));
                float of_23[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_23[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_23[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_23[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_23[3]))
                    : "r"(tmem_tmem + 32 + 8));
                float of_24[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_24[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_24[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_24[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_24[3]))
                    : "r"(tmem_tmem + 32 + 8 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int ob[4];
                of[0] = of[0] * osc[0];
                of[1] = of[1] * osc[1];
                of[2] = of[2] * osc[0];
                of[3] = of[3] * osc[1];
                uint32_t of_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of[_lp*2 + 0], of[_lp*2+1 + 0]));
                    of_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob[0] = of_bf16[0];
                ob[1] = of_bf16[1];
                of_22[0] = of_22[0] * osc[0];
                of_22[1] = of_22[1] * osc[1];
                of_22[2] = of_22[2] * osc[0];
                of_22[3] = of_22[3] * osc[1];
                uint32_t of_22_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_22[_lp*2 + 0], of_22[_lp*2+1 + 0]));
                    of_22_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob[2] = of_22_bf16[0];
                ob[3] = of_22_bf16[1];
                unsigned int oaddr = qt_addr + (unsigned int)(pt_atom * 2048) + (unsigned int)(r8 * 128) + (unsigned int)(pt_chunk << 4);
                uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(oaddr);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&ob[0])), "r"(*reinterpret_cast<const uint32_t*>(&ob[1])), "r"(*reinterpret_cast<const uint32_t*>(&ob[2])), "r"(*reinterpret_cast<const uint32_t*>(&ob[3]))
                    : "memory");
                unsigned int ob_25[4];
                of_23[0] = of_23[0] * osc[2];
                of_23[1] = of_23[1] * osc[3];
                of_23[2] = of_23[2] * osc[2];
                of_23[3] = of_23[3] * osc[3];
                uint32_t of_23_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_23[_lp*2 + 0], of_23[_lp*2+1 + 0]));
                    of_23_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_25[0] = of_23_bf16[0];
                ob_25[1] = of_23_bf16[1];
                of_24[0] = of_24[0] * osc[2];
                of_24[1] = of_24[1] * osc[3];
                of_24[2] = of_24[2] * osc[2];
                of_24[3] = of_24[3] * osc[3];
                uint32_t of_24_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_24[_lp*2 + 0], of_24[_lp*2+1 + 0]));
                    of_24_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_25[2] = of_24_bf16[0];
                ob_25[3] = of_24_bf16[1];
                unsigned int oaddr_26 = qt_addr + (unsigned int)(pt_atom * 2048) + (unsigned int)((8 + r8) * 128) + (unsigned int)(pt_chunk << 4);
                uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(oaddr_26);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&ob_25[0])), "r"(*reinterpret_cast<const uint32_t*>(&ob_25[1])), "r"(*reinterpret_cast<const uint32_t*>(&ob_25[2])), "r"(*reinterpret_cast<const uint32_t*>(&ob_25[3]))
                    : "memory");
                float of_27[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_27[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_27[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_27[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_27[3]))
                    : "r"(tmem_tmem + 32 + 16));
                float of_28[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_28[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_28[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_28[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_28[3]))
                    : "r"(tmem_tmem + 32 + 16 + 1048576));
                float of_29[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_29[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_29[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_29[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_29[3]))
                    : "r"(tmem_tmem + 32 + 16 + 8));
                float of_30[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_30[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_30[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_30[3]))
                    : "r"(tmem_tmem + 32 + 16 + 8 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int ob_31[4];
                of_27[0] = of_27[0] * osc[0];
                of_27[1] = of_27[1] * osc[1];
                of_27[2] = of_27[2] * osc[0];
                of_27[3] = of_27[3] * osc[1];
                uint32_t of_27_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_27[_lp*2 + 0], of_27[_lp*2+1 + 0]));
                    of_27_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_31[0] = of_27_bf16[0];
                ob_31[1] = of_27_bf16[1];
                of_28[0] = of_28[0] * osc[0];
                of_28[1] = of_28[1] * osc[1];
                of_28[2] = of_28[2] * osc[0];
                of_28[3] = of_28[3] * osc[1];
                uint32_t of_28_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_28[_lp*2 + 0], of_28[_lp*2+1 + 0]));
                    of_28_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_31[2] = of_28_bf16[0];
                ob_31[3] = of_28_bf16[1];
                unsigned int oaddr_32 = qt_addr + (unsigned int)((2 + pt_atom) * 2048) + (unsigned int)(r8 * 128) + (unsigned int)(pt_chunk << 4);
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(oaddr_32);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&ob_31[0])), "r"(*reinterpret_cast<const uint32_t*>(&ob_31[1])), "r"(*reinterpret_cast<const uint32_t*>(&ob_31[2])), "r"(*reinterpret_cast<const uint32_t*>(&ob_31[3]))
                    : "memory");
                unsigned int ob_33[4];
                of_29[0] = of_29[0] * osc[2];
                of_29[1] = of_29[1] * osc[3];
                of_29[2] = of_29[2] * osc[2];
                of_29[3] = of_29[3] * osc[3];
                uint32_t of_29_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_29[_lp*2 + 0], of_29[_lp*2+1 + 0]));
                    of_29_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_33[0] = of_29_bf16[0];
                ob_33[1] = of_29_bf16[1];
                of_30[0] = of_30[0] * osc[2];
                of_30[1] = of_30[1] * osc[3];
                of_30[2] = of_30[2] * osc[2];
                of_30[3] = of_30[3] * osc[3];
                uint32_t of_30_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_30[_lp*2 + 0], of_30[_lp*2+1 + 0]));
                    of_30_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_33[2] = of_30_bf16[0];
                ob_33[3] = of_30_bf16[1];
                unsigned int oaddr_34 = qt_addr + (unsigned int)((2 + pt_atom) * 2048) + (unsigned int)((8 + r8) * 128) + (unsigned int)(pt_chunk << 4);
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(oaddr_34);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&ob_33[0])), "r"(*reinterpret_cast<const uint32_t*>(&ob_33[1])), "r"(*reinterpret_cast<const uint32_t*>(&ob_33[2])), "r"(*reinterpret_cast<const uint32_t*>(&ob_33[3]))
                    : "memory");
                float of_35[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_35[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_35[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_35[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_35[3]))
                    : "r"(tmem_tmem + 32 + 32));
                float of_36[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_36[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_36[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_36[3]))
                    : "r"(tmem_tmem + 32 + 32 + 1048576));
                float of_37[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_37[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_37[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_37[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_37[3]))
                    : "r"(tmem_tmem + 32 + 32 + 8));
                float of_38[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_38[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_38[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_38[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_38[3]))
                    : "r"(tmem_tmem + 32 + 32 + 8 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int ob_39[4];
                of_35[0] = of_35[0] * osc[0];
                of_35[1] = of_35[1] * osc[1];
                of_35[2] = of_35[2] * osc[0];
                of_35[3] = of_35[3] * osc[1];
                uint32_t of_35_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_35[_lp*2 + 0], of_35[_lp*2+1 + 0]));
                    of_35_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_39[0] = of_35_bf16[0];
                ob_39[1] = of_35_bf16[1];
                of_36[0] = of_36[0] * osc[0];
                of_36[1] = of_36[1] * osc[1];
                of_36[2] = of_36[2] * osc[0];
                of_36[3] = of_36[3] * osc[1];
                uint32_t of_36_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_36[_lp*2 + 0], of_36[_lp*2+1 + 0]));
                    of_36_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_39[2] = of_36_bf16[0];
                ob_39[3] = of_36_bf16[1];
                unsigned int oaddr_40 = qt_addr + (unsigned int)((4 + pt_atom) * 2048) + (unsigned int)(r8 * 128) + (unsigned int)(pt_chunk << 4);
                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(oaddr_40);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&ob_39[0])), "r"(*reinterpret_cast<const uint32_t*>(&ob_39[1])), "r"(*reinterpret_cast<const uint32_t*>(&ob_39[2])), "r"(*reinterpret_cast<const uint32_t*>(&ob_39[3]))
                    : "memory");
                unsigned int ob_41[4];
                of_37[0] = of_37[0] * osc[2];
                of_37[1] = of_37[1] * osc[3];
                of_37[2] = of_37[2] * osc[2];
                of_37[3] = of_37[3] * osc[3];
                uint32_t of_37_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_37[_lp*2 + 0], of_37[_lp*2+1 + 0]));
                    of_37_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_41[0] = of_37_bf16[0];
                ob_41[1] = of_37_bf16[1];
                of_38[0] = of_38[0] * osc[2];
                of_38[1] = of_38[1] * osc[3];
                of_38[2] = of_38[2] * osc[2];
                of_38[3] = of_38[3] * osc[3];
                uint32_t of_38_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_38[_lp*2 + 0], of_38[_lp*2+1 + 0]));
                    of_38_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_41[2] = of_38_bf16[0];
                ob_41[3] = of_38_bf16[1];
                unsigned int oaddr_42 = qt_addr + (unsigned int)((4 + pt_atom) * 2048) + (unsigned int)((8 + r8) * 128) + (unsigned int)(pt_chunk << 4);
                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(oaddr_42);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&ob_41[0])), "r"(*reinterpret_cast<const uint32_t*>(&ob_41[1])), "r"(*reinterpret_cast<const uint32_t*>(&ob_41[2])), "r"(*reinterpret_cast<const uint32_t*>(&ob_41[3]))
                    : "memory");
                float of_43[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_43[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_43[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_43[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_43[3]))
                    : "r"(tmem_tmem + 32 + 48));
                float of_44[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_44[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_44[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_44[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_44[3]))
                    : "r"(tmem_tmem + 32 + 48 + 1048576));
                float of_45[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_45[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_45[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_45[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_45[3]))
                    : "r"(tmem_tmem + 32 + 48 + 8));
                float of_46[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&of_46[0])), "=r"(*reinterpret_cast<uint32_t*>(&of_46[1])), "=r"(*reinterpret_cast<uint32_t*>(&of_46[2])), "=r"(*reinterpret_cast<uint32_t*>(&of_46[3]))
                    : "r"(tmem_tmem + 32 + 48 + 8 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                unsigned int ob_47[4];
                of_43[0] = of_43[0] * osc[0];
                of_43[1] = of_43[1] * osc[1];
                of_43[2] = of_43[2] * osc[0];
                of_43[3] = of_43[3] * osc[1];
                uint32_t of_43_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_43[_lp*2 + 0], of_43[_lp*2+1 + 0]));
                    of_43_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_47[0] = of_43_bf16[0];
                ob_47[1] = of_43_bf16[1];
                of_44[0] = of_44[0] * osc[0];
                of_44[1] = of_44[1] * osc[1];
                of_44[2] = of_44[2] * osc[0];
                of_44[3] = of_44[3] * osc[1];
                uint32_t of_44_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_44[_lp*2 + 0], of_44[_lp*2+1 + 0]));
                    of_44_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_47[2] = of_44_bf16[0];
                ob_47[3] = of_44_bf16[1];
                unsigned int oaddr_48 = qt_addr + (unsigned int)((6 + pt_atom) * 2048) + (unsigned int)(r8 * 128) + (unsigned int)(pt_chunk << 4);
                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(oaddr_48);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&ob_47[0])), "r"(*reinterpret_cast<const uint32_t*>(&ob_47[1])), "r"(*reinterpret_cast<const uint32_t*>(&ob_47[2])), "r"(*reinterpret_cast<const uint32_t*>(&ob_47[3]))
                    : "memory");
                unsigned int ob_49[4];
                of_45[0] = of_45[0] * osc[2];
                of_45[1] = of_45[1] * osc[3];
                of_45[2] = of_45[2] * osc[2];
                of_45[3] = of_45[3] * osc[3];
                uint32_t of_45_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_45[_lp*2 + 0], of_45[_lp*2+1 + 0]));
                    of_45_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_49[0] = of_45_bf16[0];
                ob_49[1] = of_45_bf16[1];
                of_46[0] = of_46[0] * osc[2];
                of_46[1] = of_46[1] * osc[3];
                of_46[2] = of_46[2] * osc[2];
                of_46[3] = of_46[3] * osc[3];
                uint32_t of_46_bf16[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(of_46[_lp*2 + 0], of_46[_lp*2+1 + 0]));
                    of_46_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                ob_49[2] = of_46_bf16[0];
                ob_49[3] = of_46_bf16[1];
                unsigned int oaddr_50 = qt_addr + (unsigned int)((6 + pt_atom) * 2048) + (unsigned int)((8 + r8) * 128) + (unsigned int)(pt_chunk << 4);
                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(oaddr_50);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&ob_49[0])), "r"(*reinterpret_cast<const uint32_t*>(&ob_49[1])), "r"(*reinterpret_cast<const uint32_t*>(&ob_49[2])), "r"(*reinterpret_cast<const uint32_t*>(&ob_49[3]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 0, 128;" ::: "memory");
                if (w == 0) {
                    if (elect_sync()) {
                        tma_store_4d((&out), 0, 0, 0, query, qt_addr);
                        tma_store_4d((&out), 0, 0, 1, query, qt_addr + 2048);
                        tma_store_4d((&out), 0, 0, 2, query, qt_addr + 4096);
                        tma_store_4d((&out), 0, 0, 3, query, qt_addr + 6144);
                        tma_store_4d((&out), 0, 0, 4, query, qt_addr + 8192);
                        tma_store_4d((&out), 0, 0, 5, query, qt_addr + 10240);
                        tma_store_4d((&out), 0, 0, 6, query, qt_addr + 12288);
                        tma_store_4d((&out), 0, 0, 7, query, qt_addr + 14336);
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        mbarrier_arrive(q_free_addr);
                    }
                }
                mbarrier_wait(clc_full_addr, outer_phase);
                uint32_t _clc_valid_0 = 0;
                uint32_t _clc_ctaid_x_0;
                uint32_t _clc_ctaid_y_0;
                uint32_t _clc_ctaid_z_0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_0), "=r"(_clc_ctaid_y_0), "=r"(_clc_ctaid_z_0), "=r"(_clc_valid_0)
                    : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                    : "memory");
                mbarrier_arrive(clc_empty_addr);
                if (_clc_valid_0 == 0) {
                    break;
                }
                query = _clc_ctaid_x_0;
                outer_phase = outer_phase ^ 1;
            }
            if (warp == 3) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(128));
            }
        }
    }
    // ---- Role: producer ----
    if (warp >= 4 && warp <= 7) {
        { // producer_main
            int query_1 = blockIdx.x;
            unsigned int outer_phase_1 = 0;
            int tcur_1 = 0;
            int producer_warp = warp - 4;
            #pragma unroll 1
            for (int outer_1 = 0; outer_1 < num_queries; outer_1++) {
                int active_1 = topk;
                if (derive_length != 0) {
                    int len_slot_1 = outer_1 & 1;
                    unsigned int len_phase_1 = outer_1 >> 1 & 1;
                    mbarrier_wait(len_ready_addr + (len_slot_1) * 8, len_phase_1);
                    int _max_59 = ((len_cells[len_slot_1]) > (0) ? (len_cells[len_slot_1]) : (0));
                    int _min_2 = ((_max_59) < (topk) ? (_max_59) : (topk));
                    active_1 = _min_2;
                } else if (has_topk_length != 0) {
                    int _max_60 = ((topk_length[query_1]) > (0) ? (topk_length[query_1]) : (0));
                    int _min_3 = ((_max_60) < (topk) ? (_max_60) : (topk));
                    active_1 = _min_3;
                }
                int active_0_1 = active_1;
                int _max_61 = (((active_0_1 + 128 - 1) / 128) > (1) ? ((active_0_1 + 128 - 1) / 128) : (1));
                int blocks_1 = _max_61;
                long long row_base = (long long)indices_offset + (long long)query_1 * (long long)idx_stride;
                #pragma unroll 1
                for (int block_1 = 0; block_1 < blocks_1; block_1++) {
                    if (elect_sync()) {
                        int idx[32];
                        int smallest = num_kv;
                        int largest = -1;
                        int whole_tile = active_0_1 >= block_1 * 128 + 128 && ((idx_stride | indices_offset) & 7) == 0;
                        if (whole_tile != 0) {
                            int _vec_load_0[4];
                            {
                                const int4* _ivptr_0 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 128 + producer_warp * 4)) + 0);
                                int4 _ivld_0;
                                _ivld_0 = *_ivptr_0;
                                _vec_load_0[0 + 0] = _ivld_0.x;
                                _vec_load_0[0 + 1] = _ivld_0.y;
                                _vec_load_0[0 + 2] = _ivld_0.z;
                                _vec_load_0[0 + 3] = _ivld_0.w;
                            }
                            idx[0] = _vec_load_0[0];
                            idx[1] = _vec_load_0[1];
                            idx[2] = _vec_load_0[2];
                            idx[3] = _vec_load_0[3];
                            int _max_62 = ((_vec_load_0[0]) > (_vec_load_0[1]) ? (_vec_load_0[0]) : (_vec_load_0[1]));
                            int _max_63 = ((_vec_load_0[2]) > (_vec_load_0[3]) ? (_vec_load_0[2]) : (_vec_load_0[3]));
                            int _max_64 = ((_max_62) > (_max_63) ? (_max_62) : (_max_63));
                            int _max_65 = ((largest) > (_max_64) ? (largest) : (_max_64));
                            largest = _max_65;
                            int _min_4 = ((_vec_load_0[0]) < (_vec_load_0[1]) ? (_vec_load_0[0]) : (_vec_load_0[1]));
                            int _min_5 = ((_vec_load_0[2]) < (_vec_load_0[3]) ? (_vec_load_0[2]) : (_vec_load_0[3]));
                            int _min_6 = ((_min_4) < (_min_5) ? (_min_4) : (_min_5));
                            int _min_7 = ((smallest) < (_min_6) ? (smallest) : (_min_6));
                            smallest = _min_7;
                            int _vec_load_1[4];
                            {
                                const int4* _ivptr_1 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 128 + 16 + producer_warp * 4)) + 0);
                                int4 _ivld_1;
                                _ivld_1 = *_ivptr_1;
                                _vec_load_1[0 + 0] = _ivld_1.x;
                                _vec_load_1[0 + 1] = _ivld_1.y;
                                _vec_load_1[0 + 2] = _ivld_1.z;
                                _vec_load_1[0 + 3] = _ivld_1.w;
                            }
                            idx[4] = _vec_load_1[0];
                            idx[5] = _vec_load_1[1];
                            idx[6] = _vec_load_1[2];
                            idx[7] = _vec_load_1[3];
                            int _max_66 = ((_vec_load_1[0]) > (_vec_load_1[1]) ? (_vec_load_1[0]) : (_vec_load_1[1]));
                            int _max_67 = ((_vec_load_1[2]) > (_vec_load_1[3]) ? (_vec_load_1[2]) : (_vec_load_1[3]));
                            int _max_68 = ((_max_66) > (_max_67) ? (_max_66) : (_max_67));
                            int _max_69 = ((largest) > (_max_68) ? (largest) : (_max_68));
                            largest = _max_69;
                            int _min_8 = ((_vec_load_1[0]) < (_vec_load_1[1]) ? (_vec_load_1[0]) : (_vec_load_1[1]));
                            int _min_9 = ((_vec_load_1[2]) < (_vec_load_1[3]) ? (_vec_load_1[2]) : (_vec_load_1[3]));
                            int _min_10 = ((_min_8) < (_min_9) ? (_min_8) : (_min_9));
                            int _min_11 = ((smallest) < (_min_10) ? (smallest) : (_min_10));
                            smallest = _min_11;
                            int _vec_load_2[4];
                            {
                                const int4* _ivptr_2 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 128 + 32 + producer_warp * 4)) + 0);
                                int4 _ivld_2;
                                _ivld_2 = *_ivptr_2;
                                _vec_load_2[0 + 0] = _ivld_2.x;
                                _vec_load_2[0 + 1] = _ivld_2.y;
                                _vec_load_2[0 + 2] = _ivld_2.z;
                                _vec_load_2[0 + 3] = _ivld_2.w;
                            }
                            idx[8] = _vec_load_2[0];
                            idx[9] = _vec_load_2[1];
                            idx[10] = _vec_load_2[2];
                            idx[11] = _vec_load_2[3];
                            int _max_70 = ((_vec_load_2[0]) > (_vec_load_2[1]) ? (_vec_load_2[0]) : (_vec_load_2[1]));
                            int _max_71 = ((_vec_load_2[2]) > (_vec_load_2[3]) ? (_vec_load_2[2]) : (_vec_load_2[3]));
                            int _max_72 = ((_max_70) > (_max_71) ? (_max_70) : (_max_71));
                            int _max_73 = ((largest) > (_max_72) ? (largest) : (_max_72));
                            largest = _max_73;
                            int _min_12 = ((_vec_load_2[0]) < (_vec_load_2[1]) ? (_vec_load_2[0]) : (_vec_load_2[1]));
                            int _min_13 = ((_vec_load_2[2]) < (_vec_load_2[3]) ? (_vec_load_2[2]) : (_vec_load_2[3]));
                            int _min_14 = ((_min_12) < (_min_13) ? (_min_12) : (_min_13));
                            int _min_15 = ((smallest) < (_min_14) ? (smallest) : (_min_14));
                            smallest = _min_15;
                            int _vec_load_3[4];
                            {
                                const int4* _ivptr_3 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 128 + 48 + producer_warp * 4)) + 0);
                                int4 _ivld_3;
                                _ivld_3 = *_ivptr_3;
                                _vec_load_3[0 + 0] = _ivld_3.x;
                                _vec_load_3[0 + 1] = _ivld_3.y;
                                _vec_load_3[0 + 2] = _ivld_3.z;
                                _vec_load_3[0 + 3] = _ivld_3.w;
                            }
                            idx[12] = _vec_load_3[0];
                            idx[13] = _vec_load_3[1];
                            idx[14] = _vec_load_3[2];
                            idx[15] = _vec_load_3[3];
                            int _max_74 = ((_vec_load_3[0]) > (_vec_load_3[1]) ? (_vec_load_3[0]) : (_vec_load_3[1]));
                            int _max_75 = ((_vec_load_3[2]) > (_vec_load_3[3]) ? (_vec_load_3[2]) : (_vec_load_3[3]));
                            int _max_76 = ((_max_74) > (_max_75) ? (_max_74) : (_max_75));
                            int _max_77 = ((largest) > (_max_76) ? (largest) : (_max_76));
                            largest = _max_77;
                            int _min_16 = ((_vec_load_3[0]) < (_vec_load_3[1]) ? (_vec_load_3[0]) : (_vec_load_3[1]));
                            int _min_17 = ((_vec_load_3[2]) < (_vec_load_3[3]) ? (_vec_load_3[2]) : (_vec_load_3[3]));
                            int _min_18 = ((_min_16) < (_min_17) ? (_min_16) : (_min_17));
                            int _min_19 = ((smallest) < (_min_18) ? (smallest) : (_min_18));
                            smallest = _min_19;
                            int _vec_load_4[4];
                            {
                                const int4* _ivptr_4 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 128 + 64 + producer_warp * 4)) + 0);
                                int4 _ivld_4;
                                _ivld_4 = *_ivptr_4;
                                _vec_load_4[0 + 0] = _ivld_4.x;
                                _vec_load_4[0 + 1] = _ivld_4.y;
                                _vec_load_4[0 + 2] = _ivld_4.z;
                                _vec_load_4[0 + 3] = _ivld_4.w;
                            }
                            idx[16] = _vec_load_4[0];
                            idx[17] = _vec_load_4[1];
                            idx[18] = _vec_load_4[2];
                            idx[19] = _vec_load_4[3];
                            int _max_78 = ((_vec_load_4[0]) > (_vec_load_4[1]) ? (_vec_load_4[0]) : (_vec_load_4[1]));
                            int _max_79 = ((_vec_load_4[2]) > (_vec_load_4[3]) ? (_vec_load_4[2]) : (_vec_load_4[3]));
                            int _max_80 = ((_max_78) > (_max_79) ? (_max_78) : (_max_79));
                            int _max_81 = ((largest) > (_max_80) ? (largest) : (_max_80));
                            largest = _max_81;
                            int _min_20 = ((_vec_load_4[0]) < (_vec_load_4[1]) ? (_vec_load_4[0]) : (_vec_load_4[1]));
                            int _min_21 = ((_vec_load_4[2]) < (_vec_load_4[3]) ? (_vec_load_4[2]) : (_vec_load_4[3]));
                            int _min_22 = ((_min_20) < (_min_21) ? (_min_20) : (_min_21));
                            int _min_23 = ((smallest) < (_min_22) ? (smallest) : (_min_22));
                            smallest = _min_23;
                            int _vec_load_5[4];
                            {
                                const int4* _ivptr_5 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 128 + 80 + producer_warp * 4)) + 0);
                                int4 _ivld_5;
                                _ivld_5 = *_ivptr_5;
                                _vec_load_5[0 + 0] = _ivld_5.x;
                                _vec_load_5[0 + 1] = _ivld_5.y;
                                _vec_load_5[0 + 2] = _ivld_5.z;
                                _vec_load_5[0 + 3] = _ivld_5.w;
                            }
                            idx[20] = _vec_load_5[0];
                            idx[21] = _vec_load_5[1];
                            idx[22] = _vec_load_5[2];
                            idx[23] = _vec_load_5[3];
                            int _max_82 = ((_vec_load_5[0]) > (_vec_load_5[1]) ? (_vec_load_5[0]) : (_vec_load_5[1]));
                            int _max_83 = ((_vec_load_5[2]) > (_vec_load_5[3]) ? (_vec_load_5[2]) : (_vec_load_5[3]));
                            int _max_84 = ((_max_82) > (_max_83) ? (_max_82) : (_max_83));
                            int _max_85 = ((largest) > (_max_84) ? (largest) : (_max_84));
                            largest = _max_85;
                            int _min_24 = ((_vec_load_5[0]) < (_vec_load_5[1]) ? (_vec_load_5[0]) : (_vec_load_5[1]));
                            int _min_25 = ((_vec_load_5[2]) < (_vec_load_5[3]) ? (_vec_load_5[2]) : (_vec_load_5[3]));
                            int _min_26 = ((_min_24) < (_min_25) ? (_min_24) : (_min_25));
                            int _min_27 = ((smallest) < (_min_26) ? (smallest) : (_min_26));
                            smallest = _min_27;
                            int _vec_load_6[4];
                            {
                                const int4* _ivptr_6 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 128 + 96 + producer_warp * 4)) + 0);
                                int4 _ivld_6;
                                _ivld_6 = *_ivptr_6;
                                _vec_load_6[0 + 0] = _ivld_6.x;
                                _vec_load_6[0 + 1] = _ivld_6.y;
                                _vec_load_6[0 + 2] = _ivld_6.z;
                                _vec_load_6[0 + 3] = _ivld_6.w;
                            }
                            idx[24] = _vec_load_6[0];
                            idx[25] = _vec_load_6[1];
                            idx[26] = _vec_load_6[2];
                            idx[27] = _vec_load_6[3];
                            int _max_86 = ((_vec_load_6[0]) > (_vec_load_6[1]) ? (_vec_load_6[0]) : (_vec_load_6[1]));
                            int _max_87 = ((_vec_load_6[2]) > (_vec_load_6[3]) ? (_vec_load_6[2]) : (_vec_load_6[3]));
                            int _max_88 = ((_max_86) > (_max_87) ? (_max_86) : (_max_87));
                            int _max_89 = ((largest) > (_max_88) ? (largest) : (_max_88));
                            largest = _max_89;
                            int _min_28 = ((_vec_load_6[0]) < (_vec_load_6[1]) ? (_vec_load_6[0]) : (_vec_load_6[1]));
                            int _min_29 = ((_vec_load_6[2]) < (_vec_load_6[3]) ? (_vec_load_6[2]) : (_vec_load_6[3]));
                            int _min_30 = ((_min_28) < (_min_29) ? (_min_28) : (_min_29));
                            int _min_31 = ((smallest) < (_min_30) ? (smallest) : (_min_30));
                            smallest = _min_31;
                            int _vec_load_7[4];
                            {
                                const int4* _ivptr_7 = reinterpret_cast<const int4*>(indices + (row_base + (long long)(block_1 * 128 + 112 + producer_warp * 4)) + 0);
                                int4 _ivld_7;
                                _ivld_7 = *_ivptr_7;
                                _vec_load_7[0 + 0] = _ivld_7.x;
                                _vec_load_7[0 + 1] = _ivld_7.y;
                                _vec_load_7[0 + 2] = _ivld_7.z;
                                _vec_load_7[0 + 3] = _ivld_7.w;
                            }
                            idx[28] = _vec_load_7[0];
                            idx[29] = _vec_load_7[1];
                            idx[30] = _vec_load_7[2];
                            idx[31] = _vec_load_7[3];
                            int _max_90 = ((_vec_load_7[0]) > (_vec_load_7[1]) ? (_vec_load_7[0]) : (_vec_load_7[1]));
                            int _max_91 = ((_vec_load_7[2]) > (_vec_load_7[3]) ? (_vec_load_7[2]) : (_vec_load_7[3]));
                            int _max_92 = ((_max_90) > (_max_91) ? (_max_90) : (_max_91));
                            int _max_93 = ((largest) > (_max_92) ? (largest) : (_max_92));
                            largest = _max_93;
                            int _min_32 = ((_vec_load_7[0]) < (_vec_load_7[1]) ? (_vec_load_7[0]) : (_vec_load_7[1]));
                            int _min_33 = ((_vec_load_7[2]) < (_vec_load_7[3]) ? (_vec_load_7[2]) : (_vec_load_7[3]));
                            int _min_34 = ((_min_32) < (_min_33) ? (_min_32) : (_min_33));
                            int _min_35 = ((smallest) < (_min_34) ? (smallest) : (_min_34));
                            smallest = _min_35;
                        } else {
                            int position = block_1 * 128 + producer_warp * 4;
                            int _min_36 = ((position) < (topk - 1) ? (position) : (topk - 1));
                            int clamped = _min_36;
                            int value = indices[row_base + (long long)clamped];
                            if (position >= active_0_1) {
                                value = -1;
                            }
                            idx[0] = value;
                            int _max_94 = ((largest) > (value) ? (largest) : (value));
                            largest = _max_94;
                            int _min_37 = ((smallest) < (value) ? (smallest) : (value));
                            smallest = _min_37;
                            int position_0 = block_1 * 128 + producer_warp * 4 + 1;
                            int _min_38 = ((position_0) < (topk - 1) ? (position_0) : (topk - 1));
                            int clamped_1 = _min_38;
                            int value_2 = indices[row_base + (long long)clamped_1];
                            if (position_0 >= active_0_1) {
                                value_2 = -1;
                            }
                            idx[1] = value_2;
                            int _max_95 = ((largest) > (value_2) ? (largest) : (value_2));
                            largest = _max_95;
                            int _min_39 = ((smallest) < (value_2) ? (smallest) : (value_2));
                            smallest = _min_39;
                            int position_3 = block_1 * 128 + producer_warp * 4 + 2;
                            int _min_40 = ((position_3) < (topk - 1) ? (position_3) : (topk - 1));
                            int clamped_4 = _min_40;
                            int value_5 = indices[row_base + (long long)clamped_4];
                            if (position_3 >= active_0_1) {
                                value_5 = -1;
                            }
                            idx[2] = value_5;
                            int _max_96 = ((largest) > (value_5) ? (largest) : (value_5));
                            largest = _max_96;
                            int _min_41 = ((smallest) < (value_5) ? (smallest) : (value_5));
                            smallest = _min_41;
                            int position_6 = block_1 * 128 + producer_warp * 4 + 3;
                            int _min_42 = ((position_6) < (topk - 1) ? (position_6) : (topk - 1));
                            int clamped_7 = _min_42;
                            int value_8 = indices[row_base + (long long)clamped_7];
                            if (position_6 >= active_0_1) {
                                value_8 = -1;
                            }
                            idx[3] = value_8;
                            int _max_97 = ((largest) > (value_8) ? (largest) : (value_8));
                            largest = _max_97;
                            int _min_43 = ((smallest) < (value_8) ? (smallest) : (value_8));
                            smallest = _min_43;
                            int position_9 = block_1 * 128 + 16 + producer_warp * 4;
                            int _min_44 = ((position_9) < (topk - 1) ? (position_9) : (topk - 1));
                            int clamped_10 = _min_44;
                            int value_11 = indices[row_base + (long long)clamped_10];
                            if (position_9 >= active_0_1) {
                                value_11 = -1;
                            }
                            idx[4] = value_11;
                            int _max_98 = ((largest) > (value_11) ? (largest) : (value_11));
                            largest = _max_98;
                            int _min_45 = ((smallest) < (value_11) ? (smallest) : (value_11));
                            smallest = _min_45;
                            int position_12 = block_1 * 128 + 16 + producer_warp * 4 + 1;
                            int _min_46 = ((position_12) < (topk - 1) ? (position_12) : (topk - 1));
                            int clamped_13 = _min_46;
                            int value_14 = indices[row_base + (long long)clamped_13];
                            if (position_12 >= active_0_1) {
                                value_14 = -1;
                            }
                            idx[5] = value_14;
                            int _max_99 = ((largest) > (value_14) ? (largest) : (value_14));
                            largest = _max_99;
                            int _min_47 = ((smallest) < (value_14) ? (smallest) : (value_14));
                            smallest = _min_47;
                            int position_15 = block_1 * 128 + 16 + producer_warp * 4 + 2;
                            int _min_48 = ((position_15) < (topk - 1) ? (position_15) : (topk - 1));
                            int clamped_16 = _min_48;
                            int value_17 = indices[row_base + (long long)clamped_16];
                            if (position_15 >= active_0_1) {
                                value_17 = -1;
                            }
                            idx[6] = value_17;
                            int _max_100 = ((largest) > (value_17) ? (largest) : (value_17));
                            largest = _max_100;
                            int _min_49 = ((smallest) < (value_17) ? (smallest) : (value_17));
                            smallest = _min_49;
                            int position_18 = block_1 * 128 + 16 + producer_warp * 4 + 3;
                            int _min_50 = ((position_18) < (topk - 1) ? (position_18) : (topk - 1));
                            int clamped_19 = _min_50;
                            int value_20 = indices[row_base + (long long)clamped_19];
                            if (position_18 >= active_0_1) {
                                value_20 = -1;
                            }
                            idx[7] = value_20;
                            int _max_101 = ((largest) > (value_20) ? (largest) : (value_20));
                            largest = _max_101;
                            int _min_51 = ((smallest) < (value_20) ? (smallest) : (value_20));
                            smallest = _min_51;
                            int position_21 = block_1 * 128 + 32 + producer_warp * 4;
                            int _min_52 = ((position_21) < (topk - 1) ? (position_21) : (topk - 1));
                            int clamped_22 = _min_52;
                            int value_23 = indices[row_base + (long long)clamped_22];
                            if (position_21 >= active_0_1) {
                                value_23 = -1;
                            }
                            idx[8] = value_23;
                            int _max_102 = ((largest) > (value_23) ? (largest) : (value_23));
                            largest = _max_102;
                            int _min_53 = ((smallest) < (value_23) ? (smallest) : (value_23));
                            smallest = _min_53;
                            int position_24 = block_1 * 128 + 32 + producer_warp * 4 + 1;
                            int _min_54 = ((position_24) < (topk - 1) ? (position_24) : (topk - 1));
                            int clamped_25 = _min_54;
                            int value_26 = indices[row_base + (long long)clamped_25];
                            if (position_24 >= active_0_1) {
                                value_26 = -1;
                            }
                            idx[9] = value_26;
                            int _max_103 = ((largest) > (value_26) ? (largest) : (value_26));
                            largest = _max_103;
                            int _min_55 = ((smallest) < (value_26) ? (smallest) : (value_26));
                            smallest = _min_55;
                            int position_27 = block_1 * 128 + 32 + producer_warp * 4 + 2;
                            int _min_56 = ((position_27) < (topk - 1) ? (position_27) : (topk - 1));
                            int clamped_28 = _min_56;
                            int value_29 = indices[row_base + (long long)clamped_28];
                            if (position_27 >= active_0_1) {
                                value_29 = -1;
                            }
                            idx[10] = value_29;
                            int _max_104 = ((largest) > (value_29) ? (largest) : (value_29));
                            largest = _max_104;
                            int _min_57 = ((smallest) < (value_29) ? (smallest) : (value_29));
                            smallest = _min_57;
                            int position_30 = block_1 * 128 + 32 + producer_warp * 4 + 3;
                            int _min_58 = ((position_30) < (topk - 1) ? (position_30) : (topk - 1));
                            int clamped_31 = _min_58;
                            int value_32 = indices[row_base + (long long)clamped_31];
                            if (position_30 >= active_0_1) {
                                value_32 = -1;
                            }
                            idx[11] = value_32;
                            int _max_105 = ((largest) > (value_32) ? (largest) : (value_32));
                            largest = _max_105;
                            int _min_59 = ((smallest) < (value_32) ? (smallest) : (value_32));
                            smallest = _min_59;
                            int position_33 = block_1 * 128 + 48 + producer_warp * 4;
                            int _min_60 = ((position_33) < (topk - 1) ? (position_33) : (topk - 1));
                            int clamped_34 = _min_60;
                            int value_35 = indices[row_base + (long long)clamped_34];
                            if (position_33 >= active_0_1) {
                                value_35 = -1;
                            }
                            idx[12] = value_35;
                            int _max_106 = ((largest) > (value_35) ? (largest) : (value_35));
                            largest = _max_106;
                            int _min_61 = ((smallest) < (value_35) ? (smallest) : (value_35));
                            smallest = _min_61;
                            int position_36 = block_1 * 128 + 48 + producer_warp * 4 + 1;
                            int _min_62 = ((position_36) < (topk - 1) ? (position_36) : (topk - 1));
                            int clamped_37 = _min_62;
                            int value_38 = indices[row_base + (long long)clamped_37];
                            if (position_36 >= active_0_1) {
                                value_38 = -1;
                            }
                            idx[13] = value_38;
                            int _max_107 = ((largest) > (value_38) ? (largest) : (value_38));
                            largest = _max_107;
                            int _min_63 = ((smallest) < (value_38) ? (smallest) : (value_38));
                            smallest = _min_63;
                            int position_39 = block_1 * 128 + 48 + producer_warp * 4 + 2;
                            int _min_64 = ((position_39) < (topk - 1) ? (position_39) : (topk - 1));
                            int clamped_40 = _min_64;
                            int value_41 = indices[row_base + (long long)clamped_40];
                            if (position_39 >= active_0_1) {
                                value_41 = -1;
                            }
                            idx[14] = value_41;
                            int _max_108 = ((largest) > (value_41) ? (largest) : (value_41));
                            largest = _max_108;
                            int _min_65 = ((smallest) < (value_41) ? (smallest) : (value_41));
                            smallest = _min_65;
                            int position_42 = block_1 * 128 + 48 + producer_warp * 4 + 3;
                            int _min_66 = ((position_42) < (topk - 1) ? (position_42) : (topk - 1));
                            int clamped_43 = _min_66;
                            int value_44 = indices[row_base + (long long)clamped_43];
                            if (position_42 >= active_0_1) {
                                value_44 = -1;
                            }
                            idx[15] = value_44;
                            int _max_109 = ((largest) > (value_44) ? (largest) : (value_44));
                            largest = _max_109;
                            int _min_67 = ((smallest) < (value_44) ? (smallest) : (value_44));
                            smallest = _min_67;
                            int position_45 = block_1 * 128 + 64 + producer_warp * 4;
                            int _min_68 = ((position_45) < (topk - 1) ? (position_45) : (topk - 1));
                            int clamped_46 = _min_68;
                            int value_47 = indices[row_base + (long long)clamped_46];
                            if (position_45 >= active_0_1) {
                                value_47 = -1;
                            }
                            idx[16] = value_47;
                            int _max_110 = ((largest) > (value_47) ? (largest) : (value_47));
                            largest = _max_110;
                            int _min_69 = ((smallest) < (value_47) ? (smallest) : (value_47));
                            smallest = _min_69;
                            int position_48 = block_1 * 128 + 64 + producer_warp * 4 + 1;
                            int _min_70 = ((position_48) < (topk - 1) ? (position_48) : (topk - 1));
                            int clamped_49 = _min_70;
                            int value_50 = indices[row_base + (long long)clamped_49];
                            if (position_48 >= active_0_1) {
                                value_50 = -1;
                            }
                            idx[17] = value_50;
                            int _max_111 = ((largest) > (value_50) ? (largest) : (value_50));
                            largest = _max_111;
                            int _min_71 = ((smallest) < (value_50) ? (smallest) : (value_50));
                            smallest = _min_71;
                            int position_51 = block_1 * 128 + 64 + producer_warp * 4 + 2;
                            int _min_72 = ((position_51) < (topk - 1) ? (position_51) : (topk - 1));
                            int clamped_52 = _min_72;
                            int value_53 = indices[row_base + (long long)clamped_52];
                            if (position_51 >= active_0_1) {
                                value_53 = -1;
                            }
                            idx[18] = value_53;
                            int _max_112 = ((largest) > (value_53) ? (largest) : (value_53));
                            largest = _max_112;
                            int _min_73 = ((smallest) < (value_53) ? (smallest) : (value_53));
                            smallest = _min_73;
                            int position_54 = block_1 * 128 + 64 + producer_warp * 4 + 3;
                            int _min_74 = ((position_54) < (topk - 1) ? (position_54) : (topk - 1));
                            int clamped_55 = _min_74;
                            int value_56 = indices[row_base + (long long)clamped_55];
                            if (position_54 >= active_0_1) {
                                value_56 = -1;
                            }
                            idx[19] = value_56;
                            int _max_113 = ((largest) > (value_56) ? (largest) : (value_56));
                            largest = _max_113;
                            int _min_75 = ((smallest) < (value_56) ? (smallest) : (value_56));
                            smallest = _min_75;
                            int position_57 = block_1 * 128 + 80 + producer_warp * 4;
                            int _min_76 = ((position_57) < (topk - 1) ? (position_57) : (topk - 1));
                            int clamped_58 = _min_76;
                            int value_59 = indices[row_base + (long long)clamped_58];
                            if (position_57 >= active_0_1) {
                                value_59 = -1;
                            }
                            idx[20] = value_59;
                            int _max_114 = ((largest) > (value_59) ? (largest) : (value_59));
                            largest = _max_114;
                            int _min_77 = ((smallest) < (value_59) ? (smallest) : (value_59));
                            smallest = _min_77;
                            int position_60 = block_1 * 128 + 80 + producer_warp * 4 + 1;
                            int _min_78 = ((position_60) < (topk - 1) ? (position_60) : (topk - 1));
                            int clamped_61 = _min_78;
                            int value_62 = indices[row_base + (long long)clamped_61];
                            if (position_60 >= active_0_1) {
                                value_62 = -1;
                            }
                            idx[21] = value_62;
                            int _max_115 = ((largest) > (value_62) ? (largest) : (value_62));
                            largest = _max_115;
                            int _min_79 = ((smallest) < (value_62) ? (smallest) : (value_62));
                            smallest = _min_79;
                            int position_63 = block_1 * 128 + 80 + producer_warp * 4 + 2;
                            int _min_80 = ((position_63) < (topk - 1) ? (position_63) : (topk - 1));
                            int clamped_64 = _min_80;
                            int value_65 = indices[row_base + (long long)clamped_64];
                            if (position_63 >= active_0_1) {
                                value_65 = -1;
                            }
                            idx[22] = value_65;
                            int _max_116 = ((largest) > (value_65) ? (largest) : (value_65));
                            largest = _max_116;
                            int _min_81 = ((smallest) < (value_65) ? (smallest) : (value_65));
                            smallest = _min_81;
                            int position_66 = block_1 * 128 + 80 + producer_warp * 4 + 3;
                            int _min_82 = ((position_66) < (topk - 1) ? (position_66) : (topk - 1));
                            int clamped_67 = _min_82;
                            int value_68 = indices[row_base + (long long)clamped_67];
                            if (position_66 >= active_0_1) {
                                value_68 = -1;
                            }
                            idx[23] = value_68;
                            int _max_117 = ((largest) > (value_68) ? (largest) : (value_68));
                            largest = _max_117;
                            int _min_83 = ((smallest) < (value_68) ? (smallest) : (value_68));
                            smallest = _min_83;
                            int position_69 = block_1 * 128 + 96 + producer_warp * 4;
                            int _min_84 = ((position_69) < (topk - 1) ? (position_69) : (topk - 1));
                            int clamped_70 = _min_84;
                            int value_71 = indices[row_base + (long long)clamped_70];
                            if (position_69 >= active_0_1) {
                                value_71 = -1;
                            }
                            idx[24] = value_71;
                            int _max_118 = ((largest) > (value_71) ? (largest) : (value_71));
                            largest = _max_118;
                            int _min_85 = ((smallest) < (value_71) ? (smallest) : (value_71));
                            smallest = _min_85;
                            int position_72 = block_1 * 128 + 96 + producer_warp * 4 + 1;
                            int _min_86 = ((position_72) < (topk - 1) ? (position_72) : (topk - 1));
                            int clamped_73 = _min_86;
                            int value_74 = indices[row_base + (long long)clamped_73];
                            if (position_72 >= active_0_1) {
                                value_74 = -1;
                            }
                            idx[25] = value_74;
                            int _max_119 = ((largest) > (value_74) ? (largest) : (value_74));
                            largest = _max_119;
                            int _min_87 = ((smallest) < (value_74) ? (smallest) : (value_74));
                            smallest = _min_87;
                            int position_75 = block_1 * 128 + 96 + producer_warp * 4 + 2;
                            int _min_88 = ((position_75) < (topk - 1) ? (position_75) : (topk - 1));
                            int clamped_76 = _min_88;
                            int value_77 = indices[row_base + (long long)clamped_76];
                            if (position_75 >= active_0_1) {
                                value_77 = -1;
                            }
                            idx[26] = value_77;
                            int _max_120 = ((largest) > (value_77) ? (largest) : (value_77));
                            largest = _max_120;
                            int _min_89 = ((smallest) < (value_77) ? (smallest) : (value_77));
                            smallest = _min_89;
                            int position_78 = block_1 * 128 + 96 + producer_warp * 4 + 3;
                            int _min_90 = ((position_78) < (topk - 1) ? (position_78) : (topk - 1));
                            int clamped_79 = _min_90;
                            int value_80 = indices[row_base + (long long)clamped_79];
                            if (position_78 >= active_0_1) {
                                value_80 = -1;
                            }
                            idx[27] = value_80;
                            int _max_121 = ((largest) > (value_80) ? (largest) : (value_80));
                            largest = _max_121;
                            int _min_91 = ((smallest) < (value_80) ? (smallest) : (value_80));
                            smallest = _min_91;
                            int position_81 = block_1 * 128 + 112 + producer_warp * 4;
                            int _min_92 = ((position_81) < (topk - 1) ? (position_81) : (topk - 1));
                            int clamped_82 = _min_92;
                            int value_83 = indices[row_base + (long long)clamped_82];
                            if (position_81 >= active_0_1) {
                                value_83 = -1;
                            }
                            idx[28] = value_83;
                            int _max_122 = ((largest) > (value_83) ? (largest) : (value_83));
                            largest = _max_122;
                            int _min_93 = ((smallest) < (value_83) ? (smallest) : (value_83));
                            smallest = _min_93;
                            int position_84 = block_1 * 128 + 112 + producer_warp * 4 + 1;
                            int _min_94 = ((position_84) < (topk - 1) ? (position_84) : (topk - 1));
                            int clamped_85 = _min_94;
                            int value_86 = indices[row_base + (long long)clamped_85];
                            if (position_84 >= active_0_1) {
                                value_86 = -1;
                            }
                            idx[29] = value_86;
                            int _max_123 = ((largest) > (value_86) ? (largest) : (value_86));
                            largest = _max_123;
                            int _min_95 = ((smallest) < (value_86) ? (smallest) : (value_86));
                            smallest = _min_95;
                            int position_87 = block_1 * 128 + 112 + producer_warp * 4 + 2;
                            int _min_96 = ((position_87) < (topk - 1) ? (position_87) : (topk - 1));
                            int clamped_88 = _min_96;
                            int value_89 = indices[row_base + (long long)clamped_88];
                            if (position_87 >= active_0_1) {
                                value_89 = -1;
                            }
                            idx[30] = value_89;
                            int _max_124 = ((largest) > (value_89) ? (largest) : (value_89));
                            largest = _max_124;
                            int _min_97 = ((smallest) < (value_89) ? (smallest) : (value_89));
                            smallest = _min_97;
                            int position_90 = block_1 * 128 + 112 + producer_warp * 4 + 3;
                            int _min_98 = ((position_90) < (topk - 1) ? (position_90) : (topk - 1));
                            int clamped_91 = _min_98;
                            int value_92 = indices[row_base + (long long)clamped_91];
                            if (position_90 >= active_0_1) {
                                value_92 = -1;
                            }
                            idx[31] = value_92;
                            int _max_125 = ((largest) > (value_92) ? (largest) : (value_92));
                            largest = _max_125;
                            int _min_99 = ((smallest) < (value_92) ? (smallest) : (value_92));
                            smallest = _min_99;
                        }
                        int skip = (smallest == num_kv || largest == -1) && block_1 >= 2;
                        int gatom = tcur_1 * 8;
                        int patom = gatom % 12;
                        {
                            mbarrier_wait(sv_done_addr + (patom / 2) * 8, gatom / 12 & 1 ^ 1);
                        }
                        if (skip == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom * 16384) + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(0), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (patom) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom * 16384) + 2048 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(0), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (patom) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom * 16384) + 4096 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(0), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (patom) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom * 16384) + 6144 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(0), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (patom) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom * 16384) + 8192 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(0), "r"(idx[16]), "r"(idx[17]), "r"(idx[18]), "r"(idx[19]), "r"(kv_ready_addr + (patom) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom * 16384) + 10240 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(0), "r"(idx[20]), "r"(idx[21]), "r"(idx[22]), "r"(idx[23]), "r"(kv_ready_addr + (patom) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom * 16384) + 12288 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(0), "r"(idx[24]), "r"(idx[25]), "r"(idx[26]), "r"(idx[27]), "r"(kv_ready_addr + (patom) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom * 16384) + 14336 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(0), "r"(idx[28]), "r"(idx[29]), "r"(idx[30]), "r"(idx[31]), "r"(kv_ready_addr + (patom) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_addr + (patom) * 8), "r"((uint32_t)(4096)) : "memory");
                        }
                        int gatom_0 = tcur_1 * 8 + 1;
                        int patom_1 = gatom_0 % 12;
                        if (skip == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_1 * 16384) + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(64), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (patom_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_1 * 16384) + 2048 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(64), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (patom_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_1 * 16384) + 4096 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(64), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (patom_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_1 * 16384) + 6144 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(64), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (patom_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_1 * 16384) + 8192 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(64), "r"(idx[16]), "r"(idx[17]), "r"(idx[18]), "r"(idx[19]), "r"(kv_ready_addr + (patom_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_1 * 16384) + 10240 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(64), "r"(idx[20]), "r"(idx[21]), "r"(idx[22]), "r"(idx[23]), "r"(kv_ready_addr + (patom_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_1 * 16384) + 12288 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(64), "r"(idx[24]), "r"(idx[25]), "r"(idx[26]), "r"(idx[27]), "r"(kv_ready_addr + (patom_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_1 * 16384) + 14336 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(64), "r"(idx[28]), "r"(idx[29]), "r"(idx[30]), "r"(idx[31]), "r"(kv_ready_addr + (patom_1) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_addr + (patom_1) * 8), "r"((uint32_t)(4096)) : "memory");
                        }
                        int gatom_2 = tcur_1 * 8 + 2;
                        int patom_3 = gatom_2 % 12;
                        {
                            mbarrier_wait(sv_done_addr + (patom_3 / 2) * 8, gatom_2 / 12 & 1 ^ 1);
                        }
                        if (skip == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_3 * 16384) + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(128), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (patom_3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_3 * 16384) + 2048 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(128), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (patom_3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_3 * 16384) + 4096 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(128), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (patom_3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_3 * 16384) + 6144 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(128), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (patom_3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_3 * 16384) + 8192 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(128), "r"(idx[16]), "r"(idx[17]), "r"(idx[18]), "r"(idx[19]), "r"(kv_ready_addr + (patom_3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_3 * 16384) + 10240 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(128), "r"(idx[20]), "r"(idx[21]), "r"(idx[22]), "r"(idx[23]), "r"(kv_ready_addr + (patom_3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_3 * 16384) + 12288 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(128), "r"(idx[24]), "r"(idx[25]), "r"(idx[26]), "r"(idx[27]), "r"(kv_ready_addr + (patom_3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_3 * 16384) + 14336 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(128), "r"(idx[28]), "r"(idx[29]), "r"(idx[30]), "r"(idx[31]), "r"(kv_ready_addr + (patom_3) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_addr + (patom_3) * 8), "r"((uint32_t)(4096)) : "memory");
                        }
                        int gatom_4 = tcur_1 * 8 + 3;
                        int patom_5 = gatom_4 % 12;
                        if (skip == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_5 * 16384) + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(192), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (patom_5) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_5 * 16384) + 2048 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(192), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (patom_5) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_5 * 16384) + 4096 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(192), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (patom_5) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_5 * 16384) + 6144 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(192), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (patom_5) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_5 * 16384) + 8192 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(192), "r"(idx[16]), "r"(idx[17]), "r"(idx[18]), "r"(idx[19]), "r"(kv_ready_addr + (patom_5) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_5 * 16384) + 10240 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(192), "r"(idx[20]), "r"(idx[21]), "r"(idx[22]), "r"(idx[23]), "r"(kv_ready_addr + (patom_5) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_5 * 16384) + 12288 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(192), "r"(idx[24]), "r"(idx[25]), "r"(idx[26]), "r"(idx[27]), "r"(kv_ready_addr + (patom_5) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_5 * 16384) + 14336 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(192), "r"(idx[28]), "r"(idx[29]), "r"(idx[30]), "r"(idx[31]), "r"(kv_ready_addr + (patom_5) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_addr + (patom_5) * 8), "r"((uint32_t)(4096)) : "memory");
                        }
                        int gatom_6 = tcur_1 * 8 + 4;
                        int patom_7 = gatom_6 % 12;
                        {
                            mbarrier_wait(sv_done_addr + (patom_7 / 2) * 8, gatom_6 / 12 & 1 ^ 1);
                        }
                        if (skip == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_7 * 16384) + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(256), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (patom_7) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_7 * 16384) + 2048 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(256), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (patom_7) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_7 * 16384) + 4096 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(256), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (patom_7) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_7 * 16384) + 6144 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(256), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (patom_7) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_7 * 16384) + 8192 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(256), "r"(idx[16]), "r"(idx[17]), "r"(idx[18]), "r"(idx[19]), "r"(kv_ready_addr + (patom_7) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_7 * 16384) + 10240 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(256), "r"(idx[20]), "r"(idx[21]), "r"(idx[22]), "r"(idx[23]), "r"(kv_ready_addr + (patom_7) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_7 * 16384) + 12288 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(256), "r"(idx[24]), "r"(idx[25]), "r"(idx[26]), "r"(idx[27]), "r"(kv_ready_addr + (patom_7) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_7 * 16384) + 14336 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(256), "r"(idx[28]), "r"(idx[29]), "r"(idx[30]), "r"(idx[31]), "r"(kv_ready_addr + (patom_7) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_addr + (patom_7) * 8), "r"((uint32_t)(4096)) : "memory");
                        }
                        int gatom_8 = tcur_1 * 8 + 5;
                        int patom_9 = gatom_8 % 12;
                        if (skip == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_9 * 16384) + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(320), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (patom_9) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_9 * 16384) + 2048 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(320), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (patom_9) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_9 * 16384) + 4096 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(320), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (patom_9) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_9 * 16384) + 6144 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(320), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (patom_9) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_9 * 16384) + 8192 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(320), "r"(idx[16]), "r"(idx[17]), "r"(idx[18]), "r"(idx[19]), "r"(kv_ready_addr + (patom_9) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_9 * 16384) + 10240 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(320), "r"(idx[20]), "r"(idx[21]), "r"(idx[22]), "r"(idx[23]), "r"(kv_ready_addr + (patom_9) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_9 * 16384) + 12288 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(320), "r"(idx[24]), "r"(idx[25]), "r"(idx[26]), "r"(idx[27]), "r"(kv_ready_addr + (patom_9) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_9 * 16384) + 14336 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(320), "r"(idx[28]), "r"(idx[29]), "r"(idx[30]), "r"(idx[31]), "r"(kv_ready_addr + (patom_9) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_addr + (patom_9) * 8), "r"((uint32_t)(4096)) : "memory");
                        }
                        int gatom_10 = tcur_1 * 8 + 6;
                        int patom_11 = gatom_10 % 12;
                        {
                            mbarrier_wait(sv_done_addr + (patom_11 / 2) * 8, gatom_10 / 12 & 1 ^ 1);
                        }
                        if (skip == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_11 * 16384) + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(384), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (patom_11) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_11 * 16384) + 2048 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(384), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (patom_11) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_11 * 16384) + 4096 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(384), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (patom_11) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_11 * 16384) + 6144 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(384), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (patom_11) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_11 * 16384) + 8192 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(384), "r"(idx[16]), "r"(idx[17]), "r"(idx[18]), "r"(idx[19]), "r"(kv_ready_addr + (patom_11) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_11 * 16384) + 10240 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(384), "r"(idx[20]), "r"(idx[21]), "r"(idx[22]), "r"(idx[23]), "r"(kv_ready_addr + (patom_11) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_11 * 16384) + 12288 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(384), "r"(idx[24]), "r"(idx[25]), "r"(idx[26]), "r"(idx[27]), "r"(kv_ready_addr + (patom_11) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_11 * 16384) + 14336 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(384), "r"(idx[28]), "r"(idx[29]), "r"(idx[30]), "r"(idx[31]), "r"(kv_ready_addr + (patom_11) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_addr + (patom_11) * 8), "r"((uint32_t)(4096)) : "memory");
                        }
                        int gatom_12 = tcur_1 * 8 + 7;
                        int patom_13 = gatom_12 % 12;
                        if (skip == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_13 * 16384) + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(448), "r"(idx[0]), "r"(idx[1]), "r"(idx[2]), "r"(idx[3]), "r"(kv_ready_addr + (patom_13) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_13 * 16384) + 2048 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(448), "r"(idx[4]), "r"(idx[5]), "r"(idx[6]), "r"(idx[7]), "r"(kv_ready_addr + (patom_13) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_13 * 16384) + 4096 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(448), "r"(idx[8]), "r"(idx[9]), "r"(idx[10]), "r"(idx[11]), "r"(kv_ready_addr + (patom_13) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_13 * 16384) + 6144 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(448), "r"(idx[12]), "r"(idx[13]), "r"(idx[14]), "r"(idx[15]), "r"(kv_ready_addr + (patom_13) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_13 * 16384) + 8192 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(448), "r"(idx[16]), "r"(idx[17]), "r"(idx[18]), "r"(idx[19]), "r"(kv_ready_addr + (patom_13) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_13 * 16384) + 10240 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(448), "r"(idx[20]), "r"(idx[21]), "r"(idx[22]), "r"(idx[23]), "r"(kv_ready_addr + (patom_13) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_13 * 16384) + 12288 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(448), "r"(idx[24]), "r"(idx[25]), "r"(idx[26]), "r"(idx[27]), "r"(kv_ready_addr + (patom_13) * 8), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
                                ".mbarrier::complete_tx::bytes.L2::cache_hint ""[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                :: "r"(ka_addr + (unsigned int)(patom_13 * 16384) + 14336 + (unsigned int)(producer_warp * 512)), "l"((&kv_latent)), "r"(448), "r"(idx[28]), "r"(idx[29]), "r"(idx[30]), "r"(idx[31]), "r"(kv_ready_addr + (patom_13) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        } else {
                            asm volatile("mbarrier.complete_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                                :: "r"(kv_ready_addr + (patom_13) * 8), "r"((uint32_t)(4096)) : "memory");
                        }
                    }
                    tcur_1 = tcur_1 + 1;
                }
                __syncwarp();
                mbarrier_wait(clc_full_addr, outer_phase_1);
                uint32_t _clc_valid_1 = 0;
                uint32_t _clc_ctaid_x_1;
                uint32_t _clc_ctaid_y_1;
                uint32_t _clc_ctaid_z_1;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_1), "=r"(_clc_ctaid_y_1), "=r"(_clc_ctaid_z_1), "=r"(_clc_valid_1)
                    : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                    : "memory");
                mbarrier_arrive(clc_empty_addr);
                if (_clc_valid_1 == 0) {
                    break;
                }
                query_1 = _clc_ctaid_x_1;
                outer_phase_1 = outer_phase_1 ^ 1;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            int query_2 = blockIdx.x;
            unsigned int outer_phase_2 = 0;
            int tcur_2 = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int outer_2 = 0; outer_2 < num_queries; outer_2++) {
                    int active_2 = topk;
                    if (derive_length != 0) {
                        int len_slot_2 = outer_2 & 1;
                        unsigned int len_phase_2 = outer_2 >> 1 & 1;
                        mbarrier_wait(len_ready_addr + (len_slot_2) * 8, len_phase_2);
                        int _max_126 = ((len_cells[len_slot_2]) > (0) ? (len_cells[len_slot_2]) : (0));
                        int _min_100 = ((_max_126) < (topk) ? (_max_126) : (topk));
                        active_2 = _min_100;
                    } else if (has_topk_length != 0) {
                        int _max_127 = ((topk_length[query_2]) > (0) ? (topk_length[query_2]) : (0));
                        int _min_101 = ((_max_127) < (topk) ? (_max_127) : (topk));
                        active_2 = _min_101;
                    }
                    int active_0_2 = active_2;
                    int _max_128 = (((active_0_2 + 128 - 1) / 128) > (1) ? ((active_0_2 + 128 - 1) / 128) : (1));
                    int blocks_2 = _max_128;
                    mbarrier_wait(q_free_addr, outer_phase_2 ^ 1);
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qt_addr), "l"((&q_latent)), "r"(0), "r"(0), "r"(0), "r"(query_2),
                           "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qt_addr + 2048), "l"((&q_latent)), "r"(0), "r"(0), "r"(1), "r"(query_2),
                           "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qt_addr + 4096), "l"((&q_latent)), "r"(0), "r"(0), "r"(2), "r"(query_2),
                           "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qt_addr + 6144), "l"((&q_latent)), "r"(0), "r"(0), "r"(3), "r"(query_2),
                           "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qt_addr + 8192), "l"((&q_latent)), "r"(0), "r"(0), "r"(4), "r"(query_2),
                           "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qt_addr + 10240), "l"((&q_latent)), "r"(0), "r"(0), "r"(5), "r"(query_2),
                           "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qt_addr + 12288), "l"((&q_latent)), "r"(0), "r"(0), "r"(6), "r"(query_2),
                           "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(qt_addr + 14336), "l"((&q_latent)), "r"(0), "r"(0), "r"(7), "r"(query_2),
                           "r"(q_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(q_full_addr, 16384);
                    mbarrier_wait(q_full_addr, outer_phase_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int block_2 = 0; block_2 < blocks_2 + 1; block_2++) {
                        int buf_1 = tcur_2 & 1;
                        if (blocks_2 > block_2) {
                            mbarrier_wait(p_free_addr + (buf_1) * 8, tcur_2 >> 1 & 1 ^ 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int gatom_1 = tcur_2 * 8;
                            int patom_2 = gatom_1 % 12;
                            mbarrier_arrive_expect_tx(kv_ready_addr + (patom_2) * 8, 16384);
                            mbarrier_wait(kv_ready_addr + (patom_2) * 8, gatom_1 / 12 & 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0 = (((ka_addr) >> 4) & 0x3FFF) + (patom_2) * 1024;
                            int _mma_b_lo_0 = (((qt_addr) >> 4) & 0x3FFF) + (0) * 1024;
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
                    "mov.b32 id, 134481040;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem + (buf_1 * 16))), "r"(0));
                            int gatom_0_1 = tcur_2 * 8 + 1;
                            int patom_1_1 = gatom_0_1 % 12;
                            mbarrier_arrive_expect_tx(kv_ready_addr + (patom_1_1) * 8, 16384);
                            mbarrier_wait(kv_ready_addr + (patom_1_1) * 8, gatom_0_1 / 12 & 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_1 = (((ka_addr) >> 4) & 0x3FFF) + (patom_1_1) * 1024;
                            int _mma_b_lo_1 = (((qt_addr + 2048) >> 4) & 0x3FFF) + (0) * 1024;
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
                    "mov.b32 id, 134481040;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem + (buf_1 * 16))), "r"(1));
                            int gatom_2_1 = tcur_2 * 8 + 2;
                            int patom_3_1 = gatom_2_1 % 12;
                            mbarrier_arrive_expect_tx(kv_ready_addr + (patom_3_1) * 8, 16384);
                            mbarrier_wait(kv_ready_addr + (patom_3_1) * 8, gatom_2_1 / 12 & 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_2 = (((ka_addr) >> 4) & 0x3FFF) + (patom_3_1) * 1024;
                            int _mma_b_lo_2 = (((qt_addr + 4096) >> 4) & 0x3FFF) + (0) * 1024;
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
                    "mov.b32 id, 134481040;\n\t"
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem + (buf_1 * 16))), "r"(1));
                            int gatom_4_1 = tcur_2 * 8 + 3;
                            int patom_5_1 = gatom_4_1 % 12;
                            mbarrier_arrive_expect_tx(kv_ready_addr + (patom_5_1) * 8, 16384);
                            mbarrier_wait(kv_ready_addr + (patom_5_1) * 8, gatom_4_1 / 12 & 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_3 = (((ka_addr) >> 4) & 0x3FFF) + (patom_5_1) * 1024;
                            int _mma_b_lo_3 = (((qt_addr + 6144) >> 4) & 0x3FFF) + (0) * 1024;
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
                    "mov.b32 id, 134481040;\n\t"
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_tmem + (buf_1 * 16))), "r"(1));
                        }
                        if (block_2 > 0) {
                            mbarrier_wait(so_ready_addr, tcur_2 - 1 & 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_4 = ((((vt_addr) >> 4) & 0x3FFF) | 0x4000000) + (4 * (tcur_2 - 1) % 6 % 6) * 2048;
                            int _mma_b_lo_4 = (((pt_addr) >> 4) & 0x3FFF) | 0x800000;
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
                    "mov.b32 id, 134513808;\n\t"
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
                    "add.u32 blo, blo, 122;\n\t"
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_tmem + (32))), "r"(((block_2 == 1) ? 0 : 1)));
                            tcgen05_commit(sv_done_addr + (4 * (tcur_2 - 1) % 6 % 6) * 8);
                            int _mma_a_lo_5 = ((((vt_addr) >> 4) & 0x3FFF) | 0x4000000) + ((4 * (tcur_2 - 1) % 6 + 1) % 6) * 2048;
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
                    "mov.b32 id, 134513808;\n\t"
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
                    "add.u32 blo, blo, 122;\n\t"
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_4), "r"((tmem_tmem + (48))), "r"(((block_2 == 1) ? 0 : 1)));
                            tcgen05_commit(sv_done_addr + ((4 * (tcur_2 - 1) % 6 + 1) % 6) * 8);
                            int _mma_a_lo_6 = ((((vt_addr) >> 4) & 0x3FFF) | 0x4000000) + ((4 * (tcur_2 - 1) % 6 + 2) % 6) * 2048;
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
                    "mov.b32 id, 134513808;\n\t"
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
                    "add.u32 blo, blo, 122;\n\t"
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
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_4), "r"((tmem_tmem + (64))), "r"(((block_2 == 1) ? 0 : 1)));
                            tcgen05_commit(sv_done_addr + ((4 * (tcur_2 - 1) % 6 + 2) % 6) * 8);
                            int _mma_a_lo_7 = ((((vt_addr) >> 4) & 0x3FFF) | 0x4000000) + ((4 * (tcur_2 - 1) % 6 + 3) % 6) * 2048;
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
                    "mov.b32 id, 134513808;\n\t"
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
                    "add.u32 blo, blo, 122;\n\t"
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
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_4), "r"((tmem_tmem + (80))), "r"(((block_2 == 1) ? 0 : 1)));
                            tcgen05_commit(sv_done_addr + ((4 * (tcur_2 - 1) % 6 + 3) % 6) * 8);
                        }
                        if (blocks_2 > block_2) {
                            int gatom2 = tcur_2 * 8 + 4;
                            int patom2 = gatom2 % 12;
                            mbarrier_arrive_expect_tx(kv_ready_addr + (patom2) * 8, 16384);
                            mbarrier_wait(kv_ready_addr + (patom2) * 8, gatom2 / 12 & 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_8 = (((ka_addr) >> 4) & 0x3FFF) + (patom2) * 1024;
                            int _mma_b_lo_8 = (((qt_addr + 8192) >> 4) & 0x3FFF) + (0) * 1024;
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
                    "mov.b32 id, 134481040;\n\t"
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
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"((tmem_tmem + (buf_1 * 16))), "r"(1));
                            int gatom2_0 = tcur_2 * 8 + 5;
                            int patom2_1 = gatom2_0 % 12;
                            mbarrier_arrive_expect_tx(kv_ready_addr + (patom2_1) * 8, 16384);
                            mbarrier_wait(kv_ready_addr + (patom2_1) * 8, gatom2_0 / 12 & 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_9 = (((ka_addr) >> 4) & 0x3FFF) + (patom2_1) * 1024;
                            int _mma_b_lo_9 = (((qt_addr + 10240) >> 4) & 0x3FFF) + (0) * 1024;
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
                    "mov.b32 id, 134481040;\n\t"
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
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"((tmem_tmem + (buf_1 * 16))), "r"(1));
                            int gatom2_2 = tcur_2 * 8 + 6;
                            int patom2_3 = gatom2_2 % 12;
                            mbarrier_arrive_expect_tx(kv_ready_addr + (patom2_3) * 8, 16384);
                            mbarrier_wait(kv_ready_addr + (patom2_3) * 8, gatom2_2 / 12 & 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_10 = (((ka_addr) >> 4) & 0x3FFF) + (patom2_3) * 1024;
                            int _mma_b_lo_10 = (((qt_addr + 12288) >> 4) & 0x3FFF) + (0) * 1024;
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
                    "mov.b32 id, 134481040;\n\t"
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
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"((tmem_tmem + (buf_1 * 16))), "r"(1));
                            int gatom2_4 = tcur_2 * 8 + 7;
                            int patom2_5 = gatom2_4 % 12;
                            mbarrier_arrive_expect_tx(kv_ready_addr + (patom2_5) * 8, 16384);
                            mbarrier_wait(kv_ready_addr + (patom2_5) * 8, gatom2_4 / 12 & 1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_11 = (((ka_addr) >> 4) & 0x3FFF) + (patom2_5) * 1024;
                            int _mma_b_lo_11 = (((qt_addr + 14336) >> 4) & 0x3FFF) + (0) * 1024;
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
                    "mov.b32 id, 134481040;\n\t"
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
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"((tmem_tmem + (buf_1 * 16))), "r"(1));
                            tcgen05_commit(qk_done_addr + (buf_1) * 8);
                        }
                        tcur_2 = tcur_2 + 1;
                    }
                    tcur_2 = tcur_2 - 1;
                    mbarrier_wait(clc_full_addr, outer_phase_2);
                    uint32_t _clc_valid_2 = 0;
                    uint32_t _clc_ctaid_x_2;
                    uint32_t _clc_ctaid_y_2;
                    uint32_t _clc_ctaid_z_2;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_x_2), "=r"(_clc_ctaid_y_2), "=r"(_clc_ctaid_z_2), "=r"(_clc_valid_2)
                        : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                        : "memory");
                    mbarrier_arrive(clc_empty_addr);
                    if (_clc_valid_2 == 0) {
                        break;
                    }
                    query_2 = _clc_ctaid_x_2;
                    outer_phase_2 = outer_phase_2 ^ 1;
                }
            }
        }
    }
    // ---- Role: metadata ----
    if (warp == 9) {
        { // metadata_main
            if (lane < 8) {
                int query_3 = blockIdx.x;
                unsigned int outer_phase_3 = 0;
                int tcur_3 = 0;
                #pragma unroll 1
                for (int outer_3 = 0; outer_3 < num_queries; outer_3++) {
                    int active_3 = topk;
                    if (derive_length != 0) {
                        int len_slot_3 = outer_3 & 1;
                        unsigned int len_phase_3 = outer_3 >> 1 & 1;
                        mbarrier_wait(len_ready_addr + (len_slot_3) * 8, len_phase_3);
                        int _max_132 = ((len_cells[len_slot_3]) > (0) ? (len_cells[len_slot_3]) : (0));
                        int _min_104 = ((_max_132) < (topk) ? (_max_132) : (topk));
                        active_3 = _min_104;
                    } else if (has_topk_length != 0) {
                        int _max_133 = ((topk_length[query_3]) > (0) ? (topk_length[query_3]) : (0));
                        int _min_105 = ((_max_133) < (topk) ? (_max_133) : (topk));
                        active_3 = _min_105;
                    }
                    int active_0_3 = active_3;
                    int _max_134 = (((active_0_3 + 128 - 1) / 128) > (1) ? ((active_0_3 + 128 - 1) / 128) : (1));
                    int blocks_3 = _max_134;
                    long long row_base_1 = (long long)indices_offset + (long long)query_3 * (long long)idx_stride;
                    #pragma unroll 1
                    for (int block_3 = 0; block_3 < blocks_3; block_3++) {
                        unsigned int mask = 0;
                        int position_1 = block_3 * 128 + lane * 16;
                        int whole_tile_1 = active_0_3 >= block_3 * 128 + 128 && ((idx_stride | indices_offset) & 7) == 0;
                        if (whole_tile_1 != 0) {
                            int _vec_load_8[8];
                            {
                                uint32_t _iv_0_0;
                                uint32_t _iv_0_1;
                                uint32_t _iv_0_2;
                                uint32_t _iv_0_3;
                                uint32_t _iv_0_4;
                                uint32_t _iv_0_5;
                                uint32_t _iv_0_6;
                                uint32_t _iv_0_7;
                                asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(_iv_0_0), "=r"(_iv_0_1), "=r"(_iv_0_2), "=r"(_iv_0_3), "=r"(_iv_0_4), "=r"(_iv_0_5), "=r"(_iv_0_6), "=r"(_iv_0_7) : "l"((const void*)(indices + (row_base_1 + (long long)position_1) + (0))) : "memory");
                                _vec_load_8[0 + 0] = (int32_t)_iv_0_0;
                                _vec_load_8[0 + 1] = (int32_t)_iv_0_1;
                                _vec_load_8[0 + 2] = (int32_t)_iv_0_2;
                                _vec_load_8[0 + 3] = (int32_t)_iv_0_3;
                                _vec_load_8[0 + 4] = (int32_t)_iv_0_4;
                                _vec_load_8[0 + 5] = (int32_t)_iv_0_5;
                                _vec_load_8[0 + 6] = (int32_t)_iv_0_6;
                                _vec_load_8[0 + 7] = (int32_t)_iv_0_7;
                            }
                            if (_vec_load_8[0] >= 0 && _vec_load_8[0] < num_kv) {
                                mask = mask | 1;
                            }
                            if (_vec_load_8[1] >= 0 && _vec_load_8[1] < num_kv) {
                                mask = mask | 2;
                            }
                            if (_vec_load_8[2] >= 0 && _vec_load_8[2] < num_kv) {
                                mask = mask | 4;
                            }
                            if (_vec_load_8[3] >= 0 && _vec_load_8[3] < num_kv) {
                                mask = mask | 8;
                            }
                            if (_vec_load_8[4] >= 0 && _vec_load_8[4] < num_kv) {
                                mask = mask | 16;
                            }
                            if (_vec_load_8[5] >= 0 && _vec_load_8[5] < num_kv) {
                                mask = mask | 32;
                            }
                            if (_vec_load_8[6] >= 0 && _vec_load_8[6] < num_kv) {
                                mask = mask | 64;
                            }
                            if (_vec_load_8[7] >= 0 && _vec_load_8[7] < num_kv) {
                                mask = mask | 128;
                            }
                            int _vec_load_9[8];
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
                                    : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(indices + (row_base_1 + (long long)position_1 + 8) + (0))) : "memory");
                                _vec_load_9[0 + 0] = (int32_t)_iv_1_0;
                                _vec_load_9[0 + 1] = (int32_t)_iv_1_1;
                                _vec_load_9[0 + 2] = (int32_t)_iv_1_2;
                                _vec_load_9[0 + 3] = (int32_t)_iv_1_3;
                                _vec_load_9[0 + 4] = (int32_t)_iv_1_4;
                                _vec_load_9[0 + 5] = (int32_t)_iv_1_5;
                                _vec_load_9[0 + 6] = (int32_t)_iv_1_6;
                                _vec_load_9[0 + 7] = (int32_t)_iv_1_7;
                            }
                            if (_vec_load_9[0] >= 0 && _vec_load_9[0] < num_kv) {
                                mask = mask | 256;
                            }
                            if (_vec_load_9[1] >= 0 && _vec_load_9[1] < num_kv) {
                                mask = mask | 512;
                            }
                            if (_vec_load_9[2] >= 0 && _vec_load_9[2] < num_kv) {
                                mask = mask | 1024;
                            }
                            if (_vec_load_9[3] >= 0 && _vec_load_9[3] < num_kv) {
                                mask = mask | 2048;
                            }
                            if (_vec_load_9[4] >= 0 && _vec_load_9[4] < num_kv) {
                                mask = mask | 4096;
                            }
                            if (_vec_load_9[5] >= 0 && _vec_load_9[5] < num_kv) {
                                mask = mask | 8192;
                            }
                            if (_vec_load_9[6] >= 0 && _vec_load_9[6] < num_kv) {
                                mask = mask | 16384;
                            }
                            if (_vec_load_9[7] >= 0 && _vec_load_9[7] < num_kv) {
                                mask = mask | 32768;
                            }
                        } else {
                            int _min_106 = ((position_1) < (topk - 1) ? (position_1) : (topk - 1));
                            int clamped_2 = _min_106;
                            int value_1 = indices[row_base_1 + (long long)clamped_2];
                            if (value_1 >= 0 && value_1 < num_kv && active_0_3 > position_1) {
                                mask = mask | 1;
                            }
                            int _min_107 = ((position_1 + 1) < (topk - 1) ? (position_1 + 1) : (topk - 1));
                            int clamped_0 = _min_107;
                            int value_1_1 = indices[row_base_1 + (long long)clamped_0];
                            if (value_1_1 >= 0 && value_1_1 < num_kv && active_0_3 > position_1 + 1) {
                                mask = mask | 2;
                            }
                            int _min_108 = ((position_1 + 2) < (topk - 1) ? (position_1 + 2) : (topk - 1));
                            int clamped_2_1 = _min_108;
                            int value_3 = indices[row_base_1 + (long long)clamped_2_1];
                            if (value_3 >= 0 && value_3 < num_kv && active_0_3 > position_1 + 2) {
                                mask = mask | 4;
                            }
                            int _min_109 = ((position_1 + 3) < (topk - 1) ? (position_1 + 3) : (topk - 1));
                            int clamped_4_1 = _min_109;
                            int value_5_1 = indices[row_base_1 + (long long)clamped_4_1];
                            if (value_5_1 >= 0 && value_5_1 < num_kv && active_0_3 > position_1 + 3) {
                                mask = mask | 8;
                            }
                            int _min_110 = ((position_1 + 4) < (topk - 1) ? (position_1 + 4) : (topk - 1));
                            int clamped_6 = _min_110;
                            int value_7 = indices[row_base_1 + (long long)clamped_6];
                            if (value_7 >= 0 && value_7 < num_kv && active_0_3 > position_1 + 4) {
                                mask = mask | 16;
                            }
                            int _min_111 = ((position_1 + 5) < (topk - 1) ? (position_1 + 5) : (topk - 1));
                            int clamped_8 = _min_111;
                            int value_9 = indices[row_base_1 + (long long)clamped_8];
                            if (value_9 >= 0 && value_9 < num_kv && active_0_3 > position_1 + 5) {
                                mask = mask | 32;
                            }
                            int _min_112 = ((position_1 + 6) < (topk - 1) ? (position_1 + 6) : (topk - 1));
                            int clamped_10_1 = _min_112;
                            int value_11_1 = indices[row_base_1 + (long long)clamped_10_1];
                            if (value_11_1 >= 0 && value_11_1 < num_kv && active_0_3 > position_1 + 6) {
                                mask = mask | 64;
                            }
                            int _min_113 = ((position_1 + 7) < (topk - 1) ? (position_1 + 7) : (topk - 1));
                            int clamped_12 = _min_113;
                            int value_13 = indices[row_base_1 + (long long)clamped_12];
                            if (value_13 >= 0 && value_13 < num_kv && active_0_3 > position_1 + 7) {
                                mask = mask | 128;
                            }
                            int _min_114 = ((position_1 + 8) < (topk - 1) ? (position_1 + 8) : (topk - 1));
                            int clamped_14 = _min_114;
                            int value_15 = indices[row_base_1 + (long long)clamped_14];
                            if (value_15 >= 0 && value_15 < num_kv && active_0_3 > position_1 + 8) {
                                mask = mask | 256;
                            }
                            int _min_115 = ((position_1 + 9) < (topk - 1) ? (position_1 + 9) : (topk - 1));
                            int clamped_16_1 = _min_115;
                            int value_17_1 = indices[row_base_1 + (long long)clamped_16_1];
                            if (value_17_1 >= 0 && value_17_1 < num_kv && active_0_3 > position_1 + 9) {
                                mask = mask | 512;
                            }
                            int _min_116 = ((position_1 + 10) < (topk - 1) ? (position_1 + 10) : (topk - 1));
                            int clamped_18 = _min_116;
                            int value_19 = indices[row_base_1 + (long long)clamped_18];
                            if (value_19 >= 0 && value_19 < num_kv && active_0_3 > position_1 + 10) {
                                mask = mask | 1024;
                            }
                            int _min_117 = ((position_1 + 11) < (topk - 1) ? (position_1 + 11) : (topk - 1));
                            int clamped_20 = _min_117;
                            int value_21 = indices[row_base_1 + (long long)clamped_20];
                            if (value_21 >= 0 && value_21 < num_kv && active_0_3 > position_1 + 11) {
                                mask = mask | 2048;
                            }
                            int _min_118 = ((position_1 + 12) < (topk - 1) ? (position_1 + 12) : (topk - 1));
                            int clamped_22_1 = _min_118;
                            int value_23_1 = indices[row_base_1 + (long long)clamped_22_1];
                            if (value_23_1 >= 0 && value_23_1 < num_kv && active_0_3 > position_1 + 12) {
                                mask = mask | 4096;
                            }
                            int _min_119 = ((position_1 + 13) < (topk - 1) ? (position_1 + 13) : (topk - 1));
                            int clamped_24 = _min_119;
                            int value_25 = indices[row_base_1 + (long long)clamped_24];
                            if (value_25 >= 0 && value_25 < num_kv && active_0_3 > position_1 + 13) {
                                mask = mask | 8192;
                            }
                            int _min_120 = ((position_1 + 14) < (topk - 1) ? (position_1 + 14) : (topk - 1));
                            int clamped_26 = _min_120;
                            int value_27 = indices[row_base_1 + (long long)clamped_26];
                            if (value_27 >= 0 && value_27 < num_kv && active_0_3 > position_1 + 14) {
                                mask = mask | 16384;
                            }
                            int _min_121 = ((position_1 + 15) < (topk - 1) ? (position_1 + 15) : (topk - 1));
                            int clamped_28_1 = _min_121;
                            int value_29_1 = indices[row_base_1 + (long long)clamped_28_1];
                            if (value_29_1 >= 0 && value_29_1 < num_kv && active_0_3 > position_1 + 15) {
                                mask = mask | 32768;
                            }
                        }
                        int vstage_1 = tcur_3 % 3;
                        mbarrier_wait(valid_free_addr + (vstage_1) * 8, tcur_3 / 3 & 1 ^ 1);
                        validity[vstage_1 * 16 + lane * 2] = mask & 255;
                        validity[vstage_1 * 16 + lane * 2 + 1] = mask >> 8 & 255;
                        mbarrier_arrive(valid_ready_addr + (vstage_1) * 8);
                        tcur_3 = tcur_3 + 1;
                    }
                    mbarrier_wait(clc_full_addr, outer_phase_3);
                    uint32_t _clc_valid_4 = 0;
                    uint32_t _clc_ctaid_x_4;
                    uint32_t _clc_ctaid_y_4;
                    uint32_t _clc_ctaid_z_4;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_x_4), "=r"(_clc_ctaid_y_4), "=r"(_clc_ctaid_z_4), "=r"(_clc_valid_4)
                        : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                        : "memory");
                    mbarrier_arrive(clc_empty_addr);
                    if (_clc_valid_4 == 0) {
                        break;
                    }
                    query_3 = _clc_ctaid_x_4;
                    outer_phase_3 = outer_phase_3 ^ 1;
                }
            } else if (lane == 8) {
                unsigned int outer_phase_4 = 0;
                #pragma unroll 1
                for (int outer_4 = 0; outer_4 < num_queries; outer_4++) {
                    mbarrier_wait(clc_empty_addr, outer_phase_4 ^ 1);
                    asm volatile(
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(clc_payload_addr + 0 * 16 + 0 * 16), "r"(clc_full_addr + 0 * 8)
                        : "memory");
                    mbarrier_arrive_expect_tx(clc_full_addr, 16);
                    mbarrier_wait(clc_full_addr, outer_phase_4);
                    uint32_t _clc_valid_5 = 0;
                    uint32_t _clc_ctaid_x_5;
                    uint32_t _clc_ctaid_y_5;
                    uint32_t _clc_ctaid_z_5;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_x_5), "=r"(_clc_ctaid_y_5), "=r"(_clc_ctaid_z_5), "=r"(_clc_valid_5)
                        : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                        : "memory");
                    mbarrier_arrive(clc_empty_addr);
                    if (_clc_valid_5 == 0) {
                        break;
                    }
                    outer_phase_4 = outer_phase_4 ^ 1;
                }
            } else {
                if (lane >= 16) {
                    int query_4 = blockIdx.x;
                    unsigned int outer_phase_5 = 0;
                    int lane16 = lane - 16;
                    int scan_aligned = ((idx_stride | indices_offset) & 7) == 0;
                    #pragma unroll 1
                    for (int outer_5 = 0; outer_5 < num_queries; outer_5++) {
                        if (derive_length != 0) {
                            long long row_base_2 = (long long)indices_offset + (long long)query_4 * (long long)idx_stride;
                            int last = 0;
                            int n_full = topk / 128;
                            if (scan_aligned == 0) {
                                n_full = 0;
                            }
                            #pragma unroll 4
                            for (int g = 0; g < n_full; g++) {
                                int spos = g * 128 + lane16 * 8;
                                int _vec_load_10[8];
                                {
                                    uint32_t _iv_2_0;
                                    uint32_t _iv_2_1;
                                    uint32_t _iv_2_2;
                                    uint32_t _iv_2_3;
                                    uint32_t _iv_2_4;
                                    uint32_t _iv_2_5;
                                    uint32_t _iv_2_6;
                                    uint32_t _iv_2_7;
                                    asm volatile("ld.global.nc.L1::evict_first.L2::evict_normal.L2::256B.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(indices + (row_base_2 + (long long)spos) + (0))) : "memory");
                                    _vec_load_10[0 + 0] = (int32_t)_iv_2_0;
                                    _vec_load_10[0 + 1] = (int32_t)_iv_2_1;
                                    _vec_load_10[0 + 2] = (int32_t)_iv_2_2;
                                    _vec_load_10[0 + 3] = (int32_t)_iv_2_3;
                                    _vec_load_10[0 + 4] = (int32_t)_iv_2_4;
                                    _vec_load_10[0 + 5] = (int32_t)_iv_2_5;
                                    _vec_load_10[0 + 6] = (int32_t)_iv_2_6;
                                    _vec_load_10[0 + 7] = (int32_t)_iv_2_7;
                                }
                                if (_vec_load_10[0] >= 0 && _vec_load_10[0] < num_kv) {
                                    last = spos + 1;
                                }
                                if (_vec_load_10[1] >= 0 && _vec_load_10[1] < num_kv) {
                                    last = spos + 1 + 1;
                                }
                                if (_vec_load_10[2] >= 0 && _vec_load_10[2] < num_kv) {
                                    last = spos + 2 + 1;
                                }
                                if (_vec_load_10[3] >= 0 && _vec_load_10[3] < num_kv) {
                                    last = spos + 3 + 1;
                                }
                                if (_vec_load_10[4] >= 0 && _vec_load_10[4] < num_kv) {
                                    last = spos + 4 + 1;
                                }
                                if (_vec_load_10[5] >= 0 && _vec_load_10[5] < num_kv) {
                                    last = spos + 5 + 1;
                                }
                                if (_vec_load_10[6] >= 0 && _vec_load_10[6] < num_kv) {
                                    last = spos + 6 + 1;
                                }
                                if (_vec_load_10[7] >= 0 && _vec_load_10[7] < num_kv) {
                                    last = spos + 7 + 1;
                                }
                            }
                            int tail0 = n_full * 128 + lane16;
                            int n_tail = (topk - n_full * 128 - lane16 + 15) / 16;
                            #pragma unroll 1
                            for (int t = 0; t < n_tail; t++) {
                                int pos = tail0 + t * 16;
                                int value_4 = indices[row_base_2 + (long long)pos];
                                if (value_4 >= 0 && value_4 < num_kv) {
                                    last = pos + 1;
                                }
                            }
                            int part = last;
                            int _shfl_xor_24 = __shfl_xor_sync(4294901760u, part, 8);
                            int peer = _shfl_xor_24;
                            int _max_135 = ((part) > (peer) ? (part) : (peer));
                            part = _max_135;
                            int _shfl_xor_25 = __shfl_xor_sync(4294901760u, part, 4);
                            int peer_0 = _shfl_xor_25;
                            int _max_136 = ((part) > (peer_0) ? (part) : (peer_0));
                            part = _max_136;
                            int _shfl_xor_26 = __shfl_xor_sync(4294901760u, part, 2);
                            int peer_1 = _shfl_xor_26;
                            int _max_137 = ((part) > (peer_1) ? (part) : (peer_1));
                            part = _max_137;
                            int _shfl_xor_27 = __shfl_xor_sync(4294901760u, part, 1);
                            int peer_2 = _shfl_xor_27;
                            int _max_138 = ((part) > (peer_2) ? (part) : (peer_2));
                            part = _max_138;
                            if (lane == 16) {
                                len_cells[outer_5 & 1] = part;
                                topk_length[(long long)query_4] = part;
                                mbarrier_arrive(len_ready_addr + (outer_5 & 1) * 8);
                            }
                        }
                        mbarrier_wait(clc_full_addr, outer_phase_5);
                        uint32_t _clc_valid_6 = 0;
                        uint32_t _clc_ctaid_x_6;
                        uint32_t _clc_ctaid_y_6;
                        uint32_t _clc_ctaid_z_6;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %3, 1, 0, p1;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_x_6), "=r"(_clc_ctaid_y_6), "=r"(_clc_ctaid_z_6), "=r"(_clc_valid_6)
                            : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                            : "memory");
                        mbarrier_arrive(clc_empty_addr);
                        if (_clc_valid_6 == 0) {
                            break;
                        }
                        query_4 = _clc_ctaid_x_6;
                        outer_phase_5 = outer_phase_5 ^ 1;
                    }
                }
            }
        }
    }
    // ---- Role: rope_load ----
    if (warp >= 10 && warp <= 11) {
        { // rope_load_main
            int query_5 = blockIdx.x;
            unsigned int outer_phase_6 = 0;
            int tcur_4 = 0;
            int thread = (warp - 10) * 32 + lane;
            int rrow = thread / 8;
            int rcol = thread % 8 * 8;
            #pragma unroll 1
            for (int outer_6 = 0; outer_6 < num_queries; outer_6++) {
                int active_4 = topk;
                if (derive_length != 0) {
                    int len_slot_4 = outer_6 & 1;
                    unsigned int len_phase_4 = outer_6 >> 1 & 1;
                    mbarrier_wait(len_ready_addr + (len_slot_4) * 8, len_phase_4);
                    int _max_129 = ((len_cells[len_slot_4]) > (0) ? (len_cells[len_slot_4]) : (0));
                    int _min_102 = ((_max_129) < (topk) ? (_max_129) : (topk));
                    active_4 = _min_102;
                } else if (has_topk_length != 0) {
                    int _max_130 = ((topk_length[query_5]) > (0) ? (topk_length[query_5]) : (0));
                    int _min_103 = ((_max_130) < (topk) ? (_max_130) : (topk));
                    active_4 = _min_103;
                }
                int active_0_4 = active_4;
                int _max_131 = (((active_0_4 + 128 - 1) / 128) > (1) ? ((active_0_4 + 128 - 1) / 128) : (1));
                int blocks_4 = _max_131;
                long long row_base_3 = (long long)indices_offset + (long long)query_5 * (long long)idx_stride;
                mbarrier_wait(clc_full_addr, outer_phase_6);
                uint32_t _clc_valid_3 = 0;
                uint32_t _clc_ctaid_x_3;
                uint32_t _clc_ctaid_y_3;
                uint32_t _clc_ctaid_z_3;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.acquire.cta.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %2, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_3), "=r"(_clc_ctaid_y_3), "=r"(_clc_ctaid_z_3), "=r"(_clc_valid_3)
                    : "r"(clc_payload_addr + 0 * 16 + 0 * 16)
                    : "memory");
                mbarrier_arrive(clc_empty_addr);
                if (_clc_valid_3 == 0) {
                    break;
                }
                query_5 = _clc_ctaid_x_3;
                outer_phase_6 = outer_phase_6 ^ 1;
            }
        }
    }

    // Cleanup
}

} // extern "C"
