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
#define TMEM_TMEM_ACC_OFFSET 0
#define NUM_Q_PIPE_STAGES 1
#define NUM_K_PIPE_STAGES 4
#define NUM_TMEM_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 65536
#define SMEM_SMEM_Q_STRIDE 65536
#define SMEM_SMEM_K_OFF 66560
#define SMEM_SMEM_K_STAGE_BYTES 32768
#define SMEM_SMEM_K_STRIDE 32768
#define SMEM_S_W_OFF 197632
#define SMEM_S_W_STAGE_BYTES 1024
#define SMEM_S_W_STRIDE 1024
#define SMEM_TOTAL 198656
#define THREADS 320
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

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_dsa_train_e55b4b44b1fe9a685974(const __grid_constant__ CUtensorMap index_q, const __grid_constant__ CUtensorMap index_k, float* __restrict__ weights, int* __restrict__ cu_ks, int* __restrict__ cu_ke, float* __restrict__ logits, int seq_len, int seq_len_kv)
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
    #define k_full_addr (mbar_base + 8)
    #define k_empty_addr (mbar_base + 40)
    #define umma_full_addr (mbar_base + 72)
    #define umma_empty_addr (mbar_base + 88)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_Q_OFF);
    const int smem_q_addr = smem + SMEM_SMEM_Q_OFF;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_K_OFF);
    const int smem_k_addr = smem + SMEM_SMEM_K_OFF;
    float* s_w = reinterpret_cast<float*>(smem_raw + SMEM_S_W_OFF);
    const int s_w_addr = smem + SMEM_S_W_OFF;
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&index_q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&index_k))) : "memory"); }

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 13 barriers)
    // Mbarriers at smem_raw[0..104)

    if (warp == 8) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 4 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // k_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // --- pipeline 'tmem_pipe' ---
            // umma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // umma_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            mbarrier_init(smem + 96, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 104);
    if (warp == 9) {
        int _tmem_hold = smem + 104;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_acc = taddr;

    // ---- Role: math ----
    if (warp <= 7) {
        { // math_main
            int warp_in_wg = warp % 4;
            int wg = warp / 4;
            int lane_0 = lane;
            int b = blockIdx.y;
            int n_blocks = (seq_len + 7) / 8;
            int blk = blockIdx.x;
            blk = n_blocks - 1 - blockIdx.x;
            int q0 = blk * 8;
            int row0 = (b * seq_len + q0) * 32;
            int kv_row0 = b * seq_len_kv;
            int k_lo = seq_len_kv;
            int k_hi = 0;
            #pragma unroll
            for (int qi = 0; qi < 8; qi++) {
                if (q0 + qi < seq_len) {
                    int _min_6 = ((cu_ks[q0 + qi]) < (seq_len_kv) ? (cu_ks[q0 + qi]) : (seq_len_kv));
                    int _min_7 = ((k_lo) < (_min_6) ? (k_lo) : (_min_6));
                    k_lo = _min_7;
                    int _min_8 = ((cu_ke[q0 + qi]) < (seq_len_kv) ? (cu_ke[q0 + qi]) : (seq_len_kv));
                    int _max_2 = ((k_hi) > (_min_8) ? (k_hi) : (_min_8));
                    k_hi = _max_2;
                }
            }
            int num_tiles = (k_hi - k_lo + 127) / 128;
            int wc = warp * 32 + lane_0;
            float w_val = 0.0f;
            if (q0 + wc / 32 < seq_len) {
                w_val = weights[(long long)row0 + (long long)wc];
            }
            s_w[wc] = w_val;
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            int key_in_tile = warp_in_wg * 32 + lane_0;
            int w_base = wg * 4 * 32 * 4;
            unsigned int t_stage = 0;
            float wv[8];
            float chain[4];
            unsigned int _phase_umma_full = 0;
            #pragma unroll 1
            for (int t = 0; t < num_tiles; t++) {
                int key = k_lo + t * 128 + key_in_tile;
                mbarrier_wait(umma_full_addr + (t_stage) * 8, _phase_umma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[128];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(taddr + t_stage * 256 + (unsigned int)(wg * 128) + (unsigned int)(warp_in_wg * 32 << 16)));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                    : "r"(taddr + t_stage * 256 + (unsigned int)(wg * 128) + (unsigned int)(warp_in_wg * 32 << 16) + 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95])
                    : "r"(taddr + t_stage * 256 + (unsigned int)(wg * 128) + (unsigned int)(warp_in_wg * 32 << 16) + 64));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                    : "r"(taddr + t_stage * 256 + (unsigned int)(wg * 128) + (unsigned int)(warp_in_wg * 32 << 16) + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(umma_empty_addr + (t_stage) * 8);
                t_stage += 1;
                if (t_stage == 2) { t_stage = 0; _phase_umma_full ^= 1; }
                #pragma unroll
                for (int ql = 0; ql < 4; ql++) {
                    int q = q0 + wg * 4 + ql;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        chain[j] = 0.0f;
                    }
                    #pragma unroll
                    for (int nb = 0; nb < 4; nb++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&wv[0])), "=r"(*reinterpret_cast<uint32_t*>(&wv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wv[(0) + 3]))
                            : "r"(s_w_addr + (unsigned int)w_base + (unsigned int)((ql * 32 + nb * 8) * 4)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&wv[4])), "=r"(*reinterpret_cast<uint32_t*>(&wv[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wv[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wv[(4) + 3]))
                            : "r"(s_w_addr + (unsigned int)w_base + (unsigned int)((ql * 32 + nb * 8) * 4) + 16));
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 4; j_1++) {
                            #pragma unroll
                            for (int e = 0; e < 2; e++) {
                                const int h = nb * 8 + 2 * j_1 + e;
                                float _max_3 = max_noftz(_tmem_load_0[ql * 32 + h], 0.0f);
                                chain[j_1] = chain[j_1] + _max_3 * wv[2 * j_1 + e];
                            }
                        }
                    }
                    float part = chain[0] + chain[1] + (chain[2] + chain[3]);
                    if (q < seq_len) {
                        if (key < k_hi) {
                            long long lrow = ((long long)b * (long long)seq_len + (long long)q) * (long long)seq_len_kv;
                            logits[lrow + (long long)key] = part;
                        }
                    }
                }
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            if (warp == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(512));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            int b_1 = blockIdx.y;
            int n_blocks_1 = (seq_len + 7) / 8;
            int blk_1 = blockIdx.x;
            blk_1 = n_blocks_1 - 1 - blockIdx.x;
            int q0_1 = blk_1 * 8;
            int row0_1 = (b_1 * seq_len + q0_1) * 32;
            int kv_row0_1 = b_1 * seq_len_kv;
            int k_lo_1 = seq_len_kv;
            int k_hi_1 = 0;
            #pragma unroll
            for (int qi_1 = 0; qi_1 < 8; qi_1++) {
                if (q0_1 + qi_1 < seq_len) {
                    int _min_0 = ((cu_ks[q0_1 + qi_1]) < (seq_len_kv) ? (cu_ks[q0_1 + qi_1]) : (seq_len_kv));
                    int _min_1 = ((k_lo_1) < (_min_0) ? (k_lo_1) : (_min_0));
                    k_lo_1 = _min_1;
                    int _min_2 = ((cu_ke[q0_1 + qi_1]) < (seq_len_kv) ? (cu_ke[q0_1 + qi_1]) : (seq_len_kv));
                    int _max_0 = ((k_hi_1) > (_min_2) ? (k_hi_1) : (_min_2));
                    k_hi_1 = _max_0;
                }
            }
            int num_tiles_1 = (k_hi_1 - k_lo_1 + 127) / 128;
            unsigned int k_stage_l = 0;
            unsigned int _phase_k_empty = 1;
            if (elect_sync()) {
                tma_3d_gmem2smem(smem_q_addr, (&index_q), 0, row0_1, 0, q_full_addr);
                mbarrier_arrive_expect_tx(q_full_addr, 65536);
                #pragma unroll 1
                for (int t_1 = 0; t_1 < num_tiles_1; t_1++) {
                    mbarrier_wait(k_empty_addr + (k_stage_l) * 8, _phase_k_empty);
                    tma_3d_gmem2smem(smem_k_addr + k_stage_l * 32768, (&index_k), 0, kv_row0_1 + k_lo_1 + t_1 * 128, 0, k_full_addr + (k_stage_l) * 8);
                    mbarrier_arrive_expect_tx(k_full_addr + (k_stage_l) * 8, 32768);
                    k_stage_l += 1;
                    if (k_stage_l == 4) { k_stage_l = 0; _phase_k_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            int b_2 = blockIdx.y;
            int n_blocks_2 = (seq_len + 7) / 8;
            int blk_2 = blockIdx.x;
            blk_2 = n_blocks_2 - 1 - blockIdx.x;
            int q0_2 = blk_2 * 8;
            int row0_2 = (b_2 * seq_len + q0_2) * 32;
            int kv_row0_2 = b_2 * seq_len_kv;
            int k_lo_2 = seq_len_kv;
            int k_hi_2 = 0;
            #pragma unroll
            for (int qi_2 = 0; qi_2 < 8; qi_2++) {
                if (q0_2 + qi_2 < seq_len) {
                    int _min_3 = ((cu_ks[q0_2 + qi_2]) < (seq_len_kv) ? (cu_ks[q0_2 + qi_2]) : (seq_len_kv));
                    int _min_4 = ((k_lo_2) < (_min_3) ? (k_lo_2) : (_min_3));
                    k_lo_2 = _min_4;
                    int _min_5 = ((cu_ke[q0_2 + qi_2]) < (seq_len_kv) ? (cu_ke[q0_2 + qi_2]) : (seq_len_kv));
                    int _max_1 = ((k_hi_2) > (_min_5) ? (k_hi_2) : (_min_5));
                    k_hi_2 = _max_1;
                }
            }
            int num_tiles_2 = (k_hi_2 - k_lo_2 + 127) / 128;
            unsigned int k_stage_m = 0;
            unsigned int t_stage_m = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_umma_empty = 1;
            if (elect_sync()) {
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                #pragma unroll 1
                for (int t_2 = 0; t_2 < num_tiles_2; t_2++) {
                    mbarrier_wait(k_full_addr + (k_stage_m) * 8, _phase_k_full);
                    mbarrier_wait(umma_empty_addr + (t_stage_m) * 8, _phase_umma_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage_m) * 2048;
                    int _mma_b_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 4096;
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
                    "mov.b32 id, 138413200;\n\t"
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
                    "add.u32 blo, blo, 2042;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_acc + (t_stage_m * 256))), "r"(0));
                    tcgen05_commit(umma_full_addr + (t_stage_m) * 8);
                    tcgen05_commit(k_empty_addr + (k_stage_m) * 8);
                    t_stage_m += 1;
                    if (t_stage_m == 2) { t_stage_m = 0; _phase_umma_empty ^= 1; }
                    k_stage_m += 1;
                    if (k_stage_m == 4) { k_stage_m = 0; _phase_k_full ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
