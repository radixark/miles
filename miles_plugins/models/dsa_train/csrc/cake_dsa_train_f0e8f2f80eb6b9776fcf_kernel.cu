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
#define SMEM_S_Q_OFF 0
#define SMEM_S_Q_STAGE_BYTES 16384
#define SMEM_S_Q_STRIDE 16384
#define SMEM_S_K_OFF 16384
#define SMEM_S_K_STAGE_BYTES 32768
#define SMEM_S_K_STRIDE 32768
#define SMEM_S_W_OFF 81920
#define SMEM_S_W_STAGE_BYTES 256
#define SMEM_S_W_STRIDE 256
#define SMEM_TOTAL 82176
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_dsa_train_f0e8f2f80eb6b9776fcf(__nv_bfloat16* __restrict__ index_q, __nv_bfloat16* __restrict__ index_k, float* __restrict__ weights, int* __restrict__ cu_ks, int* __restrict__ cu_ke, float* __restrict__ logits, int seq_len, int seq_len_kv)
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
    __nv_bfloat16* s_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_q_addr = smem + 0;
    __nv_bfloat16* s_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_k_addr = smem + 16384;
    float* s_w = reinterpret_cast<float*>(smem_raw + 81920);
    const int s_w_addr = smem + 81920;

    // === Task calls (dependency order) ===
    int b = blockIdx.y;
    int q0 = blockIdx.x * 2;
    int tid_0 = tid;
    int lane_1 = lane;
    int warp_2 = warp;
    int r_a = lane_1 / 4;
    int r_b = r_a + 8;
    int col_q = lane_1 % 4 * 2;
    long long row0 = ((long long)b * (long long)seq_len + (long long)q0) * 32;
    long long kv_row0 = (long long)b * (long long)seq_len_kv;
    int k_lo = seq_len_kv;
    int k_hi = 0;
    #pragma unroll
    for (int qi = 0; qi < 2; qi++) {
        if (q0 + qi < seq_len) {
            int _min_0 = ((cu_ks[q0 + qi]) < (seq_len_kv) ? (cu_ks[q0 + qi]) : (seq_len_kv));
            int _min_1 = ((k_lo) < (_min_0) ? (k_lo) : (_min_0));
            k_lo = _min_1;
            int _min_2 = ((cu_ke[q0 + qi]) < (seq_len_kv) ? (cu_ke[q0 + qi]) : (seq_len_kv));
            int _max_0 = ((k_hi) > (_min_2) ? (k_hi) : (_min_2));
            k_hi = _max_0;
        }
    }
    int num_tiles = (k_hi - k_lo + 128 - 1) / 128;
    if (tid_0 < 64) {
        float w_val = 0.0f;
        if (q0 + tid_0 / 32 < seq_len) {
            w_val = weights[row0 + (long long)tid_0];
        }
        s_w[tid_0] = w_val;
    }
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_q + ((row0 + (long long)(tid / 16)) * 128 + (long long)(tid % 16 * 8))), "r"((q0 + tid / 16 / 32 < seq_len) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_q + ((row0 + (long long)((256 + tid) / 16)) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((q0 + (256 + tid) / 16 / 32 < seq_len) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_q + ((row0 + (long long)((512 + tid) / 16)) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((q0 + (512 + tid) / 16 / 32 < seq_len) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_q_addr + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_q + ((row0 + (long long)((768 + tid) / 16)) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((q0 + (768 + tid) / 16 / 32 < seq_len) ? 16 : 0));
    if (num_tiles > 0) {
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_k_addr + (unsigned int)(tid % 16 * 8 / 64 * 16384 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(k_lo + tid / 16)) * 128 + (long long)(tid % 16 * 8))), "r"((k_hi > k_lo + tid / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_k_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 16384 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(k_lo + (256 + tid) / 16)) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((k_hi > k_lo + (256 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_k_addr + (unsigned int)((512 + tid) % 16 * 8 / 64 * 16384 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(k_lo + (512 + tid) / 16)) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((k_hi > k_lo + (512 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_k_addr + (unsigned int)((768 + tid) % 16 * 8 / 64 * 16384 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(k_lo + (768 + tid) / 16)) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((k_hi > k_lo + (768 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_k_addr + (unsigned int)((1024 + tid) % 16 * 8 / 64 * 16384 + ((1024 + tid) / 16 * 128 + (1024 + tid) % 16 * 8 % 64 * 2 ^ ((1024 + tid) / 16 * 128 + (1024 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(k_lo + (1024 + tid) / 16)) * 128 + (long long)((1024 + tid) % 16 * 8))), "r"((k_hi > k_lo + (1024 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_k_addr + (unsigned int)((1280 + tid) % 16 * 8 / 64 * 16384 + ((1280 + tid) / 16 * 128 + (1280 + tid) % 16 * 8 % 64 * 2 ^ ((1280 + tid) / 16 * 128 + (1280 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(k_lo + (1280 + tid) / 16)) * 128 + (long long)((1280 + tid) % 16 * 8))), "r"((k_hi > k_lo + (1280 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_k_addr + (unsigned int)((1536 + tid) % 16 * 8 / 64 * 16384 + ((1536 + tid) / 16 * 128 + (1536 + tid) % 16 * 8 % 64 * 2 ^ ((1536 + tid) / 16 * 128 + (1536 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(k_lo + (1536 + tid) / 16)) * 128 + (long long)((1536 + tid) % 16 * 8))), "r"((k_hi > k_lo + (1536 + tid) / 16) ? 16 : 0));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(s_k_addr + (unsigned int)((1792 + tid) % 16 * 8 / 64 * 16384 + ((1792 + tid) / 16 * 128 + (1792 + tid) % 16 * 8 % 64 * 2 ^ ((1792 + tid) / 16 * 128 + (1792 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(k_lo + (1792 + tid) / 16)) * 128 + (long long)((1792 + tid) % 16 * 8))), "r"((k_hi > k_lo + (1792 + tid) / 16) ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;");
    unsigned int a_frag[4];
    unsigned int b_frag[4];
    float acc[32];
    #pragma unroll 1
    for (int t = 0; t < num_tiles; t++) {
        int st = t % 2;
        int nst = 1 - st;
        int key0 = k_lo + t * 128;
        __syncthreads();
        if (num_tiles > t + 1) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_k_addr + (unsigned int)(nst * 32768) + (unsigned int)(tid % 16 * 8 / 64 * 16384 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(key0 + 128 + tid / 16)) * 128 + (long long)(tid % 16 * 8))), "r"((k_hi > key0 + 128 + tid / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_k_addr + (unsigned int)(nst * 32768) + (unsigned int)((256 + tid) % 16 * 8 / 64 * 16384 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(key0 + 128 + (256 + tid) / 16)) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((k_hi > key0 + 128 + (256 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_k_addr + (unsigned int)(nst * 32768) + (unsigned int)((512 + tid) % 16 * 8 / 64 * 16384 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(key0 + 128 + (512 + tid) / 16)) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((k_hi > key0 + 128 + (512 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_k_addr + (unsigned int)(nst * 32768) + (unsigned int)((768 + tid) % 16 * 8 / 64 * 16384 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(key0 + 128 + (768 + tid) / 16)) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((k_hi > key0 + 128 + (768 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_k_addr + (unsigned int)(nst * 32768) + (unsigned int)((1024 + tid) % 16 * 8 / 64 * 16384 + ((1024 + tid) / 16 * 128 + (1024 + tid) % 16 * 8 % 64 * 2 ^ ((1024 + tid) / 16 * 128 + (1024 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(key0 + 128 + (1024 + tid) / 16)) * 128 + (long long)((1024 + tid) % 16 * 8))), "r"((k_hi > key0 + 128 + (1024 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_k_addr + (unsigned int)(nst * 32768) + (unsigned int)((1280 + tid) % 16 * 8 / 64 * 16384 + ((1280 + tid) / 16 * 128 + (1280 + tid) % 16 * 8 % 64 * 2 ^ ((1280 + tid) / 16 * 128 + (1280 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(key0 + 128 + (1280 + tid) / 16)) * 128 + (long long)((1280 + tid) % 16 * 8))), "r"((k_hi > key0 + 128 + (1280 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_k_addr + (unsigned int)(nst * 32768) + (unsigned int)((1536 + tid) % 16 * 8 / 64 * 16384 + ((1536 + tid) / 16 * 128 + (1536 + tid) % 16 * 8 % 64 * 2 ^ ((1536 + tid) / 16 * 128 + (1536 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(key0 + 128 + (1536 + tid) / 16)) * 128 + (long long)((1536 + tid) % 16 * 8))), "r"((k_hi > key0 + 128 + (1536 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(s_k_addr + (unsigned int)(nst * 32768) + (unsigned int)((1792 + tid) % 16 * 8 / 64 * 16384 + ((1792 + tid) / 16 * 128 + (1792 + tid) % 16 * 8 % 64 * 2 ^ ((1792 + tid) / 16 * 128 + (1792 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(index_k + ((kv_row0 + (long long)(key0 + 128 + (1792 + tid) / 16)) * 128 + (long long)((1792 + tid) % 16 * 8))), "r"((k_hi > key0 + 128 + (1792 + tid) / 16) ? 16 : 0));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        int k_base = s_k_addr + (unsigned int)(st * 32768);
        #pragma unroll
        for (int kb = 0; kb < 8; kb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                : "r"(k_base + ((kb * 16 + lane / 16 * 8) / 64 * 16384 + ((warp_2 * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_2 * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            #pragma unroll
            for (int nb2 = 0; nb2 < 4; nb2++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                    : "r"(s_q_addr + (unsigned int)((kb * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((nb2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((nb2 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc + nb2 * 8)[0]), "=f"((acc + nb2 * 8)[1]), "=f"((acc + nb2 * 8)[2]), "=f"((acc + nb2 * 8)[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8)[0])), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8)[1])), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8)[2])), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8)[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc + nb2 * 8 + 4)[0]), "=f"((acc + nb2 * 8 + 4)[1]), "=f"((acc + nb2 * 8 + 4)[2]), "=f"((acc + nb2 * 8 + 4)[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8 + 4)[0])), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8 + 4)[1])), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8 + 4)[2])), "f"(((kb == 0) ? 0.0f : (acc + nb2 * 8 + 4)[3])));
            }
        }
        int key_a = key0 + warp_2 * 16 + r_a;
        int key_b = key_a + 8;
        #pragma unroll
        for (int qi2 = 0; qi2 < 2; qi2++) {
            float part_a = 0.0f;
            float part_b = 0.0f;
            #pragma unroll
            for (int nbq = 0; nbq < 4; nbq++) {
                const int nb = qi2 * 4 + nbq;
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    float w_col = s_w[nb * 8 + col_q + e];
                    float _max_1 = max_noftz(acc[nb * 4 + e], 0.0f);
                    part_a = part_a + _max_1 * w_col;
                    float _max_2 = max_noftz(acc[nb * 4 + 2 + e], 0.0f);
                    part_b = part_b + _max_2 * w_col;
                }
            }
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, part_a, 1);
            part_a = part_a + _shfl_xor_0;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, part_a, 2);
            part_a = part_a + _shfl_xor_1;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, part_b, 1);
            part_b = part_b + _shfl_xor_2;
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, part_b, 2);
            part_b = part_b + _shfl_xor_3;
            if (lane_1 % 4 == 0) {
                if (q0 + qi2 < seq_len) {
                    long long lrow = ((long long)b * (long long)seq_len + (long long)(q0 + qi2)) * (long long)seq_len_kv;
                    if (key_a < k_hi) {
                        logits[lrow + (long long)key_a] = part_a;
                    }
                    if (key_b < k_hi) {
                        logits[lrow + (long long)key_b] = part_b;
                    }
                }
            }
        }
    }
}

} // extern "C"
