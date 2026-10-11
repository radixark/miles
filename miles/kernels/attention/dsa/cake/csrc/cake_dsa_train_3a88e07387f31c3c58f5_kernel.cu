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
#define NUM_MAIN_STAGES 1
#define SMEM_KEYS_OFF 0
#define SMEM_KEYS_STAGE_BYTES 65536
#define SMEM_KEYS_STRIDE 65536
#define SMEM_PAIRS_OFF 65536
#define SMEM_PAIRS_STAGE_BYTES 4096
#define SMEM_PAIRS_STRIDE 4096
#define SMEM_HIST_OFF 69632
#define SMEM_HIST_STAGE_BYTES 1024
#define SMEM_HIST_STRIDE 1024
#define SMEM_CNT_SEL_OFF 70656
#define SMEM_CNT_SEL_STAGE_BYTES 2048
#define SMEM_CNT_SEL_STRIDE 2048
#define SMEM_CNT_TIE_OFF 72704
#define SMEM_CNT_TIE_STAGE_BYTES 2048
#define SMEM_CNT_TIE_STRIDE 2048
#define SMEM_WARP_SUMS_OFF 74752
#define SMEM_WARP_SUMS_STAGE_BYTES 64
#define SMEM_WARP_SUMS_STRIDE 64
#define SMEM_STATE_OFF 74816
#define SMEM_STATE_STAGE_BYTES 16
#define SMEM_STATE_STRIDE 16
#define SMEM_TOTAL 74880
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_dsa_train_3a88e07387f31c3c58f5(float* __restrict__ logits, int* __restrict__ out, int* __restrict__ row_end, int n_kv, int has_row_end)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    unsigned int* keys = reinterpret_cast<unsigned int*>(smem_raw + SMEM_KEYS_OFF);
    const int keys_addr = smem + SMEM_KEYS_OFF;
    unsigned long long* pairs = reinterpret_cast<unsigned long long*>(smem_raw + SMEM_PAIRS_OFF);
    const int pairs_addr = smem + SMEM_PAIRS_OFF;
    unsigned int* hist = reinterpret_cast<unsigned int*>(smem_raw + SMEM_HIST_OFF);
    const int hist_addr = smem + SMEM_HIST_OFF;
    unsigned int* cnt_sel = reinterpret_cast<unsigned int*>(smem_raw + SMEM_CNT_SEL_OFF);
    const int cnt_sel_addr = smem + SMEM_CNT_SEL_OFF;
    unsigned int* cnt_tie = reinterpret_cast<unsigned int*>(smem_raw + SMEM_CNT_TIE_OFF);
    const int cnt_tie_addr = smem + SMEM_CNT_TIE_OFF;
    unsigned int* warp_sums = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WARP_SUMS_OFF);
    const int warp_sums_addr = smem + SMEM_WARP_SUMS_OFF;
    unsigned int* state = reinterpret_cast<unsigned int*>(smem_raw + SMEM_STATE_OFF);
    const int state_addr = smem + SMEM_STATE_OFF;

    // === Task calls (dependency order) ===
    int row = bid;
    unsigned long long row_base = (unsigned long long)row * (unsigned long long)n_kv;
    int limit = n_kv;
    if (has_row_end != 0) {
        int end_i = row_end[row];
        if (end_i < limit) {
            limit = end_i;
        }
    }
    if (limit < 0) {
        limit = 0;
    }
    if (tid < 256) {
        hist[tid] = 0;
    }
    #pragma unroll 4
    for (int idx = tid; idx < 16384; idx += 256) {
        unsigned int key = 0;
        if (limit > idx) {
            float x = logits[row_base + (unsigned long long)idx];
            unsigned int bits = __as_u32(x);
            bits = ((bits == 2147483648u) ? 0 : bits);
            unsigned int sign_mask = (unsigned int)((int)bits >> 31);
            unsigned int enc = bits ^ (sign_mask | 2147483648u);
            unsigned int magnitude = bits & 2147483647;
            bool non_selectable = magnitude > 2139095040 || bits == 4286578688u;
            key = ((non_selectable) ? 0 : enc);
        }
        keys[idx] = key;
    }
    __syncthreads();
    unsigned int prefix = 0;
    unsigned int remaining = 512;
    unsigned int bins[8];
    #pragma unroll 4
    for (int idx0 = tid; idx0 < 16384; idx0 += 256) {
        unsigned int key0 = keys[idx0];
        atomicAdd(&hist[key0 >> 24], 1);
    }
    __syncthreads();
    if (warp == 0) {
        unsigned int lane_total = 0;
        unsigned int hb = hist[lane * 8];
        bins[0] = hb;
        lane_total = lane_total + hb;
        unsigned int hb_0 = hist[lane * 8 + 1];
        bins[1] = hb_0;
        lane_total = lane_total + hb_0;
        unsigned int hb_1 = hist[lane * 8 + 2];
        bins[2] = hb_1;
        lane_total = lane_total + hb_1;
        unsigned int hb_2 = hist[lane * 8 + 3];
        bins[3] = hb_2;
        lane_total = lane_total + hb_2;
        unsigned int hb_3 = hist[lane * 8 + 4];
        bins[4] = hb_3;
        lane_total = lane_total + hb_3;
        unsigned int hb_4 = hist[lane * 8 + 5];
        bins[5] = hb_4;
        lane_total = lane_total + hb_4;
        unsigned int hb_5 = hist[lane * 8 + 6];
        bins[6] = hb_5;
        lane_total = lane_total + hb_5;
        unsigned int hb_6 = hist[lane * 8 + 7];
        bins[7] = hb_6;
        lane_total = lane_total + hb_6;
        unsigned int above_incl = lane_total;
        unsigned int _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, above_incl, 1, 32);
        unsigned int nxt = _shfl_down_0;
        if (lane + 1 < 32) {
            above_incl = above_incl + nxt;
        }
        unsigned int _shfl_down_1 = __shfl_down_sync(0xFFFFFFFF, above_incl, 2, 32);
        unsigned int nxt_7 = _shfl_down_1;
        if (lane + 2 < 32) {
            above_incl = above_incl + nxt_7;
        }
        unsigned int _shfl_down_2 = __shfl_down_sync(0xFFFFFFFF, above_incl, 4, 32);
        unsigned int nxt_8 = _shfl_down_2;
        if (lane + 4 < 32) {
            above_incl = above_incl + nxt_8;
        }
        unsigned int _shfl_down_3 = __shfl_down_sync(0xFFFFFFFF, above_incl, 8, 32);
        unsigned int nxt_9 = _shfl_down_3;
        if (lane + 8 < 32) {
            above_incl = above_incl + nxt_9;
        }
        unsigned int _shfl_down_4 = __shfl_down_sync(0xFFFFFFFF, above_incl, 16, 32);
        unsigned int nxt_10 = _shfl_down_4;
        if (lane + 16 < 32) {
            above_incl = above_incl + nxt_10;
        }
        unsigned int running = above_incl - lane_total;
        unsigned int hit = 4294967295u;
        unsigned int cnt_j = bins[7];
        if (running < remaining) {
            if (remaining <= running + cnt_j) {
                hit = (unsigned int)(lane * 8 + 7) << 16 | remaining - running;
            }
        }
        running = running + cnt_j;
        unsigned int cnt_j_11 = bins[6];
        if (running < remaining) {
            if (remaining <= running + cnt_j_11) {
                hit = (unsigned int)(lane * 8 + 6) << 16 | remaining - running;
            }
        }
        running = running + cnt_j_11;
        unsigned int cnt_j_12 = bins[5];
        if (running < remaining) {
            if (remaining <= running + cnt_j_12) {
                hit = (unsigned int)(lane * 8 + 5) << 16 | remaining - running;
            }
        }
        running = running + cnt_j_12;
        unsigned int cnt_j_13 = bins[4];
        if (running < remaining) {
            if (remaining <= running + cnt_j_13) {
                hit = (unsigned int)(lane * 8 + 4) << 16 | remaining - running;
            }
        }
        running = running + cnt_j_13;
        unsigned int cnt_j_14 = bins[3];
        if (running < remaining) {
            if (remaining <= running + cnt_j_14) {
                hit = (unsigned int)(lane * 8 + 3) << 16 | remaining - running;
            }
        }
        running = running + cnt_j_14;
        unsigned int cnt_j_15 = bins[2];
        if (running < remaining) {
            if (remaining <= running + cnt_j_15) {
                hit = (unsigned int)(lane * 8 + 2) << 16 | remaining - running;
            }
        }
        running = running + cnt_j_15;
        unsigned int cnt_j_16 = bins[1];
        if (running < remaining) {
            if (remaining <= running + cnt_j_16) {
                hit = (unsigned int)(lane * 8 + 1) << 16 | remaining - running;
            }
        }
        running = running + cnt_j_16;
        unsigned int cnt_j_17 = bins[0];
        if (running < remaining) {
            if (remaining <= running + cnt_j_17) {
                hit = (unsigned int)(lane * 8) << 16 | remaining - running;
            }
        }
        running = running + cnt_j_17;
        unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, hit != 4294967295u);
        unsigned int hit_mask = _vote_0;
        int _ffs_0 = __ffs(hit_mask);
        unsigned int src_lane = (unsigned int)(_ffs_0 - 1) & 31;
        unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, hit, src_lane);
        unsigned int chosen = _shfl_0;
        hist[lane * 8] = 0;
        hist[lane * 8 + 1] = 0;
        hist[lane * 8 + 2] = 0;
        hist[lane * 8 + 3] = 0;
        hist[lane * 8 + 4] = 0;
        hist[lane * 8 + 5] = 0;
        hist[lane * 8 + 6] = 0;
        hist[lane * 8 + 7] = 0;
        if (lane == 0) {
            state[0] = chosen;
        }
    }
    __syncthreads();
    unsigned int chosen_all = state[0];
    prefix = prefix | chosen_all >> 16 << 24;
    remaining = chosen_all & 65535;
    unsigned int prefix_hi = prefix >> 24;
    #pragma unroll 4
    for (int idx1 = tid; idx1 < 16384; idx1 += 256) {
        unsigned int key1 = keys[idx1];
        unsigned int key_hi1 = key1 >> 24;
        if (key_hi1 == prefix_hi) {
            atomicAdd(&hist[key1 >> 16 & 255], 1);
        }
    }
    __syncthreads();
    if (warp == 0) {
        unsigned int lane_total1 = 0;
        unsigned int hb1 = hist[lane * 8];
        bins[0] = hb1;
        lane_total1 = lane_total1 + hb1;
        unsigned int hb1_0 = hist[lane * 8 + 1];
        bins[1] = hb1_0;
        lane_total1 = lane_total1 + hb1_0;
        unsigned int hb1_1 = hist[lane * 8 + 2];
        bins[2] = hb1_1;
        lane_total1 = lane_total1 + hb1_1;
        unsigned int hb1_2 = hist[lane * 8 + 3];
        bins[3] = hb1_2;
        lane_total1 = lane_total1 + hb1_2;
        unsigned int hb1_3 = hist[lane * 8 + 4];
        bins[4] = hb1_3;
        lane_total1 = lane_total1 + hb1_3;
        unsigned int hb1_4 = hist[lane * 8 + 5];
        bins[5] = hb1_4;
        lane_total1 = lane_total1 + hb1_4;
        unsigned int hb1_5 = hist[lane * 8 + 6];
        bins[6] = hb1_5;
        lane_total1 = lane_total1 + hb1_5;
        unsigned int hb1_6 = hist[lane * 8 + 7];
        bins[7] = hb1_6;
        lane_total1 = lane_total1 + hb1_6;
        unsigned int above_incl1 = lane_total1;
        unsigned int _shfl_down_5 = __shfl_down_sync(0xFFFFFFFF, above_incl1, 1, 32);
        unsigned int nxt1 = _shfl_down_5;
        if (lane + 1 < 32) {
            above_incl1 = above_incl1 + nxt1;
        }
        unsigned int _shfl_down_6 = __shfl_down_sync(0xFFFFFFFF, above_incl1, 2, 32);
        unsigned int nxt1_7 = _shfl_down_6;
        if (lane + 2 < 32) {
            above_incl1 = above_incl1 + nxt1_7;
        }
        unsigned int _shfl_down_7 = __shfl_down_sync(0xFFFFFFFF, above_incl1, 4, 32);
        unsigned int nxt1_8 = _shfl_down_7;
        if (lane + 4 < 32) {
            above_incl1 = above_incl1 + nxt1_8;
        }
        unsigned int _shfl_down_8 = __shfl_down_sync(0xFFFFFFFF, above_incl1, 8, 32);
        unsigned int nxt1_9 = _shfl_down_8;
        if (lane + 8 < 32) {
            above_incl1 = above_incl1 + nxt1_9;
        }
        unsigned int _shfl_down_9 = __shfl_down_sync(0xFFFFFFFF, above_incl1, 16, 32);
        unsigned int nxt1_10 = _shfl_down_9;
        if (lane + 16 < 32) {
            above_incl1 = above_incl1 + nxt1_10;
        }
        unsigned int running1 = above_incl1 - lane_total1;
        unsigned int hit1 = 4294967295u;
        unsigned int cnt_j1 = bins[7];
        if (running1 < remaining) {
            if (remaining <= running1 + cnt_j1) {
                hit1 = (unsigned int)(lane * 8 + 7) << 16 | remaining - running1;
            }
        }
        running1 = running1 + cnt_j1;
        unsigned int cnt_j1_11 = bins[6];
        if (running1 < remaining) {
            if (remaining <= running1 + cnt_j1_11) {
                hit1 = (unsigned int)(lane * 8 + 6) << 16 | remaining - running1;
            }
        }
        running1 = running1 + cnt_j1_11;
        unsigned int cnt_j1_12 = bins[5];
        if (running1 < remaining) {
            if (remaining <= running1 + cnt_j1_12) {
                hit1 = (unsigned int)(lane * 8 + 5) << 16 | remaining - running1;
            }
        }
        running1 = running1 + cnt_j1_12;
        unsigned int cnt_j1_13 = bins[4];
        if (running1 < remaining) {
            if (remaining <= running1 + cnt_j1_13) {
                hit1 = (unsigned int)(lane * 8 + 4) << 16 | remaining - running1;
            }
        }
        running1 = running1 + cnt_j1_13;
        unsigned int cnt_j1_14 = bins[3];
        if (running1 < remaining) {
            if (remaining <= running1 + cnt_j1_14) {
                hit1 = (unsigned int)(lane * 8 + 3) << 16 | remaining - running1;
            }
        }
        running1 = running1 + cnt_j1_14;
        unsigned int cnt_j1_15 = bins[2];
        if (running1 < remaining) {
            if (remaining <= running1 + cnt_j1_15) {
                hit1 = (unsigned int)(lane * 8 + 2) << 16 | remaining - running1;
            }
        }
        running1 = running1 + cnt_j1_15;
        unsigned int cnt_j1_16 = bins[1];
        if (running1 < remaining) {
            if (remaining <= running1 + cnt_j1_16) {
                hit1 = (unsigned int)(lane * 8 + 1) << 16 | remaining - running1;
            }
        }
        running1 = running1 + cnt_j1_16;
        unsigned int cnt_j1_17 = bins[0];
        if (running1 < remaining) {
            if (remaining <= running1 + cnt_j1_17) {
                hit1 = (unsigned int)(lane * 8) << 16 | remaining - running1;
            }
        }
        running1 = running1 + cnt_j1_17;
        unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, hit1 != 4294967295u);
        unsigned int hit_mask1 = _vote_1;
        int _ffs_1 = __ffs(hit_mask1);
        unsigned int src_lane1 = (unsigned int)(_ffs_1 - 1) & 31;
        unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, hit1, src_lane1);
        unsigned int chosen1 = _shfl_1;
        hist[lane * 8] = 0;
        hist[lane * 8 + 1] = 0;
        hist[lane * 8 + 2] = 0;
        hist[lane * 8 + 3] = 0;
        hist[lane * 8 + 4] = 0;
        hist[lane * 8 + 5] = 0;
        hist[lane * 8 + 6] = 0;
        hist[lane * 8 + 7] = 0;
        if (lane == 0) {
            state[0] = chosen1;
        }
    }
    __syncthreads();
    unsigned int chosen_all1 = state[0];
    prefix = prefix | chosen_all1 >> 16 << 16;
    remaining = chosen_all1 & 65535;
    unsigned int prefix_hi_0 = prefix >> 16;
    #pragma unroll 4
    for (int idx1_1 = tid; idx1_1 < 16384; idx1_1 += 256) {
        unsigned int key1_1 = keys[idx1_1];
        unsigned int key_hi1_1 = key1_1 >> 16;
        if (key_hi1_1 == prefix_hi_0) {
            atomicAdd(&hist[key1_1 >> 8 & 255], 1);
        }
    }
    __syncthreads();
    if (warp == 0) {
        unsigned int lane_total1_1 = 0;
        unsigned int hb1_7 = hist[lane * 8];
        bins[0] = hb1_7;
        lane_total1_1 = lane_total1_1 + hb1_7;
        unsigned int hb1_0_1 = hist[lane * 8 + 1];
        bins[1] = hb1_0_1;
        lane_total1_1 = lane_total1_1 + hb1_0_1;
        unsigned int hb1_1_1 = hist[lane * 8 + 2];
        bins[2] = hb1_1_1;
        lane_total1_1 = lane_total1_1 + hb1_1_1;
        unsigned int hb1_2_1 = hist[lane * 8 + 3];
        bins[3] = hb1_2_1;
        lane_total1_1 = lane_total1_1 + hb1_2_1;
        unsigned int hb1_3_1 = hist[lane * 8 + 4];
        bins[4] = hb1_3_1;
        lane_total1_1 = lane_total1_1 + hb1_3_1;
        unsigned int hb1_4_1 = hist[lane * 8 + 5];
        bins[5] = hb1_4_1;
        lane_total1_1 = lane_total1_1 + hb1_4_1;
        unsigned int hb1_5_1 = hist[lane * 8 + 6];
        bins[6] = hb1_5_1;
        lane_total1_1 = lane_total1_1 + hb1_5_1;
        unsigned int hb1_6_1 = hist[lane * 8 + 7];
        bins[7] = hb1_6_1;
        lane_total1_1 = lane_total1_1 + hb1_6_1;
        unsigned int above_incl1_1 = lane_total1_1;
        unsigned int _shfl_down_10 = __shfl_down_sync(0xFFFFFFFF, above_incl1_1, 1, 32);
        unsigned int nxt1_1 = _shfl_down_10;
        if (lane + 1 < 32) {
            above_incl1_1 = above_incl1_1 + nxt1_1;
        }
        unsigned int _shfl_down_11 = __shfl_down_sync(0xFFFFFFFF, above_incl1_1, 2, 32);
        unsigned int nxt1_7_1 = _shfl_down_11;
        if (lane + 2 < 32) {
            above_incl1_1 = above_incl1_1 + nxt1_7_1;
        }
        unsigned int _shfl_down_12 = __shfl_down_sync(0xFFFFFFFF, above_incl1_1, 4, 32);
        unsigned int nxt1_8_1 = _shfl_down_12;
        if (lane + 4 < 32) {
            above_incl1_1 = above_incl1_1 + nxt1_8_1;
        }
        unsigned int _shfl_down_13 = __shfl_down_sync(0xFFFFFFFF, above_incl1_1, 8, 32);
        unsigned int nxt1_9_1 = _shfl_down_13;
        if (lane + 8 < 32) {
            above_incl1_1 = above_incl1_1 + nxt1_9_1;
        }
        unsigned int _shfl_down_14 = __shfl_down_sync(0xFFFFFFFF, above_incl1_1, 16, 32);
        unsigned int nxt1_10_1 = _shfl_down_14;
        if (lane + 16 < 32) {
            above_incl1_1 = above_incl1_1 + nxt1_10_1;
        }
        unsigned int running1_1 = above_incl1_1 - lane_total1_1;
        unsigned int hit1_1 = 4294967295u;
        unsigned int cnt_j1_1 = bins[7];
        if (running1_1 < remaining) {
            if (remaining <= running1_1 + cnt_j1_1) {
                hit1_1 = (unsigned int)(lane * 8 + 7) << 16 | remaining - running1_1;
            }
        }
        running1_1 = running1_1 + cnt_j1_1;
        unsigned int cnt_j1_11_1 = bins[6];
        if (running1_1 < remaining) {
            if (remaining <= running1_1 + cnt_j1_11_1) {
                hit1_1 = (unsigned int)(lane * 8 + 6) << 16 | remaining - running1_1;
            }
        }
        running1_1 = running1_1 + cnt_j1_11_1;
        unsigned int cnt_j1_12_1 = bins[5];
        if (running1_1 < remaining) {
            if (remaining <= running1_1 + cnt_j1_12_1) {
                hit1_1 = (unsigned int)(lane * 8 + 5) << 16 | remaining - running1_1;
            }
        }
        running1_1 = running1_1 + cnt_j1_12_1;
        unsigned int cnt_j1_13_1 = bins[4];
        if (running1_1 < remaining) {
            if (remaining <= running1_1 + cnt_j1_13_1) {
                hit1_1 = (unsigned int)(lane * 8 + 4) << 16 | remaining - running1_1;
            }
        }
        running1_1 = running1_1 + cnt_j1_13_1;
        unsigned int cnt_j1_14_1 = bins[3];
        if (running1_1 < remaining) {
            if (remaining <= running1_1 + cnt_j1_14_1) {
                hit1_1 = (unsigned int)(lane * 8 + 3) << 16 | remaining - running1_1;
            }
        }
        running1_1 = running1_1 + cnt_j1_14_1;
        unsigned int cnt_j1_15_1 = bins[2];
        if (running1_1 < remaining) {
            if (remaining <= running1_1 + cnt_j1_15_1) {
                hit1_1 = (unsigned int)(lane * 8 + 2) << 16 | remaining - running1_1;
            }
        }
        running1_1 = running1_1 + cnt_j1_15_1;
        unsigned int cnt_j1_16_1 = bins[1];
        if (running1_1 < remaining) {
            if (remaining <= running1_1 + cnt_j1_16_1) {
                hit1_1 = (unsigned int)(lane * 8 + 1) << 16 | remaining - running1_1;
            }
        }
        running1_1 = running1_1 + cnt_j1_16_1;
        unsigned int cnt_j1_17_1 = bins[0];
        if (running1_1 < remaining) {
            if (remaining <= running1_1 + cnt_j1_17_1) {
                hit1_1 = (unsigned int)(lane * 8) << 16 | remaining - running1_1;
            }
        }
        running1_1 = running1_1 + cnt_j1_17_1;
        unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, hit1_1 != 4294967295u);
        unsigned int hit_mask1_1 = _vote_2;
        int _ffs_2 = __ffs(hit_mask1_1);
        unsigned int src_lane1_1 = (unsigned int)(_ffs_2 - 1) & 31;
        unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, hit1_1, src_lane1_1);
        unsigned int chosen1_1 = _shfl_2;
        hist[lane * 8] = 0;
        hist[lane * 8 + 1] = 0;
        hist[lane * 8 + 2] = 0;
        hist[lane * 8 + 3] = 0;
        hist[lane * 8 + 4] = 0;
        hist[lane * 8 + 5] = 0;
        hist[lane * 8 + 6] = 0;
        hist[lane * 8 + 7] = 0;
        if (lane == 0) {
            state[0] = chosen1_1;
        }
    }
    __syncthreads();
    unsigned int chosen_all1_1 = state[0];
    prefix = prefix | chosen_all1_1 >> 16 << 8;
    remaining = chosen_all1_1 & 65535;
    unsigned int prefix_hi_2 = prefix >> 8;
    #pragma unroll 4
    for (int idx1_2 = tid; idx1_2 < 16384; idx1_2 += 256) {
        unsigned int key1_2 = keys[idx1_2];
        unsigned int key_hi1_2 = key1_2 >> 8;
        if (key_hi1_2 == prefix_hi_2) {
            atomicAdd(&hist[key1_2 & 255], 1);
        }
    }
    __syncthreads();
    if (warp == 0) {
        unsigned int lane_total1_2 = 0;
        unsigned int hb1_8 = hist[lane * 8];
        bins[0] = hb1_8;
        lane_total1_2 = lane_total1_2 + hb1_8;
        unsigned int hb1_0_2 = hist[lane * 8 + 1];
        bins[1] = hb1_0_2;
        lane_total1_2 = lane_total1_2 + hb1_0_2;
        unsigned int hb1_1_2 = hist[lane * 8 + 2];
        bins[2] = hb1_1_2;
        lane_total1_2 = lane_total1_2 + hb1_1_2;
        unsigned int hb1_2_2 = hist[lane * 8 + 3];
        bins[3] = hb1_2_2;
        lane_total1_2 = lane_total1_2 + hb1_2_2;
        unsigned int hb1_3_2 = hist[lane * 8 + 4];
        bins[4] = hb1_3_2;
        lane_total1_2 = lane_total1_2 + hb1_3_2;
        unsigned int hb1_4_2 = hist[lane * 8 + 5];
        bins[5] = hb1_4_2;
        lane_total1_2 = lane_total1_2 + hb1_4_2;
        unsigned int hb1_5_2 = hist[lane * 8 + 6];
        bins[6] = hb1_5_2;
        lane_total1_2 = lane_total1_2 + hb1_5_2;
        unsigned int hb1_6_2 = hist[lane * 8 + 7];
        bins[7] = hb1_6_2;
        lane_total1_2 = lane_total1_2 + hb1_6_2;
        unsigned int above_incl1_2 = lane_total1_2;
        unsigned int _shfl_down_15 = __shfl_down_sync(0xFFFFFFFF, above_incl1_2, 1, 32);
        unsigned int nxt1_2 = _shfl_down_15;
        if (lane + 1 < 32) {
            above_incl1_2 = above_incl1_2 + nxt1_2;
        }
        unsigned int _shfl_down_16 = __shfl_down_sync(0xFFFFFFFF, above_incl1_2, 2, 32);
        unsigned int nxt1_7_2 = _shfl_down_16;
        if (lane + 2 < 32) {
            above_incl1_2 = above_incl1_2 + nxt1_7_2;
        }
        unsigned int _shfl_down_17 = __shfl_down_sync(0xFFFFFFFF, above_incl1_2, 4, 32);
        unsigned int nxt1_8_2 = _shfl_down_17;
        if (lane + 4 < 32) {
            above_incl1_2 = above_incl1_2 + nxt1_8_2;
        }
        unsigned int _shfl_down_18 = __shfl_down_sync(0xFFFFFFFF, above_incl1_2, 8, 32);
        unsigned int nxt1_9_2 = _shfl_down_18;
        if (lane + 8 < 32) {
            above_incl1_2 = above_incl1_2 + nxt1_9_2;
        }
        unsigned int _shfl_down_19 = __shfl_down_sync(0xFFFFFFFF, above_incl1_2, 16, 32);
        unsigned int nxt1_10_2 = _shfl_down_19;
        if (lane + 16 < 32) {
            above_incl1_2 = above_incl1_2 + nxt1_10_2;
        }
        unsigned int running1_2 = above_incl1_2 - lane_total1_2;
        unsigned int hit1_2 = 4294967295u;
        unsigned int cnt_j1_2 = bins[7];
        if (running1_2 < remaining) {
            if (remaining <= running1_2 + cnt_j1_2) {
                hit1_2 = (unsigned int)(lane * 8 + 7) << 16 | remaining - running1_2;
            }
        }
        running1_2 = running1_2 + cnt_j1_2;
        unsigned int cnt_j1_11_2 = bins[6];
        if (running1_2 < remaining) {
            if (remaining <= running1_2 + cnt_j1_11_2) {
                hit1_2 = (unsigned int)(lane * 8 + 6) << 16 | remaining - running1_2;
            }
        }
        running1_2 = running1_2 + cnt_j1_11_2;
        unsigned int cnt_j1_12_2 = bins[5];
        if (running1_2 < remaining) {
            if (remaining <= running1_2 + cnt_j1_12_2) {
                hit1_2 = (unsigned int)(lane * 8 + 5) << 16 | remaining - running1_2;
            }
        }
        running1_2 = running1_2 + cnt_j1_12_2;
        unsigned int cnt_j1_13_2 = bins[4];
        if (running1_2 < remaining) {
            if (remaining <= running1_2 + cnt_j1_13_2) {
                hit1_2 = (unsigned int)(lane * 8 + 4) << 16 | remaining - running1_2;
            }
        }
        running1_2 = running1_2 + cnt_j1_13_2;
        unsigned int cnt_j1_14_2 = bins[3];
        if (running1_2 < remaining) {
            if (remaining <= running1_2 + cnt_j1_14_2) {
                hit1_2 = (unsigned int)(lane * 8 + 3) << 16 | remaining - running1_2;
            }
        }
        running1_2 = running1_2 + cnt_j1_14_2;
        unsigned int cnt_j1_15_2 = bins[2];
        if (running1_2 < remaining) {
            if (remaining <= running1_2 + cnt_j1_15_2) {
                hit1_2 = (unsigned int)(lane * 8 + 2) << 16 | remaining - running1_2;
            }
        }
        running1_2 = running1_2 + cnt_j1_15_2;
        unsigned int cnt_j1_16_2 = bins[1];
        if (running1_2 < remaining) {
            if (remaining <= running1_2 + cnt_j1_16_2) {
                hit1_2 = (unsigned int)(lane * 8 + 1) << 16 | remaining - running1_2;
            }
        }
        running1_2 = running1_2 + cnt_j1_16_2;
        unsigned int cnt_j1_17_2 = bins[0];
        if (running1_2 < remaining) {
            if (remaining <= running1_2 + cnt_j1_17_2) {
                hit1_2 = (unsigned int)(lane * 8) << 16 | remaining - running1_2;
            }
        }
        running1_2 = running1_2 + cnt_j1_17_2;
        unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, hit1_2 != 4294967295u);
        unsigned int hit_mask1_2 = _vote_3;
        int _ffs_3 = __ffs(hit_mask1_2);
        unsigned int src_lane1_2 = (unsigned int)(_ffs_3 - 1) & 31;
        unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, hit1_2, src_lane1_2);
        unsigned int chosen1_2 = _shfl_3;
        hist[lane * 8] = 0;
        hist[lane * 8 + 1] = 0;
        hist[lane * 8 + 2] = 0;
        hist[lane * 8 + 3] = 0;
        hist[lane * 8 + 4] = 0;
        hist[lane * 8 + 5] = 0;
        hist[lane * 8 + 6] = 0;
        hist[lane * 8 + 7] = 0;
        if (lane == 0) {
            state[0] = chosen1_2;
        }
    }
    __syncthreads();
    unsigned int chosen_all1_3 = state[0];
    prefix = prefix | chosen_all1_3 >> 16;
    remaining = chosen_all1_3 & 65535;
    unsigned int pivot = prefix;
    unsigned int quota = remaining;
    if (pivot == 0) {
        quota = 0;
    }
    unsigned int count_gt = 512 - remaining;
    unsigned int n_selected = count_gt + quota;
    #pragma unroll 1
    for (int c = 0; c < 64; c++) {
        int idx2 = c * 256 + tid;
        unsigned int k2 = keys[idx2];
        unsigned int _vote_4 = __ballot_sync(0xFFFFFFFF, k2 > pivot);
        unsigned int sel_mask = _vote_4;
        unsigned int _vote_5 = __ballot_sync(0xFFFFFFFF, k2 == pivot);
        unsigned int tie_mask = _vote_5;
        if (lane == 0) {
            int slot = c * 8 + warp;
            int _popc_0 = __popc(sel_mask);
            cnt_sel[slot] = (unsigned int)_popc_0;
            int _popc_1 = __popc(tie_mask);
            cnt_tie[slot] = (unsigned int)_popc_1;
        }
    }
    #pragma unroll 1
    for (int s = tid; s < 512; s += 256) {
        pairs[s] = 0;
    }
    __syncthreads();
    unsigned int local_sel = 0;
    unsigned int local_tie = 0;
    int f = tid * 2;
    if (f < 512) {
        local_sel = local_sel + cnt_sel[f];
        local_tie = local_tie + cnt_tie[f];
    }
    int f_4 = tid * 2 + 1;
    if (f_4 < 512) {
        local_sel = local_sel + cnt_sel[f_4];
        local_tie = local_tie + cnt_tie[f_4];
    }
    unsigned int incl_sel = local_sel;
    unsigned int incl_tie = local_tie;
    unsigned int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 1, 32);
    unsigned int up_sel = _shfl_up_0;
    unsigned int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 1, 32);
    unsigned int up_tie = _shfl_up_1;
    if (lane >= 1) {
        incl_sel = incl_sel + up_sel;
        incl_tie = incl_tie + up_tie;
    }
    unsigned int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 2, 32);
    unsigned int up_sel_5 = _shfl_up_2;
    unsigned int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 2, 32);
    unsigned int up_tie_6 = _shfl_up_3;
    if (lane >= 2) {
        incl_sel = incl_sel + up_sel_5;
        incl_tie = incl_tie + up_tie_6;
    }
    unsigned int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 4, 32);
    unsigned int up_sel_7 = _shfl_up_4;
    unsigned int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 4, 32);
    unsigned int up_tie_8 = _shfl_up_5;
    if (lane >= 4) {
        incl_sel = incl_sel + up_sel_7;
        incl_tie = incl_tie + up_tie_8;
    }
    unsigned int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 8, 32);
    unsigned int up_sel_9 = _shfl_up_6;
    unsigned int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 8, 32);
    unsigned int up_tie_10 = _shfl_up_7;
    if (lane >= 8) {
        incl_sel = incl_sel + up_sel_9;
        incl_tie = incl_tie + up_tie_10;
    }
    unsigned int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 16, 32);
    unsigned int up_sel_11 = _shfl_up_8;
    unsigned int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 16, 32);
    unsigned int up_tie_12 = _shfl_up_9;
    if (lane >= 16) {
        incl_sel = incl_sel + up_sel_11;
        incl_tie = incl_tie + up_tie_12;
    }
    if (lane == 31) {
        warp_sums[warp] = incl_sel;
        warp_sums[8 + warp] = incl_tie;
    }
    __syncthreads();
    unsigned int base_sel = 0;
    unsigned int base_tie = 0;
    unsigned int ws = warp_sums[0];
    unsigned int wt = warp_sums[8];
    if (warp > 0) {
        base_sel = base_sel + ws;
        base_tie = base_tie + wt;
    }
    unsigned int ws_13 = warp_sums[1];
    unsigned int wt_14 = warp_sums[9];
    if (warp > 1) {
        base_sel = base_sel + ws_13;
        base_tie = base_tie + wt_14;
    }
    unsigned int ws_15 = warp_sums[2];
    unsigned int wt_16 = warp_sums[10];
    if (warp > 2) {
        base_sel = base_sel + ws_15;
        base_tie = base_tie + wt_16;
    }
    unsigned int ws_17 = warp_sums[3];
    unsigned int wt_18 = warp_sums[11];
    if (warp > 3) {
        base_sel = base_sel + ws_17;
        base_tie = base_tie + wt_18;
    }
    unsigned int ws_19 = warp_sums[4];
    unsigned int wt_20 = warp_sums[12];
    if (warp > 4) {
        base_sel = base_sel + ws_19;
        base_tie = base_tie + wt_20;
    }
    unsigned int ws_21 = warp_sums[5];
    unsigned int wt_22 = warp_sums[13];
    if (warp > 5) {
        base_sel = base_sel + ws_21;
        base_tie = base_tie + wt_22;
    }
    unsigned int ws_23 = warp_sums[6];
    unsigned int wt_24 = warp_sums[14];
    if (warp > 6) {
        base_sel = base_sel + ws_23;
        base_tie = base_tie + wt_24;
    }
    unsigned int ws_25 = warp_sums[7];
    unsigned int wt_26 = warp_sums[15];
    if (warp > 7) {
        base_sel = base_sel + ws_25;
        base_tie = base_tie + wt_26;
    }
    unsigned int run_sel = base_sel + incl_sel - local_sel;
    unsigned int run_tie = base_tie + incl_tie - local_tie;
    int f2 = tid * 2;
    if (f2 < 512) {
        unsigned int cur_sel = cnt_sel[f2];
        unsigned int cur_tie = cnt_tie[f2];
        cnt_sel[f2] = run_sel;
        cnt_tie[f2] = run_tie;
        run_sel = run_sel + cur_sel;
        run_tie = run_tie + cur_tie;
    }
    int f2_27 = tid * 2 + 1;
    if (f2_27 < 512) {
        unsigned int cur_sel_1 = cnt_sel[f2_27];
        unsigned int cur_tie_1 = cnt_tie[f2_27];
        cnt_sel[f2_27] = run_sel;
        cnt_tie[f2_27] = run_tie;
        run_sel = run_sel + cur_sel_1;
        run_tie = run_tie + cur_tie_1;
    }
    __syncthreads();
    unsigned int lt_mask = ~(4294967295u << (unsigned int)lane);
    #pragma unroll 1
    for (int c2 = 0; c2 < 64; c2++) {
        int idx3 = c2 * 256 + tid;
        unsigned int k3 = keys[idx3];
        bool is_sel = k3 > pivot;
        bool is_tie = k3 == pivot;
        unsigned int _vote_6 = __ballot_sync(0xFFFFFFFF, is_sel);
        unsigned int sel_mask2 = _vote_6;
        unsigned int _vote_7 = __ballot_sync(0xFFFFFFFF, is_tie);
        unsigned int tie_mask2 = _vote_7;
        int slot2 = c2 * 8 + warp;
        unsigned int inv_idx = (unsigned int)idx3 ^ 4294967295u;
        unsigned long long composite = (unsigned long long)k3 << 32 | (unsigned long long)inv_idx;
        if (is_sel) {
            int _popc_2 = __popc(sel_mask2 & lt_mask);
            unsigned int pos = cnt_sel[slot2] + (unsigned int)_popc_2;
            pairs[pos] = composite;
        } else if (is_tie) {
            int _popc_3 = __popc(tie_mask2 & lt_mask);
            unsigned int rank = cnt_tie[slot2] + (unsigned int)_popc_3;
            if (rank < quota) {
                pairs[count_gt + rank] = composite;
            }
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t = tid; t < 256; t += 256) {
        int low = t * 2;
        int high = low + 1;
        unsigned long long a = pairs[low];
        unsigned long long b = pairs[high];
        int dir_bit = low & 2;
        if (dir_bit == 0) {
            if (a < b) {
                pairs[low] = b;
                pairs[high] = a;
            }
        } else if (a > b) {
            pairs[low] = b;
            pairs[high] = a;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_1 = tid; t_1 < 256; t_1 += 256) {
        int low_1 = t_1 / 2 * 4 + t_1 % 2;
        int high_1 = low_1 + 2;
        unsigned long long a_1 = pairs[low_1];
        unsigned long long b_1 = pairs[high_1];
        int dir_bit_1 = low_1 & 4;
        if (dir_bit_1 == 0) {
            if (a_1 < b_1) {
                pairs[low_1] = b_1;
                pairs[high_1] = a_1;
            }
        } else if (a_1 > b_1) {
            pairs[low_1] = b_1;
            pairs[high_1] = a_1;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_2 = tid; t_2 < 256; t_2 += 256) {
        int low_2 = t_2 * 2;
        int high_2 = low_2 + 1;
        unsigned long long a_2 = pairs[low_2];
        unsigned long long b_2 = pairs[high_2];
        int dir_bit_2 = low_2 & 4;
        if (dir_bit_2 == 0) {
            if (a_2 < b_2) {
                pairs[low_2] = b_2;
                pairs[high_2] = a_2;
            }
        } else if (a_2 > b_2) {
            pairs[low_2] = b_2;
            pairs[high_2] = a_2;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_3 = tid; t_3 < 256; t_3 += 256) {
        int low_3 = t_3 / 4 * 8 + t_3 % 4;
        int high_3 = low_3 + 4;
        unsigned long long a_3 = pairs[low_3];
        unsigned long long b_3 = pairs[high_3];
        int dir_bit_3 = low_3 & 8;
        if (dir_bit_3 == 0) {
            if (a_3 < b_3) {
                pairs[low_3] = b_3;
                pairs[high_3] = a_3;
            }
        } else if (a_3 > b_3) {
            pairs[low_3] = b_3;
            pairs[high_3] = a_3;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_4 = tid; t_4 < 256; t_4 += 256) {
        int low_4 = t_4 / 2 * 4 + t_4 % 2;
        int high_4 = low_4 + 2;
        unsigned long long a_4 = pairs[low_4];
        unsigned long long b_4 = pairs[high_4];
        int dir_bit_4 = low_4 & 8;
        if (dir_bit_4 == 0) {
            if (a_4 < b_4) {
                pairs[low_4] = b_4;
                pairs[high_4] = a_4;
            }
        } else if (a_4 > b_4) {
            pairs[low_4] = b_4;
            pairs[high_4] = a_4;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_5 = tid; t_5 < 256; t_5 += 256) {
        int low_5 = t_5 * 2;
        int high_5 = low_5 + 1;
        unsigned long long a_5 = pairs[low_5];
        unsigned long long b_5 = pairs[high_5];
        int dir_bit_5 = low_5 & 8;
        if (dir_bit_5 == 0) {
            if (a_5 < b_5) {
                pairs[low_5] = b_5;
                pairs[high_5] = a_5;
            }
        } else if (a_5 > b_5) {
            pairs[low_5] = b_5;
            pairs[high_5] = a_5;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_6 = tid; t_6 < 256; t_6 += 256) {
        int low_6 = t_6 / 8 * 16 + t_6 % 8;
        int high_6 = low_6 + 8;
        unsigned long long a_6 = pairs[low_6];
        unsigned long long b_6 = pairs[high_6];
        int dir_bit_6 = low_6 & 16;
        if (dir_bit_6 == 0) {
            if (a_6 < b_6) {
                pairs[low_6] = b_6;
                pairs[high_6] = a_6;
            }
        } else if (a_6 > b_6) {
            pairs[low_6] = b_6;
            pairs[high_6] = a_6;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_7 = tid; t_7 < 256; t_7 += 256) {
        int low_7 = t_7 / 4 * 8 + t_7 % 4;
        int high_7 = low_7 + 4;
        unsigned long long a_7 = pairs[low_7];
        unsigned long long b_7 = pairs[high_7];
        int dir_bit_7 = low_7 & 16;
        if (dir_bit_7 == 0) {
            if (a_7 < b_7) {
                pairs[low_7] = b_7;
                pairs[high_7] = a_7;
            }
        } else if (a_7 > b_7) {
            pairs[low_7] = b_7;
            pairs[high_7] = a_7;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_8 = tid; t_8 < 256; t_8 += 256) {
        int low_8 = t_8 / 2 * 4 + t_8 % 2;
        int high_8 = low_8 + 2;
        unsigned long long a_8 = pairs[low_8];
        unsigned long long b_8 = pairs[high_8];
        int dir_bit_8 = low_8 & 16;
        if (dir_bit_8 == 0) {
            if (a_8 < b_8) {
                pairs[low_8] = b_8;
                pairs[high_8] = a_8;
            }
        } else if (a_8 > b_8) {
            pairs[low_8] = b_8;
            pairs[high_8] = a_8;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_9 = tid; t_9 < 256; t_9 += 256) {
        int low_9 = t_9 * 2;
        int high_9 = low_9 + 1;
        unsigned long long a_9 = pairs[low_9];
        unsigned long long b_9 = pairs[high_9];
        int dir_bit_9 = low_9 & 16;
        if (dir_bit_9 == 0) {
            if (a_9 < b_9) {
                pairs[low_9] = b_9;
                pairs[high_9] = a_9;
            }
        } else if (a_9 > b_9) {
            pairs[low_9] = b_9;
            pairs[high_9] = a_9;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_10 = tid; t_10 < 256; t_10 += 256) {
        int low_10 = t_10 / 16 * 32 + t_10 % 16;
        int high_10 = low_10 + 16;
        unsigned long long a_10 = pairs[low_10];
        unsigned long long b_10 = pairs[high_10];
        int dir_bit_10 = low_10 & 32;
        if (dir_bit_10 == 0) {
            if (a_10 < b_10) {
                pairs[low_10] = b_10;
                pairs[high_10] = a_10;
            }
        } else if (a_10 > b_10) {
            pairs[low_10] = b_10;
            pairs[high_10] = a_10;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_11 = tid; t_11 < 256; t_11 += 256) {
        int low_11 = t_11 / 8 * 16 + t_11 % 8;
        int high_11 = low_11 + 8;
        unsigned long long a_11 = pairs[low_11];
        unsigned long long b_11 = pairs[high_11];
        int dir_bit_11 = low_11 & 32;
        if (dir_bit_11 == 0) {
            if (a_11 < b_11) {
                pairs[low_11] = b_11;
                pairs[high_11] = a_11;
            }
        } else if (a_11 > b_11) {
            pairs[low_11] = b_11;
            pairs[high_11] = a_11;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_12 = tid; t_12 < 256; t_12 += 256) {
        int low_12 = t_12 / 4 * 8 + t_12 % 4;
        int high_12 = low_12 + 4;
        unsigned long long a_12 = pairs[low_12];
        unsigned long long b_12 = pairs[high_12];
        int dir_bit_12 = low_12 & 32;
        if (dir_bit_12 == 0) {
            if (a_12 < b_12) {
                pairs[low_12] = b_12;
                pairs[high_12] = a_12;
            }
        } else if (a_12 > b_12) {
            pairs[low_12] = b_12;
            pairs[high_12] = a_12;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_13 = tid; t_13 < 256; t_13 += 256) {
        int low_13 = t_13 / 2 * 4 + t_13 % 2;
        int high_13 = low_13 + 2;
        unsigned long long a_13 = pairs[low_13];
        unsigned long long b_13 = pairs[high_13];
        int dir_bit_13 = low_13 & 32;
        if (dir_bit_13 == 0) {
            if (a_13 < b_13) {
                pairs[low_13] = b_13;
                pairs[high_13] = a_13;
            }
        } else if (a_13 > b_13) {
            pairs[low_13] = b_13;
            pairs[high_13] = a_13;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_14 = tid; t_14 < 256; t_14 += 256) {
        int low_14 = t_14 * 2;
        int high_14 = low_14 + 1;
        unsigned long long a_14 = pairs[low_14];
        unsigned long long b_14 = pairs[high_14];
        int dir_bit_14 = low_14 & 32;
        if (dir_bit_14 == 0) {
            if (a_14 < b_14) {
                pairs[low_14] = b_14;
                pairs[high_14] = a_14;
            }
        } else if (a_14 > b_14) {
            pairs[low_14] = b_14;
            pairs[high_14] = a_14;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_15 = tid; t_15 < 256; t_15 += 256) {
        int low_15 = t_15 / 32 * 64 + t_15 % 32;
        int high_15 = low_15 + 32;
        unsigned long long a_15 = pairs[low_15];
        unsigned long long b_15 = pairs[high_15];
        int dir_bit_15 = low_15 & 64;
        if (dir_bit_15 == 0) {
            if (a_15 < b_15) {
                pairs[low_15] = b_15;
                pairs[high_15] = a_15;
            }
        } else if (a_15 > b_15) {
            pairs[low_15] = b_15;
            pairs[high_15] = a_15;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_16 = tid; t_16 < 256; t_16 += 256) {
        int low_16 = t_16 / 16 * 32 + t_16 % 16;
        int high_16 = low_16 + 16;
        unsigned long long a_16 = pairs[low_16];
        unsigned long long b_16 = pairs[high_16];
        int dir_bit_16 = low_16 & 64;
        if (dir_bit_16 == 0) {
            if (a_16 < b_16) {
                pairs[low_16] = b_16;
                pairs[high_16] = a_16;
            }
        } else if (a_16 > b_16) {
            pairs[low_16] = b_16;
            pairs[high_16] = a_16;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_17 = tid; t_17 < 256; t_17 += 256) {
        int low_17 = t_17 / 8 * 16 + t_17 % 8;
        int high_17 = low_17 + 8;
        unsigned long long a_17 = pairs[low_17];
        unsigned long long b_17 = pairs[high_17];
        int dir_bit_17 = low_17 & 64;
        if (dir_bit_17 == 0) {
            if (a_17 < b_17) {
                pairs[low_17] = b_17;
                pairs[high_17] = a_17;
            }
        } else if (a_17 > b_17) {
            pairs[low_17] = b_17;
            pairs[high_17] = a_17;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_18 = tid; t_18 < 256; t_18 += 256) {
        int low_18 = t_18 / 4 * 8 + t_18 % 4;
        int high_18 = low_18 + 4;
        unsigned long long a_18 = pairs[low_18];
        unsigned long long b_18 = pairs[high_18];
        int dir_bit_18 = low_18 & 64;
        if (dir_bit_18 == 0) {
            if (a_18 < b_18) {
                pairs[low_18] = b_18;
                pairs[high_18] = a_18;
            }
        } else if (a_18 > b_18) {
            pairs[low_18] = b_18;
            pairs[high_18] = a_18;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_19 = tid; t_19 < 256; t_19 += 256) {
        int low_19 = t_19 / 2 * 4 + t_19 % 2;
        int high_19 = low_19 + 2;
        unsigned long long a_19 = pairs[low_19];
        unsigned long long b_19 = pairs[high_19];
        int dir_bit_19 = low_19 & 64;
        if (dir_bit_19 == 0) {
            if (a_19 < b_19) {
                pairs[low_19] = b_19;
                pairs[high_19] = a_19;
            }
        } else if (a_19 > b_19) {
            pairs[low_19] = b_19;
            pairs[high_19] = a_19;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_20 = tid; t_20 < 256; t_20 += 256) {
        int low_20 = t_20 * 2;
        int high_20 = low_20 + 1;
        unsigned long long a_20 = pairs[low_20];
        unsigned long long b_20 = pairs[high_20];
        int dir_bit_20 = low_20 & 64;
        if (dir_bit_20 == 0) {
            if (a_20 < b_20) {
                pairs[low_20] = b_20;
                pairs[high_20] = a_20;
            }
        } else if (a_20 > b_20) {
            pairs[low_20] = b_20;
            pairs[high_20] = a_20;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_21 = tid; t_21 < 256; t_21 += 256) {
        int low_21 = t_21 / 64 * 128 + t_21 % 64;
        int high_21 = low_21 + 64;
        unsigned long long a_21 = pairs[low_21];
        unsigned long long b_21 = pairs[high_21];
        int dir_bit_21 = low_21 & 128;
        if (dir_bit_21 == 0) {
            if (a_21 < b_21) {
                pairs[low_21] = b_21;
                pairs[high_21] = a_21;
            }
        } else if (a_21 > b_21) {
            pairs[low_21] = b_21;
            pairs[high_21] = a_21;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_22 = tid; t_22 < 256; t_22 += 256) {
        int low_22 = t_22 / 32 * 64 + t_22 % 32;
        int high_22 = low_22 + 32;
        unsigned long long a_22 = pairs[low_22];
        unsigned long long b_22 = pairs[high_22];
        int dir_bit_22 = low_22 & 128;
        if (dir_bit_22 == 0) {
            if (a_22 < b_22) {
                pairs[low_22] = b_22;
                pairs[high_22] = a_22;
            }
        } else if (a_22 > b_22) {
            pairs[low_22] = b_22;
            pairs[high_22] = a_22;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_23 = tid; t_23 < 256; t_23 += 256) {
        int low_23 = t_23 / 16 * 32 + t_23 % 16;
        int high_23 = low_23 + 16;
        unsigned long long a_23 = pairs[low_23];
        unsigned long long b_23 = pairs[high_23];
        int dir_bit_23 = low_23 & 128;
        if (dir_bit_23 == 0) {
            if (a_23 < b_23) {
                pairs[low_23] = b_23;
                pairs[high_23] = a_23;
            }
        } else if (a_23 > b_23) {
            pairs[low_23] = b_23;
            pairs[high_23] = a_23;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_24 = tid; t_24 < 256; t_24 += 256) {
        int low_24 = t_24 / 8 * 16 + t_24 % 8;
        int high_24 = low_24 + 8;
        unsigned long long a_24 = pairs[low_24];
        unsigned long long b_24 = pairs[high_24];
        int dir_bit_24 = low_24 & 128;
        if (dir_bit_24 == 0) {
            if (a_24 < b_24) {
                pairs[low_24] = b_24;
                pairs[high_24] = a_24;
            }
        } else if (a_24 > b_24) {
            pairs[low_24] = b_24;
            pairs[high_24] = a_24;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_25 = tid; t_25 < 256; t_25 += 256) {
        int low_25 = t_25 / 4 * 8 + t_25 % 4;
        int high_25 = low_25 + 4;
        unsigned long long a_25 = pairs[low_25];
        unsigned long long b_25 = pairs[high_25];
        int dir_bit_25 = low_25 & 128;
        if (dir_bit_25 == 0) {
            if (a_25 < b_25) {
                pairs[low_25] = b_25;
                pairs[high_25] = a_25;
            }
        } else if (a_25 > b_25) {
            pairs[low_25] = b_25;
            pairs[high_25] = a_25;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_26 = tid; t_26 < 256; t_26 += 256) {
        int low_26 = t_26 / 2 * 4 + t_26 % 2;
        int high_26 = low_26 + 2;
        unsigned long long a_26 = pairs[low_26];
        unsigned long long b_26 = pairs[high_26];
        int dir_bit_26 = low_26 & 128;
        if (dir_bit_26 == 0) {
            if (a_26 < b_26) {
                pairs[low_26] = b_26;
                pairs[high_26] = a_26;
            }
        } else if (a_26 > b_26) {
            pairs[low_26] = b_26;
            pairs[high_26] = a_26;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_27 = tid; t_27 < 256; t_27 += 256) {
        int low_27 = t_27 * 2;
        int high_27 = low_27 + 1;
        unsigned long long a_27 = pairs[low_27];
        unsigned long long b_27 = pairs[high_27];
        int dir_bit_27 = low_27 & 128;
        if (dir_bit_27 == 0) {
            if (a_27 < b_27) {
                pairs[low_27] = b_27;
                pairs[high_27] = a_27;
            }
        } else if (a_27 > b_27) {
            pairs[low_27] = b_27;
            pairs[high_27] = a_27;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_28 = tid; t_28 < 256; t_28 += 256) {
        int low_28 = t_28 / 128 * 256 + t_28 % 128;
        int high_28 = low_28 + 128;
        unsigned long long a_28 = pairs[low_28];
        unsigned long long b_28 = pairs[high_28];
        int dir_bit_28 = low_28 & 256;
        if (dir_bit_28 == 0) {
            if (a_28 < b_28) {
                pairs[low_28] = b_28;
                pairs[high_28] = a_28;
            }
        } else if (a_28 > b_28) {
            pairs[low_28] = b_28;
            pairs[high_28] = a_28;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_29 = tid; t_29 < 256; t_29 += 256) {
        int low_29 = t_29 / 64 * 128 + t_29 % 64;
        int high_29 = low_29 + 64;
        unsigned long long a_29 = pairs[low_29];
        unsigned long long b_29 = pairs[high_29];
        int dir_bit_29 = low_29 & 256;
        if (dir_bit_29 == 0) {
            if (a_29 < b_29) {
                pairs[low_29] = b_29;
                pairs[high_29] = a_29;
            }
        } else if (a_29 > b_29) {
            pairs[low_29] = b_29;
            pairs[high_29] = a_29;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_30 = tid; t_30 < 256; t_30 += 256) {
        int low_30 = t_30 / 32 * 64 + t_30 % 32;
        int high_30 = low_30 + 32;
        unsigned long long a_30 = pairs[low_30];
        unsigned long long b_30 = pairs[high_30];
        int dir_bit_30 = low_30 & 256;
        if (dir_bit_30 == 0) {
            if (a_30 < b_30) {
                pairs[low_30] = b_30;
                pairs[high_30] = a_30;
            }
        } else if (a_30 > b_30) {
            pairs[low_30] = b_30;
            pairs[high_30] = a_30;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_31 = tid; t_31 < 256; t_31 += 256) {
        int low_31 = t_31 / 16 * 32 + t_31 % 16;
        int high_31 = low_31 + 16;
        unsigned long long a_31 = pairs[low_31];
        unsigned long long b_31 = pairs[high_31];
        int dir_bit_31 = low_31 & 256;
        if (dir_bit_31 == 0) {
            if (a_31 < b_31) {
                pairs[low_31] = b_31;
                pairs[high_31] = a_31;
            }
        } else if (a_31 > b_31) {
            pairs[low_31] = b_31;
            pairs[high_31] = a_31;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_32 = tid; t_32 < 256; t_32 += 256) {
        int low_32 = t_32 / 8 * 16 + t_32 % 8;
        int high_32 = low_32 + 8;
        unsigned long long a_32 = pairs[low_32];
        unsigned long long b_32 = pairs[high_32];
        int dir_bit_32 = low_32 & 256;
        if (dir_bit_32 == 0) {
            if (a_32 < b_32) {
                pairs[low_32] = b_32;
                pairs[high_32] = a_32;
            }
        } else if (a_32 > b_32) {
            pairs[low_32] = b_32;
            pairs[high_32] = a_32;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_33 = tid; t_33 < 256; t_33 += 256) {
        int low_33 = t_33 / 4 * 8 + t_33 % 4;
        int high_33 = low_33 + 4;
        unsigned long long a_33 = pairs[low_33];
        unsigned long long b_33 = pairs[high_33];
        int dir_bit_33 = low_33 & 256;
        if (dir_bit_33 == 0) {
            if (a_33 < b_33) {
                pairs[low_33] = b_33;
                pairs[high_33] = a_33;
            }
        } else if (a_33 > b_33) {
            pairs[low_33] = b_33;
            pairs[high_33] = a_33;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_34 = tid; t_34 < 256; t_34 += 256) {
        int low_34 = t_34 / 2 * 4 + t_34 % 2;
        int high_34 = low_34 + 2;
        unsigned long long a_34 = pairs[low_34];
        unsigned long long b_34 = pairs[high_34];
        int dir_bit_34 = low_34 & 256;
        if (dir_bit_34 == 0) {
            if (a_34 < b_34) {
                pairs[low_34] = b_34;
                pairs[high_34] = a_34;
            }
        } else if (a_34 > b_34) {
            pairs[low_34] = b_34;
            pairs[high_34] = a_34;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_35 = tid; t_35 < 256; t_35 += 256) {
        int low_35 = t_35 * 2;
        int high_35 = low_35 + 1;
        unsigned long long a_35 = pairs[low_35];
        unsigned long long b_35 = pairs[high_35];
        int dir_bit_35 = low_35 & 256;
        if (dir_bit_35 == 0) {
            if (a_35 < b_35) {
                pairs[low_35] = b_35;
                pairs[high_35] = a_35;
            }
        } else if (a_35 > b_35) {
            pairs[low_35] = b_35;
            pairs[high_35] = a_35;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_36 = tid; t_36 < 256; t_36 += 256) {
        int low_36 = t_36 / 256 * 512 + t_36 % 256;
        int high_36 = low_36 + 256;
        unsigned long long a_36 = pairs[low_36];
        unsigned long long b_36 = pairs[high_36];
        int dir_bit_36 = low_36 & 512;
        if (dir_bit_36 == 0) {
            if (a_36 < b_36) {
                pairs[low_36] = b_36;
                pairs[high_36] = a_36;
            }
        } else if (a_36 > b_36) {
            pairs[low_36] = b_36;
            pairs[high_36] = a_36;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_37 = tid; t_37 < 256; t_37 += 256) {
        int low_37 = t_37 / 128 * 256 + t_37 % 128;
        int high_37 = low_37 + 128;
        unsigned long long a_37 = pairs[low_37];
        unsigned long long b_37 = pairs[high_37];
        int dir_bit_37 = low_37 & 512;
        if (dir_bit_37 == 0) {
            if (a_37 < b_37) {
                pairs[low_37] = b_37;
                pairs[high_37] = a_37;
            }
        } else if (a_37 > b_37) {
            pairs[low_37] = b_37;
            pairs[high_37] = a_37;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_38 = tid; t_38 < 256; t_38 += 256) {
        int low_38 = t_38 / 64 * 128 + t_38 % 64;
        int high_38 = low_38 + 64;
        unsigned long long a_38 = pairs[low_38];
        unsigned long long b_38 = pairs[high_38];
        int dir_bit_38 = low_38 & 512;
        if (dir_bit_38 == 0) {
            if (a_38 < b_38) {
                pairs[low_38] = b_38;
                pairs[high_38] = a_38;
            }
        } else if (a_38 > b_38) {
            pairs[low_38] = b_38;
            pairs[high_38] = a_38;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_39 = tid; t_39 < 256; t_39 += 256) {
        int low_39 = t_39 / 32 * 64 + t_39 % 32;
        int high_39 = low_39 + 32;
        unsigned long long a_39 = pairs[low_39];
        unsigned long long b_39 = pairs[high_39];
        int dir_bit_39 = low_39 & 512;
        if (dir_bit_39 == 0) {
            if (a_39 < b_39) {
                pairs[low_39] = b_39;
                pairs[high_39] = a_39;
            }
        } else if (a_39 > b_39) {
            pairs[low_39] = b_39;
            pairs[high_39] = a_39;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_40 = tid; t_40 < 256; t_40 += 256) {
        int low_40 = t_40 / 16 * 32 + t_40 % 16;
        int high_40 = low_40 + 16;
        unsigned long long a_40 = pairs[low_40];
        unsigned long long b_40 = pairs[high_40];
        int dir_bit_40 = low_40 & 512;
        if (dir_bit_40 == 0) {
            if (a_40 < b_40) {
                pairs[low_40] = b_40;
                pairs[high_40] = a_40;
            }
        } else if (a_40 > b_40) {
            pairs[low_40] = b_40;
            pairs[high_40] = a_40;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_41 = tid; t_41 < 256; t_41 += 256) {
        int low_41 = t_41 / 8 * 16 + t_41 % 8;
        int high_41 = low_41 + 8;
        unsigned long long a_41 = pairs[low_41];
        unsigned long long b_41 = pairs[high_41];
        int dir_bit_41 = low_41 & 512;
        if (dir_bit_41 == 0) {
            if (a_41 < b_41) {
                pairs[low_41] = b_41;
                pairs[high_41] = a_41;
            }
        } else if (a_41 > b_41) {
            pairs[low_41] = b_41;
            pairs[high_41] = a_41;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_42 = tid; t_42 < 256; t_42 += 256) {
        int low_42 = t_42 / 4 * 8 + t_42 % 4;
        int high_42 = low_42 + 4;
        unsigned long long a_42 = pairs[low_42];
        unsigned long long b_42 = pairs[high_42];
        int dir_bit_42 = low_42 & 512;
        if (dir_bit_42 == 0) {
            if (a_42 < b_42) {
                pairs[low_42] = b_42;
                pairs[high_42] = a_42;
            }
        } else if (a_42 > b_42) {
            pairs[low_42] = b_42;
            pairs[high_42] = a_42;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_43 = tid; t_43 < 256; t_43 += 256) {
        int low_43 = t_43 / 2 * 4 + t_43 % 2;
        int high_43 = low_43 + 2;
        unsigned long long a_43 = pairs[low_43];
        unsigned long long b_43 = pairs[high_43];
        int dir_bit_43 = low_43 & 512;
        if (dir_bit_43 == 0) {
            if (a_43 < b_43) {
                pairs[low_43] = b_43;
                pairs[high_43] = a_43;
            }
        } else if (a_43 > b_43) {
            pairs[low_43] = b_43;
            pairs[high_43] = a_43;
        }
    }
    __syncthreads();
    #pragma unroll 1
    for (int t_44 = tid; t_44 < 256; t_44 += 256) {
        int low_44 = t_44 * 2;
        int high_44 = low_44 + 1;
        unsigned long long a_44 = pairs[low_44];
        unsigned long long b_44 = pairs[high_44];
        int dir_bit_44 = low_44 & 512;
        if (dir_bit_44 == 0) {
            if (a_44 < b_44) {
                pairs[low_44] = b_44;
                pairs[high_44] = a_44;
            }
        } else if (a_44 > b_44) {
            pairs[low_44] = b_44;
            pairs[high_44] = a_44;
        }
    }
    __syncthreads();
    unsigned long long out_base = (unsigned long long)row * 512;
    #pragma unroll 1
    for (int o = tid; o < 512; o += 256) {
        unsigned long long comp = pairs[o];
        unsigned int key_hi = (unsigned int)(comp >> 32);
        unsigned int inv_lo = (unsigned int)(comp & 4294967295);
        int index = (int)(inv_lo ^ 4294967295u);
        int value = ((key_hi != 0) ? index : -1);
        out[out_base + (unsigned long long)o] = value;
    }
}

} // extern "C"
