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
#define SMEM_WARP_SUMS_STAGE_BYTES 128
#define SMEM_WARP_SUMS_STRIDE 128
#define SMEM_STATE_OFF 75008
#define SMEM_STATE_STAGE_BYTES 16
#define SMEM_STATE_STRIDE 16
#define SMEM_PAIRS_B_OFF 0
#define SMEM_PAIRS_B_STAGE_BYTES 4096
#define SMEM_PAIRS_B_STRIDE 4096
#define SMEM_TOTAL 75136
#define THREADS 512

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
kernel_cake_dsa_train_0a0ca4f0b13765095fc5(float* __restrict__ logits, int* __restrict__ out, int* __restrict__ row_end, int n_kv, int has_row_end)
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
    unsigned long long* pairs_b = reinterpret_cast<unsigned long long*>(smem_raw + SMEM_PAIRS_B_OFF);
    const int pairs_b_addr = smem + SMEM_PAIRS_B_OFF;

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
    int n_chunks_rt = (limit + 511) / 512;
    int stage_end = n_chunks_rt * 512;
    int n_slots_rt = n_chunks_rt * 16;
    int hidx = tid;
    if (hidx < 256) {
        hist[hidx] = 0;
    }
    if (tid == 0) {
        state[2] = 0;
    }
    #pragma unroll 1
    for (int s = tid; s < 512; s += 512) {
        pairs[s] = 0;
    }
    int last_sel = -1;
    #pragma unroll 4
    for (int idx = tid; idx < stage_end; idx += 512) {
        unsigned int key = 0;
        if (limit > idx) {
            float x = logits[row_base + (unsigned long long)idx];
            unsigned int bits = __as_u32(x);
            bits = ((bits == 2147483648u) ? 0 : bits);
            unsigned int sign_mask = (unsigned int)((int)bits >> 31);
            unsigned int enc = bits ^ (sign_mask | 2147483648u);
            bool selectable = enc - 8388608 <= 4278190080u;
            key = ((selectable) ? enc : 0);
            last_sel = ((selectable) ? idx : last_sel);
        }
        keys[idx] = key;
    }
    int warp_last = last_sel;
    int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, warp_last, 1);
    int other_last = _shfl_xor_0;
    warp_last = ((other_last > warp_last) ? other_last : warp_last);
    int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, warp_last, 2);
    int other_last_0 = _shfl_xor_1;
    warp_last = ((other_last_0 > warp_last) ? other_last_0 : warp_last);
    int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, warp_last, 4);
    int other_last_1 = _shfl_xor_2;
    warp_last = ((other_last_1 > warp_last) ? other_last_1 : warp_last);
    int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, warp_last, 8);
    int other_last_2 = _shfl_xor_3;
    warp_last = ((other_last_2 > warp_last) ? other_last_2 : warp_last);
    int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, warp_last, 16);
    int other_last_3 = _shfl_xor_4;
    warp_last = ((other_last_3 > warp_last) ? other_last_3 : warp_last);
    if (lane == 0) {
        warp_sums[warp] = (unsigned int)(warp_last + 1);
    }
    __syncthreads();
    int limit_eff = 0;
    int wl = (int)warp_sums[0];
    limit_eff = ((wl > limit_eff) ? wl : limit_eff);
    int wl_4 = (int)warp_sums[1];
    limit_eff = ((wl_4 > limit_eff) ? wl_4 : limit_eff);
    int wl_5 = (int)warp_sums[2];
    limit_eff = ((wl_5 > limit_eff) ? wl_5 : limit_eff);
    int wl_6 = (int)warp_sums[3];
    limit_eff = ((wl_6 > limit_eff) ? wl_6 : limit_eff);
    int wl_7 = (int)warp_sums[4];
    limit_eff = ((wl_7 > limit_eff) ? wl_7 : limit_eff);
    int wl_8 = (int)warp_sums[5];
    limit_eff = ((wl_8 > limit_eff) ? wl_8 : limit_eff);
    int wl_9 = (int)warp_sums[6];
    limit_eff = ((wl_9 > limit_eff) ? wl_9 : limit_eff);
    int wl_10 = (int)warp_sums[7];
    limit_eff = ((wl_10 > limit_eff) ? wl_10 : limit_eff);
    int wl_11 = (int)warp_sums[8];
    limit_eff = ((wl_11 > limit_eff) ? wl_11 : limit_eff);
    int wl_12 = (int)warp_sums[9];
    limit_eff = ((wl_12 > limit_eff) ? wl_12 : limit_eff);
    int wl_13 = (int)warp_sums[10];
    limit_eff = ((wl_13 > limit_eff) ? wl_13 : limit_eff);
    int wl_14 = (int)warp_sums[11];
    limit_eff = ((wl_14 > limit_eff) ? wl_14 : limit_eff);
    int wl_15 = (int)warp_sums[12];
    limit_eff = ((wl_15 > limit_eff) ? wl_15 : limit_eff);
    int wl_16 = (int)warp_sums[13];
    limit_eff = ((wl_16 > limit_eff) ? wl_16 : limit_eff);
    int wl_17 = (int)warp_sums[14];
    limit_eff = ((wl_17 > limit_eff) ? wl_17 : limit_eff);
    int wl_18 = (int)warp_sums[15];
    limit_eff = ((wl_18 > limit_eff) ? wl_18 : limit_eff);
    n_chunks_rt = (limit_eff + 511) / 512;
    stage_end = n_chunks_rt * 512;
    n_slots_rt = n_chunks_rt * 16;
    #pragma unroll 4
    for (int idx0 = tid; idx0 < stage_end; idx0 += 512) {
        unsigned int key0 = keys[idx0];
        if (key0 != 0) {
            atomicAdd(&hist[key0 >> 24], 1);
        }
    }
    __syncthreads();
    if (warp == 0) {
        unsigned int bins[8];
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
        unsigned int at = running + bins[7];
        if (running < 512) {
            if (at >= 512) {
                hit = (unsigned int)(lane * 8 + 7) << 16 | 512 - running;
                if (at == 512) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at;
        unsigned int at_11 = running + bins[6];
        if (running < 512) {
            if (at_11 >= 512) {
                hit = (unsigned int)(lane * 8 + 6) << 16 | 512 - running;
                if (at_11 == 512) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_11;
        unsigned int at_12 = running + bins[5];
        if (running < 512) {
            if (at_12 >= 512) {
                hit = (unsigned int)(lane * 8 + 5) << 16 | 512 - running;
                if (at_12 == 512) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_12;
        unsigned int at_13 = running + bins[4];
        if (running < 512) {
            if (at_13 >= 512) {
                hit = (unsigned int)(lane * 8 + 4) << 16 | 512 - running;
                if (at_13 == 512) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_13;
        unsigned int at_14 = running + bins[3];
        if (running < 512) {
            if (at_14 >= 512) {
                hit = (unsigned int)(lane * 8 + 3) << 16 | 512 - running;
                if (at_14 == 512) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_14;
        unsigned int at_15 = running + bins[2];
        if (running < 512) {
            if (at_15 >= 512) {
                hit = (unsigned int)(lane * 8 + 2) << 16 | 512 - running;
                if (at_15 == 512) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_15;
        unsigned int at_16 = running + bins[1];
        if (running < 512) {
            if (at_16 >= 512) {
                hit = (unsigned int)(lane * 8 + 1) << 16 | 512 - running;
                if (at_16 == 512) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_16;
        unsigned int at_17 = running + bins[0];
        if (running < 512) {
            if (at_17 >= 512) {
                hit = (unsigned int)(lane * 8) << 16 | 512 - running;
                if (at_17 == 512) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_17;
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
            state[1] = above_incl;
        }
    }
    __syncthreads();
    unsigned int chosen0 = state[0];
    unsigned int total = state[1];
    unsigned int pivot = 0;
    unsigned int quota = 0;
    unsigned int count_gt = 0;
    unsigned int prefix = 0;
    unsigned int remaining = 512;
    unsigned int done = 0;
    if (total <= 512) {
        count_gt = total;
        done = 1;
    } else {
        prefix = (chosen0 >> 16 & 255) << 24;
        remaining = chosen0 & 65535;
        if ((chosen0 & 2147483648u) != 0) {
            if (prefix != 0) {
                pivot = prefix - 1;
                count_gt = 512;
                done = 1;
            }
        }
    }
    unsigned int prefix_hi = prefix >> 24;
    if (done == 0) {
        #pragma unroll 4
        for (int idx1 = tid; idx1 < stage_end; idx1 += 512) {
            unsigned int key1 = keys[idx1];
            if (key1 != 0) {
                if (key1 >> 24 == prefix_hi) {
                    atomicAdd(&hist[key1 >> 16 & 255], 1);
                }
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        unsigned int bins_1[8];
        unsigned int lane_total_1 = 0;
        unsigned int hb_7 = hist[lane * 8];
        bins_1[0] = hb_7;
        lane_total_1 = lane_total_1 + hb_7;
        unsigned int hb_0_1 = hist[lane * 8 + 1];
        bins_1[1] = hb_0_1;
        lane_total_1 = lane_total_1 + hb_0_1;
        unsigned int hb_1_1 = hist[lane * 8 + 2];
        bins_1[2] = hb_1_1;
        lane_total_1 = lane_total_1 + hb_1_1;
        unsigned int hb_2_1 = hist[lane * 8 + 3];
        bins_1[3] = hb_2_1;
        lane_total_1 = lane_total_1 + hb_2_1;
        unsigned int hb_3_1 = hist[lane * 8 + 4];
        bins_1[4] = hb_3_1;
        lane_total_1 = lane_total_1 + hb_3_1;
        unsigned int hb_4_1 = hist[lane * 8 + 5];
        bins_1[5] = hb_4_1;
        lane_total_1 = lane_total_1 + hb_4_1;
        unsigned int hb_5_1 = hist[lane * 8 + 6];
        bins_1[6] = hb_5_1;
        lane_total_1 = lane_total_1 + hb_5_1;
        unsigned int hb_6_1 = hist[lane * 8 + 7];
        bins_1[7] = hb_6_1;
        lane_total_1 = lane_total_1 + hb_6_1;
        unsigned int above_incl_1 = lane_total_1;
        unsigned int _shfl_down_5 = __shfl_down_sync(0xFFFFFFFF, above_incl_1, 1, 32);
        unsigned int nxt_1 = _shfl_down_5;
        if (lane + 1 < 32) {
            above_incl_1 = above_incl_1 + nxt_1;
        }
        unsigned int _shfl_down_6 = __shfl_down_sync(0xFFFFFFFF, above_incl_1, 2, 32);
        unsigned int nxt_7_1 = _shfl_down_6;
        if (lane + 2 < 32) {
            above_incl_1 = above_incl_1 + nxt_7_1;
        }
        unsigned int _shfl_down_7 = __shfl_down_sync(0xFFFFFFFF, above_incl_1, 4, 32);
        unsigned int nxt_8_1 = _shfl_down_7;
        if (lane + 4 < 32) {
            above_incl_1 = above_incl_1 + nxt_8_1;
        }
        unsigned int _shfl_down_8 = __shfl_down_sync(0xFFFFFFFF, above_incl_1, 8, 32);
        unsigned int nxt_9_1 = _shfl_down_8;
        if (lane + 8 < 32) {
            above_incl_1 = above_incl_1 + nxt_9_1;
        }
        unsigned int _shfl_down_9 = __shfl_down_sync(0xFFFFFFFF, above_incl_1, 16, 32);
        unsigned int nxt_10_1 = _shfl_down_9;
        if (lane + 16 < 32) {
            above_incl_1 = above_incl_1 + nxt_10_1;
        }
        unsigned int running_1 = above_incl_1 - lane_total_1;
        unsigned int hit_1 = 4294967295u;
        unsigned int at_1 = running_1 + bins_1[7];
        if (running_1 < remaining) {
            if (at_1 >= remaining) {
                hit_1 = (unsigned int)(lane * 8 + 7) << 16 | remaining - running_1;
                if (at_1 == remaining) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_1;
        unsigned int at_11_1 = running_1 + bins_1[6];
        if (running_1 < remaining) {
            if (at_11_1 >= remaining) {
                hit_1 = (unsigned int)(lane * 8 + 6) << 16 | remaining - running_1;
                if (at_11_1 == remaining) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_11_1;
        unsigned int at_12_1 = running_1 + bins_1[5];
        if (running_1 < remaining) {
            if (at_12_1 >= remaining) {
                hit_1 = (unsigned int)(lane * 8 + 5) << 16 | remaining - running_1;
                if (at_12_1 == remaining) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_12_1;
        unsigned int at_13_1 = running_1 + bins_1[4];
        if (running_1 < remaining) {
            if (at_13_1 >= remaining) {
                hit_1 = (unsigned int)(lane * 8 + 4) << 16 | remaining - running_1;
                if (at_13_1 == remaining) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_13_1;
        unsigned int at_14_1 = running_1 + bins_1[3];
        if (running_1 < remaining) {
            if (at_14_1 >= remaining) {
                hit_1 = (unsigned int)(lane * 8 + 3) << 16 | remaining - running_1;
                if (at_14_1 == remaining) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_14_1;
        unsigned int at_15_1 = running_1 + bins_1[2];
        if (running_1 < remaining) {
            if (at_15_1 >= remaining) {
                hit_1 = (unsigned int)(lane * 8 + 2) << 16 | remaining - running_1;
                if (at_15_1 == remaining) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_15_1;
        unsigned int at_16_1 = running_1 + bins_1[1];
        if (running_1 < remaining) {
            if (at_16_1 >= remaining) {
                hit_1 = (unsigned int)(lane * 8 + 1) << 16 | remaining - running_1;
                if (at_16_1 == remaining) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_16_1;
        unsigned int at_17_1 = running_1 + bins_1[0];
        if (running_1 < remaining) {
            if (at_17_1 >= remaining) {
                hit_1 = (unsigned int)(lane * 8) << 16 | remaining - running_1;
                if (at_17_1 == remaining) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_17_1;
        unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, hit_1 != 4294967295u);
        unsigned int hit_mask_1 = _vote_1;
        int _ffs_1 = __ffs(hit_mask_1);
        unsigned int src_lane_1 = (unsigned int)(_ffs_1 - 1) & 31;
        unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, hit_1, src_lane_1);
        unsigned int chosen_1 = _shfl_1;
        hist[lane * 8] = 0;
        hist[lane * 8 + 1] = 0;
        hist[lane * 8 + 2] = 0;
        hist[lane * 8 + 3] = 0;
        hist[lane * 8 + 4] = 0;
        hist[lane * 8 + 5] = 0;
        hist[lane * 8 + 6] = 0;
        hist[lane * 8 + 7] = 0;
        if (lane == 0) {
            state[0] = chosen_1;
            state[1] = above_incl_1;
        }
    }
    __syncthreads();
    unsigned int chosen1 = state[0];
    if (done == 0) {
        prefix = prefix | (chosen1 >> 16 & 255) << 16;
        remaining = chosen1 & 65535;
        if ((chosen1 & 2147483648u) != 0) {
            if (prefix != 0) {
                pivot = prefix - 1;
                count_gt = 512;
                done = 1;
            }
        }
    }
    unsigned int prefix_hi_19 = prefix >> 16;
    if (done == 0) {
        #pragma unroll 4
        for (int idx1_1 = tid; idx1_1 < stage_end; idx1_1 += 512) {
            unsigned int key1_1 = keys[idx1_1];
            if (key1_1 != 0) {
                if (key1_1 >> 16 == prefix_hi_19) {
                    atomicAdd(&hist[key1_1 >> 8 & 255], 1);
                }
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        unsigned int bins_2[8];
        unsigned int lane_total_2 = 0;
        unsigned int hb_8 = hist[lane * 8];
        bins_2[0] = hb_8;
        lane_total_2 = lane_total_2 + hb_8;
        unsigned int hb_0_2 = hist[lane * 8 + 1];
        bins_2[1] = hb_0_2;
        lane_total_2 = lane_total_2 + hb_0_2;
        unsigned int hb_1_2 = hist[lane * 8 + 2];
        bins_2[2] = hb_1_2;
        lane_total_2 = lane_total_2 + hb_1_2;
        unsigned int hb_2_2 = hist[lane * 8 + 3];
        bins_2[3] = hb_2_2;
        lane_total_2 = lane_total_2 + hb_2_2;
        unsigned int hb_3_2 = hist[lane * 8 + 4];
        bins_2[4] = hb_3_2;
        lane_total_2 = lane_total_2 + hb_3_2;
        unsigned int hb_4_2 = hist[lane * 8 + 5];
        bins_2[5] = hb_4_2;
        lane_total_2 = lane_total_2 + hb_4_2;
        unsigned int hb_5_2 = hist[lane * 8 + 6];
        bins_2[6] = hb_5_2;
        lane_total_2 = lane_total_2 + hb_5_2;
        unsigned int hb_6_2 = hist[lane * 8 + 7];
        bins_2[7] = hb_6_2;
        lane_total_2 = lane_total_2 + hb_6_2;
        unsigned int above_incl_2 = lane_total_2;
        unsigned int _shfl_down_10 = __shfl_down_sync(0xFFFFFFFF, above_incl_2, 1, 32);
        unsigned int nxt_2 = _shfl_down_10;
        if (lane + 1 < 32) {
            above_incl_2 = above_incl_2 + nxt_2;
        }
        unsigned int _shfl_down_11 = __shfl_down_sync(0xFFFFFFFF, above_incl_2, 2, 32);
        unsigned int nxt_7_2 = _shfl_down_11;
        if (lane + 2 < 32) {
            above_incl_2 = above_incl_2 + nxt_7_2;
        }
        unsigned int _shfl_down_12 = __shfl_down_sync(0xFFFFFFFF, above_incl_2, 4, 32);
        unsigned int nxt_8_2 = _shfl_down_12;
        if (lane + 4 < 32) {
            above_incl_2 = above_incl_2 + nxt_8_2;
        }
        unsigned int _shfl_down_13 = __shfl_down_sync(0xFFFFFFFF, above_incl_2, 8, 32);
        unsigned int nxt_9_2 = _shfl_down_13;
        if (lane + 8 < 32) {
            above_incl_2 = above_incl_2 + nxt_9_2;
        }
        unsigned int _shfl_down_14 = __shfl_down_sync(0xFFFFFFFF, above_incl_2, 16, 32);
        unsigned int nxt_10_2 = _shfl_down_14;
        if (lane + 16 < 32) {
            above_incl_2 = above_incl_2 + nxt_10_2;
        }
        unsigned int running_2 = above_incl_2 - lane_total_2;
        unsigned int hit_2 = 4294967295u;
        unsigned int at_2 = running_2 + bins_2[7];
        if (running_2 < remaining) {
            if (at_2 >= remaining) {
                hit_2 = (unsigned int)(lane * 8 + 7) << 16 | remaining - running_2;
                if (at_2 == remaining) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_2;
        unsigned int at_11_2 = running_2 + bins_2[6];
        if (running_2 < remaining) {
            if (at_11_2 >= remaining) {
                hit_2 = (unsigned int)(lane * 8 + 6) << 16 | remaining - running_2;
                if (at_11_2 == remaining) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_11_2;
        unsigned int at_12_2 = running_2 + bins_2[5];
        if (running_2 < remaining) {
            if (at_12_2 >= remaining) {
                hit_2 = (unsigned int)(lane * 8 + 5) << 16 | remaining - running_2;
                if (at_12_2 == remaining) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_12_2;
        unsigned int at_13_2 = running_2 + bins_2[4];
        if (running_2 < remaining) {
            if (at_13_2 >= remaining) {
                hit_2 = (unsigned int)(lane * 8 + 4) << 16 | remaining - running_2;
                if (at_13_2 == remaining) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_13_2;
        unsigned int at_14_2 = running_2 + bins_2[3];
        if (running_2 < remaining) {
            if (at_14_2 >= remaining) {
                hit_2 = (unsigned int)(lane * 8 + 3) << 16 | remaining - running_2;
                if (at_14_2 == remaining) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_14_2;
        unsigned int at_15_2 = running_2 + bins_2[2];
        if (running_2 < remaining) {
            if (at_15_2 >= remaining) {
                hit_2 = (unsigned int)(lane * 8 + 2) << 16 | remaining - running_2;
                if (at_15_2 == remaining) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_15_2;
        unsigned int at_16_2 = running_2 + bins_2[1];
        if (running_2 < remaining) {
            if (at_16_2 >= remaining) {
                hit_2 = (unsigned int)(lane * 8 + 1) << 16 | remaining - running_2;
                if (at_16_2 == remaining) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_16_2;
        unsigned int at_17_2 = running_2 + bins_2[0];
        if (running_2 < remaining) {
            if (at_17_2 >= remaining) {
                hit_2 = (unsigned int)(lane * 8) << 16 | remaining - running_2;
                if (at_17_2 == remaining) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_17_2;
        unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, hit_2 != 4294967295u);
        unsigned int hit_mask_2 = _vote_2;
        int _ffs_2 = __ffs(hit_mask_2);
        unsigned int src_lane_2 = (unsigned int)(_ffs_2 - 1) & 31;
        unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, hit_2, src_lane_2);
        unsigned int chosen_2 = _shfl_2;
        hist[lane * 8] = 0;
        hist[lane * 8 + 1] = 0;
        hist[lane * 8 + 2] = 0;
        hist[lane * 8 + 3] = 0;
        hist[lane * 8 + 4] = 0;
        hist[lane * 8 + 5] = 0;
        hist[lane * 8 + 6] = 0;
        hist[lane * 8 + 7] = 0;
        if (lane == 0) {
            state[0] = chosen_2;
            state[1] = above_incl_2;
        }
    }
    __syncthreads();
    unsigned int chosen1_20 = state[0];
    if (done == 0) {
        prefix = prefix | (chosen1_20 >> 16 & 255) << 8;
        remaining = chosen1_20 & 65535;
        if ((chosen1_20 & 2147483648u) != 0) {
            if (prefix != 0) {
                pivot = prefix - 1;
                count_gt = 512;
                done = 1;
            }
        }
    }
    unsigned int prefix_hi_21 = prefix >> 8;
    if (done == 0) {
        #pragma unroll 4
        for (int idx1_2 = tid; idx1_2 < stage_end; idx1_2 += 512) {
            unsigned int key1_2 = keys[idx1_2];
            if (key1_2 != 0) {
                if (key1_2 >> 8 == prefix_hi_21) {
                    atomicAdd(&hist[key1_2 & 255], 1);
                }
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        unsigned int bins_3[8];
        unsigned int lane_total_3 = 0;
        unsigned int hb_9 = hist[lane * 8];
        bins_3[0] = hb_9;
        lane_total_3 = lane_total_3 + hb_9;
        unsigned int hb_0_3 = hist[lane * 8 + 1];
        bins_3[1] = hb_0_3;
        lane_total_3 = lane_total_3 + hb_0_3;
        unsigned int hb_1_3 = hist[lane * 8 + 2];
        bins_3[2] = hb_1_3;
        lane_total_3 = lane_total_3 + hb_1_3;
        unsigned int hb_2_3 = hist[lane * 8 + 3];
        bins_3[3] = hb_2_3;
        lane_total_3 = lane_total_3 + hb_2_3;
        unsigned int hb_3_3 = hist[lane * 8 + 4];
        bins_3[4] = hb_3_3;
        lane_total_3 = lane_total_3 + hb_3_3;
        unsigned int hb_4_3 = hist[lane * 8 + 5];
        bins_3[5] = hb_4_3;
        lane_total_3 = lane_total_3 + hb_4_3;
        unsigned int hb_5_3 = hist[lane * 8 + 6];
        bins_3[6] = hb_5_3;
        lane_total_3 = lane_total_3 + hb_5_3;
        unsigned int hb_6_3 = hist[lane * 8 + 7];
        bins_3[7] = hb_6_3;
        lane_total_3 = lane_total_3 + hb_6_3;
        unsigned int above_incl_3 = lane_total_3;
        unsigned int _shfl_down_15 = __shfl_down_sync(0xFFFFFFFF, above_incl_3, 1, 32);
        unsigned int nxt_3 = _shfl_down_15;
        if (lane + 1 < 32) {
            above_incl_3 = above_incl_3 + nxt_3;
        }
        unsigned int _shfl_down_16 = __shfl_down_sync(0xFFFFFFFF, above_incl_3, 2, 32);
        unsigned int nxt_7_3 = _shfl_down_16;
        if (lane + 2 < 32) {
            above_incl_3 = above_incl_3 + nxt_7_3;
        }
        unsigned int _shfl_down_17 = __shfl_down_sync(0xFFFFFFFF, above_incl_3, 4, 32);
        unsigned int nxt_8_3 = _shfl_down_17;
        if (lane + 4 < 32) {
            above_incl_3 = above_incl_3 + nxt_8_3;
        }
        unsigned int _shfl_down_18 = __shfl_down_sync(0xFFFFFFFF, above_incl_3, 8, 32);
        unsigned int nxt_9_3 = _shfl_down_18;
        if (lane + 8 < 32) {
            above_incl_3 = above_incl_3 + nxt_9_3;
        }
        unsigned int _shfl_down_19 = __shfl_down_sync(0xFFFFFFFF, above_incl_3, 16, 32);
        unsigned int nxt_10_3 = _shfl_down_19;
        if (lane + 16 < 32) {
            above_incl_3 = above_incl_3 + nxt_10_3;
        }
        unsigned int running_3 = above_incl_3 - lane_total_3;
        unsigned int hit_3 = 4294967295u;
        unsigned int at_3 = running_3 + bins_3[7];
        if (running_3 < remaining) {
            if (at_3 >= remaining) {
                hit_3 = (unsigned int)(lane * 8 + 7) << 16 | remaining - running_3;
                if (at_3 == remaining) {
                    hit_3 = hit_3 | 2147483648u;
                }
            }
        }
        running_3 = at_3;
        unsigned int at_11_3 = running_3 + bins_3[6];
        if (running_3 < remaining) {
            if (at_11_3 >= remaining) {
                hit_3 = (unsigned int)(lane * 8 + 6) << 16 | remaining - running_3;
                if (at_11_3 == remaining) {
                    hit_3 = hit_3 | 2147483648u;
                }
            }
        }
        running_3 = at_11_3;
        unsigned int at_12_3 = running_3 + bins_3[5];
        if (running_3 < remaining) {
            if (at_12_3 >= remaining) {
                hit_3 = (unsigned int)(lane * 8 + 5) << 16 | remaining - running_3;
                if (at_12_3 == remaining) {
                    hit_3 = hit_3 | 2147483648u;
                }
            }
        }
        running_3 = at_12_3;
        unsigned int at_13_3 = running_3 + bins_3[4];
        if (running_3 < remaining) {
            if (at_13_3 >= remaining) {
                hit_3 = (unsigned int)(lane * 8 + 4) << 16 | remaining - running_3;
                if (at_13_3 == remaining) {
                    hit_3 = hit_3 | 2147483648u;
                }
            }
        }
        running_3 = at_13_3;
        unsigned int at_14_3 = running_3 + bins_3[3];
        if (running_3 < remaining) {
            if (at_14_3 >= remaining) {
                hit_3 = (unsigned int)(lane * 8 + 3) << 16 | remaining - running_3;
                if (at_14_3 == remaining) {
                    hit_3 = hit_3 | 2147483648u;
                }
            }
        }
        running_3 = at_14_3;
        unsigned int at_15_3 = running_3 + bins_3[2];
        if (running_3 < remaining) {
            if (at_15_3 >= remaining) {
                hit_3 = (unsigned int)(lane * 8 + 2) << 16 | remaining - running_3;
                if (at_15_3 == remaining) {
                    hit_3 = hit_3 | 2147483648u;
                }
            }
        }
        running_3 = at_15_3;
        unsigned int at_16_3 = running_3 + bins_3[1];
        if (running_3 < remaining) {
            if (at_16_3 >= remaining) {
                hit_3 = (unsigned int)(lane * 8 + 1) << 16 | remaining - running_3;
                if (at_16_3 == remaining) {
                    hit_3 = hit_3 | 2147483648u;
                }
            }
        }
        running_3 = at_16_3;
        unsigned int at_17_3 = running_3 + bins_3[0];
        if (running_3 < remaining) {
            if (at_17_3 >= remaining) {
                hit_3 = (unsigned int)(lane * 8) << 16 | remaining - running_3;
                if (at_17_3 == remaining) {
                    hit_3 = hit_3 | 2147483648u;
                }
            }
        }
        running_3 = at_17_3;
        unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, hit_3 != 4294967295u);
        unsigned int hit_mask_3 = _vote_3;
        int _ffs_3 = __ffs(hit_mask_3);
        unsigned int src_lane_3 = (unsigned int)(_ffs_3 - 1) & 31;
        unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, hit_3, src_lane_3);
        unsigned int chosen_3 = _shfl_3;
        hist[lane * 8] = 0;
        hist[lane * 8 + 1] = 0;
        hist[lane * 8 + 2] = 0;
        hist[lane * 8 + 3] = 0;
        hist[lane * 8 + 4] = 0;
        hist[lane * 8 + 5] = 0;
        hist[lane * 8 + 6] = 0;
        hist[lane * 8 + 7] = 0;
        if (lane == 0) {
            state[0] = chosen_3;
            state[1] = above_incl_3;
        }
    }
    __syncthreads();
    unsigned int chosen1_22 = state[0];
    if (done == 0) {
        prefix = prefix | chosen1_22 >> 16 & 255;
        remaining = chosen1_22 & 65535;
        if ((chosen1_22 & 2147483648u) != 0) {
            if (prefix != 0) {
                pivot = prefix - 1;
                count_gt = 512;
                done = 1;
            }
        }
    }
    if (done == 0) {
        pivot = prefix;
        quota = remaining;
        if (pivot == 0) {
            quota = 0;
        }
        count_gt = 512 - remaining;
    }
    unsigned int lt_mask = ~(4294967295u << (unsigned int)lane);
    if (quota == 0) {
        #pragma unroll 1
        for (int cp = 0; cp < n_chunks_rt; cp++) {
            int idxp = cp * 512 + tid;
            unsigned int kp = keys[idxp];
            bool is_selp = kp > pivot;
            unsigned int _vote_4 = __ballot_sync(0xFFFFFFFF, is_selp);
            unsigned int sel_maskp = _vote_4;
            int _popc_0 = __popc(sel_maskp);
            unsigned int n_selp = (unsigned int)_popc_0;
            unsigned int _atomic_old_0 = 0;
            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.u32 %0, [%1], %2;\n}"
                : "+r"(_atomic_old_0) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&state[2]))), "r"(static_cast<unsigned int>(n_selp)), "r"(static_cast<uint32_t>((unsigned int)(lane == 0))) : "memory");
            unsigned int basep = _atomic_old_0;
            unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, basep, 0);
            basep = _shfl_4;
            if (is_selp) {
                unsigned int inv_idxp = (unsigned int)idxp ^ 4294967295u;
                unsigned long long compositep = (unsigned long long)kp << 32 | (unsigned long long)inv_idxp;
                int _popc_1 = __popc(sel_maskp & lt_mask);
                pairs[basep + (unsigned int)_popc_1] = compositep;
            }
        }
    } else {
        #pragma unroll 1
        for (int c = 0; c < n_chunks_rt; c++) {
            int idx2 = c * 512 + tid;
            unsigned int k2 = keys[idx2];
            unsigned int _vote_5 = __ballot_sync(0xFFFFFFFF, k2 > pivot);
            unsigned int sel_mask = _vote_5;
            unsigned int _vote_6 = __ballot_sync(0xFFFFFFFF, k2 == pivot);
            unsigned int tie_mask = _vote_6;
            if (lane == 0) {
                int slot = c * 16 + warp;
                int _popc_2 = __popc(sel_mask);
                cnt_sel[slot] = (unsigned int)_popc_2;
                int _popc_3 = __popc(tie_mask);
                cnt_tie[slot] = (unsigned int)_popc_3;
            }
        }
    }
    __syncthreads();
    unsigned int local_sel = 0;
    unsigned int local_tie = 0;
    unsigned int incl_sel = 0;
    unsigned int incl_tie = 0;
    if (quota != 0) {
        int f = tid;
        if (f < n_slots_rt) {
            local_sel = local_sel + cnt_sel[f];
            local_tie = local_tie + cnt_tie[f];
        }
        incl_sel = local_sel;
        incl_tie = local_tie;
        unsigned int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 1, 32);
        unsigned int up_sel = _shfl_up_0;
        unsigned int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 1, 32);
        unsigned int up_tie = _shfl_up_1;
        if (lane >= 1) {
            incl_sel = incl_sel + up_sel;
            incl_tie = incl_tie + up_tie;
        }
        unsigned int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 2, 32);
        unsigned int up_sel_0 = _shfl_up_2;
        unsigned int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 2, 32);
        unsigned int up_tie_1 = _shfl_up_3;
        if (lane >= 2) {
            incl_sel = incl_sel + up_sel_0;
            incl_tie = incl_tie + up_tie_1;
        }
        unsigned int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 4, 32);
        unsigned int up_sel_2 = _shfl_up_4;
        unsigned int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 4, 32);
        unsigned int up_tie_3 = _shfl_up_5;
        if (lane >= 4) {
            incl_sel = incl_sel + up_sel_2;
            incl_tie = incl_tie + up_tie_3;
        }
        unsigned int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 8, 32);
        unsigned int up_sel_4 = _shfl_up_6;
        unsigned int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 8, 32);
        unsigned int up_tie_5 = _shfl_up_7;
        if (lane >= 8) {
            incl_sel = incl_sel + up_sel_4;
            incl_tie = incl_tie + up_tie_5;
        }
        unsigned int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 16, 32);
        unsigned int up_sel_6 = _shfl_up_8;
        unsigned int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 16, 32);
        unsigned int up_tie_7 = _shfl_up_9;
        if (lane >= 16) {
            incl_sel = incl_sel + up_sel_6;
            incl_tie = incl_tie + up_tie_7;
        }
        if (lane == 31) {
            warp_sums[warp] = incl_sel;
            warp_sums[16 + warp] = incl_tie;
        }
    }
    __syncthreads();
    if (quota != 0) {
        unsigned int base_sel = 0;
        unsigned int base_tie = 0;
        unsigned int ws = warp_sums[0];
        unsigned int wt = warp_sums[16];
        if (warp > 0) {
            base_sel = base_sel + ws;
            base_tie = base_tie + wt;
        }
        unsigned int ws_0 = warp_sums[1];
        unsigned int wt_1 = warp_sums[17];
        if (warp > 1) {
            base_sel = base_sel + ws_0;
            base_tie = base_tie + wt_1;
        }
        unsigned int ws_2 = warp_sums[2];
        unsigned int wt_3 = warp_sums[18];
        if (warp > 2) {
            base_sel = base_sel + ws_2;
            base_tie = base_tie + wt_3;
        }
        unsigned int ws_4 = warp_sums[3];
        unsigned int wt_5 = warp_sums[19];
        if (warp > 3) {
            base_sel = base_sel + ws_4;
            base_tie = base_tie + wt_5;
        }
        unsigned int ws_6 = warp_sums[4];
        unsigned int wt_7 = warp_sums[20];
        if (warp > 4) {
            base_sel = base_sel + ws_6;
            base_tie = base_tie + wt_7;
        }
        unsigned int ws_8 = warp_sums[5];
        unsigned int wt_9 = warp_sums[21];
        if (warp > 5) {
            base_sel = base_sel + ws_8;
            base_tie = base_tie + wt_9;
        }
        unsigned int ws_10 = warp_sums[6];
        unsigned int wt_11 = warp_sums[22];
        if (warp > 6) {
            base_sel = base_sel + ws_10;
            base_tie = base_tie + wt_11;
        }
        unsigned int ws_12 = warp_sums[7];
        unsigned int wt_13 = warp_sums[23];
        if (warp > 7) {
            base_sel = base_sel + ws_12;
            base_tie = base_tie + wt_13;
        }
        unsigned int ws_14 = warp_sums[8];
        unsigned int wt_15 = warp_sums[24];
        if (warp > 8) {
            base_sel = base_sel + ws_14;
            base_tie = base_tie + wt_15;
        }
        unsigned int ws_16 = warp_sums[9];
        unsigned int wt_17 = warp_sums[25];
        if (warp > 9) {
            base_sel = base_sel + ws_16;
            base_tie = base_tie + wt_17;
        }
        unsigned int ws_18 = warp_sums[10];
        unsigned int wt_19 = warp_sums[26];
        if (warp > 10) {
            base_sel = base_sel + ws_18;
            base_tie = base_tie + wt_19;
        }
        unsigned int ws_20 = warp_sums[11];
        unsigned int wt_21 = warp_sums[27];
        if (warp > 11) {
            base_sel = base_sel + ws_20;
            base_tie = base_tie + wt_21;
        }
        unsigned int ws_22 = warp_sums[12];
        unsigned int wt_23 = warp_sums[28];
        if (warp > 12) {
            base_sel = base_sel + ws_22;
            base_tie = base_tie + wt_23;
        }
        unsigned int ws_24 = warp_sums[13];
        unsigned int wt_25 = warp_sums[29];
        if (warp > 13) {
            base_sel = base_sel + ws_24;
            base_tie = base_tie + wt_25;
        }
        unsigned int ws_26 = warp_sums[14];
        unsigned int wt_27 = warp_sums[30];
        if (warp > 14) {
            base_sel = base_sel + ws_26;
            base_tie = base_tie + wt_27;
        }
        unsigned int ws_28 = warp_sums[15];
        unsigned int wt_29 = warp_sums[31];
        if (warp > 15) {
            base_sel = base_sel + ws_28;
            base_tie = base_tie + wt_29;
        }
        unsigned int run_sel = base_sel + incl_sel - local_sel;
        unsigned int run_tie = base_tie + incl_tie - local_tie;
        int f2 = tid;
        if (f2 < n_slots_rt) {
            unsigned int cur_sel = cnt_sel[f2];
            unsigned int cur_tie = cnt_tie[f2];
            cnt_sel[f2] = run_sel;
            cnt_tie[f2] = run_tie;
            run_sel = run_sel + cur_sel;
            run_tie = run_tie + cur_tie;
        }
    }
    __syncthreads();
    if (quota != 0) {
        #pragma unroll 1
        for (int c2 = 0; c2 < n_chunks_rt; c2++) {
            int idx3 = c2 * 512 + tid;
            unsigned int k3 = keys[idx3];
            bool is_sel = k3 > pivot;
            bool is_tie = k3 == pivot;
            unsigned int _vote_7 = __ballot_sync(0xFFFFFFFF, is_sel);
            unsigned int sel_mask2 = _vote_7;
            unsigned int _vote_8 = __ballot_sync(0xFFFFFFFF, is_tie);
            unsigned int tie_mask2 = _vote_8;
            int slot2 = c2 * 16 + warp;
            unsigned int inv_idx = (unsigned int)idx3 ^ 4294967295u;
            unsigned long long composite = (unsigned long long)k3 << 32 | (unsigned long long)inv_idx;
            if (is_sel) {
                int _popc_4 = __popc(sel_mask2 & lt_mask);
                unsigned int pos = cnt_sel[slot2] + (unsigned int)_popc_4;
                pairs[pos] = composite;
            } else if (is_tie) {
                int _popc_5 = __popc(tie_mask2 & lt_mask);
                unsigned int rank = cnt_tie[slot2] + (unsigned int)_popc_5;
                if (rank < quota) {
                    pairs[count_gt + rank] = composite;
                }
            }
        }
    }
    __syncthreads();
    unsigned long long vals[2];
    if (warp < 8) {
        int pos_l = warp * 64 + lane;
        vals[0] = pairs[pos_l];
        int pos_l_0 = warp * 64 + (32 + lane);
        vals[1] = pairs[pos_l_0];
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 1);
        unsigned long long other = _shfl_xor_5;
        unsigned int gt_s = (unsigned int)(other > vals[0]);
        int pos_s = warp * 64 + lane;
        unsigned int keep_max_s = (unsigned int)((pos_s & 1) == 0) ^ (unsigned int)((pos_s & 2) == 0) ^ 1;
        vals[0] = ((gt_s == keep_max_s) ? other : vals[0]);
        unsigned long long _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 1);
        unsigned long long other_0 = _shfl_xor_6;
        unsigned int gt_s_1 = (unsigned int)(other_0 > vals[1]);
        int pos_s_2 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3 = (unsigned int)((pos_s_2 & 1) == 0) ^ (unsigned int)((pos_s_2 & 2) == 0) ^ 1;
        vals[1] = ((gt_s_1 == keep_max_s_3) ? other_0 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 2);
        unsigned long long other_1 = _shfl_xor_7;
        unsigned int gt_s_2 = (unsigned int)(other_1 > vals[0]);
        int pos_s_1 = warp * 64 + lane;
        unsigned int keep_max_s_1 = (unsigned int)((pos_s_1 & 2) == 0) ^ (unsigned int)((pos_s_1 & 4) == 0) ^ 1;
        vals[0] = ((gt_s_2 == keep_max_s_1) ? other_1 : vals[0]);
        unsigned long long _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 2);
        unsigned long long other_0_1 = _shfl_xor_8;
        unsigned int gt_s_1_1 = (unsigned int)(other_0_1 > vals[1]);
        int pos_s_2_1 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_1 = (unsigned int)((pos_s_2_1 & 2) == 0) ^ (unsigned int)((pos_s_2_1 & 4) == 0) ^ 1;
        vals[1] = ((gt_s_1_1 == keep_max_s_3_1) ? other_0_1 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 1);
        unsigned long long other_2 = _shfl_xor_9;
        unsigned int gt_s_3 = (unsigned int)(other_2 > vals[0]);
        int pos_s_3 = warp * 64 + lane;
        unsigned int keep_max_s_2 = (unsigned int)((pos_s_3 & 1) == 0) ^ (unsigned int)((pos_s_3 & 4) == 0) ^ 1;
        vals[0] = ((gt_s_3 == keep_max_s_2) ? other_2 : vals[0]);
        unsigned long long _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 1);
        unsigned long long other_0_2 = _shfl_xor_10;
        unsigned int gt_s_1_2 = (unsigned int)(other_0_2 > vals[1]);
        int pos_s_2_2 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_2 = (unsigned int)((pos_s_2_2 & 1) == 0) ^ (unsigned int)((pos_s_2_2 & 4) == 0) ^ 1;
        vals[1] = ((gt_s_1_2 == keep_max_s_3_2) ? other_0_2 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 4);
        unsigned long long other_3 = _shfl_xor_11;
        unsigned int gt_s_4 = (unsigned int)(other_3 > vals[0]);
        int pos_s_4 = warp * 64 + lane;
        unsigned int keep_max_s_4 = (unsigned int)((pos_s_4 & 4) == 0) ^ (unsigned int)((pos_s_4 & 8) == 0) ^ 1;
        vals[0] = ((gt_s_4 == keep_max_s_4) ? other_3 : vals[0]);
        unsigned long long _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 4);
        unsigned long long other_0_3 = _shfl_xor_12;
        unsigned int gt_s_1_3 = (unsigned int)(other_0_3 > vals[1]);
        int pos_s_2_3 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_3 = (unsigned int)((pos_s_2_3 & 4) == 0) ^ (unsigned int)((pos_s_2_3 & 8) == 0) ^ 1;
        vals[1] = ((gt_s_1_3 == keep_max_s_3_3) ? other_0_3 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 2);
        unsigned long long other_4 = _shfl_xor_13;
        unsigned int gt_s_5 = (unsigned int)(other_4 > vals[0]);
        int pos_s_5 = warp * 64 + lane;
        unsigned int keep_max_s_5 = (unsigned int)((pos_s_5 & 2) == 0) ^ (unsigned int)((pos_s_5 & 8) == 0) ^ 1;
        vals[0] = ((gt_s_5 == keep_max_s_5) ? other_4 : vals[0]);
        unsigned long long _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 2);
        unsigned long long other_0_4 = _shfl_xor_14;
        unsigned int gt_s_1_4 = (unsigned int)(other_0_4 > vals[1]);
        int pos_s_2_4 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_4 = (unsigned int)((pos_s_2_4 & 2) == 0) ^ (unsigned int)((pos_s_2_4 & 8) == 0) ^ 1;
        vals[1] = ((gt_s_1_4 == keep_max_s_3_4) ? other_0_4 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 1);
        unsigned long long other_5 = _shfl_xor_15;
        unsigned int gt_s_6 = (unsigned int)(other_5 > vals[0]);
        int pos_s_6 = warp * 64 + lane;
        unsigned int keep_max_s_6 = (unsigned int)((pos_s_6 & 1) == 0) ^ (unsigned int)((pos_s_6 & 8) == 0) ^ 1;
        vals[0] = ((gt_s_6 == keep_max_s_6) ? other_5 : vals[0]);
        unsigned long long _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 1);
        unsigned long long other_0_5 = _shfl_xor_16;
        unsigned int gt_s_1_5 = (unsigned int)(other_0_5 > vals[1]);
        int pos_s_2_5 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_5 = (unsigned int)((pos_s_2_5 & 1) == 0) ^ (unsigned int)((pos_s_2_5 & 8) == 0) ^ 1;
        vals[1] = ((gt_s_1_5 == keep_max_s_3_5) ? other_0_5 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 8);
        unsigned long long other_6 = _shfl_xor_17;
        unsigned int gt_s_7 = (unsigned int)(other_6 > vals[0]);
        int pos_s_7 = warp * 64 + lane;
        unsigned int keep_max_s_7 = (unsigned int)((pos_s_7 & 8) == 0) ^ (unsigned int)((pos_s_7 & 16) == 0) ^ 1;
        vals[0] = ((gt_s_7 == keep_max_s_7) ? other_6 : vals[0]);
        unsigned long long _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 8);
        unsigned long long other_0_6 = _shfl_xor_18;
        unsigned int gt_s_1_6 = (unsigned int)(other_0_6 > vals[1]);
        int pos_s_2_6 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_6 = (unsigned int)((pos_s_2_6 & 8) == 0) ^ (unsigned int)((pos_s_2_6 & 16) == 0) ^ 1;
        vals[1] = ((gt_s_1_6 == keep_max_s_3_6) ? other_0_6 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 4);
        unsigned long long other_7 = _shfl_xor_19;
        unsigned int gt_s_8 = (unsigned int)(other_7 > vals[0]);
        int pos_s_8 = warp * 64 + lane;
        unsigned int keep_max_s_8 = (unsigned int)((pos_s_8 & 4) == 0) ^ (unsigned int)((pos_s_8 & 16) == 0) ^ 1;
        vals[0] = ((gt_s_8 == keep_max_s_8) ? other_7 : vals[0]);
        unsigned long long _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 4);
        unsigned long long other_0_7 = _shfl_xor_20;
        unsigned int gt_s_1_7 = (unsigned int)(other_0_7 > vals[1]);
        int pos_s_2_7 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_7 = (unsigned int)((pos_s_2_7 & 4) == 0) ^ (unsigned int)((pos_s_2_7 & 16) == 0) ^ 1;
        vals[1] = ((gt_s_1_7 == keep_max_s_3_7) ? other_0_7 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 2);
        unsigned long long other_8 = _shfl_xor_21;
        unsigned int gt_s_9 = (unsigned int)(other_8 > vals[0]);
        int pos_s_9 = warp * 64 + lane;
        unsigned int keep_max_s_9 = (unsigned int)((pos_s_9 & 2) == 0) ^ (unsigned int)((pos_s_9 & 16) == 0) ^ 1;
        vals[0] = ((gt_s_9 == keep_max_s_9) ? other_8 : vals[0]);
        unsigned long long _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 2);
        unsigned long long other_0_8 = _shfl_xor_22;
        unsigned int gt_s_1_8 = (unsigned int)(other_0_8 > vals[1]);
        int pos_s_2_8 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_8 = (unsigned int)((pos_s_2_8 & 2) == 0) ^ (unsigned int)((pos_s_2_8 & 16) == 0) ^ 1;
        vals[1] = ((gt_s_1_8 == keep_max_s_3_8) ? other_0_8 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 1);
        unsigned long long other_9 = _shfl_xor_23;
        unsigned int gt_s_10 = (unsigned int)(other_9 > vals[0]);
        int pos_s_10 = warp * 64 + lane;
        unsigned int keep_max_s_10 = (unsigned int)((pos_s_10 & 1) == 0) ^ (unsigned int)((pos_s_10 & 16) == 0) ^ 1;
        vals[0] = ((gt_s_10 == keep_max_s_10) ? other_9 : vals[0]);
        unsigned long long _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 1);
        unsigned long long other_0_9 = _shfl_xor_24;
        unsigned int gt_s_1_9 = (unsigned int)(other_0_9 > vals[1]);
        int pos_s_2_9 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_9 = (unsigned int)((pos_s_2_9 & 1) == 0) ^ (unsigned int)((pos_s_2_9 & 16) == 0) ^ 1;
        vals[1] = ((gt_s_1_9 == keep_max_s_3_9) ? other_0_9 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 16);
        unsigned long long other_10 = _shfl_xor_25;
        unsigned int gt_s_11 = (unsigned int)(other_10 > vals[0]);
        int pos_s_11 = warp * 64 + lane;
        unsigned int keep_max_s_11 = (unsigned int)((pos_s_11 & 16) == 0) ^ (unsigned int)((pos_s_11 & 32) == 0) ^ 1;
        vals[0] = ((gt_s_11 == keep_max_s_11) ? other_10 : vals[0]);
        unsigned long long _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 16);
        unsigned long long other_0_10 = _shfl_xor_26;
        unsigned int gt_s_1_10 = (unsigned int)(other_0_10 > vals[1]);
        int pos_s_2_10 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_10 = (unsigned int)((pos_s_2_10 & 16) == 0) ^ (unsigned int)((pos_s_2_10 & 32) == 0) ^ 1;
        vals[1] = ((gt_s_1_10 == keep_max_s_3_10) ? other_0_10 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 8);
        unsigned long long other_11 = _shfl_xor_27;
        unsigned int gt_s_12 = (unsigned int)(other_11 > vals[0]);
        int pos_s_12 = warp * 64 + lane;
        unsigned int keep_max_s_12 = (unsigned int)((pos_s_12 & 8) == 0) ^ (unsigned int)((pos_s_12 & 32) == 0) ^ 1;
        vals[0] = ((gt_s_12 == keep_max_s_12) ? other_11 : vals[0]);
        unsigned long long _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 8);
        unsigned long long other_0_11 = _shfl_xor_28;
        unsigned int gt_s_1_11 = (unsigned int)(other_0_11 > vals[1]);
        int pos_s_2_11 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_11 = (unsigned int)((pos_s_2_11 & 8) == 0) ^ (unsigned int)((pos_s_2_11 & 32) == 0) ^ 1;
        vals[1] = ((gt_s_1_11 == keep_max_s_3_11) ? other_0_11 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 4);
        unsigned long long other_12 = _shfl_xor_29;
        unsigned int gt_s_13 = (unsigned int)(other_12 > vals[0]);
        int pos_s_13 = warp * 64 + lane;
        unsigned int keep_max_s_13 = (unsigned int)((pos_s_13 & 4) == 0) ^ (unsigned int)((pos_s_13 & 32) == 0) ^ 1;
        vals[0] = ((gt_s_13 == keep_max_s_13) ? other_12 : vals[0]);
        unsigned long long _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 4);
        unsigned long long other_0_12 = _shfl_xor_30;
        unsigned int gt_s_1_12 = (unsigned int)(other_0_12 > vals[1]);
        int pos_s_2_12 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_12 = (unsigned int)((pos_s_2_12 & 4) == 0) ^ (unsigned int)((pos_s_2_12 & 32) == 0) ^ 1;
        vals[1] = ((gt_s_1_12 == keep_max_s_3_12) ? other_0_12 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 2);
        unsigned long long other_13 = _shfl_xor_31;
        unsigned int gt_s_14 = (unsigned int)(other_13 > vals[0]);
        int pos_s_14 = warp * 64 + lane;
        unsigned int keep_max_s_14 = (unsigned int)((pos_s_14 & 2) == 0) ^ (unsigned int)((pos_s_14 & 32) == 0) ^ 1;
        vals[0] = ((gt_s_14 == keep_max_s_14) ? other_13 : vals[0]);
        unsigned long long _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 2);
        unsigned long long other_0_13 = _shfl_xor_32;
        unsigned int gt_s_1_13 = (unsigned int)(other_0_13 > vals[1]);
        int pos_s_2_13 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_13 = (unsigned int)((pos_s_2_13 & 2) == 0) ^ (unsigned int)((pos_s_2_13 & 32) == 0) ^ 1;
        vals[1] = ((gt_s_1_13 == keep_max_s_3_13) ? other_0_13 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 1);
        unsigned long long other_14 = _shfl_xor_33;
        unsigned int gt_s_15 = (unsigned int)(other_14 > vals[0]);
        int pos_s_15 = warp * 64 + lane;
        unsigned int keep_max_s_15 = (unsigned int)((pos_s_15 & 1) == 0) ^ (unsigned int)((pos_s_15 & 32) == 0) ^ 1;
        vals[0] = ((gt_s_15 == keep_max_s_15) ? other_14 : vals[0]);
        unsigned long long _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 1);
        unsigned long long other_0_14 = _shfl_xor_34;
        unsigned int gt_s_1_14 = (unsigned int)(other_0_14 > vals[1]);
        int pos_s_2_14 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_14 = (unsigned int)((pos_s_2_14 & 1) == 0) ^ (unsigned int)((pos_s_2_14 & 32) == 0) ^ 1;
        vals[1] = ((gt_s_1_14 == keep_max_s_3_14) ? other_0_14 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long a_lo = vals[0];
        unsigned long long b_hi = vals[1];
        unsigned int gt_t = (unsigned int)(b_hi > a_lo);
        int pos_lo = warp * 64 + lane;
        unsigned int desc_t = (unsigned int)((pos_lo & 64) == 0);
        bool swap_t = gt_t == desc_t;
        vals[0] = ((swap_t) ? b_hi : a_lo);
        vals[1] = ((swap_t) ? a_lo : b_hi);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 16);
        unsigned long long other_15 = _shfl_xor_35;
        unsigned int gt_s_16 = (unsigned int)(other_15 > vals[0]);
        int pos_s_16 = warp * 64 + lane;
        unsigned int keep_max_s_16 = (unsigned int)((pos_s_16 & 16) == 0) ^ (unsigned int)((pos_s_16 & 64) == 0) ^ 1;
        vals[0] = ((gt_s_16 == keep_max_s_16) ? other_15 : vals[0]);
        unsigned long long _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 16);
        unsigned long long other_0_15 = _shfl_xor_36;
        unsigned int gt_s_1_15 = (unsigned int)(other_0_15 > vals[1]);
        int pos_s_2_15 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_15 = (unsigned int)((pos_s_2_15 & 16) == 0) ^ (unsigned int)((pos_s_2_15 & 64) == 0) ^ 1;
        vals[1] = ((gt_s_1_15 == keep_max_s_3_15) ? other_0_15 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 8);
        unsigned long long other_16 = _shfl_xor_37;
        unsigned int gt_s_17 = (unsigned int)(other_16 > vals[0]);
        int pos_s_17 = warp * 64 + lane;
        unsigned int keep_max_s_17 = (unsigned int)((pos_s_17 & 8) == 0) ^ (unsigned int)((pos_s_17 & 64) == 0) ^ 1;
        vals[0] = ((gt_s_17 == keep_max_s_17) ? other_16 : vals[0]);
        unsigned long long _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 8);
        unsigned long long other_0_16 = _shfl_xor_38;
        unsigned int gt_s_1_16 = (unsigned int)(other_0_16 > vals[1]);
        int pos_s_2_16 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_16 = (unsigned int)((pos_s_2_16 & 8) == 0) ^ (unsigned int)((pos_s_2_16 & 64) == 0) ^ 1;
        vals[1] = ((gt_s_1_16 == keep_max_s_3_16) ? other_0_16 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 4);
        unsigned long long other_17 = _shfl_xor_39;
        unsigned int gt_s_18 = (unsigned int)(other_17 > vals[0]);
        int pos_s_18 = warp * 64 + lane;
        unsigned int keep_max_s_18 = (unsigned int)((pos_s_18 & 4) == 0) ^ (unsigned int)((pos_s_18 & 64) == 0) ^ 1;
        vals[0] = ((gt_s_18 == keep_max_s_18) ? other_17 : vals[0]);
        unsigned long long _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 4);
        unsigned long long other_0_17 = _shfl_xor_40;
        unsigned int gt_s_1_17 = (unsigned int)(other_0_17 > vals[1]);
        int pos_s_2_17 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_17 = (unsigned int)((pos_s_2_17 & 4) == 0) ^ (unsigned int)((pos_s_2_17 & 64) == 0) ^ 1;
        vals[1] = ((gt_s_1_17 == keep_max_s_3_17) ? other_0_17 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 2);
        unsigned long long other_18 = _shfl_xor_41;
        unsigned int gt_s_19 = (unsigned int)(other_18 > vals[0]);
        int pos_s_19 = warp * 64 + lane;
        unsigned int keep_max_s_19 = (unsigned int)((pos_s_19 & 2) == 0) ^ (unsigned int)((pos_s_19 & 64) == 0) ^ 1;
        vals[0] = ((gt_s_19 == keep_max_s_19) ? other_18 : vals[0]);
        unsigned long long _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 2);
        unsigned long long other_0_18 = _shfl_xor_42;
        unsigned int gt_s_1_18 = (unsigned int)(other_0_18 > vals[1]);
        int pos_s_2_18 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_18 = (unsigned int)((pos_s_2_18 & 2) == 0) ^ (unsigned int)((pos_s_2_18 & 64) == 0) ^ 1;
        vals[1] = ((gt_s_1_18 == keep_max_s_3_18) ? other_0_18 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 1);
        unsigned long long other_19 = _shfl_xor_43;
        unsigned int gt_s_20 = (unsigned int)(other_19 > vals[0]);
        int pos_s_20 = warp * 64 + lane;
        unsigned int keep_max_s_20 = (unsigned int)((pos_s_20 & 1) == 0) ^ (unsigned int)((pos_s_20 & 64) == 0) ^ 1;
        vals[0] = ((gt_s_20 == keep_max_s_20) ? other_19 : vals[0]);
        unsigned long long _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 1);
        unsigned long long other_0_19 = _shfl_xor_44;
        unsigned int gt_s_1_19 = (unsigned int)(other_0_19 > vals[1]);
        int pos_s_2_19 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_19 = (unsigned int)((pos_s_2_19 & 1) == 0) ^ (unsigned int)((pos_s_2_19 & 64) == 0) ^ 1;
        vals[1] = ((gt_s_1_19 == keep_max_s_3_19) ? other_0_19 : vals[1]);
    }
    if (warp < 8) {
        int pos_w = warp * 64 + lane;
        pairs[pos_w] = vals[0];
        int pos_w_0 = warp * 64 + (32 + lane);
        pairs[pos_w_0] = vals[1];
    }
    __syncthreads();
    if (warp < 8) {
        int pos_r = warp * 64 + lane;
        unsigned long long partner = pairs[pos_r ^ 64];
        unsigned int gt_p = (unsigned int)(partner > vals[0]);
        unsigned int keep_max_p = (unsigned int)((pos_r & 64) == 0) ^ (unsigned int)((pos_r & 128) == 0) ^ 1;
        vals[0] = ((gt_p == keep_max_p) ? partner : vals[0]);
        int pos_r_0 = warp * 64 + (32 + lane);
        unsigned long long partner_1 = pairs[pos_r_0 ^ 64];
        unsigned int gt_p_2 = (unsigned int)(partner_1 > vals[1]);
        unsigned int keep_max_p_3 = (unsigned int)((pos_r_0 & 64) == 0) ^ (unsigned int)((pos_r_0 & 128) == 0) ^ 1;
        vals[1] = ((gt_p_2 == keep_max_p_3) ? partner_1 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long a_lo_1 = vals[0];
        unsigned long long b_hi_1 = vals[1];
        unsigned int gt_t_1 = (unsigned int)(b_hi_1 > a_lo_1);
        int pos_lo_1 = warp * 64 + lane;
        unsigned int desc_t_1 = (unsigned int)((pos_lo_1 & 128) == 0);
        bool swap_t_1 = gt_t_1 == desc_t_1;
        vals[0] = ((swap_t_1) ? b_hi_1 : a_lo_1);
        vals[1] = ((swap_t_1) ? a_lo_1 : b_hi_1);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 16);
        unsigned long long other_20 = _shfl_xor_45;
        unsigned int gt_s_21 = (unsigned int)(other_20 > vals[0]);
        int pos_s_21 = warp * 64 + lane;
        unsigned int keep_max_s_21 = (unsigned int)((pos_s_21 & 16) == 0) ^ (unsigned int)((pos_s_21 & 128) == 0) ^ 1;
        vals[0] = ((gt_s_21 == keep_max_s_21) ? other_20 : vals[0]);
        unsigned long long _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 16);
        unsigned long long other_0_20 = _shfl_xor_46;
        unsigned int gt_s_1_20 = (unsigned int)(other_0_20 > vals[1]);
        int pos_s_2_20 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_20 = (unsigned int)((pos_s_2_20 & 16) == 0) ^ (unsigned int)((pos_s_2_20 & 128) == 0) ^ 1;
        vals[1] = ((gt_s_1_20 == keep_max_s_3_20) ? other_0_20 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 8);
        unsigned long long other_21 = _shfl_xor_47;
        unsigned int gt_s_22 = (unsigned int)(other_21 > vals[0]);
        int pos_s_22 = warp * 64 + lane;
        unsigned int keep_max_s_22 = (unsigned int)((pos_s_22 & 8) == 0) ^ (unsigned int)((pos_s_22 & 128) == 0) ^ 1;
        vals[0] = ((gt_s_22 == keep_max_s_22) ? other_21 : vals[0]);
        unsigned long long _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 8);
        unsigned long long other_0_21 = _shfl_xor_48;
        unsigned int gt_s_1_21 = (unsigned int)(other_0_21 > vals[1]);
        int pos_s_2_21 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_21 = (unsigned int)((pos_s_2_21 & 8) == 0) ^ (unsigned int)((pos_s_2_21 & 128) == 0) ^ 1;
        vals[1] = ((gt_s_1_21 == keep_max_s_3_21) ? other_0_21 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 4);
        unsigned long long other_22 = _shfl_xor_49;
        unsigned int gt_s_23 = (unsigned int)(other_22 > vals[0]);
        int pos_s_23 = warp * 64 + lane;
        unsigned int keep_max_s_23 = (unsigned int)((pos_s_23 & 4) == 0) ^ (unsigned int)((pos_s_23 & 128) == 0) ^ 1;
        vals[0] = ((gt_s_23 == keep_max_s_23) ? other_22 : vals[0]);
        unsigned long long _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 4);
        unsigned long long other_0_22 = _shfl_xor_50;
        unsigned int gt_s_1_22 = (unsigned int)(other_0_22 > vals[1]);
        int pos_s_2_22 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_22 = (unsigned int)((pos_s_2_22 & 4) == 0) ^ (unsigned int)((pos_s_2_22 & 128) == 0) ^ 1;
        vals[1] = ((gt_s_1_22 == keep_max_s_3_22) ? other_0_22 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 2);
        unsigned long long other_23 = _shfl_xor_51;
        unsigned int gt_s_24 = (unsigned int)(other_23 > vals[0]);
        int pos_s_24 = warp * 64 + lane;
        unsigned int keep_max_s_24 = (unsigned int)((pos_s_24 & 2) == 0) ^ (unsigned int)((pos_s_24 & 128) == 0) ^ 1;
        vals[0] = ((gt_s_24 == keep_max_s_24) ? other_23 : vals[0]);
        unsigned long long _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 2);
        unsigned long long other_0_23 = _shfl_xor_52;
        unsigned int gt_s_1_23 = (unsigned int)(other_0_23 > vals[1]);
        int pos_s_2_23 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_23 = (unsigned int)((pos_s_2_23 & 2) == 0) ^ (unsigned int)((pos_s_2_23 & 128) == 0) ^ 1;
        vals[1] = ((gt_s_1_23 == keep_max_s_3_23) ? other_0_23 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 1);
        unsigned long long other_24 = _shfl_xor_53;
        unsigned int gt_s_25 = (unsigned int)(other_24 > vals[0]);
        int pos_s_25 = warp * 64 + lane;
        unsigned int keep_max_s_25 = (unsigned int)((pos_s_25 & 1) == 0) ^ (unsigned int)((pos_s_25 & 128) == 0) ^ 1;
        vals[0] = ((gt_s_25 == keep_max_s_25) ? other_24 : vals[0]);
        unsigned long long _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 1);
        unsigned long long other_0_24 = _shfl_xor_54;
        unsigned int gt_s_1_24 = (unsigned int)(other_0_24 > vals[1]);
        int pos_s_2_24 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_24 = (unsigned int)((pos_s_2_24 & 1) == 0) ^ (unsigned int)((pos_s_2_24 & 128) == 0) ^ 1;
        vals[1] = ((gt_s_1_24 == keep_max_s_3_24) ? other_0_24 : vals[1]);
    }
    if (warp < 8) {
        int pos_w_1 = warp * 64 + lane;
        pairs_b[pos_w_1] = vals[0];
        int pos_w_0_1 = warp * 64 + (32 + lane);
        pairs_b[pos_w_0_1] = vals[1];
    }
    __syncthreads();
    if (warp < 8) {
        int pos_r_1 = warp * 64 + lane;
        unsigned long long partner_2 = pairs_b[pos_r_1 ^ 128];
        unsigned int gt_p_1 = (unsigned int)(partner_2 > vals[0]);
        unsigned int keep_max_p_1 = (unsigned int)((pos_r_1 & 128) == 0) ^ (unsigned int)((pos_r_1 & 256) == 0) ^ 1;
        vals[0] = ((gt_p_1 == keep_max_p_1) ? partner_2 : vals[0]);
        int pos_r_0_1 = warp * 64 + (32 + lane);
        unsigned long long partner_1_1 = pairs_b[pos_r_0_1 ^ 128];
        unsigned int gt_p_2_1 = (unsigned int)(partner_1_1 > vals[1]);
        unsigned int keep_max_p_3_1 = (unsigned int)((pos_r_0_1 & 128) == 0) ^ (unsigned int)((pos_r_0_1 & 256) == 0) ^ 1;
        vals[1] = ((gt_p_2_1 == keep_max_p_3_1) ? partner_1_1 : vals[1]);
    }
    if (warp < 8) {
        int pos_w_2 = warp * 64 + lane;
        pairs[pos_w_2] = vals[0];
        int pos_w_0_2 = warp * 64 + (32 + lane);
        pairs[pos_w_0_2] = vals[1];
    }
    __syncthreads();
    if (warp < 8) {
        int pos_r_2 = warp * 64 + lane;
        unsigned long long partner_3 = pairs[pos_r_2 ^ 64];
        unsigned int gt_p_3 = (unsigned int)(partner_3 > vals[0]);
        unsigned int keep_max_p_2 = (unsigned int)((pos_r_2 & 64) == 0) ^ (unsigned int)((pos_r_2 & 256) == 0) ^ 1;
        vals[0] = ((gt_p_3 == keep_max_p_2) ? partner_3 : vals[0]);
        int pos_r_0_2 = warp * 64 + (32 + lane);
        unsigned long long partner_1_2 = pairs[pos_r_0_2 ^ 64];
        unsigned int gt_p_2_2 = (unsigned int)(partner_1_2 > vals[1]);
        unsigned int keep_max_p_3_2 = (unsigned int)((pos_r_0_2 & 64) == 0) ^ (unsigned int)((pos_r_0_2 & 256) == 0) ^ 1;
        vals[1] = ((gt_p_2_2 == keep_max_p_3_2) ? partner_1_2 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long a_lo_2 = vals[0];
        unsigned long long b_hi_2 = vals[1];
        unsigned int gt_t_2 = (unsigned int)(b_hi_2 > a_lo_2);
        int pos_lo_2 = warp * 64 + lane;
        unsigned int desc_t_2 = (unsigned int)((pos_lo_2 & 256) == 0);
        bool swap_t_2 = gt_t_2 == desc_t_2;
        vals[0] = ((swap_t_2) ? b_hi_2 : a_lo_2);
        vals[1] = ((swap_t_2) ? a_lo_2 : b_hi_2);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 16);
        unsigned long long other_25 = _shfl_xor_55;
        unsigned int gt_s_26 = (unsigned int)(other_25 > vals[0]);
        int pos_s_26 = warp * 64 + lane;
        unsigned int keep_max_s_26 = (unsigned int)((pos_s_26 & 16) == 0) ^ (unsigned int)((pos_s_26 & 256) == 0) ^ 1;
        vals[0] = ((gt_s_26 == keep_max_s_26) ? other_25 : vals[0]);
        unsigned long long _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 16);
        unsigned long long other_0_25 = _shfl_xor_56;
        unsigned int gt_s_1_25 = (unsigned int)(other_0_25 > vals[1]);
        int pos_s_2_25 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_25 = (unsigned int)((pos_s_2_25 & 16) == 0) ^ (unsigned int)((pos_s_2_25 & 256) == 0) ^ 1;
        vals[1] = ((gt_s_1_25 == keep_max_s_3_25) ? other_0_25 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 8);
        unsigned long long other_26 = _shfl_xor_57;
        unsigned int gt_s_27 = (unsigned int)(other_26 > vals[0]);
        int pos_s_27 = warp * 64 + lane;
        unsigned int keep_max_s_27 = (unsigned int)((pos_s_27 & 8) == 0) ^ (unsigned int)((pos_s_27 & 256) == 0) ^ 1;
        vals[0] = ((gt_s_27 == keep_max_s_27) ? other_26 : vals[0]);
        unsigned long long _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 8);
        unsigned long long other_0_26 = _shfl_xor_58;
        unsigned int gt_s_1_26 = (unsigned int)(other_0_26 > vals[1]);
        int pos_s_2_26 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_26 = (unsigned int)((pos_s_2_26 & 8) == 0) ^ (unsigned int)((pos_s_2_26 & 256) == 0) ^ 1;
        vals[1] = ((gt_s_1_26 == keep_max_s_3_26) ? other_0_26 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 4);
        unsigned long long other_27 = _shfl_xor_59;
        unsigned int gt_s_28 = (unsigned int)(other_27 > vals[0]);
        int pos_s_28 = warp * 64 + lane;
        unsigned int keep_max_s_28 = (unsigned int)((pos_s_28 & 4) == 0) ^ (unsigned int)((pos_s_28 & 256) == 0) ^ 1;
        vals[0] = ((gt_s_28 == keep_max_s_28) ? other_27 : vals[0]);
        unsigned long long _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 4);
        unsigned long long other_0_27 = _shfl_xor_60;
        unsigned int gt_s_1_27 = (unsigned int)(other_0_27 > vals[1]);
        int pos_s_2_27 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_27 = (unsigned int)((pos_s_2_27 & 4) == 0) ^ (unsigned int)((pos_s_2_27 & 256) == 0) ^ 1;
        vals[1] = ((gt_s_1_27 == keep_max_s_3_27) ? other_0_27 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 2);
        unsigned long long other_28 = _shfl_xor_61;
        unsigned int gt_s_29 = (unsigned int)(other_28 > vals[0]);
        int pos_s_29 = warp * 64 + lane;
        unsigned int keep_max_s_29 = (unsigned int)((pos_s_29 & 2) == 0) ^ (unsigned int)((pos_s_29 & 256) == 0) ^ 1;
        vals[0] = ((gt_s_29 == keep_max_s_29) ? other_28 : vals[0]);
        unsigned long long _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 2);
        unsigned long long other_0_28 = _shfl_xor_62;
        unsigned int gt_s_1_28 = (unsigned int)(other_0_28 > vals[1]);
        int pos_s_2_28 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_28 = (unsigned int)((pos_s_2_28 & 2) == 0) ^ (unsigned int)((pos_s_2_28 & 256) == 0) ^ 1;
        vals[1] = ((gt_s_1_28 == keep_max_s_3_28) ? other_0_28 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 1);
        unsigned long long other_29 = _shfl_xor_63;
        unsigned int gt_s_30 = (unsigned int)(other_29 > vals[0]);
        int pos_s_30 = warp * 64 + lane;
        unsigned int keep_max_s_30 = (unsigned int)((pos_s_30 & 1) == 0) ^ (unsigned int)((pos_s_30 & 256) == 0) ^ 1;
        vals[0] = ((gt_s_30 == keep_max_s_30) ? other_29 : vals[0]);
        unsigned long long _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 1);
        unsigned long long other_0_29 = _shfl_xor_64;
        unsigned int gt_s_1_29 = (unsigned int)(other_0_29 > vals[1]);
        int pos_s_2_29 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_29 = (unsigned int)((pos_s_2_29 & 1) == 0) ^ (unsigned int)((pos_s_2_29 & 256) == 0) ^ 1;
        vals[1] = ((gt_s_1_29 == keep_max_s_3_29) ? other_0_29 : vals[1]);
    }
    if (warp < 8) {
        int pos_w_3 = warp * 64 + lane;
        pairs_b[pos_w_3] = vals[0];
        int pos_w_0_3 = warp * 64 + (32 + lane);
        pairs_b[pos_w_0_3] = vals[1];
    }
    __syncthreads();
    if (warp < 8) {
        int pos_r_3 = warp * 64 + lane;
        unsigned long long partner_4 = pairs_b[pos_r_3 ^ 256];
        unsigned int gt_p_4 = (unsigned int)(partner_4 > vals[0]);
        unsigned int keep_max_p_4 = (unsigned int)((pos_r_3 & 256) == 0) ^ (unsigned int)((pos_r_3 & 512) == 0) ^ 1;
        vals[0] = ((gt_p_4 == keep_max_p_4) ? partner_4 : vals[0]);
        int pos_r_0_3 = warp * 64 + (32 + lane);
        unsigned long long partner_1_3 = pairs_b[pos_r_0_3 ^ 256];
        unsigned int gt_p_2_3 = (unsigned int)(partner_1_3 > vals[1]);
        unsigned int keep_max_p_3_3 = (unsigned int)((pos_r_0_3 & 256) == 0) ^ (unsigned int)((pos_r_0_3 & 512) == 0) ^ 1;
        vals[1] = ((gt_p_2_3 == keep_max_p_3_3) ? partner_1_3 : vals[1]);
    }
    if (warp < 8) {
        int pos_w_4 = warp * 64 + lane;
        pairs[pos_w_4] = vals[0];
        int pos_w_0_4 = warp * 64 + (32 + lane);
        pairs[pos_w_0_4] = vals[1];
    }
    __syncthreads();
    if (warp < 8) {
        int pos_r_4 = warp * 64 + lane;
        unsigned long long partner_5 = pairs[pos_r_4 ^ 128];
        unsigned int gt_p_5 = (unsigned int)(partner_5 > vals[0]);
        unsigned int keep_max_p_5 = (unsigned int)((pos_r_4 & 128) == 0) ^ (unsigned int)((pos_r_4 & 512) == 0) ^ 1;
        vals[0] = ((gt_p_5 == keep_max_p_5) ? partner_5 : vals[0]);
        int pos_r_0_4 = warp * 64 + (32 + lane);
        unsigned long long partner_1_4 = pairs[pos_r_0_4 ^ 128];
        unsigned int gt_p_2_4 = (unsigned int)(partner_1_4 > vals[1]);
        unsigned int keep_max_p_3_4 = (unsigned int)((pos_r_0_4 & 128) == 0) ^ (unsigned int)((pos_r_0_4 & 512) == 0) ^ 1;
        vals[1] = ((gt_p_2_4 == keep_max_p_3_4) ? partner_1_4 : vals[1]);
    }
    if (warp < 8) {
        int pos_w_5 = warp * 64 + lane;
        pairs_b[pos_w_5] = vals[0];
        int pos_w_0_5 = warp * 64 + (32 + lane);
        pairs_b[pos_w_0_5] = vals[1];
    }
    __syncthreads();
    if (warp < 8) {
        int pos_r_5 = warp * 64 + lane;
        unsigned long long partner_6 = pairs_b[pos_r_5 ^ 64];
        unsigned int gt_p_6 = (unsigned int)(partner_6 > vals[0]);
        unsigned int keep_max_p_6 = (unsigned int)((pos_r_5 & 64) == 0) ^ (unsigned int)((pos_r_5 & 512) == 0) ^ 1;
        vals[0] = ((gt_p_6 == keep_max_p_6) ? partner_6 : vals[0]);
        int pos_r_0_5 = warp * 64 + (32 + lane);
        unsigned long long partner_1_5 = pairs_b[pos_r_0_5 ^ 64];
        unsigned int gt_p_2_5 = (unsigned int)(partner_1_5 > vals[1]);
        unsigned int keep_max_p_3_5 = (unsigned int)((pos_r_0_5 & 64) == 0) ^ (unsigned int)((pos_r_0_5 & 512) == 0) ^ 1;
        vals[1] = ((gt_p_2_5 == keep_max_p_3_5) ? partner_1_5 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long a_lo_3 = vals[0];
        unsigned long long b_hi_3 = vals[1];
        unsigned int gt_t_3 = (unsigned int)(b_hi_3 > a_lo_3);
        int pos_lo_3 = warp * 64 + lane;
        unsigned int desc_t_3 = (unsigned int)((pos_lo_3 & 512) == 0);
        bool swap_t_3 = gt_t_3 == desc_t_3;
        vals[0] = ((swap_t_3) ? b_hi_3 : a_lo_3);
        vals[1] = ((swap_t_3) ? a_lo_3 : b_hi_3);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 16);
        unsigned long long other_30 = _shfl_xor_65;
        unsigned int gt_s_31 = (unsigned int)(other_30 > vals[0]);
        int pos_s_31 = warp * 64 + lane;
        unsigned int keep_max_s_31 = (unsigned int)((pos_s_31 & 16) == 0) ^ (unsigned int)((pos_s_31 & 512) == 0) ^ 1;
        vals[0] = ((gt_s_31 == keep_max_s_31) ? other_30 : vals[0]);
        unsigned long long _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 16);
        unsigned long long other_0_30 = _shfl_xor_66;
        unsigned int gt_s_1_30 = (unsigned int)(other_0_30 > vals[1]);
        int pos_s_2_30 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_30 = (unsigned int)((pos_s_2_30 & 16) == 0) ^ (unsigned int)((pos_s_2_30 & 512) == 0) ^ 1;
        vals[1] = ((gt_s_1_30 == keep_max_s_3_30) ? other_0_30 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 8);
        unsigned long long other_31 = _shfl_xor_67;
        unsigned int gt_s_32 = (unsigned int)(other_31 > vals[0]);
        int pos_s_32 = warp * 64 + lane;
        unsigned int keep_max_s_32 = (unsigned int)((pos_s_32 & 8) == 0) ^ (unsigned int)((pos_s_32 & 512) == 0) ^ 1;
        vals[0] = ((gt_s_32 == keep_max_s_32) ? other_31 : vals[0]);
        unsigned long long _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 8);
        unsigned long long other_0_31 = _shfl_xor_68;
        unsigned int gt_s_1_31 = (unsigned int)(other_0_31 > vals[1]);
        int pos_s_2_31 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_31 = (unsigned int)((pos_s_2_31 & 8) == 0) ^ (unsigned int)((pos_s_2_31 & 512) == 0) ^ 1;
        vals[1] = ((gt_s_1_31 == keep_max_s_3_31) ? other_0_31 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 4);
        unsigned long long other_32 = _shfl_xor_69;
        unsigned int gt_s_33 = (unsigned int)(other_32 > vals[0]);
        int pos_s_33 = warp * 64 + lane;
        unsigned int keep_max_s_33 = (unsigned int)((pos_s_33 & 4) == 0) ^ (unsigned int)((pos_s_33 & 512) == 0) ^ 1;
        vals[0] = ((gt_s_33 == keep_max_s_33) ? other_32 : vals[0]);
        unsigned long long _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 4);
        unsigned long long other_0_32 = _shfl_xor_70;
        unsigned int gt_s_1_32 = (unsigned int)(other_0_32 > vals[1]);
        int pos_s_2_32 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_32 = (unsigned int)((pos_s_2_32 & 4) == 0) ^ (unsigned int)((pos_s_2_32 & 512) == 0) ^ 1;
        vals[1] = ((gt_s_1_32 == keep_max_s_3_32) ? other_0_32 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 2);
        unsigned long long other_33 = _shfl_xor_71;
        unsigned int gt_s_34 = (unsigned int)(other_33 > vals[0]);
        int pos_s_34 = warp * 64 + lane;
        unsigned int keep_max_s_34 = (unsigned int)((pos_s_34 & 2) == 0) ^ (unsigned int)((pos_s_34 & 512) == 0) ^ 1;
        vals[0] = ((gt_s_34 == keep_max_s_34) ? other_33 : vals[0]);
        unsigned long long _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 2);
        unsigned long long other_0_33 = _shfl_xor_72;
        unsigned int gt_s_1_33 = (unsigned int)(other_0_33 > vals[1]);
        int pos_s_2_33 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_33 = (unsigned int)((pos_s_2_33 & 2) == 0) ^ (unsigned int)((pos_s_2_33 & 512) == 0) ^ 1;
        vals[1] = ((gt_s_1_33 == keep_max_s_3_33) ? other_0_33 : vals[1]);
    }
    if (warp < 8) {
        unsigned long long _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, vals[0], 1);
        unsigned long long other_34 = _shfl_xor_73;
        unsigned int gt_s_35 = (unsigned int)(other_34 > vals[0]);
        int pos_s_35 = warp * 64 + lane;
        unsigned int keep_max_s_35 = (unsigned int)((pos_s_35 & 1) == 0) ^ (unsigned int)((pos_s_35 & 512) == 0) ^ 1;
        vals[0] = ((gt_s_35 == keep_max_s_35) ? other_34 : vals[0]);
        unsigned long long _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, vals[1], 1);
        unsigned long long other_0_34 = _shfl_xor_74;
        unsigned int gt_s_1_34 = (unsigned int)(other_0_34 > vals[1]);
        int pos_s_2_34 = warp * 64 + (32 + lane);
        unsigned int keep_max_s_3_34 = (unsigned int)((pos_s_2_34 & 1) == 0) ^ (unsigned int)((pos_s_2_34 & 512) == 0) ^ 1;
        vals[1] = ((gt_s_1_34 == keep_max_s_3_34) ? other_0_34 : vals[1]);
    }
    unsigned long long out_base = (unsigned long long)row * 512;
    if (warp < 8) {
        int pos_o = warp * 64 + lane;
        if (pos_o < 512) {
            unsigned long long comp = vals[0];
            unsigned int key_hi = (unsigned int)(comp >> 32);
            unsigned int inv_lo = (unsigned int)(comp & 4294967295);
            int index = (int)(inv_lo ^ 4294967295u);
            int value = ((key_hi != 0) ? index : -1);
            out[out_base + (unsigned long long)pos_o] = value;
        }
        int pos_o_0 = warp * 64 + (32 + lane);
        if (pos_o_0 < 512) {
            unsigned long long comp_1 = vals[1];
            unsigned int key_hi_1 = (unsigned int)(comp_1 >> 32);
            unsigned int inv_lo_1 = (unsigned int)(comp_1 & 4294967295);
            int index_1 = (int)(inv_lo_1 ^ 4294967295u);
            int value_1 = ((key_hi_1 != 0) ? index_1 : -1);
            out[out_base + (unsigned long long)pos_o_0] = value_1;
        }
    }
}

} // extern "C"
