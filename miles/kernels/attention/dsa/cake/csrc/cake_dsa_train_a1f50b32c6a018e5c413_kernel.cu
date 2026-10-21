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
#define SMEM_PAIRS_OFF 0
#define SMEM_PAIRS_STAGE_BYTES 4096
#define SMEM_PAIRS_STRIDE 4096
#define SMEM_HIST_OFF 4096
#define SMEM_HIST_STAGE_BYTES 1024
#define SMEM_HIST_STRIDE 1024
#define SMEM_CNT_SEL_OFF 5120
#define SMEM_CNT_SEL_STAGE_BYTES 4096
#define SMEM_CNT_SEL_STRIDE 4096
#define SMEM_CNT_TIE_OFF 9216
#define SMEM_CNT_TIE_STAGE_BYTES 4096
#define SMEM_CNT_TIE_STRIDE 4096
#define SMEM_WARP_SUMS_OFF 13312
#define SMEM_WARP_SUMS_STAGE_BYTES 128
#define SMEM_WARP_SUMS_STRIDE 128
#define SMEM_STATE_OFF 13568
#define SMEM_STATE_STAGE_BYTES 16
#define SMEM_STATE_STRIDE 16
#define SMEM_PAIRS_B_OFF 13696
#define SMEM_PAIRS_B_STAGE_BYTES 4096
#define SMEM_PAIRS_B_STRIDE 4096
#define SMEM_CAND_OFF 17792
#define SMEM_CAND_STAGE_BYTES 32768
#define SMEM_CAND_STRIDE 32768
#define SMEM_TOTAL 50560
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
kernel_cake_dsa_train_a1f50b32c6a018e5c413(float* __restrict__ logits, int* __restrict__ out, int* __restrict__ row_end, int n_kv, int has_row_end)
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
    unsigned long long* cand = reinterpret_cast<unsigned long long*>(smem_raw + SMEM_CAND_OFF);
    const int cand_addr = smem + SMEM_CAND_OFF;

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
        state[3] = 0;
    }
    #pragma unroll 1
    for (int s = tid; s < 512; s += 512) {
        pairs[s] = 0;
    }
    #pragma unroll 4
    for (int sc = tid; sc < 4096; sc += 512) {
        cand[sc] = 0;
    }
    __syncthreads();
    int n_samp_rt = (limit + 511) / 512 * 32;
    unsigned int ksamp[4];
    int sq = tid;
    int psq = (sq >> 5) * 512 + (sq & 31);
    int idxc_s = ((psq < limit) ? psq : 0);
    float x_s = logits[row_base + (unsigned long long)idxc_s];
    unsigned int bits_s = __as_u32(x_s);
    bits_s = ((bits_s == 2147483648u) ? 0 : bits_s);
    unsigned int sign_s = (unsigned int)((int)bits_s >> 31);
    unsigned int enc_s = bits_s ^ (sign_s | 2147483648u);
    bool sel_s = enc_s - 8388608 <= 4278190080u;
    unsigned int ks_q = ((sel_s) ? enc_s : 0);
    ks_q = ((psq < limit) ? ks_q : 0);
    ksamp[0] = ks_q;
    int sq_0 = 512 + tid;
    int psq_1 = (sq_0 >> 5) * 512 + (sq_0 & 31);
    int idxc_s_2 = ((psq_1 < limit) ? psq_1 : 0);
    float x_s_3 = logits[row_base + (unsigned long long)idxc_s_2];
    unsigned int bits_s_4 = __as_u32(x_s_3);
    bits_s_4 = ((bits_s_4 == 2147483648u) ? 0 : bits_s_4);
    unsigned int sign_s_5 = (unsigned int)((int)bits_s_4 >> 31);
    unsigned int enc_s_6 = bits_s_4 ^ (sign_s_5 | 2147483648u);
    bool sel_s_7 = enc_s_6 - 8388608 <= 4278190080u;
    unsigned int ks_q_8 = ((sel_s_7) ? enc_s_6 : 0);
    ks_q_8 = ((psq_1 < limit) ? ks_q_8 : 0);
    ksamp[1] = ks_q_8;
    int sq_9 = 1024 + tid;
    int psq_10 = (sq_9 >> 5) * 512 + (sq_9 & 31);
    int idxc_s_11 = ((psq_10 < limit) ? psq_10 : 0);
    float x_s_12 = logits[row_base + (unsigned long long)idxc_s_11];
    unsigned int bits_s_13 = __as_u32(x_s_12);
    bits_s_13 = ((bits_s_13 == 2147483648u) ? 0 : bits_s_13);
    unsigned int sign_s_14 = (unsigned int)((int)bits_s_13 >> 31);
    unsigned int enc_s_15 = bits_s_13 ^ (sign_s_14 | 2147483648u);
    bool sel_s_16 = enc_s_15 - 8388608 <= 4278190080u;
    unsigned int ks_q_17 = ((sel_s_16) ? enc_s_15 : 0);
    ks_q_17 = ((psq_10 < limit) ? ks_q_17 : 0);
    ksamp[2] = ks_q_17;
    int sq_18 = 1536 + tid;
    int psq_19 = (sq_18 >> 5) * 512 + (sq_18 & 31);
    int idxc_s_20 = ((psq_19 < limit) ? psq_19 : 0);
    float x_s_21 = logits[row_base + (unsigned long long)idxc_s_20];
    unsigned int bits_s_22 = __as_u32(x_s_21);
    bits_s_22 = ((bits_s_22 == 2147483648u) ? 0 : bits_s_22);
    unsigned int sign_s_23 = (unsigned int)((int)bits_s_22 >> 31);
    unsigned int enc_s_24 = bits_s_22 ^ (sign_s_23 | 2147483648u);
    bool sel_s_25 = enc_s_24 - 8388608 <= 4278190080u;
    unsigned int ks_q_26 = ((sel_s_25) ? enc_s_24 : 0);
    ks_q_26 = ((psq_19 < limit) ? ks_q_26 : 0);
    ksamp[3] = ks_q_26;
    unsigned int ksq0 = ksamp[0];
    if (ksq0 != 0) {
        atomicAdd(&hist[ksq0 >> 24], 1);
    }
    unsigned int ksq0_27 = ksamp[1];
    if (ksq0_27 != 0) {
        atomicAdd(&hist[ksq0_27 >> 24], 1);
    }
    unsigned int ksq0_28 = ksamp[2];
    if (ksq0_28 != 0) {
        atomicAdd(&hist[ksq0_28 >> 24], 1);
    }
    unsigned int ksq0_29 = ksamp[3];
    if (ksq0_29 != 0) {
        atomicAdd(&hist[ksq0_29 >> 24], 1);
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
        if (running < 48) {
            if (at >= 48) {
                hit = (unsigned int)(lane * 8 + 7) << 16 | 48 - running;
                if (at == 48) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at;
        unsigned int at_11 = running + bins[6];
        if (running < 48) {
            if (at_11 >= 48) {
                hit = (unsigned int)(lane * 8 + 6) << 16 | 48 - running;
                if (at_11 == 48) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_11;
        unsigned int at_12 = running + bins[5];
        if (running < 48) {
            if (at_12 >= 48) {
                hit = (unsigned int)(lane * 8 + 5) << 16 | 48 - running;
                if (at_12 == 48) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_12;
        unsigned int at_13 = running + bins[4];
        if (running < 48) {
            if (at_13 >= 48) {
                hit = (unsigned int)(lane * 8 + 4) << 16 | 48 - running;
                if (at_13 == 48) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_13;
        unsigned int at_14 = running + bins[3];
        if (running < 48) {
            if (at_14 >= 48) {
                hit = (unsigned int)(lane * 8 + 3) << 16 | 48 - running;
                if (at_14 == 48) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_14;
        unsigned int at_15 = running + bins[2];
        if (running < 48) {
            if (at_15 >= 48) {
                hit = (unsigned int)(lane * 8 + 2) << 16 | 48 - running;
                if (at_15 == 48) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_15;
        unsigned int at_16 = running + bins[1];
        if (running < 48) {
            if (at_16 >= 48) {
                hit = (unsigned int)(lane * 8 + 1) << 16 | 48 - running;
                if (at_16 == 48) {
                    hit = hit | 2147483648u;
                }
            }
        }
        running = at_16;
        unsigned int at_17 = running + bins[0];
        if (running < 48) {
            if (at_17 >= 48) {
                hit = (unsigned int)(lane * 8) << 16 | 48 - running;
                if (at_17 == 48) {
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
    unsigned int chosen_s = state[0];
    unsigned int total_s = state[1];
    unsigned int cand_lo = 1;
    unsigned int samp_live = 0;
    unsigned int prefix_s = 0;
    unsigned int remaining_s = 48;
    if (total_s >= 48) {
        samp_live = 1;
        prefix_s = (chosen_s >> 16 & 255) << 24;
        remaining_s = chosen_s & 65535;
        cand_lo = prefix_s;
    }
    unsigned int prefix_s_hi = prefix_s >> 24;
    if (samp_live != 0) {
        unsigned int ksq1 = ksamp[0];
        if (ksq1 != 0) {
            if (ksq1 >> 24 == prefix_s_hi) {
                atomicAdd(&hist[ksq1 >> 16 & 255], 1);
            }
        }
        unsigned int ksq1_0 = ksamp[1];
        if (ksq1_0 != 0) {
            if (ksq1_0 >> 24 == prefix_s_hi) {
                atomicAdd(&hist[ksq1_0 >> 16 & 255], 1);
            }
        }
        unsigned int ksq1_1 = ksamp[2];
        if (ksq1_1 != 0) {
            if (ksq1_1 >> 24 == prefix_s_hi) {
                atomicAdd(&hist[ksq1_1 >> 16 & 255], 1);
            }
        }
        unsigned int ksq1_2 = ksamp[3];
        if (ksq1_2 != 0) {
            if (ksq1_2 >> 24 == prefix_s_hi) {
                atomicAdd(&hist[ksq1_2 >> 16 & 255], 1);
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
        if (running_1 < remaining_s) {
            if (at_1 >= remaining_s) {
                hit_1 = (unsigned int)(lane * 8 + 7) << 16 | remaining_s - running_1;
                if (at_1 == remaining_s) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_1;
        unsigned int at_11_1 = running_1 + bins_1[6];
        if (running_1 < remaining_s) {
            if (at_11_1 >= remaining_s) {
                hit_1 = (unsigned int)(lane * 8 + 6) << 16 | remaining_s - running_1;
                if (at_11_1 == remaining_s) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_11_1;
        unsigned int at_12_1 = running_1 + bins_1[5];
        if (running_1 < remaining_s) {
            if (at_12_1 >= remaining_s) {
                hit_1 = (unsigned int)(lane * 8 + 5) << 16 | remaining_s - running_1;
                if (at_12_1 == remaining_s) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_12_1;
        unsigned int at_13_1 = running_1 + bins_1[4];
        if (running_1 < remaining_s) {
            if (at_13_1 >= remaining_s) {
                hit_1 = (unsigned int)(lane * 8 + 4) << 16 | remaining_s - running_1;
                if (at_13_1 == remaining_s) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_13_1;
        unsigned int at_14_1 = running_1 + bins_1[3];
        if (running_1 < remaining_s) {
            if (at_14_1 >= remaining_s) {
                hit_1 = (unsigned int)(lane * 8 + 3) << 16 | remaining_s - running_1;
                if (at_14_1 == remaining_s) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_14_1;
        unsigned int at_15_1 = running_1 + bins_1[2];
        if (running_1 < remaining_s) {
            if (at_15_1 >= remaining_s) {
                hit_1 = (unsigned int)(lane * 8 + 2) << 16 | remaining_s - running_1;
                if (at_15_1 == remaining_s) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_15_1;
        unsigned int at_16_1 = running_1 + bins_1[1];
        if (running_1 < remaining_s) {
            if (at_16_1 >= remaining_s) {
                hit_1 = (unsigned int)(lane * 8 + 1) << 16 | remaining_s - running_1;
                if (at_16_1 == remaining_s) {
                    hit_1 = hit_1 | 2147483648u;
                }
            }
        }
        running_1 = at_16_1;
        unsigned int at_17_1 = running_1 + bins_1[0];
        if (running_1 < remaining_s) {
            if (at_17_1 >= remaining_s) {
                hit_1 = (unsigned int)(lane * 8) << 16 | remaining_s - running_1;
                if (at_17_1 == remaining_s) {
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
    unsigned int chosen_s1 = state[0];
    if (samp_live != 0) {
        cand_lo = prefix_s | (chosen_s1 >> 16 & 255) << 16;
    }
    if (cand_lo == 0) {
        cand_lo = 1;
    }
    unsigned int lt_mask = ~(4294967295u << (unsigned int)lane);
    int n_vec4 = (((n_kv & 3) == 0) ? limit / 4 : 0);
    int n_vchunks = (n_vec4 + 511) / 512;
    int last_sel = -1;
    unsigned int mkf[4];
    unsigned int kvf[4];
    float chunks[16];
    int n_vgroups = (n_vchunks + 3) / 4;
    #pragma unroll 1
    for (int vg = 0; vg < n_vgroups; vg++) {
        int v4l = vg * 4 * 512 + tid;
        int v4c = ((v4l < n_vec4) ? v4l : 0);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(logits + row_base + (unsigned long long)(v4c * 4));
            chunks[0 + 0] = _v4.x;
            chunks[0 + 1] = _v4.y;
            chunks[0 + 2] = _v4.z;
            chunks[0 + 3] = _v4.w;
        }
        int v4l_0 = (vg * 4 + 1) * 512 + tid;
        int v4c_1 = ((v4l_0 < n_vec4) ? v4l_0 : 0);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(logits + row_base + (unsigned long long)(v4c_1 * 4));
            chunks[4 + 0] = _v4.x;
            chunks[4 + 1] = _v4.y;
            chunks[4 + 2] = _v4.z;
            chunks[4 + 3] = _v4.w;
        }
        int v4l_2 = (vg * 4 + 2) * 512 + tid;
        int v4c_3 = ((v4l_2 < n_vec4) ? v4l_2 : 0);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(logits + row_base + (unsigned long long)(v4c_3 * 4));
            chunks[8 + 0] = _v4.x;
            chunks[8 + 1] = _v4.y;
            chunks[8 + 2] = _v4.z;
            chunks[8 + 3] = _v4.w;
        }
        int v4l_4 = (vg * 4 + 3) * 512 + tid;
        int v4c_5 = ((v4l_4 < n_vec4) ? v4l_4 : 0);
        {
            float4 _v4 = *reinterpret_cast<const float4*>(logits + row_base + (unsigned long long)(v4c_5 * 4));
            chunks[12 + 0] = _v4.x;
            chunks[12 + 1] = _v4.y;
            chunks[12 + 2] = _v4.z;
            chunks[12 + 3] = _v4.w;
        }
        int v4 = vg * 4 * 512 + tid;
        int pos0 = v4 * 4;
        bool vin = v4 < n_vec4;
        float xvf = chunks[0];
        unsigned int bitsv = __as_u32(xvf);
        bitsv = ((bitsv == 2147483648u) ? 0 : bitsv);
        unsigned int signv = (unsigned int)((int)bitsv >> 31);
        unsigned int encv = bitsv ^ (signv | 2147483648u);
        bool selv = encv - 8388608 <= 4278190080u;
        unsigned int keyv = ((selv) ? encv : 0);
        kvf[0] = ((vin) ? keyv : 0);
        float xvf_6 = chunks[1];
        unsigned int bitsv_7 = __as_u32(xvf_6);
        bitsv_7 = ((bitsv_7 == 2147483648u) ? 0 : bitsv_7);
        unsigned int signv_8 = (unsigned int)((int)bitsv_7 >> 31);
        unsigned int encv_9 = bitsv_7 ^ (signv_8 | 2147483648u);
        bool selv_10 = encv_9 - 8388608 <= 4278190080u;
        unsigned int keyv_11 = ((selv_10) ? encv_9 : 0);
        kvf[1] = ((vin) ? keyv_11 : 0);
        float xvf_12 = chunks[2];
        unsigned int bitsv_13 = __as_u32(xvf_12);
        bitsv_13 = ((bitsv_13 == 2147483648u) ? 0 : bitsv_13);
        unsigned int signv_14 = (unsigned int)((int)bitsv_13 >> 31);
        unsigned int encv_15 = bitsv_13 ^ (signv_14 | 2147483648u);
        bool selv_16 = encv_15 - 8388608 <= 4278190080u;
        unsigned int keyv_17 = ((selv_16) ? encv_15 : 0);
        kvf[2] = ((vin) ? keyv_17 : 0);
        float xvf_18 = chunks[3];
        unsigned int bitsv_19 = __as_u32(xvf_18);
        bitsv_19 = ((bitsv_19 == 2147483648u) ? 0 : bitsv_19);
        unsigned int signv_20 = (unsigned int)((int)bitsv_19 >> 31);
        unsigned int encv_21 = bitsv_19 ^ (signv_20 | 2147483648u);
        bool selv_22 = encv_21 - 8388608 <= 4278190080u;
        unsigned int keyv_23 = ((selv_22) ? encv_21 : 0);
        kvf[3] = ((vin) ? keyv_23 : 0);
        last_sel = ((kvf[0] != 0) ? pos0 : last_sel);
        last_sel = ((kvf[1] != 0) ? pos0 + 1 : last_sel);
        last_sel = ((kvf[2] != 0) ? pos0 + 2 : last_sel);
        last_sel = ((kvf[3] != 0) ? pos0 + 3 : last_sel);
        unsigned int nvf = 0;
        unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, kvf[0] >= cand_lo);
        mkf[0] = _vote_2;
        int _popc_0 = __popc(mkf[0]);
        nvf = nvf + (unsigned int)_popc_0;
        unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, kvf[1] >= cand_lo);
        mkf[1] = _vote_3;
        int _popc_1 = __popc(mkf[1]);
        nvf = nvf + (unsigned int)_popc_1;
        unsigned int _vote_4 = __ballot_sync(0xFFFFFFFF, kvf[2] >= cand_lo);
        mkf[2] = _vote_4;
        int _popc_2 = __popc(mkf[2]);
        nvf = nvf + (unsigned int)_popc_2;
        unsigned int _vote_5 = __ballot_sync(0xFFFFFFFF, kvf[3] >= cand_lo);
        mkf[3] = _vote_5;
        int _popc_3 = __popc(mkf[3]);
        nvf = nvf + (unsigned int)_popc_3;
        unsigned int _atomic_old_0 = 0;
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.u32 %0, [%1], %2;\n}"
            : "+r"(_atomic_old_0) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&state[3]))), "r"(static_cast<unsigned int>(nvf)), "r"(static_cast<uint32_t>((unsigned int)(lane == 0))) : "memory");
        unsigned int basevf = _atomic_old_0;
        unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, basevf, 0);
        basevf = _shfl_2;
        unsigned int kfj = kvf[0];
        unsigned int mfj = mkf[0];
        if (kfj >= cand_lo) {
            int _popc_4 = __popc(mfj & lt_mask);
            unsigned int slotvf = basevf + (unsigned int)_popc_4;
            if (slotvf < 4096) {
                unsigned int inv_vf = (unsigned int)pos0 ^ 4294967295u;
                cand[slotvf] = (unsigned long long)kfj << 32 | (unsigned long long)inv_vf;
            }
        }
        int _popc_5 = __popc(mfj);
        basevf = basevf + (unsigned int)_popc_5;
        unsigned int kfj_24 = kvf[1];
        unsigned int mfj_25 = mkf[1];
        if (kfj_24 >= cand_lo) {
            int _popc_6 = __popc(mfj_25 & lt_mask);
            unsigned int slotvf_1 = basevf + (unsigned int)_popc_6;
            if (slotvf_1 < 4096) {
                unsigned int inv_vf_1 = (unsigned int)(pos0 + 1) ^ 4294967295u;
                cand[slotvf_1] = (unsigned long long)kfj_24 << 32 | (unsigned long long)inv_vf_1;
            }
        }
        int _popc_7 = __popc(mfj_25);
        basevf = basevf + (unsigned int)_popc_7;
        unsigned int kfj_26 = kvf[2];
        unsigned int mfj_27 = mkf[2];
        if (kfj_26 >= cand_lo) {
            int _popc_8 = __popc(mfj_27 & lt_mask);
            unsigned int slotvf_2 = basevf + (unsigned int)_popc_8;
            if (slotvf_2 < 4096) {
                unsigned int inv_vf_2 = (unsigned int)(pos0 + 2) ^ 4294967295u;
                cand[slotvf_2] = (unsigned long long)kfj_26 << 32 | (unsigned long long)inv_vf_2;
            }
        }
        int _popc_9 = __popc(mfj_27);
        basevf = basevf + (unsigned int)_popc_9;
        unsigned int kfj_28 = kvf[3];
        unsigned int mfj_29 = mkf[3];
        if (kfj_28 >= cand_lo) {
            int _popc_10 = __popc(mfj_29 & lt_mask);
            unsigned int slotvf_3 = basevf + (unsigned int)_popc_10;
            if (slotvf_3 < 4096) {
                unsigned int inv_vf_3 = (unsigned int)(pos0 + 3) ^ 4294967295u;
                cand[slotvf_3] = (unsigned long long)kfj_28 << 32 | (unsigned long long)inv_vf_3;
            }
        }
        int _popc_11 = __popc(mfj_29);
        basevf = basevf + (unsigned int)_popc_11;
        int v4_30 = (vg * 4 + 1) * 512 + tid;
        int pos0_31 = v4_30 * 4;
        bool vin_32 = v4_30 < n_vec4;
        float xvf_33 = chunks[4];
        unsigned int bitsv_34 = __as_u32(xvf_33);
        bitsv_34 = ((bitsv_34 == 2147483648u) ? 0 : bitsv_34);
        unsigned int signv_35 = (unsigned int)((int)bitsv_34 >> 31);
        unsigned int encv_36 = bitsv_34 ^ (signv_35 | 2147483648u);
        bool selv_37 = encv_36 - 8388608 <= 4278190080u;
        unsigned int keyv_38 = ((selv_37) ? encv_36 : 0);
        kvf[0] = ((vin_32) ? keyv_38 : 0);
        float xvf_39 = chunks[5];
        unsigned int bitsv_40 = __as_u32(xvf_39);
        bitsv_40 = ((bitsv_40 == 2147483648u) ? 0 : bitsv_40);
        unsigned int signv_41 = (unsigned int)((int)bitsv_40 >> 31);
        unsigned int encv_42 = bitsv_40 ^ (signv_41 | 2147483648u);
        bool selv_43 = encv_42 - 8388608 <= 4278190080u;
        unsigned int keyv_44 = ((selv_43) ? encv_42 : 0);
        kvf[1] = ((vin_32) ? keyv_44 : 0);
        float xvf_45 = chunks[6];
        unsigned int bitsv_46 = __as_u32(xvf_45);
        bitsv_46 = ((bitsv_46 == 2147483648u) ? 0 : bitsv_46);
        unsigned int signv_47 = (unsigned int)((int)bitsv_46 >> 31);
        unsigned int encv_48 = bitsv_46 ^ (signv_47 | 2147483648u);
        bool selv_49 = encv_48 - 8388608 <= 4278190080u;
        unsigned int keyv_50 = ((selv_49) ? encv_48 : 0);
        kvf[2] = ((vin_32) ? keyv_50 : 0);
        float xvf_51 = chunks[7];
        unsigned int bitsv_52 = __as_u32(xvf_51);
        bitsv_52 = ((bitsv_52 == 2147483648u) ? 0 : bitsv_52);
        unsigned int signv_53 = (unsigned int)((int)bitsv_52 >> 31);
        unsigned int encv_54 = bitsv_52 ^ (signv_53 | 2147483648u);
        bool selv_55 = encv_54 - 8388608 <= 4278190080u;
        unsigned int keyv_56 = ((selv_55) ? encv_54 : 0);
        kvf[3] = ((vin_32) ? keyv_56 : 0);
        last_sel = ((kvf[0] != 0) ? pos0_31 : last_sel);
        last_sel = ((kvf[1] != 0) ? pos0_31 + 1 : last_sel);
        last_sel = ((kvf[2] != 0) ? pos0_31 + 2 : last_sel);
        last_sel = ((kvf[3] != 0) ? pos0_31 + 3 : last_sel);
        unsigned int nvf_57 = 0;
        unsigned int _vote_6 = __ballot_sync(0xFFFFFFFF, kvf[0] >= cand_lo);
        mkf[0] = _vote_6;
        int _popc_12 = __popc(mkf[0]);
        nvf_57 = nvf_57 + (unsigned int)_popc_12;
        unsigned int _vote_7 = __ballot_sync(0xFFFFFFFF, kvf[1] >= cand_lo);
        mkf[1] = _vote_7;
        int _popc_13 = __popc(mkf[1]);
        nvf_57 = nvf_57 + (unsigned int)_popc_13;
        unsigned int _vote_8 = __ballot_sync(0xFFFFFFFF, kvf[2] >= cand_lo);
        mkf[2] = _vote_8;
        int _popc_14 = __popc(mkf[2]);
        nvf_57 = nvf_57 + (unsigned int)_popc_14;
        unsigned int _vote_9 = __ballot_sync(0xFFFFFFFF, kvf[3] >= cand_lo);
        mkf[3] = _vote_9;
        int _popc_15 = __popc(mkf[3]);
        nvf_57 = nvf_57 + (unsigned int)_popc_15;
        unsigned int _atomic_old_1 = 0;
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.u32 %0, [%1], %2;\n}"
            : "+r"(_atomic_old_1) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&state[3]))), "r"(static_cast<unsigned int>(nvf_57)), "r"(static_cast<uint32_t>((unsigned int)(lane == 0))) : "memory");
        unsigned int basevf_58 = _atomic_old_1;
        unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, basevf_58, 0);
        basevf_58 = _shfl_3;
        unsigned int kfj_59 = kvf[0];
        unsigned int mfj_60 = mkf[0];
        if (kfj_59 >= cand_lo) {
            int _popc_16 = __popc(mfj_60 & lt_mask);
            unsigned int slotvf_4 = basevf_58 + (unsigned int)_popc_16;
            if (slotvf_4 < 4096) {
                unsigned int inv_vf_4 = (unsigned int)pos0_31 ^ 4294967295u;
                cand[slotvf_4] = (unsigned long long)kfj_59 << 32 | (unsigned long long)inv_vf_4;
            }
        }
        int _popc_17 = __popc(mfj_60);
        basevf_58 = basevf_58 + (unsigned int)_popc_17;
        unsigned int kfj_61 = kvf[1];
        unsigned int mfj_62 = mkf[1];
        if (kfj_61 >= cand_lo) {
            int _popc_18 = __popc(mfj_62 & lt_mask);
            unsigned int slotvf_5 = basevf_58 + (unsigned int)_popc_18;
            if (slotvf_5 < 4096) {
                unsigned int inv_vf_5 = (unsigned int)(pos0_31 + 1) ^ 4294967295u;
                cand[slotvf_5] = (unsigned long long)kfj_61 << 32 | (unsigned long long)inv_vf_5;
            }
        }
        int _popc_19 = __popc(mfj_62);
        basevf_58 = basevf_58 + (unsigned int)_popc_19;
        unsigned int kfj_63 = kvf[2];
        unsigned int mfj_64 = mkf[2];
        if (kfj_63 >= cand_lo) {
            int _popc_20 = __popc(mfj_64 & lt_mask);
            unsigned int slotvf_6 = basevf_58 + (unsigned int)_popc_20;
            if (slotvf_6 < 4096) {
                unsigned int inv_vf_6 = (unsigned int)(pos0_31 + 2) ^ 4294967295u;
                cand[slotvf_6] = (unsigned long long)kfj_63 << 32 | (unsigned long long)inv_vf_6;
            }
        }
        int _popc_21 = __popc(mfj_64);
        basevf_58 = basevf_58 + (unsigned int)_popc_21;
        unsigned int kfj_65 = kvf[3];
        unsigned int mfj_66 = mkf[3];
        if (kfj_65 >= cand_lo) {
            int _popc_22 = __popc(mfj_66 & lt_mask);
            unsigned int slotvf_7 = basevf_58 + (unsigned int)_popc_22;
            if (slotvf_7 < 4096) {
                unsigned int inv_vf_7 = (unsigned int)(pos0_31 + 3) ^ 4294967295u;
                cand[slotvf_7] = (unsigned long long)kfj_65 << 32 | (unsigned long long)inv_vf_7;
            }
        }
        int _popc_23 = __popc(mfj_66);
        basevf_58 = basevf_58 + (unsigned int)_popc_23;
        int v4_67 = (vg * 4 + 2) * 512 + tid;
        int pos0_68 = v4_67 * 4;
        bool vin_69 = v4_67 < n_vec4;
        float xvf_70 = chunks[8];
        unsigned int bitsv_71 = __as_u32(xvf_70);
        bitsv_71 = ((bitsv_71 == 2147483648u) ? 0 : bitsv_71);
        unsigned int signv_72 = (unsigned int)((int)bitsv_71 >> 31);
        unsigned int encv_73 = bitsv_71 ^ (signv_72 | 2147483648u);
        bool selv_74 = encv_73 - 8388608 <= 4278190080u;
        unsigned int keyv_75 = ((selv_74) ? encv_73 : 0);
        kvf[0] = ((vin_69) ? keyv_75 : 0);
        float xvf_76 = chunks[9];
        unsigned int bitsv_77 = __as_u32(xvf_76);
        bitsv_77 = ((bitsv_77 == 2147483648u) ? 0 : bitsv_77);
        unsigned int signv_78 = (unsigned int)((int)bitsv_77 >> 31);
        unsigned int encv_79 = bitsv_77 ^ (signv_78 | 2147483648u);
        bool selv_80 = encv_79 - 8388608 <= 4278190080u;
        unsigned int keyv_81 = ((selv_80) ? encv_79 : 0);
        kvf[1] = ((vin_69) ? keyv_81 : 0);
        float xvf_82 = chunks[10];
        unsigned int bitsv_83 = __as_u32(xvf_82);
        bitsv_83 = ((bitsv_83 == 2147483648u) ? 0 : bitsv_83);
        unsigned int signv_84 = (unsigned int)((int)bitsv_83 >> 31);
        unsigned int encv_85 = bitsv_83 ^ (signv_84 | 2147483648u);
        bool selv_86 = encv_85 - 8388608 <= 4278190080u;
        unsigned int keyv_87 = ((selv_86) ? encv_85 : 0);
        kvf[2] = ((vin_69) ? keyv_87 : 0);
        float xvf_88 = chunks[11];
        unsigned int bitsv_89 = __as_u32(xvf_88);
        bitsv_89 = ((bitsv_89 == 2147483648u) ? 0 : bitsv_89);
        unsigned int signv_90 = (unsigned int)((int)bitsv_89 >> 31);
        unsigned int encv_91 = bitsv_89 ^ (signv_90 | 2147483648u);
        bool selv_92 = encv_91 - 8388608 <= 4278190080u;
        unsigned int keyv_93 = ((selv_92) ? encv_91 : 0);
        kvf[3] = ((vin_69) ? keyv_93 : 0);
        last_sel = ((kvf[0] != 0) ? pos0_68 : last_sel);
        last_sel = ((kvf[1] != 0) ? pos0_68 + 1 : last_sel);
        last_sel = ((kvf[2] != 0) ? pos0_68 + 2 : last_sel);
        last_sel = ((kvf[3] != 0) ? pos0_68 + 3 : last_sel);
        unsigned int nvf_94 = 0;
        unsigned int _vote_10 = __ballot_sync(0xFFFFFFFF, kvf[0] >= cand_lo);
        mkf[0] = _vote_10;
        int _popc_24 = __popc(mkf[0]);
        nvf_94 = nvf_94 + (unsigned int)_popc_24;
        unsigned int _vote_11 = __ballot_sync(0xFFFFFFFF, kvf[1] >= cand_lo);
        mkf[1] = _vote_11;
        int _popc_25 = __popc(mkf[1]);
        nvf_94 = nvf_94 + (unsigned int)_popc_25;
        unsigned int _vote_12 = __ballot_sync(0xFFFFFFFF, kvf[2] >= cand_lo);
        mkf[2] = _vote_12;
        int _popc_26 = __popc(mkf[2]);
        nvf_94 = nvf_94 + (unsigned int)_popc_26;
        unsigned int _vote_13 = __ballot_sync(0xFFFFFFFF, kvf[3] >= cand_lo);
        mkf[3] = _vote_13;
        int _popc_27 = __popc(mkf[3]);
        nvf_94 = nvf_94 + (unsigned int)_popc_27;
        unsigned int _atomic_old_2 = 0;
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.u32 %0, [%1], %2;\n}"
            : "+r"(_atomic_old_2) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&state[3]))), "r"(static_cast<unsigned int>(nvf_94)), "r"(static_cast<uint32_t>((unsigned int)(lane == 0))) : "memory");
        unsigned int basevf_95 = _atomic_old_2;
        unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, basevf_95, 0);
        basevf_95 = _shfl_4;
        unsigned int kfj_96 = kvf[0];
        unsigned int mfj_97 = mkf[0];
        if (kfj_96 >= cand_lo) {
            int _popc_28 = __popc(mfj_97 & lt_mask);
            unsigned int slotvf_8 = basevf_95 + (unsigned int)_popc_28;
            if (slotvf_8 < 4096) {
                unsigned int inv_vf_8 = (unsigned int)pos0_68 ^ 4294967295u;
                cand[slotvf_8] = (unsigned long long)kfj_96 << 32 | (unsigned long long)inv_vf_8;
            }
        }
        int _popc_29 = __popc(mfj_97);
        basevf_95 = basevf_95 + (unsigned int)_popc_29;
        unsigned int kfj_98 = kvf[1];
        unsigned int mfj_99 = mkf[1];
        if (kfj_98 >= cand_lo) {
            int _popc_30 = __popc(mfj_99 & lt_mask);
            unsigned int slotvf_9 = basevf_95 + (unsigned int)_popc_30;
            if (slotvf_9 < 4096) {
                unsigned int inv_vf_9 = (unsigned int)(pos0_68 + 1) ^ 4294967295u;
                cand[slotvf_9] = (unsigned long long)kfj_98 << 32 | (unsigned long long)inv_vf_9;
            }
        }
        int _popc_31 = __popc(mfj_99);
        basevf_95 = basevf_95 + (unsigned int)_popc_31;
        unsigned int kfj_100 = kvf[2];
        unsigned int mfj_101 = mkf[2];
        if (kfj_100 >= cand_lo) {
            int _popc_32 = __popc(mfj_101 & lt_mask);
            unsigned int slotvf_10 = basevf_95 + (unsigned int)_popc_32;
            if (slotvf_10 < 4096) {
                unsigned int inv_vf_10 = (unsigned int)(pos0_68 + 2) ^ 4294967295u;
                cand[slotvf_10] = (unsigned long long)kfj_100 << 32 | (unsigned long long)inv_vf_10;
            }
        }
        int _popc_33 = __popc(mfj_101);
        basevf_95 = basevf_95 + (unsigned int)_popc_33;
        unsigned int kfj_102 = kvf[3];
        unsigned int mfj_103 = mkf[3];
        if (kfj_102 >= cand_lo) {
            int _popc_34 = __popc(mfj_103 & lt_mask);
            unsigned int slotvf_11 = basevf_95 + (unsigned int)_popc_34;
            if (slotvf_11 < 4096) {
                unsigned int inv_vf_11 = (unsigned int)(pos0_68 + 3) ^ 4294967295u;
                cand[slotvf_11] = (unsigned long long)kfj_102 << 32 | (unsigned long long)inv_vf_11;
            }
        }
        int _popc_35 = __popc(mfj_103);
        basevf_95 = basevf_95 + (unsigned int)_popc_35;
        int v4_104 = (vg * 4 + 3) * 512 + tid;
        int pos0_105 = v4_104 * 4;
        bool vin_106 = v4_104 < n_vec4;
        float xvf_107 = chunks[12];
        unsigned int bitsv_108 = __as_u32(xvf_107);
        bitsv_108 = ((bitsv_108 == 2147483648u) ? 0 : bitsv_108);
        unsigned int signv_109 = (unsigned int)((int)bitsv_108 >> 31);
        unsigned int encv_110 = bitsv_108 ^ (signv_109 | 2147483648u);
        bool selv_111 = encv_110 - 8388608 <= 4278190080u;
        unsigned int keyv_112 = ((selv_111) ? encv_110 : 0);
        kvf[0] = ((vin_106) ? keyv_112 : 0);
        float xvf_113 = chunks[13];
        unsigned int bitsv_114 = __as_u32(xvf_113);
        bitsv_114 = ((bitsv_114 == 2147483648u) ? 0 : bitsv_114);
        unsigned int signv_115 = (unsigned int)((int)bitsv_114 >> 31);
        unsigned int encv_116 = bitsv_114 ^ (signv_115 | 2147483648u);
        bool selv_117 = encv_116 - 8388608 <= 4278190080u;
        unsigned int keyv_118 = ((selv_117) ? encv_116 : 0);
        kvf[1] = ((vin_106) ? keyv_118 : 0);
        float xvf_119 = chunks[14];
        unsigned int bitsv_120 = __as_u32(xvf_119);
        bitsv_120 = ((bitsv_120 == 2147483648u) ? 0 : bitsv_120);
        unsigned int signv_121 = (unsigned int)((int)bitsv_120 >> 31);
        unsigned int encv_122 = bitsv_120 ^ (signv_121 | 2147483648u);
        bool selv_123 = encv_122 - 8388608 <= 4278190080u;
        unsigned int keyv_124 = ((selv_123) ? encv_122 : 0);
        kvf[2] = ((vin_106) ? keyv_124 : 0);
        float xvf_125 = chunks[15];
        unsigned int bitsv_126 = __as_u32(xvf_125);
        bitsv_126 = ((bitsv_126 == 2147483648u) ? 0 : bitsv_126);
        unsigned int signv_127 = (unsigned int)((int)bitsv_126 >> 31);
        unsigned int encv_128 = bitsv_126 ^ (signv_127 | 2147483648u);
        bool selv_129 = encv_128 - 8388608 <= 4278190080u;
        unsigned int keyv_130 = ((selv_129) ? encv_128 : 0);
        kvf[3] = ((vin_106) ? keyv_130 : 0);
        last_sel = ((kvf[0] != 0) ? pos0_105 : last_sel);
        last_sel = ((kvf[1] != 0) ? pos0_105 + 1 : last_sel);
        last_sel = ((kvf[2] != 0) ? pos0_105 + 2 : last_sel);
        last_sel = ((kvf[3] != 0) ? pos0_105 + 3 : last_sel);
        unsigned int nvf_131 = 0;
        unsigned int _vote_14 = __ballot_sync(0xFFFFFFFF, kvf[0] >= cand_lo);
        mkf[0] = _vote_14;
        int _popc_36 = __popc(mkf[0]);
        nvf_131 = nvf_131 + (unsigned int)_popc_36;
        unsigned int _vote_15 = __ballot_sync(0xFFFFFFFF, kvf[1] >= cand_lo);
        mkf[1] = _vote_15;
        int _popc_37 = __popc(mkf[1]);
        nvf_131 = nvf_131 + (unsigned int)_popc_37;
        unsigned int _vote_16 = __ballot_sync(0xFFFFFFFF, kvf[2] >= cand_lo);
        mkf[2] = _vote_16;
        int _popc_38 = __popc(mkf[2]);
        nvf_131 = nvf_131 + (unsigned int)_popc_38;
        unsigned int _vote_17 = __ballot_sync(0xFFFFFFFF, kvf[3] >= cand_lo);
        mkf[3] = _vote_17;
        int _popc_39 = __popc(mkf[3]);
        nvf_131 = nvf_131 + (unsigned int)_popc_39;
        unsigned int _atomic_old_3 = 0;
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.u32 %0, [%1], %2;\n}"
            : "+r"(_atomic_old_3) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&state[3]))), "r"(static_cast<unsigned int>(nvf_131)), "r"(static_cast<uint32_t>((unsigned int)(lane == 0))) : "memory");
        unsigned int basevf_132 = _atomic_old_3;
        unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, basevf_132, 0);
        basevf_132 = _shfl_5;
        unsigned int kfj_133 = kvf[0];
        unsigned int mfj_134 = mkf[0];
        if (kfj_133 >= cand_lo) {
            int _popc_40 = __popc(mfj_134 & lt_mask);
            unsigned int slotvf_12 = basevf_132 + (unsigned int)_popc_40;
            if (slotvf_12 < 4096) {
                unsigned int inv_vf_12 = (unsigned int)pos0_105 ^ 4294967295u;
                cand[slotvf_12] = (unsigned long long)kfj_133 << 32 | (unsigned long long)inv_vf_12;
            }
        }
        int _popc_41 = __popc(mfj_134);
        basevf_132 = basevf_132 + (unsigned int)_popc_41;
        unsigned int kfj_135 = kvf[1];
        unsigned int mfj_136 = mkf[1];
        if (kfj_135 >= cand_lo) {
            int _popc_42 = __popc(mfj_136 & lt_mask);
            unsigned int slotvf_13 = basevf_132 + (unsigned int)_popc_42;
            if (slotvf_13 < 4096) {
                unsigned int inv_vf_13 = (unsigned int)(pos0_105 + 1) ^ 4294967295u;
                cand[slotvf_13] = (unsigned long long)kfj_135 << 32 | (unsigned long long)inv_vf_13;
            }
        }
        int _popc_43 = __popc(mfj_136);
        basevf_132 = basevf_132 + (unsigned int)_popc_43;
        unsigned int kfj_137 = kvf[2];
        unsigned int mfj_138 = mkf[2];
        if (kfj_137 >= cand_lo) {
            int _popc_44 = __popc(mfj_138 & lt_mask);
            unsigned int slotvf_14 = basevf_132 + (unsigned int)_popc_44;
            if (slotvf_14 < 4096) {
                unsigned int inv_vf_14 = (unsigned int)(pos0_105 + 2) ^ 4294967295u;
                cand[slotvf_14] = (unsigned long long)kfj_137 << 32 | (unsigned long long)inv_vf_14;
            }
        }
        int _popc_45 = __popc(mfj_138);
        basevf_132 = basevf_132 + (unsigned int)_popc_45;
        unsigned int kfj_139 = kvf[3];
        unsigned int mfj_140 = mkf[3];
        if (kfj_139 >= cand_lo) {
            int _popc_46 = __popc(mfj_140 & lt_mask);
            unsigned int slotvf_15 = basevf_132 + (unsigned int)_popc_46;
            if (slotvf_15 < 4096) {
                unsigned int inv_vf_15 = (unsigned int)(pos0_105 + 3) ^ 4294967295u;
                cand[slotvf_15] = (unsigned long long)kfj_139 << 32 | (unsigned long long)inv_vf_15;
            }
        }
        int _popc_47 = __popc(mfj_140);
        basevf_132 = basevf_132 + (unsigned int)_popc_47;
    }
    int tail_start = n_vec4 * 4;
    int n_tchunks = (limit - tail_start + 511) / 512;
    #pragma unroll 1
    for (int tc = 0; tc < n_tchunks; tc++) {
        int idxt = tail_start + tc * 512 + tid;
        int idxc_t = ((idxt < limit) ? idxt : 0);
        float x_t = logits[row_base + (unsigned long long)idxc_t];
        unsigned int bits_t = __as_u32(x_t);
        bits_t = ((bits_t == 2147483648u) ? 0 : bits_t);
        unsigned int sign_t = (unsigned int)((int)bits_t >> 31);
        unsigned int enc_t = bits_t ^ (sign_t | 2147483648u);
        bool sel_t = enc_t - 8388608 <= 4278190080u;
        unsigned int kt = ((sel_t) ? enc_t : 0);
        kt = ((idxt < limit) ? kt : 0);
        last_sel = ((kt != 0) ? idxt : last_sel);
        bool is_ct = kt >= cand_lo;
        unsigned int _vote_18 = __ballot_sync(0xFFFFFFFF, is_ct);
        unsigned int mt = _vote_18;
        int _popc_48 = __popc(mt);
        unsigned int nt = (unsigned int)_popc_48;
        unsigned int _atomic_old_4 = 0;
        asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.u32 %0, [%1], %2;\n}"
            : "+r"(_atomic_old_4) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&state[3]))), "r"(static_cast<unsigned int>(nt)), "r"(static_cast<uint32_t>((unsigned int)(lane == 0))) : "memory");
        unsigned int baset = _atomic_old_4;
        unsigned int _shfl_6 = __shfl_sync(0xFFFFFFFF, baset, 0);
        baset = _shfl_6;
        if (is_ct) {
            int _popc_49 = __popc(mt & lt_mask);
            unsigned int slott = baset + (unsigned int)_popc_49;
            if (slott < 4096) {
                unsigned int inv_t = (unsigned int)idxt ^ 4294967295u;
                cand[slott] = (unsigned long long)kt << 32 | (unsigned long long)inv_t;
            }
        }
    }
    int warp_last = last_sel;
    int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, warp_last, 1);
    int other_last = _shfl_xor_0;
    warp_last = ((other_last > warp_last) ? other_last : warp_last);
    int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, warp_last, 2);
    int other_last_30 = _shfl_xor_1;
    warp_last = ((other_last_30 > warp_last) ? other_last_30 : warp_last);
    int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, warp_last, 4);
    int other_last_31 = _shfl_xor_2;
    warp_last = ((other_last_31 > warp_last) ? other_last_31 : warp_last);
    int _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, warp_last, 8);
    int other_last_32 = _shfl_xor_3;
    warp_last = ((other_last_32 > warp_last) ? other_last_32 : warp_last);
    int _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, warp_last, 16);
    int other_last_33 = _shfl_xor_4;
    warp_last = ((other_last_33 > warp_last) ? other_last_33 : warp_last);
    if (lane == 0) {
        warp_sums[warp] = (unsigned int)(warp_last + 1);
    }
    __syncthreads();
    int limit_eff = 0;
    int wl = (int)warp_sums[0];
    limit_eff = ((wl > limit_eff) ? wl : limit_eff);
    int wl_34 = (int)warp_sums[1];
    limit_eff = ((wl_34 > limit_eff) ? wl_34 : limit_eff);
    int wl_35 = (int)warp_sums[2];
    limit_eff = ((wl_35 > limit_eff) ? wl_35 : limit_eff);
    int wl_36 = (int)warp_sums[3];
    limit_eff = ((wl_36 > limit_eff) ? wl_36 : limit_eff);
    int wl_37 = (int)warp_sums[4];
    limit_eff = ((wl_37 > limit_eff) ? wl_37 : limit_eff);
    int wl_38 = (int)warp_sums[5];
    limit_eff = ((wl_38 > limit_eff) ? wl_38 : limit_eff);
    int wl_39 = (int)warp_sums[6];
    limit_eff = ((wl_39 > limit_eff) ? wl_39 : limit_eff);
    int wl_40 = (int)warp_sums[7];
    limit_eff = ((wl_40 > limit_eff) ? wl_40 : limit_eff);
    int wl_41 = (int)warp_sums[8];
    limit_eff = ((wl_41 > limit_eff) ? wl_41 : limit_eff);
    int wl_42 = (int)warp_sums[9];
    limit_eff = ((wl_42 > limit_eff) ? wl_42 : limit_eff);
    int wl_43 = (int)warp_sums[10];
    limit_eff = ((wl_43 > limit_eff) ? wl_43 : limit_eff);
    int wl_44 = (int)warp_sums[11];
    limit_eff = ((wl_44 > limit_eff) ? wl_44 : limit_eff);
    int wl_45 = (int)warp_sums[12];
    limit_eff = ((wl_45 > limit_eff) ? wl_45 : limit_eff);
    int wl_46 = (int)warp_sums[13];
    limit_eff = ((wl_46 > limit_eff) ? wl_46 : limit_eff);
    int wl_47 = (int)warp_sums[14];
    limit_eff = ((wl_47 > limit_eff) ? wl_47 : limit_eff);
    int wl_48 = (int)warp_sums[15];
    limit_eff = ((wl_48 > limit_eff) ? wl_48 : limit_eff);
    limit = limit_eff;
    n_chunks_rt = (limit + 511) / 512;
    stage_end = n_chunks_rt * 512;
    n_slots_rt = n_chunks_rt * 16;
    int n_cand_rt = (int)state[3];
    int n_chunks_fb = n_chunks_rt;
    if (n_cand_rt >= 512) {
        if (n_cand_rt <= 4096) {
            n_chunks_fb = 0;
        } else {
            n_cand_rt = 0;
        }
    } else {
        n_cand_rt = 0;
    }
    int n_cchunks = (n_cand_rt + 511) / 512;
    #pragma unroll 8
    for (int c0 = 0; c0 < n_chunks_fb; c0++) {
        int idx0 = c0 * 512 + tid;
        int idxc_0 = ((idx0 < limit) ? idx0 : 0);
        float x_0 = logits[row_base + (unsigned long long)idxc_0];
        unsigned int bits_0 = __as_u32(x_0);
        bits_0 = ((bits_0 == 2147483648u) ? 0 : bits_0);
        unsigned int sign_0 = (unsigned int)((int)bits_0 >> 31);
        unsigned int enc_0 = bits_0 ^ (sign_0 | 2147483648u);
        bool sel_0 = enc_0 - 8388608 <= 4278190080u;
        unsigned int key0 = ((sel_0) ? enc_0 : 0);
        key0 = ((idx0 < limit) ? key0 : 0);
        if (key0 != 0) {
            atomicAdd(&hist[key0 >> 24], 1);
        }
    }
    #pragma unroll 8
    for (int cc0 = 0; cc0 < n_cchunks; cc0++) {
        int ic0 = cc0 * 512 + tid;
        unsigned long long comp0 = cand[ic0];
        unsigned int k0c = (unsigned int)(comp0 >> 32);
        if (k0c != 0) {
            atomicAdd(&hist[k0c >> 24], 1);
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
        if (running_2 < 512) {
            if (at_2 >= 512) {
                hit_2 = (unsigned int)(lane * 8 + 7) << 16 | 512 - running_2;
                if (at_2 == 512) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_2;
        unsigned int at_11_2 = running_2 + bins_2[6];
        if (running_2 < 512) {
            if (at_11_2 >= 512) {
                hit_2 = (unsigned int)(lane * 8 + 6) << 16 | 512 - running_2;
                if (at_11_2 == 512) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_11_2;
        unsigned int at_12_2 = running_2 + bins_2[5];
        if (running_2 < 512) {
            if (at_12_2 >= 512) {
                hit_2 = (unsigned int)(lane * 8 + 5) << 16 | 512 - running_2;
                if (at_12_2 == 512) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_12_2;
        unsigned int at_13_2 = running_2 + bins_2[4];
        if (running_2 < 512) {
            if (at_13_2 >= 512) {
                hit_2 = (unsigned int)(lane * 8 + 4) << 16 | 512 - running_2;
                if (at_13_2 == 512) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_13_2;
        unsigned int at_14_2 = running_2 + bins_2[3];
        if (running_2 < 512) {
            if (at_14_2 >= 512) {
                hit_2 = (unsigned int)(lane * 8 + 3) << 16 | 512 - running_2;
                if (at_14_2 == 512) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_14_2;
        unsigned int at_15_2 = running_2 + bins_2[2];
        if (running_2 < 512) {
            if (at_15_2 >= 512) {
                hit_2 = (unsigned int)(lane * 8 + 2) << 16 | 512 - running_2;
                if (at_15_2 == 512) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_15_2;
        unsigned int at_16_2 = running_2 + bins_2[1];
        if (running_2 < 512) {
            if (at_16_2 >= 512) {
                hit_2 = (unsigned int)(lane * 8 + 1) << 16 | 512 - running_2;
                if (at_16_2 == 512) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_16_2;
        unsigned int at_17_2 = running_2 + bins_2[0];
        if (running_2 < 512) {
            if (at_17_2 >= 512) {
                hit_2 = (unsigned int)(lane * 8) << 16 | 512 - running_2;
                if (at_17_2 == 512) {
                    hit_2 = hit_2 | 2147483648u;
                }
            }
        }
        running_2 = at_17_2;
        unsigned int _vote_19 = __ballot_sync(0xFFFFFFFF, hit_2 != 4294967295u);
        unsigned int hit_mask_2 = _vote_19;
        int _ffs_2 = __ffs(hit_mask_2);
        unsigned int src_lane_2 = (unsigned int)(_ffs_2 - 1) & 31;
        unsigned int _shfl_7 = __shfl_sync(0xFFFFFFFF, hit_2, src_lane_2);
        unsigned int chosen_2 = _shfl_7;
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
        #pragma unroll 8
        for (int c1 = 0; c1 < n_chunks_fb; c1++) {
            int idx1 = c1 * 512 + tid;
            int idxc_1 = ((idx1 < limit) ? idx1 : 0);
            float x_1 = logits[row_base + (unsigned long long)idxc_1];
            unsigned int bits_1 = __as_u32(x_1);
            bits_1 = ((bits_1 == 2147483648u) ? 0 : bits_1);
            unsigned int sign_1 = (unsigned int)((int)bits_1 >> 31);
            unsigned int enc_1 = bits_1 ^ (sign_1 | 2147483648u);
            bool sel_1 = enc_1 - 8388608 <= 4278190080u;
            unsigned int key1 = ((sel_1) ? enc_1 : 0);
            key1 = ((idx1 < limit) ? key1 : 0);
            if (key1 != 0) {
                if (key1 >> 24 == prefix_hi) {
                    atomicAdd(&hist[key1 >> 16 & 255], 1);
                }
            }
        }
        #pragma unroll 8
        for (int cc1 = 0; cc1 < n_cchunks; cc1++) {
            int ic = cc1 * 512 + tid;
            unsigned long long compi = cand[ic];
            unsigned int kc = (unsigned int)(compi >> 32);
            if (kc != 0) {
                if (kc >> 24 == prefix_hi) {
                    atomicAdd(&hist[kc >> 16 & 255], 1);
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
        unsigned int _vote_20 = __ballot_sync(0xFFFFFFFF, hit_3 != 4294967295u);
        unsigned int hit_mask_3 = _vote_20;
        int _ffs_3 = __ffs(hit_mask_3);
        unsigned int src_lane_3 = (unsigned int)(_ffs_3 - 1) & 31;
        unsigned int _shfl_8 = __shfl_sync(0xFFFFFFFF, hit_3, src_lane_3);
        unsigned int chosen_3 = _shfl_8;
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
    unsigned int prefix_hi_49 = prefix >> 16;
    if (done == 0) {
        #pragma unroll 8
        for (int c1_1 = 0; c1_1 < n_chunks_fb; c1_1++) {
            int idx1_1 = c1_1 * 512 + tid;
            int idxc_1_1 = ((idx1_1 < limit) ? idx1_1 : 0);
            float x_1_1 = logits[row_base + (unsigned long long)idxc_1_1];
            unsigned int bits_1_1 = __as_u32(x_1_1);
            bits_1_1 = ((bits_1_1 == 2147483648u) ? 0 : bits_1_1);
            unsigned int sign_1_1 = (unsigned int)((int)bits_1_1 >> 31);
            unsigned int enc_1_1 = bits_1_1 ^ (sign_1_1 | 2147483648u);
            bool sel_1_1 = enc_1_1 - 8388608 <= 4278190080u;
            unsigned int key1_1 = ((sel_1_1) ? enc_1_1 : 0);
            key1_1 = ((idx1_1 < limit) ? key1_1 : 0);
            if (key1_1 != 0) {
                if (key1_1 >> 16 == prefix_hi_49) {
                    atomicAdd(&hist[key1_1 >> 8 & 255], 1);
                }
            }
        }
        #pragma unroll 8
        for (int cc1_1 = 0; cc1_1 < n_cchunks; cc1_1++) {
            int ic_1 = cc1_1 * 512 + tid;
            unsigned long long compi_1 = cand[ic_1];
            unsigned int kc_1 = (unsigned int)(compi_1 >> 32);
            if (kc_1 != 0) {
                if (kc_1 >> 16 == prefix_hi_49) {
                    atomicAdd(&hist[kc_1 >> 8 & 255], 1);
                }
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        unsigned int bins_4[8];
        unsigned int lane_total_4 = 0;
        unsigned int hb_10 = hist[lane * 8];
        bins_4[0] = hb_10;
        lane_total_4 = lane_total_4 + hb_10;
        unsigned int hb_0_4 = hist[lane * 8 + 1];
        bins_4[1] = hb_0_4;
        lane_total_4 = lane_total_4 + hb_0_4;
        unsigned int hb_1_4 = hist[lane * 8 + 2];
        bins_4[2] = hb_1_4;
        lane_total_4 = lane_total_4 + hb_1_4;
        unsigned int hb_2_4 = hist[lane * 8 + 3];
        bins_4[3] = hb_2_4;
        lane_total_4 = lane_total_4 + hb_2_4;
        unsigned int hb_3_4 = hist[lane * 8 + 4];
        bins_4[4] = hb_3_4;
        lane_total_4 = lane_total_4 + hb_3_4;
        unsigned int hb_4_4 = hist[lane * 8 + 5];
        bins_4[5] = hb_4_4;
        lane_total_4 = lane_total_4 + hb_4_4;
        unsigned int hb_5_4 = hist[lane * 8 + 6];
        bins_4[6] = hb_5_4;
        lane_total_4 = lane_total_4 + hb_5_4;
        unsigned int hb_6_4 = hist[lane * 8 + 7];
        bins_4[7] = hb_6_4;
        lane_total_4 = lane_total_4 + hb_6_4;
        unsigned int above_incl_4 = lane_total_4;
        unsigned int _shfl_down_20 = __shfl_down_sync(0xFFFFFFFF, above_incl_4, 1, 32);
        unsigned int nxt_4 = _shfl_down_20;
        if (lane + 1 < 32) {
            above_incl_4 = above_incl_4 + nxt_4;
        }
        unsigned int _shfl_down_21 = __shfl_down_sync(0xFFFFFFFF, above_incl_4, 2, 32);
        unsigned int nxt_7_4 = _shfl_down_21;
        if (lane + 2 < 32) {
            above_incl_4 = above_incl_4 + nxt_7_4;
        }
        unsigned int _shfl_down_22 = __shfl_down_sync(0xFFFFFFFF, above_incl_4, 4, 32);
        unsigned int nxt_8_4 = _shfl_down_22;
        if (lane + 4 < 32) {
            above_incl_4 = above_incl_4 + nxt_8_4;
        }
        unsigned int _shfl_down_23 = __shfl_down_sync(0xFFFFFFFF, above_incl_4, 8, 32);
        unsigned int nxt_9_4 = _shfl_down_23;
        if (lane + 8 < 32) {
            above_incl_4 = above_incl_4 + nxt_9_4;
        }
        unsigned int _shfl_down_24 = __shfl_down_sync(0xFFFFFFFF, above_incl_4, 16, 32);
        unsigned int nxt_10_4 = _shfl_down_24;
        if (lane + 16 < 32) {
            above_incl_4 = above_incl_4 + nxt_10_4;
        }
        unsigned int running_4 = above_incl_4 - lane_total_4;
        unsigned int hit_4 = 4294967295u;
        unsigned int at_4 = running_4 + bins_4[7];
        if (running_4 < remaining) {
            if (at_4 >= remaining) {
                hit_4 = (unsigned int)(lane * 8 + 7) << 16 | remaining - running_4;
                if (at_4 == remaining) {
                    hit_4 = hit_4 | 2147483648u;
                }
            }
        }
        running_4 = at_4;
        unsigned int at_11_4 = running_4 + bins_4[6];
        if (running_4 < remaining) {
            if (at_11_4 >= remaining) {
                hit_4 = (unsigned int)(lane * 8 + 6) << 16 | remaining - running_4;
                if (at_11_4 == remaining) {
                    hit_4 = hit_4 | 2147483648u;
                }
            }
        }
        running_4 = at_11_4;
        unsigned int at_12_4 = running_4 + bins_4[5];
        if (running_4 < remaining) {
            if (at_12_4 >= remaining) {
                hit_4 = (unsigned int)(lane * 8 + 5) << 16 | remaining - running_4;
                if (at_12_4 == remaining) {
                    hit_4 = hit_4 | 2147483648u;
                }
            }
        }
        running_4 = at_12_4;
        unsigned int at_13_4 = running_4 + bins_4[4];
        if (running_4 < remaining) {
            if (at_13_4 >= remaining) {
                hit_4 = (unsigned int)(lane * 8 + 4) << 16 | remaining - running_4;
                if (at_13_4 == remaining) {
                    hit_4 = hit_4 | 2147483648u;
                }
            }
        }
        running_4 = at_13_4;
        unsigned int at_14_4 = running_4 + bins_4[3];
        if (running_4 < remaining) {
            if (at_14_4 >= remaining) {
                hit_4 = (unsigned int)(lane * 8 + 3) << 16 | remaining - running_4;
                if (at_14_4 == remaining) {
                    hit_4 = hit_4 | 2147483648u;
                }
            }
        }
        running_4 = at_14_4;
        unsigned int at_15_4 = running_4 + bins_4[2];
        if (running_4 < remaining) {
            if (at_15_4 >= remaining) {
                hit_4 = (unsigned int)(lane * 8 + 2) << 16 | remaining - running_4;
                if (at_15_4 == remaining) {
                    hit_4 = hit_4 | 2147483648u;
                }
            }
        }
        running_4 = at_15_4;
        unsigned int at_16_4 = running_4 + bins_4[1];
        if (running_4 < remaining) {
            if (at_16_4 >= remaining) {
                hit_4 = (unsigned int)(lane * 8 + 1) << 16 | remaining - running_4;
                if (at_16_4 == remaining) {
                    hit_4 = hit_4 | 2147483648u;
                }
            }
        }
        running_4 = at_16_4;
        unsigned int at_17_4 = running_4 + bins_4[0];
        if (running_4 < remaining) {
            if (at_17_4 >= remaining) {
                hit_4 = (unsigned int)(lane * 8) << 16 | remaining - running_4;
                if (at_17_4 == remaining) {
                    hit_4 = hit_4 | 2147483648u;
                }
            }
        }
        running_4 = at_17_4;
        unsigned int _vote_21 = __ballot_sync(0xFFFFFFFF, hit_4 != 4294967295u);
        unsigned int hit_mask_4 = _vote_21;
        int _ffs_4 = __ffs(hit_mask_4);
        unsigned int src_lane_4 = (unsigned int)(_ffs_4 - 1) & 31;
        unsigned int _shfl_9 = __shfl_sync(0xFFFFFFFF, hit_4, src_lane_4);
        unsigned int chosen_4 = _shfl_9;
        hist[lane * 8] = 0;
        hist[lane * 8 + 1] = 0;
        hist[lane * 8 + 2] = 0;
        hist[lane * 8 + 3] = 0;
        hist[lane * 8 + 4] = 0;
        hist[lane * 8 + 5] = 0;
        hist[lane * 8 + 6] = 0;
        hist[lane * 8 + 7] = 0;
        if (lane == 0) {
            state[0] = chosen_4;
            state[1] = above_incl_4;
        }
    }
    __syncthreads();
    unsigned int chosen1_50 = state[0];
    if (done == 0) {
        prefix = prefix | (chosen1_50 >> 16 & 255) << 8;
        remaining = chosen1_50 & 65535;
        if ((chosen1_50 & 2147483648u) != 0) {
            if (prefix != 0) {
                pivot = prefix - 1;
                count_gt = 512;
                done = 1;
            }
        }
    }
    unsigned int prefix_hi_51 = prefix >> 8;
    if (done == 0) {
        #pragma unroll 8
        for (int c1_2 = 0; c1_2 < n_chunks_fb; c1_2++) {
            int idx1_2 = c1_2 * 512 + tid;
            int idxc_1_2 = ((idx1_2 < limit) ? idx1_2 : 0);
            float x_1_2 = logits[row_base + (unsigned long long)idxc_1_2];
            unsigned int bits_1_2 = __as_u32(x_1_2);
            bits_1_2 = ((bits_1_2 == 2147483648u) ? 0 : bits_1_2);
            unsigned int sign_1_2 = (unsigned int)((int)bits_1_2 >> 31);
            unsigned int enc_1_2 = bits_1_2 ^ (sign_1_2 | 2147483648u);
            bool sel_1_2 = enc_1_2 - 8388608 <= 4278190080u;
            unsigned int key1_2 = ((sel_1_2) ? enc_1_2 : 0);
            key1_2 = ((idx1_2 < limit) ? key1_2 : 0);
            if (key1_2 != 0) {
                if (key1_2 >> 8 == prefix_hi_51) {
                    atomicAdd(&hist[key1_2 & 255], 1);
                }
            }
        }
        #pragma unroll 8
        for (int cc1_2 = 0; cc1_2 < n_cchunks; cc1_2++) {
            int ic_2 = cc1_2 * 512 + tid;
            unsigned long long compi_2 = cand[ic_2];
            unsigned int kc_2 = (unsigned int)(compi_2 >> 32);
            if (kc_2 != 0) {
                if (kc_2 >> 8 == prefix_hi_51) {
                    atomicAdd(&hist[kc_2 & 255], 1);
                }
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
        unsigned int bins_5[8];
        unsigned int lane_total_5 = 0;
        unsigned int hb_11 = hist[lane * 8];
        bins_5[0] = hb_11;
        lane_total_5 = lane_total_5 + hb_11;
        unsigned int hb_0_5 = hist[lane * 8 + 1];
        bins_5[1] = hb_0_5;
        lane_total_5 = lane_total_5 + hb_0_5;
        unsigned int hb_1_5 = hist[lane * 8 + 2];
        bins_5[2] = hb_1_5;
        lane_total_5 = lane_total_5 + hb_1_5;
        unsigned int hb_2_5 = hist[lane * 8 + 3];
        bins_5[3] = hb_2_5;
        lane_total_5 = lane_total_5 + hb_2_5;
        unsigned int hb_3_5 = hist[lane * 8 + 4];
        bins_5[4] = hb_3_5;
        lane_total_5 = lane_total_5 + hb_3_5;
        unsigned int hb_4_5 = hist[lane * 8 + 5];
        bins_5[5] = hb_4_5;
        lane_total_5 = lane_total_5 + hb_4_5;
        unsigned int hb_5_5 = hist[lane * 8 + 6];
        bins_5[6] = hb_5_5;
        lane_total_5 = lane_total_5 + hb_5_5;
        unsigned int hb_6_5 = hist[lane * 8 + 7];
        bins_5[7] = hb_6_5;
        lane_total_5 = lane_total_5 + hb_6_5;
        unsigned int above_incl_5 = lane_total_5;
        unsigned int _shfl_down_25 = __shfl_down_sync(0xFFFFFFFF, above_incl_5, 1, 32);
        unsigned int nxt_5 = _shfl_down_25;
        if (lane + 1 < 32) {
            above_incl_5 = above_incl_5 + nxt_5;
        }
        unsigned int _shfl_down_26 = __shfl_down_sync(0xFFFFFFFF, above_incl_5, 2, 32);
        unsigned int nxt_7_5 = _shfl_down_26;
        if (lane + 2 < 32) {
            above_incl_5 = above_incl_5 + nxt_7_5;
        }
        unsigned int _shfl_down_27 = __shfl_down_sync(0xFFFFFFFF, above_incl_5, 4, 32);
        unsigned int nxt_8_5 = _shfl_down_27;
        if (lane + 4 < 32) {
            above_incl_5 = above_incl_5 + nxt_8_5;
        }
        unsigned int _shfl_down_28 = __shfl_down_sync(0xFFFFFFFF, above_incl_5, 8, 32);
        unsigned int nxt_9_5 = _shfl_down_28;
        if (lane + 8 < 32) {
            above_incl_5 = above_incl_5 + nxt_9_5;
        }
        unsigned int _shfl_down_29 = __shfl_down_sync(0xFFFFFFFF, above_incl_5, 16, 32);
        unsigned int nxt_10_5 = _shfl_down_29;
        if (lane + 16 < 32) {
            above_incl_5 = above_incl_5 + nxt_10_5;
        }
        unsigned int running_5 = above_incl_5 - lane_total_5;
        unsigned int hit_5 = 4294967295u;
        unsigned int at_5 = running_5 + bins_5[7];
        if (running_5 < remaining) {
            if (at_5 >= remaining) {
                hit_5 = (unsigned int)(lane * 8 + 7) << 16 | remaining - running_5;
                if (at_5 == remaining) {
                    hit_5 = hit_5 | 2147483648u;
                }
            }
        }
        running_5 = at_5;
        unsigned int at_11_5 = running_5 + bins_5[6];
        if (running_5 < remaining) {
            if (at_11_5 >= remaining) {
                hit_5 = (unsigned int)(lane * 8 + 6) << 16 | remaining - running_5;
                if (at_11_5 == remaining) {
                    hit_5 = hit_5 | 2147483648u;
                }
            }
        }
        running_5 = at_11_5;
        unsigned int at_12_5 = running_5 + bins_5[5];
        if (running_5 < remaining) {
            if (at_12_5 >= remaining) {
                hit_5 = (unsigned int)(lane * 8 + 5) << 16 | remaining - running_5;
                if (at_12_5 == remaining) {
                    hit_5 = hit_5 | 2147483648u;
                }
            }
        }
        running_5 = at_12_5;
        unsigned int at_13_5 = running_5 + bins_5[4];
        if (running_5 < remaining) {
            if (at_13_5 >= remaining) {
                hit_5 = (unsigned int)(lane * 8 + 4) << 16 | remaining - running_5;
                if (at_13_5 == remaining) {
                    hit_5 = hit_5 | 2147483648u;
                }
            }
        }
        running_5 = at_13_5;
        unsigned int at_14_5 = running_5 + bins_5[3];
        if (running_5 < remaining) {
            if (at_14_5 >= remaining) {
                hit_5 = (unsigned int)(lane * 8 + 3) << 16 | remaining - running_5;
                if (at_14_5 == remaining) {
                    hit_5 = hit_5 | 2147483648u;
                }
            }
        }
        running_5 = at_14_5;
        unsigned int at_15_5 = running_5 + bins_5[2];
        if (running_5 < remaining) {
            if (at_15_5 >= remaining) {
                hit_5 = (unsigned int)(lane * 8 + 2) << 16 | remaining - running_5;
                if (at_15_5 == remaining) {
                    hit_5 = hit_5 | 2147483648u;
                }
            }
        }
        running_5 = at_15_5;
        unsigned int at_16_5 = running_5 + bins_5[1];
        if (running_5 < remaining) {
            if (at_16_5 >= remaining) {
                hit_5 = (unsigned int)(lane * 8 + 1) << 16 | remaining - running_5;
                if (at_16_5 == remaining) {
                    hit_5 = hit_5 | 2147483648u;
                }
            }
        }
        running_5 = at_16_5;
        unsigned int at_17_5 = running_5 + bins_5[0];
        if (running_5 < remaining) {
            if (at_17_5 >= remaining) {
                hit_5 = (unsigned int)(lane * 8) << 16 | remaining - running_5;
                if (at_17_5 == remaining) {
                    hit_5 = hit_5 | 2147483648u;
                }
            }
        }
        running_5 = at_17_5;
        unsigned int _vote_22 = __ballot_sync(0xFFFFFFFF, hit_5 != 4294967295u);
        unsigned int hit_mask_5 = _vote_22;
        int _ffs_5 = __ffs(hit_mask_5);
        unsigned int src_lane_5 = (unsigned int)(_ffs_5 - 1) & 31;
        unsigned int _shfl_10 = __shfl_sync(0xFFFFFFFF, hit_5, src_lane_5);
        unsigned int chosen_5 = _shfl_10;
        hist[lane * 8] = 0;
        hist[lane * 8 + 1] = 0;
        hist[lane * 8 + 2] = 0;
        hist[lane * 8 + 3] = 0;
        hist[lane * 8 + 4] = 0;
        hist[lane * 8 + 5] = 0;
        hist[lane * 8 + 6] = 0;
        hist[lane * 8 + 7] = 0;
        if (lane == 0) {
            state[0] = chosen_5;
            state[1] = above_incl_5;
        }
    }
    __syncthreads();
    unsigned int chosen1_52 = state[0];
    if (done == 0) {
        prefix = prefix | chosen1_52 >> 16 & 255;
        remaining = chosen1_52 & 65535;
        if ((chosen1_52 & 2147483648u) != 0) {
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
    if (quota == 0) {
        #pragma unroll 8
        for (int cp = 0; cp < n_chunks_fb; cp++) {
            int idxp = cp * 512 + tid;
            int idxc_p = ((idxp < limit) ? idxp : 0);
            float x_p = logits[row_base + (unsigned long long)idxc_p];
            unsigned int bits_p = __as_u32(x_p);
            bits_p = ((bits_p == 2147483648u) ? 0 : bits_p);
            unsigned int sign_p = (unsigned int)((int)bits_p >> 31);
            unsigned int enc_p = bits_p ^ (sign_p | 2147483648u);
            bool sel_p = enc_p - 8388608 <= 4278190080u;
            unsigned int kp = ((sel_p) ? enc_p : 0);
            kp = ((idxp < limit) ? kp : 0);
            bool is_selp = kp > pivot;
            unsigned int _vote_23 = __ballot_sync(0xFFFFFFFF, is_selp);
            unsigned int sel_maskp = _vote_23;
            int _popc_50 = __popc(sel_maskp);
            unsigned int n_selp = (unsigned int)_popc_50;
            unsigned int _atomic_old_5 = 0;
            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.u32 %0, [%1], %2;\n}"
                : "+r"(_atomic_old_5) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&state[2]))), "r"(static_cast<unsigned int>(n_selp)), "r"(static_cast<uint32_t>((unsigned int)(lane == 0))) : "memory");
            unsigned int basep = _atomic_old_5;
            unsigned int _shfl_11 = __shfl_sync(0xFFFFFFFF, basep, 0);
            basep = _shfl_11;
            if (is_selp) {
                unsigned int inv_idxp = (unsigned int)idxp ^ 4294967295u;
                unsigned long long compositep = (unsigned long long)kp << 32 | (unsigned long long)inv_idxp;
                int _popc_51 = __popc(sel_maskp & lt_mask);
                pairs[basep + (unsigned int)_popc_51] = compositep;
            }
        }
        #pragma unroll 8
        for (int cc = 0; cc < n_cchunks; cc++) {
            int icp = cc * 512 + tid;
            unsigned long long compc = cand[icp];
            unsigned int kcp = (unsigned int)(compc >> 32);
            bool is_selc = kcp > pivot;
            unsigned int _vote_24 = __ballot_sync(0xFFFFFFFF, is_selc);
            unsigned int sel_maskc = _vote_24;
            int _popc_52 = __popc(sel_maskc);
            unsigned int n_selc = (unsigned int)_popc_52;
            unsigned int _atomic_old_6 = 0;
            asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %3, 0;\n\t@p atom.shared.add.u32 %0, [%1], %2;\n}"
                : "+r"(_atomic_old_6) : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&state[2]))), "r"(static_cast<unsigned int>(n_selc)), "r"(static_cast<uint32_t>((unsigned int)(lane == 0))) : "memory");
            unsigned int basec = _atomic_old_6;
            unsigned int _shfl_12 = __shfl_sync(0xFFFFFFFF, basec, 0);
            basec = _shfl_12;
            if (is_selc) {
                int _popc_53 = __popc(sel_maskc & lt_mask);
                pairs[basec + (unsigned int)_popc_53] = compc;
            }
        }
    } else {
        #pragma unroll 8
        for (int c = 0; c < n_chunks_rt; c++) {
            int idx2 = c * 512 + tid;
            int idxc_2 = ((idx2 < limit) ? idx2 : 0);
            float x_2 = logits[row_base + (unsigned long long)idxc_2];
            unsigned int bits_2 = __as_u32(x_2);
            bits_2 = ((bits_2 == 2147483648u) ? 0 : bits_2);
            unsigned int sign_2 = (unsigned int)((int)bits_2 >> 31);
            unsigned int enc_2 = bits_2 ^ (sign_2 | 2147483648u);
            bool sel_2 = enc_2 - 8388608 <= 4278190080u;
            unsigned int k2 = ((sel_2) ? enc_2 : 0);
            k2 = ((idx2 < limit) ? k2 : 0);
            unsigned int _vote_25 = __ballot_sync(0xFFFFFFFF, k2 > pivot);
            unsigned int sel_mask = _vote_25;
            unsigned int _vote_26 = __ballot_sync(0xFFFFFFFF, k2 == pivot);
            unsigned int tie_mask = _vote_26;
            if (lane == 0) {
                int slot = c * 16 + warp;
                int _popc_54 = __popc(sel_mask);
                cnt_sel[slot] = (unsigned int)_popc_54;
                int _popc_55 = __popc(tie_mask);
                cnt_tie[slot] = (unsigned int)_popc_55;
            }
        }
    }
    __syncthreads();
    unsigned int local_sel = 0;
    unsigned int local_tie = 0;
    unsigned int incl_sel = 0;
    unsigned int incl_tie = 0;
    if (quota != 0) {
        int f = tid * 2;
        if (f < n_slots_rt) {
            local_sel = local_sel + cnt_sel[f];
            local_tie = local_tie + cnt_tie[f];
        }
        int f_0 = tid * 2 + 1;
        if (f_0 < n_slots_rt) {
            local_sel = local_sel + cnt_sel[f_0];
            local_tie = local_tie + cnt_tie[f_0];
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
        unsigned int up_sel_1 = _shfl_up_2;
        unsigned int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 2, 32);
        unsigned int up_tie_2 = _shfl_up_3;
        if (lane >= 2) {
            incl_sel = incl_sel + up_sel_1;
            incl_tie = incl_tie + up_tie_2;
        }
        unsigned int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 4, 32);
        unsigned int up_sel_3 = _shfl_up_4;
        unsigned int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 4, 32);
        unsigned int up_tie_4 = _shfl_up_5;
        if (lane >= 4) {
            incl_sel = incl_sel + up_sel_3;
            incl_tie = incl_tie + up_tie_4;
        }
        unsigned int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 8, 32);
        unsigned int up_sel_5 = _shfl_up_6;
        unsigned int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 8, 32);
        unsigned int up_tie_6 = _shfl_up_7;
        if (lane >= 8) {
            incl_sel = incl_sel + up_sel_5;
            incl_tie = incl_tie + up_tie_6;
        }
        unsigned int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, incl_sel, 16, 32);
        unsigned int up_sel_7 = _shfl_up_8;
        unsigned int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, incl_tie, 16, 32);
        unsigned int up_tie_8 = _shfl_up_9;
        if (lane >= 16) {
            incl_sel = incl_sel + up_sel_7;
            incl_tie = incl_tie + up_tie_8;
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
        int f2 = tid * 2;
        if (f2 < n_slots_rt) {
            unsigned int cur_sel = cnt_sel[f2];
            unsigned int cur_tie = cnt_tie[f2];
            cnt_sel[f2] = run_sel;
            cnt_tie[f2] = run_tie;
            run_sel = run_sel + cur_sel;
            run_tie = run_tie + cur_tie;
        }
        int f2_30 = tid * 2 + 1;
        if (f2_30 < n_slots_rt) {
            unsigned int cur_sel_1 = cnt_sel[f2_30];
            unsigned int cur_tie_1 = cnt_tie[f2_30];
            cnt_sel[f2_30] = run_sel;
            cnt_tie[f2_30] = run_tie;
            run_sel = run_sel + cur_sel_1;
            run_tie = run_tie + cur_tie_1;
        }
    }
    __syncthreads();
    if (quota != 0) {
        #pragma unroll 8
        for (int c2 = 0; c2 < n_chunks_rt; c2++) {
            int idx3 = c2 * 512 + tid;
            int idxc_3 = ((idx3 < limit) ? idx3 : 0);
            float x_3 = logits[row_base + (unsigned long long)idxc_3];
            unsigned int bits_3 = __as_u32(x_3);
            bits_3 = ((bits_3 == 2147483648u) ? 0 : bits_3);
            unsigned int sign_3 = (unsigned int)((int)bits_3 >> 31);
            unsigned int enc_3 = bits_3 ^ (sign_3 | 2147483648u);
            bool sel_3 = enc_3 - 8388608 <= 4278190080u;
            unsigned int k3 = ((sel_3) ? enc_3 : 0);
            k3 = ((idx3 < limit) ? k3 : 0);
            bool is_sel = k3 > pivot;
            bool is_tie = k3 == pivot;
            unsigned int _vote_27 = __ballot_sync(0xFFFFFFFF, is_sel);
            unsigned int sel_mask2 = _vote_27;
            unsigned int _vote_28 = __ballot_sync(0xFFFFFFFF, is_tie);
            unsigned int tie_mask2 = _vote_28;
            int slot2 = c2 * 16 + warp;
            unsigned int inv_idx = (unsigned int)idx3 ^ 4294967295u;
            unsigned long long composite = (unsigned long long)k3 << 32 | (unsigned long long)inv_idx;
            if (is_sel) {
                int _popc_56 = __popc(sel_mask2 & lt_mask);
                unsigned int pos = cnt_sel[slot2] + (unsigned int)_popc_56;
                pairs[pos] = composite;
            } else if (is_tie) {
                int _popc_57 = __popc(tie_mask2 & lt_mask);
                unsigned int rank = cnt_tie[slot2] + (unsigned int)_popc_57;
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
