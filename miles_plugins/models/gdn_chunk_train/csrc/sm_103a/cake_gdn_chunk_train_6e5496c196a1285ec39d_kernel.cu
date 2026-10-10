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
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_gdn_chunk_train_6e5496c196a1285ec39d(int* __restrict__ cu_seqlens, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seq_chunk_start, int num_seqs, int num_chunks_max, int total_tokens)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int tid_0 = tid;
    if (tid_0 == 0) {
        int acc = 0;
        seq_chunk_start[0] = 0;
        #pragma unroll 1
        for (int s0 = 0; s0 < num_seqs; s0++) {
            int seq_len = cu_seqlens[s0 + 1] - cu_seqlens[s0];
            acc = acc + (seq_len + 63) / 64;
            seq_chunk_start[s0 + 1] = acc;
        }
    }
    __syncthreads();
    int total = seq_chunk_start[num_seqs];
    #pragma unroll 1
    for (int it = 0; it < (num_seqs + 127) / 128; it++) {
        int seq = it * 128 + tid_0;
        if (seq < num_seqs) {
            int c0 = seq_chunk_start[seq];
            int n_chunks = seq_chunk_start[seq + 1] - c0;
            int seq_tok0 = cu_seqlens[seq];
            int seq_end = cu_seqlens[seq + 1];
            #pragma unroll 1
            for (int j = 0; j < n_chunks; j++) {
                int start = seq_tok0 + j * 64;
                chunk_start[c0 + j] = start;
                int _min_0 = ((seq_end - start) < (64) ? (seq_end - start) : (64));
                chunk_len[c0 + j] = _min_0;
            }
        }
    }
    #pragma unroll 1
    for (int it2 = 0; it2 < (num_chunks_max + 127) / 128; it2++) {
        int pad = total + it2 * 128 + tid_0;
        if (pad < num_chunks_max) {
            chunk_start[pad] = total_tokens;
            chunk_len[pad] = 0;
        }
    }
}

} // extern "C"
