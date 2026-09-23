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
#define SMEM_S_W_OFF 0
#define SMEM_S_W_STAGE_BYTES 16384
#define SMEM_S_W_STRIDE 40960
#define SMEM_S_KN_OFF 16384
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 40960
#define SMEM_S_U_OFF 32768
#define SMEM_S_U_STAGE_BYTES 8192
#define SMEM_S_U_STRIDE 40960
#define SMEM_S_G_OFF 81920
#define SMEM_S_G_STAGE_BYTES 512
#define SMEM_S_G_STRIDE 512
#define SMEM_S_STAGE_OFF 82432
#define SMEM_S_STAGE_STAGE_BYTES 8192
#define SMEM_S_STAGE_STRIDE 8192
#define SMEM_S_RED_OFF 90624
#define SMEM_S_RED_STAGE_BYTES 16384
#define SMEM_S_RED_STRIDE 16384
#define SMEM_TOTAL 107008
#define THREADS 128

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void fma_f32x2_noftz_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("add.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("sub.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 add_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_gdn_chunk_train_3bb5ea7c18d7965632f7(__nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ u, float* __restrict__ g_cs, float* __restrict__ h0, __nv_bfloat16* __restrict__ h_out, __nv_bfloat16* __restrict__ v_new, float* __restrict__ final_state, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int* __restrict__ seq_chunk_start, int num_heads, int num_v_heads, int use_initial_state, int store_final_state)
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
    __nv_bfloat16* s_w = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_w_addr = smem + 0;
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_kn_addr = smem + 16384;
    __nv_bfloat16* s_u = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_u_addr = smem + 32768;
    float* s_g = reinterpret_cast<float*>(smem_raw + 81920);
    const int s_g_addr = smem + 81920;
    int* s_stage = reinterpret_cast<int*>(smem_raw + 82432);
    const int s_stage_addr = smem + 82432;
    float* s_red = reinterpret_cast<float*>(smem_raw + 90624);
    const int s_red_addr = smem + 90624;

    // === Task calls (dependency order) ===
    int seq = blockIdx.x;
    int hv = blockIdx.y;
    int v0 = blockIdx.z * 32;
    int c_begin = seq_chunk_start[seq];
    int c_end = seq_chunk_start[seq + 1];
    int group = num_v_heads / num_heads;
    int hq = hv / group;
    int lane_0 = lane;
    int warp_1 = warp;
    int tid_2 = tid;
    int quad = lane_0 % 4;
    int vwarp = warp_1 / 2;
    int khalf = warp_1 % 2;
    int kbase = khalf * 64;
    int vloc_a = vwarp * 16 + lane_0 / 4;
    int vrow_a = v0 + vloc_a;
    int vrow_b = vrow_a + 8;
    long long tok_stride = (long long)num_v_heads * 128;
    int stage_base = s_stage_addr + (unsigned int)(warp_1 * 2048);
    int stage_u32_base = warp_1 * 512;
    unsigned int vw[16];
    float acc_h[32];
    float acc_v[32];
    unsigned int hpk[16];
    unsigned int vpk[16];
    unsigned int u_frag[4];
    unsigned int b_frag_k[4];
    unsigned int b_frag_mn[4];
    float g_next[1];
    float g_far[1];
    long long state_base = ((long long)seq * (long long)num_v_heads + (long long)hv) * 16384;
    acc_h[0] = 0.0f;
    acc_h[1] = 0.0f;
    acc_h[2] = 0.0f;
    acc_h[3] = 0.0f;
    acc_h[4] = 0.0f;
    acc_h[5] = 0.0f;
    acc_h[6] = 0.0f;
    acc_h[7] = 0.0f;
    acc_h[8] = 0.0f;
    acc_h[9] = 0.0f;
    acc_h[10] = 0.0f;
    acc_h[11] = 0.0f;
    acc_h[12] = 0.0f;
    acc_h[13] = 0.0f;
    acc_h[14] = 0.0f;
    acc_h[15] = 0.0f;
    acc_h[16] = 0.0f;
    acc_h[17] = 0.0f;
    acc_h[18] = 0.0f;
    acc_h[19] = 0.0f;
    acc_h[20] = 0.0f;
    acc_h[21] = 0.0f;
    acc_h[22] = 0.0f;
    acc_h[23] = 0.0f;
    acc_h[24] = 0.0f;
    acc_h[25] = 0.0f;
    acc_h[26] = 0.0f;
    acc_h[27] = 0.0f;
    acc_h[28] = 0.0f;
    acc_h[29] = 0.0f;
    acc_h[30] = 0.0f;
    acc_h[31] = 0.0f;
    if (use_initial_state != 0) {
        #pragma unroll
        for (int nb = 0; nb < 8; nb++) {
            int kcol = kbase + nb * 8 + quad * 2;
            acc_h[nb * 4] = h0[state_base + (long long)kcol * 128 + (long long)vrow_a];
            acc_h[nb * 4 + 1] = h0[state_base + (long long)(kcol + 1) * 128 + (long long)vrow_a];
            acc_h[nb * 4 + 2] = h0[state_base + (long long)kcol * 128 + (long long)vrow_b];
            acc_h[nb * 4 + 3] = h0[state_base + (long long)(kcol + 1) * 128 + (long long)vrow_b];
        }
    }
    int num_chunks = c_end - c_begin;
    int seq_tok0 = chunk_start[c_begin];
    int seq_len = chunk_start[c_end - 1] + chunk_len[c_end - 1] - seq_tok0;
    int stage_off = 0;
    int g_off = 0;
    int _min_0 = ((seq_len) < (64) ? (seq_len) : (64));
    int r0 = tid / 16;
    int c0 = tid % 16 * 8;
    long long row_stride = (long long)num_v_heads * 128;
    long long src_row = (long long)(seq_tok0 + r0) * row_stride + (long long)hv * 128 + (long long)c0;
    int dst0 = s_w_addr + (unsigned int)stage_off + (unsigned int)(c0 / 64 * 8192 + (r0 * 128 + c0 % 64 * 2 ^ (r0 * 128 + c0 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0), "l"(w + src_row), "r"((_min_0 > r0) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 1024), "l"(w + (src_row + 8 * row_stride)), "r"((_min_0 > r0 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 2048), "l"(w + (src_row + 16 * row_stride)), "r"((_min_0 > r0 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 3072), "l"(w + (src_row + 24 * row_stride)), "r"((_min_0 > r0 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 4096), "l"(w + (src_row + 32 * row_stride)), "r"((_min_0 > r0 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 5120), "l"(w + (src_row + 40 * row_stride)), "r"((_min_0 > r0 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 6144), "l"(w + (src_row + 48 * row_stride)), "r"((_min_0 > r0 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0 + 7168), "l"(w + (src_row + 56 * row_stride)), "r"((_min_0 > r0 + 56) ? 16 : 0));
    int r0_3 = tid / 16;
    int c0_4 = tid % 16 * 8;
    long long row_stride_5 = (long long)num_heads * 128;
    long long src_row_6 = (long long)(seq_tok0 + r0_3) * row_stride_5 + (long long)hq * 128 + (long long)c0_4;
    int dst0_7 = s_kn_addr + (unsigned int)stage_off + (unsigned int)(c0_4 / 64 * 8192 + (r0_3 * 128 + c0_4 % 64 * 2 ^ (r0_3 * 128 + c0_4 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7), "l"(kn + src_row_6), "r"((_min_0 > r0_3) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 1024), "l"(kn + (src_row_6 + 8 * row_stride_5)), "r"((_min_0 > r0_3 + 8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 2048), "l"(kn + (src_row_6 + 16 * row_stride_5)), "r"((_min_0 > r0_3 + 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 3072), "l"(kn + (src_row_6 + 24 * row_stride_5)), "r"((_min_0 > r0_3 + 24) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 4096), "l"(kn + (src_row_6 + 32 * row_stride_5)), "r"((_min_0 > r0_3 + 32) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 5120), "l"(kn + (src_row_6 + 40 * row_stride_5)), "r"((_min_0 > r0_3 + 40) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 6144), "l"(kn + (src_row_6 + 48 * row_stride_5)), "r"((_min_0 > r0_3 + 48) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_7 + 7168), "l"(kn + (src_row_6 + 56 * row_stride_5)), "r"((_min_0 > r0_3 + 56) ? 16 : 0));
    int r0_8 = tid / 4;
    int c0_9 = tid % 4 * 8;
    long long row_stride_10 = (long long)num_v_heads * 128;
    long long src_row_11 = (long long)(seq_tok0 + r0_8) * row_stride_10 + (long long)hv * 128 + (long long)v0 + (long long)c0_9;
    int dst0_12 = s_u_addr + (unsigned int)stage_off + (unsigned int)(c0_9 / 64 * 8192 + (r0_8 * 128 + c0_9 % 64 * 2 ^ (r0_8 * 128 + c0_9 % 64 * 2 >> 7 & 7) << 4));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12), "l"(u + src_row_11), "r"((_min_0 > r0_8) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(dst0_12 + 4096), "l"(u + (src_row_11 + 32 * row_stride_10)), "r"((_min_0 > r0_8 + 32) ? 16 : 0));
    asm volatile("cp.async.commit_group;");
    int _min_1 = ((seq_len) < (64) ? (seq_len) : (64));
    int tok = tid;
    if (tok < 64) {
        float gate_val = 0.0f;
        if (tok < _min_1) {
            gate_val = g_cs[(long long)(seq_tok0 + tok) * (long long)num_v_heads + (long long)hv];
        }
        s_g[tok] = gate_val;
    }
    int _min_2 = ((seq_len - 64) < (64) ? (seq_len - 64) : (64));
    int tok_13 = tid;
    g_next[0] = 0.0f;
    if (tok_13 < _min_2) {
        g_next[0] = g_cs[(long long)(seq_tok0 + 64 + tok_13) * (long long)num_v_heads + (long long)hv];
    }
    #pragma unroll 1
    for (int it_chunk = 0; it_chunk < num_chunks; it_chunk++) {
        int chunk = c_begin + it_chunk;
        int tok0 = seq_tok0 + it_chunk * 64;
        int _min_3 = ((seq_len - it_chunk * 64) < (64) ? (seq_len - it_chunk * 64) : (64));
        int n_valid = _min_3;
        int next_off = 40960 - stage_off;
        int next_g_off = 64 - g_off;
        int _min_4 = ((seq_len - (it_chunk + 2) * 64) < (64) ? (seq_len - (it_chunk + 2) * 64) : (64));
        int tok_0 = tid;
        g_far[0] = 0.0f;
        if (tok_0 < _min_4) {
            g_far[0] = g_cs[(long long)(tok0 + 128 + tok_0) * (long long)num_v_heads + (long long)hv];
        }
        if (num_chunks > it_chunk + 1) {
            int next_tok0 = tok0 + 64;
            int _min_5 = ((seq_len - (it_chunk + 1) * 64) < (64) ? (seq_len - (it_chunk + 1) * 64) : (64));
            int next_len = _min_5;
            int r0_0 = tid / 16;
            int c0_1 = tid % 16 * 8;
            long long row_stride_2 = (long long)num_v_heads * 128;
            long long src_row_3 = (long long)(next_tok0 + r0_0) * row_stride_2 + (long long)hv * 128 + (long long)c0_1;
            int dst0_4 = s_w_addr + (unsigned int)next_off + (unsigned int)(c0_1 / 64 * 8192 + (r0_0 * 128 + c0_1 % 64 * 2 ^ (r0_0 * 128 + c0_1 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4), "l"(w + src_row_3), "r"((next_len > r0_0) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 1024), "l"(w + (src_row_3 + 8 * row_stride_2)), "r"((next_len > r0_0 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 2048), "l"(w + (src_row_3 + 16 * row_stride_2)), "r"((next_len > r0_0 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 3072), "l"(w + (src_row_3 + 24 * row_stride_2)), "r"((next_len > r0_0 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 4096), "l"(w + (src_row_3 + 32 * row_stride_2)), "r"((next_len > r0_0 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 5120), "l"(w + (src_row_3 + 40 * row_stride_2)), "r"((next_len > r0_0 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 6144), "l"(w + (src_row_3 + 48 * row_stride_2)), "r"((next_len > r0_0 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_4 + 7168), "l"(w + (src_row_3 + 56 * row_stride_2)), "r"((next_len > r0_0 + 56) ? 16 : 0));
            int r0_5 = tid / 16;
            int c0_6 = tid % 16 * 8;
            long long row_stride_7 = (long long)num_heads * 128;
            long long src_row_8 = (long long)(next_tok0 + r0_5) * row_stride_7 + (long long)hq * 128 + (long long)c0_6;
            int dst0_9 = s_kn_addr + (unsigned int)next_off + (unsigned int)(c0_6 / 64 * 8192 + (r0_5 * 128 + c0_6 % 64 * 2 ^ (r0_5 * 128 + c0_6 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9), "l"(kn + src_row_8), "r"((next_len > r0_5) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 1024), "l"(kn + (src_row_8 + 8 * row_stride_7)), "r"((next_len > r0_5 + 8) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 2048), "l"(kn + (src_row_8 + 16 * row_stride_7)), "r"((next_len > r0_5 + 16) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 3072), "l"(kn + (src_row_8 + 24 * row_stride_7)), "r"((next_len > r0_5 + 24) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 4096), "l"(kn + (src_row_8 + 32 * row_stride_7)), "r"((next_len > r0_5 + 32) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 5120), "l"(kn + (src_row_8 + 40 * row_stride_7)), "r"((next_len > r0_5 + 40) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 6144), "l"(kn + (src_row_8 + 48 * row_stride_7)), "r"((next_len > r0_5 + 48) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_9 + 7168), "l"(kn + (src_row_8 + 56 * row_stride_7)), "r"((next_len > r0_5 + 56) ? 16 : 0));
            int r0_10 = tid / 4;
            int c0_11 = tid % 4 * 8;
            long long row_stride_12 = (long long)num_v_heads * 128;
            long long src_row_13 = (long long)(next_tok0 + r0_10) * row_stride_12 + (long long)hv * 128 + (long long)v0 + (long long)c0_11;
            int dst0_14 = s_u_addr + (unsigned int)next_off + (unsigned int)(c0_11 / 64 * 8192 + (r0_10 * 128 + c0_11 % 64 * 2 ^ (r0_10 * 128 + c0_11 % 64 * 2 >> 7 & 7) << 4));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14), "l"(u + src_row_13), "r"((next_len > r0_10) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(dst0_14 + 4096), "l"(u + (src_row_13 + 32 * row_stride_12)), "r"((next_len > r0_10 + 32) ? 16 : 0));
            asm volatile("cp.async.commit_group;");
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        #pragma unroll
        for (int _lp = 0; _lp < 16; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_h[_lp*2 + 0], acc_h[_lp*2+1 + 0]));
            hpk[_lp] = *(uint32_t*)&_bf2;
        }
        __syncthreads();
        float g_last = s_g[g_off + n_valid - 1];
        float _exp2_0 = approx_exp2(g_last);
        const float2 _scale2_0 = {_exp2_0, _exp2_0};
        #pragma unroll
        for (int _ls = 0; _ls < 16; _ls++)
            mul_f32x2_inplace(&reinterpret_cast<float2*>(acc_h)[_ls], _scale2_0);
        #pragma unroll
        for (int ks = 0; ks < ((0) ? 0 : 4); ks++) {
            #pragma unroll
            for (int nb3 = 0; nb3 < 4; nb3++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                    : "r"(s_w_addr + (unsigned int)stage_off + (unsigned int)((kbase + ks * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((nb3 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kbase + ks * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((nb3 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kbase + ks * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_v + nb3 * 8)[0]), "=f"((acc_v + nb3 * 8)[1]), "=f"((acc_v + nb3 * 8)[2]), "=f"((acc_v + nb3 * 8)[3])
                    : "r"((hpk + ks * 4)[0]), "r"((hpk + ks * 4)[1]), "r"((hpk + ks * 4)[2]), "r"((hpk + ks * 4)[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[0])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[1])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[2])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8)[3])));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                    : "=f"((acc_v + nb3 * 8 + 4)[0]), "=f"((acc_v + nb3 * 8 + 4)[1]), "=f"((acc_v + nb3 * 8 + 4)[2]), "=f"((acc_v + nb3 * 8 + 4)[3])
                    : "r"((hpk + ks * 4)[0]), "r"((hpk + ks * 4)[1]), "r"((hpk + ks * 4)[2]), "r"((hpk + ks * 4)[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[0])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[1])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[2])), "f"(((ks == 0) ? 0.0f : (acc_v + nb3 * 8 + 4)[3])));
            }
        }
        long long tape_base = ((long long)chunk * (long long)num_v_heads + (long long)hv) * 16384 + (long long)(v0 + vwarp * 16) * 128 + (long long)kbase;
        int m_idx = lane_0 / 8;
        int r_idx = lane_0 % 8;
        int st_row = m_idx % 2 * 8 + r_idx;
        int st_nb = m_idx / 2;
        uint32_t _stmatrix_addr_1 = static_cast<uint32_t>(stage_base + st_row * 128 + ((st_nb ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_1), "r"(*reinterpret_cast<const uint32_t*>(&hpk[0])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[1])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[2])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[3]))
            : "memory");
        int st_nb_1 = 2 + m_idx / 2;
        uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(stage_base + st_row * 128 + ((st_nb_1 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&hpk[4])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[5])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[6])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[7]))
            : "memory");
        int st_nb_2 = 4 + m_idx / 2;
        uint32_t _stmatrix_addr_3 = static_cast<uint32_t>(stage_base + st_row * 128 + ((st_nb_2 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_3), "r"(*reinterpret_cast<const uint32_t*>(&hpk[8])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[9])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[10])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[11]))
            : "memory");
        int st_nb_3 = 6 + m_idx / 2;
        uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(stage_base + st_row * 128 + ((st_nb_3 ^ r_idx) << 4));
        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
            :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&hpk[12])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[13])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[14])), "r"(*reinterpret_cast<const uint32_t*>(&hpk[15]))
            : "memory");
        __syncwarp();
        int ld_row_lo = lane_0 / 8;
        int ld_ch = lane_0 % 8;
        int ld_row = ld_row_lo;
        int words[4];
        {
            const int4* _ivptr_5 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row * 32 + ((ld_ch ^ ld_row % 8) << 2));
            int4 _ivld_5;
            _ivld_5 = *_ivptr_5;
            words[0 + 0] = _ivld_5.x;
            words[0 + 1] = _ivld_5.y;
            words[0 + 2] = _ivld_5.z;
            words[0 + 3] = _ivld_5.w;
        }
        {
            int4 _iv4 = make_int4(words[0 + 0], words[0 + 1], words[0 + 2], words[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_4 = 4 + ld_row_lo;
        int words_5[4];
        {
            const int4* _ivptr_6 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_4 * 32 + ((ld_ch ^ ld_row_4 % 8) << 2));
            int4 _ivld_6;
            _ivld_6 = *_ivptr_6;
            words_5[0 + 0] = _ivld_6.x;
            words_5[0 + 1] = _ivld_6.y;
            words_5[0 + 2] = _ivld_6.z;
            words_5[0 + 3] = _ivld_6.w;
        }
        {
            int4 _iv4 = make_int4(words_5[0 + 0], words_5[0 + 1], words_5[0 + 2], words_5[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_4 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_6 = 8 + ld_row_lo;
        int words_7[4];
        {
            const int4* _ivptr_7 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_6 * 32 + ((ld_ch ^ ld_row_6 % 8) << 2));
            int4 _ivld_7;
            _ivld_7 = *_ivptr_7;
            words_7[0 + 0] = _ivld_7.x;
            words_7[0 + 1] = _ivld_7.y;
            words_7[0 + 2] = _ivld_7.z;
            words_7[0 + 3] = _ivld_7.w;
        }
        {
            int4 _iv4 = make_int4(words_7[0 + 0], words_7[0 + 1], words_7[0 + 2], words_7[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_6 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        int ld_row_8 = 12 + ld_row_lo;
        int words_9[4];
        {
            const int4* _ivptr_8 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_row_8 * 32 + ((ld_ch ^ ld_row_8 % 8) << 2));
            int4 _ivld_8;
            _ivld_8 = *_ivptr_8;
            words_9[0 + 0] = _ivld_8.x;
            words_9[0 + 1] = _ivld_8.y;
            words_9[0 + 2] = _ivld_8.z;
            words_9[0 + 3] = _ivld_8.w;
        }
        {
            int4 _iv4 = make_int4(words_9[0 + 0], words_9[0 + 1], words_9[0 + 2], words_9[0 + 3]);
            *reinterpret_cast<int4*>(reinterpret_cast<int*>(h_out) + (tape_base + (long long)(ld_row_8 * 128) + (long long)(ld_ch * 8)) / 2) = _iv4;
        }
        __syncwarp();
        int own_base = warp_1 * 1024 + lane_0 * 4;
        int partner_base = (warp_1 ^ 1) * 1024 + lane_0 * 4;
        {
            float4 _v4 = make_float4(acc_v[0 + 0], acc_v[0 + 1], acc_v[0 + 2], acc_v[0 + 3]);
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(s_red) + own_base) = _v4;
        }
        {
            float4 _v4 = make_float4(acc_v[4 + 0], acc_v[4 + 1], acc_v[4 + 2], acc_v[4 + 3]);
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(s_red) + own_base + 128) = _v4;
        }
        {
            float4 _v4 = make_float4(acc_v[8 + 0], acc_v[8 + 1], acc_v[8 + 2], acc_v[8 + 3]);
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(s_red) + own_base + 256) = _v4;
        }
        {
            float4 _v4 = make_float4(acc_v[12 + 0], acc_v[12 + 1], acc_v[12 + 2], acc_v[12 + 3]);
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(s_red) + own_base + 384) = _v4;
        }
        {
            float4 _v4 = make_float4(acc_v[16 + 0], acc_v[16 + 1], acc_v[16 + 2], acc_v[16 + 3]);
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(s_red) + own_base + 512) = _v4;
        }
        {
            float4 _v4 = make_float4(acc_v[20 + 0], acc_v[20 + 1], acc_v[20 + 2], acc_v[20 + 3]);
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(s_red) + own_base + 640) = _v4;
        }
        {
            float4 _v4 = make_float4(acc_v[24 + 0], acc_v[24 + 1], acc_v[24 + 2], acc_v[24 + 3]);
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(s_red) + own_base + 768) = _v4;
        }
        {
            float4 _v4 = make_float4(acc_v[28 + 0], acc_v[28 + 1], acc_v[28 + 2], acc_v[28 + 3]);
            *reinterpret_cast<float4*>(reinterpret_cast<float*>(s_red) + own_base + 896) = _v4;
        }
        __syncthreads();
        float part[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(s_red) + partner_base);
            part[0 + 0] = _v4.x;
            part[0 + 1] = _v4.y;
            part[0 + 2] = _v4.z;
            part[0 + 3] = _v4.w;
        }
        acc_v[0] = acc_v[0] + part[0];
        acc_v[1] = acc_v[1] + part[1];
        acc_v[2] = acc_v[2] + part[2];
        acc_v[3] = acc_v[3] + part[3];
        float part_10[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(s_red) + partner_base + 128);
            part_10[0 + 0] = _v4.x;
            part_10[0 + 1] = _v4.y;
            part_10[0 + 2] = _v4.z;
            part_10[0 + 3] = _v4.w;
        }
        acc_v[4] = acc_v[4] + part_10[0];
        acc_v[5] = acc_v[5] + part_10[1];
        acc_v[6] = acc_v[6] + part_10[2];
        acc_v[7] = acc_v[7] + part_10[3];
        float part_11[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(s_red) + partner_base + 256);
            part_11[0 + 0] = _v4.x;
            part_11[0 + 1] = _v4.y;
            part_11[0 + 2] = _v4.z;
            part_11[0 + 3] = _v4.w;
        }
        acc_v[8] = acc_v[8] + part_11[0];
        acc_v[9] = acc_v[9] + part_11[1];
        acc_v[10] = acc_v[10] + part_11[2];
        acc_v[11] = acc_v[11] + part_11[3];
        float part_12[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(s_red) + partner_base + 384);
            part_12[0 + 0] = _v4.x;
            part_12[0 + 1] = _v4.y;
            part_12[0 + 2] = _v4.z;
            part_12[0 + 3] = _v4.w;
        }
        acc_v[12] = acc_v[12] + part_12[0];
        acc_v[13] = acc_v[13] + part_12[1];
        acc_v[14] = acc_v[14] + part_12[2];
        acc_v[15] = acc_v[15] + part_12[3];
        float part_13[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(s_red) + partner_base + 512);
            part_13[0 + 0] = _v4.x;
            part_13[0 + 1] = _v4.y;
            part_13[0 + 2] = _v4.z;
            part_13[0 + 3] = _v4.w;
        }
        acc_v[16] = acc_v[16] + part_13[0];
        acc_v[17] = acc_v[17] + part_13[1];
        acc_v[18] = acc_v[18] + part_13[2];
        acc_v[19] = acc_v[19] + part_13[3];
        float part_14[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(s_red) + partner_base + 640);
            part_14[0 + 0] = _v4.x;
            part_14[0 + 1] = _v4.y;
            part_14[0 + 2] = _v4.z;
            part_14[0 + 3] = _v4.w;
        }
        acc_v[20] = acc_v[20] + part_14[0];
        acc_v[21] = acc_v[21] + part_14[1];
        acc_v[22] = acc_v[22] + part_14[2];
        acc_v[23] = acc_v[23] + part_14[3];
        float part_15[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(s_red) + partner_base + 768);
            part_15[0 + 0] = _v4.x;
            part_15[0 + 1] = _v4.y;
            part_15[0 + 2] = _v4.z;
            part_15[0 + 3] = _v4.w;
        }
        acc_v[24] = acc_v[24] + part_15[0];
        acc_v[25] = acc_v[25] + part_15[1];
        acc_v[26] = acc_v[26] + part_15[2];
        acc_v[27] = acc_v[27] + part_15[3];
        float part_16[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<float*>(s_red) + partner_base + 896);
            part_16[0 + 0] = _v4.x;
            part_16[0 + 1] = _v4.y;
            part_16[0 + 2] = _v4.z;
            part_16[0 + 3] = _v4.w;
        }
        acc_v[28] = acc_v[28] + part_16[0];
        acc_v[29] = acc_v[29] + part_16[1];
        acc_v[30] = acc_v[30] + part_16[2];
        acc_v[31] = acc_v[31] + part_16[3];
        float vn[4];
        #pragma unroll
        for (int tb = 0; tb < ((0) ? 0 : 4); tb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(u_frag[0]), "=r"(u_frag[1]), "=r"(u_frag[2]), "=r"(u_frag[3])
                : "r"(s_u_addr + (unsigned int)stage_off + (unsigned int)((vwarp * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((tb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((tb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (vwarp * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                #pragma unroll
                for (int e = 0; e < 2; e++) {
                    int tok_1 = (tb * 2 + half) * 8 + quad * 2 + e;
                    float decay = 0.0f;
                    if (tok_1 < n_valid) {
                        float _exp2_1 = approx_exp2(g_last - s_g[g_off + tok_1]);
                        decay = _exp2_1;
                    }
                    float u_a = ((e == 0) ? __uint_as_float(u_frag[half * 2] << 16) : __uint_as_float(u_frag[half * 2] & 4294901760));
                    float u_b = ((e == 0) ? __uint_as_float(u_frag[half * 2 + 1] << 16) : __uint_as_float(u_frag[half * 2 + 1] & 4294901760));
                    vn[e] = u_a - acc_v[(tb * 2 + half) * 4 + e];
                    vn[2 + e] = u_b - acc_v[(tb * 2 + half) * 4 + 2 + e];
                    acc_v[(tb * 2 + half) * 4 + e] = vn[e] * decay;
                    acc_v[(tb * 2 + half) * 4 + 2 + e] = vn[2 + e] * decay;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(vn[_lp*2 + 0], vn[_lp*2+1 + 0]));
                    vw[((tb * 2 + half) * 2) + _lp] = *(uint32_t*)&_bf2;
                }
            }
        }
        #pragma unroll
        for (int _lp = 0; _lp < 16; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_v[_lp*2 + 0], acc_v[_lp*2+1 + 0]));
            vpk[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int ks2 = 0; ks2 < ((0) ? 0 : 4); ks2++) {
            #pragma unroll
            for (int nb5 = 0; nb5 < 4; nb5++) {
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                    : "r"(s_kn_addr + (unsigned int)stage_off + (unsigned int)((kbase + nb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((ks2 * 16 + lane % 16) * 128 + (kbase + nb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((ks2 * 16 + lane % 16) * 128 + (kbase + nb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                    : "memory");
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_h + nb5 * 8)[0]), "+f"((acc_h + nb5 * 8)[1]), "+f"((acc_h + nb5 * 8)[2]), "+f"((acc_h + nb5 * 8)[3])
                    : "r"((vpk + ks2 * 4)[0]), "r"((vpk + ks2 * 4)[1]), "r"((vpk + ks2 * 4)[2]), "r"((vpk + ks2 * 4)[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_h + nb5 * 8 + 4)[0]), "+f"((acc_h + nb5 * 8 + 4)[1]), "+f"((acc_h + nb5 * 8 + 4)[2]), "+f"((acc_h + nb5 * 8 + 4)[3])
                    : "r"((vpk + ks2 * 4)[0]), "r"((vpk + ks2 * 4)[1]), "r"((vpk + ks2 * 4)[2]), "r"((vpk + ks2 * 4)[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
            }
        }
        if (khalf == 0) {
            long long row_base = ((long long)tok0 * (long long)num_v_heads + (long long)hv) * 128 + (long long)v0 + (long long)(vwarp * 16);
            int m_idx_0 = lane_0 / 8;
            int r_idx_1 = lane_0 % 8;
            int st_half = m_idx_0 % 2;
            int st_tok = m_idx_0 / 2 * 8 + r_idx_1;
            uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(stage_base + st_tok * 32 + ((st_half ^ st_tok >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&vw[0])), "r"(*reinterpret_cast<const uint32_t*>(&vw[1])), "r"(*reinterpret_cast<const uint32_t*>(&vw[2])), "r"(*reinterpret_cast<const uint32_t*>(&vw[3]))
                : "memory");
            int st_tok_2 = (2 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(stage_base + st_tok_2 * 32 + ((st_half ^ st_tok_2 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&vw[4])), "r"(*reinterpret_cast<const uint32_t*>(&vw[5])), "r"(*reinterpret_cast<const uint32_t*>(&vw[6])), "r"(*reinterpret_cast<const uint32_t*>(&vw[7]))
                : "memory");
            int st_tok_3 = (4 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(stage_base + st_tok_3 * 32 + ((st_half ^ st_tok_3 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&vw[8])), "r"(*reinterpret_cast<const uint32_t*>(&vw[9])), "r"(*reinterpret_cast<const uint32_t*>(&vw[10])), "r"(*reinterpret_cast<const uint32_t*>(&vw[11]))
                : "memory");
            int st_tok_4 = (6 + m_idx_0 / 2) * 8 + r_idx_1;
            uint32_t _stmatrix_addr_20 = static_cast<uint32_t>(stage_base + st_tok_4 * 32 + ((st_half ^ st_tok_4 >> 2 & 1) << 4));
            asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                :: "r"(_stmatrix_addr_20), "r"(*reinterpret_cast<const uint32_t*>(&vw[12])), "r"(*reinterpret_cast<const uint32_t*>(&vw[13])), "r"(*reinterpret_cast<const uint32_t*>(&vw[14])), "r"(*reinterpret_cast<const uint32_t*>(&vw[15]))
                : "memory");
            __syncwarp();
            int ld_half = lane_0 % 2;
            int ld_tok = lane_0 / 2;
            int words_6[4];
            {
                const int4* _ivptr_21 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok * 8 + ((ld_half ^ ld_tok >> 2 & 1) << 2));
                int4 _ivld_21;
                _ivld_21 = *_ivptr_21;
                words_6[0 + 0] = _ivld_21.x;
                words_6[0 + 1] = _ivld_21.y;
                words_6[0 + 2] = _ivld_21.z;
                words_6[0 + 3] = _ivld_21.w;
            }
            if (ld_tok < n_valid) {
                {
                    int4 _iv4 = make_int4(words_6[0 + 0], words_6[0 + 1], words_6[0 + 2], words_6[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_7 = 16 + lane_0 / 2;
            int words_8[4];
            {
                const int4* _ivptr_22 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_7 * 8 + ((ld_half ^ ld_tok_7 >> 2 & 1) << 2));
                int4 _ivld_22;
                _ivld_22 = *_ivptr_22;
                words_8[0 + 0] = _ivld_22.x;
                words_8[0 + 1] = _ivld_22.y;
                words_8[0 + 2] = _ivld_22.z;
                words_8[0 + 3] = _ivld_22.w;
            }
            if (ld_tok_7 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_8[0 + 0], words_8[0 + 1], words_8[0 + 2], words_8[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_7 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_9 = 32 + lane_0 / 2;
            int words_10[4];
            {
                const int4* _ivptr_23 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_9 * 8 + ((ld_half ^ ld_tok_9 >> 2 & 1) << 2));
                int4 _ivld_23;
                _ivld_23 = *_ivptr_23;
                words_10[0 + 0] = _ivld_23.x;
                words_10[0 + 1] = _ivld_23.y;
                words_10[0 + 2] = _ivld_23.z;
                words_10[0 + 3] = _ivld_23.w;
            }
            if (ld_tok_9 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_10[0 + 0], words_10[0 + 1], words_10[0 + 2], words_10[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_9 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            int ld_tok_11 = 48 + lane_0 / 2;
            int words_12[4];
            {
                const int4* _ivptr_24 = reinterpret_cast<const int4*>(reinterpret_cast<int*>(s_stage) + stage_u32_base + ld_tok_11 * 8 + ((ld_half ^ ld_tok_11 >> 2 & 1) << 2));
                int4 _ivld_24;
                _ivld_24 = *_ivptr_24;
                words_12[0 + 0] = _ivld_24.x;
                words_12[0 + 1] = _ivld_24.y;
                words_12[0 + 2] = _ivld_24.z;
                words_12[0 + 3] = _ivld_24.w;
            }
            if (ld_tok_11 < n_valid) {
                {
                    int4 _iv4 = make_int4(words_12[0 + 0], words_12[0 + 1], words_12[0 + 2], words_12[0 + 3]);
                    *reinterpret_cast<int4*>(reinterpret_cast<int*>(v_new) + (row_base + (long long)ld_tok_11 * tok_stride + (long long)(ld_half * 8)) / 2) = _iv4;
                }
            }
            __syncwarp();
        }
        if (num_chunks > it_chunk + 1) {
            #pragma unroll
            for (int gi2 = 0; gi2 < 1; gi2++) {
                int gtok2 = gi2 * 128 + tid_2;
                if (gtok2 < 64) {
                    s_g[next_g_off + gtok2] = g_next[gi2];
                }
            }
        }
        #pragma unroll
        for (int gi3 = 0; gi3 < 1; gi3++) {
            g_next[gi3] = g_far[gi3];
        }
        __syncthreads();
        stage_off = next_off;
        g_off = next_g_off;
    }
    if (store_final_state != 0) {
        #pragma unroll
        for (int nb6 = 0; nb6 < 8; nb6++) {
            int kcol2 = kbase + nb6 * 8 + quad * 2;
            final_state[state_base + (long long)kcol2 * 128 + (long long)vrow_a] = acc_h[nb6 * 4];
            final_state[state_base + (long long)(kcol2 + 1) * 128 + (long long)vrow_a] = acc_h[nb6 * 4 + 1];
            final_state[state_base + (long long)kcol2 * 128 + (long long)vrow_b] = acc_h[nb6 * 4 + 2];
            final_state[state_base + (long long)(kcol2 + 1) * 128 + (long long)vrow_b] = acc_h[nb6 * 4 + 3];
        }
    }
}

} // extern "C"
