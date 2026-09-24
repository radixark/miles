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
#define SMEM_S_DO_OFF 0
#define SMEM_S_DO_STAGE_BYTES 16384
#define SMEM_S_DO_STRIDE 16384
#define SMEM_S_VN_OFF 16384
#define SMEM_S_VN_STAGE_BYTES 16384
#define SMEM_S_VN_STRIDE 16384
#define SMEM_S_QN_OFF 32768
#define SMEM_S_QN_STAGE_BYTES 16384
#define SMEM_S_QN_STRIDE 16384
#define SMEM_S_KN_OFF 49152
#define SMEM_S_KN_STAGE_BYTES 16384
#define SMEM_S_KN_STRIDE 16384
#define SMEM_S_DV_OFF 65536
#define SMEM_S_DV_STAGE_BYTES 16384
#define SMEM_S_DV_STRIDE 16384
#define SMEM_S_H_OFF 81920
#define SMEM_S_H_STAGE_BYTES 32768
#define SMEM_S_H_STRIDE 32768
#define SMEM_S_DH_OFF 114688
#define SMEM_S_DH_STAGE_BYTES 32768
#define SMEM_S_DH_STRIDE 32768
#define SMEM_S_DS_OFF 147456
#define SMEM_S_DS_STAGE_BYTES 8192
#define SMEM_S_DS_STRIDE 8192
#define SMEM_S_G_OFF 155648
#define SMEM_S_G_STAGE_BYTES 256
#define SMEM_S_G_STRIDE 256
#define SMEM_S_DGQ_OFF 155904
#define SMEM_S_DGQ_STAGE_BYTES 256
#define SMEM_S_DGQ_STRIDE 256
#define SMEM_S_DGK_OFF 156160
#define SMEM_S_DGK_STAGE_BYTES 256
#define SMEM_S_DGK_STRIDE 256
#define SMEM_S_RED_OFF 156416
#define SMEM_S_RED_STAGE_BYTES 512
#define SMEM_S_RED_STRIDE 512
#define SMEM_TOTAL 156928
#define THREADS 128

#include <math_constants.h>

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
kernel_cake_gdn_chunk_train_6d7c261b3ab8339ea352(__nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, __nv_bfloat16* __restrict__ v_new, __nv_bfloat16* __restrict__ do_, __nv_bfloat16* __restrict__ dv2, __nv_bfloat16* __restrict__ h, __nv_bfloat16* __restrict__ dh, float* __restrict__ g_cs, __nv_bfloat16* __restrict__ dq_out, __nv_bfloat16* __restrict__ dk_out, __nv_bfloat16* __restrict__ dw_out, float* __restrict__ dg_out, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, float scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __nv_bfloat16* s_do = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_do_addr = smem + 0;
    __nv_bfloat16* s_vn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_vn_addr = smem + 16384;
    __nv_bfloat16* s_qn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32768);
    const int s_qn_addr = smem + 32768;
    __nv_bfloat16* s_kn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 49152);
    const int s_kn_addr = smem + 49152;
    __nv_bfloat16* s_dv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 65536);
    const int s_dv_addr = smem + 65536;
    __nv_bfloat16* s_h = reinterpret_cast<__nv_bfloat16*>(smem_raw + 81920);
    const int s_h_addr = smem + 81920;
    __nv_bfloat16* s_dh = reinterpret_cast<__nv_bfloat16*>(smem_raw + 114688);
    const int s_dh_addr = smem + 114688;
    __nv_bfloat16* s_ds = reinterpret_cast<__nv_bfloat16*>(smem_raw + 147456);
    const int s_ds_addr = smem + 147456;
    float* s_g = reinterpret_cast<float*>(smem_raw + 155648);
    const int s_g_addr = smem + 155648;
    float* s_dgq = reinterpret_cast<float*>(smem_raw + 155904);
    const int s_dgq_addr = smem + 155904;
    float* s_dgk = reinterpret_cast<float*>(smem_raw + 156160);
    const int s_dgk_addr = smem + 156160;
    float* s_red = reinterpret_cast<float*>(smem_raw + 156416);
    const int s_red_addr = smem + 156416;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int hv = blockIdx.y;
    int tok0 = chunk_start[chunk];
    int n_valid = chunk_len[chunk];
    int group = num_v_heads / num_heads;
    int hq = hv / group;
    int lane_0 = lane;
    int warp_1 = warp;
    int tid_2 = tid;
    int row_a = warp_1 * 16 + lane_0 / 4;
    int row_b = row_a + 8;
    int col_q = lane_0 % 4 * 2;
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)(tok0 + tid / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)(tid % 16 * 8))), "r"((n_valid > tid / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)(tok0 + (128 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((n_valid > (128 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)(tok0 + (256 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((n_valid > (256 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)(tok0 + (384 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((n_valid > (384 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)(tok0 + (512 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((n_valid > (512 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)(tok0 + (640 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((n_valid > (640 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)(tok0 + (768 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((n_valid > (768 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_do_addr + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(do_ + (((long long)(tok0 + (896 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((n_valid > (896 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_vn_addr + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v_new + (((long long)(tok0 + tid / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)(tid % 16 * 8))), "r"((n_valid > tid / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_vn_addr + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v_new + (((long long)(tok0 + (128 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((n_valid > (128 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_vn_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v_new + (((long long)(tok0 + (256 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((n_valid > (256 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_vn_addr + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v_new + (((long long)(tok0 + (384 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((n_valid > (384 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_vn_addr + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v_new + (((long long)(tok0 + (512 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((n_valid > (512 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_vn_addr + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v_new + (((long long)(tok0 + (640 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((n_valid > (640 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_vn_addr + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v_new + (((long long)(tok0 + (768 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((n_valid > (768 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_vn_addr + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(v_new + (((long long)(tok0 + (896 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((n_valid > (896 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_qn_addr + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(qn + (((long long)(tok0 + tid / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)(tid % 16 * 8))), "r"((n_valid > tid / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_qn_addr + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(qn + (((long long)(tok0 + (128 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((n_valid > (128 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_qn_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(qn + (((long long)(tok0 + (256 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((n_valid > (256 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_qn_addr + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(qn + (((long long)(tok0 + (384 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((n_valid > (384 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_qn_addr + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(qn + (((long long)(tok0 + (512 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((n_valid > (512 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_qn_addr + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(qn + (((long long)(tok0 + (640 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((n_valid > (640 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_qn_addr + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(qn + (((long long)(tok0 + (768 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((n_valid > (768 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_qn_addr + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(qn + (((long long)(tok0 + (896 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((n_valid > (896 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + tid / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)(tid % 16 * 8))), "r"((n_valid > tid / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (128 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((n_valid > (128 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (256 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((n_valid > (256 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (384 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((n_valid > (384 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (512 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((n_valid > (512 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (640 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((n_valid > (640 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (768 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((n_valid > (768 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_kn_addr + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(kn + (((long long)(tok0 + (896 + tid) / 16) * (long long)num_heads + (long long)hq) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((n_valid > (896 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dv_addr + (unsigned int)(tid % 16 * 8 / 64 * 8192 + (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 ^ (tid / 16 * 128 + tid % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dv2 + (((long long)(tok0 + tid / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)(tid % 16 * 8))), "r"((n_valid > tid / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dv_addr + (unsigned int)((128 + tid) % 16 * 8 / 64 * 8192 + ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 ^ ((128 + tid) / 16 * 128 + (128 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dv2 + (((long long)(tok0 + (128 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((128 + tid) % 16 * 8))), "r"((n_valid > (128 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dv_addr + (unsigned int)((256 + tid) % 16 * 8 / 64 * 8192 + ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 ^ ((256 + tid) / 16 * 128 + (256 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dv2 + (((long long)(tok0 + (256 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((256 + tid) % 16 * 8))), "r"((n_valid > (256 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dv_addr + (unsigned int)((384 + tid) % 16 * 8 / 64 * 8192 + ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 ^ ((384 + tid) / 16 * 128 + (384 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dv2 + (((long long)(tok0 + (384 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((384 + tid) % 16 * 8))), "r"((n_valid > (384 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dv_addr + (unsigned int)((512 + tid) % 16 * 8 / 64 * 8192 + ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 ^ ((512 + tid) / 16 * 128 + (512 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dv2 + (((long long)(tok0 + (512 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((512 + tid) % 16 * 8))), "r"((n_valid > (512 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dv_addr + (unsigned int)((640 + tid) % 16 * 8 / 64 * 8192 + ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 ^ ((640 + tid) / 16 * 128 + (640 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dv2 + (((long long)(tok0 + (640 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((640 + tid) % 16 * 8))), "r"((n_valid > (640 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dv_addr + (unsigned int)((768 + tid) % 16 * 8 / 64 * 8192 + ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 ^ ((768 + tid) / 16 * 128 + (768 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dv2 + (((long long)(tok0 + (768 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((768 + tid) % 16 * 8))), "r"((n_valid > (768 + tid) / 16) ? 16 : 0));
    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
        :: "r"(s_dv_addr + (unsigned int)((896 + tid) % 16 * 8 / 64 * 8192 + ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 ^ ((896 + tid) / 16 * 128 + (896 + tid) % 16 * 8 % 64 * 2 >> 7 & 7) << 4))), "l"(dv2 + (((long long)(tok0 + (896 + tid) / 16) * (long long)num_v_heads + (long long)hv) * 128 + (long long)((896 + tid) % 16 * 8))), "r"((n_valid > (896 + tid) / 16) ? 16 : 0));
    long long state_base = ((long long)chunk * (long long)num_v_heads + (long long)hv) * 16384;
    #pragma unroll
    for (int it = 0; it < 16; it++) {
        int idx = it * 128 + tid_2;
        int h_row = idx / 16;
        int h_col = idx % 16 * 8;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(s_h_addr + (unsigned int)(h_col / 64 * 16384 + (h_row * 128 + h_col % 64 * 2 ^ (h_row * 128 + h_col % 64 * 2 >> 7 & 7) << 4))), "l"(h + (state_base + (long long)h_row * 128 + (long long)h_col)));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(s_dh_addr + (unsigned int)(h_col / 64 * 16384 + (h_row * 128 + h_col % 64 * 2 ^ (h_row * 128 + h_col % 64 * 2 >> 7 & 7) << 4))), "l"(dh + (state_base + (long long)h_row * 128 + (long long)h_col)));
    }
    asm volatile("cp.async.commit_group;");
    if (tid_2 < 64) {
        float gate_val = 0.0f;
        if (tid_2 < n_valid) {
            gate_val = g_cs[(long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv];
        }
        s_g[tid_2] = gate_val;
    }
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    unsigned int a_frag_k[4];
    unsigned int a_frag_mn[4];
    unsigned int b_frag_k[4];
    unsigned int b_frag_mn[4];
    float acc_s[32];
    float acc[64];
    float g_a = s_g[row_a];
    float g_b = s_g[row_b];
    int _max_0 = ((n_valid) > (1) ? (n_valid) : (1));
    float g_last = s_g[_max_0 - 1];
    long long out_a = ((long long)(tok0 + row_a) * (long long)num_v_heads + (long long)hv) * 128 + (long long)col_q;
    long long out_b = ((long long)(tok0 + row_b) * (long long)num_v_heads + (long long)hv) * 128 + (long long)col_q;
    #pragma unroll
    for (int kb = 0; kb < 8; kb++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_do_addr + (unsigned int)((kb * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb = 0; nb < 4; nb++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_k[0]), "=r"(b_frag_k[1]), "=r"(b_frag_k[2]), "=r"(b_frag_k[3])
                : "r"(s_vn_addr + (unsigned int)((kb * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((nb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((nb * 16 + 8 * (lane / 16) + lane % 8) * 128 + (kb * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb * 8)[0]), "=f"((acc_s + nb * 8)[1]), "=f"((acc_s + nb * 8)[2]), "=f"((acc_s + nb * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[0]), "r"(b_frag_k[1]), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8)[0])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8)[1])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8)[2])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc_s + nb * 8 + 4)[0]), "=f"((acc_s + nb * 8 + 4)[1]), "=f"((acc_s + nb * 8 + 4)[2]), "=f"((acc_s + nb * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_k[2]), "r"(b_frag_k[(2) + 1]), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8 + 4)[0])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8 + 4)[1])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8 + 4)[2])), "f"(((kb == 0) ? 0.0f : (acc_s + nb * 8 + 4)[3])));
        }
    }
    #pragma unroll
    for (int nb2 = 0; nb2 < 8; nb2++) {
        #pragma unroll
        for (int e = 0; e < 2; e++) {
            int col = nb2 * 8 + col_q + e;
            float g_col = s_g[col];
            float val_a = 0.0f;
            float val_b = 0.0f;
            if (col <= row_a) {
                if (row_a < n_valid) {
                    float _exp2_0 = approx_exp2(g_a - g_col);
                    val_a = acc_s[nb2 * 4 + e] * _exp2_0 * scale;
                }
            }
            if (col <= row_b) {
                if (row_b < n_valid) {
                    float _exp2_1 = approx_exp2(g_b - g_col);
                    val_b = acc_s[nb2 * 4 + 2 + e] * _exp2_1 * scale;
                }
            }
            {
                __nv_bfloat16 _bval_0 = __float2bfloat16_rn(val_a);
                uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                uint32_t _addr_0 = static_cast<uint32_t>((s_ds_addr + (unsigned int)(row_a * 128 + col * 2 ^ (row_a * 128 + col * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
            }
            {
                __nv_bfloat16 _bval_1 = __float2bfloat16_rn(val_b);
                uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                uint32_t _addr_1 = static_cast<uint32_t>((s_ds_addr + (unsigned int)(row_b * 128 + col * 2 ^ (row_b * 128 + col * 2 >> 7 & 7) << 4)));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
            }
        }
    }
    #pragma unroll
    for (int kb2 = 0; kb2 < 8; kb2++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_do_addr + (unsigned int)((kb2 * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb2 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb3 = 0; nb3 < 8; nb3++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_h_addr + (unsigned int)((nb3 * 16 + lane / 16 * 8) / 64 * 16384 + ((kb2 * 16 + lane % 16) * 128 + (nb3 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb2 * 16 + lane % 16) * 128 + (nb3 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb3 * 8)[0]), "=f"((acc + nb3 * 8)[1]), "=f"((acc + nb3 * 8)[2]), "=f"((acc + nb3 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[0])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[1])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[2])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb3 * 8 + 4)[0]), "=f"((acc + nb3 * 8 + 4)[1]), "=f"((acc + nb3 * 8 + 4)[2]), "=f"((acc + nb3 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[0])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[1])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[2])), "f"(((kb2 == 0) ? 0.0f : (acc + nb3 * 8 + 4)[3])));
        }
    }
    float _exp2_2 = approx_exp2(g_a);
    float gain_a = _exp2_2 * scale;
    float _exp2_3 = approx_exp2(g_b);
    float gain_b = _exp2_3 * scale;
    #pragma unroll
    for (int nb4 = 0; nb4 < 16; nb4++) {
        acc[nb4 * 4] = acc[nb4 * 4] * gain_a;
        acc[nb4 * 4 + 1] = acc[nb4 * 4 + 1] * gain_a;
        acc[nb4 * 4 + 2] = acc[nb4 * 4 + 2] * gain_b;
        acc[nb4 * 4 + 3] = acc[nb4 * 4 + 3] * gain_b;
    }
    __syncthreads();
    #pragma unroll
    for (int kb3 = 0; kb3 < 4; kb3++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_ds_addr + (unsigned int)((kb3 * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb3 * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb3 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb5 = 0; nb5 < 8; nb5++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_kn_addr + (unsigned int)((nb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb3 * 16 + lane % 16) * 128 + (nb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb3 * 16 + lane % 16) * 128 + (nb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc + nb5 * 8)[0]), "+f"((acc + nb5 * 8)[1]), "+f"((acc + nb5 * 8)[2]), "+f"((acc + nb5 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc + nb5 * 8 + 4)[0]), "+f"((acc + nb5 * 8 + 4)[1]), "+f"((acc + nb5 * 8 + 4)[2]), "+f"((acc + nb5 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
        }
    }
    float dgq_a = 0.0f;
    float dgq_b = 0.0f;
    #pragma unroll
    for (int nb6 = 0; nb6 < 16; nb6++) {
        #pragma unroll
        for (int e2 = 0; e2 < 2; e2++) {
            int qcol = nb6 * 8 + col_q + e2;
            float _fma_0 = __fmaf_rn(acc[nb6 * 4 + e2], (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_qn) + (qcol / 64 * 8192 + (row_a * 128 + qcol % 64 * 2 ^ (row_a * 128 + qcol % 64 * 2 >> 7 & 7) << 4)))[0], dgq_a);
            dgq_a = _fma_0;
            float _fma_1 = __fmaf_rn(acc[nb6 * 4 + 2 + e2], (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_qn) + (qcol / 64 * 8192 + (row_b * 128 + qcol % 64 * 2 ^ (row_b * 128 + qcol % 64 * 2 >> 7 & 7) << 4)))[0], dgq_b);
            dgq_b = _fma_1;
        }
        if (row_a < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb6 * 4 + 0], acc[nb6 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dq_out))[out_a + (long long)(nb6 * 8)]) = _pk;
            }
        }
        if (row_b < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb6 * 4 + 2 + 0], acc[nb6 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dq_out))[out_b + (long long)(nb6 * 8)]) = _pk;
            }
        }
    }
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, dgq_a, 1);
    dgq_a = dgq_a + _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, dgq_a, 2);
    dgq_a = dgq_a + _shfl_xor_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, dgq_b, 1);
    dgq_b = dgq_b + _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, dgq_b, 2);
    dgq_b = dgq_b + _shfl_xor_3;
    if (lane_0 % 4 == 0) {
        s_dgq[row_a] = dgq_a;
        s_dgq[row_b] = dgq_b;
    }
    #pragma unroll
    for (int kb4 = 0; kb4 < 8; kb4++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_dv_addr + (unsigned int)((kb4 * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb4 * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb4 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb7 = 0; nb7 < 8; nb7++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_h_addr + (unsigned int)((nb7 * 16 + lane / 16 * 8) / 64 * 16384 + ((kb4 * 16 + lane % 16) * 128 + (nb7 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb4 * 16 + lane % 16) * 128 + (nb7 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb7 * 8)[0]), "=f"((acc + nb7 * 8)[1]), "=f"((acc + nb7 * 8)[2]), "=f"((acc + nb7 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8)[0])), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8)[1])), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8)[2])), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb7 * 8 + 4)[0]), "=f"((acc + nb7 * 8 + 4)[1]), "=f"((acc + nb7 * 8 + 4)[2]), "=f"((acc + nb7 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8 + 4)[0])), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8 + 4)[1])), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8 + 4)[2])), "f"(((kb4 == 0) ? 0.0f : (acc + nb7 * 8 + 4)[3])));
        }
    }
    const float2 _scale2_2 = {-1.0f, -1.0f};
    #pragma unroll
    for (int _ls = 0; _ls < 32; _ls++)
        mul_f32x2_inplace(&reinterpret_cast<float2*>(acc)[_ls], _scale2_2);
    #pragma unroll
    for (int nb8 = 0; nb8 < 16; nb8++) {
        if (row_a < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb8 * 4 + 0], acc[nb8 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dw_out))[out_a + (long long)(nb8 * 8)]) = _pk;
            }
        }
        if (row_b < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb8 * 4 + 2 + 0], acc[nb8 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dw_out))[out_b + (long long)(nb8 * 8)]) = _pk;
            }
        }
    }
    #pragma unroll
    for (int kb5 = 0; kb5 < 8; kb5++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_k[0]), "=r"(a_frag_k[1]), "=r"(a_frag_k[2]), "=r"(a_frag_k[3])
            : "r"(s_vn_addr + (unsigned int)((kb5 * 16 + lane / 16 * 8) / 64 * 8192 + ((warp_1 * 16 + lane % 16) * 128 + (kb5 * 16 + lane / 16 * 8) % 64 * 2 ^ ((warp_1 * 16 + lane % 16) * 128 + (kb5 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb9 = 0; nb9 < 8; nb9++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_dh_addr + (unsigned int)((nb9 * 16 + lane / 16 * 8) / 64 * 16384 + ((kb5 * 16 + lane % 16) * 128 + (nb9 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb5 * 16 + lane % 16) * 128 + (nb9 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb9 * 8)[0]), "=f"((acc + nb9 * 8)[1]), "=f"((acc + nb9 * 8)[2]), "=f"((acc + nb9 * 8)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8)[0])), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8)[1])), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8)[2])), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8)[3])));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                : "=f"((acc + nb9 * 8 + 4)[0]), "=f"((acc + nb9 * 8 + 4)[1]), "=f"((acc + nb9 * 8 + 4)[2]), "=f"((acc + nb9 * 8 + 4)[3])
                : "r"(a_frag_k[0]), "r"(a_frag_k[1]), "r"(a_frag_k[2]), "r"(a_frag_k[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8 + 4)[0])), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8 + 4)[1])), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8 + 4)[2])), "f"(((kb5 == 0) ? 0.0f : (acc + nb9 * 8 + 4)[3])));
        }
    }
    float decay_a = 0.0f;
    float decay_b = 0.0f;
    if (row_a < n_valid) {
        float _exp2_4 = approx_exp2(g_last - g_a);
        decay_a = _exp2_4;
    }
    if (row_b < n_valid) {
        float _exp2_5 = approx_exp2(g_last - g_b);
        decay_b = _exp2_5;
    }
    float dg_last_part = 0.0f;
    #pragma unroll
    for (int nb10 = 0; nb10 < 16; nb10++) {
        acc[nb10 * 4] = acc[nb10 * 4] * decay_a;
        acc[nb10 * 4 + 1] = acc[nb10 * 4 + 1] * decay_a;
        acc[nb10 * 4 + 2] = acc[nb10 * 4 + 2] * decay_b;
        acc[nb10 * 4 + 3] = acc[nb10 * 4 + 3] * decay_b;
        #pragma unroll
        for (int e3 = 0; e3 < 2; e3++) {
            int kcol = nb10 * 8 + col_q + e3;
            float _fma_2 = __fmaf_rn(acc[nb10 * 4 + e3], (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kn) + (kcol / 64 * 8192 + (row_a * 128 + kcol % 64 * 2 ^ (row_a * 128 + kcol % 64 * 2 >> 7 & 7) << 4)))[0], dg_last_part);
            dg_last_part = _fma_2;
            float _fma_3 = __fmaf_rn(acc[nb10 * 4 + 2 + e3], (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kn) + (kcol / 64 * 8192 + (row_b * 128 + kcol % 64 * 2 ^ (row_b * 128 + kcol % 64 * 2 >> 7 & 7) << 4)))[0], dg_last_part);
            dg_last_part = _fma_3;
        }
    }
    float hdh = 0.0f;
    #pragma unroll
    for (int it2 = 0; it2 < 128; it2++) {
        int eidx = it2 * 128 + tid_2;
        int e_row = eidx / 128;
        int e_col = eidx % 128;
        int e_off = e_col / 64 * 16384 + (e_row * 128 + e_col % 64 * 2 ^ (e_row * 128 + e_col % 64 * 2 >> 7 & 7) << 4);
        float _fma_4 = __fmaf_rn((float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_h) + e_off)[0], (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_dh) + e_off)[0], hdh);
        hdh = _fma_4;
    }
    float _exp2_6 = approx_exp2(g_last);
    s_red[tid_2] = hdh * _exp2_6 + dg_last_part;
    #pragma unroll
    for (int kb6 = 0; kb6 < 4; kb6++) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(a_frag_mn[0]), "=r"(a_frag_mn[1]), "=r"(a_frag_mn[2]), "=r"(a_frag_mn[3])
            : "r"(s_ds_addr + (unsigned int)((warp_1 * 16 + lane % 16 / 8 * 8) / 64 * 8192 + ((kb6 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (warp_1 * 16 + lane % 16 / 8 * 8) % 64 * 2 ^ ((kb6 * 16 + 8 * (lane / 16) + lane % 8) * 128 + (warp_1 * 16 + lane % 16 / 8 * 8) % 64 * 2 >> 7 & 7) << 4)))
            : "memory");
        #pragma unroll
        for (int nb11 = 0; nb11 < 8; nb11++) {
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b_frag_mn[0]), "=r"(b_frag_mn[1]), "=r"(b_frag_mn[2]), "=r"(b_frag_mn[3])
                : "r"(s_qn_addr + (unsigned int)((nb11 * 16 + lane / 16 * 8) / 64 * 8192 + ((kb6 * 16 + lane % 16) * 128 + (nb11 * 16 + lane / 16 * 8) % 64 * 2 ^ ((kb6 * 16 + lane % 16) * 128 + (nb11 * 16 + lane / 16 * 8) % 64 * 2 >> 7 & 7) << 4)))
                : "memory");
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc + nb11 * 8)[0]), "+f"((acc + nb11 * 8)[1]), "+f"((acc + nb11 * 8)[2]), "+f"((acc + nb11 * 8)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[0]), "r"(b_frag_mn[1]));
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc + nb11 * 8 + 4)[0]), "+f"((acc + nb11 * 8 + 4)[1]), "+f"((acc + nb11 * 8 + 4)[2]), "+f"((acc + nb11 * 8 + 4)[3])
                : "r"(a_frag_mn[0]), "r"(a_frag_mn[1]), "r"(a_frag_mn[2]), "r"(a_frag_mn[3]), "r"(b_frag_mn[2]), "r"(b_frag_mn[(2) + 1]));
        }
    }
    float dgk_a = 0.0f;
    float dgk_b = 0.0f;
    #pragma unroll
    for (int nb12 = 0; nb12 < 16; nb12++) {
        #pragma unroll
        for (int e4 = 0; e4 < 2; e4++) {
            int kcol2 = nb12 * 8 + col_q + e4;
            float _fma_5 = __fmaf_rn(acc[nb12 * 4 + e4], (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kn) + (kcol2 / 64 * 8192 + (row_a * 128 + kcol2 % 64 * 2 ^ (row_a * 128 + kcol2 % 64 * 2 >> 7 & 7) << 4)))[0], dgk_a);
            dgk_a = _fma_5;
            float _fma_6 = __fmaf_rn(acc[nb12 * 4 + 2 + e4], (float)reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const uint8_t*>(s_kn) + (kcol2 / 64 * 8192 + (row_b * 128 + kcol2 % 64 * 2 ^ (row_b * 128 + kcol2 % 64 * 2 >> 7 & 7) << 4)))[0], dgk_b);
            dgk_b = _fma_6;
        }
        if (row_a < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb12 * 4 + 0], acc[nb12 * 4 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dk_out))[out_a + (long long)(nb12 * 8)]) = _pk;
            }
        }
        if (row_b < n_valid) {
            {
                __nv_bfloat162 _pk = __floats2bfloat162_rn(acc[nb12 * 4 + 2 + 0], acc[nb12 * 4 + 2 + 1]);
                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(dk_out))[out_b + (long long)(nb12 * 8)]) = _pk;
            }
        }
    }
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, dgk_a, 1);
    dgk_a = dgk_a + _shfl_xor_4;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, dgk_a, 2);
    dgk_a = dgk_a + _shfl_xor_5;
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, dgk_b, 1);
    dgk_b = dgk_b + _shfl_xor_6;
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, dgk_b, 2);
    dgk_b = dgk_b + _shfl_xor_7;
    if (lane_0 % 4 == 0) {
        s_dgk[row_a] = dgk_a;
        s_dgk[row_b] = dgk_b;
    }
    __syncthreads();
    if (tid_2 < n_valid) {
        float dg_last = 0.0f;
        #pragma unroll
        for (int r = 0; r < 128; r++) {
            dg_last = dg_last + s_red[r];
        }
        float dg_val = s_dgq[tid_2] - s_dgk[tid_2];
        if (tid_2 == n_valid - 1) {
            dg_val = dg_val + dg_last;
        }
        dg_out[(long long)(tok0 + tid_2) * (long long)num_v_heads + (long long)hv] = dg_val;
    }
}

} // extern "C"
