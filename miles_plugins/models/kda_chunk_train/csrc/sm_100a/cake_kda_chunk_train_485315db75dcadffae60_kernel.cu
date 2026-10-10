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
#define SMEM_X_S_OFF 0
#define SMEM_X_S_STAGE_BYTES 34816
#define SMEM_X_S_STRIDE 34816
#define SMEM_G_S_OFF 34816
#define SMEM_G_S_STAGE_BYTES 34816
#define SMEM_G_S_STRIDE 34816
#define SMEM_BETA_S_OFF 69632
#define SMEM_BETA_S_STAGE_BYTES 256
#define SMEM_BETA_S_STRIDE 256
#define SMEM_DIAG_S_OFF 69888
#define SMEM_DIAG_S_STAGE_BYTES 5120
#define SMEM_DIAG_S_STRIDE 5120
#define SMEM_AKKW_S_OFF 0
#define SMEM_AKKW_S_STAGE_BYTES 18432
#define SMEM_AKKW_S_STRIDE 18432
#define SMEM_AKKO_S_OFF 34816
#define SMEM_AKKO_S_STAGE_BYTES 18432
#define SMEM_AKKO_S_STRIDE 18432
#define SMEM_SCR_S_OFF 53248
#define SMEM_SCR_S_STAGE_BYTES 4608
#define SMEM_SCR_S_STRIDE 4608
#define SMEM_TOTAL 75008
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 3

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_chunk_train_485315db75dcadffae60(__nv_bfloat16* __restrict__ q_norm, __nv_bfloat16* __restrict__ k_norm, float* __restrict__ gk, float* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ aqk_ext, __nv_bfloat16* __restrict__ akk_ext, __nv_bfloat16* __restrict__ akk_int, int num_heads, int num_qk_heads, int group, int write_int, float scale)
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
    __nv_bfloat16* x_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_X_S_OFF);
    const int x_s_addr = smem + SMEM_X_S_OFF;
    float* g_s = reinterpret_cast<float*>(smem_raw + SMEM_G_S_OFF);
    const int g_s_addr = smem + SMEM_G_S_OFF;
    float* beta_s = reinterpret_cast<float*>(smem_raw + SMEM_BETA_S_OFF);
    const int beta_s_addr = smem + SMEM_BETA_S_OFF;
    float* diag_s = reinterpret_cast<float*>(smem_raw + SMEM_DIAG_S_OFF);
    const int diag_s_addr = smem + SMEM_DIAG_S_OFF;
    float* akkw_s = reinterpret_cast<float*>(smem_raw + SMEM_AKKW_S_OFF);
    const int akkw_s_addr = smem + SMEM_AKKW_S_OFF;
    float* akko_s = reinterpret_cast<float*>(smem_raw + SMEM_AKKO_S_OFF);
    const int akko_s_addr = smem + SMEM_AKKO_S_OFF;
    float* scr_s = reinterpret_cast<float*>(smem_raw + SMEM_SCR_S_OFF);
    const int scr_s_addr = smem + SMEM_SCR_S_OFF;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int head = blockIdx.y;
    int qk_head = head / group;
    int row0 = chunk * 64;
    int tid_0 = tid;
    long long ext0 = (long long)chunk_bos[chunk];
    int clen = chunk_len[chunk];
    int clast = clen - 1;
    if (clast < 0) {
        clast = 0;
    }
    long long abase = (ext0 * (long long)num_heads + (long long)head) * 64;
    int ars = num_heads * 64;
    if (tid_0 < 64) {
        float b_v = 0.0f;
        if (tid_0 < clen) {
            b_v = beta[(ext0 + (long long)tid_0) * (long long)num_heads + (long long)head];
        }
        beta_s[tid_0] = b_v;
    }
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int e1 = (r * 256 + tid_0) * 8;
        int t_b = e1 / 128;
        int d_b = e1 - t_b * 128;
        int t_c = t_b;
        if (t_b > clast) {
            t_c = clast;
        }
        long long qk_idx = ((ext0 + (long long)t_c) * (long long)num_qk_heads + (long long)qk_head) * 128 + (long long)d_b;
        float qv[8];
        float kv[8];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(q_norm + qk_idx);
            uint4 _vld_0[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_0[_blk] = _vptr_0[_blk];
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&qv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&qv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_1 = reinterpret_cast<const uint4*>(k_norm + qk_idx);
            uint4 _vld_1[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_1[_blk] = _vptr_1[_blk];
                uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&kv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                    (&kv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int e = 0; e < 8; e++) {
            float q_e = 0.0f;
            float k_e = 0.0f;
            if (t_b < clen) {
                q_e = qv[e];
                k_e = kv[e];
            }
            qv[e] = q_e;
            kv[e] = k_e;
        }
        uint32_t qv_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qv[_lp*2 + 0], qv[_lp*2+1 + 0]));
            qv_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t kv_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kv[_lp*2 + 0], kv[_lp*2+1 + 0]));
            kv_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_s_addr + (unsigned int)((t_b * 136 + d_b) * 2)), "r"(qv_bf16[0]), "r"(qv_bf16[1]), "r"(qv_bf16[2]), "r"(qv_bf16[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(x_s_addr + (unsigned int)(((64 + t_b) * 136 + d_b) * 2)), "r"(kv_bf16[0]), "r"(kv_bf16[1]), "r"(kv_bf16[2]), "r"(kv_bf16[3]) : "memory");
    }
    #pragma unroll
    for (int r_1 = 0; r_1 < 8; r_1++) {
        int e4 = (r_1 * 256 + tid_0) * 4;
        int t_g = e4 / 128;
        int d_g = e4 - t_g * 128;
        long long g_idx = (((long long)row0 + (long long)t_g) * (long long)num_heads + (long long)head) * 128 + (long long)d_g;
        float gv[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(gk + g_idx);
            gv[0 + 0] = _v4.x;
            gv[0 + 1] = _v4.y;
            gv[0 + 2] = _v4.z;
            gv[0 + 3] = _v4.w;
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(g_s_addr + (unsigned int)((t_g * 136 + d_g) * 4)), "f"(gv[0]), "f"(gv[1]), "f"(gv[2]), "f"(gv[3]) : "memory");
    }
    __syncthreads();
    int grp = lane / 4;
    int tq = lane % 4;
    float acc_q[8];
    float acc_k[8];
    int i_0 = warp;
    int j_0 = warp;
    int diag_0 = 1;
    if (warp >= 4) {
        diag_0 = 0;
        i_0 = 1;
        j_0 = 0;
        if (warp == 5) {
            i_0 = 2;
        }
        if (warp == 6) {
            i_0 = 2;
            j_0 = 1;
        }
        if (warp == 7) {
            i_0 = 3;
            j_0 = 2;
        }
    }
    int arow_0 = i_0 * 16;
    if (diag_0 != 0) {
        arow_0 = arow_0 + 8;
    }
    acc_q[0] = 0.0f;
    acc_q[1] = 0.0f;
    acc_q[2] = 0.0f;
    acc_q[3] = 0.0f;
    acc_q[4] = 0.0f;
    acc_q[5] = 0.0f;
    acc_q[6] = 0.0f;
    acc_q[7] = 0.0f;
    acc_k[0] = 0.0f;
    acc_k[1] = 0.0f;
    acc_k[2] = 0.0f;
    acc_k[3] = 0.0f;
    acc_k[4] = 0.0f;
    acc_k[5] = 0.0f;
    acc_k[6] = 0.0f;
    acc_k[7] = 0.0f;
    #pragma unroll
    for (int kk = 0; kk < 16; kk++) {
        int d = kk * 8 + tq * 2;
        float an[2];
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&an[0])), "=r"(*reinterpret_cast<uint32_t*>(&an[(0) + 1]))
            : "r"(g_s_addr + (unsigned int)((arow_0 * 136 + d) * 4)));
        float gA[4];
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&gA[0])), "=r"(*reinterpret_cast<uint32_t*>(&gA[(0) + 1]))
            : "r"(g_s_addr + (unsigned int)(((i_0 * 16 + grp) * 136 + d) * 4)));
        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
            : "=r"(*reinterpret_cast<uint32_t*>(&gA[2])), "=r"(*reinterpret_cast<uint32_t*>(&gA[(2) + 1]))
            : "r"(g_s_addr + (unsigned int)(((i_0 * 16 + grp + 8) * 136 + d) * 4)));
        float eA[4];
        float _exp2_0 = approx_exp2(gA[0] - an[0]);
        eA[0] = _exp2_0;
        float _exp2_1 = approx_exp2(gA[2] - an[0]);
        eA[1] = _exp2_1;
        float _exp2_2 = approx_exp2(gA[1] - an[1]);
        eA[2] = _exp2_2;
        float _exp2_3 = approx_exp2(gA[3] - an[1]);
        eA[3] = _exp2_3;
        unsigned int qw[2];
        unsigned int kw[2];
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&qw[0])) : "r"(x_s_addr + (unsigned int)(((i_0 * 16 + grp) * 136 + d) * 2)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&qw[1])) : "r"(x_s_addr + (unsigned int)(((i_0 * 16 + grp + 8) * 136 + d) * 2)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw[0])) : "r"(x_s_addr + (unsigned int)(((64 + i_0 * 16 + grp) * 136 + d) * 2)));
        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw[1])) : "r"(x_s_addr + (unsigned int)(((64 + i_0 * 16 + grp + 8) * 136 + d) * 2)));
        float qw_f32[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&qw_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) << 16);
            (&qw_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw[_pair]) & 0xffff0000u);
        }
        float kw_f32[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&kw_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw[_pair]) << 16);
            (&kw_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw[_pair]) & 0xffff0000u);
        }
        float aq[4];
        float ak[4];
        aq[0] = qw_f32[0] * eA[0];
        aq[1] = qw_f32[2] * eA[1];
        aq[2] = qw_f32[1] * eA[2];
        aq[3] = qw_f32[3] * eA[3];
        ak[0] = kw_f32[0] * eA[0];
        ak[1] = kw_f32[2] * eA[1];
        ak[2] = kw_f32[1] * eA[2];
        ak[3] = kw_f32[3] * eA[3];
        unsigned int aqr[4];
        unsigned int akr[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            aqr[_lp] = __float_as_uint(aq[_lp + 0]);
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            akr[_lp] = __float_as_uint(ak[_lp + 0]);
        }
        #pragma unroll
        for (int nn = 0; nn < 2; nn++) {
            int s = j_0 * 16 + nn * 8 + grp;
            float gB[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gB[0])), "=r"(*reinterpret_cast<uint32_t*>(&gB[(0) + 1]))
                : "r"(g_s_addr + (unsigned int)((s * 136 + d) * 4)));
            unsigned int bw[1];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bw[0])) : "r"(x_s_addr + (unsigned int)(((64 + s) * 136 + d) * 2)));
            float bw_f32[2];
            #pragma unroll
            for (int _pair = 0; _pair < 1; _pair++) {
                (&bw_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(bw[_pair]) << 16);
                (&bw_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(bw[_pair]) & 0xffff0000u);
            }
            float b[2];
            float _exp2_4 = approx_exp2(an[0] - gB[0]);
            b[0] = bw_f32[0] * _exp2_4;
            float _exp2_5 = approx_exp2(an[1] - gB[1]);
            b[1] = bw_f32[1] * _exp2_5;
            unsigned int br[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                br[_lp] = __float_as_uint(b[_lp + 0]);
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc_q + nn * 4)[0]), "+f"((acc_q + nn * 4)[1]), "+f"((acc_q + nn * 4)[2]), "+f"((acc_q + nn * 4)[3])
                : "r"(aqr[0]), "r"(aqr[1]), "r"(aqr[2]), "r"(aqr[3]), "r"(br[0]), "r"(br[1]));
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((acc_k + nn * 4)[0]), "+f"((acc_k + nn * 4)[1]), "+f"((acc_k + nn * 4)[2]), "+f"((acc_k + nn * 4)[3])
                : "r"(akr[0]), "r"(akr[1]), "r"(akr[2]), "r"(akr[3]), "r"(br[0]), "r"(br[1]));
        }
    }
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        int t_d = i_0 * 16 + grp + half * 8;
        #pragma unroll
        for (int nn_1 = 0; nn_1 < 2; nn_1++) {
            int s_d = j_0 * 16 + nn_1 * 8 + tq * 2;
            float v0 = acc_q[nn_1 * 4 + half * 2] * scale;
            float v1 = acc_q[nn_1 * 4 + half * 2 + 1] * scale;
            if (diag_0 != 0) {
                if (s_d > t_d) {
                    v0 = 0.0f;
                }
                if (t_d < s_d + 1) {
                    v1 = 0.0f;
                }
            }
            float pr[2];
            pr[0] = v0;
            pr[1] = v1;
            if (t_d < clen) {
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(pr[0 + 0], pr[0 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(aqk_ext + (abase + (long long)(t_d * ars) + (long long)s_d)))[0]) = _pk;
                }
            }
        }
    }
    if (warp < 4) {
        float zero8[8];
        zero8[0] = 0.0f;
        zero8[1] = 0.0f;
        zero8[2] = 0.0f;
        zero8[3] = 0.0f;
        zero8[4] = 0.0f;
        zero8[5] = 0.0f;
        zero8[6] = 0.0f;
        zero8[7] = 0.0f;
        int rr = lane / 2;
        int part = lane - rr * 2;
        int t_d_1 = i_0 * 16 + rr;
        int nchunks = (3 - i_0) * 2;
        #pragma unroll
        for (int c = 0; c < 3; c++) {
            int cc = part + c * 2;
            if (cc < nchunks) {
                if (t_d_1 < clen) {
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(zero8[0 + 0], zero8[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(zero8[0 + 2], zero8[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(zero8[0 + 4], zero8[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(zero8[0 + 6], zero8[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(aqk_ext + (abase + (long long)(t_d_1 * ars) + (long long)((i_0 + 1) * 16 + cc * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
        }
        #pragma unroll
        for (int half_1 = 0; half_1 < 2; half_1++) {
            int t_d_0 = i_0 * 16 + grp + half_1 * 8;
            float bt_d = beta_s[t_d_0];
            #pragma unroll
            for (int nn_2 = 0; nn_2 < 2; nn_2++) {
                int s_d_1 = i_0 * 16 + nn_2 * 8 + tq * 2;
                float v0_1 = acc_k[nn_2 * 4 + half_1 * 2] * bt_d;
                float v1_1 = acc_k[nn_2 * 4 + half_1 * 2 + 1] * bt_d;
                if (s_d_1 >= t_d_0) {
                    v0_1 = 0.0f;
                }
                if (t_d_0 <= s_d_1 + 1) {
                    v1_1 = 0.0f;
                }
                float pr_1[2];
                pr_1[0] = v0_1;
                pr_1[1] = v1_1;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(diag_s_addr + (unsigned int)((i_0 * 16 * 20 + (grp + half_1 * 8) * 20 + nn_2 * 8 + tq * 2) * 4)), "f"(pr_1[0]), "f"(pr_1[1]) : "memory");
            }
        }
        asm volatile("barrier.sync 1, 256;" ::: "memory");
    }
    if (warp < 2) {
        int i_1 = 3;
        int j_1 = warp;
        acc_q[0] = 0.0f;
        acc_q[1] = 0.0f;
        acc_q[2] = 0.0f;
        acc_q[3] = 0.0f;
        acc_q[4] = 0.0f;
        acc_q[5] = 0.0f;
        acc_q[6] = 0.0f;
        acc_q[7] = 0.0f;
        acc_k[0] = 0.0f;
        acc_k[1] = 0.0f;
        acc_k[2] = 0.0f;
        acc_k[3] = 0.0f;
        acc_k[4] = 0.0f;
        acc_k[5] = 0.0f;
        acc_k[6] = 0.0f;
        acc_k[7] = 0.0f;
        #pragma unroll
        for (int kk_1 = 0; kk_1 < 16; kk_1++) {
            int d_1 = kk_1 * 8 + tq * 2;
            float an_1[2];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&an_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&an_1[(0) + 1]))
                : "r"(g_s_addr + (unsigned int)((i_1 * 16 * 136 + d_1) * 4)));
            float gA_1[4];
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gA_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&gA_1[(0) + 1]))
                : "r"(g_s_addr + (unsigned int)(((i_1 * 16 + grp) * 136 + d_1) * 4)));
            asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gA_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&gA_1[(2) + 1]))
                : "r"(g_s_addr + (unsigned int)(((i_1 * 16 + grp + 8) * 136 + d_1) * 4)));
            float eA_1[4];
            float _exp2_6 = approx_exp2(gA_1[0] - an_1[0]);
            eA_1[0] = _exp2_6;
            float _exp2_7 = approx_exp2(gA_1[2] - an_1[0]);
            eA_1[1] = _exp2_7;
            float _exp2_8 = approx_exp2(gA_1[1] - an_1[1]);
            eA_1[2] = _exp2_8;
            float _exp2_9 = approx_exp2(gA_1[3] - an_1[1]);
            eA_1[3] = _exp2_9;
            unsigned int qw_1[2];
            unsigned int kw_1[2];
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&qw_1[0])) : "r"(x_s_addr + (unsigned int)(((i_1 * 16 + grp) * 136 + d_1) * 2)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&qw_1[1])) : "r"(x_s_addr + (unsigned int)(((i_1 * 16 + grp + 8) * 136 + d_1) * 2)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw_1[0])) : "r"(x_s_addr + (unsigned int)(((64 + i_1 * 16 + grp) * 136 + d_1) * 2)));
            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&kw_1[1])) : "r"(x_s_addr + (unsigned int)(((64 + i_1 * 16 + grp + 8) * 136 + d_1) * 2)));
            float qw_f32_1[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&qw_f32_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw_1[_pair]) << 16);
                (&qw_f32_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw_1[_pair]) & 0xffff0000u);
            }
            float kw_f32_1[4];
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&kw_f32_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw_1[_pair]) << 16);
                (&kw_f32_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw_1[_pair]) & 0xffff0000u);
            }
            float aq_1[4];
            float ak_1[4];
            aq_1[0] = qw_f32_1[0] * eA_1[0];
            aq_1[1] = qw_f32_1[2] * eA_1[1];
            aq_1[2] = qw_f32_1[1] * eA_1[2];
            aq_1[3] = qw_f32_1[3] * eA_1[3];
            ak_1[0] = kw_f32_1[0] * eA_1[0];
            ak_1[1] = kw_f32_1[2] * eA_1[1];
            ak_1[2] = kw_f32_1[1] * eA_1[2];
            ak_1[3] = kw_f32_1[3] * eA_1[3];
            unsigned int aqr_1[4];
            unsigned int akr_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                aqr_1[_lp] = __float_as_uint(aq_1[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                akr_1[_lp] = __float_as_uint(ak_1[_lp + 0]);
            }
            #pragma unroll
            for (int nn_3 = 0; nn_3 < 2; nn_3++) {
                int s_1 = j_1 * 16 + nn_3 * 8 + grp;
                float gB_1[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&gB_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&gB_1[(0) + 1]))
                    : "r"(g_s_addr + (unsigned int)((s_1 * 136 + d_1) * 4)));
                unsigned int bw_1[1];
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bw_1[0])) : "r"(x_s_addr + (unsigned int)(((64 + s_1) * 136 + d_1) * 2)));
                float bw_f32_1[2];
                #pragma unroll
                for (int _pair = 0; _pair < 1; _pair++) {
                    (&bw_f32_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(bw_1[_pair]) << 16);
                    (&bw_f32_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(bw_1[_pair]) & 0xffff0000u);
                }
                float b_1[2];
                float _exp2_10 = approx_exp2(an_1[0] - gB_1[0]);
                b_1[0] = bw_f32_1[0] * _exp2_10;
                float _exp2_11 = approx_exp2(an_1[1] - gB_1[1]);
                b_1[1] = bw_f32_1[1] * _exp2_11;
                unsigned int br_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_1[_lp] = __float_as_uint(b_1[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_q + nn_3 * 4)[0]), "+f"((acc_q + nn_3 * 4)[1]), "+f"((acc_q + nn_3 * 4)[2]), "+f"((acc_q + nn_3 * 4)[3])
                    : "r"(aqr_1[0]), "r"(aqr_1[1]), "r"(aqr_1[2]), "r"(aqr_1[3]), "r"(br_1[0]), "r"(br_1[1]));
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc_k + nn_3 * 4)[0]), "+f"((acc_k + nn_3 * 4)[1]), "+f"((acc_k + nn_3 * 4)[2]), "+f"((acc_k + nn_3 * 4)[3])
                    : "r"(akr_1[0]), "r"(akr_1[1]), "r"(akr_1[2]), "r"(akr_1[3]), "r"(br_1[0]), "r"(br_1[1]));
            }
        }
        #pragma unroll
        for (int half_2 = 0; half_2 < 2; half_2++) {
            int t_d_2 = i_1 * 16 + grp + half_2 * 8;
            #pragma unroll
            for (int nn_4 = 0; nn_4 < 2; nn_4++) {
                int s_d_2 = j_1 * 16 + nn_4 * 8 + tq * 2;
                float v0_2 = acc_q[nn_4 * 4 + half_2 * 2] * scale;
                float v1_2 = acc_q[nn_4 * 4 + half_2 * 2 + 1] * scale;
                float pr_2[2];
                pr_2[0] = v0_2;
                pr_2[1] = v1_2;
                if (t_d_2 < clen) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(pr_2[0 + 0], pr_2[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(aqk_ext + (abase + (long long)(t_d_2 * ars) + (long long)s_d_2)))[0]) = _pk;
                    }
                }
            }
        }
    }
    if (warp >= 4) {
        asm volatile("barrier.sync 1, 256;" ::: "memory");
        int di = warp - 4;
        int c_1 = lane % 16;
        int dbase = di * 16 * 20;
        float Ai[16];
        #pragma unroll
        for (int r_2 = 0; r_2 < 16; r_2++) {
            float a_r = 0.0f;
            if (lane < 16) {
                a_r = -diag_s[dbase + r_2 * 20 + c_1];
            }
            Ai[r_2] = a_r;
        }
        #pragma unroll
        for (int i2 = 2; i2 < 16; i2++) {
            float a_c = Ai[i2];
            float tot = a_c;
            #pragma unroll
            for (int jj = 0; jj < i2; jj++) {
                float _shfl_0 = __shfl_sync(0xFFFFFFFF, a_c, jj);
                float a_j = _shfl_0;
                tot = tot + a_j * Ai[jj];
            }
            Ai[i2] = tot;
        }
        if (lane < 16) {
            #pragma unroll
            for (int r_3 = 0; r_3 < 16; r_3++) {
                float v_r = Ai[r_3];
                if (r_3 == c_1) {
                    v_r = v_r + 1.0f;
                }
                diag_s[dbase + r_3 * 20 + c_1] = v_r;
            }
        }
    }
    __syncthreads();
    if (warp >= 4) {
        #pragma unroll
        for (int half_3 = 0; half_3 < 2; half_3++) {
            int t_d_3 = i_0 * 16 + grp + half_3 * 8;
            float bt_d_1 = beta_s[t_d_3];
            #pragma unroll
            for (int nn_5 = 0; nn_5 < 2; nn_5++) {
                float pr_3[2];
                pr_3[0] = acc_k[nn_5 * 4 + half_3 * 2] * bt_d_1;
                pr_3[1] = acc_k[nn_5 * 4 + half_3 * 2 + 1] * bt_d_1;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akkw_s_addr + (unsigned int)((t_d_3 * 72 + j_0 * 16 + nn_5 * 8 + tq * 2) * 4)), "f"(pr_3[0]), "f"(pr_3[1]) : "memory");
            }
        }
    }
    if (warp < 2) {
        #pragma unroll
        for (int half_4 = 0; half_4 < 2; half_4++) {
            int t_d_4 = 48 + grp + half_4 * 8;
            float bt_d_2 = beta_s[t_d_4];
            #pragma unroll
            for (int nn_6 = 0; nn_6 < 2; nn_6++) {
                float pr_4[2];
                pr_4[0] = acc_k[nn_6 * 4 + half_4 * 2] * bt_d_2;
                pr_4[1] = acc_k[nn_6 * 4 + half_4 * 2 + 1] * bt_d_2;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akkw_s_addr + (unsigned int)((t_d_4 * 72 + warp * 16 + nn_6 * 8 + tq * 2) * 4)), "f"(pr_4[0]), "f"(pr_4[1]) : "memory");
            }
        }
    }
    __syncthreads();
    float acc[8];
    if (warp < 3) {
        int i1 = warp + 1;
        int j1 = warp;
        int so1 = warp * 16 * 24;
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_2 = 0; kk_2 < 2; kk_2++) {
            float a[4];
            a[0] = diag_s[i1 * 16 * 20 + grp * 20 + kk_2 * 8 + tq];
            a[1] = diag_s[i1 * 16 * 20 + (grp + 8) * 20 + kk_2 * 8 + tq];
            a[2] = diag_s[i1 * 16 * 20 + grp * 20 + kk_2 * 8 + tq + 4];
            a[3] = diag_s[i1 * 16 * 20 + (grp + 8) * 20 + kk_2 * 8 + tq + 4];
            unsigned int ar[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar[_lp] = __float_as_uint(a[_lp + 0]);
            }
            #pragma unroll
            for (int nn_7 = 0; nn_7 < 2; nn_7++) {
                float b_2[2];
                b_2[0] = akkw_s[i1 * 16 * 72 + j1 * 16 + (kk_2 * 8 + tq) * 72 + nn_7 * 8 + grp];
                b_2[1] = akkw_s[i1 * 16 * 72 + j1 * 16 + (kk_2 * 8 + tq + 4) * 72 + nn_7 * 8 + grp];
                unsigned int br_2[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_2[_lp] = __float_as_uint(b_2[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_7 * 4)[0]), "+f"((acc + nn_7 * 4)[1]), "+f"((acc + nn_7 * 4)[2]), "+f"((acc + nn_7 * 4)[3])
                    : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "r"(br_2[0]), "r"(br_2[1]));
            }
        }
        #pragma unroll
        for (int nn_8 = 0; nn_8 < 2; nn_8++) {
            #pragma unroll
            for (int half_5 = 0; half_5 < 2; half_5++) {
                float v0_3 = acc[nn_8 * 4 + half_5 * 2];
                float v1_3 = acc[nn_8 * 4 + half_5 * 2 + 1];
                float pr_5[2];
                pr_5[0] = v0_3;
                pr_5[1] = v1_3;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(scr_s_addr + (unsigned int)((so1 + (grp + half_5 * 8) * 24 + nn_8 * 8 + tq * 2) * 4)), "f"(pr_5[0]), "f"(pr_5[1]) : "memory");
            }
        }
        __syncwarp();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_3 = 0; kk_3 < 2; kk_3++) {
            float a_1[4];
            a_1[0] = scr_s[so1 + grp * 24 + kk_3 * 8 + tq];
            a_1[1] = scr_s[so1 + (grp + 8) * 24 + kk_3 * 8 + tq];
            a_1[2] = scr_s[so1 + grp * 24 + kk_3 * 8 + tq + 4];
            a_1[3] = scr_s[so1 + (grp + 8) * 24 + kk_3 * 8 + tq + 4];
            unsigned int ar_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_1[_lp] = __float_as_uint(a_1[_lp + 0]);
            }
            #pragma unroll
            for (int nn_9 = 0; nn_9 < 2; nn_9++) {
                float b_3[2];
                b_3[0] = diag_s[j1 * 16 * 20 + (kk_3 * 8 + tq) * 20 + nn_9 * 8 + grp];
                b_3[1] = diag_s[j1 * 16 * 20 + (kk_3 * 8 + tq + 4) * 20 + nn_9 * 8 + grp];
                unsigned int br_3[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_3[_lp] = __float_as_uint(b_3[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_9 * 4)[0]), "+f"((acc + nn_9 * 4)[1]), "+f"((acc + nn_9 * 4)[2]), "+f"((acc + nn_9 * 4)[3])
                    : "r"(ar_1[0]), "r"(ar_1[1]), "r"(ar_1[2]), "r"(ar_1[3]), "r"(br_3[0]), "r"(br_3[1]));
            }
        }
        #pragma unroll
        for (int nn_10 = 0; nn_10 < 2; nn_10++) {
            #pragma unroll
            for (int half_6 = 0; half_6 < 2; half_6++) {
                float v0_4 = acc[nn_10 * 4 + half_6 * 2];
                float v1_4 = acc[nn_10 * 4 + half_6 * 2 + 1];
                v0_4 = -v0_4;
                v1_4 = -v1_4;
                float pr_6[2];
                pr_6[0] = v0_4;
                pr_6[1] = v1_4;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akko_s_addr + (unsigned int)((i1 * 16 * 72 + j1 * 16 + (grp + half_6 * 8) * 72 + nn_10 * 8 + tq * 2) * 4)), "f"(pr_6[0]), "f"(pr_6[1]) : "memory");
            }
        }
    }
    __syncthreads();
    if (warp < 2) {
        int i2_ = warp + 2;
        int j2_ = warp;
        int m2_ = warp + 1;
        int so2 = warp * 16 * 24;
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_4 = 0; kk_4 < 2; kk_4++) {
            float a_2[4];
            a_2[0] = akkw_s[i2_ * 16 * 72 + j2_ * 16 + grp * 72 + kk_4 * 8 + tq];
            a_2[1] = akkw_s[i2_ * 16 * 72 + j2_ * 16 + (grp + 8) * 72 + kk_4 * 8 + tq];
            a_2[2] = akkw_s[i2_ * 16 * 72 + j2_ * 16 + grp * 72 + kk_4 * 8 + tq + 4];
            a_2[3] = akkw_s[i2_ * 16 * 72 + j2_ * 16 + (grp + 8) * 72 + kk_4 * 8 + tq + 4];
            unsigned int ar_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_2[_lp] = __float_as_uint(a_2[_lp + 0]);
            }
            #pragma unroll
            for (int nn_11 = 0; nn_11 < 2; nn_11++) {
                float b_4[2];
                b_4[0] = diag_s[j2_ * 16 * 20 + (kk_4 * 8 + tq) * 20 + nn_11 * 8 + grp];
                b_4[1] = diag_s[j2_ * 16 * 20 + (kk_4 * 8 + tq + 4) * 20 + nn_11 * 8 + grp];
                unsigned int br_4[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_4[_lp] = __float_as_uint(b_4[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_11 * 4)[0]), "+f"((acc + nn_11 * 4)[1]), "+f"((acc + nn_11 * 4)[2]), "+f"((acc + nn_11 * 4)[3])
                    : "r"(ar_2[0]), "r"(ar_2[1]), "r"(ar_2[2]), "r"(ar_2[3]), "r"(br_4[0]), "r"(br_4[1]));
            }
        }
        #pragma unroll
        for (int kk_5 = 0; kk_5 < 2; kk_5++) {
            float a_3[4];
            a_3[0] = akkw_s[i2_ * 16 * 72 + m2_ * 16 + grp * 72 + kk_5 * 8 + tq];
            a_3[1] = akkw_s[i2_ * 16 * 72 + m2_ * 16 + (grp + 8) * 72 + kk_5 * 8 + tq];
            a_3[2] = akkw_s[i2_ * 16 * 72 + m2_ * 16 + grp * 72 + kk_5 * 8 + tq + 4];
            a_3[3] = akkw_s[i2_ * 16 * 72 + m2_ * 16 + (grp + 8) * 72 + kk_5 * 8 + tq + 4];
            unsigned int ar_3[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_3[_lp] = __float_as_uint(a_3[_lp + 0]);
            }
            #pragma unroll
            for (int nn_12 = 0; nn_12 < 2; nn_12++) {
                float b_5[2];
                b_5[0] = akko_s[m2_ * 16 * 72 + j2_ * 16 + (kk_5 * 8 + tq) * 72 + nn_12 * 8 + grp];
                b_5[1] = akko_s[m2_ * 16 * 72 + j2_ * 16 + (kk_5 * 8 + tq + 4) * 72 + nn_12 * 8 + grp];
                unsigned int br_5[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_5[_lp] = __float_as_uint(b_5[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_12 * 4)[0]), "+f"((acc + nn_12 * 4)[1]), "+f"((acc + nn_12 * 4)[2]), "+f"((acc + nn_12 * 4)[3])
                    : "r"(ar_3[0]), "r"(ar_3[1]), "r"(ar_3[2]), "r"(ar_3[3]), "r"(br_5[0]), "r"(br_5[1]));
            }
        }
        #pragma unroll
        for (int nn_13 = 0; nn_13 < 2; nn_13++) {
            #pragma unroll
            for (int half_7 = 0; half_7 < 2; half_7++) {
                float v0_5 = acc[nn_13 * 4 + half_7 * 2];
                float v1_5 = acc[nn_13 * 4 + half_7 * 2 + 1];
                float pr_7[2];
                pr_7[0] = v0_5;
                pr_7[1] = v1_5;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(scr_s_addr + (unsigned int)((so2 + (grp + half_7 * 8) * 24 + nn_13 * 8 + tq * 2) * 4)), "f"(pr_7[0]), "f"(pr_7[1]) : "memory");
            }
        }
        __syncwarp();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_6 = 0; kk_6 < 2; kk_6++) {
            float a_4[4];
            a_4[0] = diag_s[i2_ * 16 * 20 + grp * 20 + kk_6 * 8 + tq];
            a_4[1] = diag_s[i2_ * 16 * 20 + (grp + 8) * 20 + kk_6 * 8 + tq];
            a_4[2] = diag_s[i2_ * 16 * 20 + grp * 20 + kk_6 * 8 + tq + 4];
            a_4[3] = diag_s[i2_ * 16 * 20 + (grp + 8) * 20 + kk_6 * 8 + tq + 4];
            unsigned int ar_4[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_4[_lp] = __float_as_uint(a_4[_lp + 0]);
            }
            #pragma unroll
            for (int nn_14 = 0; nn_14 < 2; nn_14++) {
                float b_6[2];
                b_6[0] = scr_s[so2 + (kk_6 * 8 + tq) * 24 + nn_14 * 8 + grp];
                b_6[1] = scr_s[so2 + (kk_6 * 8 + tq + 4) * 24 + nn_14 * 8 + grp];
                unsigned int br_6[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_6[_lp] = __float_as_uint(b_6[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_14 * 4)[0]), "+f"((acc + nn_14 * 4)[1]), "+f"((acc + nn_14 * 4)[2]), "+f"((acc + nn_14 * 4)[3])
                    : "r"(ar_4[0]), "r"(ar_4[1]), "r"(ar_4[2]), "r"(ar_4[3]), "r"(br_6[0]), "r"(br_6[1]));
            }
        }
        #pragma unroll
        for (int nn_15 = 0; nn_15 < 2; nn_15++) {
            #pragma unroll
            for (int half_8 = 0; half_8 < 2; half_8++) {
                float v0_6 = acc[nn_15 * 4 + half_8 * 2];
                float v1_6 = acc[nn_15 * 4 + half_8 * 2 + 1];
                v0_6 = -v0_6;
                v1_6 = -v1_6;
                float pr_8[2];
                pr_8[0] = v0_6;
                pr_8[1] = v1_6;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akko_s_addr + (unsigned int)((i2_ * 16 * 72 + j2_ * 16 + (grp + half_8 * 8) * 72 + nn_15 * 8 + tq * 2) * 4)), "f"(pr_8[0]), "f"(pr_8[1]) : "memory");
            }
        }
    }
    __syncthreads();
    if (warp < 1) {
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_7 = 0; kk_7 < 2; kk_7++) {
            float a_5[4];
            a_5[0] = akkw_s[3456 + grp * 72 + kk_7 * 8 + tq];
            a_5[1] = akkw_s[3456 + (grp + 8) * 72 + kk_7 * 8 + tq];
            a_5[2] = akkw_s[3456 + grp * 72 + kk_7 * 8 + tq + 4];
            a_5[3] = akkw_s[3456 + (grp + 8) * 72 + kk_7 * 8 + tq + 4];
            unsigned int ar_5[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_5[_lp] = __float_as_uint(a_5[_lp + 0]);
            }
            #pragma unroll
            for (int nn_16 = 0; nn_16 < 2; nn_16++) {
                float b_7[2];
                b_7[0] = diag_s[(kk_7 * 8 + tq) * 20 + nn_16 * 8 + grp];
                b_7[1] = diag_s[(kk_7 * 8 + tq + 4) * 20 + nn_16 * 8 + grp];
                unsigned int br_7[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_7[_lp] = __float_as_uint(b_7[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_16 * 4)[0]), "+f"((acc + nn_16 * 4)[1]), "+f"((acc + nn_16 * 4)[2]), "+f"((acc + nn_16 * 4)[3])
                    : "r"(ar_5[0]), "r"(ar_5[1]), "r"(ar_5[2]), "r"(ar_5[3]), "r"(br_7[0]), "r"(br_7[1]));
            }
        }
        #pragma unroll
        for (int m3 = 1; m3 < 3; m3++) {
            #pragma unroll
            for (int kk_8 = 0; kk_8 < 2; kk_8++) {
                float a_6[4];
                a_6[0] = akkw_s[3456 + m3 * 16 + grp * 72 + kk_8 * 8 + tq];
                a_6[1] = akkw_s[3456 + m3 * 16 + (grp + 8) * 72 + kk_8 * 8 + tq];
                a_6[2] = akkw_s[3456 + m3 * 16 + grp * 72 + kk_8 * 8 + tq + 4];
                a_6[3] = akkw_s[3456 + m3 * 16 + (grp + 8) * 72 + kk_8 * 8 + tq + 4];
                unsigned int ar_6[4];
                #pragma unroll
                for (int _lp = 0; _lp < 4; _lp++) {
                    ar_6[_lp] = __float_as_uint(a_6[_lp + 0]);
                }
                #pragma unroll
                for (int nn_17 = 0; nn_17 < 2; nn_17++) {
                    float b_8[2];
                    b_8[0] = akko_s[m3 * 16 * 72 + (kk_8 * 8 + tq) * 72 + nn_17 * 8 + grp];
                    b_8[1] = akko_s[m3 * 16 * 72 + (kk_8 * 8 + tq + 4) * 72 + nn_17 * 8 + grp];
                    unsigned int br_8[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        br_8[_lp] = __float_as_uint(b_8[_lp + 0]);
                    }
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+f"((acc + nn_17 * 4)[0]), "+f"((acc + nn_17 * 4)[1]), "+f"((acc + nn_17 * 4)[2]), "+f"((acc + nn_17 * 4)[3])
                        : "r"(ar_6[0]), "r"(ar_6[1]), "r"(ar_6[2]), "r"(ar_6[3]), "r"(br_8[0]), "r"(br_8[1]));
                }
            }
        }
        #pragma unroll
        for (int nn_18 = 0; nn_18 < 2; nn_18++) {
            #pragma unroll
            for (int half_9 = 0; half_9 < 2; half_9++) {
                float v0_7 = acc[nn_18 * 4 + half_9 * 2];
                float v1_7 = acc[nn_18 * 4 + half_9 * 2 + 1];
                float pr_9[2];
                pr_9[0] = v0_7;
                pr_9[1] = v1_7;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(scr_s_addr + (unsigned int)(((grp + half_9 * 8) * 24 + nn_18 * 8 + tq * 2) * 4)), "f"(pr_9[0]), "f"(pr_9[1]) : "memory");
            }
        }
        __syncwarp();
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        #pragma unroll
        for (int kk_9 = 0; kk_9 < 2; kk_9++) {
            float a_7[4];
            a_7[0] = diag_s[960 + grp * 20 + kk_9 * 8 + tq];
            a_7[1] = diag_s[960 + (grp + 8) * 20 + kk_9 * 8 + tq];
            a_7[2] = diag_s[960 + grp * 20 + kk_9 * 8 + tq + 4];
            a_7[3] = diag_s[960 + (grp + 8) * 20 + kk_9 * 8 + tq + 4];
            unsigned int ar_7[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                ar_7[_lp] = __float_as_uint(a_7[_lp + 0]);
            }
            #pragma unroll
            for (int nn_19 = 0; nn_19 < 2; nn_19++) {
                float b_9[2];
                b_9[0] = scr_s[(kk_9 * 8 + tq) * 24 + nn_19 * 8 + grp];
                b_9[1] = scr_s[(kk_9 * 8 + tq + 4) * 24 + nn_19 * 8 + grp];
                unsigned int br_9[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br_9[_lp] = __float_as_uint(b_9[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((acc + nn_19 * 4)[0]), "+f"((acc + nn_19 * 4)[1]), "+f"((acc + nn_19 * 4)[2]), "+f"((acc + nn_19 * 4)[3])
                    : "r"(ar_7[0]), "r"(ar_7[1]), "r"(ar_7[2]), "r"(ar_7[3]), "r"(br_9[0]), "r"(br_9[1]));
            }
        }
        #pragma unroll
        for (int nn_20 = 0; nn_20 < 2; nn_20++) {
            #pragma unroll
            for (int half_10 = 0; half_10 < 2; half_10++) {
                float v0_8 = acc[nn_20 * 4 + half_10 * 2];
                float v1_8 = acc[nn_20 * 4 + half_10 * 2 + 1];
                v0_8 = -v0_8;
                v1_8 = -v1_8;
                float pr_10[2];
                pr_10[0] = v0_8;
                pr_10[1] = v1_8;
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(akko_s_addr + (unsigned int)((3456 + (grp + half_10 * 8) * 72 + nn_20 * 8 + tq * 2) * 4)), "f"(pr_10[0]), "f"(pr_10[1]) : "memory");
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int r_4 = 0; r_4 < 2; r_4++) {
        int gi = r_4 * 256 + tid_0;
        int t_e = gi / 8;
        int s0 = (gi - t_e * 8) * 8;
        int bi = t_e / 16;
        int bj = s0 / 16;
        int upper = 0;
        if (bj > bi) {
            upper = 1;
        }
        int src = akko_s_addr + (unsigned int)((t_e * 72 + s0) * 4);
        if (bj == bi) {
            src = diag_s_addr + (unsigned int)((bi * 16 * 20 + (t_e - bi * 16) * 20 + (s0 - bj * 16)) * 4);
        }
        float vk[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vk[0])), "=r"(*reinterpret_cast<uint32_t*>(&vk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vk[(0) + 3]))
            : "r"(src));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&vk[4])), "=r"(*reinterpret_cast<uint32_t*>(&vk[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&vk[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&vk[(4) + 3]))
            : "r"(src + 16));
        #pragma unroll
        for (int e_1 = 0; e_1 < 8; e_1++) {
            float vk_e = vk[e_1];
            if (upper != 0) {
                vk_e = 0.0f;
            }
            vk[e_1] = vk_e;
        }
        if (t_e < clen) {
            long long eidx = abase + (long long)(t_e * ars) + (long long)s0;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vk[0 + 0], vk[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vk[0 + 2], vk[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vk[0 + 4], vk[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vk[0 + 6], vk[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(akk_ext + eidx))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
        if (write_int != 0) {
            #pragma unroll
            for (int e_2 = 0; e_2 < 8; e_2++) {
                float vk2_e = vk[e_2];
                if (t_e >= clen) {
                    vk2_e = 0.0f;
                }
                vk[e_2] = vk2_e;
            }
            long long iidx = (((long long)row0 + (long long)t_e) * (long long)num_heads + (long long)head) * 64 + (long long)s0;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(vk[0 + 0], vk[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vk[0 + 2], vk[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(vk[0 + 4], vk[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(vk[0 + 6], vk[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(akk_int + iidx))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }
}

} // extern "C"
