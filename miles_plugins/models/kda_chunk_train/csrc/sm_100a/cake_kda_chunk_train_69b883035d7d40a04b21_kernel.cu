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
#define SMEM_GRAW_OFF 0
#define SMEM_GRAW_STAGE_BYTES 32768
#define SMEM_GRAW_STRIDE 32768
#define SMEM_GTOT_OFF 32768
#define SMEM_GTOT_STAGE_BYTES 4096
#define SMEM_GTOT_STRIDE 4096
#define SMEM_TOTAL 36864
#define THREADS 512
#define LAUNCH_MIN_BLOCKS 2

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(THREADS, LAUNCH_MIN_BLOCKS) void
kernel_cake_kda_chunk_train_69b883035d7d40a04b21(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ g_raw, __nv_bfloat16* __restrict__ beta_logits, float* __restrict__ A_log, float* __restrict__ dt_bias, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ q_norm_out, __nv_bfloat16* __restrict__ k_norm_out, float* __restrict__ q_rstd_out, float* __restrict__ k_rstd_out, float* __restrict__ beta_out, float* __restrict__ gk_out, __nv_bfloat16* __restrict__ vb_out, __nv_bfloat16* __restrict__ kb_out, __nv_bfloat16* __restrict__ kg_out, __nv_bfloat16* __restrict__ qg_out, int num_qk_heads, int num_heads, int group, int write_qg, float lower_bound)
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
    float* graw = reinterpret_cast<float*>(smem_raw + SMEM_GRAW_OFF);
    const int graw_addr = smem + SMEM_GRAW_OFF;
    float* gtot = reinterpret_cast<float*>(smem_raw + SMEM_GTOT_OFF);
    const int gtot_addr = smem + SMEM_GTOT_OFF;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int head = blockIdx.y;
    int qk_head = head / group;
    int warp_0 = warp;
    int lane_1 = lane;
    int d0 = lane_1 * 4;
    long long ext0 = (long long)chunk_bos[chunk];
    int clen = chunk_len[chunk];
    long long int0 = (long long)chunk * 64;
    int clast = clen - 1;
    if (clast < 0) {
        clast = 0;
    }
    int rs = num_heads * 128;
    int qs = num_qk_heads * 128;
    long long ibase = (int0 * (long long)num_heads + (long long)head) * 128 + (long long)d0;
    long long ebase = (ext0 * (long long)num_heads + (long long)head) * 128 + (long long)d0;
    long long qbase = (ext0 * (long long)num_qk_heads + (long long)qk_head) * 128 + (long long)d0;
    long long bbase = ext0 * (long long)num_heads + (long long)head;
    long long rbase = ext0 * (long long)num_qk_heads + (long long)qk_head;
    int owns_qk = 0;
    if (head - qk_head * group == 0) {
        owns_qk = 1;
    }
    float _expf_0 = __expf(A_log[head]);
    float rate = _expf_0;
    float bias[4];
    {
        float4 _v4 = *reinterpret_cast<const float4*>(dt_bias + (long long)(head * 128 + d0));
        bias[0 + 0] = _v4.x;
        bias[0 + 1] = _v4.y;
        bias[0 + 2] = _v4.z;
        bias[0 + 3] = _v4.w;
    }
    unsigned int xw[8];
    unsigned int qlw[8];
    unsigned int klw[8];
    unsigned int vw[8];
    float blv[4];
    #pragma unroll
    for (int u = 0; u < 4; u++) {
        int t_u = warp_0 * 4 + u;
        int t_c = t_u;
        if (t_u > clast) {
            t_c = clast;
        }
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(g_raw + ebase + (long long)(t_c * rs));
            uint2* _vdst_1 = reinterpret_cast<uint2*>(&xw[u * 2]);
            *_vdst_1 = _vld_1;
        }
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(q + qbase + (long long)(t_c * qs));
            uint2* _vdst_2 = reinterpret_cast<uint2*>(&qlw[u * 2]);
            *_vdst_2 = _vld_2;
        }
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(k + qbase + (long long)(t_c * qs));
            uint2* _vdst_3 = reinterpret_cast<uint2*>(&klw[u * 2]);
            *_vdst_3 = _vld_3;
        }
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(v + ebase + (long long)(t_c * rs));
            uint2* _vdst_4 = reinterpret_cast<uint2*>(&vw[u * 2]);
            *_vdst_4 = _vld_4;
        }
        __nv_bfloat16 bl = beta_logits[bbase + (long long)(t_c * num_heads)];
        float _cvt_f32_0 = __bfloat162float(bl);
        blv[u] = _cvt_f32_0;
    }
    unsigned int qw[8];
    unsigned int kw[8];
    float btv[4];
    #pragma unroll
    for (int u_1 = 0; u_1 < 4; u_1++) {
        int t_u_1 = warp_0 * 4 + u_1;
        int t_c_1 = t_u_1;
        if (t_u_1 > clast) {
            t_c_1 = clast;
        }
        long long qidx = qbase + (long long)(t_c_1 * qs);
        unsigned int xw_u[2];
        unsigned int qlw_u[2];
        unsigned int klw_u[2];
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            xw_u[h] = xw[u_1 * 2 + h];
            qlw_u[h] = qlw[u_1 * 2 + h];
            klw_u[h] = klw[u_1 * 2 + h];
        }
        float xf[4];
        float qf[4];
        float kf[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&xf[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(xw_u[_pair]) << 16);
            (&xf[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(xw_u[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&qf[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qlw_u[_pair]) << 16);
            (&qf[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qlw_u[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&kf[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(klw_u[_pair]) << 16);
            (&kf[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(klw_u[_pair]) & 0xffff0000u);
        }
        float sq = qf[0] * qf[0];
        float sk = kf[0] * kf[0];
        #pragma unroll
        for (int e = 1; e < 4; e++) {
            sq = sq + qf[e] * qf[e];
            sk = sk + kf[e] * kf[e];
        }
        float _warp_reduce_0 = sq;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
        sq = _warp_reduce_0;
        float _warp_reduce_1 = sk;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
        sk = _warp_reduce_1;
        float _sqrt_0;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_0) : "f"(sq + 1e-06f));
        float _fdiv_rn_0 = __fdiv_rn(1.0f, _sqrt_0);
        float rq = _fdiv_rn_0;
        float _sqrt_1;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_1) : "f"(sk + 1e-06f));
        float _fdiv_rn_1 = __fdiv_rn(1.0f, _sqrt_1);
        float rk = _fdiv_rn_1;
        float _exp_0 = expf(-blv[u_1]);
        float _fdiv_rn_2 = __fdiv_rn(1.0f, 1.0f + _exp_0);
        float bt_l = _fdiv_rn_2;
        float bt_u = 0.0f;
        if (t_u_1 < clen) {
            bt_u = bt_l;
        }
        btv[u_1] = bt_u;
        float qn4[4];
        float kn4[4];
        float gq[4];
        #pragma unroll
        for (int e_1 = 0; e_1 < 4; e_1++) {
            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(qf[e_1] * rq);
            float _cvt_f32_1 = __bfloat162float(_cvt_bf16_0);
            float qn_l = _cvt_f32_1;
            __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(kf[e_1] * rk);
            float _cvt_f32_2 = __bfloat162float(_cvt_bf16_1);
            float kn_l = _cvt_f32_2;
            float _expf_1 = __expf(-(rate * (xf[e_1] + bias[e_1])));
            float _rcp_0 = approx_rcp(1.0f + _expf_1);
            float gate_l = lower_bound * _rcp_0;
            float qn_u = 0.0f;
            float kn_u = 0.0f;
            float gate_u = 0.0f;
            if (t_u_1 < clen) {
                qn_u = qn_l;
                kn_u = kn_l;
                gate_u = gate_l;
            }
            qn4[e_1] = qn_u;
            kn4[e_1] = kn_u;
            gq[e_1] = gate_u;
        }
        uint32_t qn4_bf16[2];
        #pragma unroll
        for (int _lp = 0; _lp < 2; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qn4[_lp*2 + 0], qn4[_lp*2+1 + 0]));
            qn4_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t kn4_bf16[2];
        #pragma unroll
        for (int _lp = 0; _lp < 2; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kn4[_lp*2 + 0], kn4[_lp*2+1 + 0]));
            kn4_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int h_1 = 0; h_1 < 2; h_1++) {
            qw[u_1 * 2 + h_1] = qn4_bf16[h_1];
            kw[u_1 * 2 + h_1] = kn4_bf16[h_1];
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(graw_addr + (unsigned int)((t_u_1 * 128 + d0) * 4)), "f"(gq[0]), "f"(gq[1]), "f"(gq[2]), "f"(gq[3]) : "memory");
        if (t_u_1 < clen) {
            if (owns_qk != 0) {
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(qn4[0 + 0], qn4[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(qn4[0 + 2], qn4[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(q_norm_out + qidx))[0]) = _pk2;
                }
                {
                    uint2 _pk2;
                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                    _pk[0] = __floats2bfloat162_rn(kn4[0 + 0], kn4[0 + 1]);
                    _pk[1] = __floats2bfloat162_rn(kn4[0 + 2], kn4[0 + 3]);
                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(k_norm_out + qidx))[0]) = _pk2;
                }
                if (lane_1 == 0) {
                    q_rstd_out[rbase + (long long)(t_c_1 * num_qk_heads)] = rq;
                    k_rstd_out[rbase + (long long)(t_c_1 * num_qk_heads)] = rk;
                }
            }
            if (lane_1 == 0) {
                beta_out[bbase + (long long)(t_c_1 * num_heads)] = bt_l;
            }
        }
    }
    __syncthreads();
    int grp = warp_0 / 2;
    int pos0 = (warp_0 - grp * 2) * 4;
    float local[4];
    #pragma unroll
    for (int e_2 = 0; e_2 < 4; e_2++) {
        local[e_2] = 0.0f;
    }
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        if (pos0 >= p) {
            float gp[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gp[0])), "=r"(*reinterpret_cast<uint32_t*>(&gp[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gp[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gp[(0) + 3]))
                : "r"(graw_addr + (unsigned int)(((grp * 8 + p) * 128 + d0) * 4)));
            #pragma unroll
            for (int e_3 = 0; e_3 < 4; e_3++) {
                local[e_3] = local[e_3] + gp[e_3];
            }
        }
    }
    if (pos0 == 4) {
        float tot[4];
        #pragma unroll
        for (int e_4 = 0; e_4 < 4; e_4++) {
            tot[e_4] = local[e_4];
        }
        #pragma unroll
        for (int u_2 = 1; u_2 < 4; u_2++) {
            float gu[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gu[0])), "=r"(*reinterpret_cast<uint32_t*>(&gu[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gu[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gu[(0) + 3]))
                : "r"(graw_addr + (unsigned int)(((warp_0 * 4 + u_2) * 128 + d0) * 4)));
            #pragma unroll
            for (int e_5 = 0; e_5 < 4; e_5++) {
                tot[e_5] = tot[e_5] + gu[e_5];
            }
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(gtot_addr + (unsigned int)((grp * 128 + d0) * 4)), "f"(tot[0]), "f"(tot[1]), "f"(tot[2]), "f"(tot[3]) : "memory");
    }
    __syncthreads();
    float off[4];
    float total[4];
    float gk_n[4];
    #pragma unroll
    for (int e_6 = 0; e_6 < 4; e_6++) {
        off[e_6] = 0.0f;
        total[e_6] = 0.0f;
    }
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        float tg[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tg[0])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 3]))
            : "r"(gtot_addr + (unsigned int)((g * 128 + d0) * 4)));
        if (grp > g) {
            #pragma unroll
            for (int e_7 = 0; e_7 < 4; e_7++) {
                off[e_7] = off[e_7] + tg[e_7];
            }
        }
        #pragma unroll
        for (int e_8 = 0; e_8 < 4; e_8++) {
            total[e_8] = total[e_8] + tg[e_8];
        }
    }
    #pragma unroll
    for (int e_9 = 0; e_9 < 4; e_9++) {
        gk_n[e_9] = total[e_9] * 1.4426950408889634f;
    }
    float acc[4];
    #pragma unroll
    for (int e_10 = 0; e_10 < 4; e_10++) {
        acc[e_10] = local[e_10];
    }
    #pragma unroll
    for (int u_3 = 0; u_3 < 4; u_3++) {
        int t_u_2 = warp_0 * 4 + u_3;
        long long iidx = ibase + (long long)(t_u_2 * rs);
        float bt = btv[u_3];
        if (u_3 > 0) {
            float gu2[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gu2[0])), "=r"(*reinterpret_cast<uint32_t*>(&gu2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gu2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gu2[(0) + 3]))
                : "r"(graw_addr + (unsigned int)((t_u_2 * 128 + d0) * 4)));
            #pragma unroll
            for (int e_11 = 0; e_11 < 4; e_11++) {
                acc[e_11] = acc[e_11] + gu2[e_11];
            }
        }
        unsigned int vw_u[2];
        #pragma unroll
        for (int h_2 = 0; h_2 < 2; h_2++) {
            vw_u[h_2] = vw[u_3 * 2 + h_2];
        }
        float vf[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&vf[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(vw_u[_pair]) << 16);
            (&vf[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(vw_u[_pair]) & 0xffff0000u);
        }
        unsigned int qw2[2];
        unsigned int kw2[2];
        #pragma unroll
        for (int h_3 = 0; h_3 < 2; h_3++) {
            qw2[h_3] = qw[u_3 * 2 + h_3];
            kw2[h_3] = kw[u_3 * 2 + h_3];
        }
        float qw2_f32[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&qw2_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(qw2[_pair]) << 16);
            (&qw2_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(qw2[_pair]) & 0xffff0000u);
        }
        float kw2_f32[4];
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&kw2_f32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(kw2[_pair]) << 16);
            (&kw2_f32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(kw2[_pair]) & 0xffff0000u);
        }
        float gkt[4];
        float vb[4];
        float kb[4];
        float kg[4];
        float qg[4];
        #pragma unroll
        for (int e_12 = 0; e_12 < 4; e_12++) {
            gkt[e_12] = (off[e_12] + acc[e_12]) * 1.4426950408889634f;
            float _exp2_0 = approx_exp2(gkt[e_12]);
            float e_g = _exp2_0;
            float v_e = 0.0f;
            if (t_u_2 < clen) {
                v_e = vf[e_12];
            }
            float kn_e = kw2_f32[e_12];
            vb[e_12] = v_e * bt;
            kb[e_12] = kn_e * bt * e_g;
            float _exp2_1 = approx_exp2(gk_n[e_12] - gkt[e_12]);
            kg[e_12] = kn_e * _exp2_1;
            qg[e_12] = qw2_f32[e_12] * e_g;
        }
        {
            float4 _v4 = make_float4(gkt[0 + 0], gkt[0 + 1], gkt[0 + 2], gkt[0 + 3]);
            *reinterpret_cast<float4*>((gk_out + iidx) + 0) = _v4;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(vb[0 + 0], vb[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(vb[0 + 2], vb[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(vb_out + iidx))[0]) = _pk2;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(kb[0 + 0], kb[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(kb[0 + 2], kb[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(kb_out + iidx))[0]) = _pk2;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(kg[0 + 0], kg[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(kg[0 + 2], kg[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(kg_out + iidx))[0]) = _pk2;
        }
        if (write_qg != 0) {
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(qg[0 + 0], qg[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(qg[0 + 2], qg[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(qg_out + iidx))[0]) = _pk2;
            }
        }
    }
}

} // extern "C"
