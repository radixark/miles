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
#define SMEM_GRAW_OFF 0
#define SMEM_GRAW_STAGE_BYTES 32768
#define SMEM_GRAW_STRIDE 32768
#define SMEM_GTOT_OFF 32768
#define SMEM_GTOT_STAGE_BYTES 4096
#define SMEM_GTOT_STRIDE 4096
#define SMEM_TOTAL 36864
#define THREADS 1024

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

__global__ __launch_bounds__(1024) void
kernel_cake_kda_chunk_train_35c13d4668c0cafffff2(__nv_bfloat16* __restrict__ g_raw, __nv_bfloat16* __restrict__ q_norm, __nv_bfloat16* __restrict__ k_norm, __nv_bfloat16* __restrict__ v, float* __restrict__ beta, float* __restrict__ A_log, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ aqk, __nv_bfloat16* __restrict__ akk, __nv_bfloat16* __restrict__ do_, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, float* __restrict__ gk_out, __nv_bfloat16* __restrict__ vb_out, __nv_bfloat16* __restrict__ kb_out, __nv_bfloat16* __restrict__ qg_out, __nv_bfloat16* __restrict__ kg_out, __nv_bfloat16* __restrict__ ke_out, __nv_bfloat16* __restrict__ qe_out, __nv_bfloat16* __restrict__ aqk_tril, __nv_bfloat16* __restrict__ akk_int, __nv_bfloat16* __restrict__ do_int, __nv_bfloat16* __restrict__ v_int, float* __restrict__ beta_int, int num_qk_heads, int num_heads, int group, int copy_inputs, float lower_bound)
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
    float* graw = reinterpret_cast<float*>(smem_raw + 0);
    const int graw_addr = smem + 0;
    float* gtot = reinterpret_cast<float*>(smem_raw + 32768);
    const int gtot_addr = smem + 32768;

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
    float gates[8];
    #pragma unroll
    for (int u = 0; u < 2; u++) {
        int t_u = warp_0 * 2 + u;
        int t_c = t_u;
        if (t_u > clast) {
            t_c = clast;
        }
        long long gidx = ebase + (long long)(t_c * rs);
        float x[4];
        {
            uint2 _vld_1;
            _vld_1 = *reinterpret_cast<const uint2*>(g_raw + gidx);
            uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&x[0 + _pair * 2])[0]), "=f"((&x[0 + _pair * 2])[1])
                    : "r"(_vpairs_1[_pair]));
            }
        }
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            float _expf_1 = __expf(-(rate * (x[e] + bias[e])));
            float _rcp_0 = approx_rcp(1.0f + _expf_1);
            float gate_l = lower_bound * _rcp_0;
            float gate_u = 0.0f;
            if (t_u < clen) {
                gate_u = gate_l;
            }
            gates[u * 4 + e] = gate_u;
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(graw_addr + (unsigned int)((t_u * 128 + d0) * 4)), "f"(gates[u * 4]), "f"(gates[u * 4 + 1]), "f"(gates[u * 4 + 2]), "f"(gates[u * 4 + 3]) : "memory");
    }
    __syncthreads();
    int grp = warp_0 / 4;
    int pos0 = (warp_0 - grp * 4) * 2;
    float local[4];
    #pragma unroll
    for (int e_1 = 0; e_1 < 4; e_1++) {
        local[e_1] = 0.0f;
    }
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        if (pos0 >= p) {
            float gp[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&gp[0])), "=r"(*reinterpret_cast<uint32_t*>(&gp[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gp[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gp[(0) + 3]))
                : "r"(graw_addr + (unsigned int)(((grp * 8 + p) * 128 + d0) * 4)));
            #pragma unroll
            for (int e_2 = 0; e_2 < 4; e_2++) {
                local[e_2] = local[e_2] + gp[e_2];
            }
        }
    }
    #pragma unroll
    for (int e_3 = 0; e_3 < 4; e_3++) {
        gates[e_3] = local[e_3];
        gates[4 + e_3] = local[e_3] + gates[4 + e_3];
    }
    if (pos0 == 6) {
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(gtot_addr + (unsigned int)((grp * 128 + d0) * 4)), "f"(gates[4]), "f"(gates[5]), "f"(gates[6]), "f"(gates[7]) : "memory");
    }
    __syncthreads();
    float off[4];
    float total[4];
    float gk_n[4];
    #pragma unroll
    for (int e_4 = 0; e_4 < 4; e_4++) {
        off[e_4] = 0.0f;
        total[e_4] = 0.0f;
    }
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        float tg[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&tg[0])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tg[(0) + 3]))
            : "r"(gtot_addr + (unsigned int)((g * 128 + d0) * 4)));
        if (grp > g) {
            #pragma unroll
            for (int e_5 = 0; e_5 < 4; e_5++) {
                off[e_5] = off[e_5] + tg[e_5];
            }
        }
        #pragma unroll
        for (int e_6 = 0; e_6 < 4; e_6++) {
            total[e_6] = total[e_6] + tg[e_6];
        }
    }
    #pragma unroll
    for (int e_7 = 0; e_7 < 4; e_7++) {
        gk_n[e_7] = total[e_7] * 1.4426950408889634f;
    }
    #pragma unroll
    for (int u_1 = 0; u_1 < 2; u_1++) {
        int t_u_1 = warp_0 * 2 + u_1;
        int t_c_1 = t_u_1;
        if (t_u_1 > clast) {
            t_c_1 = clast;
        }
        long long iidx = ibase + (long long)(t_u_1 * rs);
        long long eidx = ebase + (long long)(t_c_1 * rs);
        long long qidx = qbase + (long long)(t_c_1 * qs);
        float bt_l = beta[bbase + (long long)(t_c_1 * num_heads)];
        float kv[4];
        float qv[4];
        float vv[4];
        {
            uint2 _vld_2;
            _vld_2 = *reinterpret_cast<const uint2*>(k_norm + qidx);
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&kv[0 + _pair * 2])[0]), "=f"((&kv[0 + _pair * 2])[1])
                    : "r"(_vpairs_2[_pair]));
            }
        }
        {
            uint2 _vld_3;
            _vld_3 = *reinterpret_cast<const uint2*>(q_norm + qidx);
            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&qv[0 + _pair * 2])[0]), "=f"((&qv[0 + _pair * 2])[1])
                    : "r"(_vpairs_3[_pair]));
            }
        }
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(v + eidx);
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&vv[0 + _pair * 2])[0]), "=f"((&vv[0 + _pair * 2])[1])
                    : "r"(_vpairs_4[_pair]));
            }
        }
        float bt = 0.0f;
        if (t_u_1 < clen) {
            bt = bt_l;
        }
        float gkt[4];
        float vb[4];
        float kb[4];
        float qg[4];
        float kg[4];
        float ke[4];
        float qe[4];
        #pragma unroll
        for (int e_8 = 0; e_8 < 4; e_8++) {
            gkt[e_8] = (off[e_8] + gates[u_1 * 4 + e_8]) * 1.4426950408889634f;
            float _exp2_0 = approx_exp2(gkt[e_8]);
            float e_g = _exp2_0;
            float k_e = 0.0f;
            float q_e = 0.0f;
            float v_e = 0.0f;
            if (t_u_1 < clen) {
                k_e = kv[e_8];
                q_e = qv[e_8];
                v_e = vv[e_8];
            }
            vb[e_8] = v_e * bt;
            __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(k_e * bt);
            float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
            kb[e_8] = _cvt_f32_0 * e_g;
            qg[e_8] = q_e * e_g;
            float _exp2_1 = approx_exp2(gk_n[e_8] - gkt[e_8]);
            kg[e_8] = k_e * _exp2_1;
            ke[e_8] = k_e;
            qe[e_8] = q_e;
        }
        {
            float4 _v4 = make_float4(gkt[0 + 0], gkt[0 + 1], gkt[0 + 2], gkt[0 + 3]);
            *reinterpret_cast<float4*>(gk_out + iidx + 0) = _v4;
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
            _pk[0] = __floats2bfloat162_rn(qg[0 + 0], qg[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(qg[0 + 2], qg[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(qg_out + iidx))[0]) = _pk2;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(kg[0 + 0], kg[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(kg[0 + 2], kg[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(kg_out + iidx))[0]) = _pk2;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(ke[0 + 0], ke[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(ke[0 + 2], ke[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(ke_out + iidx))[0]) = _pk2;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(qe[0 + 0], qe[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(qe[0 + 2], qe[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(qe_out + iidx))[0]) = _pk2;
        }
    }
    if (copy_inputs != 0) {
        #pragma unroll
        for (int u_2 = 0; u_2 < 2; u_2++) {
            int t_u_2 = warp_0 * 2 + u_2;
            int t_c_2 = t_u_2;
            if (t_u_2 > clast) {
                t_c_2 = clast;
            }
            long long iidx_1 = ibase + (long long)(t_u_2 * rs);
            long long eidx_1 = ebase + (long long)(t_c_2 * rs);
            float vc[4];
            float dc[4];
            {
                uint2 _vld_5;
                _vld_5 = *reinterpret_cast<const uint2*>(v + eidx_1);
                uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&vc[0 + _pair * 2])[0]), "=f"((&vc[0 + _pair * 2])[1])
                        : "r"(_vpairs_5[_pair]));
                }
            }
            {
                uint2 _vld_6;
                _vld_6 = *reinterpret_cast<const uint2*>(do_ + eidx_1);
                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    asm volatile(
                        "{\n\t"
                        "shl.b32 %0, %2, 16;\n\t"
                        "and.b32 %1, %2, 0xffff0000;\n\t"
                        "}\n"
                        : "=f"((&dc[0 + _pair * 2])[0]), "=f"((&dc[0 + _pair * 2])[1])
                        : "r"(_vpairs_6[_pair]));
                }
            }
            float bc_l = beta[bbase + (long long)(t_c_2 * num_heads)];
            float bc = 0.0f;
            if (t_u_2 < clen) {
                bc = bc_l;
            }
            #pragma unroll
            for (int e_9 = 0; e_9 < 4; e_9++) {
                float vc_e = 0.0f;
                float dc_e = 0.0f;
                if (t_u_2 < clen) {
                    vc_e = vc[e_9];
                    dc_e = dc[e_9];
                }
                vc[e_9] = vc_e;
                dc[e_9] = dc_e;
            }
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(vc[0 + 0], vc[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(vc[0 + 2], vc[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(v_int + iidx_1))[0]) = _pk2;
            }
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(dc[0 + 0], dc[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(dc[0 + 2], dc[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(do_int + iidx_1))[0]) = _pk2;
            }
            if (lane_1 == 0) {
                beta_int[(int0 + (long long)t_u_2) * (long long)num_heads + (long long)head] = bc;
            }
        }
    }
    int t_a = tid / 16;
    int s0 = (tid - t_a * 16) * 4;
    int t_ac = t_a;
    if (t_a > clast) {
        t_ac = clast;
    }
    long long iidx_a = ((int0 + (long long)t_a) * (long long)num_heads + (long long)head) * 64 + (long long)s0;
    long long eidx_a = ((ext0 + (long long)t_ac) * (long long)num_heads + (long long)head) * 64 + (long long)s0;
    float raw[4];
    {
        uint2 _vld_7;
        _vld_7 = *reinterpret_cast<const uint2*>(aqk + eidx_a);
        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&raw[0 + _pair * 2])[0]), "=f"((&raw[0 + _pair * 2])[1])
                : "r"(_vpairs_7[_pair]));
        }
    }
    #pragma unroll
    for (int e_10 = 0; e_10 < 4; e_10++) {
        float val = 0.0f;
        if (t_a < clen) {
            if (t_a >= s0 + e_10) {
                val = raw[e_10];
            }
        }
        raw[e_10] = val;
    }
    {
        uint2 _pk2;
        __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
        _pk[0] = __floats2bfloat162_rn(raw[0 + 0], raw[0 + 1]);
        _pk[1] = __floats2bfloat162_rn(raw[0 + 2], raw[0 + 3]);
        *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(aqk_tril + iidx_a))[0]) = _pk2;
    }
    if (copy_inputs != 0) {
        float rawk[4];
        {
            uint2 _vld_8;
            _vld_8 = *reinterpret_cast<const uint2*>(akk + eidx_a);
            uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    "shl.b32 %0, %2, 16;\n\t"
                    "and.b32 %1, %2, 0xffff0000;\n\t"
                    "}\n"
                    : "=f"((&rawk[0 + _pair * 2])[0]), "=f"((&rawk[0 + _pair * 2])[1])
                    : "r"(_vpairs_8[_pair]));
            }
        }
        #pragma unroll
        for (int e_11 = 0; e_11 < 4; e_11++) {
            float valk = 0.0f;
            if (t_a < clen) {
                valk = rawk[e_11];
            }
            rawk[e_11] = valk;
        }
        {
            uint2 _pk2;
            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
            _pk[0] = __floats2bfloat162_rn(rawk[0 + 0], rawk[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(rawk[0 + 2], rawk[0 + 3]);
            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(akk_int + iidx_a))[0]) = _pk2;
        }
    }
}

} // extern "C"
