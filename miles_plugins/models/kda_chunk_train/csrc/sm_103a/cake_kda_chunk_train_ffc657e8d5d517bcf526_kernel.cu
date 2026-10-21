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
#define SMEM_DAQK_S_OFF 0
#define SMEM_DAQK_S_STAGE_BYTES 17408
#define SMEM_DAQK_S_STRIDE 17408
#define SMEM_DAKK_S_OFF 17408
#define SMEM_DAKK_S_STAGE_BYTES 17408
#define SMEM_DAKK_S_STRIDE 17408
#define SMEM_G_S_OFF 34816
#define SMEM_G_S_STAGE_BYTES 34816
#define SMEM_G_S_STRIDE 34816
#define SMEM_K_S_OFF 69632
#define SMEM_K_S_STAGE_BYTES 17408
#define SMEM_K_S_STRIDE 17408
#define SMEM_Q_S_OFF 87040
#define SMEM_Q_S_STAGE_BYTES 17408
#define SMEM_Q_S_STRIDE 17408
#define SMEM_BETA_S_OFF 104448
#define SMEM_BETA_S_STAGE_BYTES 256
#define SMEM_BETA_S_STRIDE 256
#define SMEM_DQ_S_OFF 104704
#define SMEM_DQ_S_STAGE_BYTES 34816
#define SMEM_DQ_S_STRIDE 34816
#define SMEM_DK_S_OFF 139520
#define SMEM_DK_S_STAGE_BYTES 34816
#define SMEM_DK_S_STRIDE 34816
#define SMEM_RQ_S_OFF 174336
#define SMEM_RQ_S_STAGE_BYTES 256
#define SMEM_RQ_S_STRIDE 256
#define SMEM_RK_S_OFF 174592
#define SMEM_RK_S_STAGE_BYTES 256
#define SMEM_RK_S_STRIDE 256
#define SMEM_DOTQ_S_OFF 174848
#define SMEM_DOTQ_S_STAGE_BYTES 256
#define SMEM_DOTQ_S_STRIDE 256
#define SMEM_DOTK_S_OFF 175104
#define SMEM_DOTK_S_STAGE_BYTES 256
#define SMEM_DOTK_S_STRIDE 256
#define SMEM_TOTAL 175360
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_kda_chunk_train_ffc657e8d5d517bcf526(float* __restrict__ dAqk, float* __restrict__ dAkk, float* __restrict__ gk, __nv_bfloat16* __restrict__ k_e, __nv_bfloat16* __restrict__ q_e, float* __restrict__ beta, float* __restrict__ dq_f, float* __restrict__ dk_f, float* __restrict__ dg_f, float* __restrict__ db_f, float* __restrict__ q_rstd, float* __restrict__ k_rstd, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ dq_ext, __nv_bfloat16* __restrict__ dk_ext, float* __restrict__ dg_out, float* __restrict__ db_out, int num_heads, int num_qk_heads, int group)
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
    float* dAqk_s = reinterpret_cast<float*>(smem_raw + SMEM_DAQK_S_OFF);
    const int dAqk_s_addr = smem + SMEM_DAQK_S_OFF;
    float* dAkk_s = reinterpret_cast<float*>(smem_raw + SMEM_DAKK_S_OFF);
    const int dAkk_s_addr = smem + SMEM_DAKK_S_OFF;
    float* g_s = reinterpret_cast<float*>(smem_raw + SMEM_G_S_OFF);
    const int g_s_addr = smem + SMEM_G_S_OFF;
    __nv_bfloat16* k_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_K_S_OFF);
    const int k_s_addr = smem + SMEM_K_S_OFF;
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_Q_S_OFF);
    const int q_s_addr = smem + SMEM_Q_S_OFF;
    float* beta_s = reinterpret_cast<float*>(smem_raw + SMEM_BETA_S_OFF);
    const int beta_s_addr = smem + SMEM_BETA_S_OFF;
    float* dq_s = reinterpret_cast<float*>(smem_raw + SMEM_DQ_S_OFF);
    const int dq_s_addr = smem + SMEM_DQ_S_OFF;
    float* dk_s = reinterpret_cast<float*>(smem_raw + SMEM_DK_S_OFF);
    const int dk_s_addr = smem + SMEM_DK_S_OFF;
    float* rq_s = reinterpret_cast<float*>(smem_raw + SMEM_RQ_S_OFF);
    const int rq_s_addr = smem + SMEM_RQ_S_OFF;
    float* rk_s = reinterpret_cast<float*>(smem_raw + SMEM_RK_S_OFF);
    const int rk_s_addr = smem + SMEM_RK_S_OFF;
    float* dotq_s = reinterpret_cast<float*>(smem_raw + SMEM_DOTQ_S_OFF);
    const int dotq_s_addr = smem + SMEM_DOTQ_S_OFF;
    float* dotk_s = reinterpret_cast<float*>(smem_raw + SMEM_DOTK_S_OFF);
    const int dotk_s_addr = smem + SMEM_DOTK_S_OFF;

    // === Task calls (dependency order) ===
    int chunk = blockIdx.x;
    int head = blockIdx.y;
    int qk_head = head / group;
    int row0 = chunk * 64;
    int tid_0 = tid;
    long long ext0 = (long long)chunk_bos[chunk];
    int clen = chunk_len[chunk];
    if (tid_0 < 64) {
        beta_s[tid_0] = beta[(row0 + tid_0) * num_heads + head];
        float rq_v = 0.0f;
        float rk_v = 0.0f;
        if (tid_0 < clen) {
            rq_v = q_rstd[(ext0 + (long long)tid_0) * (long long)num_qk_heads + (long long)qk_head];
            rk_v = k_rstd[(ext0 + (long long)tid_0) * (long long)num_qk_heads + (long long)qk_head];
        }
        rq_s[tid_0] = rq_v;
        rk_s[tid_0] = rk_v;
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int e0 = (r * 256 + tid_0) * 4;
        int t_a = e0 / 64;
        int s_a = e0 - t_a * 64;
        int gidx = ((row0 + t_a) * num_heads + head) * 64 + s_a;
        float va[4];
        float vb[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(dAqk + gidx);
            va[0 + 0] = _v4.x;
            va[0 + 1] = _v4.y;
            va[0 + 2] = _v4.z;
            va[0 + 3] = _v4.w;
        }
        {
            float4 _v4 = *reinterpret_cast<const float4*>(dAkk + gidx);
            vb[0 + 0] = _v4.x;
            vb[0 + 1] = _v4.y;
            vb[0 + 2] = _v4.z;
            vb[0 + 3] = _v4.w;
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dAqk_s_addr + (unsigned int)((t_a * 68 + s_a) * 4)), "f"(va[0]), "f"(va[1]), "f"(va[2]), "f"(va[3]) : "memory");
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(dAkk_s_addr + (unsigned int)((t_a * 68 + s_a) * 4)), "f"(vb[0]), "f"(vb[1]), "f"(vb[2]), "f"(vb[3]) : "memory");
    }
    #pragma unroll
    for (int r_1 = 0; r_1 < 8; r_1++) {
        int e1 = (r_1 * 256 + tid_0) * 4;
        int t_b = e1 / 128;
        int d_b = e1 - t_b * 128;
        int gidx2 = ((row0 + t_b) * num_heads + head) * 128 + d_b;
        float gv[4];
        {
            float4 _v4 = *reinterpret_cast<const float4*>(gk + gidx2);
            gv[0 + 0] = _v4.x;
            gv[0 + 1] = _v4.y;
            gv[0 + 2] = _v4.z;
            gv[0 + 3] = _v4.w;
        }
        asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};" :: "r"(g_s_addr + (unsigned int)((t_b * 136 + d_b) * 4)), "f"(gv[0]), "f"(gv[1]), "f"(gv[2]), "f"(gv[3]) : "memory");
    }
    #pragma unroll
    for (int r_2 = 0; r_2 < 4; r_2++) {
        int e2 = (r_2 * 256 + tid_0) * 8;
        int t_c = e2 / 128;
        int d_c = e2 - t_c * 128;
        float kv[8];
        float qv[8];
        int qk_idx2 = ((row0 + t_c) * num_qk_heads + qk_head) * 128 + d_c;
        {
            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(k_e + qk_idx2);
            uint4 _vld_3[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_3[_blk] = _vptr_3[_blk];
                uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&kv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                    (&kv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                }
            }
        }
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(q_e + qk_idx2);
            uint4 _vld_4[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_4[_blk] = _vptr_4[_blk];
                uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&qv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                    (&qv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                }
            }
        }
        uint32_t kv_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kv[_lp*2 + 0], kv[_lp*2+1 + 0]));
            kv_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        uint32_t qv_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qv[_lp*2 + 0], qv[_lp*2+1 + 0]));
            qv_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(k_s_addr + (unsigned int)((t_c * 136 + d_c) * 2)), "r"(kv_bf16[0]), "r"(kv_bf16[1]), "r"(kv_bf16[2]), "r"(kv_bf16[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q_s_addr + (unsigned int)((t_c * 136 + d_c) * 2)), "r"(qv_bf16[0]), "r"(qv_bf16[1]), "r"(qv_bf16[2]), "r"(qv_bf16[3]) : "memory");
    }
    __syncthreads();
    int i = warp % 4;
    int kb_half = warp / 4;
    int grp = lane / 4;
    int tq = lane % 4;
    int r0 = i * 16;
    float db_lo = 0.0f;
    float db_hi = 0.0f;
    float dotq_lo = 0.0f;
    float dotq_hi = 0.0f;
    float dotk_lo = 0.0f;
    float dotk_hi = 0.0f;
    #pragma unroll
    for (int kbi = 0; kbi < 2; kbi++) {
        int kb = kb_half * 2 + kbi;
        int c_base = kb * 32;
        float dqf_pre[16];
        float dkf_pre[16];
        float dgf_pre[16];
        #pragma unroll
        for (int nn_p = 0; nn_p < 4; nn_p++) {
            #pragma unroll
            for (int half_p = 0; half_p < 2; half_p++) {
                int gbase_p = ((row0 + r0 + grp + half_p * 8) * num_heads + head) * 128 + c_base + nn_p * 8 + tq * 2;
                {
                    float2 _v2_5 = *reinterpret_cast<const float2*>(dq_f + gbase_p);
                    dqf_pre[nn_p * 4 + half_p * 2] = _v2_5.x;
                    dqf_pre[nn_p * 4 + half_p * 2 + 1] = _v2_5.y;
                }
                {
                    float2 _v2_6 = *reinterpret_cast<const float2*>(dk_f + gbase_p);
                    dkf_pre[nn_p * 4 + half_p * 2] = _v2_6.x;
                    dkf_pre[nn_p * 4 + half_p * 2 + 1] = _v2_6.y;
                }
                {
                    float2 _v2_7 = *reinterpret_cast<const float2*>(dg_f + gbase_p);
                    dgf_pre[nn_p * 4 + half_p * 2] = _v2_7.x;
                    dgf_pre[nn_p * 4 + half_p * 2 + 1] = _v2_7.y;
                }
            }
        }
        float gn_b[4];
        float gmid_b[4];
        float gnl_b[4];
        #pragma unroll
        for (int nn = 0; nn < 4; nn++) {
            int kd_b = c_base + nn * 8 + grp;
            gn_b[nn] = g_s[r0 * 136 + kd_b];
            gmid_b[nn] = g_s[(r0 + 8) * 136 + kd_b];
            gnl_b[nn] = g_s[(r0 + 16 - 1) * 136 + kd_b];
        }
        float at_off[16];
        float at_dg[16];
        at_off[0] = 0.0f;
        at_off[1] = 0.0f;
        at_off[2] = 0.0f;
        at_off[3] = 0.0f;
        at_off[4] = 0.0f;
        at_off[5] = 0.0f;
        at_off[6] = 0.0f;
        at_off[7] = 0.0f;
        at_off[8] = 0.0f;
        at_off[9] = 0.0f;
        at_off[10] = 0.0f;
        at_off[11] = 0.0f;
        at_off[12] = 0.0f;
        at_off[13] = 0.0f;
        at_off[14] = 0.0f;
        at_off[15] = 0.0f;
        at_dg[0] = 0.0f;
        at_dg[1] = 0.0f;
        at_dg[2] = 0.0f;
        at_dg[3] = 0.0f;
        at_dg[4] = 0.0f;
        at_dg[5] = 0.0f;
        at_dg[6] = 0.0f;
        at_dg[7] = 0.0f;
        at_dg[8] = 0.0f;
        at_dg[9] = 0.0f;
        at_dg[10] = 0.0f;
        at_dg[11] = 0.0f;
        at_dg[12] = 0.0f;
        at_dg[13] = 0.0f;
        at_dg[14] = 0.0f;
        at_dg[15] = 0.0f;
        float atq_nx[4];
        float atk_nx[4];
        float g2_nx[8];
        float bt_nx[2];
        #pragma unroll
        for (int wd = 0; wd < 4; wd++) {
            int tt_p = r0 + grp + wd % 2 * 8;
            int ss_p = r0 + tq + wd / 2 * 4;
            atq_nx[wd] = dAqk_s[ss_p * 68 + tt_p];
            atk_nx[wd] = dAkk_s[ss_p * 68 + tt_p];
        }
        #pragma unroll
        for (int nn_1 = 0; nn_1 < 4; nn_1++) {
            int kd_p = c_base + nn_1 * 8 + grp;
            #pragma unroll
            for (int wd_1 = 0; wd_1 < 2; wd_1++) {
                int ss_q = r0 + tq + wd_1 * 4;
                g2_nx[nn_1 * 2 + wd_1] = g_s[ss_q * 136 + kd_p];
            }
        }
        #pragma unroll
        for (int wd_2 = 0; wd_2 < 2; wd_2++) {
            bt_nx[wd_2] = beta_s[r0 + tq + wd_2 * 4];
        }
        float aeq0[4];
        float aek0[4];
        #pragma unroll
        for (int wd_3 = 0; wd_3 < 4; wd_3++) {
            int tt0 = r0 + grp + wd_3 % 2 * 8;
            int ss30 = r0 + tq + wd_3 / 2 * 4;
            float vq0_ = atq_nx[wd_3];
            float vk20_ = atk_nx[wd_3];
            if (ss30 < tt0) {
                vq0_ = 0.0f;
                vk20_ = 0.0f;
            }
            aeq0[wd_3] = vq0_;
            aek0[wd_3] = vk20_;
        }
        unsigned int aeq0r[4];
        unsigned int aek0r[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            aeq0r[_lp] = __float_as_uint(aeq0[_lp + 0]);
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            aek0r[_lp] = __float_as_uint(aek0[_lp + 0]);
        }
        #pragma unroll
        for (int nn_2 = 0; nn_2 < 4; nn_2++) {
            int kd = c_base + nn_2 * 8 + grp;
            float b1d0[2];
            float b2d0[2];
            #pragma unroll
            for (int wd_4 = 0; wd_4 < 2; wd_4++) {
                int ss = r0 + tq + wd_4 * 4;
                float _exp2_0 = approx_exp2(g2_nx[nn_2 * 2 + wd_4] - gmid_b[nn_2]);
                float efd0 = _exp2_0;
                __nv_bfloat16 qd0_b = q_s[ss * 136 + kd];
                float _cvt_f32_0 = __bfloat162float(qd0_b);
                b1d0[wd_4] = _cvt_f32_0 * efd0;
                __nv_bfloat16 kd0_raw = k_s[ss * 136 + kd];
                float _cvt_f32_1 = __bfloat162float(kd0_raw);
                __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_cvt_f32_1 * bt_nx[wd_4]);
                __nv_bfloat16 kbd0_b = _cvt_bf16_0;
                float _cvt_f32_2 = __bfloat162float(kbd0_b);
                b2d0[wd_4] = _cvt_f32_2 * efd0;
            }
            unsigned int b1d0r[2];
            unsigned int b2d0r[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                b1d0r[_lp] = __float_as_uint(b1d0[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                b2d0r[_lp] = __float_as_uint(b2d0[_lp + 0]);
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((at_dg + nn_2 * 4)[0]), "+f"((at_dg + nn_2 * 4)[1]), "+f"((at_dg + nn_2 * 4)[2]), "+f"((at_dg + nn_2 * 4)[3])
                : "r"(aeq0r[0]), "r"(aeq0r[1]), "r"(aeq0r[2]), "r"(aeq0r[3]), "r"(b1d0r[0]), "r"(b1d0r[1]));
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((at_dg + nn_2 * 4)[0]), "+f"((at_dg + nn_2 * 4)[1]), "+f"((at_dg + nn_2 * 4)[2]), "+f"((at_dg + nn_2 * 4)[3])
                : "r"(aek0r[0]), "r"(aek0r[1]), "r"(aek0r[2]), "r"(aek0r[3]), "r"(b2d0r[0]), "r"(b2d0r[1]));
        }
        int s0e = r0 + 8;
        float aeq[4];
        float aek[4];
        #pragma unroll
        for (int wd_5 = 0; wd_5 < 4; wd_5++) {
            int tt = r0 + grp + wd_5 % 2 * 8;
            int ss3 = s0e + tq + wd_5 / 2 * 4;
            float vq_ = dAqk_s[ss3 * 68 + tt];
            float vk2_ = dAkk_s[ss3 * 68 + tt];
            if (ss3 < tt) {
                vq_ = 0.0f;
                vk2_ = 0.0f;
            }
            aeq[wd_5] = vq_;
            aek[wd_5] = vk2_;
        }
        unsigned int aeqr[4];
        unsigned int aekr[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            aeqr[_lp] = __float_as_uint(aeq[_lp + 0]);
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            aekr[_lp] = __float_as_uint(aek[_lp + 0]);
        }
        #pragma unroll
        for (int nn_3 = 0; nn_3 < 4; nn_3++) {
            int kd_1 = c_base + nn_3 * 8 + grp;
            float b1d[2];
            float b2d[2];
            #pragma unroll
            for (int wd_6 = 0; wd_6 < 2; wd_6++) {
                int ss_1 = s0e + tq + wd_6 * 4;
                float _exp2_1 = approx_exp2(g_s[ss_1 * 136 + kd_1] - gmid_b[nn_3]);
                float efd = _exp2_1;
                __nv_bfloat16 _q_s_v4 = q_s[ss_1 * 136 + kd_1];
                float _cvt_f32_3 = __bfloat162float(_q_s_v4);
                b1d[wd_6] = _cvt_f32_3 * efd;
                __nv_bfloat16 _kraw__kb_s_v5 = k_s[ss_1 * 136 + kd_1];
                float _cvt_f32_4 = __bfloat162float(_kraw__kb_s_v5);
                __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(_cvt_f32_4 * beta_s[ss_1]);
                __nv_bfloat16 _kb_s_v5 = _cvt_bf16_1;
                float _cvt_f32_5 = __bfloat162float(_kb_s_v5);
                b2d[wd_6] = _cvt_f32_5 * efd;
            }
            unsigned int b1dr[2];
            unsigned int b2dr[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                b1dr[_lp] = __float_as_uint(b1d[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                b2dr[_lp] = __float_as_uint(b2d[_lp + 0]);
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((at_dg + nn_3 * 4)[0]), "+f"((at_dg + nn_3 * 4)[1]), "+f"((at_dg + nn_3 * 4)[2]), "+f"((at_dg + nn_3 * 4)[3])
                : "r"(aeqr[0]), "r"(aeqr[1]), "r"(aeqr[2]), "r"(aeqr[3]), "r"(b1dr[0]), "r"(b1dr[1]));
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((at_dg + nn_3 * 4)[0]), "+f"((at_dg + nn_3 * 4)[1]), "+f"((at_dg + nn_3 * 4)[2]), "+f"((at_dg + nn_3 * 4)[3])
                : "r"(aekr[0]), "r"(aekr[1]), "r"(aekr[2]), "r"(aekr[3]), "r"(b2dr[0]), "r"(b2dr[1]));
        }
        int jn0 = i + 1;
        if (jn0 > 3) {
            jn0 = 3;
        }
        int s0n0 = jn0 * 16;
        #pragma unroll
        for (int wd_7 = 0; wd_7 < 4; wd_7++) {
            int tt_n0 = r0 + grp + wd_7 % 2 * 8;
            int ss_n0 = s0n0 + tq + wd_7 / 2 * 4;
            atq_nx[wd_7] = dAqk_s[ss_n0 * 68 + tt_n0];
            atk_nx[wd_7] = dAkk_s[ss_n0 * 68 + tt_n0];
        }
        #pragma unroll
        for (int nn_4 = 0; nn_4 < 4; nn_4++) {
            int kd_n0 = c_base + nn_4 * 8 + grp;
            #pragma unroll
            for (int wd_8 = 0; wd_8 < 2; wd_8++) {
                int ss_m0 = s0n0 + tq + wd_8 * 4;
                g2_nx[nn_4 * 2 + wd_8] = g_s[ss_m0 * 136 + kd_n0];
            }
        }
        #pragma unroll
        for (int wd_9 = 0; wd_9 < 2; wd_9++) {
            bt_nx[wd_9] = beta_s[s0n0 + tq + wd_9 * 4];
        }
        #pragma unroll 1
        for (int j = i + 1; j < 4; j++) {
            int s0t = j * 16;
            unsigned int atqr0[4];
            unsigned int atkr0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                atqr0[_lp] = __float_as_uint(atq_nx[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                atkr0[_lp] = __float_as_uint(atk_nx[_lp + 0]);
            }
            #pragma unroll
            for (int nn_5 = 0; nn_5 < 4; nn_5++) {
                int kd_2 = c_base + nn_5 * 8 + grp;
                float b1c[2];
                float b2c[2];
                #pragma unroll
                for (int wd_10 = 0; wd_10 < 2; wd_10++) {
                    int ss_2 = s0t + tq + wd_10 * 4;
                    float _exp2_2 = approx_exp2(g2_nx[nn_5 * 2 + wd_10] - gnl_b[nn_5]);
                    float efc = _exp2_2;
                    __nv_bfloat16 qc_b = q_s[ss_2 * 136 + kd_2];
                    float _cvt_f32_6 = __bfloat162float(qc_b);
                    b1c[wd_10] = _cvt_f32_6 * efc;
                    __nv_bfloat16 kc_raw = k_s[ss_2 * 136 + kd_2];
                    float _cvt_f32_7 = __bfloat162float(kc_raw);
                    __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(_cvt_f32_7 * bt_nx[wd_10]);
                    __nv_bfloat16 kbc_b = _cvt_bf16_2;
                    float _cvt_f32_8 = __bfloat162float(kbc_b);
                    b2c[wd_10] = _cvt_f32_8 * efc;
                }
                unsigned int b1cr[2];
                unsigned int b2cr[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    b1cr[_lp] = __float_as_uint(b1c[_lp + 0]);
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    b2cr[_lp] = __float_as_uint(b2c[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((at_off + nn_5 * 4)[0]), "+f"((at_off + nn_5 * 4)[1]), "+f"((at_off + nn_5 * 4)[2]), "+f"((at_off + nn_5 * 4)[3])
                    : "r"(atqr0[0]), "r"(atqr0[1]), "r"(atqr0[2]), "r"(atqr0[3]), "r"(b1cr[0]), "r"(b1cr[1]));
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((at_off + nn_5 * 4)[0]), "+f"((at_off + nn_5 * 4)[1]), "+f"((at_off + nn_5 * 4)[2]), "+f"((at_off + nn_5 * 4)[3])
                    : "r"(atkr0[0]), "r"(atkr0[1]), "r"(atkr0[2]), "r"(atkr0[3]), "r"(b2cr[0]), "r"(b2cr[1]));
            }
            int jn = j + 1;
            if (jn > 3) {
                jn = 3;
            }
            int s0n = jn * 16;
            #pragma unroll
            for (int wd_11 = 0; wd_11 < 4; wd_11++) {
                int tt_n = r0 + grp + wd_11 % 2 * 8;
                int ss_n = s0n + tq + wd_11 / 2 * 4;
                atq_nx[wd_11] = dAqk_s[ss_n * 68 + tt_n];
                atk_nx[wd_11] = dAkk_s[ss_n * 68 + tt_n];
            }
            #pragma unroll
            for (int nn_6 = 0; nn_6 < 4; nn_6++) {
                int kd_n = c_base + nn_6 * 8 + grp;
                #pragma unroll
                for (int wd_12 = 0; wd_12 < 2; wd_12++) {
                    int ss_m = s0n + tq + wd_12 * 4;
                    g2_nx[nn_6 * 2 + wd_12] = g_s[ss_m * 136 + kd_n];
                }
            }
            #pragma unroll
            for (int wd_13 = 0; wd_13 < 2; wd_13++) {
                bt_nx[wd_13] = beta_s[s0n + tq + wd_13 * 4];
            }
            int s0t1 = j * 16 + 8;
            float atq[4];
            float atk[4];
            #pragma unroll
            for (int wd_14 = 0; wd_14 < 4; wd_14++) {
                int tt_1 = r0 + grp + wd_14 % 2 * 8;
                int ss2 = s0t1 + tq + wd_14 / 2 * 4;
                atq[wd_14] = dAqk_s[ss2 * 68 + tt_1];
                atk[wd_14] = dAkk_s[ss2 * 68 + tt_1];
            }
            unsigned int atqr[4];
            unsigned int atkr[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                atqr[_lp] = __float_as_uint(atq[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                atkr[_lp] = __float_as_uint(atk[_lp + 0]);
            }
            #pragma unroll
            for (int nn_7 = 0; nn_7 < 4; nn_7++) {
                int kd_3 = c_base + nn_7 * 8 + grp;
                float b1[2];
                float b2[2];
                #pragma unroll
                for (int wd_15 = 0; wd_15 < 2; wd_15++) {
                    int ss_3 = s0t1 + tq + wd_15 * 4;
                    float _exp2_3 = approx_exp2(g_s[ss_3 * 136 + kd_3] - gnl_b[nn_7]);
                    float ef = _exp2_3;
                    __nv_bfloat16 _q_s_v2 = q_s[ss_3 * 136 + kd_3];
                    float _cvt_f32_9 = __bfloat162float(_q_s_v2);
                    b1[wd_15] = _cvt_f32_9 * ef;
                    __nv_bfloat16 _kraw__kb_s_v3 = k_s[ss_3 * 136 + kd_3];
                    float _cvt_f32_10 = __bfloat162float(_kraw__kb_s_v3);
                    __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(_cvt_f32_10 * beta_s[ss_3]);
                    __nv_bfloat16 _kb_s_v3 = _cvt_bf16_3;
                    float _cvt_f32_11 = __bfloat162float(_kb_s_v3);
                    b2[wd_15] = _cvt_f32_11 * ef;
                }
                unsigned int b1r[2];
                unsigned int b2r[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    b1r[_lp] = __float_as_uint(b1[_lp + 0]);
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    b2r[_lp] = __float_as_uint(b2[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((at_off + nn_7 * 4)[0]), "+f"((at_off + nn_7 * 4)[1]), "+f"((at_off + nn_7 * 4)[2]), "+f"((at_off + nn_7 * 4)[3])
                    : "r"(atqr[0]), "r"(atqr[1]), "r"(atqr[2]), "r"(atqr[3]), "r"(b1r[0]), "r"(b1r[1]));
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((at_off + nn_7 * 4)[0]), "+f"((at_off + nn_7 * 4)[1]), "+f"((at_off + nn_7 * 4)[2]), "+f"((at_off + nn_7 * 4)[3])
                    : "r"(atkr[0]), "r"(atkr[1]), "r"(atkr[2]), "r"(atkr[3]), "r"(b2r[0]), "r"(b2r[1]));
            }
        }
        float gmid_cf[8];
        float gnl_cf[8];
        #pragma unroll
        for (int nn_8 = 0; nn_8 < 4; nn_8++) {
            #pragma unroll
            for (int e = 0; e < 2; e++) {
                int col_a = c_base + nn_8 * 8 + tq * 2 + e;
                gmid_cf[nn_8 * 2 + e] = g_s[(r0 + 8) * 136 + col_a];
                gnl_cf[nn_8 * 2 + e] = g_s[(r0 + 16 - 1) * 136 + col_a];
            }
        }
        float dkt_pre[16];
        #pragma unroll
        for (int nn_9 = 0; nn_9 < 4; nn_9++) {
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                int t_f = r0 + grp + half * 8;
                int col0_f = c_base + nn_9 * 8 + tq * 2;
                #pragma unroll
                for (int e_1 = 0; e_1 < 2; e_1++) {
                    int col_f = col0_f + e_1;
                    const int reg_f = nn_9 * 4 + half * 2 + e_1;
                    float gtk_f = g_s[t_f * 136 + col_f];
                    float gmid_f = gmid_cf[nn_9 * 2 + e_1];
                    float gnl_f = gnl_cf[nn_9 * 2 + e_1];
                    float _exp2_4 = approx_exp2(gnl_f - gtk_f);
                    float _exp2_5 = approx_exp2(gmid_f - gtk_f);
                    dkt_pre[reg_f] = at_off[reg_f] * _exp2_4 + at_dg[reg_f] * _exp2_5;
                }
            }
        }
        float aq_off[16];
        float ak_off[16];
        float aq_dg[16];
        float ak_dg[16];
        aq_off[0] = 0.0f;
        aq_off[1] = 0.0f;
        aq_off[2] = 0.0f;
        aq_off[3] = 0.0f;
        aq_off[4] = 0.0f;
        aq_off[5] = 0.0f;
        aq_off[6] = 0.0f;
        aq_off[7] = 0.0f;
        aq_off[8] = 0.0f;
        aq_off[9] = 0.0f;
        aq_off[10] = 0.0f;
        aq_off[11] = 0.0f;
        aq_off[12] = 0.0f;
        aq_off[13] = 0.0f;
        aq_off[14] = 0.0f;
        aq_off[15] = 0.0f;
        ak_off[0] = 0.0f;
        ak_off[1] = 0.0f;
        ak_off[2] = 0.0f;
        ak_off[3] = 0.0f;
        ak_off[4] = 0.0f;
        ak_off[5] = 0.0f;
        ak_off[6] = 0.0f;
        ak_off[7] = 0.0f;
        ak_off[8] = 0.0f;
        ak_off[9] = 0.0f;
        ak_off[10] = 0.0f;
        ak_off[11] = 0.0f;
        ak_off[12] = 0.0f;
        ak_off[13] = 0.0f;
        ak_off[14] = 0.0f;
        ak_off[15] = 0.0f;
        aq_dg[0] = 0.0f;
        aq_dg[1] = 0.0f;
        aq_dg[2] = 0.0f;
        aq_dg[3] = 0.0f;
        aq_dg[4] = 0.0f;
        aq_dg[5] = 0.0f;
        aq_dg[6] = 0.0f;
        aq_dg[7] = 0.0f;
        aq_dg[8] = 0.0f;
        aq_dg[9] = 0.0f;
        aq_dg[10] = 0.0f;
        aq_dg[11] = 0.0f;
        aq_dg[12] = 0.0f;
        aq_dg[13] = 0.0f;
        aq_dg[14] = 0.0f;
        aq_dg[15] = 0.0f;
        ak_dg[0] = 0.0f;
        ak_dg[1] = 0.0f;
        ak_dg[2] = 0.0f;
        ak_dg[3] = 0.0f;
        ak_dg[4] = 0.0f;
        ak_dg[5] = 0.0f;
        ak_dg[6] = 0.0f;
        ak_dg[7] = 0.0f;
        ak_dg[8] = 0.0f;
        ak_dg[9] = 0.0f;
        ak_dg[10] = 0.0f;
        ak_dg[11] = 0.0f;
        ak_dg[12] = 0.0f;
        ak_dg[13] = 0.0f;
        ak_dg[14] = 0.0f;
        ak_dg[15] = 0.0f;
        float aq_nx[4];
        float ak_nx[4];
        float g1_nx[8];
        #pragma unroll
        for (int wd_16 = 0; wd_16 < 4; wd_16++) {
            int rr_p = r0 + grp + wd_16 % 2 * 8;
            int cc_p = tq + wd_16 / 2 * 4;
            aq_nx[wd_16] = dAqk_s[rr_p * 68 + cc_p];
            ak_nx[wd_16] = dAkk_s[rr_p * 68 + cc_p];
        }
        #pragma unroll
        for (int nn_10 = 0; nn_10 < 4; nn_10++) {
            int kd_q = c_base + nn_10 * 8 + grp;
            #pragma unroll
            for (int wd_17 = 0; wd_17 < 2; wd_17++) {
                int ss_r = tq + wd_17 * 4;
                g1_nx[nn_10 * 2 + wd_17] = g_s[ss_r * 136 + kd_q];
            }
        }
        #pragma unroll 1
        for (int j_1 = 0; j_1 < i; j_1++) {
            int s0 = j_1 * 16;
            unsigned int aqr0[4];
            unsigned int akr0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                aqr0[_lp] = __float_as_uint(aq_nx[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                akr0[_lp] = __float_as_uint(ak_nx[_lp + 0]);
            }
            #pragma unroll
            for (int nn_11 = 0; nn_11 < 4; nn_11++) {
                int kd_4 = c_base + nn_11 * 8 + grp;
                float bv0[2];
                #pragma unroll
                for (int wd_18 = 0; wd_18 < 2; wd_18++) {
                    int ss_4 = s0 + tq + wd_18 * 4;
                    __nv_bfloat16 k0_b = k_s[ss_4 * 136 + kd_4];
                    float _cvt_f32_12 = __bfloat162float(k0_b);
                    float _exp2_6 = approx_exp2(gn_b[nn_11] - g1_nx[nn_11 * 2 + wd_18]);
                    bv0[wd_18] = _cvt_f32_12 * _exp2_6;
                }
                unsigned int br0[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br0[_lp] = __float_as_uint(bv0[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((aq_off + nn_11 * 4)[0]), "+f"((aq_off + nn_11 * 4)[1]), "+f"((aq_off + nn_11 * 4)[2]), "+f"((aq_off + nn_11 * 4)[3])
                    : "r"(aqr0[0]), "r"(aqr0[1]), "r"(aqr0[2]), "r"(aqr0[3]), "r"(br0[0]), "r"(br0[1]));
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((ak_off + nn_11 * 4)[0]), "+f"((ak_off + nn_11 * 4)[1]), "+f"((ak_off + nn_11 * 4)[2]), "+f"((ak_off + nn_11 * 4)[3])
                    : "r"(akr0[0]), "r"(akr0[1]), "r"(akr0[2]), "r"(akr0[3]), "r"(br0[0]), "r"(br0[1]));
            }
            int s0n1 = (j_1 + 1) * 16;
            #pragma unroll
            for (int wd_19 = 0; wd_19 < 4; wd_19++) {
                int rr_n = r0 + grp + wd_19 % 2 * 8;
                int cc_n = s0n1 + tq + wd_19 / 2 * 4;
                aq_nx[wd_19] = dAqk_s[rr_n * 68 + cc_n];
                ak_nx[wd_19] = dAkk_s[rr_n * 68 + cc_n];
            }
            #pragma unroll
            for (int nn_12 = 0; nn_12 < 4; nn_12++) {
                int kd_m = c_base + nn_12 * 8 + grp;
                #pragma unroll
                for (int wd_20 = 0; wd_20 < 2; wd_20++) {
                    int ss_t = s0n1 + tq + wd_20 * 4;
                    g1_nx[nn_12 * 2 + wd_20] = g_s[ss_t * 136 + kd_m];
                }
            }
            int s01 = j_1 * 16 + 8;
            float aqv[4];
            float akv[4];
            #pragma unroll
            for (int wd_21 = 0; wd_21 < 4; wd_21++) {
                int rr = r0 + grp + wd_21 % 2 * 8;
                int cc = s01 + tq + wd_21 / 2 * 4;
                aqv[wd_21] = dAqk_s[rr * 68 + cc];
                akv[wd_21] = dAkk_s[rr * 68 + cc];
            }
            unsigned int aqr[4];
            unsigned int akr[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                aqr[_lp] = __float_as_uint(aqv[_lp + 0]);
            }
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                akr[_lp] = __float_as_uint(akv[_lp + 0]);
            }
            #pragma unroll
            for (int nn_13 = 0; nn_13 < 4; nn_13++) {
                int kd_5 = c_base + nn_13 * 8 + grp;
                float bv[2];
                #pragma unroll
                for (int wd_22 = 0; wd_22 < 2; wd_22++) {
                    int ss_5 = s01 + tq + wd_22 * 4;
                    __nv_bfloat16 _k_s_v0 = k_s[ss_5 * 136 + kd_5];
                    float _cvt_f32_13 = __bfloat162float(_k_s_v0);
                    float _exp2_7 = approx_exp2(gn_b[nn_13] - g_s[ss_5 * 136 + kd_5]);
                    bv[wd_22] = _cvt_f32_13 * _exp2_7;
                }
                unsigned int br[2];
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    br[_lp] = __float_as_uint(bv[_lp + 0]);
                }
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((aq_off + nn_13 * 4)[0]), "+f"((aq_off + nn_13 * 4)[1]), "+f"((aq_off + nn_13 * 4)[2]), "+f"((aq_off + nn_13 * 4)[3])
                    : "r"(aqr[0]), "r"(aqr[1]), "r"(aqr[2]), "r"(aqr[3]), "r"(br[0]), "r"(br[1]));
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+f"((ak_off + nn_13 * 4)[0]), "+f"((ak_off + nn_13 * 4)[1]), "+f"((ak_off + nn_13 * 4)[2]), "+f"((ak_off + nn_13 * 4)[3])
                    : "r"(akr[0]), "r"(akr[1]), "r"(akr[2]), "r"(akr[3]), "r"(br[0]), "r"(br[1]));
            }
        }
        float aqd0[4];
        float akd0[4];
        #pragma unroll
        for (int wd_23 = 0; wd_23 < 4; wd_23++) {
            int rr0 = r0 + grp + wd_23 % 2 * 8;
            int cc0 = r0 + tq + wd_23 / 2 * 4;
            float va0_ = aq_nx[wd_23];
            float vk0_ = ak_nx[wd_23];
            if (cc0 > rr0) {
                va0_ = 0.0f;
                vk0_ = 0.0f;
            }
            aqd0[wd_23] = va0_;
            akd0[wd_23] = vk0_;
        }
        unsigned int aqd0r[4];
        unsigned int akd0r[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            aqd0r[_lp] = __float_as_uint(aqd0[_lp + 0]);
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            akd0r[_lp] = __float_as_uint(akd0[_lp + 0]);
        }
        #pragma unroll
        for (int nn_14 = 0; nn_14 < 4; nn_14++) {
            int kd_6 = c_base + nn_14 * 8 + grp;
            float bvd0[2];
            #pragma unroll
            for (int wd_24 = 0; wd_24 < 2; wd_24++) {
                int ss_6 = r0 + tq + wd_24 * 4;
                __nv_bfloat16 kd0_b = k_s[ss_6 * 136 + kd_6];
                float _cvt_f32_14 = __bfloat162float(kd0_b);
                float _exp2_8 = approx_exp2(gmid_b[nn_14] - g1_nx[nn_14 * 2 + wd_24]);
                bvd0[wd_24] = _cvt_f32_14 * _exp2_8;
            }
            unsigned int brd0[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                brd0[_lp] = __float_as_uint(bvd0[_lp + 0]);
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((aq_dg + nn_14 * 4)[0]), "+f"((aq_dg + nn_14 * 4)[1]), "+f"((aq_dg + nn_14 * 4)[2]), "+f"((aq_dg + nn_14 * 4)[3])
                : "r"(aqd0r[0]), "r"(aqd0r[1]), "r"(aqd0r[2]), "r"(aqd0r[3]), "r"(brd0[0]), "r"(brd0[1]));
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((ak_dg + nn_14 * 4)[0]), "+f"((ak_dg + nn_14 * 4)[1]), "+f"((ak_dg + nn_14 * 4)[2]), "+f"((ak_dg + nn_14 * 4)[3])
                : "r"(akd0r[0]), "r"(akd0r[1]), "r"(akd0r[2]), "r"(akd0r[3]), "r"(brd0[0]), "r"(brd0[1]));
        }
        int s0d = r0 + 8;
        float aqd[4];
        float akd[4];
        #pragma unroll
        for (int wd_25 = 0; wd_25 < 4; wd_25++) {
            int rr_1 = r0 + grp + wd_25 % 2 * 8;
            int cc_1 = s0d + tq + wd_25 / 2 * 4;
            float va_ = dAqk_s[rr_1 * 68 + cc_1];
            float vk_ = dAkk_s[rr_1 * 68 + cc_1];
            if (cc_1 > rr_1) {
                va_ = 0.0f;
                vk_ = 0.0f;
            }
            aqd[wd_25] = va_;
            akd[wd_25] = vk_;
        }
        unsigned int aqdr[4];
        unsigned int akdr[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            aqdr[_lp] = __float_as_uint(aqd[_lp + 0]);
        }
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            akdr[_lp] = __float_as_uint(akd[_lp + 0]);
        }
        #pragma unroll
        for (int nn_15 = 0; nn_15 < 4; nn_15++) {
            int kd_7 = c_base + nn_15 * 8 + grp;
            float bvd[2];
            #pragma unroll
            for (int wd_26 = 0; wd_26 < 2; wd_26++) {
                int ss_7 = s0d + tq + wd_26 * 4;
                __nv_bfloat16 _k_s_v1 = k_s[ss_7 * 136 + kd_7];
                float _cvt_f32_15 = __bfloat162float(_k_s_v1);
                float _exp2_9 = approx_exp2(gmid_b[nn_15] - g_s[ss_7 * 136 + kd_7]);
                bvd[wd_26] = _cvt_f32_15 * _exp2_9;
            }
            unsigned int brd[2];
            #pragma unroll
            for (int _lp = 0; _lp < 2; _lp++) {
                brd[_lp] = __float_as_uint(bvd[_lp + 0]);
            }
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((aq_dg + nn_15 * 4)[0]), "+f"((aq_dg + nn_15 * 4)[1]), "+f"((aq_dg + nn_15 * 4)[2]), "+f"((aq_dg + nn_15 * 4)[3])
                : "r"(aqdr[0]), "r"(aqdr[1]), "r"(aqdr[2]), "r"(aqdr[3]), "r"(brd[0]), "r"(brd[1]));
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                : "+f"((ak_dg + nn_15 * 4)[0]), "+f"((ak_dg + nn_15 * 4)[1]), "+f"((ak_dg + nn_15 * 4)[2]), "+f"((ak_dg + nn_15 * 4)[3])
                : "r"(akdr[0]), "r"(akdr[1]), "r"(akdr[2]), "r"(akdr[3]), "r"(brd[0]), "r"(brd[1]));
        }
        float gn_ce[8];
        float gmid_ce[8];
        #pragma unroll
        for (int nn_16 = 0; nn_16 < 4; nn_16++) {
            #pragma unroll
            for (int e_2 = 0; e_2 < 2; e_2++) {
                int col_b = c_base + nn_16 * 8 + tq * 2 + e_2;
                gn_ce[nn_16 * 2 + e_2] = g_s[r0 * 136 + col_b];
                gmid_ce[nn_16 * 2 + e_2] = g_s[(r0 + 8) * 136 + col_b];
            }
        }
        #pragma unroll
        for (int nn_17 = 0; nn_17 < 4; nn_17++) {
            #pragma unroll
            for (int half_1 = 0; half_1 < 2; half_1++) {
                int t_loc = r0 + grp + half_1 * 8;
                int grow = row0 + t_loc;
                int gbase = (grow * num_heads + head) * 128;
                int col0 = c_base + nn_17 * 8 + tq * 2;
                float dq_o[2];
                float dk_o[2];
                float dg_o[2];
                float bt_t = beta_s[t_loc];
                #pragma unroll
                for (int e_3 = 0; e_3 < 2; e_3++) {
                    int col = col0 + e_3;
                    const int reg = nn_17 * 4 + half_1 * 2 + e_3;
                    float gtk = g_s[t_loc * 136 + col];
                    float gn_c = gn_ce[nn_17 * 2 + e_3];
                    float gmid_c = gmid_ce[nn_17 * 2 + e_3];
                    float _exp2_10 = approx_exp2(gtk - gn_c);
                    float e_off = _exp2_10;
                    float _exp2_11 = approx_exp2(gtk - gmid_c);
                    float e_dg = _exp2_11;
                    float dq2 = aq_off[reg] * e_off + aq_dg[reg] * e_dg;
                    float dk2 = ak_off[reg] * e_off + ak_dg[reg] * e_dg;
                    __nv_bfloat16 _k_s_v6 = k_s[t_loc * 136 + col];
                    float _cvt_f32_16 = __bfloat162float(_k_s_v6);
                    float kt = _cvt_f32_16;
                    if (half_1 == 0) {
                        db_lo = db_lo + dk2 * kt;
                    } else {
                        db_hi = db_hi + dk2 * kt;
                    }
                    dk2 = dk2 * bt_t;
                    __nv_bfloat16 _q_s_v7 = q_s[t_loc * 136 + col];
                    float _cvt_f32_17 = __bfloat162float(_q_s_v7);
                    float qt = _cvt_f32_17;
                    float dg2 = qt * dq2;
                    float dkt = dkt_pre[reg];
                    dq_o[e_3] = dq2 + dqf_pre[reg];
                    dg_o[e_3] = dg2 + (dk2 - dkt) * kt + dgf_pre[reg];
                    dk_o[e_3] = dk2 + dkf_pre[reg] + dkt;
                    if (half_1 == 0) {
                        dotq_lo = dotq_lo + dq_o[e_3] * qt;
                        dotk_lo = dotk_lo + dk_o[e_3] * kt;
                    } else {
                        dotq_hi = dotq_hi + dq_o[e_3] * qt;
                        dotk_hi = dotk_hi + dk_o[e_3] * kt;
                    }
                }
                #pragma unroll
                for (int e_4 = 0; e_4 < 2; e_4++) {
                    dq_s[t_loc * 136 + col0 + e_4] = dq_o[e_4];
                    dk_s[t_loc * 136 + col0 + e_4] = dk_o[e_4];
                }
                {
                    float2 _v2 = make_float2(dg_o[0 + 0], dg_o[0 + 1]);
                    *reinterpret_cast<float2*>((dg_out + (gbase + col0)) + 0) = _v2;
                }
            }
        }
        if (kbi == 0) {
            int t_lo_p = r0 + grp;
            int t_hi_p = r0 + grp + 8;
            int col_p = c_base + 32 + tq * 2;
            dq_s[t_lo_p * 136 + col_p] = db_lo;
            dq_s[t_lo_p * 136 + col_p + 1] = dotq_lo;
            dq_s[t_hi_p * 136 + col_p] = db_hi;
            dq_s[t_hi_p * 136 + col_p + 1] = dotq_hi;
            dk_s[t_lo_p * 136 + col_p] = dotk_lo;
            dk_s[t_hi_p * 136 + col_p] = dotk_hi;
            __syncwarp();
        }
    }
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, db_lo, 1);
    db_lo = db_lo + _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, db_lo, 2);
    db_lo = db_lo + _shfl_xor_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, db_hi, 1);
    db_hi = db_hi + _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, db_hi, 2);
    db_hi = db_hi + _shfl_xor_3;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, dotq_lo, 1);
    dotq_lo = dotq_lo + _shfl_xor_4;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, dotq_lo, 2);
    dotq_lo = dotq_lo + _shfl_xor_5;
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, dotq_hi, 1);
    dotq_hi = dotq_hi + _shfl_xor_6;
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, dotq_hi, 2);
    dotq_hi = dotq_hi + _shfl_xor_7;
    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, dotk_lo, 1);
    dotk_lo = dotk_lo + _shfl_xor_8;
    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, dotk_lo, 2);
    dotk_lo = dotk_lo + _shfl_xor_9;
    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, dotk_hi, 1);
    dotk_hi = dotk_hi + _shfl_xor_10;
    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, dotk_hi, 2);
    dotk_hi = dotk_hi + _shfl_xor_11;
    __syncthreads();
    if (tq == 0) {
        if (kb_half == 1) {
            beta_s[r0 + grp] = db_lo;
            beta_s[r0 + grp + 8] = db_hi;
            dotq_s[r0 + grp] = dotq_lo;
            dotq_s[r0 + grp + 8] = dotq_hi;
            dotk_s[r0 + grp] = dotk_lo;
            dotk_s[r0 + grp + 8] = dotk_hi;
        }
    }
    __syncthreads();
    if (tq == 0) {
        if (kb_half == 0) {
            int idx_lo = (row0 + r0 + grp) * num_heads + head;
            int idx_hi = (row0 + r0 + grp + 8) * num_heads + head;
            db_out[idx_lo] = db_f[idx_lo] + (db_lo + beta_s[r0 + grp]);
            db_out[idx_hi] = db_f[idx_hi] + (db_hi + beta_s[r0 + grp + 8]);
            dotq_s[r0 + grp] = dotq_lo + dotq_s[r0 + grp];
            dotq_s[r0 + grp + 8] = dotq_hi + dotq_s[r0 + grp + 8];
            dotk_s[r0 + grp] = dotk_lo + dotk_s[r0 + grp];
            dotk_s[r0 + grp + 8] = dotk_hi + dotk_s[r0 + grp + 8];
        }
    }
    __syncthreads();
    int d0 = lane * 4;
    #pragma unroll
    for (int it = 0; it < 8; it++) {
        int t = it * 8 + warp;
        long long erow = ext0 + (long long)t;
        float dyq[4];
        float dyk[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&dyq[0])), "=r"(*reinterpret_cast<uint32_t*>(&dyq[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dyq[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dyq[(0) + 3]))
            : "r"(dq_s_addr + (unsigned int)((t * 136 + d0) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&dyk[0])), "=r"(*reinterpret_cast<uint32_t*>(&dyk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&dyk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&dyk[(0) + 3]))
            : "r"(dk_s_addr + (unsigned int)((t * 136 + d0) * 4)));
        float yq[4];
        float yk[4];
        #pragma unroll
        for (int e_5 = 0; e_5 < 4; e_5++) {
            __nv_bfloat16 _yq_b = q_s[t * 136 + d0 + e_5];
            __nv_bfloat16 _yk_b = k_s[t * 136 + d0 + e_5];
            float yq_e = 0.0f;
            float yk_e = 0.0f;
            if (t < clen) {
                float _cvt_f32_18 = __bfloat162float(_yq_b);
                yq_e = _cvt_f32_18;
                float _cvt_f32_19 = __bfloat162float(_yk_b);
                yk_e = _cvt_f32_19;
            }
            yq[e_5] = yq_e;
            yk[e_5] = yk_e;
        }
        float dot_q = dotq_s[t];
        float dot_k = dotk_s[t];
        float rq = rq_s[t];
        float rk = rk_s[t];
        float oq[4];
        float ok[4];
        #pragma unroll
        for (int e_6 = 0; e_6 < 4; e_6++) {
            oq[e_6] = dyq[e_6] * rq - dot_q * yq[e_6] * rq;
            ok[e_6] = dyk[e_6] * rk - dot_k * yk[e_6] * rk;
        }
        long long qindex = (erow * (long long)num_qk_heads + (long long)qk_head) * 128 + (long long)d0;
        if (t < clen) {
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(oq[0 + 0], oq[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(oq[0 + 2], oq[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(dq_ext + qindex))[0]) = _pk2;
            }
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(ok[0 + 0], ok[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(ok[0 + 2], ok[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(dk_ext + qindex))[0]) = _pk2;
            }
        }
    }
}

} // extern "C"
