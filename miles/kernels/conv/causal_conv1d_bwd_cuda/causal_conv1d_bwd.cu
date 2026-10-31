/******************************************************************************
 * Backward of y = silu(causal_conv1d(x, weight)) per packed sequence: depthwise, width 4, channels-last bf16
 * activations, fp32 or bf16 weight. dx and per-block dweight partials (summed by the caller in a fixed order).
 *
 * Tiles are sequence-aligned: the tile list (fla's chunk_indices, 64 rows) cuts every sequence on its own, so no
 * tile spans a boundary. Rows of another sequence (halos, the padding of a short last tile) are loaded as zero,
 * and the math is the plain no-boundary convolution: no per-row sequence ids or compares, unlike the seq_idx
 * path this replaces.
 *
 * Adapted from the channel-last backward of causal-conv1d (csrc/causal_conv1d_bwd.cu,
 * https://github.com/Dao-AILab/causal-conv1d)
 ******************************************************************************/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

using input_t = at::BFloat16;
using vec_t = uint4;  // 16-byte vector: 8 bf16 channels

constexpr int kNThreads = 128;
constexpr int kWidth = 4;
constexpr int kChunkSizeL = 64;  // tile rows; fla's chunk_indices are built for 64
constexpr int kNElts = 8;        // channels per vector load
constexpr int kChunkSizeC = 64;  // channels per block: one 128-byte cache line per row
// Load role: 8 threads move one row of the block's 64 channels, 16 rows per step, 4 steps per tile.
constexpr int kNThreadsPerC = kChunkSizeC / kNElts;
constexpr int kLPerLoad = kNThreads / kNThreadsPerC;
constexpr int kNLoads = kChunkSizeL / kLPerLoad;
// Compute role: each thread owns one channel and kLPerThread consecutive rows; 2 threads share a channel.
constexpr int kLPerThread = kChunkSizeL * kChunkSizeC / kNThreads;
constexpr int kNThreadsPerRow = kChunkSizeL / kLPerThread;
static_assert(kNLoads * kLPerLoad == kChunkSizeL);
static_assert(kNThreadsPerRow == 2, "pair_sum assumes two threads per channel");

struct Params {
    int dim;
    int n_tiles;
    int64_t x_l_stride, dout_l_stride, dx_l_stride;
    int64_t weight_c_stride, weight_width_stride;
    const input_t* x;
    const input_t* dout;
    input_t* dx;
    const void* weight;
    const int* tiles;       // [n_tiles, 2]: (sequence, piece of that sequence)
    const int* cu_seqlens;  // [num_sequences + 1]
    float* dweight_partials;  // [gridDim.y, dim, kWidth]
};

__device__ inline float pair_sum(float v) { return v + __shfl_xor_sync(0xffffffffu, v, 1); }

template <typename weight_t>
__global__ __launch_bounds__(kNThreads) void causal_conv1d_bwd_kernel(Params params) {
    // x_smem: 3-row left halo | 64 tile rows | 3-row right halo; dout_smem: 64 tile rows | 3-row right halo.
    // The extra kNElts columns pad each row against bank conflicts in the transposed reads.
    __shared__ input_t dout_smem[kChunkSizeL + kWidth - 1][kChunkSizeC + kNElts];
    __shared__ input_t x_smem[kWidth - 1 + kChunkSizeL + kWidth - 1][kChunkSizeC + kNElts];

    const int chunk_c_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int l_idx = tid / kNThreadsPerC;
    const int c_idx = tid % kNThreadsPerC;
    const bool c_load_ok = chunk_c_id * kChunkSizeC + c_idx * kNElts < params.dim;
    const input_t* x = params.x + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    const input_t* dout = params.dout + chunk_c_id * kChunkSizeC + c_idx * kNElts;
    input_t* dx = params.dx + chunk_c_id * kChunkSizeC + c_idx * kNElts;

    const int row_idx = tid / kNThreadsPerRow;
    const int col_idx = tid % kNThreadsPerRow;
    const bool c_row_ok = chunk_c_id * kChunkSizeC + row_idx < params.dim;
    const weight_t* weight =
        reinterpret_cast<const weight_t*>(params.weight) + (chunk_c_id * kChunkSizeC + row_idx) * params.weight_c_stride;
    float weight_vals[kWidth] = {0};
    if (c_row_ok) {
#pragma unroll
        for (int w = 0; w < kWidth; ++w) { weight_vals[w] = float(weight[w * params.weight_width_stride]); }
    }
    float dweight_acc[kWidth] = {0};

    // Grid-stride over the tile list: block y takes tiles y, y + gridDim.y, ...
    for (int it = blockIdx.y; it < params.n_tiles; it += gridDim.y) {
        const int seq = params.tiles[2 * it], piece = params.tiles[2 * it + 1];
        const int bos = params.cu_seqlens[seq], eos = params.cu_seqlens[seq + 1];
        const int t0 = bos + piece * kChunkSizeL;
        const int t_end = min(t0 + kChunkSizeL, eos);

        // Rows outside [bos, eos) load as zero; that is all the boundary handling the math below needs.
        // All of the tile's global loads are issued before its first shared-memory store, so they are in flight
        // together; storing each row as it arrives serializes them (one memory round trip per row step).
        vec_t dout_rows[kNLoads], x_rows[kNLoads];
#pragma unroll
        for (int l = 0; l < kNLoads; ++l) {
            const int p = t0 + l * kLPerLoad + l_idx;
            const bool in_tile = p < t_end && c_load_ok;
            dout_rows[l] = in_tile ? *reinterpret_cast<const vec_t*>(dout + p * params.dout_l_stride) : vec_t{0, 0, 0, 0};
            x_rows[l] = in_tile ? *reinterpret_cast<const vec_t*>(x + p * params.x_l_stride) : vec_t{0, 0, 0, 0};
        }
#pragma unroll
        for (int l = 0; l < kNLoads; ++l) {
            reinterpret_cast<vec_t*>(dout_smem[l * kLPerLoad + l_idx])[c_idx] = dout_rows[l];
            reinterpret_cast<vec_t*>(x_smem[kWidth - 1 + l * kLPerLoad + l_idx])[c_idx] = x_rows[l];
        }
        if (l_idx < kWidth - 1) {
            input_t dout_load[kNElts] = {0};
            input_t x_left_load[kNElts] = {0};
            input_t x_right_load[kNElts] = {0};
            const int p_right = t0 + kChunkSizeL + l_idx;
            const int p_left = t0 + l_idx - (kWidth - 1);
            if (p_right < eos && c_load_ok) {
                reinterpret_cast<vec_t*>(dout_load)[0] = *reinterpret_cast<const vec_t*>(dout + p_right * params.dout_l_stride);
                reinterpret_cast<vec_t*>(x_right_load)[0] = *reinterpret_cast<const vec_t*>(x + p_right * params.x_l_stride);
            }
            if (p_left >= bos && c_load_ok) {
                reinterpret_cast<vec_t*>(x_left_load)[0] = *reinterpret_cast<const vec_t*>(x + p_left * params.x_l_stride);
            }
            reinterpret_cast<vec_t*>(dout_smem[kChunkSizeL + l_idx])[c_idx] = reinterpret_cast<vec_t*>(dout_load)[0];
            reinterpret_cast<vec_t*>(x_smem[l_idx])[c_idx] = reinterpret_cast<vec_t*>(x_left_load)[0];
            reinterpret_cast<vec_t*>(x_smem[kWidth - 1 + kChunkSizeL + l_idx])[c_idx] =
                reinterpret_cast<vec_t*>(x_right_load)[0];
        }
        __syncthreads();

        float dout_vals[kLPerThread + kWidth - 1];
        float x_vals[kWidth - 1 + kLPerThread + kWidth - 1];
#pragma unroll
        for (int i = 0; i < kWidth - 1 + kLPerThread; ++i) {
            dout_vals[i] = float(dout_smem[col_idx * kLPerThread + i][row_idx]);
            x_vals[i] = float(x_smem[col_idx * kLPerThread + i][row_idx]);
        }
#pragma unroll
        for (int i = kWidth - 1 + kLPerThread; i < kWidth - 1 + kLPerThread + kWidth - 1; ++i) {
            x_vals[i] = float(x_smem[col_idx * kLPerThread + i][row_idx]);
        }
        // Recompute the conv output and take dout through SiLU's derivative.
#pragma unroll
        for (int i = 0; i < kLPerThread + kWidth - 1; ++i) {
            float out_val = 0.f;
#pragma unroll
            for (int w = 0; w < kWidth; ++w) { out_val += weight_vals[w] * x_vals[i + w]; }
            const float out_sigmoid = 1.f / (1.f + expf(-out_val));
            dout_vals[i] *= out_sigmoid * (1 + out_val * (1 - out_sigmoid));
        }

#pragma unroll
        for (int w = 0; w < kWidth; ++w) {
            float dweight_val = 0.f;
#pragma unroll
            for (int i = 0; i < kLPerThread; ++i) { dweight_val += x_vals[i + w] * dout_vals[i]; }
            dweight_acc[w] += pair_sum(dweight_val);
        }

        float dx_vals[kLPerThread] = {0};
#pragma unroll
        for (int i = 0; i < kLPerThread; ++i) {
#pragma unroll
            for (int w = 0; w < kWidth; ++w) { dx_vals[i] += weight_vals[kWidth - 1 - w] * dout_vals[i + w]; }
        }

        // dx goes back through x_smem so the global store is in the load role's coalesced layout.
        __syncthreads();
#pragma unroll
        for (int i = 0; i < kLPerThread; ++i) { x_smem[kWidth - 1 + col_idx * kLPerThread + i][row_idx] = dx_vals[i]; }
        __syncthreads();
#pragma unroll
        for (int l = 0; l < kNLoads; ++l) {
            const int p = t0 + l * kLPerLoad + l_idx;
            if (p < t_end && c_load_ok) {
                *reinterpret_cast<vec_t*>(dx + p * params.dx_l_stride) =
                    reinterpret_cast<vec_t*>(x_smem[kWidth - 1 + l * kLPerLoad + l_idx])[c_idx];
            }
        }
        __syncthreads();  // shared memory is refilled by the next tile
    }

    // One partial per (block y, channel); every slot is written, so the buffer needs no zeroing.
    if (col_idx == 0 && c_row_ok) {
        float* out = params.dweight_partials
            + (int64_t(blockIdx.y) * params.dim + chunk_c_id * kChunkSizeC + row_idx) * kWidth;
#pragma unroll
        for (int w = 0; w < kWidth; ++w) { out[w] = dweight_acc[w]; }
    }
}

bool vector_aligned(const at::Tensor& t) {
    return t.stride(2) == 1 && t.stride(1) % kNElts == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

void check_rows(const at::Tensor& t, const at::Tensor& x, const char* name) {
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, name, " must be a bf16 CUDA tensor");
    TORCH_CHECK(t.sizes() == x.sizes(), name, " must have x's shape");
    TORCH_CHECK(vector_aligned(t), name, " needs contiguous channels and 16-byte aligned rows");
}

}  // namespace

void causal_conv1d_bwd(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& dout,
                       const at::Tensor& cu_seqlens, const at::Tensor& chunk_indices, int64_t tile_rows,
                       at::Tensor& dx, at::Tensor& dweight_partials) {
    // chunk_indices are built by the caller; tiles of another height would be read at the wrong rows.
    TORCH_CHECK(tile_rows == kChunkSizeL, "chunk_indices must be built for ", kChunkSizeL, "-row tiles, got ",
                tile_rows);
    TORCH_CHECK(x.dim() == 3 && x.size(0) == 1, "x must be [1, T, C]");
    const int dim = static_cast<int>(x.size(2));
    TORCH_CHECK(dim % kNElts == 0, "channels must be a multiple of 8");
    check_rows(x, x, "x");
    check_rows(dout, x, "dout");
    check_rows(dx, x, "dx");
    TORCH_CHECK(weight.is_cuda() && weight.dim() == 2 && weight.size(0) == dim && weight.size(1) == kWidth,
                "weight must be a [C, 4] CUDA tensor");
    TORCH_CHECK(weight.scalar_type() == at::kFloat || weight.scalar_type() == at::kBFloat16,
                "weight must be fp32 or bf16");
    TORCH_CHECK(cu_seqlens.is_cuda() && cu_seqlens.scalar_type() == at::kInt && cu_seqlens.dim() == 1
                    && cu_seqlens.is_contiguous(), "cu_seqlens must be a contiguous int32 CUDA vector");
    TORCH_CHECK(chunk_indices.is_cuda() && chunk_indices.scalar_type() == at::kInt && chunk_indices.dim() == 2
                    && chunk_indices.size(1) == 2 && chunk_indices.is_contiguous(),
                "chunk_indices must be a contiguous int32 [num_tiles, 2] CUDA tensor");
    TORCH_CHECK(dweight_partials.scalar_type() == at::kFloat && dweight_partials.is_contiguous()
                    && dweight_partials.dim() == 3 && dweight_partials.size(0) >= 1 && dweight_partials.size(1) == dim
                    && dweight_partials.size(2) == kWidth, "dweight_partials must be a contiguous fp32 [k, C, 4]");

    Params params;
    params.dim = dim;
    params.n_tiles = static_cast<int>(chunk_indices.size(0));
    params.x_l_stride = x.stride(1);
    params.dout_l_stride = dout.stride(1);
    params.dx_l_stride = dx.stride(1);
    params.weight_c_stride = weight.stride(0);
    params.weight_width_stride = weight.stride(1);
    params.x = reinterpret_cast<const input_t*>(x.data_ptr());
    params.dout = reinterpret_cast<const input_t*>(dout.data_ptr());
    params.dx = reinterpret_cast<input_t*>(dx.data_ptr());
    params.weight = weight.data_ptr();
    params.tiles = chunk_indices.data_ptr<int>();
    params.cu_seqlens = cu_seqlens.data_ptr<int>();
    params.dweight_partials = dweight_partials.data_ptr<float>();

    const at::cuda::CUDAGuard device_guard(x.device());
    const dim3 grid((dim + kChunkSizeC - 1) / kChunkSizeC, static_cast<unsigned>(dweight_partials.size(0)));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (weight.scalar_type() == at::kFloat) {
        causal_conv1d_bwd_kernel<float><<<grid, kNThreads, 0, stream>>>(params);
    } else {
        causal_conv1d_bwd_kernel<at::BFloat16><<<grid, kNThreads, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("causal_conv1d_bwd", &causal_conv1d_bwd,
          "Sequence-aligned backward of silu(causal_conv1d(x, weight)): dx and per-block dweight partials");
}
