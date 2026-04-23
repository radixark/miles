"""Tilelang FP8 activation quantization used by the compressor and indexer QAT paths."""

import tilelang
import tilelang.language as T
import torch

tilelang.set_log_level("WARNING")

_pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"


def _fast_log2_ceil(x):
    bits_x = T.reinterpret("uint32", x)
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def _fast_pow2(x):
    bits_x = (x + 127) << 23
    return T.reinterpret("float32", bits_x)


def _fast_round_scale(amax, fp8_max_inv):
    return _fast_pow2(_fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(pass_configs=_pass_configs)
def act_quant_kernel(N, block_size=128, in_dtype=BF16, out_dtype=FP8, scale_dtype=FP32, round_scale=False):
    M = T.symbolic("M")
    fp8_min = -448.0
    fp8_max = 448.0
    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale else 2
    blk_m = 32
    group_size = block_size

    @T.prim_func
    def act_quant_kernel_(
        X: T.Tensor[(M, N), in_dtype],
        Y: T.Tensor[(M, N), out_dtype],
        S: T.Tensor[(M, T.ceildiv(N, group_size)), scale_dtype],
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
            pid_m,
            pid_n,
        ):
            x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
            x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
            amax_local = T.alloc_fragment((blk_m,), scale_dtype)
            s_local = T.alloc_fragment((blk_m,), scale_dtype)
            y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
            y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

            for _ in T.Pipelined(1, num_stages=num_stages):
                T.copy(X[pid_m * blk_m, pid_n * group_size], x_shared)
                T.copy(x_shared, x_local)
                T.reduce_absmax(x_local, amax_local, dim=1)
                for i in T.Parallel(blk_m):
                    amax_local[i] = T.max(amax_local[i], 1e-4)
                    if round_scale:
                        s_local[i] = _fast_round_scale(amax_local[i], fp8_max_inv)
                    else:
                        s_local[i] = amax_local[i] * fp8_max_inv
                for i, j in T.Parallel(blk_m, group_size):
                    y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)
                for i in T.Parallel(blk_m):
                    S[pid_m * blk_m + i, pid_n] = s_local[i]
                T.copy(y_local, y_shared)
                T.copy(y_shared, Y[pid_m * blk_m, pid_n * group_size])

    return act_quant_kernel_


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert (
        x.size(-1) % block_size == 0
    ), f"Last dimension size must be divisible by block_size (block_size={block_size})"
    N = x.size(-1)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], N // block_size, dtype=torch.float32)
    kernel = act_quant_kernel(N, block_size, round_scale=scale_fmt is not None)
    kernel(x.view(-1, N), y.view(-1, N), s.view(-1, N // block_size))
    return y, s
