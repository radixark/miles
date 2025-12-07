# EVOLVE-BLOCK-START
"""Constructor-based function for C₂ autoconvolution lower-bound optimization"""
import numpy as np
from typing import Tuple
from tqdm import trange


def _simpson_l2sq(conv: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute ||f*f||_2^2 via Simpson-like piecewise-linear rule with endpoint zeros,
    and return its gradient w.r.t conv (same length as conv).

    l2_sq = sum_{i=0..M} (dx/3) * (y_i^2 + y_i*y_{i+1} + y_{i+1}^2),
    where y = [0, conv, 0], dx = 1/(M+1).

    d l2_sq / d y_j = (dx/3) * (4*y_j + y_{j-1} + y_{j+1})
    => restrict to indices j=1..M (these correspond to conv entries)
    """
    M = conv.size
    if M == 0:
        return 0.0, np.zeros_like(conv)

    dx = 1.0 / (M + 1)

    # pad endpoints with zeros
    y = np.empty(M + 2, dtype=conv.dtype)
    y[0] = 0.0
    y[1:-1] = conv
    y[-1] = 0.0

    # l2 value
    lhs = y[:-1]
    rhs = y[1:]
    l2_sq = (dx / 3.0) * np.sum(lhs * lhs + lhs * rhs + rhs * rhs)

    # gradient wrt conv (positions 1..M of y)
    # d/d y_j : (dx/3)*(4*y_j + y_{j-1} + y_{j+1})
    grad_y = (dx / 3.0) * (4.0 * y + np.roll(y, 1) + np.roll(y, -1))
    grad_conv = grad_y[1:-1]  # strip padding

    return float(l2_sq), grad_conv


def _l1(conv: np.ndarray) -> Tuple[float, np.ndarray]:
    """ ||f*f||_1 = dx * sum(conv); gradient is dx * ones """
    M = conv.size
    dx = 1.0 / (M + 1) if M > 0 else 1.0
    val = dx * float(np.sum(conv)) if M > 0 else 0.0
    grad = np.full_like(conv, dx)
    return val, grad


def _linf(conv: np.ndarray) -> Tuple[float, np.ndarray]:
    """ ||f*f||_inf = max(conv); subgradient: uniform over argmax set """
    if conv.size == 0:
        return 0.0, np.zeros_like(conv)
    m = float(np.max(conv))
    mask = (conv == m)
    count = int(mask.sum())
    if count == 0 or m <= 0.0:
        return m, np.zeros_like(conv)
    grad = mask.astype(conv.dtype) / count
    return m, grad


def _objective_and_grad_conv(conv: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute C = l2_sq / (l1 * linf) and gradient dC/d(conv) using quotient rule.
    """
    l2_sq, g_l2 = _simpson_l2sq(conv)
    l1, g_l1 = _l1(conv)
    linf, g_linf = _linf(conv)

    if l1 <= 0.0 or linf <= 0.0:
        return 0.0, np.zeros_like(conv)

    denom = l1 * linf
    C = l2_sq / denom

    # dC = (g_l2 * denom - l2_sq * (g_l1 * linf + l1 * g_linf)) / denom^2
    num_grad = g_l2 * denom - l2_sq * (g_l1 * linf + l1 * g_linf)
    g_conv = num_grad / (denom * denom)

    return float(C), g_conv


def _grad_h_from_conv_grad(h: np.ndarray, g_conv: np.ndarray) -> np.ndarray:
    """
    Given dC/d(conv) and conv = h * h (full convolution),
    dC/dh = 2 * (g_conv convolved with reverse(h)) in 'valid' mode (length N).
    """
    h_rev = h[::-1]
    # length(g_conv)=2N-1, length(h_rev)=N; 'valid' output length is N
    g_h = np.convolve(g_conv, h_rev, mode="valid")
    return 2.0 * g_h


class _Adam:
    """Lightweight Adam optimizer for numpy arrays (per-candidate)."""
    def __init__(self, shape, lr=3e-2, beta1=0.9, beta2=0.999, eps=1e-8, dtype=np.float32):
        self.m = np.zeros(shape, dtype=dtype)
        self.v = np.zeros(shape, dtype=dtype)
        self.t = 0
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.dtype = dtype

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad * grad)
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset_like(self, params):
        self.m[...] = 0.0
        self.v[...] = 0.0
        self.t = 0


def _batch_objective(h_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation over a batch of candidates.
    Returns:
        C_vals: (B,) objective values
        conv_grads: list/array of per-candidate dC/d(conv) for backprop
    """
    B, N = h_batch.shape
    C_vals = np.zeros(B, dtype=np.float32)
    conv_grads = [None] * B
    for b in range(B):
        h = np.clip(h_batch[b], 0.0, None)
        conv = np.convolve(h, h, mode="full")
        Cb, g_conv = _objective_and_grad_conv(conv)
        C_vals[b] = Cb
        conv_grads[b] = g_conv
    return C_vals, conv_grads


def _phase_update(h_batch, opt_list, lr, add_noise=False, t=0, eta=1e-3, gamma=0.55):
    """
    One optimization step for the whole batch.
    """
    B, N = h_batch.shape
    # compute dC/dh for each candidate
    C_vals, conv_grads = _batch_objective(h_batch)
    grads = np.zeros_like(h_batch, dtype=h_batch.dtype)
    for b in range(B):
        g_h = _grad_h_from_conv_grad(np.clip(h_batch[b], 0.0, None), conv_grads[b])
        grads[b] = g_h

    if add_noise:
        sigma = eta / ((t + 1) ** gamma)
        grads = grads + sigma * np.random.normal(size=grads.shape).astype(grads.dtype)

    # apply Adam update + project to nonnegativity
    for b in range(B):
        opt = opt_list[b]
        opt.lr = lr
        h_new = opt.step(h_batch[b], grads[b].astype(h_batch.dtype))
        h_batch[b] = np.clip(h_new, 0.0, None)

    return h_batch, C_vals


def _elitist_respawn(h_batch, C_vals, keep_frac, init_sampler, opt_list):
    """
    Keep top frac, respawn the rest with fresh random samples, reset their Adam states.
    """
    B = h_batch.shape[0]
    K = max(1, int(B * keep_frac))
    idx = np.argsort(C_vals)[-K:]  # top K
    survivors = h_batch[idx].copy()

    fresh = init_sampler(B - K)
    new_batch = np.concatenate([survivors, fresh], axis=0)

    # reorder optimizers to match new batch; reset the respawned ones
    new_opts = []
    for _ in range(K):
        new_opts.append(opt_list[idx[_]])  # keep state for survivors
    for _ in range(B - K):
        opt = _Adam(shape=h_batch.shape[1:], lr=opt_list[0].lr, dtype=h_batch.dtype)
        new_opts.append(opt)

    return new_batch, new_opts


def _upsample_1d(h: np.ndarray) -> np.ndarray:
    """Linear ×2 upsampling on [-1/4,1/4] grid (index-space等距采样即可)."""
    N = h.shape[0]
    x_old = np.linspace(-0.5, 0.5, N)
    x_new = np.linspace(-0.5, 0.5, 2 * N)
    return np.interp(x_new, x_old, h)


def _single_candidate_finetune(h0: np.ndarray, lr=3e-3, steps=50_000, log_every=2_000) -> Tuple[np.ndarray, float]:
    """
    Pure exploitation (no noise) on a single vector with Adam + projection.
    """
    h = h0.astype(np.float32).copy()
    opt = _Adam(h.shape, lr=lr, dtype=h.dtype)
    last_C = 0.0
    for t in trange(steps, desc="upsampled-finetune", leave=False):
        # objective and gradient
        conv = np.convolve(np.clip(h, 0.0, None), np.clip(h, 0.0, None), mode="full")
        C, g_conv = _objective_and_grad_conv(conv)
        g_h = _grad_h_from_conv_grad(np.clip(h, 0.0, None), g_conv)

        h = np.clip(opt.step(h, g_h.astype(h.dtype)), 0.0, None)
        last_C = C
        if log_every and (t + 1) % log_every == 0:
            pass  # tqdm显示即可
    return h, float(last_C)


def construct_function():
    """
    Use the paper's 4-phase gradient-based search to maximize
        R(f) = ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf).
    Returns (heights, r_value).
    """
    # ---------- Hyperparameters (close to paper, but conservatively smaller by default) ----------
    N = 256                # search resolution (paper: 768; will truncate zeros automatically)
    B = 64                 # batch size (paper used batch; you can raise this)
    ITER = 10_000          # total iterations
    EXPLORE_STEPS = 30_000 # phase split (paper: 30k)
    DROP_EVERY = 10_000    # respawn period (paper: 20k)
    KEEP_FRAC = 0.5        # keep top fraction (paper: kappa%)
    LR_EXPLORE = 3e-2      # Adam lr for exploration (paper)
    LR_EXPLOIT = 5e-3      # Adam lr for exploitation (paper)
    ETA, GAMMA = 1e-3, 0.55  # gradient noise schedule (paper used ~0.65; 0.55 is slightly gentler)
    dtype = np.float32

    rng = np.random.default_rng()

    def init_sampler(m):
        return rng.uniform(0.0, 1.0, size=(m, N)).astype(dtype)

    # init population and per-candidate Adam
    h_batch = init_sampler(B)
    opt_list = [_Adam(shape=(N,), lr=LR_EXPLORE, dtype=dtype) for _ in range(B)]
    best_h = h_batch.copy()
    best_C = np.full(B, -np.inf, dtype=dtype)

    for t in trange(ITER, desc="optimizing", leave=False):
        if t < EXPLORE_STEPS:
            # exploration phase: higher LR + noise
            h_batch, C_vals = _phase_update(
                h_batch, opt_list, lr=LR_EXPLORE, add_noise=True, t=t, eta=ETA, gamma=GAMMA
            )
        else:
            # exploitation phase: lower LR, no noise
            h_batch, C_vals = _phase_update(
                h_batch, opt_list, lr=LR_EXPLOIT, add_noise=False, t=t, eta=ETA, gamma=GAMMA
            )

        # update per-candidate bests
        improved = C_vals > best_C
        best_C = np.where(improved, C_vals, best_C)
        best_h[improved] = h_batch[improved]

        # periodic elitist respawn
        if (t + 1) % DROP_EVERY == 0:
            h_batch, opt_list = _elitist_respawn(
                h_batch, C_vals, keep_frac=KEEP_FRAC, init_sampler=init_sampler, opt_list=opt_list
            )

    # pick the best candidate
    idx = int(np.argmax(best_C))
    h_star = np.clip(best_h[idx].astype(np.float32), 0.0, None)

    # ---------- Phase 4: Upsampling + exploitation ----------
    # 2× upsample then fine-tune；可再做一次得到 4×（和论文一致）
    h_up1 = _upsample_1d(h_star)
    h_up1, _ = _single_candidate_finetune(h_up1, lr=3e-3, steps=40_000, log_every=2_000)

    h_up2 = _upsample_1d(h_up1)
    h_up2, _ = _single_candidate_finetune(h_up2, lr=3e-3, steps=40_000, log_every=2_000)

    heights = np.clip(h_up2, 0.0, None)
    r_value = compute_c2_lower_bound(heights)
    return heights, r_value


def compute_c2_lower_bound(heights: np.ndarray) -> float:
    """
    Compute the C₂ lower bound:
        R(f) = ||f*f||_2^2 / ( ||f*f||_1 * ||f*f||_inf )

    Implementation matches the paper's verification logic:
      * full autoconvolution
      * piecewise-linear (Simpson-like) integral with zero padding for ||.||_2^2
      * L1 as dx * sum(conv), dx = 1/(M+1)
      * Linf as max(conv)
    """
    conv = np.convolve(heights, heights, mode="full")
    if conv.size == 0:
        return 0.0

    # L2 norm squared via piecewise-linear rule with endpoint zeros
    M = len(conv)
    dx = 1.0 / (M + 1)
    y = np.empty(M + 2, dtype=conv.dtype)
    y[0] = 0.0
    y[1:-1] = conv
    y[-1] = 0.0
    l2_sq = (dx / 3.0) * np.sum(y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2)

    # L1 and Linf norms
    l1 = dx * float(np.sum(conv))
    linf = float(np.max(conv))

    if l1 <= 0.0 or linf <= 0.0:
        return 0.0

    return float(l2_sq) / (l1 * linf)
# EVOLVE-BLOCK-END



def run_c2_optimization():
    """Main entry point for C₂ (second autocorrelation) lower-bound search"""
    heights, r_value = construct_function()

    # Save results in accepted format for the C2 verifier (key: 'heights')
    current_solution = {'data': heights.tolist()}
    save_search_results(
        best_perfect_solution=None,
        current_solution=current_solution,
        target_value=TARGET_VALUE
    )

    return heights, r_value


if __name__ == "__main__":

    ######## get parameters from config ########
    from openevolve.modular_utils.file_io_controller import save_search_results
    from openevolve.modular_utils.evaluation_controller import get_current_problem_config
    PROBLEM_CONFIG = get_current_problem_config()
    TARGET_VALUE = PROBLEM_CONFIG['core_parameters']['target_value']
    PROBLEM_TYPE = PROBLEM_CONFIG['problem_type']
    ###############################################

    heights, r_value = run_c2_optimization()
    print(f"\nGenerated {PROBLEM_TYPE} function (constructor approach):")
    print(f"C₂ lower bound R(f): {r_value:.6f}")
    if r_value > TARGET_VALUE:
        print(f"✅ Exceeds target {TARGET_VALUE} (improvement: {(r_value - TARGET_VALUE):.6f})")
    else:
        pct = 100.0 * r_value / TARGET_VALUE if TARGET_VALUE > 0 else 0.0
        print(f"Progress vs target {TARGET_VALUE}: {pct:.1f}%")
    print(f"Function length: {len(heights)}")