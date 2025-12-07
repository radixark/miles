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


def _smooth_linf(conv: np.ndarray, T: float, k: int = 1) -> Tuple[float, np.ndarray]:
    """
    Smooth max version of ||f*f||_inf with temperature T.
    Returns: smooth_max_value, gradient w.r.t. conv.
    """
    if T == 0.0:
        # Fall back to hard max
        if conv.size == 0:
            return 0.0, np.zeros_like(conv)
        m = float(np.max(conv))
        mask = (conv == m)
        count = int(mask.sum())
        if count == 0 or m <= 0.0:
            return m, np.zeros_like(conv)
        grad = mask.astype(conv.dtype) / count
        return m, grad
    else:
        # Smooth max using a softmax style
        if conv.size == 0:
            return 0.0, np.zeros_like(conv)
        m = float(np.max(conv))
        # Normalize by subtracting m to stabilize the exponentials
        conv_shifted = conv - m
        exps = np.exp(T * conv_shifted)
        s = np.log(np.sum(exps)) / T + m
        grad = exps / np.sum(exps)
        return s, grad


def _objective_and_grad_conv(conv: np.ndarray, T: float = 3.0, k: int = 3) -> Tuple[float, np.ndarray]:
    """
    Compute C = l2_sq / (l1 * smooth_linf) and gradient dC/d(conv) using quotient rule.
    Default T=3.0 for the smooth max.
    """
    l2_sq, g_l2 = _simpson_l2sq(conv)
    l1, g_l1 = _l1(conv)
    smooth_linf_val, g_smooth_linf = _smooth_linf(conv, 3.0, k=k)   # Change T to 3.0

    if l1 <= 0.0 or smooth_linf_val <= 0.0:
        return 0.0, np.zeros_like(conv)

    denom = l1 * smooth_linf_val
    C = l2_sq / denom

    # dC = (g_l2 * denom - l2_sq * (g_l1 * smooth_linf_val + l1 * g_smooth_linf)) / denom^2
    num_grad = g_l2 * denom - l2_sq * (g_l1 * smooth_linf_val + l1 * g_smooth_linf)
    g_conv = num_grad / (denom * denom)

    return float(C), g_conv


def _all_positive_to_center(conv: np.ndarray) -> bool:
    """Check if non-zero parts form a contiguous block"""
    # Find non-zero indices
    idx = np.flatnonzero(conv)
    if len(idx) == 0:
        return False
    start = idx.min()
    end = idx.max()
    total_len = end - start + 1
    num_nonzero = len(idx)
    # Require at least 50% overlap between intended support and actual non-zero
    if total_len == 0:
        return False
    coverage = num_nonzero / total_len
    return coverage > 0.7


def _excluded_divergence(conv: np.ndarray) -> bool:
    """Return True if autoconvolution has divergent behavior at edges due to abrupt truncation"""
    center_idx = len(conv) // 2
    # Check edges for unusually low values after initial drop
    margin = 30  # Edge check window
    left_edge = conv[:margin]
    right_edge = conv[-margin:]
    
    # Find value at center
    center_val = conv[center_idx]
    
    # Calculate mean of edge values
    edge_mean = np.mean(np.concatenate([left_edge, right_edge]))
    
    # If center is much higher than edge region, indicates discontinuity
    return (center_val - edge_mean) / center_val > 0.3


def _grad_h_from_conv_grad(h: np.ndarray, g_conv: np.ndarray, curv_diag=None) -> np.ndarray:
    """
    Given dC/d(conv) and conv = h * h (full convolution),
    dC/dh = 2 * (g_conv convolved with reverse(h)) in 'valid' mode (length N).
    If curv_diag is provided, adjust the gradient by curvature information.
    """
    h_rev = h[::-1]
    # length(g_conv)=2N-1, length(h_rev)=N; 'valid' output length is N
    g_h = np.convolve(g_conv, h_rev, mode="valid")
    
    if curv_diag is not None:
        # Use curvature information to precondition the gradient
        #   Scale the gradient by a factor that depends on the local curvature (curv_diag) at each point.
        #   Higher curvature (more negative) should be damped, and lower (positive) should be amplified.
        #   We use a simple scaling: 1 + g_curv * curv_diag, where g_curv is a smoothed version of the gradient's frequency content.
        g_curv = np.abs(np.fft.fft(g_h))
        # Normalize the curvature (avoid over-emphasis) and use a more stable scaling
        g_curv = g_curv / (np.sqrt(g_curv**2).mean())  # Normalize by mean L2 to keep scale
        # Use arithmetic mean of curv_diag (already computed for every point) to set a scaling factor
        g_h = np.multiply(g_h, (1.0 + curv_diag * 0.005 * g_curv))  # 0.005 as a small factor to enable tuning later
    
    # Boost negative curvature (damps oscillatory directions)
    # This is empirically determined: if the gradient is negative, we want to make it more positive by damping
    g_h[g_h < 0] *= 1.05  # -20% more positive gradient weight (reduced from 40% to be less aggressive)
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
        # Initialize first-moment decay states to avoid transient drops
        self.m[:] = 10.0 * self.m
        self.v[:] = 10.0 * self.v

    def step(self, params, grad):
        if np.any(np.isinf(grad)):
            grad = np.nan_to_num(grad, nan=0.0)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * (grad * grad)
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = np.clip(self.v / (1 - self.b2 ** self.t), a_min=1e-8, a_max=None)  # Avoid division by zero
        return params + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset_like(self, params):
        self.m[...] = 0.0
        self.v[...] = 0.0
        self.t = 0


def _batch_objective(h_batch: np.ndarray, T: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized evaluation over a batch of candidates.
    Returns:
        C_vals: (B,) objective values
        conv_grads: list/array of per-candidate dC/d(conv) for backprop
    """
    B, N = h_batch.shape
    C_vals = np.zeros(B, dtype=np.float32)
    conv_grads = [None] * B
    T_local = T  # for all for simplicity
    for b in range(B):
        h = np.clip(h_batch[b], 0.0, None)
        conv = np.convolve(h, h, mode="full")
        Cb, g_conv = _objective_and_grad_conv(conv, T_local)
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
    """Linear ×2 upsampling on [-1/4,1/4] grid (uniform sampling in index space)."""
    N = h.shape[0]
    x_old = np.linspace(-0.5, 0.5, N)
    x_new = np.linspace(-0.5, 0.5, 2 * N)
    return np.interp(x_new, x_old, h)


def _single_candidate_finetune(h0: np.ndarray, lr=3e-3, steps=50_000, log_every=2_000) -> Tuple[np.ndarray, float]:
    """
    Pure exploitation (no noise) on a single vector with Adam + projection.
    """
    h = h0.astype(np.float32).copy()
    # Initialize Adam state with stronger moment estimates
    opt = _Adam(h.shape, lr=lr, dtype=h.dtype)
    # Apply initialization scaling
    h = 0.2 * h  # Helps escape local optima
    last_C = 0.0
    for t in trange(steps, desc="upsampled-finetune", leave=False):
        # objective and gradient
        conv = np.convolve(np.clip(h, 0.0, None), np.clip(h, 0.0, None), mode="full")
        C, g_conv = _objective_and_grad_conv(conv)
        g_h = _grad_h_from_conv_grad(np.clip(h, 0.0, None), g_conv)

        h = np.clip(opt.step(h, g_h.astype(h.dtype)), 0.0, None)
        last_C = C
        if log_every and (t + 1) % log_every == 0:
            pass  # tqdm displays progress
    return h, float(last_C)


def construct_function():
    """
    Use the paper's 4-phase gradient-based search to maximize
        R(f) = ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf).
    Returns (heights, r_value).
    """
    # ---------- Hyperparameters (close to paper, but conservatively smaller by default) ----------
    N = 768                 # search resolution (paper: 768; will truncate zeros automatically)
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
        if m == 0:
            return np.empty((0, N), dtype=dtype)
        arr = np.empty((m, N), dtype=dtype)
        for i in range(m):
            if rng.random() < 0.3:   # 30% chance to generate the ansatz structure
                # For the ansatz: broad envelope, two asymmetric spikes, weak comb
                x = np.linspace(-0.5, 0.5, N)
                # Broad envelope: triangle from -0.15 to 0.15
                envelope = 0.3 * np.maximum(0.0, 0.6 - np.abs(x)/0.15)
                
                # Right spike: tall narrow at 0.1
                offset1 = 0.1
                sigma_s1 = 0.02
                spike1 = 0.5 * np.exp(-(x - offset1)**2 / (2 * sigma_s1**2))
                
                # Left spike: broader and weaker (lower) at -0.1
                offset2 = -0.1
                sigma_s2 = 0.04
                spike2 = 0.2 * np.exp(-(x - offset2)**2 / (2 * sigma_s2**2))
                
                # Adjust comb parameters: fewer teeth but stronger amplitude at specific multiples
                comb_spread = 0.05   # Spacing between comb teeth
                comb_num = 3         # Number of teeth
                # We'll generate the centers around the spikes to have a comb effect
                base_centers = [offset1, offset2]
                comb_centers = []
                for base in base_centers:
                    centers = [base + i * comb_spread for i in range(-comb_num//2, comb_num//2+1)]
                    comb_centers.extend([c for c in centers if c >= -0.5 and c <= 0.5])
                # Remove duplicates if any
                comb_centers = list(set(comb_centers))
                comb_ampl = 0.2      # Amplitude of the comb teeth
                comb = np.zeros_like(x)
                for center in comb_centers:
                    sigma_c = 0.04
                    comb += comb_ampl * np.exp(-(x - center)**2 / (2 * sigma_c**2))
                
                total_f = envelope + spike1 + spike2 + comb
                # Normalize by range to make it positive and of reasonable scale
                factor = 1.0 / (max(total_f) - min(total_f))
                # We must ensure total_f is nonnegative
                total_f = total_f * factor + abs(min(total_f)) * factor  # to move min to zero
                
                arr[i] = total_f.astype(dtype)
                
            else:
                arr[i] = rng.uniform(-5.0, 5.0, size=N).astype(dtype)
        return arr

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
    # 2× upsample then fine-tune; can repeat once more to get 4× (consistent with paper)
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
    """
    if heights.size == 0:
        return 0.0
        
    try:
        conv = np.convolve(heights, heights, mode="full")
        l2_sq_val, _ = _simpson_l2sq(conv)
        l1_val, _ = _l1(conv)
        smooth_linf_val, _ = _smooth_linf(conv, T=0.0)  # Use hard max (averaged)
        
        if l1_val <= 1e-6 or smooth_linf_val <= 1e-6:
            return 0.0
            
        return float(l2_sq_val) / (l1_val * smooth_linf_val)
    except Exception as e:
        print(f"Error in objective computation: {e}")
        return 0.0

# We'll implement a helper for autoconvolution and clean computations
def compute_autoconvolution(heights):
    if heights.size == 0:
        return (0.0, np.array([])), (0.0, np.array([]))
    
    conv = np.convolve(heights, heights, mode="full")
    M = len(conv)
    
    # Get grid spacing
    dx_val = 1.0 / (len(heights) + 1)  # original grid spacing
    
    l1_val = dx_val * float(np.sum(conv))
    l2_sq_val = None  # Not computed if l1_val is too small
    
    if l1_val <= 1e-6:
        return (0.0, 0.0), (0.0, np.array([]))
    
    # Compute lateral/longitudinal curvature component for preconditioning
    h_center = np.fliplr([heights])[0]
    h_curvature = np.convolve(h_center, h_center, mode='full')
    # Extended support for curvature
    if h_curvature.size > M+1:
        h_curvature = h_curvature[int((h_curvature.size - (M+1))/2):]
    idx = int(len(h_curvature)/2)
    curv_diag = np.abs(np.gradient(h_curvature, dx_val))
    if idx < curv_diag.size and np.isfinite(curv_diag[idx]):
        curv_diag[idx] *= 0.8  # Damp central curvature estimates
    
    return (l2_sq_val, l1_val), (conv, h_curvature, curv_diag)

# But due to time, let's keep it simple for now and return the changes above.
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