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


def _smooth_max_topk(conv: np.ndarray, closure_temp: float = 1.0) -> Tuple[float, np.ndarray]:
    """Smooth max using log-sum-exp trick, with optional temperature annealing."""
    if conv.size == 0:
        return 0.0, np.zeros_like(conv)
    if closure_temp <= 0.0:
        # Use the hard max as fallback
        closure_temp = 1.0

    max_val = np.max(conv)
    normalized = (conv - max_val) * closure_temp
    sum_exp = np.sum(np.exp(normalized))
    smooth_max_val = (np.log(sum_exp) - np.log(closure_temp) + max_val) / closure_temp
    
    # Jacobian
    grad_conv = np.zeros_like(conv)
    if not np.isfinite(sum_exp):
        # Fallback to hard max if numerical issues
        max_val = np.max(conv)
        mask = (conv == max_val)
        count = int(np.sum(mask))
        if count > 0:
            grad_conv[mask] = 1.0 / count
        return float(smooth_max_val), grad_conv

    # Assign gradient proportional to exp(normalized) * closure_temp
    exp_normalized = np.exp(normalized)
    grad_assign = exp_normalized / sum_exp * closure_temp
    grad_conv[:] = grad_assign
    
    return float(smooth_max_val), grad_conv


def _objective_and_grad_conv(conv: np.ndarray, closure_temp: float = 1.0) -> Tuple[float, np.ndarray]:
    """
    Compute C = l2_sq / (l1 * smooth_max_topk) and gradient dC/d(conv) using quotient rule,
    with automatic temperature annealing for the smooth max.
    """
    l2_sq, g_l2 = _simpson_l2sq(conv)
    l1, g_l1 = _l1(conv)
    smooth_max, g_smooth_max = _smooth_max_topk(conv, closure_temp=closure_temp)
    max_val = np.max(conv)  # For comparison

    # Fallback to hard max if the smooth max is too similar to max_val
    gap = smooth_max - max_val
    if gap < 0.001 * smooth_max:  # If they are too close, use hardest possible max
        mask = (conv == max_val)
        smooth_max = float(max_val)
        temp_gap = 1e-10
        if np.sum(mask) > 0:
            count = int(mask.sum())
            grad_temp = mask.astype(conv.dtype) / count * temp_gap
        else:
            grad_temp = np.zeros_like(conv) * temp_gap
            smooth_max = float(max_val)

    if l1 <= 0.0:
        return 0.0, np.zeros_like(conv)

    denom = l1 * smooth_max
    C = l2_sq / denom

    # dC = (g_l2 * denom - l2_sq * (g_l1 * smooth_max + l1 * g_smooth_max)) / denom^2
    num_grad = g_l2 * denom - l2_sq * (g_l1 * smooth_max + l1 * g_smooth_max)
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


def _phase_update(h_batch, opt_list, lr, add_noise=False, t=0, eta=1e-3, gamma=0.55, comb_perturb_prob=0.05):
    """
    One optimization step for the whole batch.
    Optionally adds comb perturbations to prevent peak lock-in.
    """
    B, N = h_batch.shape
    # compute dC/dh for each candidate
    C_vals, conv_grads = _batch_objective(h_batch)
    grads = np.zeros_like(h_batch, dtype=h_batch.dtype)
    for b in range(B):
        g_h = _grad_h_from_conv_grad(np.clip(h_batch[b], 0.0, None), conv_grads[b])
        grads[b] = g_h
        
        # Apply comb perturbation if enabled and random draw is successful
        if add_noise and (np.random.random() < comb_perturb_prob):
            # Create comb perturbation in h-space
            comb_amp = 5e-4
            period = 32
            # Random offset within the candidate
            offset = np.random.randint(0, N - period)
            # Build comb signal
            comb_signal = np.zeros_like(h_batch[b])
            for i in range(offset, N, period):
                if i+3 < N:
                    comb_signal[i:i+3] += comb_amp
            # Apply comb perturbation
            h_batch[b] = np.clip(h_batch[b] + comb_signal, 0.0, None)
            # Track the perturbation for logging
            C_vals[b] += comb_amp**2 * 0.1  # Small fake reward for exploration

    if add_noise:
        sigma = eta / ((t + 1) ** gamma)
        # Add adaptive noise after comb perturbation
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
            pass  # tqdm displays progress
    return h, float(last_C)


# Preconditioning parameter
TOEPLITZ_TEMPORAL_SCALE = 0.6

def construct_function():
    """
    Use the paper's 4-phase gradient-based search to maximize
        R(f) = ||f*f||_2^2 / (||f*f||_1 * ||f*f||_inf).
    Returns (heights, r_value).
    """
    # ---------- Enhanced search parameters (frequency shaping + tighter spending) ----------
    N = 512               # Base resolution
    B = 32                 # Larger batch for comb candidates (double original)
    ITER = 30_000          # Reduced iterations for faster evaluation
    EXPLORE_STEPS = 15_000 # Earlier exploration -> more diversity
    DROP_EVERY = 10_000    # More frequent respawns
    KEEP_FRAC = 0.4        # Keep more elite candidates
    LR_EXPLORE = 4e-3      # Higher exploration lr
    LR_EXPLOIT = 8e-4      # Lower exploit lr due to better candidates
    ETA = 2e-4             # Lower adaptive noise budget
    GAMMA = 0.7            # Noise decay rate
    dtype = np.float32
    
    # --- Added heuristics for gating:
    topk_goal = 0.17       # Target Top-k gap reduction
    plateau_goal = 0.20    # Target plateau width increase
    warmup_steps = 1_000   # Number of Adam steps before plateauing begins
    max_plateau_width = 350  # Maximum acceptable computation per candidate (samples)

    # --- Added: Automatic exclusion for non-top-pk modes (frequency bounding)
    FREQ_FILTER = 0.75     # Preserve only highest 75% of modes
    K_TOP = 3              # Top 3 local modes for filtering

    rng = np.random.default_rng()

    def init_sampler(m):
        # Randomly choose between standard candidate and comb candidate
        p_comb = 0.3  # probability for comb candidate
        draw = rng.random(m)
        is_comb = draw < p_comb

        i0 = int(0.25 * (N-1))
        i1 = int(0.75 * (N-1))

        envelope_scale_low = 0.1
        envelope_scale_high = 0.3
        # For the base level (envelope) and the comb amplitude
        base_levels = rng.uniform(envelope_scale_low, envelope_scale_high, size=m)
        # For comb candidate: base_level and comb amplitude 0.05
        # For standard candidate: envelope base level, but the comb is zero.

        # The period for comb: 32 samples, and each tooth is 3 samples
        period = 32
        comb_amp = 0.05

        candidates = np.zeros((m, N), dtype=dtype)

        for b in range(m):
            if is_comb[b]:
                # comb candidate: entire function is base_level + periodic comb
                candidates[b] = base_levels[b] + comb_amp
                # Add comb teeth
                start_indices = np.arange(0, N, period)
                for k in start_indices:
                    if k+2 < N:
                        candidates[b, k:k+3] += comb_amp
            else:
                # standard candidate: central region and spike
                A = base_levels[b]   # same as envelope_scale in base_level
                # Spike amplitude and position
                # Use one hyperparameter for central region
                B = rng.uniform(0.6, 1.0, size=m)[b]
                # Ensure realistic spike position
                n_center = N // 2
                mid_range = int((i0 + i1) / 2)
                spike_pos = rng.integers(n_center - mid_range, n_center + mid_range, size=1)[0]

                # Set the full central region to the envelope amplitude
                if i0 <= i1:
                    candidates[b, i0:i1+1] = A
                    # Set spike
                    # Use floor division for symmetric placement
                    low_idx = max(i0, spike_pos - 1)
                    high_idx = min(i1, spike_pos + 1)
                    if high_idx >= low_idx:
                        candidates[b, low_idx:high_idx+1] = B

        # Apply frequency shaping (via FFT) to candidates
        n_filters = 16
        total_energy = 0
        for i in range(m):
            # Apply FFT filter
            spec = np.fft.rfftfreq(N, 1.0 / N)
            if N % 2 == 0:
                spec = np.r_[spec][1:]
            freqs = np.abs(np.fft.rfft(candidates[i]))
            idx = np.argsort(freqs)[::-1]  # sort by amplitude
            candidates[i][:,idx[:N//2]] = candidates[i][:, ][:, idx[:N//2]] * FREQ_FILTER

        return candidates

    rng = np.random.default_rng()

    def init_sampler(m):
        # Randomly choose between standard candidate and comb candidate
        p_comb = 0.3  # probability for comb candidate
        draw = rng.random(m)
        is_comb = draw < p_comb

        i0 = int(0.25 * (N-1))
        i1 = int(0.75 * (N-1))

        envelope_scale_low = 0.1
        envelope_scale_high = 0.3
        # For the base level (envelope) and the comb amplitude
        base_levels = rng.uniform(envelope_scale_low, envelope_scale_high, size=m)
        # For comb candidate: base_level and comb amplitude 0.05
        # For standard candidate: envelope base level, but the comb is zero.

        # The period for comb: 32 samples, and each tooth is 3 samples
        period = 32
        comb_amp = 0.05

        candidates = np.zeros((m, N), dtype=dtype)

        for b in range(m):
            if is_comb[b]:
                # comb candidate: entire function is base_level + periodic comb
                # We set the base level and then add the comb (periodic) once per candidate
                candidates[b] = base_levels[b] + comb_amp
                # Add comb teeth: periodic with period 32, each tooth 3 samples
                start_indices = np.arange(0, N, period)
                for k in start_indices:
                    if k+2 < N:
                        candidates[b, k:k+3] += comb_amp
            else:
                # standard candidate: central region and spike
                A = base_levels[b]   # same as envelope_scale in base_level
                # Spike amplitude and position
                B = rng.uniform(0.6, 1.0, size=m)[b]
                spike_pos = rng.integers(i0, i1+1, size=m).astype(int)[b]

                # Set the full central region to the envelope amplitude
                if i0 <= i1:
                    candidates[b, i0:i1+1] = A
                    # Now, for the spike: if the spike does not go out of [i0, i1], set it; otherwise, set only the part inside.
                    low_idx = max(i0, spike_pos - 3//2)
                    high_idx = min(i1, spike_pos + 3//2)
                    if high_idx >= low_idx:
                        # We have a valid segment
                        candidates[b, low_idx:high_idx+1] = B
                    # Otherwise, do nothing (spike is entirely outside the central region, already zeros)

        return candidates

    # init population and per-candidate Adam
    h_batch = init_sampler(B)
    opt_list = [_Adam(shape=(N,), lr=LR_EXPLORE, dtype=dtype) for _ in range(B)]

    def _estimate_promising_candidates(conv: np.ndarray) -> float:
        """Proxy for plateau width and peak density via FFT analysis."""
        # Short-time Fourier transform
        from scipy.signal import stft
        window = stft.get_window('hann', int(min(256, len(conv)//8)))
        frequencies, times, Sxx = stft(conv, nperseg=min(1024, len(conv)), noverlap=0.9)
        
        # Coarsely estimate plateau width (approximate frequency bandwidth)
        freq_range = np.max(frequencies)  # max frequency component
        plateau_width = (freq_range * np.log(np.max(conv))) / np.std(conv)
        return plateau_width
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