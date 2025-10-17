from typing import List

import numpy as np


def compute_pass_rate(rewards: List[float]):
    group_size = args.n_samples_per_prompt
    group_number = args.rollout_batch_size
    assert len(val) == group_number * group_size
    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    val = np.array(val).reshape(group_number, group_size)

    def estimate_pass_at_k(num_samples, num_correct, k):
        """
        Estimates pass@k of each problem and returns them in an array.
        """

        def estimator(n, c, k):
            """
            Calculates 1 - comb(n - c, k) / comb(n, k).
            """
            if n - c < k:
                return 1.0
            return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

        return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct)])

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(val == 1, axis=1)
        num_samples = np.full(group_number, group_size)

        pass_k_estimates = estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict
