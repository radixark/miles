import random
from typing import Optional
from bisect import bisect_left
from Gym.environment import VerifiableEnvironment


class RangeConstrained_IncreasingSequence_Counting_Environment(VerifiableEnvironment):
    prompt_template = \
r"""Count the number of integer sequences A[0], A[1], ..., A[{N_minus_1}] of length {N} such that:
- For each A[i], it is either 0 or an integer in [L[i], R[i]]
- At least one A[i] is greater than 0
- All non-zero A[i] form a strictly increasing sequence in order (i.e., if A[i] > 0 and A[j] > 0 with i < j, then A[i] < A[j])

The bounds L[i] and R[i] for each position are given as:
{L_and_R}

Output the number of such sequences modulo {MOD}.
"""
    MOD = 10 ** 9 + 7

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the RangeConstrained_IncreasingSequence_Counting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        L = self.parameter["L"] = [random.randint(1, N * N) for _ in range(N)]
        R = self.parameter["R"] = [random.randint(Li, N * N) for Li in L]


        A, B = L.copy(), R.copy()
        coords = []
        for ai, bi in zip(A, B) :
            coords.append(ai)
            coords.append(bi + 1)

        # Coordinate compression
        coords = sorted(set(coords))
        tot = len(coords)
        for i in range(N):
            A[i] = bisect_left(coords, A[i])
            B[i] = bisect_left(coords, B[i] + 1)

        # Precompute modular inverses up to N
        inv = [0] * (N + 1)
        inv[1] = 1
        for i in range(2, N + 1):
            inv[i] = (self.MOD - self.MOD // i) * inv[self.MOD % i] % self.MOD

        # DP arrays
        # C[k] will hold binomial-like coefficients for each segment length
        C = [0] * (N + 1)
        # g[k] is number of ways ending with the k-th school (k from 0 to N)
        g = [0] * (N + 1)
        g[0] = 1  # base: no school chosen yet

        # Process each compressed segment j
        for j in range(tot - 1):
            length = coords[j + 1] - coords[j]
            # Build C array: C[k] = C(length + k - 1, k)
            C[0] = 1
            for k in range(1, N + 1):
                C[k] = C[k - 1] * (length + k - 1) % self.MOD * inv[k] % self.MOD

            # Update DP in reverse order to avoid overwriting
            for i in range(N, 0, -1):
                # If school i-1 can cover this segment
                if A[i - 1] <= j < B[i - 1]:
                    f = 0
                    m = 1
                    c_val = length
                    # Sum contributions from previous states
                    for p in range(i - 1, -1, -1):
                        f = (f + c_val * g[p]) % self.MOD
                        # If previous school (p-1) also covers, increase combination size
                        if p > 0 and A[p - 1] <= j < B[p - 1]:
                            m += 1
                            c_val = C[m]
                    g[i] = (g[i] + f) % self.MOD

        # Sum all ways where at least one school participates
        self.parameter["reference_answer"] = sum(g[1:]) % self.MOD
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            L_and_R = "\n".join("L[{}]={} R[{}]={}".format(i, Li, i, Ri) for i, (Li, Ri) in enumerate(zip(self.parameter["L"], self.parameter["R"]))),
            MOD = self.MOD,
        )


    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                int_answer = int(answer)
                return int_answer
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if not (0 <= processed_result < self.MOD) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]