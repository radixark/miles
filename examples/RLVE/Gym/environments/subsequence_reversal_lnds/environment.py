import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SubsequenceReversalLNDS_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3607
    prompt_template = \
r"""You are given a sequence A of {N} integers: {A}
You may choose a subsequence of A, defined by a strictly increasing sequence of indices i₁, ..., iₖ (1 ≤ i₁ < ... < iₖ ≤ {N}, k >= 1), and **reverse the order of the elements at those indices** (i.e., A[i₁] becomes A[iₖ], ..., A[iₖ] becomes A[i₁]). Please **maximize the length of the longest non-decreasing subsequence** (not necessarily contiguous) in the resulting array. Output a single integer — the maximum achievable length."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the SubsequenceReversalLNDS_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 4, "N should be greater than or equal to 4"

        self.parameter["A"] = [random.randint(1, N) for _ in range(N)]


        # Read heights, 1-indexed
        A = [0] + self.parameter["A"]
        M = max(A)

        # dp[l][r][L][R]: max LIS length in A[l..r] after reversing at most one subsequence,
        # considering only values in [L..R]
        # Dimensions: (N+2) x (N+2) x (M+2) x (M+2)
        dp = [[[[0] * (M+2) for _ in range(M+2)] for _ in range(N+2)] for _ in range(N+2)]

        # Base case: intervals of length 1
        for i in range(1, N+1):
            for L in range(1, A[i] + 1):
                for R in range(A[i], M + 1):
                    dp[i][i][L][R] = 1

        # Build up for intervals of length = 2..N
        for length in range(2, N+1):
            for l in range(1, N - length + 2):
                r = l + length - 1
                for span in range(1, M+1):
                    for L in range(1, M - span + 2):
                        R = L + span - 1

                        # 1) shrink the allowed value range
                        val = dp[l][r][L+1][R]
                        if dp[l][r][L][R-1] > val:
                            val = dp[l][r][L][R-1]

                        # 2) extend by taking A[l] at the left (if it matches L)
                        tmp = dp[l+1][r][L][R] + (1 if A[l] == L else 0)
                        if tmp > val:
                            val = tmp

                        # 3) extend by taking A[r] at the right (if it matches R)
                        tmp = dp[l][r-1][L][R] + (1 if A[r] == R else 0)
                        if tmp > val:
                            val = tmp

                        # 4) reverse a subsequence spanning the ends
                        tmp = dp[l+1][r-1][L][R]
                        if A[l] == R:
                            tmp += 1
                        if A[r] == L:
                            tmp += 1
                        if tmp > val:
                            val = tmp

                        dp[l][r][L][R] = val

        # The answer is dp[1][N][1][M]
        self.parameter["reference_answer"] = dp[1][N][1][M]
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = ", ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"], start = 1)),
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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]