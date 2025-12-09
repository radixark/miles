import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class LandformGenerationCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3255
    prompt_template = \
r"""You are given two arrays `H` and `C`, each of length {N}:
H: {H}
C: {C}

A permutation `p` of the indices `0` to `{N_minus_1}` (i.e., `p[0], p[1], ..., p[{N_minus_1}]`) is considered **valid** if and only if the following condition holds for every index `i` from `0` to `{N_minus_1}`: there are **fewer than** C[p[i]] indices `j` (j < i) such that H[p[j]] > H[p[i]].
Please count the number of **distinct sequences** `H[p[0]], H[p[1]], ..., H[p[{N_minus_1}]]` that can be obtained by a valid permutation `p`. (Two permutations producing the same `H`-sequence count as one.) Output the result modulo {MOD}."""

    def __init__(self,
                 max_MOD : int = 1000000000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the LandformGenerationCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_MOD = max_MOD
        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        example_H = [random.randint(1, N) for _ in range(N)]
        A = [None] * N
        for i, Hi in enumerate(example_H) :
            A[i] = (Hi, random.randint(sum(int(Hj > Hi) for Hj in example_H[: i]) + 1, sum(int(Hj > Hi) for Hj in example_H) + 1))
        random.shuffle(A)
        self.parameter["A"] = A.copy()

        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


        # ---------- pre-processing ----------
        # sort by height desc, key asc
        A.sort(key=lambda x: (-x[0], x[1]))

        # ---------- 2. contour (height) sequences ----------
        ans_heights = 1
        start = 0
        while start < N:
            end = start
            h_cur = A[start][0]
            while end + 1 < N and A[end + 1][0] == h_cur:     # same-height block
                end += 1

            processed = start + 1                              # 1-based
            dp = [0] * (processed + 2)                         # dp[0 … processed]

            first_key = A[start][1]
            for j in range(1, min(processed, first_key) + 1):
                dp[j] = 1

            for i in range(start + 1, end + 1):                # remaining in block
                key = A[i][1]
                limit = min(processed, key)
                for j in range(1, limit + 1):                  # prefix sums
                    dp[j] = (dp[j] + dp[j - 1]) % MOD

            last_key = A[end][1]
            res = sum(dp[1:min(processed, last_key) + 1]) % MOD
            ans_heights = (ans_heights * res) % MOD

            start = end + 1                                    # next block

        # ---------- output ----------
        self.parameter["reference_answer"] = ans_heights


    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            H = " ".join("H[{}]={}".format(i, Ai[0]) for i, Ai in enumerate(self.parameter["A"])),
            C = " ".join("C[{}]={}".format(i, Ai[1]) for i, Ai in enumerate(self.parameter["A"])),
            MOD = self.parameter["MOD"],
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
            if not (0 <= processed_result < self.parameter["MOD"]) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]