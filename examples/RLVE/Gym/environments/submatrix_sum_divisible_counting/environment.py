import random
from typing import Optional
from Gym.environment import VerifiableEnvironment

class SubmatrixSumDivisibleCounting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a matrix of size {N} × {M}, where each element is an integer. Count the number of **contiguous, non-empty submatrices** whose sum is divisible by {K}. The matrix is:
{matrix}

Notes:
- Two submatrices are considered different if they differ in position, even if they contain identical elements.
- The entire matrix itself is also considered a submatrix.
- Output a single non-negative integer, which is the total number of submatrices whose sum is divisible by {K}."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SubmatrixSumDivisibleCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    
    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 2, "MAX_N_M should be greater than or equal to 2"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(2, MAX_N_M), random.randint(2, MAX_N_M)
        K = self.parameter["K"] = random.randint(2, N * M)
        matrix = self.parameter["matrix"] = [[random.randint(0, K - 1) for _ in range(M)] for _ in range(N)]


        # 2D prefix sums modulo K, 1-indexed
        a = [[0] * (M + 1) for _ in range(N + 1)]

        for i in range(1, N + 1):
            row = matrix[i - 1]
            ai = a[i]
            ai_1 = a[i - 1]
            for j in range(1, M + 1):
                v = row[j - 1]  # each a[i][j] <= K per problem statement
                # a[i][j] = (v + a[i-1][j] + a[i][j-1] + K - a[i-1][j-1]) % K
                ai[j] = (v + ai_1[j] + ai[j - 1] + K - ai_1[j - 1]) % K

        ans = 0
        b = [0] * (M + 1)           # reuse across pairs of rows
        cnt = [0] * K               # frequency array modulo K (size depends on K)

        # Enumerate pairs of rows (top=i+1 .. bottom=j)
        for i in range(0, N):
            ai = a[i]
            for j in range(i + 1, N + 1):
                aj = a[j]
                cnt[0] = 1  # empty prefix
                # Sweep columns, counting subarrays with sum % K == 0
                for k in range(1, M + 1):
                    v = aj[k] - ai[k]   # both already modulo K
                    if v < 0:
                        v += K          # avoid Python modulo in inner loop
                    b[k] = v
                    ans += cnt[v]
                    cnt[v] += 1
                # reset only the touched buckets (like the C++ code)
                for k in range(1, M + 1):
                    cnt[b[k]] = 0

        self.parameter["reference_answer"] = ans
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            K = self.parameter["K"],
            matrix = "[\n" + "\n".join(", ".join(map(str, row)) for row in self.parameter["matrix"]) + "\n]",
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
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                if processed_result == 0 :
                    return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == 0)
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]