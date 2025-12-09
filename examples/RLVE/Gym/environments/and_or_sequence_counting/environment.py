import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class AndOr_Sequence_Counting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an integer array `A` of length {N}:
{A}

Please count the number of valid integer arrays `B` of length {N} that satisfy the following conditions:
- For all indices 0 <= i <= {N_minus_1}, the value B[i] must be in the range: 0 <= B[i] < 2^{M} = {power_2_M}
- For all indices 0 <= i < {N_minus_1}, the following bitwise conditions hold:
  - (A[i] & B[i]) <= (A[i + 1] & B[i + 1])
  - (A[i] | B[i]) >= (A[i + 1] | B[i + 1])
  - (Here, `&` is the bitwise AND operator and `|` is the bitwise OR operator.)

**Output Format:** Your final answer should be a single integer — the number of valid arrays `B` that satisfy all the above conditions."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the AndOr_Sequence_Counting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
      
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 1, "M should be greater than or equal to 1"

        A = self.parameter["A"] = [random.randint(0, 2 ** M - 1) for i in range(N)]


        def dp1(N, M, A) :
            F = [[[0] * N for _ in range(N)] for _ in range(2)]
            for l in range(N) :
                for r in range(l, N) :
                    F[1][l][r] = 1

            for b in range(M + 1) :
                now = b % 2
                lst = now ^ 1

                for i in range(N) :
                    for j in range(N) :
                        F[now][i][j] = 0

                Pre = [0] * (N + 1)
                for i in range(1, N + 1) :
                    Pre[i] = Pre[i - 1] + ((A[i - 1] >> b) & 1)

                for l in range(N) :
                    for r in range(l, N) :
                        for x in range(l - 1, r + 1) :
                            if Pre[r + 1] - Pre[x + 1] != (r - x) :
                                continue

                            left_count  = F[lst][l][x]   if x   >= l else 1
                            right_count = F[lst][x + 1][r] if x+1 <= r else 1
                            F[now][l][r] += left_count * right_count

            return F[M % 2][0][N - 1]

        def dp2(N, M, A) :
            F = [[[0] * N for _ in range(N)] for _ in range(2)]
            for l in range(N) :
                for r in range(l, N) :
                    F[1][l][r] = 1

            for b in range(M + 1) :
                now = b % 2
                lst = now ^ 1
                for i in range(N) :
                    for j in range(N) :
                        F[now][i][j] = 0

                Pre = [0] * (N + 1)
                for i in range(1, N + 1) :
                    Pre[i] = Pre[i - 1] + ((A[i - 1] >> b) & 1)

                for l in range(N) :
                    for r in range(l, N) :
                        for x in range(l - 1, r + 1) :
                            if Pre[r + 1] - Pre[x + 1] != 0:
                                continue

                            left_count  = F[lst][l][x] if x >= l else 1
                            right_count = F[lst][x + 1][r] if x + 1 <= r else 1
                            F[now][l][r] += left_count * right_count

            return F[M % 2][0][N - 1]

        self.parameter["reference_answer"] = dp1(N, M - 1, A) * dp2(N, M - 1, A)
    
    def _prompt_generate(self) -> str :
        N, M = self.parameter["N"], self.parameter["M"]
        return self.prompt_template.format(
            N = self.parameter["N"],
            N_minus_1 = self.parameter["N"] - 1,
            M = self.parameter["M"],
            power_2_M = 2 ** self.parameter["M"],
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
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
            
            if self.parameter["reference_answer"] == 0 :
                return self.rewards["rewarding_weight"] * (processed_result == 0)

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]