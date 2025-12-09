import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class PrefixConcatenation_Environment(VerifiableEnvironment) : # Source: https://www.luogu.com.cn/problem/P3216
    prompt_template = \
r"""Define $Concatenate(n)$ as the number formed by concatenating all positive integers from $1$ to $n$ in order. For example, when $n = 12$, $Concatenate(12) = 123456789101112$

Your task is to compute $Concatenate({N}) \bmod {M}$.

**Output Format:** Your final answer should be a **single integer** in the range $[0, {M})$, printed on a line by itself.
"""

    def __init__(self,
                 max_modulo : int = 1000000,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the PrefixConcatenation_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }

        self.max_modulo = max_modulo
    
    
    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 2, "MAX_N should be greater than or equal to 1"

        N = self.parameter["N"] = random.randint(2, MAX_N)
        M = self.parameter["M"] = random.randint(3, self.max_modulo)


        def mat_mul(A, B) :
            return [
                [(A[i][0] * B[0][j] + A[i][1] * B[1][j] + A[i][2] * B[2][j]) % M
                for j in range(3)]
                for i in range(3)
            ]

        def mat_pow(base, exp) :
            R = [[1 if i == j else 0 for j in range(3)] for i in range(3)]
            while exp :
                if exp & 1 :
                    R = mat_mul(R, base)
                base = mat_mul(base, base)
                exp >>= 1
            return R

        def mat_vec_mul(A, v) :
            return [
                (A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2]) % M,
                (A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2]) % M,
                (A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2]) % M,
            ]

        state = [0, 1, 1]
        start = 1
        power_of_10 = 10

        while start <= N :
            end = min(N, power_of_10 - 1)
            block_size = end - start + 1

            B = [
                [power_of_10 % M, 1, 0],
                [0,               1, 1],
                [0,               0, 1]
            ]

            Bk = mat_pow(B, block_size)
            state = mat_vec_mul(Bk, state)

            start = power_of_10
            power_of_10 *= 10

        self.parameter["reference_answer"] = state[0]
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], M = self.parameter["M"])


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