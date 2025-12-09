import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Fibonacci_Environment(VerifiableEnvironment) : # Source: https://www.luogu.com.cn/problem/P1349
    prompt_template = \
r"""We have a sequence $A$, where $A[1] = {A1}$, $A[2] = {A2}$, and for $n > 2$ the recurrence is defined as $A[n] = {P} \times A[n - 1] + {Q} \times A[n - 2]$. Please compute $A[{N}] \bmod {modulo}$.

Output Format: Your final answer should be a **single integer** on a line by itself, representing the value of $A[{N}] \bmod {modulo}$.
"""
    def __init__(self,
                 modulo : int = 10000,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the Fibonacci_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
        
        self.modulo = modulo
    
    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 3, "MAX_N should be greater than or equal to 3"
        
        N = self.parameter["N"] = random.randint(3, self.parameter["MAX_N"])

        A1 = self.parameter["A1"] = random.randint(0, self.modulo - 1)
        A2 = self.parameter["A2"] = random.randint(0, self.modulo - 1)
        
        P = self.parameter["P"] = random.randint(1, self.modulo - 1)
        Q = self.parameter["Q"] = random.randint(1, self.modulo - 1)
        

        def matrix_multiply(A, B, mod) :
            n = len(A)
            C = [[0] * n for _ in range(n)]
            # transpose B for cache‐friendly access
            B_T = [[B[j][i] for j in range(n)] for i in range(n)]
            for i in range(n) :
                for j in range(n) :
                    s = 0
                    for k in range(n) :
                        s += A[i][k] * B_T[j][k]
                    C[i][j] = s % mod
            return C

        def matrix_power(A, k, mod) :
            n = len(A)
            # result = identity
            result = [[0] * n for _ in range(n)]
            for i in range(n) :
                result[i][i] = 1
            base = [row[:] for row in A]
            while k > 0 :
                if k & 1 :
                    result = matrix_multiply(result, base, mod)
                base = matrix_multiply(base, base, mod)
                k >>= 1
            return result

        def solve(p, q, a1, a2, n, m) :
            # base cases
            if n == 1 :
                return a1 % m
            if n == 2 :
                return a2 % m

            # build the transformation matrix modulo m
            T = [
                [p % m, q % m],
                [1,     0    ],
            ]
            # raise T to the (n-2)th power
            Tn = matrix_power(T, n - 2, m)
            # multiply by the base vector [a2, a1]
            return (Tn[0][0] * (a2 % m) + Tn[0][1] * (a1 % m)) % m

        self.parameter["reference_answer"] = solve(P, Q, A1, A2, N, self.modulo)
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            A1 = self.parameter["A1"],
            A2 = self.parameter["A2"],
            P = self.parameter["P"],
            Q = self.parameter["Q"],
            N = self.parameter["N"],
            modulo = self.modulo,
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