import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class NotContainingStringCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3193
    prompt_template = \
r"""Please count the number of binary (0/1) strings of length {N} that do **NOT** contain the substring {pattern}

Output the result modulo {MOD}."""

    def __init__(self,
                 max_MOD : int = 10000,
                 wrong_format: float = -1.0, wrong_range: float = -0.5, correct_answer: float = +1.0, wrong_answer: float = 0.0,
                 **kwargs) -> None:
        """
        Initialize the NotContainingStringCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_MOD = max_MOD

        self.rewards = {
            "wrong_format": wrong_format,
            "wrong_range": wrong_range,
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 3, "MAX_N should be greater than or equal to 3"
        N = self.parameter["N"] = random.randint(3, MAX_N)

        assert "MAX_M" in self.parameter, "MAX_M is required in parameter"
        MAX_M = self.parameter["MAX_M"]
        assert MAX_M >= 2, "MAX_M should be greater than or equal to 2"
        M = random.randint(2, min(N - 1, MAX_M))
        one_probability = random.random()
        pattern = self.parameter["pattern"] = "".join("1" if random.random() < one_probability else "0" for _ in range(M))

        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


        def build_prefix(pattern):
            """
            Build the KMP prefix function (pi array) for the given pattern.
            pi[i] = length of the longest proper prefix of pattern[:i+1]
                    which is also a suffix of pattern[:i+1].
            """
            m = len(pattern)
            pi = [0] * m
            j = 0
            for i in range(1, m):
                while j > 0 and pattern[i] != pattern[j]:
                    j = pi[j - 1]
                if pattern[i] == pattern[j]:
                    j += 1
                pi[i] = j
            return pi

        def multiply_matrices(A, B, mod):
            """
            Multiply two square matrices A and B under modulo mod.
            """
            size = len(A)
            C = [[0] * size for _ in range(size)]
            for i in range(size):
                for k in range(size):
                    if A[i][k]:
                        aik = A[i][k]
                        for j in range(size):
                            C[i][j] = (C[i][j] + aik * B[k][j]) % mod
            return C

        def matrix_power(matrix, exponent, mod):
            """
            Raise 'matrix' to the power 'exponent' under modulo 'mod'
            using binary exponentiation.
            """
            size = len(matrix)
            # initialize result as the identity matrix
            result = [[int(i == j) for j in range(size)] for i in range(size)]
            base = matrix
            while exponent > 0:
                if exponent & 1:
                    result = multiply_matrices(result, base, mod)
                base = multiply_matrices(base, base, mod)
                exponent >>= 1
            return result

        def compute():
            # Build KMP prefix function for the forbidden pattern
            pi = build_prefix(pattern)

            # Build the (M+1) x (M+1) transition matrix
            # States 0..M-1 correspond to "currently matched prefix length"
            # State M is the absorbing forbidden state
            size = M + 1
            B = [[0] * size for _ in range(size)]

            # Fill transitions for states 0..M-1
            for state in range(M):
                for digit in map(str, range(2)):
                    k = state
                    # follow KMP fallback links
                    while k > 0 and digit != pattern[k]:
                        k = pi[k - 1]
                    if digit == pattern[k]:
                        k += 1
                    # transition from 'state' to 'k' on this digit
                    B[state][k] += 1

            # Make state M absorbing with all 2 digits
            B[M][M] = 2

            # Compute B^N mod MOD
            Bn = matrix_power(B, N, MOD)

            # Initial state is 0 (matched 0 chars), so the number of valid sequences of length N
            # that end in state j is Bn[0][j]. We sum over j = 0..M-1 (exclude forbidden state M).
            result = sum(Bn[0][j] for j in range(M)) % MOD

            return result

        self.parameter["reference_answer"] = compute()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], pattern = self.parameter["pattern"], MOD = self.parameter["MOD"])


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