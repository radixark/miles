import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class XorEquationCounting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an equation: X[1] XOR ... XOR X[{N}] = {K}
That is, the bitwise XOR of all variables X[1] through X[{N}] must equal the integer {K}. Each variable X[i] must satisfy the constraint: {L} <= X[i] <= {R} for all i = 1, ..., {N}. Please compute how many such combinations of values satisfy the equation. Give the result **modulo {MOD}**.

**Output Format:** Your final answer should be a single integer — the number of valid combinations modulo `{MOD}`."""
    MOD = 10000

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the XorEquationCounting_Environment instance.
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

        assert "RANGE" in self.parameter, "RANGE is required in parameter"
        RANGE = self.parameter["RANGE"]
        assert RANGE >= 1, "RANGE should be greater than or equal to 1"

        R = self.parameter["R"] = random.randint(0, RANGE)
        L = self.parameter["L"] = random.randint(0, R)

        K = 0
        for i in range(1, N + 1) :
            K ^= random.randint(L, R)
        self.parameter["K"] = K


        def mult(a: int, b: int) -> int:
            return a * b % self.MOD

        def add(a: int, b: int) -> int:
            s = a + b
            return s - self.MOD if s >= self.MOD else s

        def sub(a: int, b: int) -> int:
            d = a - b
            return d + self.MOD if d < 0 else d

        def power(a: int, n: int) -> int:
            result = 1
            while n > 0:
                if n & 1:
                    result = mult(result, a)
                a = mult(a, a)
                n >>= 1
            return result

        def idx3(v0: int, v1: int, v2: int) -> int:
            return v0 + (v1 << 1) + (v2 << 2)

        def idx2(v0: int, v1: int) -> int:
            return v0 + (v1 << 1)

        class Matrix:
            MOD = self.MOD
            def __init__(self):
                self.v = [[0]*8 for _ in range(8)]
            def __mul__(self, other):
                temp = [[0]*8 for _ in range(8)]
                for k in range(8):
                    for i in range(8):
                        aik = self.v[i][k]
                        if aik:
                            for j in range(8):
                                temp[i][j] += aik * other.v[k][j]
                c = Matrix()
                for i in range(8):
                    for j in range(8):
                        c.v[i][j] = temp[i][j] % self.MOD
                return c
            def __pow__(self, n):
                result = Matrix()
                for i in range(8):
                    result.v[i][i] = 1
                base = self
                while n > 0:
                    if n & 1:
                        result = result * base
                    base = base * base
                    n >>= 1
                return result

        def work4(c: int, a: int, b: int, k: int, N: int) -> int:
            if a > b:
                a, b = b, a
                c ^= (N & 1)
            if b == 0:
                return power(2, N-1) if k == 0 else 0
            w = 1 << (b.bit_length() - 1)
            if (w << 1) - 1 < k:
                return 0
            
            zy = Matrix()
            for v0 in (0,1):
                for v1 in (0,1):
                    for v2 in (0,1):
                        row = idx3(v0, v1, v2)
                        zy.v[row][idx3(v0^1, v1, v2)] = add(zy.v[row][idx3(v0^1, v1, v2)], b - w + 1)
                        zy.v[row][idx3(v0, 1, v2)] = add(zy.v[row][idx3(v0, 1, v2)], w if v1 else 1)
                        if a & w:
                            zy.v[row][idx3(v0^1, v1, v2^1)] = add(zy.v[row][idx3(v0^1, v1, v2^1)], a - w + 1)
                            zy.v[row][idx3(v0, 1, v2^1)] = add(zy.v[row][idx3(v0, 1, v2^1)], w if v1 else 1)
                        else:
                            zy.v[row][idx3(v0, v1, v2^1)] = add(zy.v[row][idx3(v0, v1, v2^1)], a + 1)

            zy = zy ** N
            bit = 1 if (k & w) else 0
            base_count = zy.v[idx3(0,0,0)][idx3(bit,1,c)]
            
            next_a = (a ^ w) if (a & w) else a
            next_b = b ^ w
            next_k = k ^ ((a & w) * c) ^ (w * (c ^ (N & 1)))
            
            return add(base_count, work4(c, next_a, next_b, next_k, N))

        def work2(b: int, k: int, N: int) -> int:
            if b == 0:
                return 1 if k == 0 else 0
            w = 1 << (b.bit_length() - 1)
            if (w << 1) - 1 < k:
                return 0
            zy = Matrix()
            for v0 in (0,1):
                for v1 in (0,1):
                    row = idx2(v0, v1)
                    zy.v[row][idx2(v0^1, v1)] = add(zy.v[row][idx2(v0^1, v1)], b - w + 1)
                    zy.v[row][idx2(v0, 1)] = add(zy.v[row][idx2(v0, 1)], w if v1 else 1)
            zy = zy ** N
            bit = 1 if (k & w) else 0
            base_count = zy.v[idx2(0,0)][idx2(bit,1)]
            next_b = b ^ w
            next_k = k ^ (w * (N & 1))
            return add(base_count, work2(next_b, next_k, N))

        self.parameter["reference_answer"] = work2(R, K, N) if L == 0 else sub(work4(0, L-1, R, K, N), work4(1, L-1, R, K, N))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K = self.parameter["K"],
            L = self.parameter["L"],
            R = self.parameter["R"],
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