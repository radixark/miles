import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Matrix_BinaryExponentiation_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""We use the integer in the $i$-th row and $j$-th column to represent the element $A[i][j]$ of a matrix.

You are given a square matrix $A$ of size {N}×{N}:
{matrix}

Please compute the matrix $A^{K}$ (i.e., matrix $A$ raised to the power of ${K}$). Since the values may become very large, take each element **modulo {modulo}**.

**Output Format:**
Your final answer — the matrix $A^{K}$ — should be printed as ${N}$ lines separated by **line breaks**. Each line should contain ${N}$ integers separated by **spaces**.
Example (do **NOT** include the backticks or quotes):
```
{all_zeros}
```
"""
    def __init__(self,
                 modulo : int = 10000,
                 wrong_format : float = -1.0, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Matrix_BinaryExponentiation_Environment instance.
        """
        super().__init__(**kwargs)

        self.modulo = modulo

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 1, "N should be greater than or equal to 1"

        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K >= 2, "MAX_K should be greater than or equal to 2"

        K = self.parameter["K"] = random.randint(2, MAX_K)

        A = self.parameter["A"] = [[random.randint(0, self.modulo - 1) for j in range(N)] for i in range(N)]


        def matrix_multiply(A, B, mod) :
            n = len(A)
            C = [[0] * n for _ in range(n)]
            
            B_T = [[B[j][i] for j in range(n)] for i in range(n)]
            
            for i in range(n) :
                for j in range(n) :
                    sum_val = 0
                    for k in range(n) :
                        sum_val += A[i][k] * B_T[j][k]
                    C[i][j] = sum_val % mod
            
            return C

        def matrix_power(A, k, mod) :
            n = len(A)
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

        self.parameter["gold_answer"] = matrix_power(A, K, self.modulo)
        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, row)) for row in self.parameter["gold_answer"])
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            matrix = "\n".join(" ".join(map(str, row)) for row in self.parameter["A"]),
            K = self.parameter["K"],
            modulo = self.modulo,
            all_zeros = "\n".join(" ".join("0" for _ in range(self.parameter["N"])) for _ in range(self.parameter["N"])),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                matrix = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        matrix.append(list(map(int, line.split())))
                return matrix
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            A_K = processed_result
            if len(A_K) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            if not all(len(row) == self.parameter["N"] for row in A_K) :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(sum(answer == gold for answer, gold in zip(answer_row, gold_row)) for answer_row, gold_row in zip(A_K, self.parameter["gold_answer"])) / (self.parameter["N"] * self.parameter["N"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return A_K == self.parameter["gold_answer"]
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]