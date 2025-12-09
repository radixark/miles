import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class BitwiseOperationSequenceCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4424
    prompt_template = \
r"""You are given an array A of {N} + 1 binary strings, each of length {M}. The strings are:
{A}

You will insert an operation (`AND` or `OR`) between every pair of adjacent elements in A, resulting in {N} operations total, to form an expression. You can evaluate the expression from left to right (without operator precedence) to get the final result of the expression.
Count the number of different ways to insert these operations such that the final result equals this binary string: {R}"""

    def __init__(self,
                wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                **kwargs) :
        """
        Initialize the BitwiseOperationSequenceCounting_Environment instance.
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

        N = self.parameter["N"] = random.randint(2, MAX_N_M)
        M = self.parameter["M"] = random.randint(2, MAX_N_M)

        self.parameter["A"] = A = [None] * (N + 1)
        A[0] = "0" * M
        result = "0" * M
        AND_probability = random.random()
        for i in range(1, N + 1) :
            one_probability = random.random()
            A[i] = "".join(str(int(random.random() < one_probability)) for _ in range(M))
            operation = "AND" if random.random() < AND_probability else "OR"
            if operation == "AND" :
                result = "".join(str(int(A[i][j]) & int(result[j])) for j in range(M))
            else :
                result = "".join(str(int(A[i][j]) | int(result[j])) for j in range(M))
        self.parameter["R"] = result


        S = A[1 :]

        # rk will store the current column order (0-indexed)
        rk = list(range(M))
        # b[j][i] will store the bit in column j, row i
        b = [[0] * N for _ in range(M)]

        # Read the N rows of the matrix, and maintain the stable partition of rk
        for i in range(N):
            s = S[i]
            # parse the bits of this row
            row = [int(ch) for ch in s]
            # fill b
            for j in range(M):
                b[j][i] = row[j]
            # stable partition rk: first zeros, then ones
            new_rk = []
            for k in rk:
                if row[k] == 0:
                    new_rk.append(k)
            for k in rk:
                if row[k] == 1:
                    new_rk.append(k)
            rk = new_rk

        # Compute Ans[j] = integer value of column j (bits b[j][N-1]...b[j][0]) mod MOD
        Ans = [0] * M
        for j in range(M):
            val = 0
            # build the number from most-significant bit b[j][N-1] down to b[j][0]
            for i in range(N - 1, -1, -1):
                val = val * 2 + b[j][i]
            Ans[j] = val

        def compute() :
            s = result
            # Find the first position in rk where the bit is '1'
            Rk_idx = M  # default to sentinel
            for idx in range(M):
                if s[rk[idx]] == '1':
                    Rk_idx = idx
                    break
            # Find the last position in rk where the bit is '0'
            Lk_idx = -1  # default to before first
            for idx in range(M - 1, -1, -1):
                if s[rk[idx]] == '0':
                    Lk_idx = idx
                    break

            # If the first '1' comes before the last '0', no valid interval
            if Rk_idx < Lk_idx:
                return 0
            else:
                # Determine the two endpoints' values
                x_val = 0 if Lk_idx == -1 else Ans[rk[Lk_idx]]
                y_val = (2 ** N) if Rk_idx == M else Ans[rk[Rk_idx]]
                # Answer is y_val - x_val
                return y_val - x_val
        
        self.parameter["reference_answer"] = compute()
        assert self.parameter["reference_answer"] > 0
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            A = "\n".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
            R = self.parameter["R"],
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
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]