import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Cube_FixedLocalMaximumCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P5400
    prompt_template = \
r"""You are given a 3D grid of size {N} × {M} × {L}. Each cell will be filled with a unique number from 1 to {total} (where {total} = {N} × {M} × {L}). The numbers are assigned randomly and uniformly — every permutation of the {total} numbers over the grid is equally likely. A cell is called **dominant** if its value is strictly greater than all other cells that share at least one coordinate (i.e., same x, y, or z index). Please compute the probability that **exactly** {K} dominant cells exist after filling the grid.

**Output Format:** Output a single integer — the required probability modulo {MOD}."""
    MOD = 998244353

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the Cube_FixedLocalMaximumCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M_L" in self.parameter, "MAX_N_M_L is required in parameter"
        MAX_N_M_L = self.parameter["MAX_N_M_L"]
        assert MAX_N_M_L >= 2, "MAX_N_M_L should be greater than or equal to 2"

        N, M, L = self.parameter["N"], self.parameter["M"], self.parameter["L"] = random.randint(2, MAX_N_M_L), random.randint(2, MAX_N_M_L), random.randint(2, MAX_N_M_L)
        K = self.parameter["K"] = random.randint(2, min(N, M, L))


        def inv_list(n):
            """Compute modular inverses of 1..n under MOD."""
            invs = [0] * (n + 1)
            invs[1] = 1
            for i in range(2, n + 1):
                invs[i] = (-(self.MOD // i) * invs[self.MOD % i]) % self.MOD
            return invs

        def modinv(x):
            """Modular inverse of x under MOD (MOD is prime)."""
            return pow(x, self.MOD - 2, self.MOD)

        def compute():
            Q = min(N, M, L)
            invs = inv_list(Q)

            # R(x) = (N-x)*(M-x)*(L-x) mod MOD
            def R(x):
                return (N - x) * (M - x) % self.MOD * (L - x) % self.MOD
            
            # Prepare arrays of length Q+1
            vals  = [0] * (Q + 1)
            iprod = [0] * (Q + 1)  # corresponds to iVals in C++
            iprod[0] = 1
            
            R0 = R(0)
            # Build prefix products of (R0 - R(i))
            for i in range(1, Q + 1):
                vals[i]   = (R0 - R(i)) % self.MOD
                iprod[i]  = iprod[i - 1] * vals[i] % self.MOD
            
            # Compute inverses of those prefix products by reversing
            inv_total = modinv(iprod[Q])
            for i in range(Q, 0, -1):
                prev = iprod[i - 1]
                iprod[i] = inv_total * prev % self.MOD
                inv_total = inv_total * vals[i] % self.MOD
            
            # Now do the main summation for the answer
            ans = 0
            C = 0
            S = 1
            for i in range(1, Q + 1):
                # S accumulates product over R(i-1) * iprod[i]
                S = S * R(i - 1) % self.MOD * iprod[i] % self.MOD
                
                # update C according to i vs K
                if i == K:
                    C = 1
                elif i > K:
                    # C = -C * i * invs[i - K]  (all mod MOD)
                    C = (-C * i * invs[i - K]) % self.MOD
                
                ans = (ans + C * S) % self.MOD
            
            return ans

        self.parameter["reference_answer"] = compute()
    

    def _prompt_generate(self) -> str :
        N, M, L = self.parameter["N"], self.parameter["M"], self.parameter["L"]
        return self.prompt_template.format(
            N = N, M = M, L = L,
            total = N * M * L,
            K = self.parameter["K"],
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