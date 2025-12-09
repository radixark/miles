import random
from typing import Optional
from Gym.environment import VerifiableEnvironment

class EnergyStorageMeter_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4067
    prompt_template = prompt_template = r"""I want to know the sum of max((i XOR j) − {K}, 0) over all pairs (i, j) such that 0 ≤ i < {N} and 0 ≤ j < {M}, where XOR denotes the bitwise XOR operation."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the EnergyStorageMeter_Environment instance.
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
        assert MAX_N_M >= 4, "MAX_N_M should be greater than or equal to 4"

        N = self.parameter["N"] = random.randint(2, MAX_N_M)
        M = self.parameter["M"] = random.randint(2, MAX_N_M)
        K = self.parameter["K"] = random.randint(0, MAX_N_M)


        def S(l, r):
            # sum of integers from l to r inclusive
            if l > r:
                return 0
            cnt = r - l + 1
            return (l + r) * cnt // 2

        def calc(l, r, x):
            # corresponds to the C++ inline calc
            if l <= x <= r:
                return S(0, r - x)
            elif r < x:
                return 0
            else:  # x < l
                return S(l - x, r - x)

        def solve():
            # collect set bit positions (0..59) for N and M
            bitsN = [i for i in range(N.bit_length() + 1) if (N >> i) & 1]
            bitsM = [j for j in range(M.bit_length() + 1) if (M >> j) & 1]

            ans = 0

            for i in bitsN:
                for j in bitsM:
                    u = i if i < j else j
                    v = i ^ j ^ u  # equals max(i, j)

                    # Clear lower (i+1) bits of N and (j+1) bits of M, then XOR
                    ni = (N >> (i + 1)) << (i + 1)
                    mj = (M >> (j + 1)) << (j + 1)
                    x = ni ^ mj

                    # Clear lower v bits of x
                    if v > 0:
                        x = (x >> v) << v

                    # r = x with its lower v bits set to 1
                    r = x | ((1 << v) - 1) if v > 0 else x

                    contrib = (1 << u) * calc(x, r, K)
                    ans += contrib

            return ans

        self.parameter["reference_answer"] = solve()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], M = self.parameter["M"], K = self.parameter["K"])


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
                if processed_result == 0 :
                    return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == 0)
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]