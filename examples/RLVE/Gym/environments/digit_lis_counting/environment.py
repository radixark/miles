import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class DigitLISCounting_Environment(VerifiableEnvironment) : # Source : https://acm.hdu.edu.cn/showproblem.php?pid=4352
    prompt_template = \
r"""Consider all integers N in the inclusive range **[{L}, {R}]**. Interpret each N as a string of decimal digits. 
The **power** of N is defined as **the length of the longest strictly increasing subsequence** of its digits.

Please count how many integers N within the range [{L}, {R}] have a **power value exactly equal to {K}**.

**Output Format:** Your final answer should be a single integer — the total number of integers between {L} and {R} inclusive, for which the power is exactly {K}."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the DigitLISCounting_Environment instance.
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

        # Generate a random integer R with exactly N digits (no leading zeros)
        R = self.parameter["R"] = random.randint(10 ** (N - 1), 10 ** N - 1)
        # Generate a random integer L, L <= R
        L = self.parameter["L"] = random.randint(0, R)
        K = self.parameter["K"] = random.randint(1, min(N, 10))

        def new_sta(x, n) :
            for i in range(n, 10) :
                if (1 << i) & x :
                    return (x ^ (1 << i)) | (1 << n)
            return x | (1 << n)

        def cal(x) :
            return bin(x).count('1')

        def dfs(pos, sta, limit, lead) :
            if pos == -1 :
                return int(cal(sta) == K)
            if not limit and not lead and dp[pos][sta][K] != -1 :
                return dp[pos][sta][K]
            up = a[pos] if limit else 9
            ans = 0
            for i in range(up + 1) :
                new_state = 0 if lead and i == 0 else new_sta(sta, i)
                ans += dfs(pos - 1, new_state, limit and i == up, lead and i == 0)
            if not limit and not lead :
                dp[pos][sta][K] = ans
            return ans
        
        def solve(x) :
            nonlocal a
            pos = -1
            while x > 0 :
                pos += 1
                a[pos] = x % 10
                x //= 10
            return dfs(pos, 0, True, True)

        dp = [[[-1 for _ in range(K + 1)] for _ in range(1025)] for _ in range(N + 1)]
        a = [0] * N
        self.parameter["reference_answer"] = solve(R) - solve(L-1)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            L = self.parameter["L"],
            R = self.parameter["R"],
            K = self.parameter["K"],
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