import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SumManhattan_CurvedSurface_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3636
    prompt_template = r"""Define P(k) as the sum of (|x| + |y| + |z|)^2 over all integer triples (x, y, z) such that x × y × z = k. Compute the sum of P(k) for all integers k in the range [{A}, {B}] (inclusive)."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SumManhattan_CurvedSurface_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_A_B" in self.parameter, "MAX_A_B is required in parameter"
        MAX_A_B = self.parameter["MAX_A_B"]
        assert MAX_A_B >= 1, "MAX_A_B should be greater than or equal to 1"

        A = self.parameter["A"] = random.randint(1, MAX_A_B)
        B = self.parameter["B"] = random.randint(A, MAX_A_B)

        
        def funa(l: int, r: int) -> int:
            """Sum of i for i in [l..r], mod mo."""
            cnt = r - l + 1
            return (l + r) * cnt // 2

        def ready(x: int) -> int:
            """Sum of i^2 for i in [1..x], mod mo."""
            return x * (x + 1) * (2 * x + 1) // 6

        def funb(l: int, r: int) -> int:
            """Sum of i^2 for i in [l..r], mod mo."""
            return ready(r) - ready(l - 1)

        def work2(n: int):
            """
            Compute the three helper sums for a given n:
            ans1 = sum_{i=1..n} floor(n/i)
            ans2 = sum_{i=1..n} [ sum_{j=1..i} j + i * sum_{j=1..floor(n/i)} j ]
            ans3 = sum_{i=1..n} [ sum_{j=1..i} j^2 + i * sum_{j=1..floor(n/i)} j^2 + 2 * (sum_{j=1..i} j) * (sum_{k=1..floor(n/i)} k) ]
            All mod mo.
            Uses divisor grouping to run in ~O(sqrt(n)).
            """
            ans1 = ans2 = ans3 = 0
            l = 1
            while l <= n:
                d = n // l
                r = n // d
                cnt = r - l + 1

                # accumulate contributions
                ans1 += cnt * d
                ans2 += funa(l, r) * d + cnt * funa(1, d)
                ans3 += funb(l, r) * d + cnt * funb(1, d) + 2 * funa(l, r) * funa(1, d)

                l = r + 1

            return ans1, ans2, ans3

        def work(n: int) -> int:
            """
            Compute the cumulative beauty sum S(n) = sum_{k=1..n} P(k)/4 (mod mo),
            where P(k) is the squared-Manhattan-distance sum on xyz=k.
            The final answer is 4*(S(b) - S(a-1)) mod mo.
            """
            ans = 0
            l = 1
            while l <= n:
                d = n // l
                r = n // d
                cnt = r - l + 1

                a1, a2, a3 = work2(d)
                ans += funb(l, r) * a1 + funa(l, r) * 2 * a2 + cnt * a3

                l = r + 1

            return ans

        result = work(B) - work(A - 1)
        result = result * 4
        assert result > 0, "Result should be positive"
        self.parameter["reference_answer"] = result
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(A = self.parameter["A"], B = self.parameter["B"])


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