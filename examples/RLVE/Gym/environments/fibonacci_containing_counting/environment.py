import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class FibonacciContainingCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3986
    prompt_template = r"""How many pairs of positive integers (a, b) are there such that, defining f by f(0)=a, f(1)=b, and f(n)=f(n−1)+f(n−2) for n≥2, there exists an n≥2 with f(n)={K}?"""
    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the FibonacciContainingCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K >= 2, "MAX_K should be greater than or equal to 2"

        K = self.parameter["K"] = random.randint(2, MAX_K)


        def gcd(a, b):
            return gcd(b, a % b) if b else a

        def lcm(a, b):
            return a // gcd(a, b) * b

        def main():
            fib = [1, 1]  # dynamic list
            e = 1
            while fib[e] + fib[e - 1] <= K:
                fib.append(fib[e] + fib[e - 1])
                e += 1

            ans = 0
            for i in range(1, e):
                a = fib[i - 1]
                b = fib[i]
                x = 1
                while (K - b * x) % a != 0 and K > b * x:
                    x += 1
                if K <= b * x:
                    continue
                ans += (K - b * x - 1) // lcm(a, b) + 1
            assert ans > 0, "The answer should be positive."
            return ans

        self.parameter["reference_answer"] = main()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(K = self.parameter["K"])
    

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