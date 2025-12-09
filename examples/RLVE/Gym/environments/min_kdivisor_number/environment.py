import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MinKDivisorNumber_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1128
    prompt_template = \
r"""Find the **smallest positive integer `M`** such that it has **exactly `{K}` distinct positive divisors**.

**Output Format:**
Your final answer should be a single integer representing the value of `M`.
Example: `10` (do **NOT** include the backticks or quotes)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = 0.0, rewarding_strategy : str = "(gold/answer)^beta", rewarding_beta : float = 2.0, rewarding_weight : float = 1.0,
                 **kwargs) :
        """
        Initialize the MinKDivisorNumber_Environment instance.
        """

        super().__init__(**kwargs)
        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def prime_factorization(self, n, limit) :
        factors = []
        d = 2
        while d * d <= n :
            e = 0
            while n % d == 0 :
                n //= d
                e += 1
            if e > 0 :
                factors.append((d, e))
            d += 1
            if d > limit :
                return None
        if n > 1 :
            factors.append((n, 1))
        return factors
    

    def _generate(self) -> None :
        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K >= 1, "MAX_K should be greater than or equal to 1"
        
        K = self.parameter["K"] = random.randint(1, MAX_K)

        sum_e = sum(e for d, e in self.prime_factorization(K, float("inf")))
        all_primes = [2]
        while len(all_primes) < sum_e :
            all_primes.append(all_primes[-1] + 1)
            def check_prime(n) :
                if n == 2 or n == 3 :
                    return True
                if n < 2 or n % 2 == 0 :
                    return False
                for i in range(3, int(n ** 0.5) + 1, 2) :
                    if n % i == 0 :
                        return False
                return True
            while not check_prime(all_primes[-1]) :
                all_primes[-1] += 1

        dpF = dict()
        def dp(p, n) :
            if n == 1 :
                return 1
            if (p, n) in dpF :
                return dpF[(p, n)]
            Ans = (all_primes[p]) ** (n - 1)
            if p + 1 < len(all_primes) :
                factors = []
                for factor in range(1, int(n ** 0.5) + 1) :
                    if n % factor == 0 :
                        factors.append(factor)
                        if n // factor > factor :
                            factors.append(n // factor)
                
                for factor in factors :
                    if factor > 1 :
                        Ans = min(Ans, (all_primes[p] ** (factor - 1)) * dp(p + 1, n // factor))
            dpF[(p, n)] = Ans
            return Ans

        self.parameter["reference_answer"] = dp(0, K)
    
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
            if processed_result <= 0 :
                return self.rewards["wrong_format"]

            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["rewarding_weight"]

            factorization_result = self.prime_factorization(processed_result, int(1E7))
            if factorization_result is None :
                return 0.0
            all_e = [e for d, e in factorization_result]
            divisor_number = 1
            for e in all_e :
                divisor_number *= (e + 1)
            
            if divisor_number != self.parameter["K"] :
                return self.rewards["invalid_answer"]

            assert processed_result >= self.parameter["reference_answer"], "processed_result should be greater than or equal to reference_answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((self.parameter["reference_answer"] / processed_result) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                assert self.parameter["reference_answer"] != processed_result
                return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]