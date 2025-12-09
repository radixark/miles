import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class GcdLcmCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1029
    prompt_template = \
r"""Find the number of pairs of positive integers `(P, Q)` that satisfy the following conditions:

1. Both `P` and `Q` are **positive integers**.
2. The **greatest common divisor (GCD)** of `P` and `Q` is **{gcd}**.
3. The **least common multiple (LCM)** of `P` and `Q` is **{lcm}**.

Your task is to determine how many such pairs `(P, Q)` satisfy **all** of the above conditions.

Output Format:
Your final answer should be a single integer — the number of valid `(P, Q)` pairs.
Example: `4` (do **NOT** include the backticks or quotes); this means there are 4 valid pairs that meet the criteria.
"""

    def __init__(self,
                 wrong_format : float = -1.0, not_power_2 : float = 0.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 answer_being_0_probability : float = 0.01,
                 **kwargs) :
        """
        Initialize the GcdLcmCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "not_power_2" : not_power_2,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
        self.answer_being_0_probability = answer_being_0_probability
    

    def _generate(self) -> None :
        assert "MAX_LCM" in self.parameter, "MAX_LCM is required in parameter"
        MAX_LCM = self.parameter["MAX_LCM"]
        assert MAX_LCM >= 3, "MAX_LCM should be greater than or equal to 3"

        if random.random() < self.answer_being_0_probability :
            while True :
                LCM = self.parameter["LCM"] = random.randint(1, MAX_LCM)
                GCD = self.parameter["GCD"] = random.randint(1, LCM)
                if LCM % GCD != 0 :
                    break
        else :
            LCM = self.parameter["LCM"] = random.randint(1, MAX_LCM)
            def all_factors(n) :
                factors = set()
                for i in range(1, int(n**0.5) + 1) :
                    if n % i == 0 :
                        factors.add(i)
                        factors.add(n // i)
                return factors
            factors = all_factors(LCM)
            GCD = self.parameter["GCD"] = random.choice(list(set(factors)))
        
        def solve(gcd, lcm) :
            def prime_factorization(n) :
                prime2count = {}
                for x in range(2, int(n**0.5) + 1) :
                    while n % x == 0 :
                        prime2count[x] = prime2count.get(x, 0) + 1
                        n //= x
                if n > 1 :
                    prime2count[n] = prime2count.get(n, 0) + 1
                return prime2count

            gcd, lcm = prime_factorization(gcd), prime_factorization(lcm)

            counting = 1
            for p in set(gcd.keys()) | set(lcm.keys()) :
                x_count, y_count = gcd.get(p, 0), lcm.get(p, 0)
                if x_count > y_count :
                    counting = 0
                    break
                if x_count == y_count :
                    counting *= 1
                else : # x_count < y_count
                    counting *= 2
            return counting
        self.parameter["reference_answer"] = solve(self.parameter["GCD"], self.parameter["LCM"])
        assert (self.parameter["reference_answer"] == 0) == (self.parameter["LCM"] % self.parameter["GCD"] != 0)
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(gcd = self.parameter["GCD"], lcm = self.parameter["LCM"])
    

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
            else :
                def power_2(n) :
                    while n and n % 2 == 0 :
                        n //= 2
                    return n <= 1
                if not power_2(processed_result) :
                    return self.rewards["not_power_2"]
                else :
                    if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                        a, b = self.parameter["reference_answer"], processed_result
                        return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
                    elif self.rewards["rewarding_strategy"] == "gold=answer" :
                        return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
                    else :
                        raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]