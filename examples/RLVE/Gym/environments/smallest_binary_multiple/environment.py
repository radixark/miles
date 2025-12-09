import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SmallestBinaryMultiple_Environment(VerifiableEnvironment) : # https://www.luogu.com.cn/problem/P2841
    prompt_template = r"""Find the **smallest positive integer** B such that the product {A} × B contains **only digits `0` and `1`** in its decimal representation. Output the value of B."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_beta : float = 5.0, rewarding_weight : float = 1.0,
                 **kwargs) :
        """
        Initialize the SmallestBinaryMultiple_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_answer": invalid_answer,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_beta": rewarding_beta,
            "rewarding_weight": rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "MAX_A" in self.parameter, "MAX_A is required in parameter"
        MAX_A = self.parameter["MAX_A"]
        assert MAX_A >= 2, "MAX_A should be greater than or equal to 2"

        A = self.parameter["A"] = random.randint(2, MAX_A)


        def solve() :
            dp = {0: 0}

            cur_value = 1          # 10^k      (a single '1' at the current digit position)
            cur_mod = 1 % A        # (10^k) mod A

            while True:
                # store new states to avoid modifying dp during iteration
                new_states = []

                for remainder, value in dp.items():
                    candidate = value + cur_value     # turn the current digit from 0 to 1
                    new_remainder = (remainder + cur_mod) % A

                    if new_remainder == 0:
                        # candidate is the first multiple of A that uses only 0/1 digits
                        B = candidate // A
                        return B

                    if new_remainder not in dp:       # first time we see this remainder
                        new_states.append((new_remainder, candidate))

                # add the freshly discovered states
                for r, v in new_states:
                    dp[r] = v

                # move to the next more-significant digit
                cur_value *= 10
                cur_mod = (cur_mod * 10) % A
        self.parameter["reference_answer"] = solve()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(A = self.parameter["A"])


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

            AB = self.parameter["A"] * processed_result
            while AB :
                if AB % 10 not in (0, 1) :
                    return self.rewards["invalid_answer"]
                AB //= 10
            
            assert self.parameter["reference_answer"] <= processed_result, "Reference answer should be less than or equal to the processed result"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((self.parameter["reference_answer"] / processed_result) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                assert self.parameter["reference_answer"] != processed_result
                return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]