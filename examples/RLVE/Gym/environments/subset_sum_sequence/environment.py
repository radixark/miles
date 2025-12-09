import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SubsetSumSequence_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1062
    prompt_template = \
r"""Consider all powers of `{K}`, and all **finite sums of distinct powers of `{K}`**.
Collect these numbers and sort them in **increasing order** (starting from index 1) to form a sequence:
`{term_0}, {term_1}, {term_2}, {term_3}, ...`

Your task is to compute the value of the **{N}-th term** in this sequence (1-based indexing), and output it in **decimal (base 10)**.

Output Format:
Your final answer should be a single decimal number to indicate the {N}-th term in the sequence.
Example: `{K}` (do **NOT** include the backticks or quotes).
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = 0.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 2.0,
                **kwargs) :
        """
        Initialize the SubsetSumSequence_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K >= 2, "MAX_K should be greater than or equal to 2"

        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 1, "MAX_N should be greater than or equal to 1"

        N = self.parameter["N"] = random.randint(1, MAX_N)
        K = self.parameter["K"] = random.randint(2, MAX_K)

        Ans = 0
        base = 1
        while N :
            if N & 1 :
                Ans += base
            N //= 2
            base *= K
        self.parameter["reference_answer"] = Ans
    
    def _prompt_generate(self) -> str :
        K = self.parameter["K"]
        term_0 = 1
        term_1 = K
        term_2 = 1 + K
        term_3 = K**2
        return self.prompt_template.format(
            K = K,
            term_0 = term_0,
            term_1 = term_1,
            term_2 = term_2,
            term_3 = term_3,
            N = self.parameter["N"],
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
            if processed_result <= 0 :
                return self.rewards["wrong_format"]

            K = self.parameter["K"]
            def check(num : int) -> bool : # Check if the answer is in base K and contains only 0s and 1s.
                while num :
                    if num % K not in (0, 1) :
                        return False
                    num //= K
                return True
            if not check(processed_result) :
                return self.rewards["invalid_solution"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]