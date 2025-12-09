import math
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Cinema_Environment(VerifiableEnvironment) : # Source: https://www.luogu.com.cn/problem/P3330
    prompt_template = \
r"""There are {N} people entering a cinema and {K} numbered seats labeled from 1 to {K}.

Each person, in order from 1 to {N}, independently picks a random integer L from 1 to {K}, uniformly at random.
- If seat L is unoccupied, they take it.
- If it's taken, they try seat L + 1, then L + 2, ..., up to seat {K}, until they find a free seat.
- If all seats from L to {K} are occupied, the person must stand.

Please compute the **probability that all {N} people get a seat** (i.e., no one ends up standing). Output the probability as a reduced fraction `A B`, where A/B is the probability and gcd(A, B) = 1.

**Output Format:** A single line with two integers `A B`, separated by a space — the reduced fraction representing the answer."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the Cinema_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_K" in self.parameter, "MAX_N_K is required in parameter"
        MAX_N_K = self.parameter["MAX_N_K"]
        assert MAX_N_K >= 2, "MAX_N_K should be greater than or equal to 2"

        N = self.parameter["N"] = random.randint(2, MAX_N_K)
        K = self.parameter["K"] = random.randint(N, MAX_N_K)
        assert N <= K, "N should be less than or equal to K"


        ans1 = ((K + 1) ** (N - 1)) * (K - N + 1)
        ans2 = K ** N
        tmp = math.gcd(ans1,ans2)
        ans1 //= tmp
        ans2 //= tmp
        self.parameter["gold_answer"] = (ans1, ans2)
        self.parameter["reference_answer"] = "{} {}".format(ans1, ans2)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], K = self.parameter["K"])


    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                a, b = map(int, answer.split())
                return a, b
            except :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, tuple) and len(processed_result) == 2, "Processed result should be a tuple of two integers"
            if processed_result == self.parameter["gold_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]