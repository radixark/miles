import math
import random
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment


class BinaryTreeLeafNumExpectation_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3978
    prompt_template = \
r"""We uniformly at random generate a **binary tree** with exactly {N} nodes (all distinct binary trees with {N} nodes are equally likely). Two binary trees are considered identical if and only if:
- both are empty, **OR**
- both are non-empty, and their left subtrees are identical and their right subtrees are identical.

What is the expected number of **leaf** nodes (nodes whose left and right children are both empty) in the generated binary tree? Output the result as `A/B` (do NOT include quotes), where A and B are positive integers separated by a slash `/`."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the BinaryTreeLeafNumExpectation_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 5, "MAX_N should be greater than or equal to 5"

        N = self.parameter["N"] = random.randint(1, MAX_N)

        A, B = N * (N + 1), 2 * (2 * N - 1)
        gcd_AB = math.gcd(A, B)
        A //= gcd_AB
        B //= gcd_AB
        self.parameter["gold_answer"] = dict(A = A, B = B)
        self.parameter["reference_answer"] = "{}/{}".format(A, B)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"])


    def _process(self, answer : Optional[str]) -> Optional[Tuple[int, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                A, B = map(int, map(str.strip, answer.split('/')))
                return (A, B)
            except :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            A, B = processed_result
            if not (A > 0 and B > 0) :
                return self.rewards["wrong_format"]
            gold_A, gold_B = self.parameter["gold_answer"]["A"], self.parameter["gold_answer"]["B"]
            gcd_AB = math.gcd(A, B)
            A //= gcd_AB
            B //= gcd_AB
            if (A, B) == (gold_A, gold_B) :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]