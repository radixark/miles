import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MinNonsubstring_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a string A = `{A}`

Your task is to find a string B such that:
(1) B consists only of the characters `a` and `b`.
(2) B is **NOT** a (contiguous) substring of A.
(3) Among all strings satisfying (1) and (2), B has the **minimum possible length**.
(4) Among all strings satisfying (1), (2), and (3), B is **lexicographically smallest**. There is exactly one such string B.

**Output Format:** Your final answer should be a single string B."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MinNonsubstring_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 1, "N should be greater than or equal to 1"

        a_probability = random.random()

        A = self.parameter["A"] = "".join("ab"[random.random() < a_probability] for _ in range(N))


        length = 1
        while True :
            found = False
            for B_mask in range(1 << length) :
                B = "".join("ab"[(B_mask >> i) & 1] for i in range(length - 1, -1, -1))
                if B not in A :
                    self.parameter["reference_answer"] = B
                    found = True
                    break
            if found :
                break
            length += 1
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(A = self.parameter["A"])


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            answer = answer.strip()
            return answer
        else :
            return None
    
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if not all(c in "ab" for c in processed_result) :
                return self.rewards["invalid_answer"]
            
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]