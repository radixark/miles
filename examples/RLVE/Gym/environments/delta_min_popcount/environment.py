import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class DeltaMinPopcount_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Define `popcount(x)` as the number of 1s in the binary representation of a non-negative integer `x`. For example, `popcount(5) = 2` because `(5)_10 = (101)_2`.

You are given a binary number `d = ({binary_string})_2` (i.e., the base-2 representation of a decimal integer `d`).
Please compute the **minimum value of** `popcount(n XOR (n + d))` over all non-negative integers `n`, where `XOR` denotes the bitwise exclusive OR operation.

**Output Format:** Your final answer should be a single base-10 integer — the minimum `popcount(n XOR (n + d))` over all `n >= 0`."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the DeltaMinPopcount_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    
    def _generate(self) -> None :
        assert "digit_num" in self.parameter, "digit_num is required in parameter"
        digit_num = self.parameter["digit_num"]
        assert digit_num >= 1, "digit_num should be greater than or equal to 1"

        self.parameter["binary_string"] = "1" + "".join(str(random.randint(0, 1)) for _ in range(digit_num - 1))


        S = self.parameter["binary_string"]
        S = S[::-1]
        S = S + "00"

        cur = ans = 0
        for i in range(len(S) - 1) :
            x = int(S[i])
            if x != cur :
                ans += 1
                if S[i + 1] == "1" :
                    cur = 1
                else :
                    cur = 0

        self.parameter["reference_answer"] = ans
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(binary_string = self.parameter["binary_string"])
    

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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]