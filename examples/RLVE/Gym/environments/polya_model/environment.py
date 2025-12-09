import random
from fractions import Fraction
from typing import Optional, Dict
from Gym.environment import VerifiableEnvironment


class PolyaModel_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4204
    prompt_template = \
r"""You have a bag with balls of {T} colors. The initial counts are: {color2num}
Process:
- At each step (starting from step 1), draw one ball uniformly at random from the bag.
- Return the drawn ball to the bag, then add {D} additional balls of the **same color** to the bag.

Given the following event(s): {events}
What's the probability that **all** specified events occur? Output a single fraction `p/q` (without quotes), where `p` and `q` are coprime non-negative integers; if the probability is 0, output `0/1`; if it is 1, output `1/1`."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the PolyaModel_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_T_N" in self.parameter, "MAX_T_N is required in parameter"
        MAX_T_N = self.parameter["MAX_T_N"]
        assert MAX_T_N >= 2, "MAX_T_N should be greater than or equal to 2"

        T = self.parameter["T"] = random.randint(2, MAX_T_N)

        color2num = self.parameter["color2num"] = [random.randint(1, MAX_T_N) for color in range(T)]
        D = self.parameter["D"] = random.randint(1, MAX_T_N)

        N = random.randint(1, MAX_T_N)
        events = self.parameter["events"] = [(step, random.randint(1, T)) for step in sorted(random.sample(range(1, N + 1), random.randint(1, N)))]
        

        ar = color2num.copy()
        s = sum(ar)
        ans = Fraction(1)
        for x, y in events:
            y -= 1
            ans *= Fraction(ar[y], s)
            ar[y] += D
            s += D
        self.parameter["reference_answer"] = str(ans)
        self.parameter["gold_answer"] = dict(numerator = int(ans.numerator), denominator = int(ans.denominator))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            T = self.parameter["T"],
            color2num = ", ".join("{} balls of color {}".format(num, color) for color, num in enumerate(self.parameter["color2num"], start = 1)),
            D = self.parameter["D"],
            events = ", ".join("at step {} the drawn ball is of color {}".format(step, color) for step,color in self.parameter["events"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[Dict[str, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                a, b = map(int, answer.split('/'))
                return dict(numerator = a, denominator = b)
            except :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self._process(output)
        if processed_result is not None :
            if processed_result == self.parameter["gold_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]