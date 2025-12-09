import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Fibtrain_Environment(VerifiableEnvironment) : # Source: https://www.luogu.com.cn/problem/P1011
    prompt_template = \
r"""A train departs from its starting station (Station 1) with {A} passengers onboard. There are {N} stations in total, numbered from 1 to {N}.

At Station 2, an equal number of passengers get on and off, so the total number of passengers onboard remains unchanged at {A}.

From Station 3 onward (including Station 3) up to Station {N_minus_1}, the boarding and alighting follow a specific rule:
- The number of **boarding** passengers at each station is the **sum of the number of boarding passengers at the previous two stations**.
- The number of **alighting** passengers at each station is **equal to the number of boarding passengers at the previous station**.

At the final station (Station {N}), **all remaining passengers get off**, and the number of passengers who get off is {M}.

Given this setup, what is the number of passengers **on the train after it departs from Station {X}**?

Output Format:
Your final answer should be a **single integer** on a line by itself, representing the number of passengers onboard **after the train departs from Station {X}**.
"""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the Fibtrain_Environment instance.
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

        N = self.parameter["N"] = random.randint(5, MAX_N)

        assert "MAX_A_B" in self.parameter, "MAX_A_B is required in parameter"
        MAX_A_B = self.parameter["MAX_A_B"]
        assert MAX_A_B >= 1, "MAX_A_B should be greater than or equal to 1"

        A = self.parameter["A"] = random.randint(1, MAX_A_B)
        B = self.parameter["B"] = random.randint(1, MAX_A_B)

        boarding, total = [0] * N, [0] * N
        boarding[1], boarding[2] = A, B
        total[1], total[2] = A, A
        for i in range(3, N) :
            boarding[i] = boarding[i - 1] + boarding[i - 2]
            total[i] = total[i - 1] + boarding[i] - boarding[i - 1]
        self.parameter["M"] = total[N - 1]

        X = self.parameter["X"] = random.randint(3, N - 1)
        self.parameter["reference_answer"] = total[X]

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            A = self.parameter["A"],
            N = self.parameter["N"],
            N_minus_1 = self.parameter["N"] - 1,
            M = self.parameter["M"],
            X = self.parameter["X"],
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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]