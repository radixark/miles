import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class LampChanging_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3937
    prompt_template = \
r"""There are {N} lamps arranged in a circle, labeled clockwise from 1 to {N}. At each next moment, the state of each lamp depends on its current state and the state of the next lamp in the clockwise direction:
- If the two lamps have the same state, then the lamp will be OFF in the next moment.
- If the two lamps have different states, then the lamp will be ON in the next moment.

The initial moment is time 0, and the initial states of all lamps are: {situations}
What's the state of lamp {K} at time {T} (Output either ON or OFF)?"""
    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the LampChanging_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_T" in self.parameter, "MAX_N_T is required in parameter"
        MAX_N_T = self.parameter["MAX_N_T"]
        assert MAX_N_T >= 3, "MAX_N_T should be greater than or equal to 3"
        self.parameter["reference_answer"] = random.choice(["ON", "OFF"])

        while True :
            N = self.parameter["N"] = random.randint(3, MAX_N_T)
            ON_probability = random.random()
            B = self.parameter["B"] = [1 if random.random() < ON_probability else 0 for _ in range(N)]
            T = self.parameter["T"] = random.randint(2, MAX_N_T)
            K = self.parameter["K"] = random.randint(1, N)

            res = 0
            for i in range(T + 1):
                if (T & i) == i:  # C(T, i) % 2 == 1  <=>  i is a submask of T
                    res ^= B[(i + K - 1) % N]  # XOR is addition mod 2
            if self.parameter["reference_answer"] == ("OFF", "ON")[res] :
                break
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            T = self.parameter["T"],
            K = self.parameter["K"],
            situations = "; ".join("Lamp {} is {}".format(i, "ON" if Bi else "OFF") for i, Bi in enumerate(self.parameter["B"], start = 1)),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result not in ("ON", "OFF") :
                return self.rewards["invalid_answer"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]