import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MinimalCyclicShift_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Here is a binary string S of length {N}: {S}
You may perform any number of cyclic shifts on S, where one shift moves the leftmost character to the rightmost position. Output the lexicographically smallest string obtainable after any number of shifts."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the MinimalCyclicShift_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 4, "N should be greater than or equal to 4"

        one_probability = random.random()
        S = self.parameter["S"] = "".join(str(int(random.random() < one_probability)) for _ in range(N))


        i, j, k = 0, 1, 0
        while i < N and j < N and k < N:
            c1 = S[(i + k) % N]
            c2 = S[(j + k) % N]
            if c1 == c2:
                k += 1
            else:
                if c1 > c2:
                    i += k + 1
                else:
                    j += k + 1
                if i == j:
                    i += 1
                k = 0

        start = min(i, j)
        ans = ''.join(S[(start + t) % N] for t in range(N))
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], S = self.parameter["S"])


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            answer = answer.strip()
            return answer
        else :
            return None
    
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if len(processed_result) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            if not all(c in "01" for c in processed_result) :
                return self.rewards["wrong_format"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["reference_answer"], processed_result)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]