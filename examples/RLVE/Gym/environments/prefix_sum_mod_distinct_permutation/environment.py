import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class PrefixSumMODDistinctPermutation_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Please find a permutation of the numbers from 1 to {N} such that all {N} prefix sums (i.e., the sum of the first i numbers for all i from 1 to {N}) are **distinct modulo {N}**. Output the permutation as {N} integers (in order) in one line, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the PrefixSumMODDistinctPermutation_Environment instance.
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
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 3, "MAX_N should be greater than or equal to 3"

        while True :
            N = self.parameter["N"] = random.randint(3, MAX_N)

            if N % 2 == 1:
                continue
            else:
                # Build the “zig‑zag” even‑N construction
                perm = [N]
                for i in range(1, N):
                    if i % 2 == 1:
                        perm.append(i)
                    else:
                        perm.append(N - i)
                self.parameter["reference_answer"] = " ".join(map(str, perm))
                break
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"])
    

    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if set(processed_result) != set(range(1, self.parameter["N"] + 1)) :
                return self.rewards["invalid_solution"]
            
            existing, prefix_sum = [False] * self.parameter["N"], 0
            for x in processed_result :
                prefix_sum = (prefix_sum + x) % self.parameter["N"]
                assert 0 <= prefix_sum < self.parameter["N"], "prefix_sum should be in the range [0, N)"
                existing[prefix_sum] = True
            satisfied = sum(existing)
            assert 1 <= satisfied <= self.parameter["N"], "satisfied should be less than or equal to N"
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == self.parameter["N"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]