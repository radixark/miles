import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class SetSplitting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Define the full set `S` as all {N} integers from `0` to `{N_minus_1}`.

Your task is to partition `S` into two **disjoint subsets** `S1` and `S2` such that:
- `S1 ∪ S2 = S` and `S1 ∩ S2 = ∅`
- For each of the following {M} subsets (each a subset of `S`), the subset is **not fully contained** in either `S1` or `S2`. That is, each subset must contain **at least one element from S1** and **at least one element from S2`.

The list of {M} subsets is as follows:
{Sets}

**Output Format:** Your final answer should be a single line containing the elements of `S1`, separated by spaces. (Subset `S2` is implicitly defined as `S \ S1`.)"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SetCover_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 2, "M should be greater than or equal to 2"

        S1 = random.sample(range(N), k = random.randint(1, N - 1))
        S2 = list(set(range(N)) - set(S1))
        assert S1 and S2, "S1 and S2 must be non-empty"
        self.parameter["reference_answer"] = " ".join(map(str, S1))

        Sets = self.parameter["Sets"] = []
        for _ in range(M) :
            subset = random.sample(S1, k = random.randint(1, len(S1))) + random.sample(S2, k = random.randint(1, len(S2)))
            random.shuffle(subset)
            Sets.append(subset)
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            M = len(self.parameter["Sets"]),
            Sets = "\n".join("{ " + ", ".join(map(str, subset)) + " }" for subset in self.parameter["Sets"]),
        )


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

            if not all(0 <= x < self.parameter["N"] for x in processed_result) :
                return self.rewards["invalid_solution"]
            if len(set(processed_result)) != len(processed_result) :
                return self.rewards["invalid_solution"]
            
            S1 = set(processed_result)
            S2 = set(range(self.parameter["N"])) - S1
            
            satisfied = sum(int(not (set(subset) <= S1 or set(subset) <= S2)) for subset in self.parameter["Sets"])
            assert sum(int(not (set(subset) <= S1 or set(subset) <= S2)) for subset in self.parameter["Sets"]) == sum(int(bool(set(subset) & S1) and bool(set(subset) & S2)) for subset in self.parameter["Sets"])
            assert satisfied <= self.parameter["M"], "satisfied should not exceed M"
            
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / self.parameter["M"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == self.parameter["M"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]