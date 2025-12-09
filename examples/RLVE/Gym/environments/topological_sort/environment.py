import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class TopologicalSort_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Please find a permutation of `0` to `{N_minus_1}` ({N} integers in total) such that the following conditions are satisfied:
{before_conditions}

**Output Format:** Your final answer should be a single line containing the permutation `p(0), p(1), ..., p({N_minus_1})`, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the TopologicalSort_Environment instance.
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
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 2"

        permutation = list(range(N))
        random.shuffle(permutation)
        self.parameter["reference_answer"] = " ".join(map(str, permutation))

        before_conditions = self.parameter["before_conditions"] = []
        for i in range(N) :
            if i == 0 :
                continue
            for j in random.sample(range(i), random.randint(1, i)) :
                before_conditions.append((permutation[j], permutation[i]))
        random.shuffle(before_conditions)
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            before_conditions = "\n".join("{} must be before {}".format(j, i) for j, i in self.parameter["before_conditions"]),
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

            permutation = processed_result
            if len(permutation) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if len(set(permutation)) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= i < self.parameter["N"] for i in permutation) :
                return self.rewards["invalid_solution"]
            
            positions = [None] * self.parameter["N"]
            for i, p in enumerate(permutation) :
                positions[p] = i

            satisfied = sum(positions[j] < positions[i] for j, i in self.parameter["before_conditions"])
            assert satisfied <= len(self.parameter["before_conditions"]), "satisfied should not exceed the number of conditions"
            
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / len(self.parameter["before_conditions"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied/all" :
                return self.rewards["rewarding_weight"] * (satisfied == len(self.parameter["before_conditions"]))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]