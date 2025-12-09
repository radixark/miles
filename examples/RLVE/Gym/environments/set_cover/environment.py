import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class SetCover_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given {N} items labeled from 0 to {N_minus_1}, and {M} sets labeled from 0 to {M_minus_1}. Each set is a subset of the items:
{sets}

Your task is to select a collection of sets such that every item is covered **by exactly one** of the selected sets.

**Output Format:** Your final answer should be a single line containing the indices of the selected sets, separated by spaces. Example: `0 {M_minus_1}` (do **NOT** include quotes or backticks); this means you selected sets 0 and {M_minus_1} to cover all items exactly once."""

    def __init__(self,
                 MAX_M_multiple : int = 2,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(covered/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SetCover_Environment instance.
        """
        super().__init__(**kwargs)

        self.MAX_M_multiple = MAX_M_multiple

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
        assert N >= 3, "N should be greater than or equal to 3"

        M = random.randint(3, N * self.MAX_M_multiple)
        constructed_M = random.randint(2, M - 1)

        Sets = self.parameter["Sets"] = [[] for m in range(constructed_M)]
        for item in range(N) :
            Sets[random.randint(0, constructed_M - 1)].append(item)
        for m in range(M - constructed_M) :
            existence_probability = random.random()
            Sets.append([item for item in range(N) if random.random() < existence_probability])
        Sets = [(Set, index < constructed_M) for index, Set in enumerate(Sets) if len(Set) > 0]
        random.shuffle(Sets)

        self.parameter["reference_answer"] = " ".join(str(index) for index in range(len(Sets)) if Sets[index][-1])
        self.parameter["Sets"] = [Set for Set, _ in Sets]
    
    def _prompt_generate(self) -> str :
        N, M = self.parameter["N"], len(self.parameter["Sets"])
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            M = M,
            M_minus_1 = M - 1,
            sets = "\n".join("Set {}: ".format(index) + "{ " + ", ".join(map(str, Set)) + " }" for index, Set in enumerate(self.parameter["Sets"])),
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

            Set_indices = set(processed_result)
            union = set()
            for index in Set_indices :
                if not (0 <= index < len(self.parameter["Sets"])) :
                    return self.rewards["invalid_solution"]
                current = set(self.parameter["Sets"][index])
                if union & current :
                    return self.rewards["invalid_solution"]
                union |= current
            
            assert len(union) <= self.parameter["N"], "union should be less than or equal to N"
            
            if self.rewards["rewarding_strategy"] == "(covered/all)^beta" :
                return self.rewards["rewarding_weight"] * ((len(union) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "covered=all" :
                return self.rewards["rewarding_weight"] * (len(union) == self.parameter["N"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]