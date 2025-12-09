import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Knapsack_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given {N} items labeled from `0` to `{N_minus_1}`. Each item has a **weight** W[i] and a **value** V[i]:
{W_and_V}

Please select a subset of **distinct items** i_1, i_2, ..., i_k such that:
- The total weight W[i_1] + W[i_2] + ... + W[i_k] is **less than or equal to** {W_max}, and
- Try your best to maximize the total value V[i_1] + V[i_2] + ... + V[i_k].

**Output Format:** Your final answer should be a single line containing the indices of the selected items, separated by spaces.
Example: `0 {N_minus_1}` (do **NOT** include quotes or backticks); this means you selected items `0` and `{N_minus_1}`."""

    def __init__(self,
                 value_range_multiple : int = 2,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Knapsack_Environment instance.
        """
        super().__init__(**kwargs)
        self.value_range_multiple = value_range_multiple

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
        assert N >= 2, "N should be greater than or equal to 2"

        W = self.parameter["W"] = [random.randint(1, N) for Wi in range(N)]
        V = self.parameter["V"] = [random.randint(1, Wi * self.value_range_multiple) for Wi in W]
        W_max = self.parameter["W_max"] = random.randint(min(W), sum(W))


        F = [0] * (W_max + 1)
        Sum_W = 0
        for Wi, Vi in zip(W, V) :
            Sum_W += Wi
            for w in range(W_max, Wi - 1, -1) :
                F[w] = max(F[w], F[w - Wi] + Vi)
        self.parameter["gold_answer"] = F[W_max]
        assert F[W_max] > 0
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            W_and_V = "\n".join("W[{}]={} V[{}]={}".format(i, self.parameter["W"][i], i, self.parameter["V"][i]) for i in range(N)),
            W_max = self.parameter["W_max"],
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

            if len(processed_result) != len(set(processed_result)) :
                return self.rewards["invalid_solution"]
            if not all(0 <= i < self.parameter["N"] for i in processed_result) :
                return self.rewards["invalid_solution"]
            if sum(self.parameter["W"][i] for i in processed_result) > self.parameter["W_max"] :
                return self.rewards["invalid_solution"]
            
            answer, gold = sum(self.parameter["V"][i] for i in processed_result), self.parameter["gold_answer"]
            assert answer <= gold, "answer should be less than or equal to gold"
            
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]