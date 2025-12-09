import random
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment


class MinXorPair_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Given an array of length {N} (index starting from 0):
{A}

Please find a pair of (i, j) such that 0 <= i < j < {N}, and try your best to minimize the value of (A[i] AND A[j]) XOR (A[i] OR A[j]), where `AND`, `OR`, and `XOR` denote bitwise operations.

Your final answer should be a single line containing the two integers i and j, separated by a space. For example: `0 2` (do **NOT** include quotes or backticks) means i = 0 and j = 2."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinXorPair_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def compute(self, i, j) :
        return (self.parameter["A"][i] & self.parameter["A"][j]) ^ (self.parameter["A"][i] | self.parameter["A"][j])
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        assert "max_bit_length" in self.parameter, "max_bit_length is required in parameter"
        max_bit_length = self.parameter["max_bit_length"]
        assert max_bit_length >= 1, "max_bit_length should be greater than or equal to 1"

        A = self.parameter["A"] = random.sample(range(1 << max_bit_length), N)
        random.shuffle(A)
        
        
        indices = self.parameter["indices"] = list(range(N))
        indices.sort(key = lambda x : A[x])

        i, j, res = indices[0], indices[1], self.compute(indices[0], indices[1])
        for _i, _j in zip(indices, indices[1 :]) :
            _res = self.compute(_i, _j)
            if _res < res :
                i, j, res = _i, _j, _res
        self.parameter["reference_answer"] = "{} {}".format(min(i, j), max(i, j))
        self.parameter["gold_answer"] = res
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = "\n".join("A[{}]={}".format(index, a) for index, a in enumerate(self.parameter["A"])),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[Tuple[int, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                if len(answer_array) != 2 :
                    return None # Invalid answer format
                return answer_array[0], answer_array[1]
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
        
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            i, j = processed_result

            if not (0 <= i < j < self.parameter["N"]) :
                return self.rewards["invalid_solution"]
            gold, answer = self.parameter["gold_answer"], self.compute(i, j)
            assert gold <= answer, "Gold answer should be less than or equal to answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]