import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class SubsetSum_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an array `A` of length `{N}`, indexed from `0` to `{N_minus_1}`:
{A}

Please find a subset of **distinct indices** `i1, i2, ..., ik` such that: the sum `A[i1] + A[i2] + ... + A[ik]` is exactly equal to {target}.

**Output Format:** Your final answer should be a single line containing the selected indices `i1, i2, ..., ik`, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the SubsetSum_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        A = self.parameter["A"] = [random.randint(1, N) for _ in range(N)]
        
        indices = random.sample(range(N), k = random.randint(2, N - 1))
        self.parameter["target"] = sum(A[index] for index in indices)
        self.parameter["reference_answer"] = " ".join(map(str, indices))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
            target = self.parameter["target"],
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

            if not all(0 <= i < self.parameter["N"] for i in processed_result) :
                return self.rewards["invalid_solution"]
            if len(processed_result) != len(set(processed_result)) :
                return self.rewards["invalid_solution"]
            
            if sum(self.parameter["A"][i] for i in processed_result) == self.parameter["target"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]