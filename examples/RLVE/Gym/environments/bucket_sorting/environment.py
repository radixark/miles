import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class BucketSorting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given the following array: {array}

Please find the number that appears **most frequently** in the array. If there are multiple numbers with the same highest frequency, you may output **any one** of them."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the  BucketSoring_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_answer": invalid_answer,
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        assert "MAX" in self.parameter, "MAX is required in parameter"
        MAX = self.parameter["MAX"]
        assert MAX >= 1, "MAX should be greater than or equal to 1"


        self.parameter["array"] = [random.randint(0, MAX) for _ in range(N)]

        self.parameter["value2count"] = {}
        for value in self.parameter["array"] :
            if value not in self.parameter["value2count"] :
                self.parameter["value2count"][value] = 0
            self.parameter["value2count"][value] += 1
        
        self.parameter["reference_answer"] = max(self.parameter["value2count"].items(), key = lambda x : x[1])[0]
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(array = " ".join(map(str, self.parameter["array"])))


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
            if processed_result not in self.parameter["value2count"] :
                return self.rewards["invalid_answer"]
            
            if self.parameter["value2count"][processed_result] == max(self.parameter["value2count"].values()) :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]