import math
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class LCM_Environment(VerifiableEnvironment) :
    prompt_templates = (
        "Please calculate the least common multiple (LCM) of {} and {}.",
        "What is the least common multiple (LCM) of {} and {}?",
        "Find the least common multiple (LCM) of {} and {}.",
        "Calculate the LCM of {} and {}.",
        "Determine the least common multiple (LCM) of {} and {}.",
        "What is the smallest positive integer that is a multiple of both {} and {}? (This is the LCM.)",
        "What is the least common multiple (LCM) of the numbers {} and {}?",
        "Compute the least common multiple (LCM) of {} and {}.",
        "Find the smallest number that is a multiple of both {} and {}. (This is the LCM.)",
        "What is the least common multiple (LCM) of these two numbers: {} and {}?",
    ) # This is probably unnecessary, but just in case we need to diversify the prompt templates.
    
    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the LCM_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }


    def _generate(self) -> None :
        assert "MAX_a_b" in self.parameter, "MAX_a_b is required in parameter"
        MAX_a_b = self.parameter["MAX_a_b"]
        assert MAX_a_b >= 2, "MAX_a_b should be greater than or equal to 2"

        self.parameter["a"] = random.randint(2, MAX_a_b)
        self.parameter["b"] = random.randint(2, MAX_a_b)
        self.parameter["reference_answer"] = math.lcm(self.parameter["a"], self.parameter["b"])

        self.parameter["prompt_template"] = random.randrange(len(self.prompt_templates))
    
    def _prompt_generate(self) -> str :
        return self.prompt_templates[self.parameter["prompt_template"]].format(self.parameter["a"], self.parameter["b"])


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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]