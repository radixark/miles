import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Multiplication_Environment(VerifiableEnvironment) :
    prompt_templates = (
        "Give me the answer of the following equation: {} * {} = ", # https://github.com/Jiayi-Pan/TinyZero/blob/main/examples/data_preprocess/multiply.py
        "What is the result of {} times {}?",
        "Calculate the product of {} and {}.",
        "What do you get when you multiply {} by {}?",
        "If you multiply {} and {}, what is the answer?",
        "What is {} multiplied by {}?",
        "Find the result of {} times {}.",
        "What is the multiplication of {} and {}?",
        "Compute the product of {} and {}.",
        "What is the answer to {} times {}?",
    ) # This is probably unnecessary, but just in case we need to diversify the prompt templates.
    
    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the Multiplication_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }


    def _generate(self) -> None :
        assert "digit_num" in self.parameter, "digit_num is required in parameter"
        digit_num = self.parameter["digit_num"]
        assert digit_num >= 1, "digit_num should be greater than or equal to 1"

        self.parameter["a"] = random.randint(0, 10 ** digit_num - 1)
        self.parameter["b"] = random.randint(0, 10 ** digit_num - 1)
        self.parameter["reference_answer"] = self.parameter["a"] * self.parameter["b"]

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