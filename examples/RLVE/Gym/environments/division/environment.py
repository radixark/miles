import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class Division_Environment(VerifiableEnvironment) :
    prompt_templates = (
        "What is the result of {} divided by {}? Round down to the nearest integer.",
        "Compute {} divided by {}, rounding down to the nearest whole number.",
        "Find the integer part of {} divided by {}.",
        "Compute {} divided by {}, discarding the remainder.",
        "What is the quotient when {} is divided by {}, using integer division?",
        "If you divide {} by {}, what is the whole number result?",
        "Give me the result of {} divided by {} (rounded down).",
        "How many full times does {} fit into {}?",
        "What do you get when you divide {} by {} and round down?",
        "Determine the integer result of {} divided by {}.",
    ) # This is probably unnecessary, but just in case we need to diversify the prompt templates.

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the Division_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }


    def _generate(self) -> None :
        assert "divisor_digit_num" in self.parameter, "divisor_digit_num is required in parameter"
        divisor_digit_num = self.parameter["divisor_digit_num"]
        assert divisor_digit_num >= 1, "divisor_digit_num should be greater than or equal to 1"

        assert "answer_digit_num" in self.parameter, "answer_digit_num is required in parameter"
        answer_digit_num = self.parameter["answer_digit_num"]
        assert answer_digit_num >= 1, "answer_digit_num should be greater than or equal to 1"

        self.parameter["b"] = random.randint(1, 10 ** divisor_digit_num - 1)
        self.parameter["a"] = self.parameter["b"] * random.randint(0, 10 ** answer_digit_num - 1) + random.randint(0, self.parameter["b"] - 1)
        
        self.parameter["reference_answer"] = self.parameter["a"] // self.parameter["b"]

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