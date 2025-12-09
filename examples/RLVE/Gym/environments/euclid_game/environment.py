import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class EuclidGame_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1290
    prompt_template = \
r"""Stan and Ollie are playing a game starting with two integers {X} and {Y}. Stan goes first.

On each turn, a player may subtract any **positive multiple** of one integer from the other, as long as the result is **non-negative**. The player who makes one of the numbers become **zero** wins the game.

If both players always play optimally, who will win — Stan or Ollie?

**Output Format:** Your final answer should be a single word: either `Stan` or `Ollie` (do **NOT** include quotes or backticks), indicating the winner."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the EuclidGame_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    
    def _generate(self) -> None :
        assert "MAX_X_Y" in self.parameter, "MAX_X_Y is required in parameter"
        MAX_X_Y = self.parameter["MAX_X_Y"]
        assert MAX_X_Y >= 1, "MAX_X_Y should be greater than or equal to 1"

        self.parameter["reference_answer"] = "Stan" if random.random() < 0.5 else "Ollie"

        while True :
            X = self.parameter["X"] = random.randint(1, MAX_X_Y)
            Y = self.parameter["Y"] = random.randint(1, MAX_X_Y)
            def check(x : int, y : int) -> bool :
                if not y :
                    return False
                if x // y != 1 :
                    return True
                return not check(y, x - y)
            if ("Stan" if check(max(X, Y), min(X, Y)) else "Ollie") == self.parameter["reference_answer"] :
                break
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(X = self.parameter["X"], Y = self.parameter["Y"])


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None
    
    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result not in ("Stan", "Ollie") :
                return self.rewards["invalid_answer"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]