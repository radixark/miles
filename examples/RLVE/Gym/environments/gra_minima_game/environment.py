import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class GraMinimaGame_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3507
    prompt_template = \
r"""There are {N} numbers: {A}
Alice and Bob are playing a game with these numbers. Alice goes first, and they take turns. On each turn, a player may choose any **non-empty subset** of the remaining numbers, add the **minimum** of that subset to their score, and then remove the entire subset from the game. The game ends when there are no numbers left.
Each player plays optimally to maximize **their score minus their opponent's score**. Please compute the final value of (Alice's score − Bob's score)."""

    def __init__(self,
                 wrong_format : float = -1.0, float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the GraMinimaGame_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 1, "N should be greater than or equal to 1"

        A = self.parameter["A"] = [random.randint(1, N * 2) for _ in range(N)]

        A = sorted(A)
        Ans = 0
        for a in A :
            Ans = max(Ans, a - Ans)
        self.parameter["reference_answer"] = Ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = " ".join(map(str, self.parameter["A"])),
        )


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