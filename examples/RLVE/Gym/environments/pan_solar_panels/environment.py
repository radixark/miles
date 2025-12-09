import math
import random
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment


class PanSolarPanels_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3579
    prompt_template = \
r"""Output two integers X and Y (separated by a space), such that:
- {A} ≤ X ≤ {B}
- {C} ≤ Y ≤ {D}
- gcd(X, Y) is maximized (where gcd stands for greatest common divisor)"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the PanSolarPanels_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_A_B_C_D" in self.parameter, "MAX_A_B_C_D is required in parameter"
        MAX_A_B_C_D = self.parameter["MAX_A_B_C_D"]
        assert MAX_A_B_C_D >= 4, "MAX_A_B_C_D should be greater than or equal to 4"

        while True :
            numbers = [random.randint(1, MAX_A_B_C_D) for _ in range(4)]
            numbers.sort()
            A, B, C, D = numbers
            if A <= B < C <= D :
                break
        if random.random() < 0.5 :
            A, B, C, D = C, D, A, B
        self.parameter["A"], self.parameter["B"], self.parameter["C"], self.parameter["D"] = A, B, C, D


        def solve(A, B, C, D):
            res = 1
            m = min(B, D)
            p = 1
            while p <= m:
                # floor-divisions for current p
                t1 = B // p
                t2 = D // p
                # find the largest r such that B//x == t1 and D//x == t2 for all x in [p..r]
                r1 = B // t1
                r2 = D // t2
                r = min(r1, r2)
                # check if multiples of r lie within the intervals
                x = (B // r) * r
                y = (D // r) * r
                if x >= A and y >= C:
                    res = r
                # jump to the next segment
                p = r + 1
            return res
        self.parameter["gold_answer"] = solve(A, B, C, D)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(A = self.parameter["A"], B = self.parameter["B"], C = self.parameter["C"], D = self.parameter["D"])
    

    def _process(self, answer : Optional[str]) -> Optional[Tuple[int, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                X, Y = map(int, answer.split())
                return X, Y
            except :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            X, Y = processed_result
            if not (self.parameter["A"] <= X <= self.parameter["B"] and self.parameter["C"] <= Y <= self.parameter["D"]) :
                return self.rewards["invalid_solution"]
            
            gold, answer = self.parameter["gold_answer"], math.gcd(X, Y)
            assert 0 < answer <= gold, "answer should be less than or equal to gold"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]