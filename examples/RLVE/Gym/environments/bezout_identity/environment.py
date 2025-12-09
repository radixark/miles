import math
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class BezoutIdentity_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an array of length {N}, denoted as A[1], ..., A[{N}]. Please find **integers** X[1], ..., X[{N}] such that the value of S = A[1] * X[1] + ... + A[{N}] * X[{N}] satisfies the condition: **S > 0**. Try your best to **minimize the value of S** while meeting this condition.

A: {A}

**Output Format:** Output a single line containing X[1], ..., X[{N}], separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the BezoutIdentity_Environment instance.
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
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "MAX_A" in self.parameter, "MAX_A is required in parameter"
        MAX_A = self.parameter["MAX_A"]
        assert MAX_A >= 2, "MAX_A should be greater than or equal to 2"

        self.parameter["A"] = A = []
        for _ in range(N) :
            picked_a, best_counting = None, -1
            for try_step in range(1024) :
                current_a = random.randint(2, MAX_A)
                counting = sum(int(math.gcd(current_a, _a) > 1) for _a in A)
                if counting > best_counting :
                    best_counting, picked_a = counting, current_a
                if best_counting == len(A) :
                    break
            if random.random() < 0.5 :
                picked_a = -picked_a
            A.append(picked_a)
        random.shuffle(A)
        assert len(A) == N, "The length of A should be equal to N"


        def exgcd(a, b):
            """
            Returns (g, x, y) such that
                g = gcd(a, b)
                a*x + b*y = g
            Ensures g >= 0.
            """
            if b == 0:
                return (abs(a), 1 if a >= 0 else -1, 0)
            g, x1, y1 = exgcd(b, a % b)
            # b*x1 + (a%b)*y1 = g
            # a%b = a - (a//b)*b
            x = y1
            y = x1 - (a // b) * y1
            return (g, x, y)

        # initialize with A[0]
        g = abs(A[0])
        X = [0] * N
        X[0] = 1 if A[0] >= 0 else -1
        
        # incorporate each A[i]
        for i in range(1, N):
            ai = A[i]
            g2, u, v = exgcd(g, ai)
            # scale previous coefficients by u
            for j in range(i):
                X[j] *= u
            # coefficient for A[i] is v
            X[i] = v
            g = g2
        
        S = sum(x * a for x, a in zip(X, A))
        assert S == g
        assert S > 0, "The sum S must be greater than 0"
        self.parameter["reference_answer"] = " ".join(map(str, X))
        self.parameter["gold_answer"] = S
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = ", ".join(map(str, self.parameter["A"])),
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

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            S = sum(x * a for x, a in zip(processed_result, self.parameter["A"]))
            if S <= 0 :
                return self.rewards["invalid_solution"]
            assert self.parameter["gold_answer"] <= S, "The computed sum S must be greater than or equal to the gold answer"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((self.parameter["gold_answer"] / S) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == S)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]