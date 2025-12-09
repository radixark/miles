import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class RecursiveSequenceSumConstruction_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3996
    prompt_template = \
r"""Define a sequence F by:
- F(0) = {F0}
- For every integer n ≥ 1, F(n) = {A} * F(n - 1) + {B}

Output any number of **distinct** positive (F(0) cannot be included) indices n1, n2, ..., nk (k ≥ 1), in one line separated by spaces, such that: F(n1) + F(n2) + ... + F(nk) = {S}."""


    def __init__(self,
                 A_is_1_probability : float = 0.3,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the RecursiveSequenceSumConstruction_Environment instance.
        """
        super().__init__(**kwargs)

        self.A_is_1_probability = A_is_1_probability
        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_F0" in self.parameter, "MAX_F0 is required in parameter"
        MAX_F0 = self.parameter["MAX_F0"]
        assert MAX_F0 >= 1, "MAX_F0 should be greater than or equal to 1"

        assert "MAX_A" in self.parameter, "MAX_A is required in parameter"
        MAX_A = self.parameter["MAX_A"]
        assert MAX_A >= 2, "MAX_A should be greater than or equal to 2"

        assert "MAX_B" in self.parameter, "MAX_B is required in parameter"
        MAX_B = self.parameter["MAX_B"]
        assert MAX_B >= 1, "MAX_B should be greater than or equal to 1"

        F0 = self.parameter["F0"] = random.randint(0, MAX_F0)
        A = self.parameter["A"] = (1 if random.random() < self.A_is_1_probability else random.randint(2, MAX_A))
        B = self.parameter["B"] = random.randint(0, MAX_B)

        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 1, "N should be greater than or equal to 1"

        F = [F0]
        for n in range(1, N + 1) :
            F.append(A * F[n - 1] + B)
        
        self.parameter["reference_answer"] = random.sample(range(1, N + 1), k = random.randint(1, N))
        self.parameter["S"] = sum(F[n] for n in self.parameter["reference_answer"])
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["reference_answer"]))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            F0 = self.parameter["F0"],
            A = self.parameter["A"],
            B = self.parameter["B"],
            S = self.parameter["S"],
        )


    def _process(self, answer : Optional[str]) -> Optional[List[int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                if not answer_array :
                    return None
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) != len(set(processed_result)) :
                return self.rewards["invalid_answer"]
            if not all(n >= 1 for n in processed_result) :
                return self.rewards["invalid_answer"]
            
            S = 0
            N = max(processed_result)
            if N > max(map(int, self.parameter["reference_answer"].split())) * 10 :
                return self.rewards["wrong_answer"]
            processed_result = set(processed_result)
            Fn_minus_1 = self.parameter["F0"]
            for n in range(1, N + 1) :
                Fn = self.parameter["A"] * Fn_minus_1 + self.parameter["B"]
                if S + Fn > self.parameter["S"] :
                    return self.rewards["wrong_answer"]
                if n in processed_result :
                    S += Fn
                Fn_minus_1 = Fn
            if S == self.parameter["S"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]