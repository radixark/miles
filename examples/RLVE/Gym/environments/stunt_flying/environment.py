import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class StuntFlying_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3918
    prompt_template = \
r"""There are {K} elements labeled from 0 to {K_minus_1}, and each element `x` has an associated value C[x]. C is: {C}
You need to build an array A of length {N}, where each A[i] is one of these elements (i.e., 0 ≤ A[i] < {K} for all 1 ≤ i ≤ {N}). Each position i in A has a value defined as **C[A[i]] × T[i]**, where T[i] is determined as follows:
- If there is no previous index j (0 ≤ j < i) such that A[j] = A[i], then T[i] = 0.
- Otherwise, let j be the largest index (basically, closest to i) such that A[j] = A[i] (0 ≤ j < i), and set T[i] = i - j.

Can you maximize the sum of all values **C[A[i]] × T[i]**? Output A[1], A[2], ..., A[{N}] in order, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the StuntFlying_Environment instance.
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
        assert N >= 4, "N should be greater than or equal to 4"

        K = self.parameter["K"] = random.randint(2, N)
        C = self.parameter["C"] = [random.randint(1, K) for _ in range(K)]


        A = C.copy()
        A.sort(reverse=True)

        ans = 0
        N -= 1
        i = 0
        while N > 0 and i < K:
            ans += N * A[i]
            i += 1
            N -= 2

        assert ans > 0, "ans should be greater than 0"
        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        K = self.parameter["K"]
        return self.prompt_template.format(
            K = K,
            K_minus_1 = K - 1,
            C = "; ".join("C[{}] = {}".format(x, Cx) for x, Cx in enumerate(self.parameter["C"])),
            N = self.parameter["N"],
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
            if not all(0 <= Ai < self.parameter["K"] for Ai in processed_result) :
                return self.rewards["invalid_solution"]

            last = [None] * self.parameter["K"]
            gold, answer = self.parameter["gold_answer"], 0
            for i, Ai, in enumerate(processed_result) :
                T = 0 if last[Ai] is None else i - last[Ai]
                answer += self.parameter["C"][Ai] * T
                last[Ai] = i
            
            assert answer <= gold, "answer should be less than or equal to gold_answer"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]