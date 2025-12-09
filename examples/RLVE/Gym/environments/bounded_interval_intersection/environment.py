import heapq
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class BoundedIntervalIntersection_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""An interval [l, r]'s length is defined as r - l. The length of an **empty intersection** is considered to be 0. The **intersection** of a set of intervals is the range covered by all of them simultaneously.

You are given {N} intervals:
{intervals}

Please count how many **non-empty subsets** (i.e., from the total of 2^{N} - 1 non-empty subsets) have an intersection of length **greater than or equal to {K}**.

**Output Format:** Your final answer should be a single integer — the number of non-empty subsets of intervals whose intersection has length at least {K}."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the BoundedIntervalIntersection_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        self.parameter["intervals"] = []
        for i in range(N) :
            l = random.randint(0, N)
            r = random.randint(l, N)
            self.parameter["intervals"].append((l, r))
        
        K = self.parameter["K"] = random.randint(1, max(min(r - l for l, r in self.parameter["intervals"]), 1))
        assert K > 0, "K should be greater than 0"
        

        intervals = self.parameter["intervals"].copy()
        intervals.sort(key = lambda x : x[0])

        Q = []
        ans = 0

        for l, r in intervals :
            if r - l >= K :
                while Q and Q[0] < l + K :
                    heapq.heappop(Q)
                ans += pow(2, len(Q))
                heapq.heappush(Q, r)
        
        self.parameter["reference_answer"] = ans
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K = self.parameter["K"],
            intervals = "\n".join(["[{}, {}]".format(l, r) for l, r in self.parameter["intervals"]]),
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
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                if self.parameter["reference_answer"] == 0 :
                    return self.rewards["rewarding_weight"] * (processed_result == 0)
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]