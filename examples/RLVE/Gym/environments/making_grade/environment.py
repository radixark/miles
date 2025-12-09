import heapq
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MakingGrade_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2893
    prompt_template = \
r"""There is an array A of length {N}: {A}
Please find an array B of length {N} such that B is either monotonically non-decreasing or monotonically non-increasing. Can you make the sum of |A[i] - B[i]| for all 1 ≤ i ≤ {N} as small as possible? Output B[1], B[2], ..., B[{N}] in one line, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MakingGrade_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }
    
    def non_decreasing(self, A : List[int]) -> bool :
        return all(a <= b for a, b in zip(A, A[1 :]))
    
    def non_increasing(self, A : List[int]) -> bool :
        return all(a >= b for a, b in zip(A, A[1 :]))

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 4, "N should be greater than or equal to 4"

        while True :
            A = self.parameter["A"] = [random.randint(0, N * N) for _ in range(N)]
            if not (self.non_decreasing(A) or self.non_increasing(A)) :
                break

        def cost_nondecreasing(seq):
            # Max-heap via negatives
            heap = []
            ans = 0
            for a in seq:
                # push a
                heapq.heappush(heap, -a)
                top = -heap[0]  # current maximum in the heap
                if a < top:
                    # add the decrease needed and replace the largest with a
                    ans += top - a
                    heapq.heapreplace(heap, -a)
            return ans
        
        # Cost to make nondecreasing (as per the provided C++ logic)
        inc_cost = cost_nondecreasing(A)
        # Cost to make nonincreasing is the same as making (-A) nondecreasing
        dec_cost = cost_nondecreasing([-x for x in A])

        self.parameter["gold_answer"] = min(inc_cost, dec_cost)
        assert self.parameter["gold_answer"] > 0, "gold_answer should be greater than 0"
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            A = ", ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"], start = 1)),
        )


    def _process(self, answer : Optional[str]) -> Optional[List[int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            B = processed_result
            if len(B) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not (self.non_decreasing(B) or self.non_increasing(B)) :
                return self.rewards["invalid_solution"]

            gold, answer = self.parameter["gold_answer"], sum(abs(Ai - Bi) for Ai, Bi in zip(self.parameter["A"], B))
            assert 0 < gold <= answer, "gold should be less than or equal to answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]