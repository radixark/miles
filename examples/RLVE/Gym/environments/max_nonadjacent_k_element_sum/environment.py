import heapq
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Max_NonAdjacent_KElementSum_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an array A of {N} positive integers:
{array}

Please select **exactly {K}** indices i1, ..., i{K}, such that:
- No two selected indices are adjacent (i.e., there does not exist any i and i + 1 such that both i and i + 1 are selected).
- The sum A[i1] + ... + A[i{K}] is maximized.

**Output Format:** A single line containing the {K} selected indices in any order, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Max_NonAdjacent_KElementSum_Environment instance.
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

        K = self.parameter["K"] = random.randint(2, N // 2)

        self.parameter["A"] = [random.randint(1, N) for _ in range(N)]


        vals = self.parameter["A"].copy()
        # Compute a dynamic "infinite" sentinel value larger than any sum of values
        INF = sum(abs(v) for v in vals) + 1

        # Initialize arrays (0..N+1) for doubly-linked list
        L = list(range(N+2))
        R = list(range(N+2))
        val = [0] * (N + 2)
        vis = [False] * (N + 2)

        # Fill in values, set up neighbors
        for i, v in enumerate(vals, start=1):
            val[i] = v
            L[i] = i - 1
            R[i] = i + 1

        # Sentinels at 0 and N+1
        val[0] = val[N+1] = -INF
        L[0] = 0
        R[0] = 1
        L[N+1] = N
        R[N+1] = N+1

        # Build max-heap via negatives
        heap = []
        for i in range(1, N + 1):
            heapq.heappush(heap, (-val[i], i))

        ans = 0
        # Perform K merges
        for _ in range(K):
            # Pop until we find an unvisited position
            while True:
                neg_x, pos = heap[0]
                if vis[pos]:
                    heapq.heappop(heap)
                else:
                    break
            x = -neg_x
            heapq.heappop(heap)

            ans += x
            l = L[pos]
            r = R[pos]
            # Bypass l and r
            L[pos] = L[l]
            R[pos] = R[r]
            R[L[pos]] = pos
            L[R[pos]] = pos

            # Mark removed neighbors
            vis[l] = True
            vis[r] = True

            # Update current value and re-push
            val[pos] = val[l] + val[r] - x
            heapq.heappush(heap, (-val[pos], pos))

        self.parameter["gold_answer"] = ans
        assert ans > 0


    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K = self.parameter["K"],
            array = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
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

            if len(processed_result) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            if len(processed_result) != len(set(processed_result)) :
                return self.rewards["invalid_solution"]
            if not all(0 <= i < self.parameter["N"] for i in processed_result) :
                return self.rewards["invalid_solution"]
            processed_result.sort()
            if any(processed_result[i] + 1 == processed_result[i + 1] for i in range(len(processed_result) - 1)) :
                return self.rewards["invalid_solution"]
            
            answer, gold = sum(self.parameter["A"][i] for i in processed_result), self.parameter["gold_answer"]
            assert answer <= gold, "answer should be less than or equal to gold"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]