import heapq
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class PalembangBridges_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3644
    prompt_template = \
r"""You are given two arrays S and T, each of length {N}, provided as: {S_and_T}

Your task is to choose {K} integers P[j] (1 <= j <= {K}) such that the following total cost is minimized: for each i from 1 to {N}, compute min(|P[j] - S[i]| + |P[j] - T[i]|) over all 1 ≤ j ≤ {K}, and take the sum over all i. Output the {K} integers P[j] (1 <= j <= {K}) in a single line, separated by spaces."""


    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the PalembangBridges_Environment instance.
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
        assert N >= 3, "N should be greater than or equal to 3"

        K = self.parameter["K"] = random.randint(1, 2)

        S = self.parameter["S"] = [random.randint(0, N) for _ in range(N)]
        T = self.parameter["T"] = [random.randint(0, N) for _ in range(N)]


        cross_pairs = []

        # process each resident
        for s, t in zip(S, T):
            cross_pairs.append((s, t))

        m = len(cross_pairs)

        class Solver:
            def __init__(self):
                # max-heap for lower half (store negatives), min-heap for upper half
                self.left = []
                self.right = []
                self.left_sum = 0
                self.right_sum = 0

            def insert(self, a: int):
                # initial insert
                if not self.left:
                    heapq.heappush(self.left, -a)
                    self.left_sum += a
                else:
                    median = -self.left[0]
                    if a <= median:
                        heapq.heappush(self.left, -a)
                        self.left_sum += a
                    else:
                        heapq.heappush(self.right, a)
                        self.right_sum += a

                # rebalance so that left has (total+1)//2 elements
                total = len(self.left) + len(self.right)
                target = (total + 1) // 2

                # if left too big, move top of left → right
                while len(self.left) > target:
                    v = -heapq.heappop(self.left)
                    self.left_sum -= v
                    heapq.heappush(self.right, v)
                    self.right_sum += v

                # if left too small, move top of right → left
                while len(self.left) < target:
                    v = heapq.heappop(self.right)
                    self.right_sum -= v
                    heapq.heappush(self.left, -v)
                    self.left_sum += v

            def query(self) -> int:
                """
                Returns the minimum sum of absolute deviations from the optimal pivot,
                which is the sum of |x_i - median| over all inserted x_i.
                """
                if not self.left:
                    return 0
                total = len(self.left) + len(self.right)
                cnt = (total + 1) // 2
                median = -self.left[0]
                # cost = sum_{i in left} (median - x_i) + sum_{j in right} (x_j - median)
                # = cnt*median - left_sum + right_sum - (total-cnt)*median
                return cnt * median - self.left_sum + self.right_sum - (total - cnt) * median

        if K == 1:
            # one bridge: place it at the median of all endpoints
            solver = Solver()
            for a, b in cross_pairs:
                solver.insert(a)
                solver.insert(b)
            self.parameter["gold_answer"] = solver.query()

        else:
            # two bridges: split the pairs into two contiguous groups by sorting on a+b
            cross_pairs.sort(key=lambda x: x[0] + x[1])

            # pre[i]: best cost for first i pairs with one bridge
            pre = [0] * (m + 1)
            solver1 = Solver()
            for i in range(m):
                a, b = cross_pairs[i]
                solver1.insert(a)
                solver1.insert(b)
                pre[i + 1] = solver1.query()

            # suf[i]: best cost for pairs i..m-1 with one bridge
            suf = [0] * (m + 2)
            solver2 = Solver()
            for i in range(m - 1, -1, -1):
                a, b = cross_pairs[i]
                solver2.insert(a)
                solver2.insert(b)
                suf[i + 1] = solver2.query()

            # try all ways to split into two groups
            best = pre[0] + suf[1]
            for i in range(m + 1):
                cost = pre[i] + suf[i + 1]
                if cost < best:
                    best = cost

            self.parameter["gold_answer"] = best
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K = self.parameter["K"],
            S_and_T = "; ".join("S[{}]={}, T[{}]={}".format(i, Si, i, Ti) for i, (Si, Ti) in enumerate(zip(self.parameter["S"], self.parameter["T"]), start = 1)),
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
            
            answer, gold = sum(min(abs(p - s) + abs(p - t) for p in processed_result) for s, t in zip(self.parameter["S"], self.parameter["T"])), self.parameter["gold_answer"]
            assert 0 <= gold <= answer, "gold_answer should be non-negative and less than or equal to answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert gold == 0, "If answer is 0, gold should also be 0"
                    return self.rewards["rewarding_weight"] * 1.0
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]