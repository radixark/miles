import random
from collections import deque
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class WarehouseConstruction_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2120
    prompt_template = \
r"""You are given {N} factories arranged from top to bottom along a mountain, indexed from 0 to {N_minus_1}. Factory 0 is at the top and factory {N_minus_1} is at the bottom.

Each factory has
- Distance from factory 0: {D}
- Number of products: {P}
- Cost to build a warehouse at that factory: {C}

You can choose to build warehouses at any subset of factories.
- A warehouse can store any number of products.
- If a factory does not build a warehouse, all its products must be sent **downhill** to a factory with a warehouse (i.e., to a factory with a higher index). Transporting one product over one unit of distance costs 1.
- The total cost is the sum of warehouse construction costs and product transportation costs. Try your best to minimize the total cost.

**Output Format:** Output a single line containing the indices of the factories where warehouses should be built, separated by spaces (in any order)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the WarehouseConstruction_Environment instance.
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

        D = random.sample(range(1, 2 * N + 1), N - 1)
        D.sort()
        self.parameter["D"] = D = [0] + D
        assert len(D) == N, "X should have length N"
        assert all(di < di1 for di, di1 in zip(D, D[1 :])), "D should be strictly increasing"

        self.parameter["P"] = P = [random.randint(0, N) for _ in range(N)]
        self.parameter["C"] = C = [random.randint(1, N * 2) for _ in range(N)]


        Q = [0] * (N+1)
        R = [0] * (N+1)
        for i in range(1, N+1):
            Q[i] = Q[i-1] + P[i-1]
            R[i] = R[i-1] + D[i-1] * P[i-1]

        # f[i] will hold the DP value corresponding to “having built a warehouse at factory i-1”
        f = [0] * (N+1)

        # Mirror the C++ helpers:
        def decx(idx):
            return Q[idx]
        def decy(idx):
            return f[idx] + R[idx]
        def maked(i, u):
            # exactly f[u] + D[i-1]*(Q[i]-Q[u]) - (R[i]-R[u]) + C[i-1]
            return f[u] + D[i-1] * (Q[i] - Q[u]) - (R[i] - R[u]) + C[i-1]

        # We'll keep a deque of candidate u-indices, with the left end = oldest,
        # right end = newest, just like the C++ circular queue.
        dq = deque([0])

        for i in range(1, N+1):
            # 1) Pop from the left (oldest) while the next‐oldest is better at x = D[i-1]:
            while len(dq) >= 2:
                u1, u2 = dq[0], dq[1]
                if decy(u2) - decy(u1) <= D[i-1] * (decx(u2) - decx(u1)):
                    dq.popleft()
                else:
                    break

            # 2) Use the best u = dq[0] to compute f[i]:
            u = dq[0]
            f[i] = maked(i, u)

            # 3) Now pop from the right (newest) while the new line i makes it obsolete:
            while len(dq) >= 2:
                u1, u2 = dq[-1], dq[-2]
                if (decy(u1) - decy(u2)) * (decx(i) - decx(u1)) \
                >= (decy(i) - decy(u1)) * (decx(u1) - decx(u2)):
                    dq.pop()
                else:
                    break

            # 4) Add the new candidate i:
            dq.append(i)

        # At the end we want the minimum f[x] among the last non-empty factory:
        ans = f[N]
        x = N
        # if the very last factory has P=0, we can skip it
        while x > 0 and P[x-1] == 0:
            x -= 1
            ans = min(ans, f[x])

        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            D = " ".join("D[{}]={}".format(i, Di) for i, Di in enumerate(self.parameter["D"])),
            P = " ".join("P[{}]={}".format(i, Pi) for i, Pi in enumerate(self.parameter["P"])),
            C = " ".join("C[{}]={}".format(i, Ci) for i, Ci in enumerate(self.parameter["C"])),
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

            answer = 0
            built = [False] * self.parameter["N"]
            for idx in processed_result :
                if 0 <= idx < self.parameter["N"] :
                    built[idx] = True
                    answer += self.parameter["C"][idx]
                else :
                    return self.rewards["invalid_solution"]
            nearest_warehouse = None
            for i in range(self.parameter["N"] - 1, -1, -1) :
                if built[i] :
                    nearest_warehouse = i
                if self.parameter["P"][i] :
                    if nearest_warehouse is None :
                        return self.rewards["invalid_solution"]
                    answer += self.parameter["P"][i] * (self.parameter["D"][nearest_warehouse] - self.parameter["D"][i])
            
            gold = self.parameter["gold_answer"]
            assert gold <= answer, "gold_answer should be less than or equal to answer"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert gold == 0, "If answer is 0, gold should also be 0"
                    return self.rewards["rewarding_weight"]
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]