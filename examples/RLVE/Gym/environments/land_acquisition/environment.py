import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class LandAcquisition_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2900
    prompt_template = \
r"""There are {N} items, and the i-th item has two attributes W[i] and L[i]. The arrays W and L are given as follows:
{W_and_L}

Partition all items into an arbitrary number of **disjoint non-empty sets**. For each set S, its cost is defined as: cost(S) = max(W[i] for i ∈ S) × max(L[i] for i ∈ S)
Can you make the total cost, which is the sum of costs of all sets, as small as possible? Output M lines, where M is the number of sets in your partition - each line should contain the indices of the items in one set (separated by spaces)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the LandAcquisition_Environment instance.
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

        while True :
            W = self.parameter["W"] = [random.randint(1, N * N) for _ in range(N)]
            L = self.parameter["L"] = [random.randint(1, N * N) for _ in range(N)]


            Land = [None] * (N + 1)
            for i, (w, l) in enumerate(zip(W, L), start = 1) :
                Land[i] = (w, l)

            # Sort by width asc, then length asc
            Land_sorted = sorted(Land[1:], key=lambda x: (x[0], x[1]))

            # Remove dominated rectangles: keep strictly decreasing lengths (stack)
            stack = []
            for w, l in Land_sorted:
                while stack and l > stack[-1][1]:
                    stack.pop()
                stack.append((w, l))

            cnt = len(stack)

            # 1-indexed 'needto' with a sentinel at the end so needto[i+1] is safe
            needto = [None] + stack + [(0, 0)]

            # DP with Convex Hull Trick (no magic INF; we compute valid states directly)
            dp = [None] * (cnt + 1)
            dp[0] = 0

            # Monotone queue of candidate j indices; q[0] = 0 as in the C++ global zero-init
            q = [0]
            head = 0

            for i in range(1, cnt + 1):
                # Move head forward while the next candidate is better
                while head < len(q) - 1:
                    j0 = q[head]
                    j1 = q[head + 1]
                    lhs = dp[j0] - dp[j1]
                    rhs = -needto[i][0] * (needto[j0 + 1][1] - needto[j1 + 1][1])
                    if lhs >= rhs:
                        head += 1
                    else:
                        break

                j = q[head]
                dp[i] = dp[j] + needto[i][0] * needto[j + 1][1]

                # Maintain convexity of the hull
                while head < len(q) - 1:
                    j_last = q[-1]
                    j_prev = q[-2]
                    left = (dp[j_last] - dp[j_prev]) * (needto[i + 1][1] - needto[j_prev + 1][1])
                    right = (dp[i] - dp[j_prev]) * (needto[j_last + 1][1] - needto[j_prev + 1][1])
                    if left <= right:
                        q.pop()
                    else:
                        break

                q.append(i)

            self.parameter["gold_answer"] = dp[cnt]
            assert self.parameter["gold_answer"] > 0
            
            item_indices = list(range(N))
            item_indices.sort(key = lambda i : (W[i], L[i]))
            naive_answer = min(max(W) * max(L), sum(Wi * Li for Wi, Li in zip(W, L)))
            for i in range(N - 1) :
                group_1 = max(W[j] for j in item_indices[: i + 1]) * max(L[j] for j in item_indices[: i + 1])
                group_2 = max(W[j] for j in item_indices[i + 1 :]) * max(L[j] for j in item_indices[i + 1:])
                naive_answer = min(naive_answer, group_1 + group_2)
            assert self.parameter["gold_answer"] <= naive_answer
            if self.parameter["gold_answer"] < naive_answer :
                break

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            W_and_L = "\n".join("W[{}]={} L[{}]={}".format(i, Wi, i, Li) for i, (Wi, Li) in enumerate(zip(self.parameter["W"], self.parameter["L"]), start = 1)),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[List[List[int]]] :
        if answer is not None :
            answer = answer.strip()
            try :
                groups = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        groups.append(list(map(int, line.split())))
                return groups
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if sum(len(group) for group in processed_result) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if set(item for group in processed_result for item in group) != set(range(1, self.parameter["N"] + 1)) :
                return self.rewards["invalid_solution"]
            
            gold, answer = self.parameter["gold_answer"], sum(max(self.parameter["W"][i - 1] for i in group) * max(self.parameter["L"][i - 1] for i in group) for group in processed_result)
            assert gold <= answer, f"Gold answer {gold} is greater than computed answer {answer}"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]