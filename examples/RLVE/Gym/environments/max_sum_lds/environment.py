import random
from bisect import bisect_left
from typing import Optional, List, Tuple
from Gym.environment import VerifiableEnvironment


class MaxSumLDS_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3971
    prompt_template = \
r"""Given a permutation of numbers from 1 to {N}, denoted as P[1], P[2], ..., P[{N}], define:
- A[0] = 0. For 1 ≤ i ≤ {N}, A[i] = max(A[j]) + 1 such that: (i) 0 ≤ j ≤ i - 1, and (ii) j = 0 **or** P[j] < P[i].
- B[{N} + 1] = 0. For {N} ≥ i ≥ 1, B[i] = max(B[j]) + 1 such that: (i) i + 1 ≤ j ≤ {N} + 1, and (ii) j = {N} + 1 **or** P[j] < P[i].

You are given the array A: {A}
Find a permutation P such that this A is obtained, and **maximize** the value of: B[1] + B[2] + ... + B[{N}]. Output P[1], P[2], ..., P[{N}] in one line, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MaxSumLDS_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "unsuccessful_solution" : unsuccessful_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def get_A_B(self, P : List[int]) -> Tuple[List[int], List[int]] :
        assert len(P) == self.parameter["N"] + 1
        assert P[0] is None, "P[0] should be None"

        A = [0] * (self.parameter["N"] + 2)
        for i in range(1, self.parameter["N"] + 1) :
            A[i] = max(A[j] for j in range(i) if j == 0 or P[j] < P[i]) + 1
        A[self.parameter["N"] + 1] = None

        B = [0] * (self.parameter["N"] + 2)
        for i in range(self.parameter["N"], 1 - 1, -1) :
            B[i] = max(B[j] for j in range(i + 1, self.parameter["N"] + 1 + 1) if j == self.parameter["N"] + 1 or P[j] < P[i]) + 1
        B[0] = None

        return A, B


    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        P = list(range(1, N + 1))
        random.shuffle(P)
        P = [None] + P

        A, B = self.get_A_B(P)
        self.parameter["A"] = A[: -1]

        B = B[1 : -1]
        assert len(B) == N, "B should have length N"
        sumB = sum(B)
        

        A = A[1 : -1]
        assert len(A) == N, "A should have length N"
        # Build the adjacency list (nodes 0..N, with 0 as a dummy root)
        adj = [[] for _ in range(N + 1)]
        last_pos = [0] * (N + 1)  # last_pos[k] = last index i with LIS length k seen so far

        for i, x in enumerate(A, start=1):
            parent = last_pos[x - 1]
            adj[parent].append(i)
            adj[i].append(parent)
            last_pos[x] = i

        # Match C++ head-insert neighbor order by reversing adjacency lists
        for nbrs in adj:
            nbrs.reverse()

        # Iterative DFS to get preorder numbers dfn[0..N]
        dfn = [0] * (N + 1)
        cnt = 0
        stack = [(0, -1, 0)]  # (node, parent, next-neighbor-index)
        while stack:
            u, p, idx = stack.pop()
            if idx == 0:
                cnt += 1
                dfn[u] = cnt
            if idx < len(adj[u]):
                v = adj[u][idx]
                stack.append((u, p, idx + 1))
                if v != p:
                    stack.append((v, u, 0))

        # Shift dfn[1..N] down by 1 (ignore dfn[0])
        for i in range(1, N + 1):
            dfn[i] -= 1

        # Build sequence B: B[i] = dfn[N - i] for i = 0..N-1 (equivalent to b[i]=dfn[n-i+1] in 1-based)
        B = [dfn[pos] for pos in range(N, 0, -1)]

        # Compute sum of LIS lengths over B (strictly increasing), using patience sorting with bisect_left
        tails = []
        ans = 0
        for v in B:
            pos = bisect_left(tails, v)
            if pos == len(tails):
                tails.append(v)
            else:
                tails[pos] = v
            ans += pos + 1

        assert 0 < sumB <= ans, "Sum of B should be less than or equal to the answer"
        self.parameter["gold_answer"] = ans


    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = ", ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
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
            if set(processed_result) != set(range(1, self.parameter["N"] + 1)) :
                return self.rewards["invalid_solution"]

            P = [None] + processed_result
            A, B = self.get_A_B(P)
            A = A[: -1]
            if A != self.parameter["A"] :
                return self.rewards["unsuccessful_solution"]
            
            B = B[1 : -1]
            answer, gold = sum(B), self.parameter["gold_answer"]
            assert answer <= gold, "answer should be less than or equal to gold"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]