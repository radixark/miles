import random
from collections import deque
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinConversionToCycleCost_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3651
    prompt_template = \
r"""You are given a **directed graph** with {N} vertices, labeled from 0 to {N_minus_1}. Each vertex `i` has exactly one incoming edge from vertex `A[i]` to vertex `i`. The initial array A is given as: {A}

You are allowed to modify A[i] to any other vertex `j` (0 ≤ j < {N}) at a cost of C[i]. The cost array is given as: {C}

Your goal is to make the entire graph form a **single directed cycle** (i.e., each vertex has exactly one incoming and one outgoing edge, and all vertices are reachable from each other). Try your best to **minimize the total cost** of modifications.

**Output Format:** A single line containing the final A[0], A[1], ..., A[{N_minus_1}], separated by **spaces**."""


    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the MinConversionToCycleCost_Environment instance.
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
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        A = self.parameter["A"] = []
        for i in range(N) :
            while True:
                a = random.randint(0, N - 1)
                if a != i :
                    A.append(a)
                    break
        assert len(A) == N, "A should have exactly N elements"
        
        C = self.parameter["C"] = [random.randint(1, N) for _ in range(N)]


        # Compute indegree h for each node in the functional graph
        h = [0] * N
        for v in A:
            h[v] += 1

        # Queue of nodes with indegree 0 (tree leaves)
        q = deque(i for i in range(N) if h[i] == 0)

        # f[v] will track the best "incoming" cost seen so far for v
        f = [0] * N
        ans = 0

        # Special case: if there are no leaves, the graph is pure cycles
        # Check if it's exactly one big cycle
        vis = [False] * N
        if not q:
            count = 0
            j = 0
            while not vis[j]:
                vis[j] = True
                count += 1
                j = A[j]
            if count == N:
                self.parameter["gold_answer"] = ans
                return

        # Peel off the trees attached to cycles, from leaves inward
        while q:
            x = q.popleft()
            y = A[x]
            if f[y]:
                # We already have one candidate edge into y; choose the cheaper
                ans += min(f[y], C[x])
                # Keep the more expensive as the "best so far" for future comparisons
                f[y] = max(f[y], C[x])
            else:
                # First edge into y
                f[y] = C[x]
            h[y] -= 1
            if h[y] == 0:
                q.append(y)

        # Now only the cycles remain (h[i] > 0 for nodes in cycles)
        for i in range(N):
            if h[i] > 0:
                # Gather all edges in this cycle
                diffs = []
                j = i
                # Walk the cycle, breaking h[] as we go
                while h[A[j]] > 0:
                    v = A[j]
                    h[v] = 0
                    ans += f[v]            # pay the best incoming from the attached tree (or 0)
                    diffs.append(f[v] - C[j])
                    j = v
                # To make this cycle strongly connected, we must drop one edge (the max diff)
                diffs.sort()
                ans -= diffs[-1]
                # And if any other diffs are positive, we can save money by replacing more edges
                for d in diffs[:-1]:
                    if d > 0:
                        ans -= d
        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
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

            A = processed_result

            if len(A) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= a < self.parameter["N"] for a in A) :
                return self.rewards["invalid_solution"]

            visited = [False] * self.parameter["N"]
            x = 0
            while True :
                assert 0 <= x < self.parameter["N"]
                if visited[x] :
                    if x == 0 :
                        break
                    else :
                        return self.rewards["unsuccessful_solution"]
                visited[x] = True
                x = A[x]
            if not all(visited) :
                return self.rewards["unsuccessful_solution"]
            
            gold, answer = self.parameter["gold_answer"], sum(Ci * int(OldAi != NewAi) for OldAi, NewAi, Ci in zip(self.parameter["A"], A, self.parameter["C"]))
            assert gold <= answer
            
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