import heapq
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Max_TreeConstrainedPermutation_Weight_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4437
    prompt_template = \
r"""You are given an array W of length {N}: {W}

Please find a permutation P of 1 to {N} such that the following conditions are satisfied:
{conditions}

Try your best to **maximize** the sum of W[P[i]] × i for all i from 1 to {N}.

**Output Format:** Output one line containing the permutation P[1], ..., P[{N}], separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Max_TreeConstrainedPermutation_Weight_Environment instance.
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

        self.parameter["W"] = [random.randint(1, N) for _ in range(N)]
        self.parameter["A"] = [random.randint(0, i) for i in range(N)]


        class Da:
            __slots__ = ('u', 'sz', 'w')
            def __init__(self, u, sz, w):
                self.u = u
                self.sz = sz
                self.w = w
            def __lt__(self, other):
                # Compare by average weight: want to pop the smallest average first
                return self.w * other.sz < other.w * self.sz

        def compute():
            parent = [0] + self.parameter["A"]
            weights_input = [0] + self.parameter["W"]

            # Build children lists for the reversed graph
            children = [[] for _ in range(N + 1)]
            for i in range(1, N + 1):
                children[parent[i]].append(i)

            # DFS from 0 to detect reachable nodes and cycles
            visited = [False] * (N + 1)
            stack = [0]
            visited[0] = True
            cnt = 1
            while stack:
                u = stack.pop()
                for v in children[u]:
                    if visited[v]:
                        print(-1)
                        return
                    visited[v] = True
                    cnt += 1
                    stack.append(v)
            # If not all nodes reachable (including 0), there's a cycle
            if cnt <= N:
                print(-1)
                return

            # Initialize DSU, sizes, and weights
            dsu = list(range(N + 1))
            size = [1] * (N + 1)
            weight = [0] * (N + 1)
            for i in range(1, N + 1):
                weight[i] = weights_input[i]

            def find(u):
                # Path-compression find
                while dsu[u] != u:
                    dsu[u] = dsu[dsu[u]]
                    u = dsu[u]
                return u

            # Build priority queue of initial nodes
            heap = []
            for i in range(1, N + 1):
                heapq.heappush(heap, Da(i, 1, weight[i]))

            ans = 0
            # Merge components in order of increasing average weight
            while heap:
                s = heapq.heappop(heap)
                u = find(s.u)
                # Skip stale entries
                if size[u] != s.sz:
                    continue
                # Merge u into its parent component
                p = find(parent[u])
                ans += weight[u] * size[p]
                weight[p] += weight[u]
                size[p] += size[u]
                dsu[u] = p
                # Push updated parent component if it's not the root 0
                if p != 0:
                    heapq.heappush(heap, Da(p, size[p], weight[p]))

            # Output the result
            assert ans > 0
            return ans

        self.parameter["gold_answer"] = compute()


    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            W = " ".join("W[{}]={}".format(i + 1, Wi) for i, Wi in enumerate(self.parameter["W"])),
            conditions = "\n".join(
                "- The element {} has no constraint.".format(i + 1)
                if Ai == 0
                else "- The element {} must come before element {}.".format(Ai, i + 1)
                for i, Ai in enumerate(self.parameter["A"])
            ),
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

            P = processed_result
            N = self.parameter["N"]
            if len(P) != N :
                return self.rewards["invalid_solution"]
            if set(P) != set(range(1, N + 1)) :
                return self.rewards["invalid_solution"]
            
            positions = [None] * (N + 1)
            for i, Pi in enumerate(P) :
                positions[Pi] = i
            for i, Ai in enumerate(self.parameter["A"]) :
                if Ai != 0 and positions[Ai] >= positions[i + 1] :
                    return self.rewards["invalid_solution"]
            
            answer, gold = sum(self.parameter["W"][Pi - 1] * (i + 1) for i, Pi in enumerate(P)), self.parameter["gold_answer"]
            assert answer <= gold, "answer should be less than or equal to gold"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]