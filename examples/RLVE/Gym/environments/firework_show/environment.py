import heapq
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class FireworkShow_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3642
    prompt_template = \
r"""You are given a **tree** with {N} vertices labeled from `1` to `{N}`, where vertex `1` is the **root**. Each vertex (except the root) has a parent `p`, and the edge connecting the vertex to its parent has length `w`. The list of (parent, weight) pairs for each non-root vertex is given as:
{parents}

Note that these vertices are leaf nodes (i.e., vertices with no children): {leaves}
You can reduce the length of any edge. Specifically, you can change an edge's length `w` to any integer `w'` such that `0 ≤ w'`; the cost of changing an edge from `w` to `w'` is abs(w - w'). You need to make the sum of the edge lengths on the path from each leaf node to the root `1` equal — in other words, all leaf-to-root paths should have the same total length. Output the **minimum total cost** required to achieve this."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the FirworkShow_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        parents = self.parameter["parents"] = []
        is_leaf = [None] + [True] * N
        for i in range(2, N + 1) :
            parent= random.randint(1, i - 1)
            parents.append((parent, random.randint(1, N)))
            is_leaf[parent] = False
        self.parameter["leaves"] = [i for i in range(2, N + 1) if is_leaf[i]]


        # adjacency and weights
        children = [[] for _ in range(N + 1)]
        w = [0] * (N + 1)
        res = 0

        for i in range(2, N + 1):
            p, c = parents[i - 2]
            children[p].append(i)
            w[i] = c
            res += c

        def dfs(x):
            assert 1 <= x <= N, "Node index out of bounds"
            # we store values as negatives so that heapq (a min-heap)
            # can pop the "largest" original value first
            heap = []
            for y in children[x]:
                child_heap = dfs(y)
                # small‐to‐large merge
                if len(heap) < len(child_heap):
                    heap, child_heap = child_heap, heap
                for val in child_heap:
                    heapq.heappush(heap, val)

            l = r = 0
            if not is_leaf[x]:
                d = len(children[x])
                assert len(children[x]) > 0, "There should be at least one child for non-leaf nodes"
                # remove the d-1 largest values
                for _ in range(d - 1):
                    if heap:
                        heapq.heappop(heap)
                # then pop the next two largest into r and l
                if heap:
                    r = -heapq.heappop(heap)
                if heap:
                    l = -heapq.heappop(heap)
            else :
                assert len(children[x]) == 0, "Leaf nodes should not have children"

            # push back with the current edge weight
            heapq.heappush(heap, -(l + w[x]))
            heapq.heappush(heap, -(r + w[x]))
            return heap

        root_heap = dfs(1)

        # Discard the single largest, then subtract every remaining value
        if root_heap:
            heapq.heappop(root_heap)
        while root_heap:
            res -= -heapq.heappop(root_heap)

        self.parameter["reference_answer"] = res
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            parents = "\n".join("Vertex {}: ({}, {})".format(i, p, w) for i, (p, w) in enumerate(self.parameter["parents"], start = 2)),
            leaves = ", ".join(map(str, self.parameter["leaves"])),
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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]