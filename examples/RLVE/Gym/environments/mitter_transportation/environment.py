import math
import random
from typing import Optional
from collections import deque, Counter
from Gym.environment import VerifiableEnvironment


class MitterTransportation_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3237
    prompt_template = \
r"""You are given a tree with {N} vertices labeled from `0` to `{N_minus_1}`, where vertex `0` is the root. For each vertex `i` (i > 0), its parent is `p[i]`. The parent array is: {parent}
Each vertex `i` initially has a value `A[i]`. The array A is: {A}

You are allowed to modify the values of any vertices. Your goal is to ensure that:
- For every vertex `i` with children, all of its children must have the **same** value; the value of `A[i]` must be equal to the **sum** of the values of its children.
- Every vertex's value should be a **positive real number**.

Please compute the **minimum number of vertices** whose `A[i]` value you must modify to satisfy these rules. Output a single integer — the minimum number of modified vertices."""


    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MitterTransportation_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        parents = self.parameter["parents"] = [None]
        adj = [[] for s in range(N)]
        for i in range(1, N) :
            parent = random.randint(0, i - 1)
            parents.append(parent)
            adj[parent].append(i)
        

        # BFS to root the tree in vertex 0 (former city 1)
        parent = [-1] * N
        child_cnt = [0] * N
        order = []                                    # parents appear before children

        q = deque([0])
        parent[0] = 0
        while q:
            v = q.popleft()
            order.append(v)
            for nxt in adj[v]:
                if nxt == parent[v]:
                    assert False, "Tree should not have cycles"
                    continue
                parent[nxt] = v
                child_cnt[v] += 1
                q.append(nxt)

        # step 2 – compute the multiplicative factors (triple-hashed)
        k1 = [0] * N
        k1[0] = 1                     # factor(root) = 1

        for v in order[1:]:                           # skip the root itself
            p = parent[v]
            k1[v] = child_cnt[p] * k1[p]
        
        A = self.parameter["A"] = [None] * self.parameter["N"]

        no_change_vertices = random.sample(range(N), random.randint(1, N - 1))
        lcm = 1
        for i in no_change_vertices :
            assert k1[i] > 0, "k1[i] should be positive"
            lcm = math.lcm(lcm, k1[i])
        maxA = 1
        for i in no_change_vertices :
            A[i] = lcm // k1[i]
            maxA = max(maxA, A[i])
        for i in range(N) :
            if A[i] is None :
                A[i] = random.randint(1, maxA)

        # step 3 – count identical triplets
        counter = Counter(
            k1[i] * A[i]
            for i in range(N)
        )

        # step 4 – result
        max_group = max(counter.values())             # largest unchanged set
        assert max_group >= len(no_change_vertices), "max_group should be at least the size of no_change_vertices"
        self.parameter["reference_answer"] = N - max_group
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            parent = " ".join("p[{}]={}".format(i, self.parameter["parents"][i]) for i in range(1, N)),
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
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