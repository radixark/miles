import random
import networkx
from typing import Optional
from Gym.environment import VerifiableEnvironment


class BanyanHeart_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""We use the following process to generate a tree with {N} vertices labeled from 1 to {N}:
- Initially, the tree contains only vertex 1, and its **heart vertex** is also 1.
- At each step, we add a new vertex `i` (2 ≤ i ≤ {N}) and connect it to an existing vertex with an undirected edge. Then, the heart vertex moves one step toward `i` (i.e., it moves to the neighbor that is closer to `i`).
- This process continues until all {N} vertices have been added.

The final tree has the following edges:
{edges}

Can you determine which vertices could be the heart vertex after the process is completed? Output a single line with {N} characters (either `T` or `F`) without separators, where the i-th character is `T` if vertex i can be the heart vertex, and `F` otherwise."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(intersection/union)^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the BanyanHeart_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 4, "N should be greater than or equal to 4"

        edges = self.parameter["edges"] = []
        permutations = list(range(1, N + 1))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v))
        random.shuffle(edges)

        for u, v in edges :
            assert 1 <= u < v <= N
        assert len(edges) == len(set(edges)) == N - 1

        tree = networkx.Graph()
        tree.add_edges_from(edges)
        assert networkx.is_tree(tree)


        # Build adjacency list dynamically
        adjacency = [[] for _ in range(N + 1)]
        for u, v in edges:
            adjacency[u].append(v)
            adjacency[v].append(u)

        # Arrays (1..N); index 0 acts as a dummy node
        dep = [0] * (N + 1)
        siz = [0] * (N + 1)
        hson = [0] * (N + 1)
        hson2 = [0] * (N + 1)
        f = [0] * (N + 1)
        ans = [False] * (N + 1)

        # cmp function: return the index with larger siz
        def cmp(x, y):
            return x if siz[x] > siz[y] else y

        # Iterative dfs1: compute dep, siz, hson, hson2, f
        stack = [(1, 0, 0)]  # (u, parent, state) state 0=enter, 1=exit
        dep[0] = 0
        while stack:
            u, fa, state = stack.pop()
            if state == 0:
                dep[u] = dep[fa] + 1
                stack.append((u, fa, 1))
                for v in adjacency[u]:
                    if v == fa:
                        continue
                    stack.append((v, u, 0))
            else:
                # post-order processing
                s = 1
                h1 = 0
                h2 = 0
                for v in adjacency[u]:
                    if v == fa:
                        continue
                    s += siz[v]
                    if siz[v] > siz[h1]:
                        h2 = h1
                        h1 = v
                    elif siz[v] > siz[h2]:
                        h2 = v
                siz[u] = s
                hson[u] = h1
                hson2[u] = h2

                if f[h1] <= (siz[u] - 1 - siz[h1]):
                    fv = (siz[u] - 1) % 2
                else:
                    fv = f[h1] - (siz[u] - 1 - siz[h1])
                f[u] = fv + 1

        # Iterative dfs2: compute ans
        stack = [(1, 0, 0)]  # (u, parent, h)
        while stack:
            u, fa, h = stack.pop()
            tmp = cmp(hson[u], h)
            if f[tmp] <= N - dep[u] - siz[tmp]:
                ans[u] = ((N & 1) == (dep[u] & 1))
            for v in adjacency[u]:
                if v == fa:
                    continue
                if v == hson[u]:
                    h_child = cmp(hson2[u], h)
                else:
                    h_child = cmp(hson[u], h)
                stack.append((v, u, h_child))

        self.parameter["reference_answer"] = "".join("T" if ans[i] else "F" for i in range(1, N + 1))
        assert "T" in self.parameter["reference_answer"], "At least one vertex should be able to be the heart vertex"
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            answer = answer.strip()
            if not(len(answer) == self.parameter["N"] and all(c in "TF" for c in answer)) :
                return None
            return answer
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            intersection = sum((a == "T" and b == "T") for a, b in zip(processed_result, self.parameter["reference_answer"]))
            union = sum((a == "T" or b == "T") for a, b in zip(processed_result, self.parameter["reference_answer"]))
            assert intersection <= union, "intersection should not exceed union"
            
            if self.rewards["rewarding_strategy"] == "(intersection/union)^beta" :
                return ((intersection / union) ** self.rewards["rewarding_beta"]) * self.rewards["rewarding_weight"]
            elif self.rewards["rewarding_strategy"] == "intersection=union" :
                return self.rewards["rewarding_weight"] * (intersection == union)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]