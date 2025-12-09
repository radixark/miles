import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class TreeDynamic_XORZeroPath_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3359
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices labeled from `0` to `{N_minus_1}`.

The tree has the following {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning there is an undirected edge between vertex `u` and vertex `v` with weight `w`:
{edges}

You will remove edges one by one in the following order: {removes}
After removing the first 0, 1, ..., {N_minus_1} edges (in the given order above), please compute the number of **paths** such that the **XOR** of the weights along the path is equal to 0. There are C({N}, 2) paths in total, where C is the binomial coefficient.

**Output Format:** A single line containing {N} integers — the number of such paths at the beginning and after each removal, separated by spaces."""


    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the TreeDynamic_XORZeroPath_Environment instance.
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

        edges = self.parameter["edges"] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, v, random.randint(0, N)))
        random.shuffle(edges)

        for u, v, w in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, w in edges)) == N - 1

        tree = networkx.Graph()
        tree.add_weighted_edges_from(edges)
        assert networkx.is_tree(tree)

        self.parameter["removes"] = removes = list(range(N - 1))
        random.shuffle(removes)


        adjacent_lists = [[] for u in range(N)]
        for u, v, w in edges :
            adjacent_lists[u].append((v, w))
            adjacent_lists[v].append((u, w))

        xor_from_0 = [0] * N
        def DFS(u, parent) :
            for v, w in adjacent_lists[u] :
                if v != parent :
                    xor_from_0[v] = xor_from_0[u] ^ w
                    DFS(v, u)
        xor_from_0[0] = 0
        DFS(0, -1)

        parent, xor2num, nodes_list = list(range(N)), [{xor : 1} for xor in xor_from_0], [[u] for u in range(N)]

        removes = reversed(removes)
        answer = [0]
        for remove in removes :
            answer.append(answer[-1])
            u, v = edges[remove][0], edges[remove][1]
            u, v = parent[u], parent[v]
            if len(nodes_list[u]) < len(nodes_list[v]) :
                u, v = v, u
            nodes_list[u].extend(nodes_list[v])
            for node in nodes_list[v] :
                answer[-1] += xor2num[u].get(xor_from_0[node], 0)
                parent[node] = u
            for node in nodes_list[v] :
                xor2num[u][xor_from_0[node]] = xor2num[u].get(xor_from_0[node], 0) + 1
        answer.reverse()

        self.parameter["gold_answer"] = answer
        self.parameter["reference_answer"] = " ".join(map(str, answer))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("edge {} : ({} {} {})".format(i, u, v, w) for i, (u, v, w) in enumerate(self.parameter["edges"])),
            removes = " ".join(map(str, self.parameter["removes"])),
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
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]