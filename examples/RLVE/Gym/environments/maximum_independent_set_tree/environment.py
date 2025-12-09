import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Maximum_IndependentSet_Tree_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1352
    prompt_template = \
r"""You are given a **tree** (i.e., a connected undirected graph with no cycles) with {N} vertices, labeled from `0` to `{N_minus_1}`.

The tree contains the following {N} - 1 = {N_minus_1} undirected edges. Each edge is represented as a tuple `(u, v)`, meaning there is an undirected edge **connecting vertex `u` to vertex `v`**:
{edges}

Each vertex has a weight, given as a list `R` of length {N}, where `R[i]` is the weight of vertex `i`. The weights are as follows:
{R}

Your task is to select a set of distinct vertices `x_1, x_2, ..., x_k` (you determine `k`), such that **no two selected vertices are adjacent**.
Your goal is to **maximize the total weight**: R[x_1] + R[x_2] + ... + R[x_k].

**Output Format:**
Your final answer should be a single line containing the selected vertices in **any order**, separated by **spaces**.
Example: `0 1 {N_minus_1}` (do **NOT** include the backticks or quotes); this means k = 3, with selected vertices x_1 = 0, x_2 = 1, and x_3 = {N_minus_1}.
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the Maximum_IndependentSet_Tree_Environment instance.
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
        childrens = [[] for u in range(N)]

        permutations = list(range(N))
        random.shuffle(permutations)
        root = permutations[0]
        for index, child in enumerate(permutations) :
            if index == 0 :
                continue
            parent = random.choice(permutations[: index])
            childrens[parent].append(child)
            u, v = min(parent, child), max(parent, child)
            edges.append((u, v))

        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)) == N - 1

        tree = networkx.Graph()
        tree.add_edges_from(edges)
        assert networkx.is_tree(tree)

        self.parameter["R"] = [random.randint(1, N) for vertex in range(N)]


        dpF = [None] * N
        def dp(u) :
            dpF[u] = [0, self.parameter["R"][u]]
            for child in childrens[u] :
                dp(child)
                dpF[u][0] += max(dpF[child])
                dpF[u][1] += dpF[child][0]
        dp(root)
        self.parameter["reference_weight"] = max(dpF[root])

        picked = []
        def Pick(u, pick) :
            if pick :
                picked.append(u)
            for child in childrens[u] :
                if pick :
                    Pick(child, False)
                else :
                    Pick(child, bool(dpF[child][0] < dpF[child][1]))
        Pick(root, dpF[root][0] < dpF[root][1])

        self.parameter["reference_answer"] = " ".join(map(str, picked))
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
            R = "\n".join("R[{}] = {}".format(i, self.parameter["R"][i]) for i in range(N)),
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

            picked = processed_result
            if len(set(picked)) != len(picked) :
                return self.rewards["invalid_solution"]
            if not all((0 <= vertex < self.parameter["N"]) for vertex in picked) :
                return self.rewards["invalid_solution"]
            picked = set(picked)
            for u, v in self.parameter["edges"] :
                if u in picked and v in picked :
                    return self.rewards["invalid_solution"]
            
            answer = sum(self.parameter["R"][u] for u in picked)
            gold = self.parameter["reference_weight"]
            assert answer <= gold

            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise ValueError("Invalid rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]