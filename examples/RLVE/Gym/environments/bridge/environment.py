import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Bridge_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices labeled from 0 to {N_minus_1}. The graph contains the following undirected edges:
{edges}

Your task is to find all edges (u, v) such that removing the edge (u, v) from the graph would disconnect vertices u and v (which are initially connected).

**Output Format:** Assuming the edges are (u_1, v_1), (u_2, v_2), ..., (u_k, v_k), your final answer should be a single line containing `u_1 v_1 u_2 v_2 ... u_k v_k`, where the vertices are separated by spaces. Example: {two_edges} (do **NOT** include quotes or backticks)."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(found/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the CutEdge_Environment instance.
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
        assert N >= 2, "N should be greater than or equal to 1"

        assert "component_num" in self.parameter, "component_num is required in parameter"
        component_num = self.parameter["component_num"]
        assert 2 <= component_num <= N, "component_num should be between 2 and N"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        while True :
            components = [random.randint(0, component_num - 1) for vertex in range(N)]
            if len(set(components)) >= 2 :
                break

        component2vertices = [[] for _ in range(component_num)]
        for vertex, component in enumerate(components) :
            component2vertices[component].append(vertex)
        
        edges = self.parameter["edges"] = []
        remaining_edges = []

        previous_vertices = []
        for component in range(component_num) :
            vertices = component2vertices[component]
            if len(vertices) == 0 :
                continue
            if previous_vertices :
                u = random.choice(previous_vertices)
                v = random.choice(vertices)
                edges.append((min(u, v), max(u, v)))
            for u in vertices :
                for v in vertices :
                    if u < v :
                        remaining_edges.append((u, v))
            previous_vertices += vertices
        
        num_edges = int(edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges :
            edges += random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
        random.shuffle(edges)

        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"


        adj = [[] for _ in range(N)]
        for u, v in edges :
            adj[u].append(v)
            adj[v].append(u)

        disc = [-1] * N
        low = [0] * N
        timer = 0
        bridges = set()

        def dfs(u : int, parent : int) :
            nonlocal timer
            disc[u] = low[u] = timer
            timer += 1
            for v in adj[u] :
                if v == parent :
                    continue
                if disc[v] == -1 :
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] > disc[u] :
                        bridges.add((min(u, v), max(u, v)))
                else :
                    low[u] = min(low[u], disc[v])

        for u in range(N) :
            if disc[u] == -1 :
                dfs(u, -1)

        self.parameter["bridges"] = bridges = list(bridges)        
        assert len(bridges) > 0, "There should be at least one bridge"
        self.parameter["reference_answer"] = " ".join("{} {}".format(u, v) for u, v in bridges)
    
    def _prompt_generate(self) -> str :
        edges = self.parameter["edges"]
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in edges),
            two_edges = " ".join("{} {}".format(u, v) for u, v in edges[: 2]),
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

            bridges = processed_result
            if len(bridges) % 2 != 0 :
                return self.rewards["wrong_format"]
            bridges = [(min(bridges[i], bridges[i + 1]), max(bridges[i], bridges[i + 1])) for i in range(0, len(bridges), 2)]

            if len(bridges) != len(set(bridges)) :
                return self.rewards["invalid_solution"]
            bridges = set(bridges)

            gold_bridges = set(map(tuple, self.parameter["bridges"]))
            if not (bridges <= gold_bridges) :
                return self.rewards["invalid_solution"]

            if self.rewards["rewarding_strategy"] == "(found/all)^beta" :
                return self.rewards["rewarding_weight"] * ((len(bridges) / len(gold_bridges)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "found=all" :
                return self.rewards["rewarding_weight"] * (len(bridges) == len(gold_bridges))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]