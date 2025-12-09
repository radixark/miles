import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class SubgraphIsomorphism_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given two **undirected graphs**, G1 and G2.

- G1 has `{N1}` vertices labeled from `0` to `{N1_minus_1}`. It has the following edge set E1:
{G1_edges}

- G2 has `{N2}` vertices labeled from `0` to `{N2_minus_1}`. It has the following edge set E2:
{G2_edges}

Please find an **injection** `p` (an injection means each vertex in G1 maps to a **unique** vertex in G2) from the vertices of G1 to the vertices of G2. This mapping `p` must satisfy the following condition: for every pair `(u, v)`, the edge `(u, v)` exists in E1 **if and only if** the edge `(p(u), p(v))` exists in E2.

**Output Format:** Your final answer should be a single line containing `p(0), p(1), ..., p({N1_minus_1})`, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SubgraphIsomorphism_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N2" in self.parameter, "N2 is required in parameter"
        N2 = self.parameter["N2"]
        assert N2 >= 3, "N2 should be greater than or equal to 3"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 < edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"
        assert int(edge_density * N2 * (N2 - 1) / 2) > 0

        G2_edges = self.parameter["G2_edges"] = random.sample([(u, v) for u in range(N2) for v in range(u + 1, N2)], int(edge_density * N2 * (N2 - 1) / 2))
        random.shuffle(G2_edges)

        N1 = self.parameter["N1"] = random.randint(3, N2)
        mapping = random.sample(range(N2), N1)
        random.shuffle(mapping)

        G1_edges = self.parameter["G1_edges"] = []
        G2_edges_set = set(G2_edges)
        for u in range(N1) :
            for v in range(u + 1, N1) :
                G2_u, G2_v = mapping[u], mapping[v]
                if G2_u > G2_v :
                    G2_u, G2_v = G2_v, G2_u
                if (G2_u, G2_v) in G2_edges_set :
                    G1_edges.append((u, v))
        random.shuffle(G1_edges)

        for edges, N in zip((G1_edges, G2_edges), (N1, N2)) :
            for u, v in edges :
                assert 0 <= u < v < N
            assert len(edges) == len(set(edges)), "edges should be unique"

        self.parameter["reference_answer"] = " ".join(map(str, mapping))
    

    def _prompt_generate(self) -> str :
        N1, N2 = self.parameter["N1"], self.parameter["N2"]
        N1_minus_1, N2_minus_1 = N1 - 1, N2 - 1
        G1_edges, G2_edges = self.parameter["G1_edges"], self.parameter["G2_edges"]
        return self.prompt_template.format(
            N1 = N1,
            N1_minus_1 = N1_minus_1,
            G1_edges = "\n".join("({}, {})".format(u, v) for u, v in G1_edges),
            N2 = N2,
            N2_minus_1 = N2_minus_1,
            G2_edges = "\n".join("({}, {})".format(u, v) for u, v in G2_edges),
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

            mapping = processed_result
            if len(mapping) != self.parameter["N1"] :
                return self.rewards["invalid_solution"]
            if len(set(mapping)) != self.parameter["N1"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= i < self.parameter["N2"] for i in mapping) :
                return self.rewards["invalid_solution"]

            G1_edge_set, G2_edges_set = set(map(tuple, self.parameter["G1_edges"])), set(map(tuple, self.parameter["G2_edges"]))
            satisfied = 0
            for u in range(self.parameter["N1"]) :
                for v in range(u + 1, self.parameter["N1"]) :
                    G2_u, G2_v = mapping[u], mapping[v]
                    if G2_u > G2_v :
                        G2_u, G2_v = G2_v, G2_u
                    satisfied += int(((u, v) in G1_edge_set) == ((G2_u, G2_v) in G2_edges_set))
            all_edges = self.parameter["N1"] * (self.parameter["N1"] - 1) // 2
            assert satisfied <= all_edges, "satisfied edges should not exceed all edges"
            
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / all_edges) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == all_edges)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]