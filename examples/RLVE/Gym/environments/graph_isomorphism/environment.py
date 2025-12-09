import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class GraphIsomorphism_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given two **undirected graphs**, G1 and G2, each with {N} vertices labeled from `0` to `{N_minus_1}`. Both graphs contain exactly {M} **undirected** edges.

- Graph G1 has the following (undirected) edge set E1:
{G1_edges}

- Graph G2 has the following (undirected) edge set E2:
{G2_edges}

Your task is to find a **bijection** (i.e., a permutation) `p` from the vertices of G1 to the vertices of G2 such that: For every edge `(u, v)` in E1, the edge `(p(u), p(v))` exists in E2, and vice versa.

**Output Format:** Your final answer should be a single line containing the permutation `p(0), p(1), ..., p({N_minus_1})`, separated by spaces. Example: `{reversed_permutation}` (do **NOT** include backticks or quotes); this means `p(0) = {N_minus_1}, ..., p({N_minus_1}) = 0`."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(overlap/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the GraphIsomorphism_Environment instance.
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
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 < edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"
        assert int(edge_density * N * (N - 1) / 2) > 0

        G1_edges = self.parameter["G1_edges"] = random.sample([(u, v) for u in range(N) for v in range(u + 1, N)], int(edge_density * N * (N - 1) / 2))
        random.shuffle(G1_edges)

        mapping = list(range(N))
        random.shuffle(mapping)
        G2_edges = self.parameter["G2_edges"] = []
        for u, v in G1_edges :
            u, v = mapping[u], mapping[v]
            if u > v :
                u, v = v, u
            G2_edges.append((u, v))
        random.shuffle(G2_edges)

        for edges in (G1_edges, G2_edges) :
            for u, v in edges :
                assert 0 <= u < v < N
            assert len(edges) == len(set(edges)), "edges should be unique"

        self.parameter["reference_answer"] = " ".join(map(str, mapping))


    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        G1_edges, G2_edges = self.parameter["G1_edges"], self.parameter["G2_edges"]
        assert len(G1_edges) == len(G2_edges), "G1_edges and G2_edges should have the same length"
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            M = len(G1_edges),
            G1_edges = "\n".join("({}, {})".format(u, v) for u, v in G1_edges),
            G2_edges = "\n".join("({}, {})".format(u, v) for u, v in G2_edges),
            reversed_permutation = " ".join(map(str, range(N - 1, -1, -1))),
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

            permutation = processed_result
            if len(permutation) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if len(set(permutation)) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= i < self.parameter["N"] for i in permutation) :
                return self.rewards["invalid_solution"]

            new_G2_edges = set()
            for u, v in self.parameter["G1_edges"] :
                u, v = permutation[u], permutation[v]
                if u > v :
                    u, v = v, u
                new_G2_edges.add((u, v))
            assert len(new_G2_edges) == len(self.parameter["G1_edges"]), "new_G2_edges should have the same length as G1_edges"
            overlap = len(new_G2_edges & set(map(tuple, self.parameter["G2_edges"])))
            assert overlap <= len(self.parameter["G2_edges"]), "overlap should be less than or equal to len(G2_edges)"

            # ---------------------------------------- Sanity Check ----------------------------------------
            G1_edge_set, G2_edges_set = set(map(tuple, self.parameter["G1_edges"])), set(map(tuple, self.parameter["G2_edges"]))
            unsatisfied = 0
            for u in range(self.parameter["N"]) :
                for v in range(u + 1, self.parameter["N"]) :
                    G2_u, G2_v = permutation[u], permutation[v]
                    if G2_u > G2_v :
                        G2_u, G2_v = G2_v, G2_u
                    unsatisfied += int(((u, v) in G1_edge_set) != ((G2_u, G2_v) in G2_edges_set))
            assert unsatisfied == (len(self.parameter["G2_edges"]) - overlap) * 2
            # ---------------------------------------- Sanity Check ----------------------------------------
            
            if self.rewards["rewarding_strategy"] == "(overlap/all)^beta" :
                return self.rewards["rewarding_weight"] * ((overlap / len(self.parameter["G2_edges"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "overlap=all" :
                return self.rewards["rewarding_weight"] * (overlap == len(self.parameter["G2_edges"]))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]