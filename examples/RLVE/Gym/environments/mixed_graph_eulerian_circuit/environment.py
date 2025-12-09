import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MixedGraphEulerianCircuit_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a **graph** with {N} vertices labeled from 0 to {N_minus_1}.

The graph contains the following **undirected** edges:
{undirected_edges}

It also contains the following **directed** edges (each `<u, v>` represents a directed edge from vertex `u` to vertex `v`):
{directed_edges}

It is guaranteed that if all directed edges are treated as undirected, the resulting graph is connected and has no repeated edges, and every vertex has an even degree.

Please find an **Eulerian circuit** in this graph — a closed path that starts and ends at the same vertex and **visits each edge exactly once**.
Output a single line containing the sequence of vertex labels visited in order, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
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

        while True :
            degrees = [0] * N
            edges = []
            for v in range(1, N - 1) :
                neighbors = random.sample(range(v), random.randint(0, v))
                for u in neighbors :
                    assert u < v, "Undirected edges should be added in increasing order"
                    edges.append((u, v))
                    degrees[u] += 1
                    degrees[v] += 1
            for u in range(N - 1) :
                if degrees[u] % 2 == 1 :
                    v = N - 1
                    edges.append((u, v))
                    degrees[u] += 1
                    degrees[v] += 1
            assert all(degree % 2 == 0 for degree in degrees), "All vertices should have even degree in undirected edges"

            random.shuffle(edges)
            assert len(edges) == len(set(edges)), "There should be no repeated undirected edges"
            for u, v in edges :
                assert 0 <= u < v < N, "Undirected edges should be within the range of vertex labels"

            # Check if the undirected graph is connected
            undirected_graph = networkx.Graph()
            undirected_graph.add_nodes_from(range(N))
            undirected_graph.add_edges_from(edges)
            if networkx.is_connected(undirected_graph) :
                assert networkx.is_eulerian(undirected_graph), "The undirected graph should be Eulerian"
                break
        

        eulerian_circuit = list(networkx.eulerian_circuit(undirected_graph))
        assert len(eulerian_circuit) == len(edges), "The Eulerian circuit should visit each edge exactly once"
        directed_flags = [False] * len(eulerian_circuit)
        for flagged in random.sample(range(len(eulerian_circuit)), random.randint(1, len(eulerian_circuit) - 1)) :
            directed_flags[flagged] = True
        
        undirected_edges, directed_edges = self.parameter["undirected_edges"], self.parameter["directed_edges"] = [], []
        self.parameter["reference_answer"] = []
        for (u, v), directed_flag in zip(eulerian_circuit, directed_flags) :
            self.parameter["reference_answer"].append(u)
            if directed_flag :
                directed_edges.append((u, v))
            else :
                undirected_edges.append((min(u, v), max(u, v)))
        self.parameter["reference_answer"].append(eulerian_circuit[-1][1])
        assert self.parameter["reference_answer"][0] == self.parameter["reference_answer"][-1], "The Eulerian circuit should start and end at the same vertex"
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["reference_answer"]))
        assert len(undirected_edges) > 0 and len(directed_edges) > 0, "There should be at least one undirected edge and one directed edge"
        random.shuffle(undirected_edges)
        random.shuffle(directed_edges)
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            undirected_edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["undirected_edges"]),
            directed_edges = "\n".join("<{}, {}>".format(u, v) for u, v in self.parameter["directed_edges"]),
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

            if len(processed_result) == 0 :
                return self.rewards["wrong_format"]

            if not all(0 <= u < self.parameter["N"] for u in processed_result) :
                return self.rewards["invalid_solution"]
            undirected_edges, directed_edges = {(u, v) : 0 for u, v in self.parameter["undirected_edges"]}, {(u, v) : 0 for u, v in self.parameter["directed_edges"]}
            if processed_result[0] != processed_result[-1] :
                return self.rewards["invalid_solution"]
            for u, v in zip(processed_result, processed_result[1 :]) :
                directed, undirected = (u, v) in directed_edges, (min(u, v), max(u, v)) in undirected_edges
                assert int(directed) + int(undirected) <= 1
                if directed :
                    directed_edges[(u, v)] += 1
                elif undirected :
                    undirected_edges[(min(u, v), max(u, v))] += 1
                else :
                    return self.rewards["invalid_solution"]
            
            satisfied = sum(count == 1 for count in directed_edges.values()) + sum(count == 1 for count in undirected_edges.values())
            assert satisfied <= len(self.parameter["undirected_edges"]) + len(self.parameter["directed_edges"]), "satisfied should be less than or equal to the total number of edges"
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / (len(self.parameter["undirected_edges"]) + len(self.parameter["directed_edges"]))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == (len(self.parameter["undirected_edges"]) + len(self.parameter["directed_edges"])))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]