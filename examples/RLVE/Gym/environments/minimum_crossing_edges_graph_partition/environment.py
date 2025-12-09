import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Minimum_CrossingEdges_GraphPartition_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices labeled from `0` to `{N_minus_1}`. The graph contains the following undirected edges:
{edges}

Partition all vertices into {K} **non-empty** sets, such that each vertex belongs to exactly one set.  
Try your best to **minimize the number of crossing edges** — an edge `(u, v)` is considered crossing if `u` and `v` are in different sets.

**Output Format:** Output a list of {N} integers (separated by space), where the `i`-th integer is the index of the set (from `0` to `{K_minus_1}`) that vertex `i` belongs to."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Minimum_CrossingEdges_GraphPartition_Environment instance.
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

        K = self.parameter["K"] = random.randint(2, N - 1)

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        edges = self.parameter["edges"] = random.sample([(u, v) for u in range(N) for v in range(u + 1, N)], int(edge_density * N * (N - 1) / 2))
        random.shuffle(edges)
        
        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"


        internal_edges = [0] * (1 << N)
        for u, v in edges :
            remaining_S = ((1 << N) - 1) - (1 << u) - (1 << v)
            S = remaining_S
            while True :
                internal_edges[S + (1 << u) + (1 << v)] += 1
                if S == 0 :
                    break
                S = (S - 1) & remaining_S
        
        F = [None] * (1 << N)
        F[0] = 0
        for k in range(K) :
            G = [None] * (1 << N)
            for S in range(1 << N) :
                if F[S] is None :
                    continue
                S_complement = ((1 << N) - 1) - S
                T = S_complement
                while T :
                    if G[S + T] is None :
                        G[S + T] = F[S] + internal_edges[T]
                    else :
                        G[S + T] = max(G[S + T], F[S] + internal_edges[T])
                    T = (T - 1) & S_complement
            F = G
        
        self.parameter["gold_answer"] = len(edges) - F[(1 << N) - 1]
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        K = self.parameter["K"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            K = K,
            K_minus_1 = K - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
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
            if not all(0 <= x < self.parameter["K"] for x in processed_result) :
                return self.rewards["invalid_solution"]
            if len(set(processed_result)) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            
            gold, answer = self.parameter["gold_answer"], 0
            for u, v in self.parameter["edges"] :
                if processed_result[u] != processed_result[v] :
                    answer += 1
            assert gold <= answer, "gold_answer should be less than or equal to answer"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    return self.rewards["rewarding_weight"]
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]