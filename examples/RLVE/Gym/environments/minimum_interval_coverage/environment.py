import random
import networkx as nx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinimumIntervalCoverage_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3980
    prompt_template = \
r"""You are given {M} intervals within [1, {N}]. Each interval is defined as [L[i], R[i]] with an associated cost C[i]. The intervals are provided as:
{intervals}

You can select each interval any number of times (including 0). For each point i in [1, {N}], you must ensure it is covered by at least NEED[i] selected intervals, where the array NEED is given as:
{NEED}

Your goal is to minimize the **total cost** of the selected intervals while satisfying the above condition.

**Output Format:** A single line containing {M} integers — the number of times you select each interval, in order, separated by spaces."""

    def __init__(self,
                 cost_multiple : int = 3,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = +5.0,
                 **kwargs) :
        """
        Initialize the MinimumIntervalCoverage_Environment instance.
        """
        super().__init__(**kwargs)

        self.cost_multiple = cost_multiple

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
        assert N >= 1, "N must be at least 1"

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 1, "M must be at least 1"

        INTERVALS = self.parameter["intervals"] = []
        for i in range(M) :
            L, R = random.randint(1, N), random.randint(1, N)
            if L > R :
                L, R = R, L
            C = random.randint(1, self.cost_multiple * (R - L + 1))
            INTERVALS.append((L, R, C))
        
        NEEDS = self.parameter["NEEDS"] = []
        for i in range(1, N + 1) :
            NEEDS.append(random.randint(0, N) if any(L <= i <= R for L, R, C in INTERVALS) else 0)


        # Pad NEED with zeros at both ends (for difference calculation)
        NEED = [0] + NEEDS + [0]  # length N+2

        # Build the demand for each node 0..N:
        # DEMANDS[k] = flow into node k minus flow out of node k
        # We want net in-out = NEED[k+1] - NEED[k]
        DEMANDS = [0] * (N + 1)
        for i in range(1, N + 2):
            DEMANDS[i - 1] = NEED[i] - NEED[i - 1]

        # Build the directed graph
        G = nx.MultiDiGraph()
        INF = sum(NEEDS)

        # Add all nodes with their 'demand' attribute
        for node, d in enumerate(DEMANDS):
            G.add_node(node, demand=d)

        # Add the "chain" edges i -> i+1 with infinite capacity and zero cost
        for i in range(N):
            G.add_edge(i, i + 1, capacity=INF, weight=0)

        # Add an edge for each volunteer type:
        # selecting one volunteer of type (s, t, c) corresponds to sending
        # one unit of flow along t -> (s-1) at cost c
        for s, t, c in INTERVALS:
            u = t      # maps to node t (since t in [1..N], node range is 0..N)
            v = s - 1  # maps to node s-1
            G.add_edge(u, v, capacity=INF, weight=c)

        # Compute the minimum-cost flow satisfying all node demands
        cost, flow_dict = nx.network_simplex(G)

        # Output the total minimum cost
        self.parameter["gold_answer"] = cost
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            intervals = "\n".join("L[{}]={} R[{}]={} C[{}]={}".format(i + 1, L, i + 1, R, i + 1, C) for i, (L, R, C) in enumerate(self.parameter["intervals"])),
            NEED = " ".join("NEED[{}]={}".format(i + 1, need) for i, need in enumerate(self.parameter["NEEDS"]))
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

            times = processed_result
            if len(times) != self.parameter["M"] :
                return self.rewards["wrong_format"]
            if any(t < 0 for t in times) :
                return self.rewards["invalid_solution"]
            for i in range(1, self.parameter["N"] + 1) :
                if sum(int(L <= i <= R) * times[j] for j, (L, R, C) in enumerate(self.parameter["intervals"])) < self.parameter["NEEDS"][i - 1] :
                    return self.rewards["invalid_solution"]
            
            answer, gold = sum(times[j] * C for j, (L, R, C) in enumerate(self.parameter["intervals"])), self.parameter["gold_answer"]
            assert gold <= answer, "answer should be greater than or equal to gold"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert gold == 0, "gold should be zero if answer is zero"
                    return self.rewards["rewarding_weight"] * 1.0  # Reward for zero answer
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]