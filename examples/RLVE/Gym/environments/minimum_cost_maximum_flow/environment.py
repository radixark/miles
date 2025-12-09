import random
import networkx
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinimumCost_MaximumFlow_Environment(VerifiableEnvironment):
    prompt_template = \
r"""You are given a **directed graph** with {N} vertices, labeled from `0` to `{N_minus_1}`. The source vertex is `0` and the sink vertex is `{N_minus_1}`.

The graph contains the following directed edges. Each edge is represented as a tuple `(s, t, c, w)`, meaning a directed edge **from vertex `s` to vertex `t` with positive capacity `c` and positive cost `w`**:
{edges}

Your task is to find a **maximum flow** from source to sink that has the **minimum possible total cost**. A valid flow must satisfy these conditions:
1. The flow through each edge (which should not be negative) must not exceed its capacity
2. For each vertex (except source and sink), the total incoming flow must equal the total outgoing flow
3. The total flow leaving the source must be equal to the total flow entering the sink

Among all possible maximum flows (flows that satisfy the above conditions and maximize the total flow from source to sink), you need to find the one with minimum total cost. The total cost is the sum of (flow x cost) for each edge.

**Output Format:**
Your final answer should be a single line containing the flow values for each edge in the same order as they appear above, separated by **spaces**.
Example: `1 2 0 3` (do **NOT** include the backticks or quotes); this means the first edge has flow 1, second edge has flow 2, third edge has flow 0, and fourth edge has flow 3."""

    def __init__(self,
                 max_capacity: int = 10, max_cost: int = 10,
                 wrong_format: float = -1.0, invalid_solution: float = -0.5,
                 rewarding_strategy_flow: str = "(answer/gold)^beta", rewarding_weight_flow: float = +0.5, rewarding_beta_flow: float = 5.0,
                 rewarding_strategy_cost: str = "(gold/answer)^beta", rewarding_weight_cost: float = +0.5, rewarding_beta_cost: float = 5.0,
                 **kwargs):
        """
        Initialize the MaxFlow_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_capacity = max_capacity
        self.max_cost = max_cost

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy_flow": rewarding_strategy_flow,
            "rewarding_weight_flow": rewarding_weight_flow,
            "rewarding_beta_flow": rewarding_beta_flow,
            "rewarding_strategy_cost": rewarding_strategy_cost,
            "rewarding_weight_cost": rewarding_weight_cost,
            "rewarding_beta_cost": rewarding_beta_cost,
        }


    def _generate(self) -> None:
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        # Generate edges
        edges = self.parameter["edges"] = []

        # First ensure there's at least one path from source to sink with sufficient capacity
        path_length = random.randint(2, min(5, N - 1))
        path = [0] + random.sample(range(1, N - 1), path_length - 1) + [N - 1]
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            assert s != t
            capacity = random.randint(self.max_capacity // 2, self.max_capacity)  # Ensure good capacity
            cost = random.randint(1, self.max_cost)
            edges.append((s, t, capacity, cost))

        # Add remaining edges randomly, ensuring the graph is well-connected
        num_edges = int(edge_density * N * (N - 1))
        if len(edges) < num_edges:
            remaining_edges = list(set((s, t) for s in range(N) for t in range(N) if s != t and t != 0 and s != N - 1) - set((s, t) for s, t, c, w in edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for s, t in remaining_edges:
                capacity = random.randint(1, self.max_capacity)
                cost = random.randint(1, self.max_cost)
                edges.append((s, t, capacity, cost))
        random.shuffle(edges)

        for s, t, c, w in edges :
            assert 0 <= s < N and s != N - 1, "Source vertex out of bounds"
            assert 0 <= t < N and t != 0, "Target vertex out of bounds"
            assert s != t, "Source and target vertices must be different"
            assert c > 0, "Capacity must be positive"
            assert w > 0, "Cost must be positive"
        assert len(edges) == len(set((s, t) for s, t, c, w in edges)), "Edges must be unique"


        # Create networkx graph and compute max flow min cost
        G = networkx.DiGraph()
        # Add all nodes first
        for v in range(N):
            G.add_node(v)
        for s, t, c, w in edges:
            G.add_edge(s, t, capacity=c, weight=w)
        
        # Compute max flow min cost in one step
        flow_dict = networkx.max_flow_min_cost(G, 0, N - 1)
        
        # Store reference answer
        reference_flows = []
        for edge in edges:
            s, t = edge[0], edge[1]
            flow = flow_dict[s][t] if t in flow_dict[s] else 0
            reference_flows.append(flow)
        self.parameter["reference_answer"] = " ".join(map(str, reference_flows))

        total_flow = sum(flow_dict[0][t] for t in flow_dict[0])  # Total flow from source
        total_cost = sum(flow_dict[s][t] * G[s][t]['weight'] for s in flow_dict for t in flow_dict[s])
        assert total_flow > 0 and total_cost > 0
        self.parameter["gold_answer"] = {"flow" : total_flow, "cost": total_cost}


    def _prompt_generate(self) -> str:
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {}, {}, {})".format(s, t, c, w) for s, t, c, w in self.parameter["edges"]),
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


    def scorer(self, output: str) -> float:
        processed_result = self.processor(output)
        if processed_result is not None:
            assert isinstance(processed_result, list), "processed_result should be a list"

            flows = processed_result
            if len(flows) != len(self.parameter["edges"]):
                return self.rewards["wrong_format"]

            # Check if flows are valid
            N = self.parameter["N"]
            
            # Initialize flow arrays for each vertex
            in_flows = [0] * N
            out_flows = [0] * N
            
            # Check flows and compute vertex flows in one pass
            for i, (s, t, capacity, cost) in enumerate(self.parameter["edges"]):
                flow = flows[i]
                # Check if flow is valid
                if not (0 <= flow <= capacity):
                    return self.rewards["invalid_solution"]
                
                # Update vertex flows
                out_flows[s] += flow
                in_flows[t] += flow

            # Check flow conservation at intermediate vertices
            for v in range(N):
                if v == 0 or v == N - 1:
                    continue
                if in_flows[v] != out_flows[v]:
                    return self.rewards["invalid_solution"]

            # Check flow balance between source and sink
            if out_flows[0] != in_flows[N - 1]:
                return self.rewards["invalid_solution"]
            

            reward = 0.0
            
            total_flow, gold_flow = out_flows[0], self.parameter["gold_answer"]["flow"]
            assert total_flow <= gold_flow, "Total flow from source exceeds gold flow"
            if self.rewards["rewarding_strategy_flow"] == "(answer/gold)^beta":
                reward += self.rewards["rewarding_weight_flow"] * ((total_flow / gold_flow) ** self.rewards["rewarding_beta_flow"])
            elif self.rewards["rewarding_strategy_flow"] == "gold=answer":
                reward += self.rewards["rewarding_weight_flow"] * (total_flow == gold_flow)
            else :
                raise NotImplementedError(f"Unknown rewarding strategy: {self.rewards['rewarding_strategy_flow']}")
            
            if total_flow == gold_flow:
                total_cost, gold_cost = sum(flows[i] * cost for i, (_, _, _, cost) in enumerate(self.parameter["edges"])), self.parameter["gold_answer"]["cost"]
                assert gold_cost <= total_cost, "Total cost exceeds gold cost"
                if self.rewards["rewarding_strategy_cost"] == "(gold/answer)^beta":
                    reward += self.rewards["rewarding_weight_cost"] * ((gold_cost / total_cost) ** self.rewards["rewarding_beta_cost"])
                elif self.rewards["rewarding_strategy_cost"] == "gold=answer":
                    reward += self.rewards["rewarding_weight_cost"] * (total_cost == gold_cost)
                else :
                    raise NotImplementedError(f"Unknown rewarding strategy: {self.rewards['rewarding_strategy_cost']}")

            return reward
        else:
            return self.rewards["wrong_format"] 
