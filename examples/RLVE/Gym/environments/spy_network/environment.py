import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class SpyNetwork_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1262
    prompt_template = \
r"""You are given a **directed graph** with {N} vertices, labeled from 0 to {N_minus_1}.

The graph contains the following directed edges. Each edge is represented as a tuple (s, t), meaning there is a directed edge **from vertex s to vertex t**:
{edges}

Each vertex i has an associated cost c[i], given as follows:
{costs}

Your task is to select a subset of vertices s_1, s_2, ..., s_k such that:
- Every vertex in the graph is reachable (i.e., there exists a path ending at that vertex) starting from at least one of the selected vertices.
- Your goal is to **minimize** the total cost of the selected vertices: c[s_1] + c[s_2] + ... + c[s_k].

**Output Format:**
Your final answer should be a single line containing the selected vertices: s_1, s_2, ..., s_k, separated by **spaces**.
Example: `0 1 {N_minus_1}` (do **NOT** include the backticks or quotes); this means the selected vertices are 0, 1, and {N_minus_1}, and the total cost is c[0] + c[1] + c[{N_minus_1}] = {c_0} + {c_1} + {c_N_minus_1} = {example_cost}.
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.3, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the SpyNetwork_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "unsuccessful_solution" : unsuccessful_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        assert "dominated_probability" in self.parameter, "dominated_probability is required in parameter"
        dominated_probability = self.parameter["dominated_probability"]

        dominated = [random.random() < dominated_probability for vertex in range(N)]
        all_edges = [(s, t) for s in range(N) for t in range(N) if s != t and (dominated[s] == False or dominated[t] == True)]
        edges = self.parameter["edges"] = random.sample(all_edges, min(len(all_edges), int(edge_density * N * (N - 1))))
        random.shuffle(edges)

        assert len(edges) == len(set(edges)), "edges should be unique"
        for s, t in edges :
            assert 0 <= s < N, "s should be in range"
            assert 0 <= t < N, "t should be in range"
            assert s != t, "s should not be equal to t"
        
        costs = self.parameter["costs"] = [random.randint(1, N) for vertex in range(N)]


        adj = [[] for _ in range(N)]
        for s, t in edges :
            adj[s].append(t)

        scc_id     = [0] * N
        pre        = [0] * N
        low        = [0] * N
        stack      = []
        in_stack   = [False] * N

        scc_count  = 0
        dfs_clock  = 0

        def tarjan(u) :
            nonlocal dfs_clock, scc_count
            dfs_clock += 1
            pre[u] = dfs_clock
            low[u] = dfs_clock
            stack.append(u)
            in_stack[u] = True

            for v in adj[u] :
                if pre[v] == 0 :
                    tarjan(v)
                    low[u] = min(low[u], low[v])
                elif in_stack[v] :
                    low[u] = min(low[u], pre[v])

            if low[u] == pre[u] :
                while True :
                    x = stack.pop()
                    in_stack[x] = False
                    scc_id[x] = scc_count
                    if x == u:
                        break
                scc_count += 1

        for i in range(N) :
            if pre[i] == 0 :
                tarjan(i)

        scc_in_degree = [False] * scc_count
        for u in range(N) :
            for v in adj[u] :
                if scc_id[u] != scc_id[v] :
                    scc_in_degree[scc_id[v]] = True

        min_costs = [None] * scc_count
        min_vertices = [None] * scc_count
        for i, _cost in enumerate(costs) :
            if _cost is None :
                continue
            s_id = scc_id[i]
            if min_costs[s_id] is None or _cost < min_costs[s_id] :
                min_costs[s_id] = _cost
                min_vertices[s_id] = i

        self.parameter["reference_answer"] = [min_vertices[s] for s in range(scc_count) if not scc_in_degree[s]]
        self.parameter["gold_answer"] = sum(costs[vertex] for vertex in self.parameter["reference_answer"])
        assert self.parameter["gold_answer"] == sum(min_costs[s] for s in range(scc_count) if not scc_in_degree[s])
        assert self.parameter["gold_answer"] > 0, "gold_answer should be greater than 0"
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["reference_answer"]))
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        costs = self.parameter["costs"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(s, t) for s, t in self.parameter["edges"]),
            costs = "\n".join("c[{}]={}".format(i, costs[i]) for i in range(N)),
            c_0 = costs[0],
            c_1 = costs[1],
            c_N_minus_1 = costs[N - 1],
            example_cost = costs[0] + costs[1] + costs[N - 1],
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

            selected_vertices = processed_result

            adj = [[] for _ in range(self.parameter["N"])]
            for s, t in self.parameter["edges"] :
                adj[s].append(t)
            
            visited = [False] * self.parameter["N"]
            def DFS(vertex) :
                if visited[vertex] :
                    return
                visited[vertex] = True
                for neighbor in adj[vertex] :
                    DFS(neighbor)
            
            if len(selected_vertices) != len(set(selected_vertices)) :
                return self.rewards["invalid_solution"]
            
            answer = 0
            for vertex in selected_vertices :
                if not (0 <= vertex < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                DFS(vertex)
                answer += self.parameter["costs"][vertex]
            
            if not all(visited) :
                return self.rewards["unsuccessful_solution"]
            
            gold = self.parameter["gold_answer"]
            assert gold <= answer
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]