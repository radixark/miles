import random
import networkx
from collections import deque
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinPathCover_DAG_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4043
    prompt_template = \
r"""You are given a **directed acyclic graph (DAG)** with {N} vertices labeled from 1 to {N}. The graph contains the following directed edges (s, t, w), meaning there is an edge from `s` to `t` with weight `w`. It is guaranteed that vertex 1 can reach all other vertices:
{edges}

Let's find a set of paths such that:
- Each path starts from vertex 1. According to the definition of paths, consecutive vertices in a path are connected by a directed edge (following the edge direction).
- All edges in the graph are covered by at least one path.

Can we **minimize the total weight** of all paths, where the weight of a path is the sum of the weights of its edges? Please output K lines, where K is the number of paths you use; each line should list the vertices of one path in order (starting from 1), separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize MinPathCover_DAG_Environment instance.
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
        assert N >= 4, "N should be greater than or equal to 4"

        edges = self.parameter["edges"] = []
        topological_order = list(range(1, N + 1))
        random.shuffle(topological_order[1 :]) # Keep 1 as the first vertex
        for i in range(1, N) :
            t = topological_order[i]
            for s in random.sample(topological_order[: i], random.randint(1, i)) :
                edges.append((s, t, random.randint(1, N * (N - 1))))
        random.shuffle(edges)

        assert len(edges) == len(set((s, t) for s, t, w in edges)), "Duplicate edges detected"
        
        G = networkx.DiGraph()
        G.add_weighted_edges_from(edges)
        assert networkx.is_directed_acyclic_graph(G), "The generated graph is not a DAG"
        assert all(networkx.has_path(G, 1, v) for v in range(2, N + 1)), "Vertex 1 cannot reach all other vertices"
        

        # Read all edges first to compute INF based on input
        A = [0] * (N + 4)                  # 1-based indexing; extra room for rT=N+1, vS=N+2, vT=N+3
        edges_data = [[] for _ in range(N + 4)]
        total_cost_sum = 0
        M = len(edges)  # total number of edges (sum of K_i)

        for i, u, t in edges :
            edges_data[i].append((u, t))
            A[u] += 1
            A[i] -= 1
            total_cost_sum += t

        # Make INF depend on the input (covers both capacity sentinel and distance sentinel)
        INF = total_cost_sum + M + 5

        size = N + 4  # nodes: 1..N, rT=N+1, vS=N+2, vT=N+3
        Graph = [[] for _ in range(size)]

        class Edge:
            __slots__ = ("to", "cap", "cost", "rev")
            def __init__(self, to, cap, cost, rev):
                self.to = to
                self.cap = cap
                self.cost = cost
                self.rev = rev

        def add_edge(u, v, cap, cost):
            Graph[u].append(Edge(v, cap, cost, len(Graph[v])))
            Graph[v].append(Edge(u, 0, -cost, len(Graph[u]) - 1))

        rS = 1
        rT = N + 1
        vS = N + 2
        vT = N + 3

        # Build edges as in the C++ code
        for i in range(1, N + 1):
            for (u, t) in edges_data[i]:
                add_edge(i, u, INF - 1, t)

        for i in range(2, N + 1):
            add_edge(i, rT, INF, 0)

        for i in range(1, N + 1):
            if A[i] > 0:
                add_edge(vS, i, A[i], 0)
            elif A[i] < 0:
                add_edge(i, vT, -A[i], 0)

        add_edge(rT, rS, INF, 0)

        S = vS
        T = vT

        Dist = [0] * size
        Cur = [0] * size
        InQ = [False] * size
        Vis = [False] * size

        # ret starts as the sum of all edge costs, then augmented during flow as in the original code
        ret = total_cost_sum

        def spfa():
            for i in range(size):
                Dist[i] = INF
                InQ[i] = False
            Dist[S] = 0
            q = deque([S])
            InQ[S] = True
            while q:
                u = q.popleft()
                InQ[u] = False
                for e in Graph[u]:
                    if e.cap > 0 and Dist[e.to] > Dist[u] + e.cost:
                        Dist[e.to] = Dist[u] + e.cost
                        if not InQ[e.to]:
                            InQ[e.to] = True
                            q.append(e.to)
            return Dist[T] < INF

        def dfs(x, f):
            nonlocal ret
            if x == T:
                return f
            Vis[x] = True
            flow = 0
            i = Cur[x]
            while i < len(Graph[x]) and flow < f:
                Cur[x] = i
                e = Graph[x][i]
                v = e.to
                if (not Vis[v]) and e.cap > 0 and Dist[v] == Dist[x] + e.cost:
                    pushed = dfs(v, min(e.cap, f - flow))
                    if pushed:
                        ret += pushed * e.cost
                        e.cap -= pushed
                        Graph[v][e.rev].cap += pushed
                        flow += pushed
                i += 1
            Vis[x] = False
            return flow

        def dinic():
            total = 0
            while spfa():
                for i in range(size):
                    Cur[i] = 0
                    Vis[i] = False
                while True:
                    pushed = dfs(S, INF)
                    if pushed == 0:
                        break
                    total += pushed
            return total

        dinic()
        self.parameter["gold_answer"] = ret
        assert self.parameter["gold_answer"] > 0
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            edges = "\n".join("({}, {}, {})".format(s, t, w) for s, t, w in self.parameter["edges"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[List[List[int]]] :
        if answer is not None :
            answer = answer.strip()
            try :
                paths = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        paths.append(list(map(int, line.split())))
                return paths
            except  :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            edges = {(s, t) : False for s, t, w in self.parameter["edges"]}
            edge2weight = {(s, t) : w for s, t, w in self.parameter["edges"]}
            gold, answer = self.parameter["gold_answer"], 0
            for path in processed_result :
                if not path :
                    return self.rewards["invalid_solution"]
                if path[0] != 1 :
                    return self.rewards["invalid_solution"]
                for i in range(len(path) - 1) :
                    s = path[i]
                    t = path[i + 1]
                    if (s, t) in edges :
                        edges[(s, t)] = True
                        answer += edge2weight[(s, t)]
                    else :
                        return self.rewards["invalid_solution"]
            
            if not all(edges.values()) :
                return self.rewards["unsuccessful_solution"]
            
            assert 0 < gold <= answer, "gold should be less than or equal to answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]