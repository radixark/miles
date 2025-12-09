import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class OddVisitation_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a **connected undirected graph** with {N} vertices labeled from 0 to {N_minus_1}. The graph contains the following undirected edges:
{edges}

Your task is to find a trajectory that visits each vertex odd numbers of times, and the starting and ending vertices can be arbitrary.
Formally, you should find a sequence of length $K$ (which is decided by you), $v_0, v_1, \\ldots, v_{{K-1}}$, such that:
(1) $v_i$ and $v_{{i+1}}$ are connected by an edge for all $0 \\leq i < K - 1$;
(2) for each vertex with label $v$ ($0 \\leq v < N$), the number of times it appears in the sequence is odd: \[\sum_{{i=0}}^{{K-1}} [v_i = v] \\equiv 1 \\pmod 2.\]

**Output Format:** Your output should be one single line of $K$ integers (you don't need to output $K$), separated by spaces, representing the sequence $v_0, v_1, \\ldots, v_{{K-1}}$."""

    def __init__(self,
                 wrong_format : float = -1.0,
                 invalid_solution : float = -0.5,
                 correct_solution : float = +1.0,
                 wrong_solution : float = 0.0,
                 **kwargs) :
        """
        Initialize the OddVisitation_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "correct_solution" : correct_solution,
            "wrong_solution" : wrong_solution,
        }


    def _generate(self) -> None:
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 2"

        assert "edge_ratio" in self.parameter, "edge_ratio is required in parameter"
        edge_ratio = self.parameter["edge_ratio"]

        edges = self.parameter["edges"] = []

        # randomly generate a spanning tree using Prufer sequence
        prufer = [random.randint(0, N - 1) for _ in range(N - 2)]
        degree = [1] * N
        for v in prufer:
            degree[v] += 1
        leaves = [i for i in range(N) if degree[i] == 1]
        for v in prufer:
            u = leaves.pop(0)
            if u > v:
                edges.append((v, u))
            else:
                edges.append((u, v))
            degree[u] -= 1
            degree[v] -= 1
            if degree[u] == 1:
                leaves.append(u)
            if degree[v] == 1 and v not in leaves:
                leaves.append(v)
        u = leaves.pop(0)
        v = leaves.pop(0)
        if u > v:
            u, v = v, u
        edges.append((u, v))

        num_edges = int(N * edge_ratio)
        if len(edges) < num_edges :
            remaining_edges = list(set((u, v) for u in range(N) for v in range(u + 1, N)) - set(edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            edges += remaining_edges
        random.shuffle(edges)

        for u, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set(edges)), "edges should be unique"
        

        # generate reference answer
        edges = [[] for _ in range(N)]
        for u, v in self.parameter["edges"]:
            edges[u].append(v)
            edges[v].append(u)
        
        sons = [[] for _ in range(N)]
        visited = [False] * N
        def dfs1(u, fa):
            visited[u] = True
            for v in edges[u]:
                if v != fa and not visited[v]:
                    sons[u].append(v)
                    dfs1(v, u)
        dfs1(0, -1)

        answer = []
        def dfs2(u):
            u_visit = 1
            answer.append(u)
            for v in sons[u]:
                finished = dfs2(v)
                u_visit += 1
                answer.append(u)
                if not finished:
                    answer.append(v)
                    u_visit += 1
                    answer.append(u)
            return u_visit % 2 == 1
        dfs2(0)
        if sum(1 for v in answer if v == 0) % 2 == 0:
            assert answer[-1] == 0, "The last vertex should be 0 to ensure odd visitation."
            answer = answer[:-1]
        
        self.parameter["reference_answer"] = " ".join(map(str, answer))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(u, v) for u, v in self.parameter["edges"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                seq = list(map(int, answer.split()))
                return seq
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        seq = self.processor(output)
        if seq is not None :
            cnt = [0] * self.parameter["N"]
            for v in seq :
                if 0 <= v < self.parameter["N"] :
                    cnt[v] += 1
                else :
                    return self.rewards["invalid_solution"]
            
            edges = set(map(tuple, self.parameter["edges"]))
            for i in range(len(seq) - 1) :
                u, v = seq[i], seq[i + 1]
                if u > v:
                    u, v = v, u
                if (u, v) not in edges:
                    return self.rewards["invalid_solution"]
            
            if any(c % 2 == 0 for c in cnt) :
                return self.rewards["wrong_solution"]
            else :
                assert all(c % 2 == 1 for c in cnt), "All vertices should be visited odd times."
                return self.rewards["correct_solution"]
        else :
            return self.rewards["wrong_format"]