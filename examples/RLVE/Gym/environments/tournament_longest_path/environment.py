import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Tournament_LongestPath_Environment(VerifiableEnvironment):
    prompt_template = \
r"""You are given a **directed graph** with {N} vertices labeled from `0` to `{N_minus_1}`. The graph contains the following directed edges. Each edge is represented as a tuple `(s, t)`, meaning there is a directed edge **from vertex `s` to vertex `t`**:
{edges}

It is guaranteed that there is **exactly one directed edge** between every pair of two distinct vertices.
Please find the **longest path** starting from vertex `{S}`, such that no vertex is visited more than once. Output the path as a sequence of vertex labels, starting from `{S}`, separated by spaces, in the order they are visited."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the Tournament_LongestPath_Environment instance.
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

        keep_probability = random.random()
        self.parameter["TO"] = TO = [[False] * N for _ in range(N)]
        for i in range(N) :
            for j in range(i + 1, N) :
                if random.random() < keep_probability :
                    TO[i][j] = True
                else :
                    TO[j][i] = True
        

        # Tarjan's algorithm for SCC
        dfn = [0] * N
        low = [0] * N
        on_stack = [False] * N
        stack = []
        scc = [0] * N
        comp_nodes = []
        time_counter = 0
        scc_count = 0

        def tarjan(u):
            nonlocal time_counter, scc_count
            time_counter += 1
            dfn[u] = low[u] = time_counter
            stack.append(u)
            on_stack[u] = True
            for v in range(N):
                if TO[u][v]:
                    if dfn[v] == 0:
                        tarjan(v)
                        low[u] = min(low[u], low[v])
                    elif on_stack[v]:
                        low[u] = min(low[u], dfn[v])
            if dfn[u] == low[u]:
                comp_nodes.append([])
                cid = scc_count
                scc_count += 1
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc[w] = cid
                    comp_nodes[cid].append(w)
                    if w == u:
                        break

        for i in range(N):
            if dfn[i] == 0:
                tarjan(i)

        # Build a Hamiltonian cycle in each non-trivial SCC
        nxt = [None] * N
        def solve(cid):
            nodes = comp_nodes[cid]
            if len(nodes) <= 1:
                return
            s = t = nodes[0]
            for x in nodes[1:]:
                if TO[t][x]:
                    nxt[t] = x; t = x
                elif TO[x][s]:
                    nxt[x] = s; s = x
                else:
                    j = s
                    while j != t:
                        nj = nxt[j]
                        if TO[j][x] and TO[x][nj]:
                            nxt[x] = nj
                            nxt[j] = x
                            break
                        j = nj
            # close the cycle
            t2 = None
            i = nxt[s]
            while i is not None:
                if TO[i][s]:
                    t2 = i
                elif t2 is not None:
                    j = s
                    while j != t2:
                        nj = nxt[j]
                        if TO[i][nj]:
                            x = nj
                            nxt[j] = nxt[t2]
                            nxt[t2] = s
                            s = x
                            t2 = i
                            break
                        j = nj
                i = nxt[i]
            nxt[t2] = s

        for cid in range(scc_count):
            solve(cid)

        # Build answers for each starting vertex
        ans = [[] for _ in range(N)]
        for i in range(N):
            x = i
            cid = scc[i]
            while True:
                ans[i].append(x)
                nodes = comp_nodes[cid]
                if len(nodes) == 1:
                    if cid == 0:
                        break
                    cid -= 1
                    x = comp_nodes[cid][0]
                    continue
                j = nxt[x]
                while j != x:
                    ans[i].append(j)
                    j = nxt[j]
                if cid == 0:
                    break
                cid -= 1
                x = comp_nodes[cid][0]

        S = self.parameter["S"] = random.randint(0, N - 1)
        path = ans[S]
        self.parameter["gold_answer"] = len(path)
        self.parameter["reference_answer"] = " ".join(map(str, path))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {})".format(s, t) for s in range(N) for t in range(N) if self.parameter["TO"][s][t]),
            S = self.parameter["S"],
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                if not answer_array :
                    return None
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            path = processed_result
            if len(path) == 0 :
                return self.rewards["wrong_format"]
            if path[0] != self.parameter["S"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= vertex < self.parameter["N"] for vertex in path) :
                return self.rewards["invalid_solution"]
            if len(set(path)) != len(path) :
                return self.rewards["invalid_solution"]
            if not all(self.parameter["TO"][s][t] for s, t in zip(path, path[1 :])) :
                return self.rewards["invalid_solution"]

            answer, gold = len(path), self.parameter["gold_answer"]
            assert 0 < answer <= gold, "Answer length should be positive and not exceed gold length"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]