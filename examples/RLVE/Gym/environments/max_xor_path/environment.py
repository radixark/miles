import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MaxXorPath_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4151
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `0` to `{N_minus_1}`.

The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning an undirected edge **connecting vertex `u` to vertex `v` with weight `w`**:
{edges}

Find a path from vertex `0` to vertex `{N_minus_1}` such that the XOR of the weights of the edges in the path is maximized. Output the maximum XOR value."""

    def __init__(self,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MaxXorPath_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        assert "MAX_bit_length" in self.parameter, "MAX_bit_length is required in parameter"
        MAX_bit_length = self.parameter["MAX_bit_length"]
        assert MAX_bit_length >= 2, "MAX_bit_length should be greater than or equal to 2"

        while True :
            adjacent = [[] for _ in range(N)]
            for u, v in random.sample([(u, v) for u in range(N) for v in range(u + 1, N)], int(edge_density * N * (N - 1) / 2)) :
                adjacent[u].append(v)
                adjacent[v].append(u)

            base_size_upper = random.randint(0, MAX_bit_length - 1)
            
            edges = self.parameter["edges"] = []

            P = [0] * MAX_bit_length
            base_size = 0

            def insert_into_basis(x: int) -> None:
                """
                Insert x into the XOR basis P.
                """
                nonlocal P, base_size
                cur = x
                for i in range(MAX_bit_length - 1, -1, -1):
                    if not ((cur >> i) & 1):
                        continue
                    if P[i] == 0:
                        P[i] = cur
                        base_size += 1
                        return
                    cur ^= P[i]

            def maximize_with_basis(x: int) -> int:
                """
                Given a number x, maximize x XOR (any combination of basis vectors).
                """
                res = x
                for i in range(MAX_bit_length - 1, -1, -1):
                    if P[i] != 0 and (res ^ P[i]) > res:
                        res ^= P[i]
                return res

            # Arrays to track visited nodes and the XOR-distance from node 0
            visited = [False] * N
            xor_to = [0] * N

            edge2weight = {}

            def DFS(u : int) -> None :
                visited[u] = True
                for nbr in adjacent[u]:
                    if not visited[nbr]:
                        w = random.randint(0, 2 ** MAX_bit_length - 1)
                        if (min(u, nbr), max(u, nbr)) not in edge2weight :
                            edges.append((min(u, nbr), max(u, nbr), w))
                            edge2weight[(min(u, nbr), max(u, nbr))] = w
                        xor_to[nbr] = xor_to[u] ^ w
                        DFS(nbr)
                    else:
                        if (min(u, nbr), max(u, nbr)) not in edge2weight :
                            if base_size < base_size_upper :
                                w = random.randint(0, 2 ** MAX_bit_length - 1)
                            else :
                                w = xor_to[u] ^ xor_to[nbr]
                                for i in range(MAX_bit_length - 1, -1, -1) :
                                    if random.random() < 0.5 :
                                        w ^= P[i]
                            edges.append((min(u, nbr), max(u, nbr), w))
                            edge2weight[(min(u, nbr), max(u, nbr))] = w
                        else :
                            w = edge2weight[(min(u, nbr), max(u, nbr))]
                        cycle_xor = xor_to[u] ^ w ^ xor_to[nbr]
                        insert_into_basis(cycle_xor)
            DFS(0)
            if not visited[N - 1] :
                continue

            self.parameter["reference_answer"] = maximize_with_basis(xor_to[N - 1])
            if self.parameter["reference_answer"] < 2 ** MAX_bit_length - 1 :
                random.shuffle(edges)
                break
        
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "edges should be unique"
        for u, v, w in edges :
            assert 0 <= u < v < N
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {}, {})".format(u, v, w) for u, v, w in self.parameter["edges"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                int_answer = int(answer)
                return int_answer
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]