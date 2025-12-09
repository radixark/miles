import random
import networkx
from typing import Optional
from Gym.environment import VerifiableEnvironment


class TreeTopologicalSequenceCounting_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Please count the number of permutations of the integers from 0 to {N_minus_1}, denoted as p[0], p[1], ..., p[{N_minus_1}], such that the following {N_minus_1} constraints are satisfied: {constraints}
Note that each constraint above is of the form `p[i] < p[j]` or `p[i] > p[j]`, and collectively, these constraints correspond to a tree — that is, a connected undirected graph with no cycles — on {N} vertices labeled from 0 to {N_minus_1}.
You should output the number of valid permutations modulo {MOD}."""
    def __init__(self,
                 max_MOD : int = 1000000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the TreeTopologicalSequenceCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_MOD = max_MOD
        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)

        p = list(range(N))
        random.shuffle(p)

        edges = self.parameter["edges"] = []
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u, "<" if p[u] < p[v] else ">", v))
        random.shuffle(edges)

        for u, w, v in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, w, v in edges)) == N - 1

        tree = networkx.Graph()
        tree.add_edges_from((u, v) for u, w, v in edges)
        assert networkx.is_tree(tree)


        # Precompute binomial coefficients up to maxN
        C = [[0] * (N + 1) for _ in range(N + 1)]
        for i in range(N + 1):
            C[i][0] = 1
            for j in range(1, i + 1):
                C[i][j] = (C[i-1][j-1] + C[i-1][j]) % MOD

        def dfs(u, parent, h1, h2):
            # f_raw[k]: number of ways (raw) to have exactly k nodes before u
            f_raw = [0, 1]   # only u itself => 1 way with k=1
            sz = 1           # size of subtree rooted at u

            # First, merge all children v where u < v (v must come after u)
            for v in h1[u]:
                if v == parent:
                    continue
                f_v, sz_v = dfs(v, u, h1, h2)
                g = f_raw[:]          # copy old
                new_sz = sz + sz_v
                new_f = [0] * (new_sz + 1)
                for j in range(1, sz + 1):
                    gj = g[j]
                    if gj == 0:
                        continue
                    for i_count in range(j, sz_v + j):
                        # Combine with child-subtree counts that place at least (i_count-j+1) before v
                        diff = f_v[sz_v] - f_v[i_count - j]
                        if diff < 0:
                            diff += MOD
                        term = gj
                        term = term * C[i_count - 1][j - 1] % MOD
                        term = term * C[sz + sz_v - i_count][sz - j] % MOD
                        term = term * diff % MOD
                        new_f[i_count] = (new_f[i_count] + term) % MOD
                f_raw = new_f
                sz = new_sz

            # Then, merge all children v where u > v (v must come before u)
            for v in h2[u]:
                if v == parent:
                    continue
                f_v, sz_v = dfs(v, u, h1, h2)
                g = f_raw[:]
                new_sz = sz + sz_v
                new_f = [0] * (new_sz + 1)
                for j in range(1, sz + 1):
                    gj = g[j]
                    if gj == 0:
                        continue
                    for i_count in range(j + 1, sz_v + j + 1):
                        # Combine with child-subtree counts that place exactly (i_count-j) before v
                        term = gj
                        term = term * C[i_count - 1][j - 1] % MOD
                        term = term * C[sz + sz_v - i_count][sz - j] % MOD
                        term = term * f_v[i_count - j] % MOD
                        new_f[i_count] = (new_f[i_count] + term) % MOD
                f_raw = new_f
                sz = new_sz

            # Turn raw counts into prefix-sums: f_pref[k] = sum_{t=1..k} f_raw[t]
            f_pref = [0] * (sz + 1)
            for i_count in range(1, sz + 1):
                s = f_pref[i_count - 1] + f_raw[i_count]
                if s >= MOD:
                    s -= MOD
                f_pref[i_count] = s

            return f_pref, sz

        # Build directed adjacency lists
        h1 = [[] for _ in range(N + 1)]
        h2 = [[] for _ in range(N + 1)]
        for a, sign, b in edges:
            x, y = a + 1, b + 1
            if sign == '<':
                h1[x].append(y)
                h2[y].append(x)
            else:
                h1[y].append(x)
                h2[x].append(y)

        f_root, _ = dfs(1, 0, h1, h2)
        # The answer is the number of ways to have all N nodes before root (i.e. full ordering)
        self.parameter["reference_answer"] = f_root[N] % MOD

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            constraints = "; ".join("p[{}] {} p[{}]".format(u, w, v) for u, w, v in self.parameter["edges"]),
            MOD = self.parameter["MOD"],
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
            if not (0 <= processed_result < self.parameter["MOD"]) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]