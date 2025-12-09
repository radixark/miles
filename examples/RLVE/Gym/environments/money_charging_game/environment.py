import random
import networkx
from typing import Optional
from Gym.environment import VerifiableEnvironment


class MoneyChargingGame_Environment(VerifiableEnvironment) : # https://www.luogu.com.cn/problem/P5405
    prompt_template = \
r"""There are {N} nodes, each associated with values A[i][1], A[i][2], and A[i][3]. For each node `i`, define: P[i][j] = A[i][j] / (A[i][1] + A[i][2] + A[i][3]) for j = 1, 2, 3. The values A are given as follows:
{A}

We define the following random process:
1. For each node `i`, randomly assign W[i] = j with probability P[i][j] for j = 1, 2, 3.
2. Starting from an empty set, repeatedly select a node `i` with probability proportional to W[i], and add it to the set (duplicates are allowed). Continue until all nodes are in the set.
3. Let T[i] denote the first time node `i` is added to the set.

You are also given a set of constraints (each of the form T[u] < T[v]) that correspond to the edges of an undirected tree:
{T_inequalities}
Please compute the total probability that all the above T[u] < T[v] conditions hold during the random process. Output the result modulo {MOD}."""
    MOD = 998244353

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the MoneyChargingGame_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        A, B, C = self.parameter["A"], self.parameter["B"], self.parameter["C"] = [random.randint(1, N) for u in range(N)], [random.randint(1, N) for u in range(N)], [random.randint(1, N) for u in range(N)]

        T_inequalities = self.parameter["T_inequalities"] = []
        permutation = list(range(N))
        swap_probability = random.random()
        random.shuffle(permutation)
        for i in range(1, N) :
            u = permutation[random.randint(0, i - 1)]
            v = permutation[i]
            if random.random() < swap_probability :
                u, v = v, u
            T_inequalities.append((u, v))
        random.shuffle(T_inequalities)

        assert len(T_inequalities) == N - 1, "T_inequalities should have exactly N-1 elements"
        assert len(T_inequalities) == len(set(T_inequalities)), "T_inequalities should not have duplicates"
        for u, v in T_inequalities :
            assert 0 <= u < N and 0 <= v < N, "T_inequalities should contain valid indices"
            assert u != v, "T_inequalities should not contain self-loops"
        tree = networkx.Graph()
        tree.add_edges_from((T_inequalities))
        assert networkx.is_tree(tree)


        S = []
        for a1, a2, a3 in zip(A, B, C):
            total = a1 + a2 + a3
            S.append(pow(total, self.MOD - 2, self.MOD))

        # 2) precompute inverses of 1..3N
        invs = [0] * (3 * N + 1)
        for k in range(1, 3 * N + 1):
            invs[k] = pow(k, self.MOD - 2, self.MOD)

        # 3) build the tree (0-indexed) with flags
        G = [[] for _ in range(N)]
        for u, v in T_inequalities :
            G[v].append((u, 1))
            G[u].append((v, 0))

        # 4) DP arrays
        f = [None] * N
        size = [0] * N

        def dfs(x, parent):
            size[x] = 1
            # fx[k] will hold the *unnormalized* convolution numerator
            fx = [0] * (3 * size[x] + 1)
            fx[1] = A[x] * S[x] % self.MOD
            fx[2] = B[x] * S[x] % self.MOD * 2 % self.MOD
            fx[3] = C[x] * S[x] % self.MOD * 3 % self.MOD

            # merge in each child
            for (v, t) in G[x]:
                if v == parent:
                    continue
                dfs(v, x)
                fy = f[v]

                new_size = size[x] + size[v]
                tmp = [0] * (3 * new_size + 1)

                # convolution with the “subtract-and-redistribute” if t==1
                for i in range(1, size[x] * 3 + 1):
                    if fx[i] == 0:
                        continue
                    for j in range(1, size[v] * 3 + 1):
                        res = fx[i] * fy[j] % self.MOD
                        if t:
                            tmp[i + j] = (tmp[i + j] - res) % self.MOD
                            tmp[i]     = (tmp[i]     + res) % self.MOD
                        else:
                            tmp[i + j] = (tmp[i + j] + res) % self.MOD

                size[x] = new_size
                fx = tmp

            # 5) **one** division pass, _after_ all children are merged
            for k in range(1, size[x] * 3 + 1):
                fx[k] = fx[k] * invs[k] % self.MOD

            f[x] = fx

        # 6) run and collect answer
        dfs(0, -1)
        self.parameter["reference_answer"] = sum(f[0][1 : 3 * size[0] + 1]) % self.MOD


    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = "\n".join("A[{}][1, 2, 3] = [{}, {}, {}]".format(i, a, b, c) for i, (a, b, c) in enumerate(zip(self.parameter["A"], self.parameter["B"], self.parameter["C"]))),
            T_inequalities = "\n".join("T[{}] < T[{}]".format(u, v) for u, v in self.parameter["T_inequalities"]),
            MOD = self.MOD,
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
            if not (0 <= processed_result < self.MOD) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]