import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class WYC_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3597
    prompt_template = \
r"""You are given a **directed graph** with {N} vertices (labeled from 1 to {N}). Each edge is represented as a tuple (s, t, w), meaning there is a directed edge from vertex `s` to vertex `t` with weight `w`. It is guaranteed that each weight `w` is either 1, 2, or 3. The list of edges is:
{edges}

Considering **all possible paths** in this graph that consist of at least one edge (a path may start and end at any vertex, and may visit vertices or edges multiple times), sort all such paths by their total edge weight in **non-decreasing order**. Output a single integer - the total weight of the {K}-th path in the sorted list."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the WYC_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K >= 1, "MAX_K should be greater than or equal to 1"


        while True :
            edges = self.parameter["edges"] = []
            for edge_index in range(random.randint(1, N * (N - 1))) :
                s, t = random.sample(range(1, N + 1), 2)
                edges.append((s, t, random.randint(1, 3)))
            random.shuffle(edges)
            for s, t, w in edges :
                assert 1 <= s <= N and 1 <= t <= N and s != t
            
            K = self.parameter["K"] = random.randint(1, MAX_K)


            def mat_mult(X, Y, cap):
                """
                Multiply two square matrices X and Y of the same dimension, capping all entries at `cap`.
                """
                D = len(X)
                Z = [[0] * D for _ in range(D)]
                for i in range(D):
                    Xi = X[i]
                    Zi = Z[i]
                    for k, Xik in enumerate(Xi):
                        if Xik:
                            Yk = Y[k]
                            for j in range(D):
                                Zi[j] += Xik * Yk[j]
                                if Zi[j] > cap:
                                    Zi[j] = cap
                return Z

            def vec_mat_mult(v, M, cap):
                """
                Multiply a row vector v by matrix M, capping all entries at `cap`.
                Returns a new row vector.
                """
                D = len(v)
                w = [0] * D
                for k, vk in enumerate(v):
                    if vk:
                        Mk = M[k]
                        for j in range(D):
                            w[j] += vk * Mk[j]
                            if w[j] > cap:
                                w[j] = cap
                return w
            
            def compute_answer() :
                # dimension of the expanded state space
                D = 3 * N + 1
                # cap counts at K + N so we never need values above that
                cap = K + N

                # build the base adjacency matrix g0 (size D x D)
                g0 = [[0] * D for _ in range(D)]
                # self-loop at state 0
                g0[0][0] = 1

                # initial row-vector A of length D
                A = [0] * D
                # set up waiting chains and finishing transitions
                for i in range(N):
                    idx1 = i * 3 + 1
                    idx2 = idx1 + 1
                    idx3 = idx1 + 2
                    A[idx1] = 1           # can start at any vertex
                    g0[idx1][0] = 1       # from "just arrived" to finish
                    g0[idx2][idx1] = 1    # wait one unit
                    g0[idx3][idx2] = 1    # wait two units

                # read the edges and add the entry-point transitions
                for u, v, w in edges:
                    u_idx = (u - 1) * 3 + 1
                    v_idx = (v - 1) * 3 + w
                    g0[u_idx][v_idx] += 1

                # store powers g[d] = g0^(2^d)
                g = [g0]

                # determine how many bits are needed instead of a fixed 64
                max_bits = max(1, K.bit_length()) * 2

                # find highest d such that number of paths of length ≤ 2^d is ≥ K
                d = 0
                while True:
                    if d >= max_bits:
                        # even at length 2^max_bits we don't reach K paths
                        return -1
                    g.append(mat_mult(g[d], g[d], cap))
                    d += 1
                    tmp = vec_mat_mult(A, g[d], cap)
                    # subtract N trivial finishes
                    if tmp[0] - N >= K:
                        break

                # binary-lift to find exact length
                ans = 0
                for bit in range(d, -1, -1):
                    tmp = vec_mat_mult(A, g[bit], cap)
                    if tmp[0] - N < K:
                        A = tmp
                        ans += 1 << bit

                return ans

            self.parameter["reference_answer"] = compute_answer()
            if self.parameter["reference_answer"] != -1 :
                break
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            edges = "\n".join("({}, {}, {})".format(s, t, w) for s, t, w in self.parameter["edges"]),
            K = self.parameter["K"],
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
            if processed_result <= 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]