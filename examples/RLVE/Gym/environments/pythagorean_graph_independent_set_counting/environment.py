import math
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class PythagoreanGraph_IndependentSetCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3213
    prompt_template = \
r"""You are given an array H of length {N}: {H}
Construct an undirected graph with vertices labeled from 0 to {N_minus_1}. There is an edge between vertex i and vertex j (i ≠ j) if and only if:
- There exists an integer C such that H[i]^2 + H[j]^2 = C^2
- gcd(H[i], H[j]) = 1 (i.e., H[i] and H[j] are coprime)

Your task is to count the number of **non-empty independent sets** in this graph — that is, subsets of vertices such that no two vertices in the subset are connected by an edge.

**Output Format:** Output a single integer — the number of non-empty independent sets modulo {MOD}."""

    def __init__(self,
                 max_MOD : int = 1000000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the PythagoreanGraph_IndependentSetCounting_Environment instance.
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
        
        while True :
            H = self.parameter["H"] = [random.randint(1, 2 * N) for _ in range(N)]

            hs = H
            # Count sticks of each length
            maxH = max(hs)
            num = [0] * (maxH + 1)
            for h in hs:
                num[h] += 1

            # Precompute powers of 2 up to N
            PW2 = [1] * (N + 1)
            for i in range(1, N + 1):
                PW2[i] = (PW2[i-1] * 2) % MOD

            # Build adjacency lists for primitive Pythagorean pairs
            to = [[] for _ in range(maxH + 1)]
            limit_i = int(math.isqrt(maxH))
            two_max = 2 * maxH
            for i in range(1, limit_i + 1):
                # j > i, 2*i*j <= maxH, j*j <= 2*maxH
                # so j_max = min(maxH//(2*i), int(sqrt(2*maxH)))
                j_max1 = maxH // (2*i)
                j_max2 = int(math.isqrt(two_max))
                j_max = min(j_max1, j_max2)
                for j in range(i+1, j_max+1):
                    x = j*j - i*i
                    y = 2*i*j
                    # we already ensured y <= maxH by j_max1, and j*j <= 2*maxH by j_max2
                    if x > maxH or y > maxH:
                        continue
                    if num[x] == 0 or num[y] == 0:
                        continue
                    if math.gcd(x, y) != 1:
                        continue
                    to[x].append(y)
                    to[y].append(x)

            # Arrays for DFS and DP
            vis = [False] * (maxH + 1)
            ins = [False] * (maxH + 1)
            sat = [0]     * (maxH + 1)
            des = [0]     * (maxH + 1)
            dp0 = [0]     * (maxH + 1)
            dp1 = [0]     * (maxH + 1)
            QE = []   # cycle nodes
            pnt = 0  # stamp for dp traversal

            # Find all back-edges to detect cycle nodes
            def dfs_init(u, parent):
                vis[u] = True
                for v in to[u]:
                    if v == parent:
                        continue
                    if not vis[v]:
                        dfs_init(v, u)
                    else:
                        # found a back-edge u-v
                        if not ins[u]:
                            QE.append(u)
                        if not ins[v]:
                            QE.append(v)
                        ins[u] = ins[v] = True

            # Check that no two forced-selected cycle nodes are adjacent
            def check():
                for u in QE:
                    if sat[u] == 1:
                        for v in to[u]:
                            if sat[v] == 1:
                                return False
                return True

            # Tree-DP for counting valid selections in a rooted tree
            def dfs_dp(u):
                nonlocal pnt
                dp0[u] = 1
                dp1[u] = (PW2[num[u]] - 1) % MOD
                des[u] = pnt
                for v in to[u]:
                    if des[v] != pnt:
                        dfs_dp(v)
                        dp0[u] = dp0[u] * (dp0[v] + dp1[v]) % MOD
                        dp1[u] = dp1[u] * dp0[v]            % MOD
                # apply forced-status constraints
                if sat[u] ==  1:
                    dp0[u] = 0
                if sat[u] == -1:
                    dp1[u] = 0
                return (dp0[u] + dp1[u]) % MOD

            # Solve one connected component
            def query(root):
                nonlocal pnt
                QE.clear()
                dfs_init(root, root)

                comp_ans = 0
                k = len(QE)
                # Enumerate all ways to force-select or force-skip the cycle nodes
                for mask in range(1 << k):
                    for i in range(k):
                        u = QE[i]
                        sat[u] = 1 if (mask >> i) & 1 else -1
                    if not check():
                        continue
                    pnt += 1
                    comp_ans = (comp_ans + dfs_dp(root)) % MOD

                # reset sat flags
                for u in QE:
                    sat[u] = 0
                return comp_ans

            # Main loop over all lengths
            answer = 1
            for length in range(1, maxH + 1):
                if num[length] > 0 and not vis[length]:
                    if not to[length]:
                        # isolated node: any subset of its sticks
                        answer = answer * PW2[num[length]] % MOD
                        vis[length] = True
                    else:
                        answer = answer * query(length) % MOD

            # subtract empty set
            if answer != PW2[N] :
                self.parameter["reference_answer"] = (answer - 1) % MOD
                break
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            H = " ".join("H[{}]={}".format(i, Hi) for i, Hi in enumerate(self.parameter["H"])),
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