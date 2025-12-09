import math
import random
from typing import Optional
from collections import Counter
from collections import defaultdict
from Gym.environment import VerifiableEnvironment


class SumSpanningTreeGCD_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given an **undirected graph** with {N} vertices, labeled from `1` to `{N}`. The graph contains the following undirected edges. Each edge is represented as a tuple `(u, v, w)`, meaning an undirected edge **connecting vertex u to vertex v with weight w**:
{edges}

Consider a subset of edges `T = [(u_1, v_1, w_1), (u_2, v_2, w_2), ..., (u_k, v_k, w_k)]` such that:
- k = {N_minus_1} (i.e., you select exactly {N_minus_1} edges),
- The selected edges form a **spanning tree** — that is, they connect all {N} vertices without forming any cycles,
- The value of this spanning tree is defined as the **greatest common divisor (GCD)** of the weights of the edges in `T`, i.e., `gcd(w_1, w_2, ..., w_k)`.

What is **the sum value** of all such spanning trees modulo {MOD}?"""
    MODs = (666623333, 998244353, 10 ** 9 + 7)

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the SumSpanningTreeGCD_Environment instance.
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
        assert N >= 3, "N should be greater than or equal to 3"

        assert "edge_density" in self.parameter, "edge_density is required in parameter"
        edge_density = self.parameter["edge_density"]
        assert 0.0 <= edge_density <= 1.0, "edge_density should be between 0.0 and 1.0"

        edges = self.parameter["edges"] = []

        common_d = random.randint(1, N)
        permutations = list(range(N))
        random.shuffle(permutations)
        for index, vertex in enumerate(permutations) :
            if index == 0 :
                continue
            u, v = vertex, random.choice(permutations[: index])
            u, v = min(u, v), max(u, v)
            edges.append((u + 1, v + 1, common_d * random.randint(1, N)))
        
        num_edges = int(edge_density * N * (N - 1) / 2)
        if len(edges) < num_edges :
            remaining_edges = list(set((u, v) for u in range(1, N + 1) for v in range(u + 1, N + 1)) - set((u, v) for u, v, w in edges))
            remaining_edges = random.sample(remaining_edges, min(len(remaining_edges), num_edges - len(edges)))
            for u, v in remaining_edges :
                edges.append((u, v, random.randint(1, N * N)))
        random.shuffle(edges)

        for u, v, w in edges :
            assert 1 <= u < v <= N
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "edges should be unique"
    
        MOD = self.parameter["MOD"] = random.choice(self.MODs)
    

        weight_counts = Counter()
        edges = []
        for u, v, w in self.parameter["edges"] :
            edges.append((u-1, v-1, w))
            weight_counts[w] += 1

        # 2) Precompute small primes for trial division up to sqrt(max_w)
        max_w = max(weight_counts) if weight_counts else 0
        limit = int(math.isqrt(max_w)) + 1
        sieve = [True] * (limit+1)
        primes = []
        for i in range(2, limit+1):
            if sieve[i]:
                primes.append(i)
                for j in range(i*i, limit+1, i):
                    sieve[j] = False

        # 3) Build S[d] = number of edges whose weight is divisible by d,
        #    and phi_map[d] = φ(d) for all divisors d that appear.
        S = defaultdict(int)
        phi_map = {}

        def gen_divisors(idx, cur_d, cur_phi, factors, cnt):
            """Recursively generate all divisors of a weight w and accumulate S, phi_map."""
            if idx == len(factors):
                S[cur_d] += cnt
                if cur_d not in phi_map:
                    phi_map[cur_d] = cur_phi
                return
            p, e = factors[idx]
            # exponent = 0
            gen_divisors(idx+1, cur_d, cur_phi, factors, cnt)
            # exponents 1..e
            p_pow = 1
            for k in range(1, e+1):
                p_pow *= p
                # φ(p^k) = p^k - p^(k-1)
                factor = p_pow - (p_pow // p)
                gen_divisors(idx+1, cur_d * p_pow, cur_phi * factor, factors, cnt)

        for w, cnt in weight_counts.items():
            # factor w into primes
            x = w
            factors = []
            for p in primes:
                if p*p > x:
                    break
                if x % p == 0:
                    e = 0
                    while x % p == 0:
                        x //= p
                        e += 1
                    factors.append((p, e))
            if x > 1:
                factors.append((x, 1))
            # generate its divisors
            gen_divisors(0, 1, 1, factors, cnt)

        # 4) Collect all d for which we have at least N-1 edges divisible by d
        candidates = [d for d, cnt in S.items() if cnt >= N-1]
        candidates.sort()

        # 5) Define a function to compute the number of spanning trees
        #    in the subgraph of edges whose weight divides d, via Kirchhoff + Gauss.
        def solve_for_d(d):
            dim = N - 1
            # build the (N-1)x(N-1) Laplacian minor
            G = [[0]*dim for _ in range(dim)]
            for u, v, w in edges:
                if w % d != 0 or u == v:
                    continue
                # only update if endpoint != the excluded node (index N-1)
                if u < dim and v < dim:
                    G[u][u] += 1
                    G[v][v] += 1
                    G[u][v] -= 1
                    G[v][u] -= 1
                elif u < dim:
                    G[u][u] += 1
                elif v < dim:
                    G[v][v] += 1
            # reduce modulo
            for i in range(dim):
                for j in range(dim):
                    G[i][j] %= MOD

            # Gaussian elimination to compute determinant MOD
            det = 1
            for i in range(dim):
                # pivot if needed
                if G[i][i] == 0:
                    for j in range(i+1, dim):
                        if G[j][i]:
                            G[i], G[j] = G[j], G[i]
                            det = -det % MOD
                            break
                    else:
                        return 0
                ai = G[i][i]
                det = det * ai % MOD
                inv = pow(ai, MOD-2, MOD)
                # eliminate below
                for j in range(i+1, dim):
                    if G[j][i]:
                        factor = G[j][i] * inv % MOD
                        row_i = G[i]
                        row_j = G[j]
                        for k in range(i, dim):
                            row_j[k] = (row_j[k] - factor * row_i[k]) % MOD
            return det

        # 6) Sum up φ(d) * (# of trees using only edges ≡ 0 MOD d)
        ans = 0
        for d in candidates:
            ans = (ans + phi_map[d] * solve_for_d(d)) % MOD

        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            edges = "\n".join("({}, {}, {})".format(u, v, w) for u, v, w in self.parameter["edges"]),
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