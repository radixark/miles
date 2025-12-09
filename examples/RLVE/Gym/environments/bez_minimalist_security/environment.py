import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class BEZMinimalistSecurity_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3544
    prompt_template = \
r"""There is an array P of length {N}. Initially, P is: {P}

Now we want to construct a new array P' of length {N}, where 0 <= P'[i] <= P[i] for all i. Additionally, there are some constraints of the form P'[u] + P'[v] = w, where u and v are indices and w is a constant (it is guaranteed that P[u] + P[v] >= w). The constraints are:
{constraints}

Please output P'[0], P'[1], ..., P'[{N_minus_1}], separated by spaces, such that they satisfy all the constraints and their sum is {minimized_or_maximized}."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5,
                 rewarding_strategy_min : str = "(gold/answer)^beta", rewarding_weight_min : float = +1.0, rewarding_beta_min : float = 5.0,
                 rewarding_strategy_max : str = "(answer/gold)^beta", rewarding_weight_max : float = +1.0, rewarding_beta_max : float = 5.0,
                 **kwargs) :
        """
        Initialize the BEZMinimalistSecurity_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy_max" : rewarding_strategy_max,
            "rewarding_weight_max" : rewarding_weight_max,
            "rewarding_beta_max" : rewarding_beta_max,
            "rewarding_strategy_min" : rewarding_strategy_min,
            "rewarding_weight_min" : rewarding_weight_min,
            "rewarding_beta_min" : rewarding_beta_min,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be at least 3"

        P_prime = [random.randint(0, N) for _ in range(N)]

        assert "edge_ratio" in self.parameter, "edge_ratio is required in parameter"
        edge_ratio = self.parameter["edge_ratio"]
        edges = self.parameter["edges"] = random.sample([(u, v, P_prime[u] + P_prime[v]) for u in range(N) for v in range(u + 1, N)], max(1, min(N * (N - 1) // 2, int(edge_ratio * N))))
        random.shuffle(edges)
        for u, v, w in edges :
            assert 0 <= u < v < N
        assert len(edges) == len(set((u, v) for u, v, w in edges)), "edges should be unique"

        P = self.parameter["P"] = [P_prime_u + random.randint(0, N) for P_prime_u in P_prime]


        # Build adjacency list (0-indexed)
        adjacency = [[] for _ in range(N)]
        for u, v, w in edges:
            adjacency[u].append((v, w))
            adjacency[v].append((u, w))

        vis = [False] * N
        sgn = [0] * N
        cons = [0] * N
        q = [0] * N
        mn = 0
        mx = 0

        def wa() :
            assert False, "Invalid solution"

        def dfs(u):  # Depth-first search on component
            nonlocal fix
            vis[u] = True
            stc.append(u)
            # Early exit if constraint too large
            if cons[u] > 10**6:
                wa()
            for v, w in adjacency[u]:
                if not vis[v]:
                    sgn[v] = -sgn[u]
                    cons[v] = w - cons[u]
                    dfs(v)
                else:
                    if sgn[u] == sgn[v]:
                        res = w - cons[u] - cons[v]
                        # Must be even
                        if res & 1:
                            wa()
                        denom = 2 * sgn[u]
                        res //= denom
                        # Check valid fixed value
                        if res < 0 or res > P[anc] or (fix is not None and fix != res):
                            wa()
                        fix = res
                    else:
                        # Sum of constants must match
                        if cons[u] + cons[v] != w:
                            wa()

        # Process each connected component
        for i in range(N):
            if not vis[i]:
                stc = []           # nodes in current component
                anc = i           # anchor node for fixed value range
                fix = None        # fixed solution parameter
                sgn[i] = 1        # sign for anchor
                cons[i] = 0       # constant offset for anchor
                dfs(i)

                if fix is not None:
                    # Unique solution determined by `fix`
                    for u in stc:
                        q[u] = sgn[u] * fix + cons[u]
                        delta = P[u] - q[u]
                        mn += delta
                        mx += delta
                        if q[u] < 0 or q[u] > P[u]:
                            wa()
                    # Verify edges
                    for u in stc:
                        for v, w in adjacency[u]:
                            if q[u] + q[v] != w:
                                wa()
                else:
                    # Range of valid `fix` values [l, r]
                    l, r = 0, P[anc]
                    for u in stc:
                        if sgn[u] == 1:
                            l = max(l, -cons[u])
                            r = min(r, P[u] - cons[u])
                        else:
                            l = max(l, cons[u] - P[u])
                            r = min(r, cons[u])
                    if l > r:
                        wa()
                    # Compute sum of reductions for minimal `fix = l`
                    base_sum = 0
                    tsign = 0
                    for u in stc:
                        base_sum += P[u] - (l * sgn[u] + cons[u])
                        tsign -= sgn[u]
                    # Depending on tsign, extremes at l or r
                    if tsign > 0:
                        mx += base_sum + tsign * (r - l)
                        mn += base_sum
                    else:
                        mx += base_sum
                        mn += base_sum + tsign * (r - l)

        self.parameter["minimized_or_maximized"] = random.choice(["minimized", "maximized"])
        if self.parameter["minimized_or_maximized"] == "minimized" :
            self.parameter["gold_answer"] = sum(P) - mx
        elif self.parameter["minimized_or_maximized"] == "maximized" :
            self.parameter["gold_answer"] = sum(P) - mn
        else :
            raise ValueError("minimized_or_maximized should be either 'minimized' or 'maximized'")
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            P = " ".join("P[{}]={}".format(i, P_i) for i, P_i in enumerate(self.parameter["P"])),
            constraints = "\n".join("P'[{}] + P'[{}] = {}".format(u, v, w) for u, v, w in self.parameter["edges"]),
            minimized_or_maximized = self.parameter["minimized_or_maximized"],
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

            P_prime = processed_result
            if len(P_prime) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(0 <= P_prime_u <= P_u for P_prime_u, P_u in zip(P_prime, self.parameter["P"])) :
                return self.rewards["invalid_solution"]
            if not all(P_prime[u] + P_prime[v] == w for u, v, w in self.parameter["edges"]) :
                return self.rewards["invalid_solution"]
            
            gold, answer = self.parameter["gold_answer"], sum(P_prime)
            if self.parameter["minimized_or_maximized"] == "minimized" :
                assert 0 <= gold <= answer, "For minimization, answer should be greater than 0 and at least as large as the gold answer"
                if self.rewards["rewarding_strategy_min"] == "(gold/answer)^beta" :
                    if answer == 0 :
                        assert gold == 0, "If answer is 0, gold should also be 0"
                        return self.rewards["rewarding_weight_min"] * 1.0
                    return self.rewards["rewarding_weight_min"] * ((gold / answer) ** self.rewards["rewarding_beta_min"])
                elif self.rewards["rewarding_strategy_min"] == "gold=answer" :
                    return self.rewards["rewarding_weight_min"] * (gold == answer)
                else :
                    raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy_min"]))
            elif self.parameter["minimized_or_maximized"] == "maximized" :
                assert 0 <= answer <= gold, "For maximization, answer should be greater than 0 and at most as large as the gold answer"
                if self.rewards["rewarding_strategy_max"] == "(answer/gold)^beta" :
                    if gold == 0 :
                        assert answer == 0, "If gold is 0, answer should also be 0"
                        return self.rewards["rewarding_weight_max"] * 1.0
                    return self.rewards["rewarding_weight_max"] * ((answer / gold) ** self.rewards["rewarding_beta_max"])
                elif self.rewards["rewarding_strategy_max"] == "gold=answer" :
                    return self.rewards["rewarding_weight_max"] * (gold == answer)
                else :
                    raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy_max"]))
            else :
                assert False, "minimize_or_maximize should be either 'minimize' or 'maximize'"
        else :
            return self.rewards["wrong_format"]