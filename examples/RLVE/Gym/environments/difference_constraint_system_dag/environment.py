import random
from collections import deque
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class DifferenceConstraintSystemDAG_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""There are {N} **positive integers** x[0], x[1], ..., x[{N_minus_1}]. They satisfy the following {M} equations/inequations:
{relations}

Please find any solution x[0], x[1], ..., x[{N_minus_1}] that satisfies all of the equations/inequations. Try your best to minimize x[0] + x[1] + ... + x[{N_minus_1}].

Output Format: Your final answer should be a single line containing x[0], x[1], ..., x[{N_minus_1}], separated by **spaces**."""

    def __init__(self,
                 wrong_format : float = -1.0,
                 invalid_solution : float = 0.0,
                 rewarding_strategy_relation : str = "(satisfied/all)^beta", rewarding_weight_relation : float = +0.5, rewarding_beta_relation : float = 5.0,
                 rewarding_strategy_sum : str = "(gold/answer)^beta", rewarding_weight_sum : float = +0.5, rewarding_beta_sum : float = 5.0,
                 **kwargs) :
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy_relation" : rewarding_strategy_relation,
            "rewarding_weight_relation" : rewarding_weight_relation,
            "rewarding_beta_relation" : rewarding_beta_relation,
            "rewarding_strategy_sum" : rewarding_strategy_sum,
            "rewarding_weight_sum" : rewarding_weight_sum,
            "rewarding_beta_sum" : rewarding_beta_sum,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 1, "M should be greater than or equal to 1"

        Xs = [random.randint(1, N) for i in range(N)]

        relations = self.parameter["relations"] = random.sample([(i, j) for i in range(N) for j in range(N) if i != j], min(M, N * (N - 1)))
        '''
        X = 1: A = B
        X = 2: A < B
        X = 3: A ≥ B
        X = 4: A > B
        X = 5: A ≤ B
        '''
        for i, (A, B) in enumerate(relations) :
            if Xs[A] == Xs[B] :
                X_choices = (1, 3, 5)
            elif Xs[A] < Xs[B] :
                X_choices = (2, 5)
            elif Xs[A] > Xs[B] :
                X_choices = (3, 4)
            else :
                assert False, "Invalid relation: X[{}]={} and X[{}]={}".format(A, Xs[A], B, Xs[B])
            relations[i] = (random.choice(X_choices), A, B)
        
        
        adj = [[] for _ in range(N)]              # adjacency[u] = list[(v, w)]

        for X, A, B in relations:
            if X == 1:                          # equal
                adj[A].append((B, 0))
                adj[B].append((A, 0))
            elif X == 2:                          # A < B   ⇒  A→B, +1
                adj[A].append((B, 1))
            elif X == 3:                          # A ≥ B   ⇒  B→A, +0
                adj[B].append((A, 0))
            elif X == 4:                          # A > B   ⇒  B→A, +1
                adj[B].append((A, 1))
            else:                                 # X == 5   A ≤ B ⇒  A→B, +0
                adj[A].append((B, 0))

        # ---------- Tarjan SCC ----------
        dfn = [-1] * N
        low = [0] * N
        stack, in_stk = [], [False] * N
        scc_id = [-1] * N
        time = 0
        sizes = []                                # size per component
        scc_cnt = 0

        def tarjan(u: int):
            nonlocal time, scc_cnt
            dfn[u] = low[u] = time
            time += 1
            stack.append(u)
            in_stk[u] = True

            for v, _ in adj[u]:
                if dfn[v] == -1:
                    tarjan(v)
                    low[u] = min(low[u], low[v])
                elif in_stk[v]:
                    low[u] = min(low[u], dfn[v])

            if low[u] == dfn[u]:                  # root of an SCC
                sizes.append(0)
                while True:
                    node = stack.pop()
                    in_stk[node] = False
                    scc_id[node] = scc_cnt
                    sizes[scc_cnt] += 1
                    if node == u:
                        break
                scc_cnt += 1

        for i in range(N):
            if dfn[i] == -1:
                tarjan(i)

        # ---------- build condensed DAG ----------
        dag = [[] for _ in range(scc_cnt)]
        indeg = [0] * scc_cnt

        for u in range(N):
            su = scc_id[u]
            for v, w in adj[u]:
                sv = scc_id[v]
                if su == sv:
                    if w == 1:                    # c ≥ c + 1  impossible
                        assert False, "Impossible relation: c >= c + 1"
                else:
                    dag[su].append((sv, w))
                    indeg[sv] += 1

        # ---------- longest path on DAG ----------
        dp = [0] * scc_cnt
        q = deque(i for i in range(scc_cnt) if indeg[i] == 0)
        for i in q:                               # sources start at 1 candy
            dp[i] = 1

        while q:
            u = q.popleft()
            for v, w in dag[u]:
                if dp[v] < dp[u] + w:
                    dp[v] = dp[u] + w
                indeg[v] -= 1
                if indeg[v] == 0:
                    if dp[v] == 0:                # isolated source
                        dp[v] = 1
                    q.append(v)

        # ---------- final answer ----------
        self.parameter["reference_answer"] = " ".join(str(dp[scc_id[i]]) for i in range(N))
        self.parameter["gold_answer"] = sum(dp[comp] * sizes[comp] for comp in range(scc_cnt))
        assert self.parameter["gold_answer"] == sum(map(int, self.parameter["reference_answer"].split())) <= sum(Xs), "Gold answer should be less than or equal to sum(X)"


    def _prompt_generate(self) -> str :
        X2symbol = {
            1 : "=",
            2 : "<",
            3 : "≥",
            4 : ">",
            5 : "≤",
        }
        return self.prompt_template.format(
            N = self.parameter["N"],
            N_minus_1 = self.parameter["N"] - 1,
            M = self.parameter["M"],
            relations = "\n".join("x[{}] {} x[{}]".format(A, X2symbol[X], B) for X, A, B in self.parameter["relations"]),
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

            x = processed_result
            if len(x) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            if not all(xi >= 1 for xi in x) :
                return self.rewards["invalid_solution"]
            
            
            reward = 0.0

            X2function = {
                1 : lambda a, b: a == b,
                2 : lambda a, b: a < b,
                3 : lambda a, b: a >= b,
                4 : lambda a, b: a > b,
                5 : lambda a, b: a <= b,
            }
            satisfied = sum(int(X2function[X](x[A], x[B])) for X, A, B in self.parameter["relations"])
            assert satisfied <= len(self.parameter["relations"]), "satisfied should be less than or equal to the number of relations"
            if self.rewards["rewarding_strategy_relation"] == "(satisfied/all)^beta" :
                reward += self.rewards["rewarding_weight_relation"] * ((satisfied / len(self.parameter["relations"])) ** self.rewards["rewarding_beta_relation"])
            elif self.rewards["rewarding_strategy_relation"] == "satisfied=all" :
                reward += self.rewards["rewarding_weight_relation"] * (satisfied == len(self.parameter["relations"]))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy_relation"]))

            if satisfied == len(self.parameter["relations"]) :
                gold, answer = self.parameter["gold_answer"], sum(x)
                assert gold <= answer, "Gold answer should be less than or equal to the answer"
                if self.rewards["rewarding_strategy_sum"] == "(gold/answer)^beta" :
                    reward += self.rewards["rewarding_weight_sum"] * ((gold / answer) ** self.rewards["rewarding_beta_sum"])
                elif self.rewards["rewarding_strategy_sum"] == "gold=answer" :
                    reward += self.rewards["rewarding_weight_sum"] * (gold == answer)
                else :
                    raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy_sum"]))
            
            return reward
        else :
            return self.rewards["wrong_format"]