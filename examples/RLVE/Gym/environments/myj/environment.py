import random
from bisect import bisect_left
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MYJ_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3592
    prompt_template = \
r"""There are {N} shops labeled from 1 to {N} (from left to right); every shop has a price, and the price of an item at shop i is P[i]. There are {M} customers; each customer is represented by a tuple (a, b, c); the customer will consider buying the item from a shop in the range [a, b] with the lowest price, but if and only if that price is at most c (if the lowest price in the range is greater than c, the customer will not buy anything):
{customers}

Please assign an item price for each shop to **maximize the total money earned** from all customers. Output P[1], P[2], ..., P[{N}] in one line, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MYJ_Environment instance.
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
        assert N >= 1, "N should be greater than or equal to 1"

        M = self.parameter["M"] = random.randint(1, N * (N + 1) // 2)

        customers = self.parameter["customers"] = []
        for _ in range(M) :
            a, b = random.randint(1, N), random.randint(1, N)
            customers.append((min(a, b), max(a, b), random.randint(1, N * (N + 1) // 2)))
        

        A = [0] * (M + 1)
        B = [0] * (M + 1)
        C = [0] * (M + 1)
        D = []

        for i in range(1, M + 1):
            a, b, c = customers[i - 1]
            A[i] = a
            B[i] = b
            C[i] = c
            D.append(c)

        # Sort costs and compress them to 1..M
        D_sorted = sorted(D)
        for i in range(1, M + 1):
            C[i] = bisect_left(D_sorted, C[i]) + 1

        # Allocate DP, traceback, bucket and answer arrays
        # f[l][r][i]: maximum total value in segment [l..r] using cost-levels >= i
        f = [
            [
                [0] * (M + 2)
                for _ in range(N + 2)
            ]
            for __ in range(N + 2)
        ]
        # tr[l][r][i]: (cost_index, position) choice for segment [l..r] at level i
        tr = [
            [
                [(0, 0)] * (M + 2)
                for _ in range(N + 2)
            ]
            for __ in range(N + 2)
        ]
        # buc[l][r]: number of customers whose interval [a_j..b_j] is contained in [l..r]
        #               among those with cost-index >= current i
        buc = [
            [0] * (N + 2)
            for _ in range(N + 2)
        ]
        # Final assigned prices
        ans = [0] * (N + 2)

        # Recursive reconstruction of the chosen positions/prices
        def dfs(l, r, i):
            if l > r:
                return
            cost_i, pos = tr[l][r][i]
            ans[pos] = D_sorted[cost_i - 1]
            dfs(l, pos - 1, cost_i)
            dfs(pos + 1, r, cost_i)

        # Main DP: process cost-levels from high to low
        for i in range(M, 0, -1):
            # Add all intervals whose compressed cost == i into the bucket counts
            for j in range(1, M + 1):
                if C[j] == i:
                    for l in range(1, A[j] + 1):
                        for r in range(B[j], N + 1):
                            buc[l][r] += 1

            # Solve subproblems for all segments [l..r]
            for length in range(1, N + 1):
                for l in range(1, N - length + 2):
                    r = l + length - 1
                    # Option 1: skip using cost-level i
                    f[l][r][i] = f[l][r][i + 1]
                    tr[l][r][i] = tr[l][r][i + 1]

                    # Option 2: pick a position p in [l..r] with price = D_sorted[i-1]
                    for p in range(l, r + 1):
                        coef = buc[l][r]
                        coef -= buc[l][p - 1] if p - 1 >= 1 else 0
                        coef -= buc[p + 1][r] if p + 1 <= N else 0
                        v = f[l][p - 1][i] + f[p + 1][r][i] + coef * D_sorted[i - 1]
                        if v > f[l][r][i]:
                            f[l][r][i] = v
                            tr[l][r][i] = (i, p)

                    # If we never picked anything at this level, default to placing at l
                    if tr[l][r][i][0] == 0:
                        tr[l][r][i] = (i, l)

        # Output the maximum total and one valid price assignment
        self.parameter["gold_answer"] = f[1][N][1]
        dfs(1, N, 1)
        self.parameter["reference_answer"] = " ".join(str(ans[i]) for i in range(1, N + 1))
    

    def _prompt_generate(self) -> str :
        customers = self.parameter["customers"]
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = len(customers),
            customers = "\n".join("({}, {}, {})".format(a, b, c) for a, b, c in customers),
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

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            P = [None] + processed_result

            answer, gold = 0, self.parameter["gold_answer"]
            for a, b, c in self.parameter["customers"]:
                min_price = min(P[a : b + 1])
                if min_price <= c:
                    answer += min_price
            assert answer <= gold, "The answer should not exceed the gold answer"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]