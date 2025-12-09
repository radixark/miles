import heapq
import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class GoldWashing_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3303
    prompt_template = \
r"""Define f(x) as the product of the digits of x. For example, f(123) = 1 × 2 × 3 = 6.

Let g(a, b) be the number of pairs (x, y) such that:
1. x, y ∈ [1, {N}]
2. f(x) = a and f(y) = b

Compute g(a, b) for all 1 ≤ a, b ≤ {N}, then sort all g(a, b) values in non-increasing order. Output the sum of the largest {K} values (i.e., the first {K} values in the sorted list)."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the GoldWashing_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 2, "MAX_N should be greater than or equal to 1"

        N = self.parameter["N"] = random.randint(2, MAX_N)
        K = self.parameter["K"] = random.randint(1, N)


        S = str(N)
        n = len(S)

        # 1) Generate all products 2^a * 3^b * 5^c * 7^d <= N
        primes = [2, 3, 5, 7]
        products = []
        def gen(idx, cur):
            if idx == 4:
                products.append(cur)
                return
            p = primes[idx]
            x = cur
            while x <= N:
                gen(idx + 1, x)
                x *= p
        gen(0, 1)

        prod_list = sorted(products)
        M_prime = len(prod_list)
        index_of = {v: i for i, v in enumerate(prod_list)}

        # 2) Precompute counts for all lengths < n (numbers without zeros)
        #    fLen[L][j] = number of L-digit numbers (all digits 1..9) whose digit‐product = prod_list[j]
        fLen = [None] * (n + 1)
        # length = 1
        f1 = [0] * M_prime
        for d in range(1, 10):
            if d > N:
                break
            j = index_of.get(d)
            if j is not None:
                f1[j] += 1
        fLen[1] = f1

        for L in range(2, n):
            prev = fLen[L - 1]
            curr = [0] * M_prime
            for j_idx, cnt in enumerate(prev):
                if cnt == 0:
                    continue
                base = prod_list[j_idx]
                for d in range(1, 10):
                    newp = base * d
                    if newp > N:
                        break
                    newj = index_of[newp]
                    curr[newj] += cnt
            fLen[L] = curr

        # 3) Digit‐DP for length = n, counting numbers in [1..N] with no zeros
        digits = list(map(int, S))
        dp_tight = [0] * M_prime   # prefix == N so far
        dp_loose = [0] * M_prime   # prefix  < N so far
        dp_tight[index_of[1]] = 1  # product = 1 at start

        for pos in range(n):
            new_tight = [0] * M_prime
            new_loose = [0] * M_prime
            ub = digits[pos]

            # transitions from loose (already < N)
            for j_idx, cnt in enumerate(dp_loose):
                if cnt == 0:
                    continue
                base = prod_list[j_idx]
                for d in range(1, 10):
                    newp = base * d
                    if newp > N:
                        break
                    newj = index_of[newp]
                    new_loose[newj] += cnt

            # transitions from tight (== N so far)
            if ub > 0:
                for j_idx, cnt in enumerate(dp_tight):
                    if cnt == 0:
                        continue
                    base = prod_list[j_idx]
                    # choose d < ub -> becomes loose
                    for d in range(1, ub):
                        newp = base * d
                        if newp > N:
                            break
                        newj = index_of[newp]
                        new_loose[newj] += cnt
                    # choose d == ub -> stays tight
                    newp_eq = base * ub
                    if newp_eq <= N:
                        newj_eq = index_of[newp_eq]
                        new_tight[newj_eq] += cnt

            dp_tight, dp_loose = new_tight, new_loose

        # fBound[j] = count of n-digit numbers <= N, no zeros, product = prod_list[j]
        fBound = [dp_tight[i] + dp_loose[i] for i in range(M_prime)]

        # 4) Total counts A[j] = sum over lengths 1..n-1 plus fBound for length n
        A = fBound[:]  # copy
        for L in range(1, n):
            row = fLen[L]
            for j_idx, cnt in enumerate(row):
                if cnt:
                    A[j_idx] += cnt

        # 5) We have sums A[j]; sort them ascending
        sums = sorted(A)

        # 6) Take the top K products from the multiset { sums[i]*sums[j] }
        #    using a max-heap over pairs (i, j) with i, j in [0..M'-1].
        if K > M_prime * M_prime:
            K = M_prime * M_prime

        heap = []
        last = M_prime - 1
        for i in range(M_prime):
            # push initial pair (i, last)
            heap.append((-sums[i] * sums[last], i, last))
        heapq.heapify(heap)

        ans = 0
        for _ in range(K):
            negval, i, j = heapq.heappop(heap)
            val = -negval
            ans += val
            if j > 0:
                new_pair = sums[i] * sums[j - 1]
                heapq.heappush(heap, (-new_pair, i, j - 1))

        assert ans > 0
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], K = self.parameter["K"])
    

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