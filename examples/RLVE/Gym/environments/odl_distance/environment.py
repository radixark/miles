import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class ODLDistance_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3532
    prompt_template = \
r"""Define an operation on an integer as either multiplying it by a prime number, or dividing it by a prime number (only if it is divisible by that prime). Define D(a, b) as the minimum number of such operations needed to transform a into b; for example, D(69, 42) = 3 because 69 → 3 → 6 → 42 (i.e., divide by 23, multiply by 2, multiply by 7).

Given an array A of length {N}: {A}
For each index i (0 <= i < {N}), find the index j (j ≠ i) such that D(A[i], A[j]) is minimized; if multiple such j exist, choose the smallest one.
Output a single line containing {N} integers — the j values for each i in order, separated by spaces."""

    def __init__(self,
                 weight_multiple : int = 4,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the ODLDistance_Environment instance.
        """
        super().__init__(**kwargs)

        self.weight_multiple = weight_multiple
        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        A = self.parameter["A"] = random.sample(range(1, N * self.weight_multiple + 1), N)


        U = max(A)

        # compute Omega(n): number of prime factors of n with multiplicity
        num = [0] * (U + 1)
        primes = []
        for i in range(2, U + 1):
            if num[i] == 0:
                primes.append(i)
                num[i] = 1
            for p in primes:
                x = p * i
                if x > U:
                    break
                num[x] = num[i] + 1
                if i % p == 0:
                    break

        # build linked lists of positions for each value
        t = [-1] * (U + 1)
        next_idx = [-1] * N
        for i, v in enumerate(A):
            next_idx[i] = t[v]
            t[v] = i

        # initialize answers
        INF = U + 1
        ans = [INF] * N
        ansj = [-1] * N

        # for each possible divisor x
        for x in range(1, U + 1):
            # collect all indices i with A[i] divisible by x
            q = []
            for m in range(x, U + 1, x):
                j = t[m]
                while j != -1:
                    q.append(j)
                    j = next_idx[j]
            if not q:
                continue

            # find index b in q with minimal num[A[b]] (tie-break on smaller index)
            b = q[0]
            for i in range(1, len(q)):
                qi = q[i]
                if num[A[qi]] < num[A[b]] or (num[A[qi]] == num[A[b]] and qi < b):
                    # swap b and q[i]
                    q[i], b = b, qi

            # update distances using this common divisor x
            common = num[x] << 1
            for i in range(1, len(q)):
                a_i = q[i]
                d = num[A[a_i]] + num[A[b]] - common

                # update for a_i
                if d < ans[a_i] or (d == ans[a_i] and b < ansj[a_i]):
                    ans[a_i] = d
                    ansj[a_i] = b

                # update for b
                if d < ans[b] or (d == ans[b] and a_i < ansj[b]):
                    ans[b] = d
                    ansj[b] = a_i

        self.parameter["gold_answer"] = ansj
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["gold_answer"]))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
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
            if not all(0 <= j < self.parameter["N"] and j != i for i, j in enumerate(processed_result)) :
                return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]