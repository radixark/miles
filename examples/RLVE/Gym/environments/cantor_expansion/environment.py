import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class CantorExpansion_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3477
    prompt_template = \
r"""Given a sequence of integers: {A}

Please count the number of distinct permutations of this sequence that are **lexicographically smaller** than the original sequence. Output a single integer — the number of such permutations modulo {MOD}.
Note: Permutations that only differ by the positions of equal elements are considered the **same**."""

    def __init__(self,
                 max_MOD : int = 100000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the CantorExpansion_Environment instance.
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

        M = random.randint(2, N)
        A = self.parameter["A"] = [random.randint(1, M) for _ in range(N)]
        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


        M = max(A)

        # 1. Factor MOD into its prime factors and compute phi(MOD)
        ph = MOD
        nt = MOD
        p_list = []
        i = 2
        while i * i <= nt:
            if nt % i == 0:
                p_list.append(i)
                ph = ph // i * (i - 1)
                while nt % i == 0:
                    nt //= i
            i += 1
        if nt > 1:
            p_list.append(nt)
            ph = ph // nt * (nt - 1)
        pc = len(p_list)

        # 2. Fenwick tree (BIT) for counting how many of the suffix elements are <= a given value
        T = [0] * (M + 1)
        def bit_add(x):
            while x <= M:
                T[x] += 1
                x += x & -x
        def bit_sum(x):
            s = 0
            while x > 0:
                s += T[x]
                x -= x & -x
            return s

        # 3. Arrays to track multiplicative state modulo MOD
        iv = [0] * (N + 2)    # iv[k] = modular inverse of k (for k co-prime to MOD), filled on the fly
        iv[1] = 1
        tp = [0] * pc         # exponent counts for each prime in p_list
        tc = 1                # current product of all co-prime parts mod MOD
        cnt = [0] * (M + 1)   # how many times each value appears in the suffix

        ans = 0

        # Seed with the last element in the permutation
        bit_add(A[N-1])
        cnt[A[N-1]] += 1

        # Process positions from right to left
        for idx in range(N - 2, -1, -1):
            # w = how many suffix elements are strictly smaller than A[idx]
            w = bit_sum(A[idx] - 1)

            # 1) Multiply in the next factorial factor: (suffix length)!
            k = (N - 1) - idx
            tmp = k
            for j, pj in enumerate(p_list):
                while tmp % pj == 0:
                    tmp //= pj
                    tp[j] += 1
            tc = tc * tmp % MOD

            # 2) Add this element into the BIT and update its count
            bit_add(A[idx])
            iv[k + 1] = pow(k + 1, ph - 1, MOD)  # inverse of k+1, co-prime part only used later
            cnt[A[idx]] += 1

            # 3) Divide out the new multiplicity factorial factor
            tmp = cnt[A[idx]]
            for j, pj in enumerate(p_list):
                while tmp % pj == 0:
                    tmp //= pj
                    tp[j] -= 1
            tc = tc * iv[tmp] % MOD

            # 4) If there are smaller choices w, add w * (remaining permutations) to the rank
            if w > 0:
                # multiply by w
                tmp = w
                for j, pj in enumerate(p_list):
                    while tmp % pj == 0:
                        tmp //= pj
                        tp[j] += 1
                tc = tc * tmp % MOD

                # compute the current value = tc * ∏ p_i^{tp_i} mod MOD
                cur = tc
                for j, pj in enumerate(p_list):
                    if tp[j]:
                        cur = cur * pow(pj, tp[j], MOD) % MOD
                ans = (ans + cur) % MOD

                # divide back by w to restore state
                tmp = w
                for j, pj in enumerate(p_list):
                    while tmp % pj == 0:
                        tmp //= pj
                        tp[j] -= 1
                tc = tc * iv[tmp] % MOD

        self.parameter["reference_answer"] = ans % MOD
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(A = ", ".join(map(str, self.parameter["A"])), MOD = self.parameter["MOD"])
    

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