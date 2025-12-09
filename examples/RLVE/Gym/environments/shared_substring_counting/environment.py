import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SharedSubstringCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3181
    prompt_template = \
r"""You are given two strings:
S = {S}  
T = {T}

Please compute the number of tuples (lS, rS, lT, rT) such that:
- 0 ≤ lS < rS ≤ len(S)
- 0 ≤ lT < rT ≤ len(T)
- The substring S[lS : rS] is equal to the substring T[lT : rT] (we are using Python-style slicing here)"""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SharedSubstringCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_LEN" in self.parameter, "MAX_LEN is required in parameter"
        MAX_LEN = self.parameter["MAX_LEN"]
        assert MAX_LEN >= 2, "MAX_LEN should be greater than or equal to 2"

        for key in ("S", "T") :
            a_probability = random.random()
            LEN = random.randint(2, MAX_LEN)
            self.parameter[key] = "".join("a" if random.random() < a_probability else "b" for _ in range(LEN))
        S, T = self.parameter["S"], self.parameter["T"]


        def SA(arr):
            """
            Given an integer array `arr` representing a string (each int is a “character” code),
            build its suffix array and LCP, then return
            sum_{0 <= i < j < n} LCP(suffix_i, suffix_j).
            """
            n = len(arr)
            if n <= 1:
                return 0

            # initial rank range
            m = max(arr) + 1

            sa = [0] * n
            rk = arr[:]        # rk[i] = rank of the suffix starting at i
            tp = [0] * n       # temporary array for sorting
            # initial radix‐sort by single character
            tax = [0] * m
            for x in rk:
                tax[x] += 1
            for i in range(1, m):
                tax[i] += tax[i-1]
            for i in range(n-1, -1, -1):
                c = rk[i]
                tax[c] -= 1
                sa[tax[c]] = i

            # doubling loop
            w = 1
            while True:
                # sort by second key: collect suffixes with i >= n-w first
                p = 0
                for i in range(n-w, n):
                    tp[p] = i; p += 1
                for i in range(n):
                    j = sa[i]
                    if j >= w:
                        tp[p] = j - w
                        p += 1

                # radix‐sort by first key
                tax = [0] * m
                for x in rk:
                    tax[x] += 1
                for i in range(1, m):
                    tax[i] += tax[i-1]
                for i in range(n-1, -1, -1):
                    j = tp[i]
                    c = rk[j]
                    tax[c] -= 1
                    sa[tax[c]] = j

                # re‐rank
                old_rk = rk
                rk = [0] * n
                rk[sa[0]] = 0
                p = 1
                for i in range(1, n):
                    prev, curr = sa[i-1], sa[i]
                    # compare pairs (old_rk[curr], old_rk[curr+w]) vs (old_rk[prev], old_rk[prev+w])
                    if (old_rk[curr] == old_rk[prev] and
                        (old_rk[curr+w] if curr+w < n else -1) ==
                        (old_rk[prev+w] if prev+w < n else -1)):
                        rk[curr] = p-1
                    else:
                        rk[curr] = p
                        p += 1

                if p >= n:
                    break
                m = p
                w <<= 1

            # build LCP array (het) via Kasai’s algorithm
            het = [0] * n
            k = 0
            for i in range(n):
                r = rk[i]
                if r == 0:
                    continue
                j = sa[r-1]
                while i + k < n and j + k < n and arr[i+k] == arr[j+k]:
                    k += 1
                het[r] = k
                if k:
                    k -= 1

            # now sum up all LCPs over i<j by the classic “sum of minima in all subarrays” trick
            stack_h = []
            stack_cnt = []
            running = 0
            total = 0
            for i in range(1, n):
                h = het[i]
                cnt = 1
                # pop until stack_h[-1] < h
                while stack_h and stack_h[-1] >= h:
                    last_h = stack_h.pop()
                    last_cnt = stack_cnt.pop()
                    running -= last_h * last_cnt
                    cnt += last_cnt
                stack_h.append(h)
                stack_cnt.append(cnt)
                running += h * cnt
                total += running

            return total


        def compute():
            # use a separator > 'z'
            SEP = ord('z') + 1

            # build concatenated array
            concat = [ord(c) for c in S] + [SEP] + [ord(c) for c in T]

            # total cross‐sum = SA(S#T) - SA(S) - SA(T)
            ans = SA(concat)
            ans -= SA([ord(c) for c in S])
            ans -= SA([ord(c) for c in T])

            return ans

        self.parameter["reference_answer"] = compute()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            S = self.parameter["S"],
            T = self.parameter["T"],
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
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                if self.parameter["reference_answer"] == 0 :
                    return self.rewards["rewarding_weight"] * (processed_result == 0)
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]