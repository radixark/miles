import random
from typing import Optional
from itertools import permutations
from Gym.environment import VerifiableEnvironment


class BoundedAdjacencyDifference_Permutation_Counting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3867
    prompt_template = r"""What is the number of permutations of 1, 2, ..., {N} such that for every two adjacent elements (i.e., the i-th and (i+1)-th elements for all 1 <= i < N), the absolute difference between them is at most {K}?"""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the BoundedAdjacencyDifference_Permutation_Counting_Environment instance.
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
        assert N >= 4, "N should be greater than or equal to 4"

        self.parameter["K"] = K = random.randint(2, min(4, N - 2))


        # Precompute factorials up to K (K <= 4)
        FACT = [1] * (K + 1)
        for i in range(1, K + 1):
            FACT[i] = FACT[i - 1] * i
        FK = FACT[K]  # K!

        # All permutations of length K in lexicographic order
        PERMS = [list(p) for p in permutations(range(K))]
        TM = (1 << (K + 1)) - 1  # mask with (K+1) ones

        # DP over i (size), ip (permutation index), ic (mask)
        # Use rolling arrays to keep memory tight and sizes appropriate
        prev = [[0] * (TM + 1) for _ in range(FK)]
        for ip in range(FK):
            prev[ip][TM] = 1  # base: i = K

        for i in range(K + 1, N + 1):
            cur = [[0] * (TM + 1) for _ in range(FK)]
            for ip in range(FK):
                tp = PERMS[ip]  # current permutation of size K
                for ic in range(TM + 1):
                    val = prev[ip][ic]
                    if not val:
                        continue
                    # Try to insert the new maximum at each available slot j
                    for j in range(K + 1):
                        if ((ic >> j) & 1) == 0:
                            continue

                        # Insert into permutation representation
                        ttp_ins = tp[:j] + [K] + tp[j:]           # length K+1, values in {0..K}
                        l0 = ttp_ins.index(0)                      # first position of '0'
                        ttp_trim = ttp_ins[:l0] + ttp_ins[l0 + 1:] # remove that '0'
                        ttp = [x - 1 for x in ttp_trim]            # now a perm of {0..K-1}

                        # Update slot mask
                        tc_bits = [ (ic >> l) & 1 for l in range(K + 1) ]
                        ttc2 = tc_bits[:j] + [1] + tc_bits[j:]     # insert a '1' at j
                        # remove index l0+1 and then clear index l0
                        ttc_removed = ttc2[:l0 + 1] + ttc2[l0 + 2:]
                        ttc_removed[l0] = 0
                        icc = 0
                        for l in range(K + 1):
                            if ttc_removed[l]:
                                icc |= (1 << l)

                        # Lehmer code -> permutation index 'ipp'
                        ipp = 0
                        seen = [0] * K
                        for pos in range(K):
                            v = ttp[pos]
                            ch = 0
                            for z in range(v):
                                if seen[z] == 0:
                                    ch += 1
                            seen[v] = 1
                            ipp += ch * FACT[K - 1 - pos]

                        cur[ipp][icc] += val
            prev = cur

        ans = 0
        for ip in range(FK):
            for ic in range(TM + 1):
                ans += prev[ip][ic]
        assert ans > 0, "The answer should be positive"
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