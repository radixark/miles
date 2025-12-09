import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class InversionPair_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1966
    prompt_template = \
r"""You are given two arrays A and B, each containing {N} **distinct** integers:
{A}
{B}

You may perform the following operation any number of times: Swap two **adjacent elements** (i.e., elements at indices i and i+1) in either A or B.
Your goal is to **minimize** the sum: (A[0] - B[0])² + (A[1] - B[1])² + ... + (A[{N_minus_1}] - B[{N_minus_1}])²
Among all ways to achieve the minimum possible sum, please output the **minimum number of adjacent swaps** needed."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the InversionPair_Environment instance.
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
        assert N >= 3, "N should be greater than or equal to 3"

        A = self.parameter["A"] = random.sample(range(2 * N), N)
        B = self.parameter["B"] = random.sample(range(2 * N), N)


        # get the permutation that maps sorted order of A to sorted order of B
        a_idx = list(range(N))
        b_idx = list(range(N))
        a_idx.sort(key=lambda i: A[i])
        b_idx.sort(key=lambda i: B[i])

        # l[i] = the rank of A[i] in B's sorted order
        l = [0] * N
        for rank in range(N):
            l[a_idx[rank]] = b_idx[rank]

        # Fenwick (BIT) for counting how many already seen have smaller rank
        BIT = [0] * (N + 1)
        def add(pos, val):
            while pos <= N:
                BIT[pos] += val
                pos += pos & -pos

        def query(pos):
            s = 0
            while pos > 0:
                s += BIT[pos]
                pos -= pos & -pos
            return s

        # count inversions in l[] by scanning from right to left
        ans = 0
        for i in range(N - 1, -1, -1):
            # our ranks in l[i] are 0..N-1, so use pos = l[i]+1 in 1-indexed BIT
            pos = l[i] + 1
            # count how many already-added positions < pos
            ans += query(pos - 1)
            add(pos, 1)

        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
            B = " ".join("B[{}]={}".format(i, Bi) for i, Bi in enumerate(self.parameter["B"])),
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
                if processed_result == 0 :
                    return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == 0)
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]