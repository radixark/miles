import math
import random
from typing import Optional, List
from bisect import bisect_left, insort
from Gym.environment import VerifiableEnvironment


class BoxScheduling_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3207
    prompt_template = \
r"""You are given a sequence C: {C}

Now, please determine two non-negative integer sequences X[1], ..., X[{N_minus_1}] and Y[1], ..., Y[{N_minus_1}] such that:
- For 1 ≤ i ≤ {N_minus_1}, define: Pos[i] = (C[i] + {D} × X[i] + Y[i]) mod {N}
- The values Pos[1], ..., Pos[{N_minus_1}] must be all distinct.
- No Pos[i] can be equal to {S}.
- Among all valid solutions:
  + First, minimize the lexicographical order of sequence Y.
  + If multiple solutions have the same Y, then choose the one with the smallest lexicographical order of X.

**Output Format:** A single line containing Pos[1], ..., Pos[{N_minus_1}], separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the BoxScheduling_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is requid in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 3, "MAX_N should be greater than or equal to 3"

        N = self.parameter["N"] = random.randint(3, MAX_N)
        C = self.parameter["C"] = [random.randint(0, N - 1) for _ in range(N - 1)]
        for iter in range(int(N ** 0.5)) :
            D = self.parameter["D"] = random.randint(1, N - 1)
            if math.gcd(D, N) > 1 :
                break
        S = self.parameter["S"] = random.randint(0, N - 1)


        c = [0] + C

        # 2) DSU for “next free” in a D‐cycle
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # 3) prepare the multiset st of (residue_mod_G, count)
        G = math.gcd(D, N)
        con = N // G
        tar = S % G

        # st will be a sorted list of (residue, remaining_slots)
        st = []
        # we'll fill p[] as we go
        p = [0] * N

        # initialize
        for r in range(G):
            if r != tar:
                # all con slots available
                insort(st, (r, con))
            else:
                # reserve one for the empty slot at i=0
                p[0] = S
                # mark S as used by linking it to (S+D)%N
                parent[S] = find((S + D) % N)
                # if there are more in this class, keep (con-1)
                if con > 1:
                    insort(st, (r, con - 1))

        # 4) assign positions for boxes 1..N-1
        for i in range(1, N):
            key = c[i] % G

            # find the first entry in st with residue >= key
            idx = bisect_left(st, (key, -1))
            if idx == len(st):
                # wrap around to the smallest residue
                idx = 0

            r, cnt = st.pop(idx)
            # if more remain in this residue‐class, put it back
            if cnt > 1:
                insort(st, (r, cnt - 1))

            # compute the base position before DSU‐skipping
            if r >= key:
                j = (c[i] + (r - key)) % N
            else:
                # jump up one multiple of G
                j = (((c[i] // G) + 1) * G + r) % N

            # find the actual next free slot in its D‐cycle
            pj = find(j)
            p[i] = pj
            # mark pj used
            parent[pj] = find((pj + D) % N)
        
        self.parameter["gold_answer"] = p[1 :]
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["gold_answer"]))


    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            C = " ".join("C[{}]={}".format(i + 1, Ci) for i, Ci in enumerate(self.parameter["C"])),
            D = self.parameter["D"],
            S = self.parameter["S"],
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

            Pos = processed_result
            if len(Pos) != self.parameter["N"] - 1 :
                return self.rewards["invalid_solution"]
            if set(Pos) != set(range(self.parameter["N"])) - {self.parameter["S"]} :
                return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], Pos)) / (self.parameter["N"] - 1)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == Pos)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]