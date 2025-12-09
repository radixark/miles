import random
from array import array
from typing import Optional
from Gym.environment import VerifiableEnvironment


class TransmissionDelay_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2929
    prompt_template = \
r"""You are given a binary (0/1) array A of length {N} (1-indexed): {A}

You can generate a new array A′ by the following operation:
1) Choose a permutation P of 1, 2, ..., {N} such that for every i (1 ≤ i ≤ {N}), |i − P[i]| ≤ {D}.
2) For every i (1 ≤ i ≤ {N}), set A′[i] = A[P[i]].

Can you tell me the number of **distinct** arrays A′ that can be obtained by such operations?"""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the TransmissionDelay_Environment instance.
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

        while True :
            zero_probability = random.random()
            A = self.parameter["A"] = [0 if random.random() < zero_probability else 1 for _ in range(N)]
            if not (2 <= sum(A) <= N - 2) :
                continue
        
            max_D = 0
            for c in (0, 1) :
                indices = [i for i, x in enumerate(A, start = 1) if x == c]
                max_D = max(max_D, max(indices[0] - 2, N - 1 - indices[-1]))
                if len(indices) > 1 :
                    max_D = max(max_D, max((indices[i] - indices[i - 1] - 2) // 2 for i in range(1, len(indices))))
            if max_D >= 1 :
                break
        D = self.parameter["D"] = random.randint(1, max_D)


        S = "".join(map(str, A))

        # 1-based indexing for convenience (match the C++ logic)
        S = " " + S

        # Collect positions of 0s and 1s (1-based); keep a dummy 0 at index 0
        p0 = [0]
        p1 = [0]
        for i in range(1, N + 1):
            if S[i] == '0':
                p0.append(i)
            else:
                p1.append(i)
        cnt0 = len(p0) - 1
        cnt1 = len(p1) - 1

        # DP tables: F for modulo counts, G for saturated counts (capped at MOD+1)
        # Use array('I') to keep memory reasonable (4 bytes per entry)
        F = [array('I', [0] * (cnt0 + 1)) for _ in range(N + 2)]
        G = [array('I', [0] * (cnt0 + 1)) for _ in range(N + 2)]

        # Base case
        F[N + 1][0] = 1
        G[N + 1][0] = 1

        # Fill DP from i = N down to 1
        for i in range(N, 0, -1):
            # Only valid states where remaining zeros j <= cnt0 and ones k <= cnt1
            # j + k = N - i + 1  =>  j in [max(0, L - cnt1), min(L, cnt0)]
            L = N - i + 1
            j_min = max(0, L - cnt1)
            j_max = min(L, cnt0)
            Fi1 = F[i + 1]  # row i+1
            Gi1 = G[i + 1]
            Fi = F[i]
            Gi = G[i]

            for j in range(j_min, j_max + 1):
                k_ones = L - j
                total_f = 0
                total_g = 0

                # Try placing a '0' at position i
                if j > 0:
                    idx0 = cnt0 - j + 1  # the "next" remaining 0 (from the end)
                    if abs(p0[idx0] - i) <= D:
                        total_f += Fi1[j - 1]
                        total_g = Gi1[j - 1] if total_g == 0 else total_g + Gi1[j - 1]

                # Try placing a '1' at position i
                if k_ones > 0:
                    idx1 = cnt1 - k_ones + 1  # the "next" remaining 1 (from the end)
                    if abs(p1[idx1] - i) <= D:
                        total_f += Fi1[j]
                        total_g = Gi1[j] if total_g == 0 else total_g + Gi1[j]

                Fi[j] = total_f
                Gi[j] = total_g

        self.parameter["reference_answer"] = F[1][cnt0]
        assert self.parameter["reference_answer"] > 0, "Reference answer should be positive" 
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = ";".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"], start = 1)),
            D = self.parameter["D"],
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