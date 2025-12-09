import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MinSumPreXor_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4060
    prompt_template = \
r"""You are given an array P of length {N}: {P}
Replace every entry P[i] that equals -1 (for 1 ≤ i ≤ {N}) with a **non-negative integer** (all other entries are fixed non-negative integers), so as to **minimize** the sum: B[1] + B[2] + ... + B[{N}], where B[1] = P[1] and for i ≥ 2, B[i] = B[i−1] XOR P[i] (XOR is the bitwise exclusive OR). Output the updated array P as {N} space-separated non-negative integers in one line."""


    def __init__(self,
                 element_range : int = 2,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinSumPreXor instance.
        """
        super().__init__(**kwargs)

        self.element_range = element_range
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
        assert N >= 3, "N should be greater than or equal to 3"

        P = self.parameter["P"] = [random.randint(0, self.element_range * N) for _ in range(N)]
        for removed_indices in random.sample(range(N), random.randint(1, N - 1)) :
            P[removed_indices] = -1
        

        A = []
        for i, ai in enumerate(P, start = 1) :
            if ai != -1 :
                A.append((i, ai))
        A.sort()
        M = len(A)

        # Compute bit width from input instead of using a magic number.
        if M > 0:
            max_val = max(x for _, x in A)
            BIT = max(1, max_val.bit_length())
        else:
            BIT = 1

        F = []     # per-block counts of set bits for each bit position
        LEN = []   # length of each block (number of known elements inside)
        tot = 0
        now = 0

        for idx in range(M):
            if idx == 0 or A[idx][0] != A[idx - 1][0] + 1:
                F.append([0] * BIT)
                LEN.append(0)
                tot += 1
                now = 0
            now ^= A[idx][1]
            for j in range(BIT):
                F[tot - 1][j] += (now >> j) & 1
            LEN[tot - 1] += 1

        ans = 0
        for i in range(tot):
            if A[i][0] == 1:
                for j in range(BIT):
                    ans += (F[i][j] << j)
            else:
                for j in range(BIT):
                    ans += (min(F[i][j], LEN[i] - F[i][j] + 1) << j)

        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            P = " ".join("P[{}]={}".format(i, Pi) for i, Pi in enumerate(self.parameter["P"], start = 1)),
        )


    def _process(self, answer : Optional[str]) -> Optional[List[int]] :
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
            if not all((previous_Pi >= 0 and now_Pi == previous_Pi) or (previous_Pi == -1 and now_Pi >= 0) for previous_Pi, now_Pi in zip(self.parameter["P"], processed_result)) :
                return self.rewards["invalid_solution"]
            
            answer, gold = 0, self.parameter["gold_answer"]
            Bi = 0
            for Pi in processed_result :
                Bi ^= Pi
                answer += Bi
            assert 0 <= gold <= answer, "gold_answer should be non-negative and not greater than answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert gold == 0, "If answer is 0, gold should also be 0"
                    return self.rewards["rewarding_weight"] * 1.0
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]