import random
from typing import Optional, List, Tuple
from Gym.environment import VerifiableEnvironment


class FaceRightWay_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2882
    prompt_template = \
r"""There is a 0/1 array A of length {N}, and initially it is: {A}

Please do the following:
- First, pick a positive integer K, which must remain fixed throughout the process.
- Then, perform M operations. In each operation, you choose an index l (1 ≤ l ≤ {N} - K + 1) and flip all values A[i] with l ≤ i < l + K (i.e., a contiguous subarray of length K).
- Finally, all elements of A must become 0.

Your goal is:
1. Minimize M (the total number of operations).
2. Among all strategies with minimal M, minimize K.

**Output Format:** Output M lines, each containing two integers l and l + K - 1 (separated by a space), representing the closed interval [l, l + K - 1] flipped in that operation. All intervals must have the same length K."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2,
                 rewarding_strategy_M : str = "(gold/answer)^beta", rewarding_weight_M : float = +0.5, rewarding_beta_M : float = 5.0,
                 rewarding_strategy_K : str = "(gold/answer)^beta", rewarding_weight_K : float = +0.5, rewarding_beta_K : float = 5.0,
                 **kwargs):
        """
        Initialize the FaceRightWay_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "unsuccessful_solution": unsuccessful_solution,
            "rewarding_strategy_M": rewarding_strategy_M,
            "rewarding_weight_M": rewarding_weight_M,
            "rewarding_beta_M": rewarding_beta_M,
            "rewarding_strategy_K": rewarding_strategy_K,
            "rewarding_weight_K": rewarding_weight_K,
            "rewarding_beta_K": rewarding_beta_K,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 4, "N should be greater than or equal to 4"

        A = self.parameter["A"] = [0] * N
        K = random.randint(2, N)
        
        left_endpoints = list(range(0, N - K + 1))
        left_endpoints = random.sample(left_endpoints, k = random.randint(1, len(left_endpoints)))
        for l in left_endpoints :
            for i in range(l, l + K) :
                A[i] ^= 1
        
        assert any(A), "A should not be all zeros initially"
        

        ansK = 1
        ansM = sum(A)
        self.parameter["reference_answer"] = "\n".join("{} {}".format(i, i) for i, Ai in enumerate(A, start = 1) if Ai)

        A = [None] + A # 1-indexed

        # Try every K and compute the minimal number of flips M for that K in O(N)
        for K in range(1, N + 1):
            flip = [0] * (N + 1)  # flip[i] == 1 if we start a flip at position i
            curr = 0  # parity of active flips affecting current position
            m = 0
            possible = True

            currect_answer = ""

            for i in range(1, N + 1):
                # Remove the effect of a flip that ends before i
                if i - K >= 1:
                    curr ^= flip[i - K]

                # After applying current parity, do we still see a 'B' at i?
                need_flip = A[i] ^ (curr == 1)
                if need_flip:
                    # Can't start a K-flip if it would exceed N
                    if i + K - 1 > N:
                        possible = False
                        break
                    currect_answer += "{} {}\n".format(i, i + K - 1)
                    flip[i] = 1
                    curr ^= 1
                    m += 1

            if possible and m < ansM:
                ansM = m
                ansK = K
                self.parameter["reference_answer"] = currect_answer.strip()

        self.parameter["gold_answer"] = {"K" : ansK, "M" : ansM}
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = "; ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"], start = 1)),
        )


    def _process(self, answer : Optional[str]) -> Optional[List[Tuple[int, int]]] :
        if answer is not None :
            answer = answer.strip()
            try :
                operations = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        l, r = map(int, line.split())
                        operations.append((l, r))
                return operations
            except :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            A = self.parameter["A"].copy()

            K = None
            for l, r in processed_result :
                if not (1 <= l <= r <= self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                if K is None :
                    K = r - l + 1
                if K != r - l + 1 :
                    return self.rewards["invalid_solution"]
                for i in range(l, r + 1) :
                    A[i - 1] ^= 1
            
            if any(A) :
                return self.rewards["unsuccessful_solution"]

            reward = 0.0
            
            answer_M, gold_M = len(processed_result), self.parameter["gold_answer"]["M"]
            assert 0 < gold_M <= answer_M, "Gold M should be less than or equal to answer M"
            if self.rewards["rewarding_strategy_M"] == "(gold/answer)^beta":
                reward += self.rewards["rewarding_weight_M"] * ((gold_M / answer_M) ** self.rewards["rewarding_beta_M"])
            elif self.rewards["rewarding_strategy_M"] == "gold=answer":
                reward += self.rewards["rewarding_weight_M"] * (gold_M == answer_M)
            else :
                raise NotImplementedError(f"Unknown rewarding strategy: {self.rewards['rewarding_strategy_M']}")

            if gold_M == answer_M :
                answer_K, gold_K = K, self.parameter["gold_answer"]["K"]
                assert 0 < gold_K <= answer_K, "Gold K should be less than or equal to answer K"
                if self.rewards["rewarding_strategy_K"] == "(gold/answer)^beta":
                    reward += self.rewards["rewarding_weight_K"] * ((gold_K / answer_K) ** self.rewards["rewarding_beta_K"])
                elif self.rewards["rewarding_strategy_K"] == "gold=answer":
                    reward += self.rewards["rewarding_weight_K"] * (gold_K == answer_K)
                else :
                    raise NotImplementedError(f"Unknown rewarding strategy: {self.rewards['rewarding_strategy_K']}")
            
            return reward
        else :
            return self.rewards["wrong_format"]