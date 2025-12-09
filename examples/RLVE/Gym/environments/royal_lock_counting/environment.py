import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class RoyalLockCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1896
    prompt_template = \
r"""On a {N} × {N} chessboard, you are to place {K} kings such that **no two kings attack each other**. How many different valid placement configurations are there? (The internal order of the kings does NOT matter.)

A king can attack up to 8 surrounding squares: the squares directly above, below, left, right, and all 4 diagonals (top-left, top-right, bottom-left, bottom-right).

**Output Format:**
Your final answer should be a single integer — the total number of valid placements."""

    def __init__(self,
                    wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                    **kwargs) :
            """
            Initialize the RoyalLockCounting_Environment instance.
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
    
        K = self.parameter["K"] = random.randint(1, max(1, N * N // 4))
    

        num_states = 1 << N

        valid_states = []
        line_valid = [False] * num_states
        king_count  = [0]     * num_states
        for s in range(num_states) :
            if s & (s << 1) :
                continue
            line_valid[s] = True
            valid_states.append(s)
            king_count[s] = s.bit_count()

        compat = {s : [] for s in valid_states}
        for s in valid_states :
            for t in valid_states :
                if s & t:
                    continue
                if (s << 1) & t :
                    continue
                if (s >> 1) & t :
                    continue
                compat[s].append(t)

        F_prev = [[0] * num_states for _ in range(K + 1)]
        F_cur  = [[0] * num_states for _ in range(K + 1)]

        F_prev[0][0] = 1

        for _row in range(1, N + 1) :
            for k in range(K + 1) :
                for s in valid_states:
                    F_cur[k][s] = 0

            for s in valid_states :
                c = king_count[s]
                for k in range(c, K + 1) :
                    prev_k = k - c
                    tot = 0
                    for t in compat[s] :
                        tot += F_prev[prev_k][t]
                    F_cur[k][s] = tot

            F_prev, F_cur = F_cur, F_prev

        self.parameter["reference_answer"] = sum(F_prev[K][s] for s in valid_states)

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
            
            if self.parameter["reference_answer"] == 0 :
                return self.rewards["rewarding_weight"] * (processed_result == 0)

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]