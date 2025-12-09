import random
from typing import Optional, List
from collections import defaultdict, deque
from Gym.environment import VerifiableEnvironment


class Minimum_MaxAbsSlicer_Environment(VerifiableEnvironment) :  # Source : https://www.luogu.com.cn/problem/P3229
    prompt_template = \
r"""You are given two arrays A and B, each of length {N} (0-indexed). A is a permutation of [1, 2, ..., {N}], and each element of B is either +1 or -1. The values are as follows:
{A_and_B}

You must divide the indices [0, 1, ..., {N_minus_1}] into {M} **consecutive batches**. Let end[1], end[2], ..., end[{M}] (0 ≤ end[1] < end[2] < ... < end[{M}] = {N_minus_1}) represent the last index of each batch. This means:
- Batch 1 contains indices from 0 to end[1]
- Batch 2 contains indices from end[1] + 1 to end[2]
- ...
- Batch {M} contains indices from end[{M_minus_1}] + 1 to end[{M}] = {N_minus_1}

For each batch i, let S[i] be the **sum of B values in that batch**. Your goal is to **minimize the maximum absolute value** among all batches, i.e., minimize max(|S[1]|, |S[2]|, ..., |S[{M}]|).
Among all such optimal partitions, choose the one with the **smallest lexicographical order** of the sequence A[end[1]], A[end[2]], ..., A[end[{M}]].

**Output Format:** Your final answer should be a single line containing A[end[1]], A[end[2]], ..., A[end[{M}]], separated by **spaces**."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5,
                 rewarding_strategy_abs : str = "(gold/answer)^beta", rewarding_weight_abs : float = +0.5, rewarding_beta_abs : float = +5.0,
                 rewarding_strategy_lex : str = "mean([gold=answer])^beta", rewarding_weight_lex : float = +0.5, rewarding_beta_lex : float = +5.0,
                 **kwargs) :
        """
        Initialize the Minimum_MaxAbsSlicer_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy_abs" : rewarding_strategy_abs,
            "rewarding_weight_abs" : rewarding_weight_abs,
            "rewarding_beta_abs" : rewarding_beta_abs,
            "rewarding_strategy_lex" : rewarding_strategy_lex,
            "rewarding_weight_lex" : rewarding_weight_lex,
            "rewarding_beta_lex" : rewarding_beta_lex,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 4, "N must be at least 4"

        M = self.parameter["M"] = random.randint(3, N - 1)

        self.parameter["A"] = list(range(1, N + 1))
        random.shuffle(self.parameter["A"])
        positive_probability = random.random()
        self.parameter["B"] = [+1 if random.random() < positive_probability else -1 for _ in range(N)]


        A = [0] * (N + 2)                # 1-based city ids
        B = [0] * (N + 2)                # +1 (sight) / –1 (no sight)

        for i in range(1, N + 1):
            A[i] = self.parameter["A"][i - 1]
            B[i] = self.parameter["B"][i - 1]

        # ---------- build suffix balance array ----------
        SUF = [0] * (N + 3)              # SUF[i] = balance on [i … N]
        for i in range(N, 0, -1):
            SUF[i] = B[i] + SUF[i + 1]

        # count how many suffixes are perfectly balanced
        tot_zero = sum(1 for i in range(1, N + 1) if SUF[i] == 0)

        OFFSET = N                       # shift to make indices non-negative

        # ---------- minimal possible maximal monthly imbalance d ----------
        if SUF[1] == 0:                  # whole trip already balanced
            d = 1 if tot_zero < M else 0
        else:
            d = (abs(SUF[1]) - 1) // M + 1   # same as ceil(|SUF[1]| / M)
        self.parameter["gold_answer_max_abs"] = d

        # ---------- monotone queues keyed by balance value ----------
        queues = defaultdict(deque)      # balance → deque[(city, pos)]

        def push(pos: int) -> None:
            """Put position `pos` into queue of balance SUF[pos+1]."""
            key = SUF[pos + 1] + OFFSET
            dq = queues[key]
            rec = (A[pos], pos)          # ordered by city id
            while dq and rec[0] < dq[-1][0]:
                dq.pop()
            dq.append(rec)

        def best_from_queue(now_pos: int, key: int, cur_best: tuple) -> tuple:
            """Try improving cur_best using front of queue `key`."""
            dq = queues.get(key)
            if not dq:
                return cur_best
            while dq and dq[0][1] < now_pos:   # outdated endpoint
                dq.popleft()
            if dq and dq[0][0] < cur_best[0]:
                return dq[0]
            return cur_best

        # ---------- CASE 1 : perfectly balanced plan possible (d == 0) ----------
        if d == 0:
            C = [i for i in range(1, N + 1) if SUF[i + 1] == 0]   # candidate cuts
            tot_c = len(C)
            now = 1
            j = 0
            answer = []

            # decide the first M-1 months
            for month in range(1, M):
                # keep at least (M - month) candidates unpushed
                while tot_c - j > M - month:
                    push(C[j])
                    j += 1
                best = (N + 1, -1)                      # (city id, pos)
                best = best_from_queue(now, OFFSET, best)
                answer.append(best[0])
                now = best[1] + 1                       # next month starts here
        else :

            # ---------- CASE 2 : need positive imbalance (d > 0) ----------
            now = 1
            r = 1
            # preload all positions that may finish the first month
            while N - r >= M - 1:
                push(r)
                r += 1

            answer = []
            months_left = M

            while months_left > 1:
                best = (N + 1, -1)
                center = SUF[now] + OFFSET

                low = max(0, center - d)
                high = min(2 * N, center + d)

                for key in range(low, high + 1):
                    # |balance| must be small enough to finish the rest in (months_left-1) months
                    if abs(key - OFFSET) <= (months_left - 1) * d:
                        best = best_from_queue(now, key, best)

                answer.append(best[0])
                now = best[1] + 1
                months_left -= 1

                # make one more position available for the next round
                if r <= N:           # guard, though algorithm ensures r ≤ N
                    push(r)
                r += 1

        answer.append(A[N])                          # last month ends here
        assert len(answer) == M, "The answer should have exactly M elements"
        self.parameter["gold_answer"] = answer
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["gold_answer"]))


    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        M = self.parameter["M"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            M = M,
            M_minus_1 = M - 1,
            A_and_B = "\n".join("A[{}]={} B[{}]={}".format(i, Ai, i, Bi) for i, (Ai, Bi) in enumerate(zip(self.parameter["A"], self.parameter["B"]))),
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

            reward = 0.0


            N = self.parameter["N"]
            if not all(1 <= Ai <= N for Ai in processed_result) :
                return self.rewards["invalid_solution"]
            Ai2i = [None] * (N + 1)
            for i, Ai in enumerate(self.parameter["A"]) :
                Ai2i[Ai] = i
            ends = [Ai2i[Ai] for Ai in processed_result]

            if len(ends) != self.parameter["M"] :
                return self.rewards["invalid_solution"]
            for i in range(len(ends)) :
                if not (0 <= ends[i] < N) :
                    return self.rewards["invalid_solution"]
                if i and not (ends[i - 1] < ends[i]) :
                    return self.rewards["invalid_solution"]
            if ends[-1] != N - 1 :
                return self.rewards["invalid_solution"]
            
            answer = abs(sum(self.parameter["B"][index] for index in range(ends[0] + 1)))
            for i in range(1, len(ends)) :
                answer = max(answer, abs(sum(self.parameter["B"][index] for index in range(ends[i - 1] + 1, ends[i] + 1))))
            gold = self.parameter["gold_answer_max_abs"]
            assert gold <= answer, "answer should be greater than or equal to gold"
            if self.rewards["rewarding_strategy_abs"] == "(gold/answer)^beta" :
                if answer == 0 :
                    assert gold == 0, "If answer is 0, gold should also be 0"
                    reward += self.rewards["rewarding_weight_abs"] * 1.0
                else :
                    reward += self.rewards["rewarding_weight_abs"] * ((gold / answer) ** self.rewards["rewarding_beta_abs"])
            elif self.rewards["rewarding_strategy_abs"] == "gold=answer" :
                reward += self.rewards["rewarding_weight_abs"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy_abs"]))


            if gold == answer :
                if self.rewards["rewarding_strategy_lex"] == "mean([gold=answer])^beta" :
                    for a, b in zip(self.parameter["gold_answer"], processed_result) :
                        if a != b :
                            assert a < b, "gold_answer should be less than or equal to processed_result"
                            break
                    reward += self.rewards["rewarding_weight_lex"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / self.parameter["M"]) ** self.rewards["rewarding_beta_lex"])
                elif self.rewards["rewarding_strategy_lex"] == "gold=answer" :
                    reward += self.rewards["rewarding_weight_lex"] * (self.parameter["gold_answer"] == processed_result)
                else :
                    raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy_lex"]))
            
            return reward
        else :
            return self.rewards["wrong_format"]