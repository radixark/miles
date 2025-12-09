import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MafMafia_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3472
    prompt_template = \
r"""There are {N} participants in a game, labeled from 0 to {N_minus_1}. Each participant `i` has a target participant TO[i]. The array TO is given as: {TO}

You are to determine a permutation P[0], P[1], ..., P[{N_minus_1}] of the {N} participants, representing the order in which they act. The game proceeds in that order as follows:
- When a participant takes their turn, if they are still alive, they attempt to kill their target TO[i].
- If the target has already been killed earlier, nothing happens.
- A participant who has already been killed cannot act.

Please find a permutation that **{minimize_or_maximize}s the number of participants who get killed** by the end of the game. Output a single line containing the permutation P[0], P[1], ..., P[{N_minus_1}], separated by spaces."""


    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5,
                 rewarding_strategy_min : str = "(gold/answer)^beta", rewarding_weight_min : float = +1.0, rewarding_beta_min : float = 5.0,
                 rewarding_strategy_max : str = "(answer/gold)^beta", rewarding_weight_max : float = +1.0, rewarding_beta_max : float = 5.0,
                 **kwargs) :
        """
        Initialize the MafMafia_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy_max" : rewarding_strategy_max,
            "rewarding_weight_max" : rewarding_weight_max,
            "rewarding_beta_max" : rewarding_beta_max,
            "rewarding_strategy_min" : rewarding_strategy_min,
            "rewarding_weight_min" : rewarding_weight_min,
            "rewarding_beta_min" : rewarding_beta_min,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        TO = self.parameter["TO"] = [random.randint(0, N - 1) for _ in range(N)]

        self.parameter["minimize_or_maximize"] = random.choice(["minimize", "maximize"])


        # Compute indegrees
        d = [0] * N
        for t in TO:
            d[t] += 1

        # Prepare queue for trimming leaves
        q = [0] * N
        head = 0
        tail = 0
        minn = 0  # will count nodes trimmed (and pure cycles) for minimum-deaths logic

        # Enqueue all initial leaves (indegree 0)
        for i in range(N):
            if d[i] == 0:
                q[tail] = i
                tail += 1
                minn += 1

        # Arrays to mark who dies in trimming, and which cycle nodes have incoming trees
        die = [False] * N
        lv = [False] * N

        # Trim all trees feeding into cycles
        while head < tail:
            x = q[head]
            head += 1
            tx = TO[x]
            # If the target is already dead, skip
            if die[tx]:
                continue
            # Mark that target as killed
            die[tx] = True
            # Flag the target-of-target as having an incoming tree branch
            y = TO[tx]
            lv[y] = True
            # Decrement indegree, and if it becomes a leaf, enqueue it
            d[y] -= 1
            if d[y] == 0:
                q[tail] = y
                tail += 1

        # 'tail' is now the total number of nodes trimmed (including those from cycles broken by trees)
        maxn = tail

        # Now handle any remaining pure cycles
        for i in range(N):
            if not die[i] and d[i] > 0:
                # Traverse this cycle exactly once
                cnt = 0
                has_branch = False
                x = i
                while not die[x]:
                    cnt += 1
                    if lv[x]:
                        has_branch = True
                    die[x] = True
                    nx = TO[x]
                    # stop once we complete the loop
                    if nx == i:
                        break
                    x = nx

                # In a cycle of length cnt, at most floor(cnt/2) die in the worst case
                maxn += cnt // 2
                # But if it's a pure cycle (no incoming tree), at minimum 1 must die
                if cnt > 1 and not has_branch:
                    minn += 1

        # Compute and print: minimum and maximum possible deaths
        # min_deaths  = N - maxn
        # max_deaths  = N - minn
        if self.parameter["minimize_or_maximize"] == "minimize" :
            answer = N - maxn
        elif self.parameter["minimize_or_maximize"] == "maximize" :
            answer = N - minn
        else :
            assert False, "minimize_or_maximize should be either 'minimize' or 'maximize'"
        assert answer > 0, "Answer should be greater than 0"
        self.parameter["gold_answer"] = answer


    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            TO = " ".join("TO[{}]={}".format(i, To_i) for i, To_i in enumerate(self.parameter["TO"])),
            minimize_or_maximize = self.parameter["minimize_or_maximize"],
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

            P = processed_result
            if len(P) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if set(P) != set(range(self.parameter["N"])) :
                return self.rewards["invalid_solution"]
            
            killed = [False] * self.parameter["N"]
            for i in P :
                if killed[i] :
                    continue
                killed[self.parameter["TO"][i]] = True
            answer, gold = sum(map(int, killed)), self.parameter["gold_answer"]
            
            if self.parameter["minimize_or_maximize"] == "minimize" :
                assert 0 < gold <= answer, "For minimization, answer should be greater than 0 and at least as large as the gold answer"
                if self.rewards["rewarding_strategy_min"] == "(gold/answer)^beta" :
                    return self.rewards["rewarding_weight_min"] * ((gold / answer) ** self.rewards["rewarding_beta_min"])
                elif self.rewards["rewarding_strategy_min"] == "gold=answer" :
                    return self.rewards["rewarding_weight_min"] * (gold == answer)
                else :
                    raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy_min"]))
            elif self.parameter["minimize_or_maximize"] == "maximize" :
                assert 0 < answer <= gold, "For maximization, answer should be greater than 0 and at most as large as the gold answer"
                if self.rewards["rewarding_strategy_max"] == "(answer/gold)^beta" :
                    return self.rewards["rewarding_weight_max"] * ((answer / gold) ** self.rewards["rewarding_beta_max"])
                elif self.rewards["rewarding_strategy_max"] == "gold=answer" :
                    return self.rewards["rewarding_weight_max"] * (gold == answer)
                else :
                    raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy_max"]))
            else :
                assert False, "minimize_or_maximize should be either 'minimize' or 'maximize'"
        else :
            return self.rewards["wrong_format"]