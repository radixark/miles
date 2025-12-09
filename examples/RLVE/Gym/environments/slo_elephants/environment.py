import random
from typing import Optional, Tuple
from Gym.environment import VerifiableEnvironment


class SLOElephants_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3482
    prompt_template = \
r"""There are {N} items labeled from 0 to {N_minus_1}. Each item labeled `i` has an associated cost C[i]. The array C is: {C}
Initially, the items are arranged in the order A (this means the item at position 0 has label A[0], at position 1 has label A[1], etc): {A}
You are required to rearrange the items into the target order B: {B}

You may perform any number of swaps. Swapping the items labeled `i` and `j` incurs a cost of C[i] + C[j]. Please minimize the total cost of all swaps.
Output multiple lines. Each line should contain two integers `i` and `j`, indicating that you swap the items labeled `i` and `j`. The swaps should be listed in the order they are applied."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the SLOElephants_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "unsuccessful_solution" : unsuccessful_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        A, B = self.parameter["A"], self.parameter["B"] = list(range(N)), list(range(N))
        while True :
            random.shuffle(A)
            random.shuffle(B)
            if A != B :
                break
        C = self.parameter["C"] = [random.randint(1, N) for _ in range(N)]


        # ---------- build permutation on elephant IDs ----------
        # dest_pos[e] = where elephant e must finally stand (index in B)
        dest_pos = [0] * N
        for idx, e in enumerate(B):
            dest_pos[e] = idx

        # next_id[e] = elephant that currently occupies e's final place
        next_id = [A[dest_pos[e]] for e in range(N)]

        # ---------- cycle decomposition & cost ----------
        visited   = [False] * N
        overall_min = min(C)            # global lightest elephant
        answer = 0

        for e in range(N):
            if visited[e]:
                continue

            # traverse the current cycle of elephants
            cycle_sum = 0
            cycle_min = 10**9
            length    = 0
            x = e
            while not visited[x]:
                visited[x] = True
                m = C[x]
                cycle_sum += m
                cycle_min = min(cycle_min, m)
                length   += 1
                x = next_id[x]

            if length <= 1:                 # already in place → no swaps
                continue

            # two ways to reorder a cycle of length L (standard POI trick)
            cost_within = cycle_sum + cycle_min * (length - 2)
            cost_global = cycle_sum + cycle_min + overall_min * (length + 1)
            answer += min(cost_within, cost_global)

        assert answer > 0, "The answer should be greater than 0"
        self.parameter["gold_answer"] = answer
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            C = " ".join("C[{}]={}".format(i, Ci) for i, Ci in enumerate(self.parameter["C"])),
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
            B = " ".join("B[{}]={}".format(i, Bi) for i, Bi in enumerate(self.parameter["B"])),
        )


    def _process(self, answer : Optional[str]) -> Optional[Tuple[int, int]] :
        if answer is not None :
            answer = answer.strip()
            try :
                swaps = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        i, j = map(int, line.split())
                        swaps.append((i, j))
                return swaps
            except :
                return None
        return None


    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            A = self.parameter["A"].copy()
            pos = [None] * self.parameter["N"]
            for i, Ai in enumerate(A) :
                pos[Ai] = i
            
            answer, gold = 0, self.parameter["gold_answer"]
            for i, j in processed_result :
                if not (0 <= i < self.parameter["N"] and 0 <= j < self.parameter["N"] and i != j) :
                    return self.rewards["invalid_solution"]
                answer += self.parameter["C"][i] + self.parameter["C"][j]
                A[pos[i]], A[pos[j]] = A[pos[j]], A[pos[i]]
                pos[i], pos[j] = pos[j], pos[i]
                assert A[pos[i]] == i and A[pos[j]] == j, "After swap, A[{}] should be {} and A[{}] should be {}".format(pos[i], i, pos[j], j)
            if A != self.parameter["B"] :
                return self.rewards["unsuccessful_solution"]
            
            assert 0 < gold <= answer, "gold should be less than or equal to answer, but got gold={}, answer={}".format(gold, answer)
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]