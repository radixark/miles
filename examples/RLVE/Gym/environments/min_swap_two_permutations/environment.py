import random
from typing import Optional, List
from collections import defaultdict
from Gym.environment import VerifiableEnvironment


class MinSwapTwoPermutations_Environment(VerifiableEnvironment) : 
    prompt_template = \
r"""You are given two arrays A and B of length {N}. Initially:
- A = {A}
- B = {B}

Your task is to find the **minimum number of indices** i₁, i₂, ..., iₖ such that, after swapping A[i₁] with B[i₁], A[i₂] with B[i₂], ..., A[iₖ] with B[iₖ], both A and B contain **no duplicate elements**. Please output a single line containing the indices i₁, ..., iₖ, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MinSwapTwoPermutations_Environment instance.
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

        A, B = self.parameter["A"], self.parameter["B"] = list(range(1, N + 1)), list(range(1, N + 1))
        while True :
            random.shuffle(A)
            random.shuffle(B)
            swapped_indices = random.sample(range(N), random.randint(1, N - 1))
            for index in swapped_indices:
                A[index], B[index] = B[index], A[index]
            if not (len(set(A)) == N and len(set(B)) == N) :
                break


        # Map each height to the list of positions (where A[i] != B[i])
        p = defaultdict(list)
        for i in range(N):
            if A[i] != B[i]:
                p[A[i]].append(i)
                p[B[i]].append(i)

        # Build graph on positions 0..N-1, with edge weights 0 or 1
        graph = [[] for _ in range(N)]
        for val, occ in p.items():
            if len(occ) == 2:
                u, v = occ
                # weight = 1 if swapping at one end preserves the "same-row" pairing, else 0
                w = 1 if (A[u] == A[v] or B[u] == B[v]) else 0
                graph[u].append((v, w))
                graph[v].append((u, w))

        visited = [False] * N
        ans = 0

        # For each connected component, do a parity-DFS to count flips vs no-flips
        for i in range(N):
            if not visited[i]:
                stack = [(i, 0)]
                cnt = [0, 0]  # cnt[0] = # nodes with parity 0, cnt[1] = # with parity 1
                while stack:
                    u, parity = stack.pop()
                    if visited[u]:
                        continue
                    visited[u] = True
                    cnt[parity] += 1
                    for v, w in graph[u]:
                        if not visited[v]:
                            stack.append((v, parity ^ w))
                # Minimum swaps for this component is min(cnt[0], cnt[1])
                ans += min(cnt)
        
        assert 0 < ans <= len(swapped_indices), "The number of swaps should be between 1 and the number of swapped indices"
        self.parameter["gold_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
            B = " ".join("B[{}]={}".format(i, Bi) for i, Bi in enumerate(self.parameter["B"])),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            A, B = self.parameter["A"].copy(), self.parameter["B"].copy()

            swapping_indices = processed_result
            for swapping_index in swapping_indices :
                if not (0 <= swapping_index < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                A[swapping_index], B[swapping_index] = B[swapping_index], A[swapping_index]
            
            if not (len(set(A)) == self.parameter["N"] and len(set(B)) == self.parameter["N"]) :
                return self.rewards["unsuccessful_solution"]

            answer, gold = len(swapping_indices), self.parameter["gold_answer"]
            assert 0 < gold <= answer, "gold should be less than or equal to answer"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]