import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class TetrisAttack_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3460
    prompt_template = \
r"""There is an array A (initially it is of length 2 × {N}, containing each integer from 0 to {N_minus_1} exactly twice). Initially, the array A is: {A}

The array follows this rule:
- If there are two adjacent equal elements A[i] == A[i + 1], they are both removed from the array.
- After each removal, the array is compacted (i.e., elements are re-indexed from 0 to the new length), and the process continues as long as such adjacent pairs exist.

Once the array becomes stable (i.e., no adjacent equal pairs remain), you may perform a **swap** between any two adjacent elements A[i] and A[i + 1] (0 ≤ i < current array length - 1). After a swap, the same removal process restarts and continues until stable again. Please **remove all elements from the array**, using the **minimum number of swaps**. Output a single line containing the indices of the swaps (space-separated), where each index `i` indicates a swap between A[i] and A[i + 1]."""

    def __init__(self,
                 cost_range : int = 10,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the TetrisAttack_Environment instance.
        """
        super().__init__(**kwargs)

        self.cost_range = cost_range
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
        assert N >= 2, "N should be greater than or equal to 2"

        A = self.parameter["A"] = list(range(N)) + list(range(N))
        while True :
            random.shuffle(A)
            if all(a != b for a, b in zip(A, A[1 :])) :
                break
        

        vis = [False] * N
        st = []
        Ans = []
        for x in A:
            if vis[x]:
                tax = []
                while st[-1] != x:
                    Ans.append(len(st) - 1)
                    tax.append(st.pop())
                # remove the matching element
                st.pop()
                # restore the other elements
                while tax:
                    st.append(tax.pop())
            else:
                st.append(x)
                vis[x] = True
        assert Ans, "There should be at least one swap to remove all elements from the array"
        self.parameter["gold_answer"] = len(Ans)
        self.parameter["reference_answer"] = " ".join(map(str, Ans))
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
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

            A = self.parameter["A"].copy()
            
            def removal() :
                nonlocal A
                removed = False
                i = 0
                while i < len(A) - 1 :
                    if A[i] == A[i + 1] :
                        A.pop(i)
                        A.pop(i)
                        i = max(0, i - 1)
                        removed = True
                    else :
                        i += 1
                return removed
            assert not removal(), "The input should not remove any elements from the array"
            for i in processed_result :
                if not (0 <= i < len(A) - 1) :
                    return self.rewards["invalid_solution"]
                A[i], A[i + 1] = A[i + 1], A[i]
                removal()
                assert not removal(), "The input should not remove any elements from the array after a swap"
            
            if A :
                return self.rewards["unsuccessful_solution"]

            gold, answer = self.parameter["gold_answer"], len(processed_result)
            assert 0 < gold <= answer, "The number of swaps in the answer should be greater than or equal to the gold answer"
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]