import random
from collections import deque
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MaximumLexicographicalOrderSubsequence_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3487
    prompt_template = \
r"""Given an array A of length {N}: {A}

Please find a (not necessarily contiguous) subsequence of length {K} (i.e., select {K} elements with increasing indices: 0 <= i1 < ... < i{K} < {N}) such that the resulting subsequence A[i1], ..., A[i{K}] is **lexicographically maximal**. Output a single line containing the selected subsequence A[i1], ..., A[i{K}], separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 5.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the MaximumLexicographicalOrderSubsequence_Environment instance.
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
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        K = self.parameter["K"] = random.randint(2, N - 1)
        A = self.parameter["A"] = [random.randint(1, N) for _ in range(N)]


        self.parameter["gold_answer"] = []
        q = deque()
        # Process each element, maintaining a monotonic queue of at most K candidates
        for i in range(N):
            # Remove smaller elements from the back
            while q and q[-1] < A[i]:
                q.pop()
            # Append current element if we still have fewer than K candidates
            if len(q) < K:
                q.append(A[i])
            # Once we've seen the first N-K+1 elements, start outputting
            if i >= N - K:
                # The front of the deque is the next lexicographically maximal element
                self.parameter["gold_answer"].append(q[0])
                # Remove it before moving on
                q.popleft()
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["gold_answer"]))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K = self.parameter["K"],
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
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

            if len(processed_result) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            
            i = 0
            for a in processed_result :
                found = False
                while i < self.parameter["N"] :
                    if self.parameter["A"][i] == a :
                        found = True
                    i += 1
                    if found :
                        break
                if not found :
                    return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / self.parameter["K"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]