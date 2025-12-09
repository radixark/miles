import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class ShortestUnicolorSubstring_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3718
    prompt_template = \
r"""You are given a binary string (i.e., consisting of only 0s and 1s) S of length {N}: {S}

Please construct a binary string T of length {N} such that:
- There are at most {K} positions where S[i] ≠ T[i].
- You try your best to **minimize** the length of the **longest consecutive segment** of the same character in T.

**Output Format:** A single line containing the string T — a binary string of length {N}."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(gold/answer)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the ShortestUnicolorSubstring_Environment instance.
        """
        super().__init__(**kwargs)

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

        one_probability = random.random()
        self.parameter["S"] = S = "".join("1" if random.random() < one_probability else "0" for _ in range(N))
        self.parameter["K"] = K = random.randint(1, N // 2)


        def compute():
            lamp = list(map(int, S))

            # Count mismatches to the two possible alternating patterns
            # Pattern A: positions 1,3,5... = 'N' (1), positions 2,4,6... = 'F' (0)
            # In 0-based index: i%2==0 -> 1, else 0
            s1 = sum(1 for i, v in enumerate(lamp) if v == (1 if i % 2 == 0 else 0))
            # The other pattern requires flipping exactly the opposite set of positions
            s2 = N - s1

            # If we can flip into a perfect alternation, the answer is 1
            if min(s1, s2) <= K:
                return 1

            # Build the lengths of consecutive same-value segments
            segments = []
            curr = lamp[0]
            length = 1
            for v in lamp[1:]:
                if v == curr:
                    length += 1
                else:
                    segments.append(length)
                    curr = v
                    length = 1
            segments.append(length)

            # Given a candidate maximum run-length x, how many flips are needed?
            # For each segment of length L, we need floor(L / (x+1)) flips
            def flips_needed(x):
                total = 0
                for L in segments:
                    total += L // (x + 1)
                return total

            # Binary search the minimal x in [2..N] such that flips_needed(x) <= K
            lo, hi = 2, N
            ans = N
            while lo <= hi:
                mid = (lo + hi) // 2
                if flips_needed(mid) > K:
                    # too many flips needed, increase x
                    lo = mid + 1
                else:
                    # feasible, try smaller
                    ans = mid
                    hi = mid - 1

            return ans
        self.parameter["gold_answer"] = compute()
        assert self.parameter["gold_answer"] >= 1, "The gold answer should be at least 1"
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], S = self.parameter["S"], K = self.parameter["K"])
    

    def _process(self, answer : Optional[str]) -> Optional[str] :
        if answer is not None :
            return answer.strip()
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            T = processed_result
            
            if len(T) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            if any(c not in "01" for c in T) :
                return self.rewards["wrong_format"]
            
            if sum(int(s != t) for s, t in zip(self.parameter["S"], T)) > self.parameter["K"] :
                return self.rewards["invalid_solution"]
            
            now_length, answer, gold = 1, 1, self.parameter["gold_answer"]
            for i in range(1, len(T)) :
                if T[i] == T[i - 1] :
                    now_length += 1
                    answer = max(answer, now_length)
                else :
                    now_length = 1
            assert gold <= answer, "The answer should not be less than the gold answer"
            
            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                return self.rewards["rewarding_weight"] * ((gold / answer) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * int(answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]