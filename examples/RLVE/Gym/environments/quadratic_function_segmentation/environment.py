import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class QuadraticFunctionSegmentation_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3628
    prompt_template = \
r"""You are given {N} numbers A[1], A[2], ..., A[{N}]. The values are given as: {A}

You may divide these numbers (in order) into some **consecutive batches**. Let the total number of batches be k (1 ≤ k ≤ {N}), and let end[1], end[2], ..., end[k] (1 ≤ end[1] < end[2] < ... < end[k] = {N}) denote the last index in each batch. This means:
- Batch 1 contains elements A[1] to A[end[1]]
- Batch 2 contains elements A[end[1] + 1] to A[end[2]]
- ...
- Batch k contains elements A[end[k−1] + 1] to A[end[k]] (with end[k] = {N})

Define the value of a batch with sum X as: **{A_coef} × X² + {B_coef} × X + {C_coef}**. The total value of the division is the **sum of values of all batches**. I am asking you to find a batch division that **maximizes** this total value.

Output a single line containing `end[1] end[2] ... end[k]`, separated by spaces (with `end[k]` always equal to {N}`).
Example: `1 2 {N}` means:
- There are 3 batches,
- First batch ends at index 1,
- Second ends at index 2,
- Third ends at index {N} and includes the remaining elements."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_beta : float = 5.0, rewarding_weight : float = 1.0,
                 **kwargs) :
        """
        Initialize the QuadraticFunctionSegmentation_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def compute_value(self, X) -> int :
        return self.parameter["A_coef"] * (X ** 2) + self.parameter["B_coef"] * X + self.parameter["C_coef"]


    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 4, "N should be greater than or equal to 4"

        while True :
            xs = self.parameter["xs"] = [random.randint(1, N) for _ in range(N)]
            A = self.parameter["A_coef"] = -random.randint(1, N)
            B = self.parameter["B_coef"] = random.randint(1, random.randint(1, N) * random.randint(1, N) * random.randint(1, N))
            C = self.parameter["C_coef"] = random.randint(-random.randint(1, N) * random.randint(1, N) * random.randint(1, N) * random.randint(1, N) * random.randint(1, N),
                                                          +random.randint(1, N) * random.randint(1, N) * random.randint(1, N) * random.randint(1, N) * random.randint(1, N))


            # prefix sums
            s = [0] * (N + 1)
            for i in range(1, N + 1):
                s[i] = s[i - 1] + xs[i - 1]

            # dp array
            d = [0] * (N + 1)

            # deque for convex hull (indices of candidate break points)
            q = [0] * (N + 1)
            head = tail = 0
            q[0] = 0

            # helper lambdas matching the C++ macros
            def K(i):
                return 2 * A * s[i]

            def X(i):
                return s[i]

            def Y(i):
                # y(i) = d[i] + A*s[i]^2 - B*s[i]
                return d[i] + A * s[i] * s[i] - B * s[i]

            def slope(i, j):
                # (Y(i)-Y(j)) / (X(i)-X(j))
                return (Y(i) - Y(j)) / (X(i) - X(j))

            for i in range(1, N + 1):
                # pop from front while next line is better for x = s[i]
                while head < tail and slope(q[head], q[head + 1]) > K(i):
                    head += 1

                j = q[head]
                # exactly the same formula as in C++
                d[i] = -(K(i) * X(j) - Y(j) - A * s[i] * s[i] - B * s[i] - C)

                # maintain convex hull by slope ordering
                while head < tail and slope(q[tail - 1], q[tail]) <= slope(q[tail], i):
                    tail -= 1

                tail += 1
                q[tail] = i

            self.parameter["gold_answer"] = d[N]

            trivial_best = max(sum(self.compute_value(x) for x in xs), self.compute_value(sum(xs)))
            prefix_sum, suffix_sum = 0, sum(xs)
            for x in xs :
                prefix_sum += x
                suffix_sum -= x
                if prefix_sum > 0 and suffix_sum > 0 :
                    trivial_best = max(trivial_best, self.compute_value(prefix_sum) + self.compute_value(suffix_sum))
            if self.parameter["gold_answer"] > trivial_best :
                if self.parameter["gold_answer"] > 0 :
                    break
            else :
                assert self.parameter["gold_answer"] == trivial_best, "Gold answer should be greater than trivial best"
        

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = "\n".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["xs"], start = 1)),
            A_coef = self.parameter["A_coef"],
            B_coef = self.parameter["B_coef"],
            C_coef = self.parameter["C_coef"],
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                if not answer_array :
                    return None
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            N = self.parameter["N"]

            ends = processed_result
            if not (1 <= len(ends) <= N) :
                return self.rewards["invalid_solution"]
            for i in range(len(ends)) :
                if not (1 <= ends[i] <= N) :
                    return self.rewards["invalid_solution"]
                if i and not (ends[i - 1] < ends[i]) :
                    return self.rewards["invalid_solution"]
            if ends[-1] != N :
                return self.rewards["invalid_solution"]
            
            A = [None] + self.parameter["xs"]
            answer = 0
            last = 0
            for end in ends :
                batch_sum = sum(A[last + 1 : end + 1])
                answer += self.compute_value(batch_sum)
                last = end
            gold = self.parameter["gold_answer"]
            assert answer <= gold, "Answer should not be greater than gold answer"
            
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                answer = max(answer, 0)
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]