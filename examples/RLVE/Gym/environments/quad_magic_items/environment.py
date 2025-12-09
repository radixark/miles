import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class QuadMagicItems_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P2119
    prompt_template = \
r"""You are given {N} items, each with a positive value. The values of the items are:
{X}

We say that four items with indices `a, b, c, d` form a **magic formation** if their values satisfy:
- X[a] < X[b] < X[c] < X[d]
- X[b] - X[a] = 2 × (X[d] - X[c])
- X[b] - X[a] < (X[c] - X[b]) / 3

In such a formation, items `a`, `b`, `c`, and `d` are called type `A`, `B`, `C`, and `D` respectively.

**Output Format:** Output {N} lines. The i-th line should contain four integers, representing the number of times the i-th item is used as an `A`, `B`, `C`, and `D` item in any valid magic formation. The four values should be separated by spaces."""

    def __init__(self,
                 weight_range_multiple : int = 1,
                 wrong_format : float = -1.0, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the QuadMagicItems_Environment instance.
        """
        super().__init__(**kwargs)

        self.weight_range_multiple = weight_range_multiple
        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 5, "N should be greater than or equal to 5"

        X = self.parameter["X"] = [random.randint(1, N * self.weight_range_multiple) for _ in range(N)]


        # Count how many items have each magic value
        MAX = max(X)
        cnt = [0] * (MAX + 1)
        for xi in X:
            cnt[xi] += 1

        # ans_val[v][0] = times value v is used as A
        # ans_val[v][1] = times value v is used as B
        # ans_val[v][2] = times value v is used as C
        # ans_val[v][3] = times value v is used as D
        ans_val = [[0, 0, 0, 0] for _ in range(MAX + 1)]

        # Enumerate t such that 9*t <= N-2
        for t in range(1, (MAX - 2) // 9 + 1):
            # Forward pass: accumulate over d increasing
            s = 0
            for d in range(9 * t + 2, MAX + 1):
                a = d - 9 * t - 1
                b = a + 2 * t
                c = d - t
                s += cnt[a] * cnt[b]
                # add all new magic arrays ending at (c, d)
                ans_val[c][2] += s * cnt[d]   # as C
                ans_val[d][3] += s * cnt[c]   # as D

            # Backward pass: accumulate over a decreasing
            s = 0
            for a in range(MAX - 9 * t - 1, 0, -1):
                b = a + 2 * t
                c = b + 6 * t + 1
                d = c + t
                s += cnt[c] * cnt[d]
                # add all new magic arrays starting at (a, b)
                ans_val[a][0] += s * cnt[b]   # as A
                ans_val[b][1] += s * cnt[a]   # as B

        # Output results for each item in input order
        self.parameter["gold_answer"] = []
        self.parameter["reference_answer"] = ""
        for xi in X:
            A_cnt, B_cnt, C_cnt, D_cnt = ans_val[xi]
            self.parameter["gold_answer"].append((A_cnt, B_cnt, C_cnt, D_cnt))
            self.parameter["reference_answer"] += "{} {} {} {}\n".format(A_cnt, B_cnt, C_cnt, D_cnt)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], X = " ".join("X[{}]={}".format(i + 1, x) for i, x in enumerate(self.parameter["X"])))


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                matrix = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        matrix.append(tuple(map(int, line.split())))
                        if len(matrix[-1]) != 4 :
                            return None
                return matrix
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]