import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class Cornfield_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3287
    prompt_template = \
r"""You are given an array `H` of length {N}. The initial values of the array are: {H}
You may perform **at most {K} operations**. In each operation, you choose an interval [L, R] (0 ≤ L ≤ R < {N}), and increment each element H[i] by 1 for all i in the range L ≤ i ≤ R. Try your best to **maximize the length of the longest non-decreasing subsequence** (not necessarily contiguous) in the final array after performing the operations.

**Output Format:** Output at most {K} lines. Each line should contain two integers L and R (0-indexed), separated by a space, indicating an interval you chose for an operation."""

    def __init__(self,
                wrong_format: float = -1.0, invalid_solution: float = -0.5, rewarding_strategy: str = "(answer/gold)^beta", rewarding_weight: float = +1.0, rewarding_beta: float = 5.0,
                **kwargs):
        """
        Initialize the Cornfield_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "invalid_solution": invalid_solution,
            "rewarding_strategy": rewarding_strategy,
            "rewarding_weight": rewarding_weight,
            "rewarding_beta": rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        H = self.parameter["H"] = [random.randint(1, 2 * N) for _ in range(N)]
        K = self.parameter["K"] = random.randint(1, max(1, min(N, sum(max(H[i - 1] - H[i], 0) for i in range(1, N)))))


        def lowbit(x: int) -> int:
            """Return the lowest set bit of x."""
            return x & -x

        def add(bit, X, Y, x, y, value):
            """2-D BIT (Fenwick) – update point (x, y) to max(current, value)."""
            while x <= X:
                yy = y
                row = bit[x]
                while yy <= Y:
                    if value > row[yy]:
                        row[yy] = value
                    yy += lowbit(yy)
                x += lowbit(x)

        def query(bit, x, y):
            """2-D BIT (Fenwick) – max over rectangle (1..x , 1..y)."""
            res = 0
            while x:
                yy = y
                row = bit[x]
                while yy:
                    v = row[yy]
                    if v > res:
                        res = v
                    yy -= lowbit(yy)
                x -= lowbit(x)
            return res

        max_height = max(H)
        X = K + 1                     # first dimension size
        Y = max_height + K            # second dimension size (heights after boosts)

        # 2-D BIT initialised with 0 (lists are 1-based for Fenwick convenience)
        BIT = [[0] * (Y + 2) for _ in range(X + 2)]

        answer = 0
        for h in H:                   # iterate through every corn stalk
            for j in range(K, -1, -1):          # j = # boosts that will still cover this stalk
                cur_height = h + j              # final possible height of this stalk
                best = query(BIT, j + 1, cur_height) + 1
                if best > answer:
                    answer = best
                add(BIT, X, Y, j + 1, cur_height, best)

        self.parameter["gold_answer"] = answer
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K = self.parameter["K"],
            H = " ".join("H[{}]={}".format(i, Hi) for i, Hi in enumerate(self.parameter["H"])),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                operations = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        L, R = map(int, line.split())
                        operations.append((L, R))
                return operations
            except :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) > self.parameter["K"] :
                return self.rewards["invalid_solution"]

            delta = [0] * self.parameter["N"]
            for L, R in processed_result :
                if not (0 <= L <= R < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                delta[L] += 1
                if R + 1 < self.parameter["N"] :
                    delta[R + 1] -= 1
            
            H = self.parameter["H"].copy()
            for i in range(self.parameter["N"]) :
                if i > 0 :
                    delta[i] += delta[i - 1]
                H[i] += delta[i]
            
            F = [0] * self.parameter["N"]
            for i in range(self.parameter["N"]) :
                F[i] = 1
                for j in range(i) :
                    if H[j] <= H[i] :
                        F[i] = max(F[i], F[j] + 1)
            answer, gold = max(F), self.parameter["gold_answer"]
            assert 1 <= answer <= gold, "answer should be between 1 and gold_answer"

            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * int(answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]