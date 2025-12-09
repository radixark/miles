import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class DoubleCrossCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3221
    prompt_template = \
r"""A **double cross** is a specific shape consisting of two horizontal and one vertical segments of `1`s. For example:
```
..........
....1.....      ..1..
..11111...      .111.
....1.....      ..1..
.1111111..      11111
....1.....      ..1..
....1.....
..........
```
A valid double cross must satisfy the following conditions:
- The two horizontal segments must not lie on adjacent rows.
- The vertical segment must extend strictly above and strictly below the two horizontal segments.
- The vertical segment must divide both horizontal segments into two equal halves.
- The upper horizontal segment must be strictly shorter than the lower one.
- (Thus, the example on the right is the smallest valid double cross.)

In the following example, we are given a 0/1 matrix:
```
10001011
10111111
10001101
11111110
11111111
11101011
```
There are 5 valid double crosses in this matrix:
```
....1...  ....1...  ....1...
...111..  ...111..  ...111..
....1...  ....1...  ....1...
..11111.  ..11111.  ....1...
....1...  ....1...  ..11111.
........  ....1...  ....1...

....1...  ....1...
...111..  ..11111.
....1...  ....1...
....1...  ....1...
.1111111  .1111111
....1...  ....1...
```

Now, given a 0/1 matrix of size {N} × {M}, where each cell is either `1` or `0`. The coordinates of 0-cells are given as follows (0-indexed):
{zero_coordinates}

More formally, a double cross in the matrix (assuming 0-indexed rows and columns) is defined by the following parameters:
- Four row indices: x_top, x_up, x_down, x_bottom, satisfying: 0 ≤ x_top < x_up, x_up + 1 < x_down < x_bottom < {N}
- One column index y_mid, and two integers up_len and down_len, such that: 0 ≤ y_mid < {M}, 1 ≤ up_len < down_len, and y_mid - down_len ≥ 0, y_mid + down_len < {M}
- The vertical segment of the cross is formed by the column y_mid spanning from x_top to x_bottom, and all cells (x, y_mid) for x_top ≤ x ≤ x_bottom must be `1`
- The upper horizontal segment lies on row x_up, and all cells (x_up, y) for y_mid - up_len ≤ y ≤ y_mid + up_len must be `1`; The lower horizontal segment lies on row x_down, and all cells (x_down, y) for y_mid - down_len ≤ y ≤ y_mid + down_len must be `1`


Please compute how many valid double crosses exist in the matrix."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the DoubleCrossCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M" in self.parameter, "MAX_N_M is required in parameter"
        MAX_N_M = self.parameter["MAX_N_M"]
        assert MAX_N_M >= 5, "MAX_N_M should be greater than or equal to 5"

        N, M = self.parameter["N"], self.parameter["M"] = random.randint(5, MAX_N_M), random.randint(5, MAX_N_M)
        assert N >= 5 and M >= 5, "N and M should be greater than or equal to 5"
        zero_coordinates = self.parameter["zero_coordinates"] = random.sample([(x, y) for x in range(N) for y in range(M)], random.randint(1, int(N * M * 0.25)))


        size = N * M + 1                          # 1-based indexing
        vis = [True] * size                       # True  => '1',  False => '0'

        for x, y in zero_coordinates:
            x += 1
            y += 1
            vis[(x - 1) * M + y] = False

        # ------------------------------------------------------------
        # 2. pre-compute arm lengths
        # ------------------------------------------------------------
        L = [0] * size            # horizontal half-length (min of both sides) – 1
        U = [0] * size            # vertical length upward – 1
        D = [0] * size            # vertical length downward – 1

        # left sweep
        for r in range(1, N + 1):
            streak = 0
            base = (r - 1) * M
            for c in range(1, M + 1):
                idx = base + c
                streak = streak + 1 if vis[idx] else 0
                L[idx] = streak

        # right sweep
        for r in range(1, N + 1):
            streak = 0
            base = (r - 1) * M
            for c in range(M, 0, -1):
                idx = base + c
                streak = streak + 1 if vis[idx] else 0
                L[idx] = min(L[idx], streak)
                if L[idx]:
                    L[idx] -= 1                    # exclude the centre cell

        # upward sweep
        for c in range(1, M + 1):
            streak = 0
            idx = c
            for r in range(1, N + 1):
                streak = streak + 1 if vis[idx] else 0
                U[idx] = streak - 1 if streak else 0
                idx += M

        # downward sweep
        for c in range(1, M + 1):
            streak = 0
            idx = (N - 1) * M + c
            for r in range(N, 0, -1):
                streak = streak + 1 if vis[idx] else 0
                D[idx] = streak - 1 if streak else 0
                idx -= M

        # ------------------------------------------------------------
        # 3. Fenwick tree with “three-dimensional” coefficient arrays A, B, C
        #    (range-update, prefix-sum query for quadratic weights)
        # ------------------------------------------------------------
        A = [0] * (M + 1)
        B = [0] * (M + 1)
        C = [0] * (M + 1)
        tag = [0] * (M + 1)        # lazy versioning for O(#updates) clearing
        version = 1

        def lb(x: int) -> int:               # lowest set bit
            return x & -x

        def fenwick_add(x: int, w: int) -> None:
            """point-update helper used by the range-add routine"""
            i = x
            while i <= M:
                if tag[i] != version:        # clear lazily if we are in a new version
                    tag[i] = version
                    A[i] = B[i] = C[i] = 0
                A[i] += w
                B[i] += x * w
                C[i] += (x * x) * w
                i += lb(i)

        def range_add(l: int, r: int, w: int) -> None:
            """add w to every position in [l, r] (1-based, inclusive)"""
            if l > r or w == 0:
                return
            fenwick_add(l, w)
            fenwick_add(r + 1, -w)

        def prefix_query(x: int) -> int:
            """∑_{t ≤ x} ( (t + 3)·t + 2 )/2 * freq(t)  where freq(t) is the value after range adds"""
            if x <= 0:
                return 0
            s1 = s2 = s3 = 0
            i = x
            while i:
                if tag[i] == version:
                    s1 += A[i]
                    s2 += B[i]
                    s3 += C[i]
                i -= lb(i)
            res = ((x + 3) * x + 2)
            res = (res * s1 + s3 - (2 * x + 3) * s2)
            return res // 2

        # ------------------------------------------------------------
        # 4. sweep each column, building counts on the fly
        # ------------------------------------------------------------
        answer = 0

        for col in range(2, M):          # centres cannot be on the very first/last column
            version += 1                 # “clear” the Fenwick tree for this column

            for row in range(3, N):      # need at least two rows above & below
                idx = (row - 1) * M + col

                if not vis[idx]:         # a ‘0’ breaks the vertical arm
                    version += 1         # (lazy clear)
                    continue

                # take current cell as *lower* horizontal bar
                if L[idx]:
                    answer += D[idx] * prefix_query(L[idx] - 1)

                # push the row immediately above as a candidate *upper* bar
                upper = idx - M
                if L[upper] and U[upper]:
                    range_add(1, L[upper], U[upper])

        self.parameter["reference_answer"] = answer


    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            zero_coordinates = "\n".join("({}, {})".format(x, y) for x, y in self.parameter["zero_coordinates"]),
        )


    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                int_answer = int(answer)
                return int_answer
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                if processed_result == 0 :
                    return self.rewards["rewarding_weight"] * (self.parameter["reference_answer"] == 0)
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]