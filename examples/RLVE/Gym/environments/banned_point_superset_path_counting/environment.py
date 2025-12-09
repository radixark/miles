import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class BannedPointSupersetPathCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3734
    prompt_template = \
r"""In a three-dimensional space, you start at point (0, 0, 0) and want to reach the point ({N}, {M}, {R}). At each step, if you are currently at (x, y, z), you may move to a new (different from the current one) point of one of the following types:
1. (x', y, z) such that x AND x' = x
2. (x, y', z) such that y AND y' = y
3. (x, y, z') such that z AND z' = z  
(AND refers to the bitwise AND operation.)

You are **not allowed** to visit any of the following points:
{obstacles}

Please count the number of distinct valid paths from (0, 0, 0) to ({N}, {M}, {R}) that avoid all forbidden points. Output the result modulo {MOD}."""

    def __init__(self,
                 max_MOD : int = 10000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) -> None:
        """
        Initialize the BannedPointSupersetPathCounting_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_MOD = max_MOD
        self.rewards = {
            "wrong_format": wrong_format,
            "wrong_range": wrong_range,
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N_M_R" in self.parameter, "MAX_N_M_R is required in parameter"
        MAX_N_M_R = self.parameter["MAX_N_M_R"]
        assert MAX_N_M_R >= 1, "MAX_N_M_R should be greater than or equal to 1"

        while True :
            N, M, R = self.parameter["N"], self.parameter["M"], self.parameter["R"] = random.randint(0, MAX_N_M_R), random.randint(0, MAX_N_M_R), random.randint(0, MAX_N_M_R)
            if (2 ** N.bit_count()) * (2 ** M.bit_count()) * (2 ** R.bit_count()) - 2 >= 1 :
                break
        
        assert "MAX_O" in self.parameter, "MAX_O is required in parameter"
        MAX_O = self.parameter["MAX_O"]
        assert MAX_O >= 1, "MAX_O should be greater than or equal to 1"
        MAX_O = min(MAX_O, (2 ** N.bit_count()) * (2 ** M.bit_count()) * (2 ** R.bit_count()) - 2)
        O = self.parameter["O"] = random.randint(1, MAX_O)

        def convert_to_bits(x) -> List[int] :
            result = []
            bit = 1
            while bit <= x :
                if x & bit :
                    result.append(bit)
                bit <<= 1
            return result
        N_bits, M_bits, R_bits = convert_to_bits(N), convert_to_bits(M), convert_to_bits(R)
        def random_subset(bits : List[int]) -> int :
            bits = random.sample(bits, random.randint(0, len(bits)))
            return sum(bits)

        obstacles = set()
        while len(obstacles) < O :
            x, y, z = random_subset(N_bits), random_subset(M_bits), random_subset(R_bits)
            if (x, y, z) != (0, 0, 0) and (x, y, z) != (N, M, R) and (x, y, z) not in obstacles:
                obstacles.add((x, y, z))
        obstacles = list(obstacles)
        random.shuffle(obstacles)
        self.parameter["obstacles"] = obstacles.copy()

        MOD = self.parameter["MOD"] = random.randint(2, self.max_MOD)


        points = [(0, 0, 0)] + obstacles
        points.sort()  # lex order by x, then y, then z
        points.append((N, M, R))
        total = len(points)

        # Determine needed bit‐count dimensions
        dx = N.bit_count()
        dy = M.bit_count()
        dz = R.bit_count()
        max_d = max(dx, dy, dz)

        # Precompute binomial coefficients up to max_d
        binom = [[0] * (max_d + 1) for _ in range(max_d + 1)]
        for i in range(max_d + 1):
            binom[i][0] = 1
            for j in range(1, i + 1):
                binom[i][j] = (binom[i - 1][j - 1] + binom[i - 1][j]) % MOD

        # Precompute f[x][y][z]: number of ways from (0,0,0) to a diff‐vector with
        # x one‐bit‐flips in X, y flips in Y, z flips in Z (ignoring obstacles).
        f = [[[0] * (dz + 1) for _ in range(dy + 1)] for __ in range(dx + 1)]
        f[0][0][0] = 1
        for x in range(dx + 1):
            for y in range(dy + 1):
                for z in range(dz + 1):
                    if x == y == z == 0:
                        continue
                    val = 0
                    # transitions increasing X
                    for i in range(x):
                        val = (val + f[i][y][z] * binom[x][i]) % MOD
                    # transitions increasing Y
                    for j in range(y):
                        val = (val + f[x][j][z] * binom[y][j]) % MOD
                    # transitions increasing Z
                    for k in range(z):
                        val = (val + f[x][y][k] * binom[z][k]) % MOD
                    f[x][y][z] = val

        # DP over the sorted points
        # g[i] = (−1) * sum_{j < i, p[j] ⊆ p[i]} g[j] * f[ popcount differences ]
        g = [0] * total
        g[0] = 1  # only one way to stay at the origin
        for i in range(1, total):
            xi, yi, zi = points[i]
            acc = 0
            for j in range(i):
                xj, yj, zj = points[j]
                # check subset on all three coordinates
                if (xj & xi) == xj and (yj & yi) == yj and (zj & zi) == zj:
                    bx = (xi ^ xj).bit_count()
                    by = (yi ^ yj).bit_count()
                    bz = (zi ^ zj).bit_count()
                    acc = (acc + g[j] * f[bx][by][bz]) % MOD
            g[i] = (-acc) % MOD

        # The answer is -g[last] mod MOD, which recovers the positive sum
        self.parameter["reference_answer"] = (-g[-1]) % MOD
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            R = self.parameter["R"],
            obstacles = "\n".join("({}, {}, {})".format(x, y, z) for x, y, z in self.parameter["obstacles"]),
            MOD = self.parameter["MOD"],
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
            if not (0 <= processed_result < self.parameter["MOD"]) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]