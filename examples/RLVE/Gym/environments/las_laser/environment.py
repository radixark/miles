import random
from typing import Optional
from functools import cmp_to_key
from Gym.environment import VerifiableEnvironment


class LASLaser_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3562
    prompt_template = \
r"""There are {N} segments in the 2D plane, given as:
{segments}

You may shoot at most {K} rays from the origin (0, 0) in any directions. Each segment is allowed to intersect with **at most one** of these rays. Please output the **maximum number of segments** that can be intersected by a single ray."""

    def __init__(self,
                 wrong_format: float = -1.0, correct_answer: float = 1.0, incorrect_answer: float = 0.0,
                 **kwargs):
        """
        Initialize the LASLaser_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format": wrong_format,
            "correct_answer": correct_answer,
            "incorrect_answer": incorrect_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        self.parameter["segments"] = segments = [(random.randint(1, 2 * N), random.randint(1, 2 * N), random.randint(1, 2 * N), random.randint(1, 2 * N)) for _ in range(N)]


        # load all 2·N endpoint vectors
        p0 = [None] * (2 * N)
        for i, (x1, y1, x2, y2) in enumerate(segments):
            p0[i]     = (x1, y1)
            p0[N + i] = (x2, y2)

        # comparator for sorting by angle via cross‐product
        def cmp(i, j):
            x1, y1 = p0[i]
            x2, y2 = p0[j]
            c = x1 * y2 - y1 * x2
            if c > 0:
                return -1   # i comes before j
            elif c < 0:
                return 1    # i comes after j
            else:
                return 0    # same direction

        # sort all endpoint‐indices by their angle from the origin
        p = list(range(2 * N))
        p.sort(key=cmp_to_key(cmp))

        # discretize unique directions into 1..top
        w = [0] * (2 * N)
        top = 1
        now = p[0]
        w[now] = 1
        for idx in p[1:]:
            # if this direction is not collinear with 'now', it's a new bucket
            if p0[idx][0] * p0[now][1] - p0[idx][1] * p0[now][0] != 0:
                top += 1
                now = idx
            w[idx] = top

        # prepare interval data structures
        size = top + 2
        INF = top + 1
        left = [INF] * size
        num  = [0]   * size

        # build intervals [x, y] on the angle‐index line for each segment
        for i in range(N):
            a = w[i]
            b = w[N + i]
            if a > b:
                a, b = b, a
            # record the leftmost start for any interval ending at b
            if a < left[b]:
                left[b] = a
            # difference array to count how many intervals cover each point
            num[a] += 1
            num[b + 1] -= 1

        # prefix‐sum to get coverage count at each discrete angle
        for i in range(1, top + 1):
            num[i] += num[i - 1]

        # make left[i] = min(left[i..top])
        for i in range(top - 1, 0, -1):
            if left[i] > left[i + 1]:
                left[i] = left[i + 1]

        # DP: f[i] = max covered with last ray chosen at or before i
        f = [0] * size
        Ks, Answers = [], []
        for K in range(1, N + 1) :
            # try placing one more ray at each i, in descending order
            for i in range(top, 0, -1):
                cand = f[left[i] - 1] + num[i]
                if cand > f[i]:
                    f[i] = cand
            # allow skipping placing at i (carry forward max)
            for i in range(1, top + 1):
                if f[i - 1] > f[i]:
                    f[i] = f[i - 1]

            if len(Answers) == 0 or f[top] > Answers[-1]:
                Ks.append(K)
                Answers.append(f[top])
            if Answers[-1] == N:
                break
        index = random.randint(0, len(Answers) - 1)
        self.parameter["K"], self.parameter["reference_answer"] = Ks[index], Answers[index]
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            segments = "\n".join("({}, {})-({}, {})".format(x1, y1, x2, y2) for (x1, y1, x2, y2) in self.parameter["segments"]),
            K = self.parameter["K"],
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
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["incorrect_answer"]
        else :
            return self.rewards["wrong_format"]