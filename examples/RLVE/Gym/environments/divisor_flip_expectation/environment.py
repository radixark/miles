import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class DivisorFlipExpectation_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3750
    prompt_template = \
r"""You are given {N} lights labeled from 1 to {N}, each in an initial state: `1` (on) or `0` (off). The initial state is:
{state}

Each light can be toggled by pressing switches. There are {N} switches, and pressing switch `i` will **toggle the state** of all lights whose indices divide `i` (including 1 and i itself). Toggling means changing from 0 to 1 or from 1 to 0.

You play the following game:
- Repeatedly select a switch **uniformly at random** and press it, until the state of all lights is 0.
- However, if at any point it becomes possible to turn off all lights using **at most {K} switch presses**, you stop random pressing and directly use an optimal (shortest-length) sequence of switches (≤ {K} presses) to turn off all lights.

Let E be the expected number of total switch presses under this strategy. Compute the integer value of E × {N}! modulo {MOD}."""

    MOD = 10**9 + 7

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the DivisorFlipExpectation_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "wrong_range" : wrong_range,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 1, "N should be greater than or equal to 1"

        K = self.parameter["K"] = random.randint(0, N)

        one_probability = random.random()
        B = [None] + [1 if random.random() < one_probability else 0 for _ in range(N)]
        self.parameter["state"] = B.copy()


        inv = [0] * (N + 1)
        inv[1] = 1
        for i in range(2, N + 1):
            inv[i] = (self.MOD - self.MOD // i) * inv[self.MOD % i] % self.MOD

        g = [[] for _ in range(N + 1)]
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                g[j].append(i)

        tp = 0
        for i in range(N, 0, -1):
            if B[i] == 1:
                for d in g[i]:
                    B[d] ^= 1
                tp += 1

        if tp <= K:
            ans = tp
        else:
            f = [0] * (N + 1)
            f[N] = 1
            for i in range(N - 1, 0, -1):
                ans_term = (f[i + 1] + 1) % self.MOD
                f[i] = (1 + (N - i) * ans_term * inv[i]) % self.MOD

            ans = 0
            for i in range(tp, K, -1):
                ans = (ans + f[i]) % self.MOD
            ans = (ans + K) % self.MOD

        fact = 1
        for i in range(1, N + 1):
            fact = fact * i % self.MOD
        ans = ans * fact % self.MOD

        self.parameter["reference_answer"] = ans


    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            K = self.parameter["K"],
            state = "\n".join("Light {}: {}".format(i, self.parameter["state"][i]) for i in range(1, N + 1)),
            MOD = self.MOD,
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
            if not (0 <= processed_result < self.MOD) :
                return self.rewards["wrong_range"]
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]