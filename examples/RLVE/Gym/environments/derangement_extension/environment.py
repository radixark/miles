import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class DerangementExtension_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P4071
    prompt_template = r"""What's the number of permutations p of 1, 2, ..., {N} such that exactly {M} indices i satisfy p[i] = i (1-indexed)? Let me know the result modulo {MOD}."""
    MODs = (666623333, 998244353, 10 ** 9 + 7)

    def __init__(self,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the DerangementExtension_Environment instance.
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
        assert N >= 4, "N should be greater than or equal to 4"

        M = self.parameter["M"] = random.randint(0, N)
        MOD = self.parameter["MOD"] = random.choice(self.MODs)


        def init(max_n):
            prod = [1] * (max_n + 1)
            inv = [0] * (max_n + 1)
            for i in range(1, max_n + 1):
                prod[i] = (prod[i - 1] * i) % MOD
                inv[i] = pow(prod[i], MOD - 2, MOD)  # modular inverse via Fermat, faithful to C++ logic

            a = [0] * (max_n + 1)  # derangements
            if max_n >= 2:
                a[2] = 1
            for i in range(3, max_n + 1):
                a[i] = (i - 1) * ((a[i - 1] + a[i - 2]) % MOD) % MOD
            return prod, inv, a

        prod, inv, a = init(N)

        def compute() :
            if M == 0:
                return a[N] % MOD
            if N == M:
                return 1
            if N - 1 == M:
                return 0
            # C(N, M) * D_{N-M}
            comb = (prod[N] * inv[M] % MOD) * inv[N - M] % MOD
            ans = (comb * a[N - M]) % MOD
            return ans
        
        self.parameter["reference_answer"] = compute()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(N = self.parameter["N"], M = self.parameter["M"], MOD = self.parameter["MOD"])
    

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