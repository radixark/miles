import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class FixedModK_Selection_Counting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3746
    prompt_template = r"""Please compute $$\left( \sum_{{i = 0}}^\infty C_{{nk}}^{{ik + r}} \right) \bmod p$$, where n = {N}, k = {K}, r = {R}, p = {MOD}."""
    
    def __init__(self,
                 MOD_range : int = 1000000,
                 wrong_format : float = -1.0, wrong_range : float = -0.5, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs):
        """
        Initialize the FixedModK_Selection_Counting_Environment instance.
        """
        super().__init__(**kwargs)
    
        self.MOD_range = MOD_range

        self.rewards = {
            "wrong_format": wrong_format,
            "wrong_range": wrong_range,
            "correct_answer": correct_answer,
            "wrong_answer": wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "MAX_N" in self.parameter, "MAX_N is required in parameter"
        MAX_N = self.parameter["MAX_N"]
        assert MAX_N >= 1, "MAX_N should be greater than or equal to 1"
        N = self.parameter["N"] = random.randint(1, MAX_N)

        assert "MAX_K" in self.parameter, "MAX_K is required in parameter"
        MAX_K = self.parameter["MAX_K"]
        assert MAX_K >= 1, "MAX_K should be greater than or equal to 1"
        K = self.parameter["K"] = random.randint(2, MAX_K)
        R = self.parameter["R"] = random.randint(0, K - 1)

        MOD = self.parameter["MOD"] = random.randint(2, self.MOD_range)
        

        def multiply(lhs, rhs, P, K):
            # Convolution modulo K with coefficients modulo P
            result = [0] * K
            for i in range(K):
                for j in range(K):
                    result[(i + j) % K] = (result[(i + j) % K] + lhs[i] * rhs[j]) % P
            return result

        def solve():
            # Prepare base vector a
            a = [0] * K
            if K == 1:
                a[0] = 2 % MOD
            else:
                a[0] = 1
                a[1] = 1

            # Identity vector for convolution exponentiation
            ans = [0] * K
            ans[0] = 1

            # Exponent: N * K
            e = N * K

            # Fast exponentiation by squaring
            while e > 0:
                if e & 1:
                    ans = multiply(ans, a, MOD, K)
                a = multiply(a, a, MOD, K)
                e >>= 1

            # Output the R-th entry
            return ans[R]

        self.parameter["reference_answer"] = solve()
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            K = self.parameter["K"],
            R = self.parameter["R"],
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