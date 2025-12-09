import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class PolynomialRemainder_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given two polynomials:
- P(x) of degree {N}: P(x) = {P}
- Q(x) of degree {M}: Q(x) = {Q}

There exists a unique polynomial R(x) such that: P(x) = Q(x) * R(x) + S(x), where S(x) is the **remainder polynomial** and its degree is **less than {M}**. Let the coefficients of S(x) be `s_0, ..., s_{M_minus_1}` (if the degree of S(x) is less than {M_minus_1}, pad the remaining coefficients with zeros); we know that the coefficients of S(x) are all integers.

**Output Format:** Your final answer should be a single line containing `s_0 ... s_{M_minus_1}` (do **NOT** include backticks or quotes), separated by spaces.
"""

    def __init__(self,
                 max_weight : int = 5,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the PolynomialRemainder_Environment instance.
        """
        super().__init__(**kwargs)

        self.max_weight = max_weight
        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert N >= M >= 2, "M should be less than or equal to N and greater than or equal to 2"

        self.parameter["Q_coeffs"] = [random.randint(-self.max_weight, self.max_weight) for degree in range(M)] + [random.randint(1, self.max_weight)]
        self.parameter["R_coeffs"] = [random.randint(-self.max_weight, self.max_weight) for degree in range(N - M)] + [random.randint(1, self.max_weight)]
        self.parameter["S_coeffs"] = [random.randint(-self.max_weight, self.max_weight) for degree in range(M)]

        self.parameter["P_coeffs"] = [0] * (N + 1)
        for Qi in range(M + 1) :
            for Ri in range(N - M + 1) :
                self.parameter["P_coeffs"][Qi + Ri] += self.parameter["Q_coeffs"][Qi] * self.parameter["R_coeffs"][Ri]
        for Si in range(M) :
            self.parameter["P_coeffs"][Si] += self.parameter["S_coeffs"][Si]
        
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["S_coeffs"]))
    

    def _prompt_generate(self) -> str :
        N, M = self.parameter["N"], self.parameter["M"]
        return self.prompt_template.format(
            N = N,
            M = M,
            M_minus_1 = M - 1,
            P = " + ".join("({}) * x^{}".format(coefficient, i) for i, coefficient in enumerate(self.parameter["P_coeffs"]) if coefficient != 0),
            Q = " + ".join("({}) * x^{}".format(coefficient, i) for i, coefficient in enumerate(self.parameter["Q_coeffs"]) if coefficient != 0),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                return answer_array
            except ValueError :
                return None # Invalid answer format
        else :
            return None # Invalid answer format
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            if len(processed_result) != self.parameter["M"] :
                return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["S_coeffs"], processed_result)) / self.parameter["M"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["S_coeffs"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]