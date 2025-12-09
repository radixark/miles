import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class PolynomialInterpolation_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""You are given a polynomial of degree {N} in the form: f(x) = a_0 * x^0 + a_1 * x^1 + ... + a_{N} * x^{N}, where the coefficients `a_0, a_1, ..., a_{N}` are integers.

It is known that the polynomial passes through the following {N_plus_1} points:  
{points}

Please determine the coefficients a_0, a_1, ..., a_{N}.

**Output Format:** Your final answer should be a single line containing `a_0 a_1 ... a_{N}` (do **NOT** include backticks or quotes), separated by spaces."""

    def __init__(self,
                 max_weight : int = 5,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 10.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the PolynomialInterpolation_Environment instance.
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
    

    def compute(self, x : int) -> int :
        return sum(coeff * (x ** i) for i, coeff in enumerate(self.parameter["coeffs"]))
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        self.parameter["coeffs"] = [random.randint(-self.max_weight, self.max_weight) for degree in range(N)] + [random.randint(1, self.max_weight)]
        self.parameter["reference_answer"] = " ".join(map(str, self.parameter["coeffs"]))

        X = self.parameter["X"] = random.sample(range(-N, +N + 1), N + 1)
        Y = self.parameter["Y"] = [self.compute(x) for x in X]
    

    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_plus_1 = N + 1,
            points = "\n".join("f({}) = {}".format(x, y) for x, y in zip(self.parameter["X"], self.parameter["Y"])),
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

            if len(processed_result) != self.parameter["N"] + 1 :
                return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["coeffs"], processed_result)) / (self.parameter["N"] + 1)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["coeffs"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]