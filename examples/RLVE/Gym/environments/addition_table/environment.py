import random
from typing import Optional, Dict
from Gym.environment import VerifiableEnvironment


class AdditionTable_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1013
    prompt_template = \
r"""You are given an unknown base-N number system (N is an integer ≥ 3), and {N} distinct digits {ALL_LETTERS} in that system. The digits satisfy the following equations in base-N:

{EQUATIONS}

Note:
- {ALL_LETTERS} are distinct digits in the range [0, N−1].
- Expressions like ba represent base-N numbers formed by **concatenation**. For example, if a=1 and b=2, then ba = "21" in base-N.

Your task is to find the correct base N (in decimal), and the values of {ALL_LETTERS} (also in decimal) that satisfy all the equations.

Output Format:
Your final answer should be a single line containing N, {ALL_LETTERS} (all in decimal), separated by **spaces**.
Example: `{N_plus_1} {EXAMPLE_1}` (do **NOT** include the backticks or quotes); this means N={N_plus_1}, {EXAMPLE_2}.
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_answer : float = -0.5, wrong_N : float = 0.0, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        """
        Initialize the AdditionTable_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_answer" : invalid_answer,
            "wrong_N" : wrong_N,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N in range(3, 26 + 1), "N should be in the range [3, 26]"

        digit2letter = self.parameter["digit2letter"] = [chr(i) for i in range(97, 97 + N)]
        random.shuffle(digit2letter)

        letter2digit = {letter : digit for digit, letter in enumerate(digit2letter)}
        self.parameter["reference_answer"] = "{} {}".format(N, " ".join([str(letter2digit[chr(i)]) for i in range(97, 97 + N)]))
    

    def convert_to_expression(self, n : int) -> str :
        N = self.parameter["N"]
        
        if n == 0 :
            return self.parameter["digit2letter"][0]
        else :
            expression = ""
            while n > 0 :
                digit = n % N
                expression = self.parameter["digit2letter"][digit] + expression
                n //= N
            return expression

    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        ALL_LETTERS = ", ".join([chr(i) for i in range(97, 97 + N)])

        digit2letter = self.parameter["digit2letter"]
        letter2digit = {letter : digit for digit, letter in enumerate(digit2letter)}

        EQUATIONS = []
        for a_ascii in range(97, 97 + N) :
            for b_ascii in range(a_ascii, 97 + N) :
                a = chr(a_ascii)
                b = chr(b_ascii)
                EQUATIONS.append("{} + {} = {}".format(a, b, self.convert_to_expression(letter2digit[a] + letter2digit[b])))
        EQUATIONS = "\n".join(EQUATIONS)

        return self.prompt_template.format(
            ALL_LETTERS = ALL_LETTERS,
            EQUATIONS = EQUATIONS,
            N = N,
            N_plus_1 = N + 1,
            EXAMPLE_1 = " ".join([str(_) for _ in range(N)]),
            EXAMPLE_2 = ", ".join(["{}={}".format(chr(i), i - 97) for i in range(97, 97 + N)]),
        )
    

    def _process(self, answer : Optional[str]) -> Optional[Dict] :
        if answer is not None :
            answer = answer.strip()
            try :
                answer_array = list(map(int, answer.split()))
                if len(answer_array) != self.parameter["N"] + 1 :
                    return dict()
                N = answer_array[0]
                digits = answer_array[1 :]
                return dict(N = N, digits = digits)
            except ValueError :
                return dict()
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if not processed_result :
                return self.rewards["invalid_answer"]
            
            N = processed_result["N"]
            if N != self.parameter["N"] :
                return self.rewards["wrong_N"]
            
            predict_digits = processed_result["digits"]
            assert len(predict_digits) == N, "digits should have the same length as N"

            letter2digit = {letter : digit for digit, letter in enumerate(self.parameter["digit2letter"])}
            assert len(letter2digit) == N, "letter2digit should have the same length as N"
            gold_digits = [letter2digit[chr(i)] for i in range(97, 97 + N)]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(float(a == b) for a, b in zip(gold_digits, predict_digits)) / N) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * all(a == b for a, b in zip(gold_digits, predict_digits))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]