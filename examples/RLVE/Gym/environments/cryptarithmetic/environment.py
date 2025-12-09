import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


def Add(a_digits : List[int], b_digits : List[int], base : int) -> List[int] :
    c_digits = []

    carray = 0
    for i in range(max(len(a_digits), len(b_digits))) :
        a = a_digits[i] if i < len(a_digits) else 0
        b = b_digits[i] if i < len(b_digits) else 0

        c = a + b + carray
        carray = c // base
        c_digits.append(c % base)
    if carray > 0 :
        c_digits.append(carray)
    
    return c_digits


class Cryptarithmetic_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""Now consider a number system with base {N}, which uses digits d[0], d[1], ..., d[{N_minus_1}].
Each d[i] is a unique integer in the range [0, {N_minus_1}], but their actual values are unknown.

We define the number `d[i0]d[i1]...d[ik]` to represent the value `d[i0] * {N}^k + d[i1] * {N}^(k-1) + ... + d[ik] * {N}^0`,
where `d[i]` is the actual digit assigned to index `i`, and the number is visually written using the digits `d[i0]`, `d[i1]`, ..., `d[ik]`.

You are given the following equation in this unknown base-{N} digit system:
{addend_1}
+
{addend_2}
=
{sum_result}

Your task is to find one possible assignment of values (in decimal) for d[0], d[1], ..., d[{N_minus_1}] such that the equation holds true.

Output Format:
Your final answer should be a single line containing the decimal values of d[0], d[1], ..., d[{N_minus_1}], in order, separated by spaces.
Example: `{all_digits_in_order}` (do **NOT** include the backticks or quotes); this means d[0] = 0, d[1] = 1, ..., d[{N_minus_1}] = {N_minus_1}.
"""

    def __init__(self,
                 wrong_format : float = -1.0, not_permutation : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 3.0,
                 **kwargs) :
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "not_permutation" : not_permutation,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        digits = self.parameter["digits"] = list(range(N))
        random.shuffle(digits)
        self.parameter["reference_answer"] = " ".join([str(digits[i]) for i in range(N)])

        assert "addend_length" in self.parameter, "addend_length is required in parameter"
        addend_length = self.parameter["addend_length"]
        addend_1 = self.parameter["addend_1"] = [random.randint(0 if _ < addend_length - 1 else 1, N - 1) for _ in range(addend_length)]
        addend_2 = self.parameter["addend_2"] = [random.randint(0 if _ < addend_length - 1 else 1, N - 1) for _ in range(addend_length)]
        self.parameter["sum_result"] = Add(addend_1, addend_2, N)
        # self.parameter["addend_1"], self.parameter["addend_2"], self.parameter["sum_result"] are all the actual digits
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        gold_digit2i = {digit : i for i, digit in enumerate(self.parameter["digits"])}
        def print_dis(digits : List[int]) -> str :
            return "".join(["d[{}]".format(gold_digit2i[digits[i]]) for i in range(len(digits) - 1, -1, -1)]) # digits[gold_digit2i[digits[i] = digit]] = digit
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            addend_1 = print_dis(self.parameter["addend_1"]),
            addend_2 = print_dis(self.parameter["addend_2"]),
            sum_result = print_dis(self.parameter["sum_result"]),
            all_digits_in_order = " ".join([str(i) for i in range(N)]),
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
            
            N = self.parameter["N"]
            if len(processed_result) != N :
                return self.rewards["not_permutation"]
            if len(set(processed_result)) != N :
                return self.rewards["not_permutation"]
            for i in processed_result :
                if not (0 <= i < N) :
                    return self.rewards["not_permutation"]
            
            digits = processed_result

            gold_digit2i = {digit : i for i, digit in enumerate(self.parameter["digits"])}
            addend_1 = [digits[gold_digit2i[digit]] for digit in self.parameter["addend_1"]]
            addend_2 = [digits[gold_digit2i[digit]] for digit in self.parameter["addend_2"]]
            sum_result = Add(addend_1, addend_2, N)
            gold_sum_result = self.parameter["sum_result"].copy()

            if len(sum_result) < len(gold_sum_result) :
                assert len(sum_result) == self.parameter["addend_length"] and len(gold_sum_result) == self.parameter["addend_length"] + 1
                sum_result.append(0)
            elif len(sum_result) > len(gold_sum_result) :
                assert len(sum_result) == self.parameter["addend_length"] + 1 and len(gold_sum_result) == self.parameter["addend_length"]
                gold_sum_result.append(0)
            else :
                pass
            assert len(sum_result) == len(gold_sum_result), "sum_result and gold_sum_result should have the same length"

            digit2i = {digit : i for i, digit in enumerate(digits)}
            sum_result = [digit2i[digit] for digit in sum_result]
            gold_sum_result = [gold_digit2i[digit] for digit in gold_sum_result]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(float(a == b) for a, b in zip(sum_result, gold_sum_result)) / len(sum_result)) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * all(a == b for a, b in zip(sum_result, gold_sum_result))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]