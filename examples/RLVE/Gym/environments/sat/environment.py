import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class SAT_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""There are {N} boolean (0/1) values x[0], x[1], ..., x[{N_minus_1}]. Each of the following {M} expressions (`|` means OR, `!` means NOT) must equal 1:
{expressions}

Please find any solution x[0], x[1], ..., x[{N_minus_1}] that satisfies the conditions above.

Output Format: Your final answer should be a single line containing x[0], x[1], ..., x[{N_minus_1}], separated by **spaces**.
Example: `{N_boolean}` (do **NOT** include quotes or backticks)."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SAT_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 2, "N should be greater than or equal to 2"

        assert "M" in self.parameter, "M is required in parameter"
        M = self.parameter["M"]
        assert M >= 1, "M should be greater than or equal to 1"

        assert "density" in self.parameter, "density is required in parameter"
        density = self.parameter["density"]
        assert 0 < density <= 1, "density should be in (0, 1]"

        x = [random.randint(0, 1) for i in range(N)]
        self.parameter["reference_answer"] = " ".join(map(str, x))

        clauses = self.parameter["clauses"] = []
        for m in range(M) :
            while True :
                clause = []
                all_or = False
                for index in range(N) :
                    if random.random() < density :
                        clause.append((index, random.random() < 0.5))
                        all_or |= (x[index] if clause[-1][-1] else not x[index])
                if len(clause) >= 2 and all_or :
                    break
            clauses.append(clause)
    
    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            N_minus_1 = self.parameter["N"] - 1,
            M = self.parameter["M"],
            expressions = "\n".join(" | ".join("({}x[{}])".format("" if is_positive else "!", index) for index, is_positive in clause) for clause in self.parameter["clauses"]),
            N_boolean = " ".join(str(i % 2) for i in range(self.parameter["N"])),
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

            x = processed_result
            if len(x) != self.parameter["N"] :
                return self.rewards["wrong_format"]
            if not all(xi in (0, 1) for xi in x) :
                return self.rewards["wrong_format"]
            
            satisfied = sum(int(any(x[index] if is_positive else not x[index] for index, is_positive in clause)) for clause in self.parameter["clauses"])
            
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / len(self.parameter["clauses"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == len(self.parameter["clauses"]))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]