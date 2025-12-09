import random
from typing import Optional, List
from itertools import combinations, product
from Gym.environment import VerifiableEnvironment


class RangeFourSequenceConstruction_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3876
    prompt_template = \
r"""Find a sequence of {N} integers, each being 0, 1, 2, or 3, such that no two adjacent elements form any of the pairs: '00', '11', '22', '33', '02', '20', '23', '32', '13', '31'. The sequence must also satisfy the following additional conditions: each condition is given in the form `(p_1, ..., p_L)`, meaning that the elements at positions p_1, ..., p_L (positions are numbered from 1 to {N} from left to right) must all be different.
{conditions}

Output the {N} integers of the sequence in order, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(satisfied/all)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the RangeFourSequenceConstruction_Environment instance.
        """
        super().__init__(**kwargs)

        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_weight" : rewarding_weight,
            "rewarding_beta" : rewarding_beta,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        distribution = [random.randint(1, N) for _ in range(4)]
        distribution = [d / sum(distribution) for d in distribution]
        A = []
        for i in range(N) :
            while True :
                Ai = random.choices([0, 1, 2, 3], distribution)[0]
                if not ((i > 0) and (A[i - 1], Ai) in ((0, 0), (1, 1), (2, 2), (3, 3), (0, 2), (2, 0), (2, 3), (3, 2), (1, 3), (3, 1))):
                    A.append(Ai)
                    break
        
        positions = [[] for _ in range(4)]
        for i, Ai in enumerate(A) :
            positions[Ai].append(i + 1)
        
        conditions = []
        for L in range(2, 4 + 1) :
            for As in combinations(range(4), L) :
                assert len(As) == len(set(As)) == L, "As should be distinct"
                for ps in product(*[positions[A] for A in As]) :
                    for p, Ap in zip(ps, As) :
                        assert A[p - 1] == Ap, "A[p - 1] should equal Ap"
                    conditions.append(list(ps))
        self.parameter["conditions"] = conditions = random.sample(conditions, random.randint(1, min(2 * N, len(conditions))))
        for condition in conditions :
            random.shuffle(condition)

        self.parameter["reference_answer"] = " ".join(map(str, A))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            conditions = "\n".join("Condition {}: ({})".format(i + 1, ", ".join(map(str, condition))) for i, condition in enumerate(self.parameter["conditions"])),
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

            A = processed_result
            if len(A) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            if not all(x in (0, 1, 2, 3) for x in A) :
                return self.rewards["invalid_solution"]
            for a, b in zip(A, A[1 :]) :
                if (a, b) in ((0, 0), (1, 1), (2, 2), (3, 3), (0, 2), (2, 0), (2, 3), (3, 2), (1, 3), (3, 1)) :
                    return self.rewards["invalid_solution"]
        
            satisfied = sum(int(all(A[p1 - 1] != A[p2 - 1] for p1, p2 in combinations(condition, 2))) for condition in self.parameter["conditions"])
            assert satisfied <= len(self.parameter["conditions"]), "satisfied should not exceed the number of conditions"
            if self.rewards["rewarding_strategy"] == "(satisfied/all)^beta" :
                return self.rewards["rewarding_weight"] * ((satisfied / len(self.parameter["conditions"])) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "satisfied=all" :
                return self.rewards["rewarding_weight"] * (satisfied == len(self.parameter["conditions"]))
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]