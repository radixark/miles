import heapq
import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class HURWarehouseStore_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3545
    prompt_template = \
r"""You are running a warehouse store for {N} days. On the morning of day i, you receive A[i] items; in the evening of the same day, a customer arrives and demands B[i] items. You can choose to satisfy the customer only if you have at least B[i] items in stock. The arrays A and B are given as follows:
{A_and_B}

Please maximize the number of customers you can satisfy. Output a single line containing the indices of the days when you satisfy the customers, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the HURWarehouseStore_Environment instance.
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

        while True :
            A = self.parameter["A"] = [random.randint(0, N) for _ in range(N)]
            B = self.parameter["B"] = [random.randint(1, N) for _ in range(N)]
            
            answer_not_zero, stock = False, 0
            for Ai, Bi in zip(A, B) :
                stock += Ai
                if stock >= Bi :
                    answer_not_zero = True
                    break
            if answer_not_zero :
                break
        

        tot = 0
        count = 0
        # max-heap of (b_value, index), implemented by pushing (-b_value, index)
        heap = []
        vis = [False] * N

        for i in range(N):
            tot += A[i]

            # If we can't satisfy B[i], but there's a previously accepted day with a larger demand,
            # remove that day instead to free up space
            if heap and tot < B[i]:
                # peek at largest b so far
                largest_b, idx = heap[0]
                largest_b = -largest_b
                if largest_b > B[i]:
                    # remove it
                    heapq.heappop(heap)
                    vis[idx] = False
                    tot += largest_b
                    count -= 1

            # Try to accept today
            if tot >= B[i]:
                tot -= B[i]
                heapq.heappush(heap, (-B[i], i))
                vis[i] = True
                count += 1

        assert count > 0, "There should be at least one customer satisfied"
        self.parameter["gold_answer"] = count
        self.parameter["reference_answer"] = " ".join(str(i+1) for i, v in enumerate(vis) if v)
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A_and_B = "\n".join("A[{}]={} B[{}]={}".format(i + 1, Ai, i + 1, Bi) for i, (Ai, Bi) in enumerate(zip(self.parameter["A"], self.parameter["B"]))),
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

            satisfy = [False] * self.parameter["N"]
            for day in processed_result :
                day -= 1
                if not (0 <= day < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                if satisfy[day] :
                    return self.rewards["invalid_solution"]
                satisfy[day] = True
            
            stock = 0
            for sold, Ai, Bi in zip(satisfy, self.parameter["A"], self.parameter["B"]) :
                stock += Ai
                if sold :
                    if stock < Bi :
                        return self.rewards["invalid_solution"]
                    stock -= Bi
            
            gold, answer = self.parameter["gold_answer"], len(processed_result)
            assert answer <= gold, "The answer should not exceed the gold answer"
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == answer)
            else :
                raise ValueError("Invalid rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]