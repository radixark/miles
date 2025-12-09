import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class CampfireParty_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P1053
    prompt_template = \
r"""There are {N} students labeled from `0` to `{N_minus_1}`. At the beginning, they are sitting in a **circle** in the order: `0, 1, ..., {N_minus_1}`. Each student has **two specific friends** they want to sit next to. Your task is to rearrange the students around the circle so that **every student is adjacent to both of their desired neighbors**.
{desired_neighbors}

To achieve this, you may perform a series of operations. Each operation is represented as a tuple `(b_1, b_2, ..., b_m)`, where:
- The student `b_1` moves to the position of `b_2`, `b_2` moves to the position of `b_3`, ..., and `b_m` moves to the position of `b_1`.
- The cost of an operation is equal to the number of students involved (`m`).
- No student may appear more than once in a single operation.

Your goal is to achieve the desired circular arrangement using the **minimum total cost** across all operations.

**Output Format:**
Your final answer should contain K lines, where K is the number of operations you perform. The K lines should each describe one operation: a space-separated list of the students involved in that operation, in the order `(b_1, b_2, ..., b_m)`.
Example (do **NOT** include the backticks or quotes):
```
0 1 2
1 2
2 3
```
This means:
- There are 3 operations,
- The first operation rotates students 0 → 1 → 2 → 0,
- The second rotates (swaps) students 1 ↔ 2,
- The third rotates (swaps) students 2 ↔ 3,
- And the total cost is `3 + 2 + 2 = 7`.
"""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, unsuccessful_solution : float = -0.2, rewarding_strategy : str = "(gold/answer)^beta", rewarding_beta : float = +3.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the CampfireParty_Environment instance.
        """
        super().__init__(**kwargs)
    
        self.rewards = {
            "wrong_format" : wrong_format,
            "invalid_solution" : invalid_solution,
            "unsuccessful_solution" : unsuccessful_solution,
            "rewarding_strategy" : rewarding_strategy,
            "rewarding_beta" : rewarding_beta,
            "rewarding_weight" : rewarding_weight,
        }
    
    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"
    
        permutation = list(range(N))
        random.shuffle(permutation)

        adjacent = self.parameter["desired_neighbors"] = [None] * N
        for i, student in enumerate(permutation) :
            a, b = permutation[(i - 1 + N) % N], permutation[(i + 1) % N]
            adjacent[student] = (a, b)
        
        for student, (a, b) in enumerate(adjacent) :
            assert student in adjacent[a], f"Student {student} is not adjacent to {a}"
            assert student in adjacent[b], f"Student {student} is not adjacent to {b}"
        

        permutation = []
        x, parent = 0, -1
        while True :
            if x == 0 and parent != -1 :
                break
            permutation.append(x)
            for y in adjacent[x] :
                assert y is not None
                if y == parent :
                    continue
                x, parent = y, x
                break

        assert len(permutation) == N, "Permutation length should be equal to N"

        def solve() :
            target = permutation.copy()
            positions = [None] * N
            for i, p in enumerate(target) :
                positions[p] = i
            
            counting = {}
            for i, position in enumerate(positions) :
                diff = (position - i + N) % N
                counting[diff] = counting.get(diff, 0) + 1
            optimal_diff = max(counting, key = lambda x : counting[x])

            start = [(i - optimal_diff) % N for i in range(N)]
            for i, p in enumerate(start) :
                positions[p] = i
            
            target_positions = [None] * N
            for i, p in enumerate(target) :
                target_positions[p] = i
            
            cycles = []
            
            point = [None] * N
            for s, position, target_position in zip(range(N), positions, target_positions) :
                if position == target_position :
                    continue
                point[s] = start[target_position]
            
            visited = [False] * N
            for s in range(N) :
                if visited[s] :
                    continue
                if point[s] is None :
                    continue
                cycle = []
                x = s
                while True :
                    cycle.append(x)
                    visited[x] = True
                    x = point[x]
                    if x == s :
                        break
                cycles.append(cycle)
            
            def operation(cycle) :
                assert len(cycle) >= 2
                assert len(cycle) == len(set(cycle))
                new_positions = [positions[i] for i in cycle]
                new_positions = new_positions[1 :] + [new_positions[0]]
                for i, new_position in zip(cycle, new_positions) :
                    start[new_position] = i
                    positions[i] = new_position
                return len(cycle)

            cost = sum(operation(cycle) for cycle in cycles)
            
            for s, t in zip(start, target) :
                assert s == t
            for i, p in enumerate(start) :
                assert positions[p] == i
            
            return cost, cycles

        cost, cycles = solve()
        permutation.reverse()
        candidate_cost, candidate_cycles = solve()
        if cost > candidate_cost :
            cost, cycles = candidate_cost, candidate_cycles
        
        self.parameter["reference_answer"] = "\n".join(" ".join(map(str, cycle)) for cycle in cycles)
        self.parameter["reference_answer_cost"] = cost
        assert cost == sum(len(cycle) for cycle in cycles)
    
    def _prompt_generate(self) -> str :
        N = self.parameter["N"]
        return self.prompt_template.format(
            N = N,
            N_minus_1 = N - 1,
            desired_neighbors = "\n".join("Student {} prefers neighbors: {} and {}".format(student, a, b) for student, (a, b) in enumerate(self.parameter["desired_neighbors"])),
        )


    def _process(self, answer : Optional[str]) -> Optional[List] :
        if answer is not None :
            answer = answer.strip()
            try :
                cycles = []
                for line in answer.splitlines() :
                    line = line.strip()
                    if line :
                        cycles.append(list(map(int, line.split())))
                return cycles
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            assert isinstance(processed_result, list), "processed_result should be a list"

            permutation = list(range(self.parameter["N"]))
            positions = list(range(self.parameter["N"]))
            for cycle in processed_result :
                for student in cycle :
                    if not (0 <= student < self.parameter["N"]) :
                        return self.rewards["invalid_solution"]
                if len(cycle) == 1 :
                    continue
                if len(cycle) != len(set(cycle)) :
                    return self.rewards["invalid_solution"]
                
                new_positions = [positions[i] for i in cycle]
                new_positions = new_positions[1 :] + [new_positions[0]]
                for i, new_position in zip(cycle, new_positions) :
                    permutation[new_position] = i
                    positions[i] = new_position
            for i, p in enumerate(permutation) :
                assert positions[p] == i
            
            for student, (a, b) in enumerate(self.parameter["desired_neighbors"]) :
                p, pa, pb = positions[student], positions[a], positions[b]
                if pa not in ((p - 1 + self.parameter["N"]) % self.parameter["N"], (p + 1) % self.parameter["N"]) :
                    return self.rewards["unsuccessful_solution"]
                if pb not in ((p - 1 + self.parameter["N"]) % self.parameter["N"], (p + 1) % self.parameter["N"]) :
                    return self.rewards["unsuccessful_solution"]
            
            cost = sum(len(cycle) for cycle in processed_result)
            gold = self.parameter["reference_answer_cost"]
            assert gold <= cost, "cost should be greater than or equal to reference_answer_cost"

            if self.rewards["rewarding_strategy"] == "(gold/answer)^beta" :
                if cost == 0 :
                    return self.rewards["rewarding_weight"]
                return self.rewards["rewarding_weight"] * ((gold / cost) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (gold == cost)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]