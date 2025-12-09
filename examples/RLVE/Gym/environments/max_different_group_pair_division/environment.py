import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class MaxDifferentGroupPairDivision_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3648
    prompt_template = \
r"""You are given an array A of {N} integers: {A}

Initially, the entire array is one single block. Let S = 0. You need to perform the following operation exactly {K} times:
- Choose a position `i` such that A[i] and A[i + 1] are still in the same block.
- Split the block into two parts: the first ends at A[i], the second starts at A[i + 1].
- Let `sum1` and `sum2` be the sums of the two blocks. Then, update S += sum1 × sum2.

After {K} operations, you will have {K} + 1 blocks. Try your best to **maximize** the final value of S.

**Output Format:** A single line containing {K} integers — the positions `i` you chose in order, separated by spaces."""

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "(answer/gold)^beta", rewarding_weight : float = +1.0, rewarding_beta : float = 5.0,
                 **kwargs) :
        """
        Initialize the MaxDifferentGroupPairDivision_Environment instance.
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
        assert N >= 4, "N should be greater than or equal to 4"

        K = self.parameter["K"] = random.randint(2, N - 2)
        A = self.parameter["A"] = [random.randint(0, N) for _ in range(N)]


        B = K + 1  # number of blocks after K splits

        # Read sequence and build prefix sums
        prefix_sum = [0] * (N + 1)
        for i in range(1, N + 1):
            prefix_sum[i] = prefix_sum[i - 1] + A[i - 1]
        sum_N = prefix_sum[N]

        # pre[j][i] will store the split position for the j-th block ending at i
        # Use array('I') for memory efficiency (stores unsigned 32-bit ints)
        pre = [[0] * (N + 1) for _ in range(B + 1)]

        # We'll keep only two rows of DP at a time
        prev_f = [0] * (N + 1)
        cur_f = [0] * (N + 1)

        # DP over number of blocks j = 1..B
        for j in range(1, B + 1):
            # Convex-hull trick: maintain deque of candidate split-points
            qx = [0] * (N + 1)  # x = prefix_sum[p]
            qy = [0] * (N + 1)  # y = prev_f[p]
            qp = [0] * (N + 1)  # p = index

            head = tail = 0
            qx[0] = 0
            qy[0] = prev_f[0]
            qp[0] = 0

            for i in range(1, N + 1):
                psi = prefix_sum[i]
                S_rem = sum_N - psi

                # Pop from front while next candidate is better
                while head < tail and (qy[head + 1] - qy[head]) >= S_rem * (qx[head + 1] - qx[head]):
                    head += 1

                # Use best candidate at front
                p = qp[head]
                pre[j][i] = p
                cur_f[i] = qy[head] + S_rem * (psi - qx[head])

                # Prepare new candidate from this position
                new_x = psi
                new_y = prev_f[i]

                # Pop from back while new candidate makes the last one obsolete
                while head < tail and (qy[tail] - qy[tail - 1]) * (new_x - qx[tail]) <= (new_y - qy[tail]) * (qx[tail] - qx[tail - 1]):
                    tail -= 1

                tail += 1
                qx[tail] = new_x
                qy[tail] = new_y
                qp[tail] = i

            # Move current row to previous for next iteration
            prev_f, cur_f = cur_f, [0] * (N + 1)

        # The answer is DP[B][N]
        self.parameter["gold_answer"] = prev_f[N]

        # Reconstruct split positions
        path = [0] * (B + 1)
        path[B] = N
        for j in range(B, 0, -1):
            path[j - 1] = pre[j][path[j]]
        # We only need the K split points: path[1], ..., path[B-1]
        splits = path[1:B]
        self.parameter["reference_answer"] = " ".join(map(str, [split - 1 for split in splits]))  # Convert to 0-based index
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = " ".join("A[{}]={}".format(i, Ai) for i, Ai in enumerate(self.parameter["A"])),
            K = self.parameter["K"],
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

            if len(processed_result) != self.parameter["K"] :
                return self.rewards["invalid_solution"]
            
            answer = 0
            block_ID, block_numbers = 0, [0] * self.parameter["N"]
            for i in processed_result :
                if not (0 <= i < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                if not (0 <= (i + 1) < self.parameter["N"]) :
                    return self.rewards["invalid_solution"]
                if block_numbers[i] != block_numbers[i + 1] :
                    return self.rewards["invalid_solution"]
                
                sum1, j = 0, i
                while j >= 0 :
                    if block_numbers[j] != block_numbers[i] :
                        break
                    sum1 += self.parameter["A"][j]
                    j -= 1
                
                block_ID += 1
                sum2, j = 0, i + 1
                while j < self.parameter["N"] :
                    if block_numbers[j] != block_numbers[i] :
                        break
                    sum2 += self.parameter["A"][j]
                    block_numbers[j] = block_ID
                    j += 1
                
                answer += sum1 * sum2
            
            gold = self.parameter["gold_answer"]
            assert answer <= gold, "answer should be less than or equal to gold"
            
            if self.rewards["rewarding_strategy"] == "(answer/gold)^beta" :
                if gold == 0 :
                    assert answer == 0, "If gold is 0, answer should also be 0"
                    return self.rewards["rewarding_weight"] * 1.0
                return self.rewards["rewarding_weight"] * ((answer / gold) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (answer == gold)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]