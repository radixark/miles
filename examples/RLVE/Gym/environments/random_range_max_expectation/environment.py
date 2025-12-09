import random
from typing import Optional, List
from Gym.environment import VerifiableEnvironment


class RandomRangeMaxExpectation_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3352
    prompt_template = \
r"""You are given an array of {N} integers: {array}

You will perform {Q} operations in order. In each operation, you uniformly select a subarray (a contiguous segment of the array) at random from all {N} × ({N} + 1) / 2 possible subarrays. Then, all elements in that subarray are changed to the **maximum** value within it.

Please compute the expected value of each position in the array after all {Q} operations. Since the expected value is a rational number with denominator ({N} × ({N} + 1) / 2)^{Q}, output the **numerator** (i.e., the expected value multiplied by ({N} × ({N} + 1) / 2)^{Q}), modulo {MOD}.

**Output Format:** A single line containing {N} integers — the scaled expected values (modulo {MOD}) for each position, separated by spaces."""
    MOD = 10000

    def __init__(self,
                 wrong_format : float = -1.0, invalid_solution : float = -0.5, rewarding_strategy : str = "mean([gold=answer])^beta", rewarding_beta : float = 3.0, rewarding_weight : float = +1.0,
                 **kwargs) :
        """
        Initialize the RandomRangeMaxExpectation_Environment instance.
        """
        super().__init__(**kwargs)

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

        Q = self.parameter["Q"] = random.randint(1, N)

        A = self.parameter["array"] = [random.randint(0, N) for _ in range(N)]


        def calc(x):
            return x * (x + 1) // 2 % self.MOD

        # sentinel INF just above any value in A
        INF = max(A) + 1
        
        # prepare DP tables
        # f[0] for previous round, f[1] for current
        f = [
            [ [0] * N for _ in range(N) ],
            [ [0] * N for _ in range(N) ]
        ]
        # g[l][r] is the weight factor
        g = [ [0] * N for _ in range(N) ]
        
        # precompute g
        for l in range(N):
            for r in range(l, N):
                length = r - l + 1
                left  = l
                right = N - 1 - r
                g[l][r] = (calc(length) + calc(left) + calc(right)) % self.MOD
        
        # base case f[0]
        for l in range(N):
            maxx = 0
            for r in range(l, N):
                # update max in A[l..r]
                if A[r] > maxx:
                    maxx = A[r]
                # case: whole array
                if l == 0 and r == N - 1:
                    f[0][l][r] = maxx % self.MOD
                else:
                    left_val  = INF if l == 0 else A[l - 1]
                    right_val = INF if r == N - 1 else A[r + 1]
                    # only intervals where both neighbors are strictly larger
                    if left_val > maxx and right_val > maxx:
                        f[0][l][r] = (maxx - min(left_val, right_val)) % self.MOD
        
        # perform Q random-interval operations in expectation
        for i in range(1, Q + 1):
            now = i & 1
            pre = 1 - now
            
            # prefix sums s1 and suffix sums s2
            s1 = [ [0]*N for _ in range(N) ]
            s2 = [ [0]*N for _ in range(N) ]
            
            # build s1: for each r, accumulate over l=0..r of f[pre][l][r] * l
            for r in range(N):
                acc = 0
                for l in range(0, r + 1):
                    acc = (acc + f[pre][l][r] * l) % self.MOD
                    s1[l][r] = acc
            
            # build s2: for each l, accumulate over r=N-1..l of f[pre][l][r] * (N-1-r)
            for l in range(N):
                acc = 0
                for r in range(N - 1, l - 1, -1):
                    acc = (acc + f[pre][l][r] * (N - 1 - r)) % self.MOD
                    s2[l][r] = acc
            
            # update f[now] using precomputed g, s1, s2
            for l in range(N):
                for r in range(l, N):
                    left_contrib  = s1[l - 1][r] if l - 1 >= 0 else 0
                    right_contrib = s2[l][r + 1] if r + 1 < N else 0
                    f[now][l][r] = (
                        f[pre][l][r] * g[l][r]
                        + left_contrib
                        + right_contrib
                    ) % self.MOD
        
        # collect and print final answers
        result = []
        final_dp = f[Q & 1]
        for i in range(N):
            ans = 0
            for l in range(0, i + 1):
                for r in range(i, N):
                    ans = (ans + final_dp[l][r]) % self.MOD
            result.append(ans)
        
        assert len(result) == N
        self.parameter["gold_answer"] = result
        self.parameter["reference_answer"] = " ".join(map(str, result))
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            array = " ".join(map(str, self.parameter["array"])),
            Q = self.parameter["Q"],
            MOD = self.MOD,
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

            if len(processed_result) != self.parameter["N"] :
                return self.rewards["invalid_solution"]
            
            if self.rewards["rewarding_strategy"] == "mean([gold=answer])^beta" :
                return self.rewards["rewarding_weight"] * ((sum(int(a == b) for a, b in zip(self.parameter["gold_answer"], processed_result)) / self.parameter["N"]) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (self.parameter["gold_answer"] == processed_result)
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]