import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class STUWell_Environment(VerifiableEnvironment) :
    prompt_template = \
r"""There is an array X of length {N}. Initially, X is: {X}
You can perform the following operation at most {M} times: pick an arbitrary index i and decrease X[i] by 1 (i.e., X[i] -= 1); at the end, you must ensure that there exists at least one index i such that X[i] = 0.
Try your best to minimize the value of max(|X[i] - X[i + 1]|) over all 0 <= i < {N} - 1 (i.e., the maximum absolute difference between any two adjacent elements in X). Output the minimum possible value of this maximum difference."""

    def __init__(self,
                 weight_multiple : int = 4,
                 wrong_format : float = -1.0, correct_answer : float = +1.0, wrong_answer : float = 0.0,
                 **kwargs) :
        """
        Initialize the STUWell_Environment instance.
        """
        super().__init__(**kwargs)

        self.weight_multiple = weight_multiple
        self.rewards = {
            "wrong_format" : wrong_format,
            "correct_answer" : correct_answer,
            "wrong_answer" : wrong_answer,
        }
    

    def _generate(self) -> None :
        assert "N" in self.parameter, "N is required in parameter"
        N = self.parameter["N"]
        assert N >= 3, "N should be greater than or equal to 3"

        X = self.parameter["X"] = [random.randint(1, N * self.weight_multiple) for _ in range(N)]
        M = self.parameter["M"] = random.randint(min(X), sum(X))


        def check(z):
            """
            Check if it's possible with maximum allowed adjacent slope z (mid in the original code)
            to dig down somewhere to water (height 0) using at most M shovel swings.
            If so, record the position in best_k and return True; else return False.
            """
            # Remaining digging power
            rem = M
            
            # 1) First, smooth the terrain so that |a[i] - a[i+1]| <= z at minimal cost
            #    We work on a copy so as not to overwrite X
            a = X[:]  
            for i in range(1, N):
                # if slope from a[i-1] up to a[i] exceeds z, shave off the excess
                excess = a[i] - (a[i-1] + z)
                if excess > 0:
                    rem -= excess
                    a[i] = a[i-1] + z
            for i in range(N-2, -1, -1):
                excess = a[i] - (a[i+1] + z)
                if excess > 0:
                    rem -= excess
                    a[i] = a[i+1] + z
            
            # If we've already used more than M shovels, fail
            if rem < 0:
                return False
            
            # 2) Build prefix sums so we can query any interval sum in O(1)
            prefix = [0] * N
            prefix[0] = a[0]
            for i in range(1, N):
                prefix[i] = prefix[i-1] + a[i]
            
            # 3) For each candidate digging spot i, we need to compute the cost to
            #    shave the terrain down to the "tent" shape that slopes up at rate z
            #    from height 0 at i.  Outside a certain window [L..R], the original
            #    a[j] is already below the tent, so no digging needed there.
            L = [0] * N
            j = 0
            for i in range(N):
                # advance j until a[j] >= z*(i-j)
                while j < N and z * (i - j) > a[j]:
                    j += 1
                L[i] = j
            
            R = [0] * N
            j = N - 1
            for i in range(N-1, -1, -1):
                # decrease j until a[j] >= z*(j-i)
                while j >= 0 and z * (j - i) > a[j]:
                    j -= 1
                R[i] = j
            
            # 4) Test each position i as the digging spot
            for i in range(N):
                li, ri = L[i], R[i]
                # sum of a[li..ri]
                segment_sum = prefix[ri] - (prefix[li-1] if li > 0 else 0)
                # cost to carve the left half of the tent (from li up to i)
                left_len = i - li
                cost_left = z * left_len * (left_len + 1) // 2
                # cost to carve the right half of the tent (from i up to ri)
                right_len = ri - i
                cost_right = z * right_len * (right_len + 1) // 2
                # total additional digs needed to form the tent
                needed = segment_sum - cost_left - cost_right
                if needed <= rem:
                    return True
            
            return False

        # 5) Binary search on z = the maximum allowed adjacent slope
        lo, hi = 0, max(X)
        best_z = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if check(mid):
                best_z = mid
                hi = mid - 1
            else:
                lo = mid + 1
        self.parameter["reference_answer"] = best_z
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            M = self.parameter["M"],
            X = " ".join("X[{}]={}".format(i, Xi) for i, Xi in enumerate(self.parameter["X"])),
        )


    def _process(self, answer : Optional[str]) -> Optional[int] :
        if answer is not None :
            answer = answer.strip()
            try :
                int_answer = int(answer)
                return int_answer
            except ValueError :
                return None
        else :
            return None
    

    def scorer(self, output : str) -> float :
        processed_result = self.processor(output)
        if processed_result is not None :
            if processed_result == self.parameter["reference_answer"] :
                return self.rewards["correct_answer"]
            else :
                return self.rewards["wrong_answer"]
        else :
            return self.rewards["wrong_format"]