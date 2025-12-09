import random
from typing import Optional
from Gym.environment import VerifiableEnvironment


class SegmentTreeSortingCounting_Environment(VerifiableEnvironment) : # Source : https://www.luogu.com.cn/problem/P3322
    prompt_template = \
r"""You are given a permutation of integers from 1 to 2^{N} (A[1], A[2], ..., A[2^{N}]). The array is: {A}

There are {N} types of operations. You may apply **each type at most once**, and you may choose to apply them in any order. The i-th type of operation (1 ≤ i ≤ {N}) is defined as follows:
- Divide the array into 2^({N} - i + 1) segments, each of length 2^(i - 1). (Each element belongs to exactly one segment.)
- You may swap **any two segments** (freely chosen by you).

Please count the number of **distinct sequences of operations** that can sort the array into increasing order. Two sequences are considered different if:
- They have different lengths, OR
- They perform **different operations at any same position** in the sequence (i.e., the type or the pair of segments swapped differs at that step)."""

    def __init__(self,
                 wrong_format : float = -1.0, rewarding_strategy : str = "(min/max)^beta", rewarding_weight : float = 1.0, rewarding_beta : float = 10.0,
                 **kwargs) :
        """
        Initialize the SegmentTreeSortingCounting problem.
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

        A = self.parameter["A"] = list(range(1, 2 ** N + 1))

        operation_types = random.sample(range(1, N + 1), random.randint(1, N))
        for operation_type in operation_types :
            seg_num, seg_size = 2 ** (N - operation_type + 1), 2 ** (operation_type - 1)
            i, j = random.sample(range(seg_num), 2)
            i_start, j_start = i * seg_size, j * seg_size
            for k in range(seg_size) :
                A[i_start + k], A[j_start + k] = A[j_start + k], A[i_start + k]
        

        # Precompute factorials up to N (maximum 12)
        po = [1] * (N + 1)
        for i in range(1, N + 1):
            po[i] = po[i - 1] * i

        ans = 0  # will hold the final count

        # Check function: for operation type k (1-based), verify segments are "good"
        def check(k):
            seg_size = 1 << k
            half = 1 << (k - 1)
            # number of segments: 2^(N-k)
            cnt = 1 << (N - k)
            for i in range(cnt):
                start = i * seg_size
                # Compare start of segment and middle of segment
                if A[start] + half != A[start + half]:
                    return False
            return True

        # Swap two segments of length 'length', starting at indices i and j (0-based)
        def swap(i, j, length):
            for m in range(length):
                A[i + m], A[j + m] = A[j + m], A[i + m]

        # Depth-first search through operation choices
        def dfs(now, num):
            nonlocal ans
            # If we've applied an operation type and the current configuration fails the check, prune
            if now > 0 and not check(now):
                return
            # If we've considered all operations, add factorial count
            if now == N:
                ans += po[num]
                return

            # Option 1: skip operation type now+1
            dfs(now + 1, num)

            # Option 2: apply an operation of this type by swapping two segments
            seg_size = 1 << now
            total_segments = 1 << (N - now)
            tmp = []
            # Identify mismatched pairs of adjacent segments
            for i in range(1, total_segments, 2):  # i = 1, 3, 5, ... (1-based segment index)
                # Convert to 0-based start indices
                s1 = (i - 1) * seg_size
                s2 = i * seg_size
                if A[s2] != A[s1] + seg_size:
                    tmp.append(i)
                    tmp.append(i + 1)
                    if len(tmp) > 4:
                        return
            if not tmp:
                return
            # Try swapping any two segments among the identified ones
            for p in range(len(tmp)):
                for q in range(p + 1, len(tmp)):
                    i_seg = tmp[p] - 1
                    j_seg = tmp[q] - 1
                    i_start = i_seg * seg_size
                    j_start = j_seg * seg_size
                    swap(i_start, j_start, seg_size)
                    dfs(now + 1, num + 1)
                    swap(i_start, j_start, seg_size)

        # Run DFS from operation type 0 with 0 operations used
        # Note: 'now' from 0 to N, mapping to operation types 1..N
        # dfs(0,0) considers operation type 1 at now=0, so check uses now>0 means skip initial

        dfs(0, 0)
        assert ans > 0
        self.parameter["reference_answer"] = ans
    

    def _prompt_generate(self) -> str :
        return self.prompt_template.format(
            N = self.parameter["N"],
            A = " ".join("A[{}]={}".format(i + 1, Ai) for i, Ai in enumerate(self.parameter["A"])),
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
            if processed_result < 0 :
                return self.rewards["wrong_format"]

            if self.rewards["rewarding_strategy"] == "(min/max)^beta" :
                a, b = self.parameter["reference_answer"], processed_result
                return self.rewards["rewarding_weight"] * (((min(a, b) / max(a, b))) ** self.rewards["rewarding_beta"])
            elif self.rewards["rewarding_strategy"] == "gold=answer" :
                return self.rewards["rewarding_weight"] * (processed_result == self.parameter["reference_answer"])
            else :
                raise NotImplementedError("Unknown rewarding strategy: {}".format(self.rewards["rewarding_strategy"]))
        else :
            return self.rewards["wrong_format"]