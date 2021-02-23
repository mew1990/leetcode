# -*- coding: utf-8 -*-
# @Author  : mew

# 这里是记录leetcode的每日1题，2月1日开始
from typing import *
from collections import *
import bisect


class Feb_2021:
    """
    主题：双指针 2月1日~2月6日
    主题：栈和队列 2月8日~
    """

    def fairCandySwap(self, A: List[int], B: List[int]) -> List[int]:
        """ p888 哈希 双指针
        """
        a = sum(A)
        b = sum(B)
        diff = (a-b)//2
        set_A = set(A)
        for i in B:
            if i+diff in set_A:
                return [i+diff, i]

    def characterReplacement(self, s: str, k: int) -> int:
        """ p424 滑动窗口/双指针
        """

        def characterReplacement_my():
            dd = {}
            slow = 0
            res = 0
            for i in range(len(s)):
                # 更新窗口
                dd[s[i]] = dd.get(s[i], 0)+1
                while (i-slow+1)-max(dd.values()) > k:
                    dd[s[slow]] -= 1
                    slow += 1
                # 判断条件
                res = max(res, i-slow)
            return res
            # 测试时间 232ms

        def characterReplacement_faster():
            slow, max_values = 0, 0
            dd = defaultdict(int)
            for i in range(len(s)):
                dd[s[i]] += 1
                max_values = max(max_values, dd[s[i]])  # 最大值只能从s[i]更新
                if (i-slow+1)-max_values > k:  # 最多需判断一次
                    dd[s[slow]] -= 1
                    slow += 1
            return min(max_values+k, len(s))  # 或 i-slow+1
            # 测试时间96ms

        return characterReplacement_faster()

    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        """ p480 滑动窗口
        """

        def medianSlidingWindow_dummy():
            def medium(lst):
                return (lst[k//2] if k%2 else (lst[k//2]+lst[(k-1)//2])/2)

            return [medium(list(sorted(nums[i-k:i]))) for i in range(k, len(nums)+1)]
            # AC 耗时5400+ms  效率=排序(N-k)*klogk，最坏N*NlogN

        def medianSlidingWindow_faster():
            tmp = list(sorted(nums[:k]))
            if k%2:
                medium = lambda lst:lst[k>>1]
            else:
                medium = lambda lst:(lst[k>>1]+lst[(k-1)>>1])/2
            res = [medium(tmp)]
            for i in range(k, len(nums)):
                tmp.pop(bisect.bisect_left(tmp, nums[i-k]))  # 用.remove(nums[i-k])差挺多？
                tmp.insert(bisect.bisect_right(tmp, nums[i]), nums[i])
                res.append(medium(tmp))
            return res
            # AC 耗时80+ms  效率=klogk+(N-k)*(k+logk)，最坏 N*logN
            # ... pop insert 引起的list结构维护实际耗时，比想象中O(N)乐观得多

        return medianSlidingWindow_faster()

    def findMaxAverage(self, nums: List[int], k: int) -> float:
        """ p643 滑动窗口
        """
        res = cur = sum(nums[:k])
        for i in range(k, len(nums)):
            cur += nums[i]-nums[i-k]
            if cur > res: res = cur
        return res/k

    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        """ p1208
        """
        res, cost = 0, 0
        for i in range(len(s)):
            cost += abs(ord(s[i])-ord(t[i]))
            if cost > maxCost:
                cost -= abs(ord(s[i-res])-ord(t[i-res]))
            else:
                res += 1
        return res

    def maxScore(self, cardPoints: List[int], k: int) -> int:
        """ p1423
        """
        res, left, right = sum(cardPoints[:k]), k-1, len(cardPoints)-1
        res_max = res
        while left >= 0:
            res += cardPoints[right]-cardPoints[left]
            if res > res_max: res_max = res
            left, right = left-1, right-1
        return res_max

    def checkPossibility(self, nums: List[int]) -> bool:
        """ p665
        """
        idx = [(i, i+1) for i in range(len(nums)-1) if nums[i] > nums[i+1]]
        if len(idx) > 1: return False
        if len(idx) == 0: return True

        i, j = idx[0]
        if i == 0 or j == len(nums)-1: return True
        else:
            if nums[i-1] <= nums[j] or nums[i] <= nums[j+1]: return True
            else: return False

    def maxTurbulenceSize(self, arr: List[int]) -> int:
        """ p978 medium

        状态转换，有点DP的意思
        """
        up, down = 1, 1
        res = 1
        for i in range(len(arr)-1):
            if arr[i] > arr[i+1]: up, down = 1, up+1
            elif arr[i] < arr[i+1]: up, down = down+1, 1
            else: up, down = 1, 1
            res = max(res, up, down)
        return res

    def subarraysWithKDistinct(self, A: List[int], K: int) -> int:
        """ p992 hard 滑动窗口

        解题：子数组不同数字刚好为K个，转化为最大（或最小）为K个的问题
            faster 的写法值得看看
        """

        def subarraysWithKDistinct_faster():  # 通过案例，有意思
            counter = {}
            res = i = diffNum = leftForward = 0

            for j in range(len(A)):
                if A[j] not in counter:
                    diffNum += 1
                    counter[A[j]] = 1
                else:
                    counter[A[j]] += 1

                if diffNum == K:
                    if A[i-1] != A[j] and i > 0:  # leftForward置零的条件 good
                        leftForward = 0
                    while diffNum == K:
                        if counter[A[i]] == 1:
                            diffNum -= 1
                            del counter[A[i]]
                        else:
                            counter[A[i]] -= 1
                        i += 1
                        leftForward += 1
                res += leftForward

            return res

        def subarraysNotLessThanKDistinct(A, K):
            dd = defaultdict(int)
            tot, left, right = 0, 0, 0
            res = 0
            while right < len(A):
                if dd[A[right]] == 0: tot += 1
                dd[A[right]] += 1
                right += 1
                while tot > K:  # 这里是 >，当超出K个，left滑动
                    dd[A[left]] -= 1
                    if dd[A[left]] == 0:
                        tot -= 1
                    left += 1
                # 计数，以right（包含）为右端，个数不超过K的子数组数量
                # 所以每次right右移一次，计数一次
                res += right-left+1  # 这里+1表示包括长度为0的子数组，但是K>=1，所以对题目结果没有影响。
            return res

        return subarraysNotLessThanKDistinct(A, K)-subarraysNotLessThanKDistinct(A, K-1)

    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        """ p239 hard 队列
        算法+++

        题解：随着窗口滑动，队列记录长度为k的窗口内的单调递减序列

        输入：nums = [1,3,-1,-3,5,3,6,7], k = 3
        输出：[3,3,5,5,6,7]
        解释：
        滑动窗口的位置                最大值
        ---------------               -----
        [1  3  -1] -3  5  3  6  7       3
         1 [3  -1  -3] 5  3  6  7       3
        """
        if k == 1: return nums
        from collections import deque
        window = deque([k-1])
        for i in range(k-2, -1, -1):
            if nums[i] >= nums[window[0]]:
                window.appendleft(i)
        res = [nums[window[0]]]
        for i in range(k, len(nums)):
            while window and nums[window[-1]] < nums[i]:  # 这里window可能变空
                window.pop()
            window.append(i)
            if window[-1]-window[0] >= k:
                window.popleft()
            res.append(nums[window[0]])
        return res

    def minKBitFlips(self, A: List[int], K: int) -> int:
        """ p995 hard 滑动窗口
        算法++

        输入：A = [0,0,0,1,0,1,1,0], K = 3
        输出：3
        解释：
        翻转 A[0],A[1],A[2]: A变成 [1,1,1,1,0,1,1,0]
        翻转 A[4],A[5],A[6]: A变成 [1,1,1,1,1,0,0,0]
        翻转 A[5],A[6],A[7]: A变成 [1,1,1,1,1,1,1,1]

        题解：差分数组
        """
        n = len(A)
        diff = [0]*(n+1)
        cur = 0
        for i in range(n):
            cur += diff[i]
            if A[i]^(cur&1) == 0:  # 需要翻转 [0^2==2, 0^(2&1)==0]
                if i > n-K: return -1
                cur += 1
                diff[i+K] = -1  # 差分数组，i+K位置翻转次数减一
        return diff.count(-1)

    def longestOnes(self, A: List[int], K: int) -> int:
        """ p1004 medium 滑动窗口
        算法+++

        题解：滑动窗口，left搭配if使用，窗口不会变小。如果是while，更符合逻辑，但是要配合max使用
        """
        left, right, cnt = 0, 0, 0
        for right in range(len(A)):
            if A[right] == 0:
                cnt += 1
            if cnt > K:  # 如果是while，要记录max。
                if A[left] == 0:
                    cnt -= 1
                left += 1
        return right-left+1

    def longestSubarray(self, nums: List[int], limit: int) -> int:
        """  p1438 medium 滑动窗口
        算法++

        题解：【绝对差不超过限制的最长连续子数组】
        """
        a = deque()
        b = deque()
        left = 0
        for right in range(len(nums)):
            while a and a[-1] > nums[right]:
                a.pop()
            a.append(nums[right])
            while b and b[-1] < nums[right]:
                b.pop()
            b.append(nums[right])
            if b[0]-a[0] > limit:
                if nums[left] == a[0]:
                    a.popleft()
                if nums[left] == b[0]:
                    b.popleft()
                left += 1
        return right-left+1

    def maxSatisfied(self, customers: List[int], grumpy: List[int], X: int) -> int:
        """ p1052 medium 滑动窗口
        算法++

        题解：【爱生气的书店老板】，滑动固定窗口，计算差分
        """
        n = len(customers)
        ans = sum(customers[i] for i in range(n) if grumpy[i] == 0)
        cur = sum(customers[i] for i in range(X) if grumpy[i] == 1)
        cur_max = cur
        for right in range(X, n):
            cur += customers[right] if grumpy[right] == 1 else 0
            cur -= customers[right-X] if grumpy[right-X] == 1 else 0
            cur_max = max(cur, cur_max)
        return ans+cur_max
