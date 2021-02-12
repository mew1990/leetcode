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
