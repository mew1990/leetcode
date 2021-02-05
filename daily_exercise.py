# -*- coding: utf-8 -*-
# @Author  : mew

# 这里是记录leetcode的每日1题，2月1日开始
from typing import *
from collections import *
import bisect


class TwoPointer:
    """
    2月1日~
    主题：双指针
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
