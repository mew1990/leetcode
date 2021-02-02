# -*- coding: utf-8 -*-
# @Author  : mew

# 这里是记录leetcode的每日1题，2月1日开始
from typing import *
from collections import *


class TwoPointer:
    """
    2月1日~
    主题：双指针
    """

    class Solution:
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
                return min(max_values+k, len(s))  # 等价为相同的，值得思考
                # 测试时间96ms

            return characterReplacement_faster()
