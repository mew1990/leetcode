# -*- coding: utf-8 -*-
# @Author  : mew

# 这里是记录leetcode的每日1题，2月1日开始
from typing import *

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
