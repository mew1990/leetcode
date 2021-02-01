# -*- coding: utf-8 -*-
# @Author  : mew

from .my_defs import *
from typing import *


class Solution:
    """ 学习编程技巧 """

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        # p19 medium 链表 指针
        a = b = head
        for i in range(n):
            a = a.next
        if a is None: return head.next  # [特殊情况处理]在删掉第一个元素条件下成立
        while a.next:  # 这里是a.next is not None 不是 a is not None
            a = a.next
            b = b.next
        b.next = b.next.next
        return head

    def isValid(self, s: str) -> bool:
        # p20 easy 栈 字符串 匹配
        dic = {'(':')', '[':']', '{':'}'}
        stack = ['?']
        for c in s:
            if c in dic: stack.append(c)
            elif dic[stack.pop()] != c: return False  # 该写法，只能用一次pop
        return len(stack) == 1

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # p21 easy 递归 链表
        if l1 and l2:
            if l1.val > l2.val: l1, l2 = l2, l1
            l1.next = self.mergeTwoLists(l1.next, l2)
        return l1 or l2  # 该写法考虑了 l1 或 l2 为空的情况

    def generateParenthesis(self, n: int) -> List[str]:
        # p22 medium 迭代
        # ... 状态迭代，(str_, int_left_pare, int_right_pare) 记录生成状态
        pass

    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # p23 hard 分治 递归 链表
        if not lists: return None
        if len(lists) == 1: return lists[0]
        mid = len(lists)>>1
        return self.mergeTwoLists(self.mergeKLists(lists[:mid]),
                                  self.mergeKLists(lists[mid:]))  # 分治，引用了p21

    def swapPairs(self, head: ListNode) -> ListNode:
        # p24 easy 递归 链表
        if head is None or head.next is None: return head
        h1 = head.next
        h2 = h1.next  # 记录个体，重组，思路要清晰
        h1.next = head
        head.next = self.swapPairs(h2)
        return h1

    def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
        # p25 hard 递归 链表
        # ... k=2时就是p24，用list存储k长度的节点，然后分情况处理
        pass

    def removeDuplicates(self, nums: List[int]) -> int:
        # p26 easy
        # ... 已排序数组去重，idx，i记录两个数组指针
        pass

    def removeElement(self, nums: List[int], val: int) -> int:
        # p27 easy
        # ... 同p26
        pass

    def strStr(self, haystack: str, needle: str) -> int:
        # p28 easy
        # ... 内置函数时间和空间效率都偏低，虽然代码量少
        # ... 但是python的意义是什么？！
        return haystack.find(needle)

    def divide(self, dividend: int, divisor: int) -> int:
        # p29 medium 位运算思想
        # ... 除法运算，用倍增思想定边界
        pass

    def findSubstring(self, s: str, words: List[str]) -> List[int]:
        # p30 hard 滑动窗口 双指针
        # ... 不同初始条件下的窗口滑动

        def simple_method(s, words):
            from collections import Counter
            c_words = Counter(words)
            n, m = len(words[0]), len(words)

            res = []
            for i in range(len(s)-n*m+1):
                if Counter(s[i+n*j:i+n*(j+1)] for j in range(m)) == c_words:
                    res.append(i)
            return res
            # 测试时间 900+ms

        def moving_window_method(s, words):
            from collections import Counter, defaultdict
            c_words = Counter(words)
            n, len_w = len(words), len(words[0])
            res = []
            for i in range(len_w):
                # 初始化
                fast, slow = n*len_w+i, i
                dd = defaultdict(int)
                for j in range(slow, fast, len_w):
                    dd[s[j:j+len_w]] += 1
                # 窗口滑动
                for _ in range(fast, len(s)+1, len_w):  # +1 使窗口滑出
                    # 判断条件
                    if dd == c_words: res.append(slow)
                    # 更新窗口
                    tmp1, tmp2 = s[slow:slow+len_w], s[fast:fast+len_w]
                    slow, fast = slow+len_w, fast+len_w
                    if dd[tmp1] == 1: dd.pop(tmp1)
                    else: dd[tmp1] -= 1
                    dd[tmp2] += 1
            return res
        # 测试时间 92ms，字典判断可能可以更优化，这里性价比已经比较高。

    def nextPermutation(self, nums: List[int]) -> None:
        # p31 medium 排序
        # ... 冒泡排序的，寻找下一个正序对。
        #     python自带itertools.permutations功能，值得体会permutations的算法

        def nextPermutation_BubbleSort():
            # 冒泡排序（从大到小排列， 目标寻找第一对正序对）
            if len(nums) < 2: return
            for i in range(len(nums)-2, -1, -1):
                if nums[i] >= nums[-1]:  # 如果不存在正序对，
                    nums[i:] = nums[i+1:]+[nums[i]]  # 冒泡到底
                else:  # 如果存在正序对，
                    for j in range(i+1, len(nums)):
                        if nums[i] < nums[j]:  # 查找第一个正序对，
                            nums[i], nums[j] = nums[j], nums[i]  # 并交换位置
                            return  # 退出

        def nextPermutation_faster():
            if len(nums) < 2: return
            idx = len(nums)-1
            while idx > 0 and nums[idx-1] >= nums[idx]: idx -= 1
            nums[idx:] = reversed(nums[idx:])  # 链表反序，效率和上面差不太多
            if idx > 0:
                for j in range(idx, len(nums)):
                    pass  # 查找第一个正序对，并交换位置，退出。代码同上


