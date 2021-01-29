# -*- coding: utf-8 -*-
# @Author  : mew

from .my_defs import *
from typing import *


class Solution:
    """ 学习编程技巧 """

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


