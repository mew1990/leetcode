# -*- coding: utf-8 -*-
# @Author  : mew

from .my_defs import *
from typing import *
from bisect import bisect_left, bisect
import collections


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

    def longestValidParentheses(self, s: str) -> int:
        # p32 hard 栈
        # ... 官方题解还有 动态规划，以及扫描简化版本。这两个版本都等价于栈，
        #     时间都是 O(N)，空间效率不同。
        #     不过如果再考虑 大括号，中括号，就用栈方便些。
        res = 0
        stack = [-1]  # 初始化做标记，表示待匹配
        for i in range(len(s)):
            if s[i] == '(':
                stack.append(i)
            else:
                k = stack.pop()
                if stack:  # 栈非空为合法匹配
                    res = max(res, i-stack[-1])  # 记录匹配的长度
                else:  # 否则为非法匹配
                    stack.append(i)  # 重置标记点
        return res

    def search(self, nums: List[int], target: int) -> int:
        # p33 medium 二分法/分治
        # ... 考虑用二分查找（必满足左边部分>=左边界>右边界>=右边部分）
        #     中间点大于左边界，则划分为[left, mid] 和其他，
        #     中间点小于右边界，则划分为[mid, right]和其他，分治。
        try:
            return nums.index(target)  # try 的特点值得关注
        except:
            return -1

    def searchRange(self, nums: List[int], target: int) -> List[int]:
        # p34 medium 二分
        # ... 二分查找代码阅读 bisect.py，会用 bisect_left 和 bisect(=bisect_right)
        #     根据 bisect.py 说明：即插入target后仍有序的最小index，最大index
        l = bisect_left(nums, target)
        if len(nums) == 0 or l == len(nums) or nums[l] != target:
            return [-1, -1]
        else:
            return [l, bisect(nums, target)-1]

    def searchInsert(self, nums: List[int], target: int) -> int:
        # p35 easy  二分
        pass

    def isValidSudoku(self, board: List[List[str]]) -> bool:
        # p36 medium 数学
        # ... 判断数独合法性，数学题，就简洁美观易懂
        col = [[False]*10 for _ in range(9)]
        row = [[False]*10 for _ in range(9)]
        box = [[False]*10 for _ in range(9)]
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.': continue
                num = int(board[i][j])
                if (row[i][num] or col[j][num] or box[i//3*3+j//3][num]):
                    return False
                row[i][num] = col[j][num] = box[i//3*3+j//3][num] = True
        return True

    def solveSudoku(self, board: List[List[str]]) -> None:
        # p37 hard 数学 DFS
        col = [[False]*10 for _ in range(9)]
        row = [[False]*10 for _ in range(9)]
        box = [[[False]*10 for _ in range(3)] for _ in range(3)]  # 与上一题box不一样

        def dfs(tag):
            if tag == 81: return True  # 注意结束标志
            i, j = tag//9, tag%9
            if board[i][j] != '.': return dfs(tag+1)
            for num in range(1, 10):
                if row[i][num] or col[j][num] or box[i//3][j//3][num]:
                    continue
                row[i][num] = col[j][num] = box[i//3][j//3][num] = True
                board[i][j] = str(num)
                if dfs(tag+1): return True  # python中DFS回溯的方式
                board[i][j] = '.'  # 这条语句的重要性，不可删
                row[i][num] = col[j][num] = box[i//3][j//3][num] = False

        for i in range(9):
            for j in range(9):
                if board[i][j] == '.': continue
                num = int(board[i][j])
                row[i][num] = col[j][num] = box[i//3][j//3][num] = True
        dfs(0)

    def countAndSay(self, n: int) -> str:
        # p38 easy math
        # ... 外观数列，是个有意思的数论问题
        pass

    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        # p39 medium DP DFS
        # ... 无重复元素，无限制重复选取，无穷背包问题 （网上查 《背包九讲》非常经典）
        # ... 这里要所有解，如果用DFS也可以 dfs(index:int, sum_list:list)
        res = [[] for _ in range(target+1)]
        res[0].append([]) # 初始化，没有元素
        for i in candidates:
            for j in range(i,  target+1): # 正序-无穷背包
                for k_list in res[j-i]:
                    res[j].append(k_list+[i])
        return res[target]

    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        # p40 medium DP DFS
        # ... 分组背包问题（存在重复元素）
        res = [[] for _ in range(target+1)]
        a = collections.Counter(candidates)
        res[0].append([])
        for k, v in a.items():
            for j in range(target, k-1, -1):  # 逆序-分组背包
                for mv in range(1, v+1):   # 分组集合
                    if j-k*mv < 0: break
                    tmp = [k]*mv
                    for k_list in res[j-k*mv]:
                        res[j].append(k_list+tmp)
        return res[target]


    def trap(self, height: List[int]) -> int:
        # p42 hard 数学 双指针/DP
        # ... DP比较容易写，双指针空间省，难写一点。官方题解很不错
        def trap_DP():
            left = [0]*len(height)
            for i in range(1, len(height)-1):
                left[i] = max(left[i-1], height[i-1])
            right = [0]*len(height)
            for i in range(len(height)-2, 0, -1):
                right[i] = max(right[i+1], height[i+1])
            res = 0
            for i in range(1, len(height)-1):
                if min(left[i], right[i]) > height[i]:
                    res += min(left[i], right[i])-height[i]
            return res

        def trap_window():
            if len(height) <= 2: return 0
            left, right = 0, len(height)-1
            left_max, right_max = height[0], height[-1]
            res = 0
            while left < right:
                if left_max < right_max:
                    left += 1
                    res += max(0, left_max-height[left])
                    left_max = max(left_max, height[left])
                else:
                    right -= 1
                    res += max(0, right_max-height[right])
                    right_max = max(right_max, height[right])
            return res
            # AC 其实真没啥差别

        return trap_DP()
