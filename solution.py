# -*- coding: utf-8 -*-
# @Author  : mew

from leetcode.my_defs import *
from typing import *
import bisect
import collections
import itertools


class Solution:
    """ 学习编程技巧 """

    def isMatch_p10(self, s, p):
        """ p10 hard 正则 DP

        完全匹配，*是重复
        """
        ns, np = len(s), len(p)
        a = [[False]*(np+1) for _ in range(ns+1)]
        a[0][0] = True  # a[i][j] 表示 s[:i] 和 p[:j] 完全匹配
        for i in range(np):
            a[0][i+1] = p[i] == '*' and a[0][i-1]  # 只有前面出现*时才匹配
        for i in range(ns):
            for j in range(np):
                if p[j] == '*':
                    # 这里是匹配p的两个 j-1, j+1，没有用到j
                    # 可能有 '.*'能匹配任何的
                    a[i+1][j+1] = a[i+1][j-1] or ((s[i] == p[j-1] or p[j-1] == '.') and a[i][j+1])
                else:
                    a[i+1][j+1] = a[i][j] and (s[i] == p[j] or p[j] == '.')
        # pprint(a)
        return a[-1][-1]

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """ p15 medium 双指针 """
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n-2):
            if i > 0 and nums[i] == nums[i-1]: continue
            l, r = i+1, n-1
            tar = -nums[i]
            while l < r:
                if nums[l]+nums[r] == tar:
                    res.append([nums[x] for x in [i, l, r]])
                    l += 1
                    while l < r and nums[l] == nums[l-1]: l += 1
                    r -= 1
                elif nums[l]+nums[r] < tar:
                    l += 1
                else:
                    r -= 1
        return res

    def threeSumClosest(self, nums: List[int], target: int) -> int:
        """ p16 medium 双指针 """
        nums.sort()
        n = len(nums)
        res = sum(nums[:3])
        for i in range(n-2):
            l, r = i+1, n-1
            while l < r:
                tmp = nums[i]+nums[l]+nums[r]
                if abs(tmp-target) < abs(res-target):
                    res = tmp
                if tmp < target:
                    l += 1
                else:
                    r -= 1
        return res

    def letterCombinations(self, digits: str) -> List[str]:
        """ p17 medium 排列

        列表扩展的方法可以考虑下：
        >>> ret = [item+a for a in alphabets for item in ret]
        """
        if not digits: return []
        func = lambda i:{"2":"abc", "3":"def", "4":"ghi", "5":"jkl",
                         "6":"mno", "7":"pqrs", "8":"tuv", "9":"wxyz"}[i]
        return [''.join(x) for x in itertools.product(*map(func, list(digits)))]

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        """ p18 medium 排序

        做各种超界剪枝，能够更快
        """
        nums.sort()
        n = len(nums)
        res = []
        for i in range(n-3):
            if i > 0 and nums[i] == nums[i-1]: continue
            for j in range(i+1, n-2):
                if j > i+1 and nums[j] == nums[j-1]: continue
                l, r = j+1, n-1
                tar = target-nums[i]-nums[j]
                while l < r:
                    if nums[l]+nums[r] == tar:
                        res.append([nums[x] for x in [i, j, l, r]])
                        l += 1
                        while l > j+1 and l < r and nums[l] == nums[l-1]: l += 1
                        r -= 1
                    elif nums[l]+nums[r] < tar:
                        l += 1
                    else:
                        r -= 1
        return res

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
        l = bisect.bisect_left(nums, target)
        if len(nums) == 0 or l == len(nums) or nums[l] != target:
            return [-1, -1]
        else:
            return [l, bisect.bisect(nums, target)-1]

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
        res[0].append([])  # 初始化，没有元素
        for i in candidates:
            for j in range(i, target+1):  # 正序-无穷背包
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
                for mv in range(1, v+1):  # 分组集合
                    if j-k*mv < 0: break
                    tmp = [k]*mv
                    for k_list in res[j-k*mv]:
                        res[j].append(k_list+tmp)
        return res[target]

    def firstMissingPositive(self, nums: List[int]) -> int:
        # p41 hard  math
        # ... 有意思的面试题
        for i in range(len(nums)):  # 将 nums[i] 放到 nums[nums[i]-1] 位置
            while 0 <= nums[i] < len(nums) and nums[nums[i]-1] != nums[i]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        i = 0
        while i < len(nums):
            if nums[i] == i+1: i += 1
            else: break
        return i+1

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

    def multiply(self, num1: str, num2: str) -> str:
        """ p43 medium math
        算法-- 面试+

        给定两个以字符串形式表示的非负整数 num1 和 num2，返回 num1 和 num2 的乘积，它们的乘积也表示为字符串形式。
        """

        class myInt:
            def __init__(self, num: str, max_len=250):
                self._max_len = max_len
                self._num = [0]*max_len
                for i, ni in enumerate(reversed(num)):
                    self._num[i] += int(ni)

            def __str__(self):
                return ''.join(map(str, reversed(self._num))).lstrip('0') or '0'

            def __mul__(self, other):
                res = myInt('')
                for i in range(self._max_len):
                    tag = 0
                    for j in range(other._max_len):
                        if i+j < self._max_len:
                            res._num[i+j] += self._num[i]*other._num[j]+tag
                            tag, res._num[i+j] = res._num[i+j]//10, res._num[i+j]%10
                        else: break
                return res

            def __add__(self, other):
                res = myInt('')
                tag = 0
                for i in range(self._max_len):
                    res._num[i] = self._num[i]+other._num[i]+tag
                    tag, res._num[i] = res._num[i]//10, res._num[i]%10
                return res

        return str(myInt(num1)*myInt(num2))

    def isMatch(self, s: str, p: str) -> bool:
        """ p44 hard 递归 DP

        通配符匹配
        """

        def isMatch_unaccepted():
            while '**' in p:
                p = p.replace('**', '*')

            def dfs(s, p):
                if p == '': return s == ''
                if p[0] == '*':
                    idx = 0
                    while idx <= len(s):
                        # 保证每次迭代p减少
                        if dfs(s[idx:], p[1:]): return True
                        idx += 1
                    return False
                elif p[0] == '?':
                    return len(s) > 0 and dfs(s[1:], p[1:])
                else:
                    return len(s) > 0 and s[0] == p[0] and dfs(s[1:], p[1:])

            return dfs(s, p)
            # 会超时

        dp = [[False]*(len(s)+1) for _ in range(len(p)+1)]
        dp[0][0] = True
        for i in range(1, len(p)+1):  # 这一段初始化很重要，表示p开头是'*'的与s开头为''匹配。
            if p[i-1] == '*':
                dp[i][0] = dp[i-1][0]
        for i in range(1, len(p)+1):
            for j in range(1, len(s)+1):
                if p[i-1] == '*':
                    dp[i][j] = dp[i][j-1] or dp[i-1][j]
                if p[i-1] == s[j-1] or p[i-1] == '?':
                    dp[i][j] = dp[i-1][j-1]
        return dp[-1][-1]

    def jump(self, nums: List[int]) -> int:
        """ p45 hard 贪心
        """
        step = end = max_pos = 0
        for i in range(len(nums)-1):  # 最后一格不关心
            max_pos = max(max_pos, i+nums[i])
            if i == end:
                end = max_pos
                step += 1
        return step

    def permute(self, nums: List[int]) -> List[List[int]]:
        """ p46 medium 递归
        面试++

        题解：也可以用itertools.permutations 了解下这个函数的机制
        """

        def dfs(idx):
            if idx == len(nums):
                res.append(nums[:])
            else:
                for i in range(idx, len(nums)):
                    nums[i], nums[idx] = nums[idx], nums[i]
                    dfs(idx+1)
                    nums[i], nums[idx] = nums[idx], nums[i]

        res = []
        dfs(0)
        return res

    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        """ p47 medium 递归
        算法+

        题解：根据递归的方法，分为需要回溯的递归（递归过程数组共享，则必须记录使用情况，并且末尾还原）
                           和不需要回溯的递归（递归过程数组不共享，则不用担心回溯时数组的变动）

        其他解法：
        1. 数组循环：每增加一个就循环一次，生成len个新数组，然后重复（不需要回溯）；
                    每增加一位就循环后面的数组，生成len-idx个新数组，然后重复（不需要回溯）
        2. 数据标记：需要回溯的情景（或者连同数据循环，用不回溯的方法）
        3. 结合有序数组
        """

        def dfs(nums, idx):
            # print(idx, nums)
            if idx == len(nums)-1:
                res.append(nums)
            else:
                dfs(nums.copy(), idx+1)  # 原始的
                for i in range(idx, len(nums)):
                    if nums[i] != nums[idx]:  # 如果不相等，必定是idx位置数值变大的序列
                        nums[i], nums[idx] = nums[idx], nums[i]
                        dfs(nums.copy(), idx+1)

        nums.sort()
        res = []
        dfs(nums, 0)
        return res

    def rotate(self, matrix: List[List[int]]) -> None:
        """ P48 easy 数学
        面试++ 算法--

        题解： 矩阵旋转，寻找位置定位(i, j)->(j, n-1-i)
            先转置后轴对称？
        """
        n = len(matrix)
        for i in range(n):
            for j in range(i+1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        for i in range(n):
            for j in range(n//2):
                matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]

    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """ p49 medium 排序 编码
        面试+

        题解：对字符串编码成key

        Input: strs = ["eat","tea","tan","ate","nat","bat"]
        Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
        """
        from collections import defaultdict
        res = defaultdict(list)
        for i in strs:
            res[''.join(sorted(i))].append(i)
        return list(res.values())

    def myPow(self, x: float, n: int) -> float:
        """ p50 medium
        面试+ 算法+

        题解：计算pow(x, n), 字宽思想
        """
        res = 1
        sign = 1 if n > 0 else -1
        n *= sign
        while n > 0:
            if n%2 == 1:
                res *= x
            n >>= 1  # n是整数
            x *= x
        return res if sign == 1 else 1/res

    def solveNQueens(self, n: int) -> List[List[str]]:
        """ p51 hard DFS

        题解：8皇后问题，硬搜，位运算编码
            行编码0~N-1，列编码0~N-1，对角编码i-j(-N+1~N-1)，对角编码i+j(0~2N-2)
        """

        def dfs(i):
            if i == n: res.append(['.'*j+'Q'+'.'*(n-j-1) for j in cur])
            else:
                for j in range(n):
                    if row[i] and col[j] and diag[i-j] and diag2[i+j]:
                        row[i] = col[j] = diag[i-j] = diag2[i+j] = False
                        cur.append(j)
                        dfs(i+1)
                        cur.pop()
                        row[i] = col[j] = diag[i-j] = diag2[i+j] = True

        res = []
        cur = []
        row = [True]*n
        col = [True]*n
        diag = [True]*(2*n-1)
        diag2 = [True]*(2*n-1)
        dfs(0)

        return res

    def totalNQueens(self, n: int) -> int:
        """ p52 hard DFS

        题解：同p51， pass
        """
        pass

    def maxSubArray(self, nums: List[int]) -> int:
        """ p53 medium 滑动窗口 线段树

        题解：可以用滑动窗口做，这里写一个线段树的版本（掌握线段树很重要）
        """
        raise NotImplemented

    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        """ p54 medium 模拟

        题解：【螺旋矩阵】模拟转向，比较喜欢这个解法
        """
        res = []
        i, j = 0, 0
        m, n = len(matrix), len(matrix[0])
        di, dj = 0, 1
        for _ in range(m*n):
            res.append(matrix[i][j])
            matrix[i][j] = 0xffff
            if not (0 <= i+di < m and 0 <= j+dj < n) or matrix[i+di][j+dj] == 0xffff:
                di, dj = dj, -di
            i, j = i+di, j+dj
        return res

    def canJump(self, nums: List[int]) -> bool:
        """ p55 medium 滑动窗口

        题解：【跳跃游戏】滑动窗口方法（此方法顺便可以求出跳几步）
            比较讨巧的方法，是倒序判断是否可以到达末尾端点，一直倒推到起点。
        """
        left, right = 0, 0+nums[0]
        while left < right and right < len(nums)-1:  # 可以继续跳，并且还没跳到最后
            left, right = right, max(i+nums[i] for i in range(left, right+1))
        return right >= len(nums)-1

    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        """ p56 meidum 排序 模拟

        题解：【合并区间】没啥营养
        """
        intervals.sort()
        res = []
        si, sj = intervals[0]
        for i, j in intervals[1:]:
            if sj >= i:
                sj = j if j > sj else sj
            else:
                res.append((si, sj))
                si, sj = i, j
        res.append([si, sj])
        return res

    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        """ p57 medium 模拟

        题解：【插入区间】
        """
        return self.merge(intervals+[newInterval])

    def lengthOfLastWord(self, s: str) -> int:
        """ p58 easy 字符串处理

        题解：最后一个单词的长度
        """
        return s.rstrip().rsplit(' ', maxsplit=1)[-1].__len__()

    def generateMatrix(self, n: int) -> List[List[int]]:
        """ p59 medium 数学

        题解：【螺旋矩阵II】，同 p54
        """
        arr = [[0]*n for _ in range(n)]
        i = j = 0
        cur = 1
        dx, dy = 0, 1
        while cur <= n*n:
            arr[i][j] = cur
            cur += 1
            if i+dx < 0 or i+dx >= n or j+dy < 0 or j+dy >= n or arr[i+dx][j+dy] > 0:
                dx, dy = dy, -dx
            i, j = i+dx, j+dy
        return arr

    def getPermutation(self, n: int, k: int) -> str:
        """ p60 hard 排序
        算法++

        题解：【排序序列】
            n=3, 共6种，123, 132, 213, 231, 312, 321
            k=5, divmod(k, 2!)= 2, 1
            divmod(1, 1!) = 1, 0
            因此序列为(2, 1, 0) 最后一位必定为0
            对应的序列为'123'中的3, '12'中的2, '1'中的1
        """

        def nfactorial(x):
            return 1 if x <= 1 else x*nfactorial(x-1)

        def _getPermutation(a: list, k: int) -> list:
            res = []
            while a:
                idx, k = divmod(k, nfactorial(len(a)-1))
                res.append(a.pop(idx))
            return res

        return ''.join(_getPermutation([str(i+1) for i in range(n)], k-1))

    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        """ p61
        面试++

        题解：【旋转链表】
        """

        def _len(head):
            return 0 if head is None else 1+_len(head.next)

        def _rotate_right(head, pre_tail, tail):
            new_head = pre_tail.next
            pre_tail.next = None
            tail.next = head
            return new_head

        def _get_kth_node(head, k):
            return head if k == 0 else _get_kth_node(head.next, k-1)

        n = _len(head)
        if n < 2 or k%n == 0: return head
        return _rotate_right(head, _get_kth_node(head, n-1-k%n), _get_kth_node(head, n-1))

    def uniquePaths(self, m: int, n: int) -> int:
        """ p62 medium DP
        题解：【不同路径】经典动规题
        """
        dp = [[1]*n for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j]+dp[i][j-1]
        return dp[m-1][n-1]

    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        """ p63 medium DP
        题解：【不同路径II】动规
        """
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0]*n for _ in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] == 0: dp[i][0] = 1
            else: break
        for j in range(n):
            if obstacleGrid[0][j] == 0: dp[0][j] = 1
            else: break
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i-1][j]+dp[i][j-1]
        return dp[-1][-1]

    def minPathSum(self, grid: List[List[int]]) -> int:
        """ p64 medium DP
        题解：【最小路径和】和p62,63类似 """
        m, n = len(grid), len(grid[0])
        dp = [grid[i].copy() for i in range(m)]
        for i in range(1, m): dp[i][0] += dp[i-1][0]
        for j in range(1, n): dp[0][j] += dp[0][j-1]
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] += min(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]

    def isNumber(self, s: str) -> bool:
        """ p65 hard 字符串处理 模拟
        """
        raise NotImplemented

    def plusOne(self, digits: List[int]) -> List[int]:
        """ p66 easy 模拟
        题解：【加一】没啥可说的"""
        cur, idx = 1, len(digits)-1
        while cur and idx >= 0:
            cur, digits[idx] = divmod(digits[idx]+cur, 10)
            idx -= 1
        if cur and idx == -1:
            digits.insert(0, cur)
        return digits

    def addBinary(self, a: str, b: str) -> str:
        """ p67 easy 字符串处理
        """
        return bin(int(a, 2)+int(b, 2))[2:]

    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        """ p68 hard 模拟
        题解：
        """
        raise NotImplemented

    def mySqrt(self, x: int) -> int:
        """ p69 easy 二分
        题解：【x的平方根】"""
        if x == 0: return 0
        l, r = 1, x
        while l+1 < r:
            mid = (l+r)>>1
            if mid*mid <= x:
                l = mid
            else:
                r = mid
        return l

    def climbStairs(self, n: int) -> int:
        """ p70 easy 数学
        题解：【爬楼梯】斐波那契数列"""

        @lru_cache
        def fib(n):
            return 1 if n <= 1 else fib(n-1)+fib(n-2)

        return fib(n)

    def simplifyPath(self, path: str) -> str:
        """ p71 medium 栈
        题解：【简化路径】"""
        stack = []
        for i in path.split('/'):
            if i == '.' or i == '': continue
            if i == '..':
                if stack: stack.pop()
                continue
            stack.append(i)
        return '/'+'/'.join(stack)

    def minDistance(self, word1: str, word2: str) -> int:
        """ p72 hard """
        m, n = len(word1), len(word2)
        dp = [[0]*(n+1) for _ in range(m+1)]  # dp[i][j]表示word1[:i],word2[:j]匹配的最少操作数
        for i in range(m+1): dp[i][0] = i  # 这里是 m+1 不是 m
        for j in range(n+1): dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if word1[i-1] == word2[j-1] else 1))
        return dp[m][n]

    def setZeroes(self, matrix: List[List[int]]) -> None:
        """ p73 medium 模拟
        为了满足，空间 O(1)
        """
        m, n = len(matrix), len(matrix[0])
        flag_row = any(matrix[i][0] == 0 for i in range(m))
        flag_col = any(matrix[0][i] == 0 for i in range(n))
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
        for i in range(1, m):
            if matrix[i][0] == 0:
                for j in range(1, n):
                    matrix[i][j] = 0
        for j in range(1, n):
            if matrix[0][j] == 0:
                for i in range(1, m):
                    matrix[i][j] = 0
        if flag_row:
            for i in range(0, m):
                matrix[i][0] = 0
        if flag_col:
            for j in range(0, n):
                matrix[0][j] = 0
        return

    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        """ p74 medium 二分
        题解：先纵坐标二分，再横坐标二分。考虑二分的语义，有意思的"""
        if target < matrix[0][0] or target > matrix[-1][-1]: return False
        row = bisect.bisect_right([matrix[i][0] for i in range(len(matrix))], target)
        col = bisect.bisect_right(matrix[row-1], target)
        return matrix[row-1][col-1] == target

    def sortColors(self, nums: List[int]) -> None:
        """ p75 medium  状态机
        面试+ 时间O(N)"""
        if len(nums) <= 1: return
        b, e, i = 0, len(nums)-1, 0
        while i <= e:
            if nums[i] == 0:
                nums[b], nums[i] = nums[i], nums[b]
                b += 1
                i += 1
            elif nums[i] == 2:
                nums[e], nums[i] = nums[i], nums[e]
                e -= 1
            else:  # ==1
                i += 1

    def minWindow(self, s: str, t: str) -> str:
        """ p76 hard 滑动窗口
        算法++

        >>> Solution().minWindow("ACCD", "ABC")
        ''
        >>> Solution().minWindow("ADOBECODEBANC", "ABC")
        'BANC'
        >>> Solution().minWindow("a", "a")
        'a'

        """
        a = Counter(t)
        tot = len(a.keys())
        left = -1
        res = '#'*(len(s)+1)  # 哨兵
        for i, si in enumerate(s):
            if si in a:
                a[si] -= 1
                if a[si] == 0: tot -= 1
                while tot == 0:  # 全部满足
                    left += 1
                    if s[left] in a:
                        a[s[left]] += 1
                        if a[s[left]] == 1:
                            tot += 1
                            if len(s[left:i+1]) < len(res): res = s[left:i+1]
        return '' if res[0] == '#' else res

    def combine(self, n: int, k: int) -> List[List[int]]:
        """ p77 medium 回溯 排列
        内置的函数是 itertools.combinations
        >>> Solution().combine(4,2)
        [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        """
        res = [[]]
        for i in range(k):
            tmp = []
            for lst in res:
                for k in range(lst[-1]+1 if lst else 1, n+1):
                    tmp.append(lst+[k])
            res = tmp
        return res

    def subsets(self, nums: List[int]) -> List[List[int]]:
        """ p78 medium 排序 回溯
        >>> Solution().subsets([1,2,3])
        [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
        """
        res = [[]]
        for i in nums:
            res.extend([j+[i] for j in res])
        return res

    def exist(self, board: List[List[str]], word: str) -> bool:
        """ p79 medium 回溯
        加个剪枝（字符统计Counter）会快很多

        >>> Solution().exist([list('ABCE'),list('SFCS'),list('ADEE')],'ABCCED')
        True
        >>> Solution().exist([list('ABCE'),list('SFCS'),list('ADEE')],'SEE')
        True
        >>> Solution().exist([list('ABCE'),list('SFCS'),list('ADEE')],'ABCB')
        False

        """

        def check(i, j, k):  # k是已匹配的数量 （对于len==1的word）
            if k == len(word): return True
            tmp, board[i][j] = board[i][j], ''
            for dx, dy in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
                if 0 <= i+dx < n and 0 <= j+dy < m:
                    if word[k] == board[i+dx][j+dy] and check(i+dx, j+dy, k+1):
                        return True
            board[i][j] = tmp
            return False

        n, m = len(board), len(board[0])
        for i in range(n):
            for j in range(m):
                if board[i][j] == word[0] and check(i, j, 1): return True
        return False

    def removeDuplicates_82(self, nums: List[int]) -> int:
        """ p80 medium 数学
        >>> Solution().removeDuplicates_82([0,0,1,1,1,1,2,3,3])
        7
        >>> Solution().removeDuplicates_82([1,1,1,2,2,3])
        5
        """
        idx = 2
        for i in range(2, len(nums)):
            if nums[idx-2] != nums[i]:
                nums[idx] = nums[i]
                idx += 1
        return idx

    def search_81(self, nums: List[int], target: int) -> bool:
        """ p81 medium 二分 """
        pass

    def deleteDuplicates(self, head: ListNode) -> ListNode:
        """p82 medium 链表操作
        """
        if not head: return head

        node = ListNode(head.val-1)
        node.next = head
        a, b = node, head
        while b:
            if b.next and b.val == b.next.val:
                while b.next and b.val == b.next.val:
                    b = b.next
                a.next = b.next
                b = b.next
            else:
                a, b = a.next, b.next

        return node.next

    def largestRectangleArea(self, heights: List[int]) -> int:
        """ p84 hard 栈
        算法++

        非常有意思的一道题，单调栈
        left 和 right 定位，以及末尾的处理（入栈后循环结束）

        """

        def largestRectangleArea_faster():
            heights.append(0)  # 做题可以这样简单处理，实际中并不好
            stack = [-1]  # 比较好的处理
            ans = 0
            for i in range(len(heights)):
                while heights[i] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = i-stack[-1]-1
                    ans = max(h*w, ans)
                stack.append(i)

        if len(heights) == 0: return 0
        stack = []
        res = 0
        MIN_HEIGHT = 0
        for i in range(len(heights)+1):
            cur = heights[i] if i < len(heights) else MIN_HEIGHT
            while stack and cur <= heights[stack[-1]]:  # = 的情景，左边高度会失真，右边高度会正确计算的。
                h = heights[stack.pop()]  # 当前index对应的高度左右扩展
                left = stack[-1]+1 if stack else (-1)+1
                right = i
                # print(left, right, h, stack)
                area = (right-left)*h
                res = max(res, area)
            stack.append(i)

        return res

    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        """ p85 hard 位运算 DP
        算法+++

        """

        @lru_cache
        def width(base):
            if base: return width(base&base>>1)+1
            else: return 0

        m = [int(''.join(i), 2) for i in matrix]
        n = len(m)
        res = 0
        for i in range(n):
            base = m[i]
            j = i+1
            while base:
                res = max(res, width(base)*(j-i))  # j-i = height
                if j == n: break
                base &= m[j]
                j += 1
        return res

    def partition(self, head: ListNode, x: int) -> ListNode:
        """ p86 medium 链表操作
        """
        dummy = ListNode(0)
        tail = dummy
        dummy2 = ListNode(0)
        tail2 = dummy2
        while head:
            if head.val < x:
                tail.next = head
                tail = tail.next
            else:
                tail2.next = head
                tail2 = tail2.next
            head = head.next
        tail2.next = None
        tail.next = dummy2.next
        return dummy.next

    def isScramble(self, s1: str, s2: str) -> bool:
        """ p87 hard DP
        """
        if Counter(s1) != Counter(s2): return False

        @lru_cache(30000)
        def dp(i, j, l):
            if l == 1: return s1[i] == s2[j]
            else:
                return any(
                        # 不交换
                        dp(i, j, k) and dp(i+k, j+k, l-k) or
                        # 交换
                        dp(i, j+l-k, k) and dp(i+k, j, l-k) for k in range(1, l))

        return dp(0, 0, len(s1))  # here len(s1)==len(s2)

    def grayCode(self, n: int) -> List[int]:
        """ p89 medium
        """
        # 0111^*011 = *100, 1000^*100 = (1^*)100
        # 所以 i^i>>1 得到的结果，仅在i二进制倒数第一个0的位置，与(i+1)^(i+1)>>1得到的结果不同，
        return [i^i>>1 for i in range(2**n)]

    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        """ p90 medium 集合背包
        """
        res = [[]]
        for k, v in Counter(nums).items():
            res.extend([lst+[k]*vi for lst in res for vi in range(1, v+1)])
        return res

    def numDistinct(self, s: str, t: str) -> int:
        """ p115 hard DP

        s中包含t的子序列数量
        >>> Solution().numDistinct('rabbbit', 'rabbit')
        3
        >>> Solution().numDistinct('babgbag', 'bag')
        5
        """
        if len(s) == 0 or len(t) == 0: return 0
        ns, nt = len(s), len(t)
        a = [[0]*nt for _ in range(ns)]
        a[0][0] = int(s[0] == t[0])
        for i in range(1, ns):
            # 这里就是在计数，也可以用再一个空列，然后累计到后面方法
            a[i][0] = int(s[i] == t[0])+a[i-1][0]
        for i in range(nt):
            # a[0][i] = int(s[0] == t[i])+a[0][i-1]
            pass  # 这里都是0， 因为 t 必须匹配
        for i in range(1, ns):
            for j in range(1, nt):  # 第二重
                if s[i] == t[j]:  # a[i][j] 表示 s[:i+1] 中子序列 t[:j+1] 的数量
                    a[i][j] = a[i-1][j]+a[i-1][j-1]
                else:
                    a[i][j] = a[i-1][j]  # 这里 j 必须匹配 所以不能跳过j
        # pprint(a)
        return a[-1][-1]
