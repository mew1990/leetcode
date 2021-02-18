# -*- coding: utf-8 -*-
# @Author  : mew

# 所有题目来自 leetcode，仅供学习用途
from typing import *


class Contest_228:
    def minOperations(self, s: str) -> int:
        """ 1758. 生成交替二进制字符串的最少操作数

        给你一个仅由字符 '0' 和 '1' 组成的字符串 s 。一步操作中，你可以将任一 '0' 变成 '1' ，或者将 '1' 变成 '0'

        Error 1 + AC
            一开始思路是第一个字符正确，后面的随之修改，这个思路是错误的。
            需要有整体思路，最终解法如下。
            更优的解法推荐：只需要记录一种情况，第二种情况是互补的。

            >>> cnt1 = 0
            >>> sign1 = '0'
            >>> for c in s:
            >>>     if c != sign1:
            >>>         cnt1 += 1
            >>>     if sign1 == '0':
            >>>         sign1 = '1'
            >>>     else:
            >>>         sign1 = '0'
            >>> return min(cnt1, len(s) - cnt1)

        """
        res = sum(si != '1' if i%2 == 0 else si != '0' for i, si in enumerate(s))
        res2 = sum(si != '1' if i%2 == 1 else si != '0' for i, si in enumerate(s))
        return min(res, res2)

    def countHomogenous(self, s: str) -> int:
        """ 1759. 统计同构子字符串的数目

        给你一个字符串 s ，返回 s 中 同构子字符串 的数目。
        同构字符串 的定义为：如果一个字符串中的所有字符都相同，那么该字符串就是同构字符串。
        子字符串 是字符串中的一个连续字符序列。

        输入：s = "abbcccaa"
        输出：13
        解释：同构子字符串如下所列：
        "a"   出现 3 次。
        "aa"  出现 1 次。
        "b"   出现 2 次。
        "bb"  出现 1 次。
        "c"   出现 3 次。
        "cc"  出现 2 次。
        "ccc" 出现 1 次。
        3 + 1 + 2 + 1 + 3 + 2 + 1 = 13

        AC
            算法方面没啥问题

        Tips
            有一个 itertools.groupby 方法合适解本题

        """

        def func(n):
            return (n*(n+1))//2

        res = a = 0
        cur = ''
        for i in s+' ':
            if i == cur:
                a += 1
            else:
                res = (func(a)+res)%1000000007
                cur = i
                a = 1
        return res

    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        """ 1760. 袋子里最少数目的球

        最多maxOperations次操作，每次分一个整数为两个正整数，使最后所有整数的最大值最小。

        输入：nums = [4, 9], maxOperations = 2
        输出：4
        解释：
            9 分成6和3, 6分成3和3，最后得到[4,3,3,3]

        Fail
            最初想到是模拟，不过这个模拟策略不好找，而且模拟必须是状态可以转移的。
            比赛结束后，看到用搜索，的确是可行的。
            希望要有整体思维。
            代码补上。
        """
        left, right = 1, max(nums)
        while left < right:
            mid = (left+right)>>1
            if sum((i-1)//mid for i in nums) > maxOperations:
                left = mid+1
            else:
                right = mid
        return left

    def minTrioDegree(self, n: int, edges: List[List[int]]) -> int:
        """ 1761. 一个图中连通三元组的最小度数

        给你一个无向图，整数 n 表示图中节点的数目，edges 数组表示图中的边，
        其中 edges[i] = [ui, vi] ，表示 ui 和 vi 之间有一条无向边。
        返回所有连通三元组中度数的 最小值 ，如果图中没有连通三元组，那么返回 -1 。

        连通三元组的度数 是所有满足此条件的边的数目：一个顶点在这个三元组内，而另一个顶点不在这个三元组内。

        输入：n = 6, edges = [[1,2],[1,3],[3,2],[4,1],[5,2],[3,6]]
        输出：3
        解释：只有一个三元组 [1,2,3] 。构成度数的边在上图中已被加粗。

        Error 2 + AC
            想到用邻接矩阵表示，关于确认联通三元组，想到的方法比较复杂，时间复杂度也不低。但还是写完了，
            >>> m = [[False]*(n+1) for _ in range(n+1)]
            >>> from collections import defaultdict
            >>> dd = defaultdict(list)
            >>> res = defaultdict(int)
            >>> for u, v in edges:
            >>>     dd[u].append(v)
            >>>     dd[v].append(u)
            >>>     m[u][v] = m[v][u] = True
            >>> for i in dd.keys():
            >>>     if len(dd[i]) < 2: continue
            >>>     nn, lst = len(dd[i]), dd[i]
            >>>     for j in range(nn):
            >>>         for k in range(j+1, nn):
            >>>             if m[lst[j]][lst[k]]:
            >>>                 res[tuple(sorted([i, lst[j], lst[k]]))] += nn-2
            >>> return min(res.values()) if res else -1
            最后想到python对这样细枝末节的操作,时间消耗会很严重,所以不如直接用二元矩阵表示关系,直接三层遍历,
            根据数据量,可能可以,果然通过了.
        """
        m = [[False]*(n+1) for _ in range(n+1)]
        lst = [-2]*(n+1)
        res = n*3
        for u, v in edges:
            lst[u] += 1
            lst[v] += 1
            m[u][v] = m[v][u] = True
        for i in range(1, n-1):
            for j in range(i+1, n):
                for k in range(j+1, n+1):
                    if m[i][j] and m[j][k] and m[k][i]:
                        res = min(res, lst[i]+lst[j]+lst[k])
        return -1 if res == n*3 else res


class Contest_227:
    def check(self, nums: List[int]) -> bool:
        """ 1752. 检查数组是否经排序和轮转得到

        输入：nums = [3,4,5,1,2]
        输出：true
        解释：[1,2,3,4,5] 为有序的源数组。
        可以轮转 x = 3 个位置，使新数组从值为 3 的元素开始：[3,4,5,1,2] 。

        AC
            简单题，这里展示的是评论中比较好的一个解法
        """
        return sum(nums[i] < nums[i-1] for i in range(len(nums))) <= 1  # i = 0 的时候非常巧妙

    def maximumScore(self, a: int, b: int, c: int) -> int:
        """ 1753. 移除石子的最大得分

        你正在玩一个单人游戏，面前放置着大小分别为 a b 和 c​的 三堆石子。
        每回合你都要从两个不同的非空堆中取出一颗石子，并在得分上加 1 分。当存在 两个或更多 的空堆时，游戏停止。

        Error 1 + AC
            数学题，对应三角形的三边，一开始忽略了无法形成三角形的情况，所以直接给了 (a+b+c)//2

        """
        tot = a+b+c
        m = max(a, b, c)
        if m > tot-m: return tot-m
        else: return tot//2

    def largestMerge(self, word1: str, word2: str) -> str:
        """ 1754. 构造字典序最大的合并字符串

        输入：word1 = "cabaa", word2 = "bcaaa"
        输出："cbcabaaaaa"
        解释：构造字典序最大的合并字符串，可行的一种方法如下所示：
        - 从 word1 中取第一个字符：merge = "c"，word1 = "abaa"，word2 = "bcaaa"
        - 从 word2 中取第一个字符：merge = "cb"，word1 = "abaa"，word2 = "caaa"
        - 从 word2 中取第一个字符：merge = "cbc"，word1 = "abaa"，word2 = "aaa"
        - 从 word1 中取第一个字符：merge = "cbca"，word1 = "baa"，word2 = "aaa"
        - 从 word1 中取第一个字符：merge = "cbcab"，word1 = "aa"，word2 = "aaa"
        - 将 word1 和 word2 中剩下的 5 个 a 附加到 merge 的末尾。

        Error 3 + AC
            错了3次，对于“字典序最大”这个要求，没有完全理解，直到好几个测试案例的模拟才明白
            整体字典序大，但是只能取第一个字符！！！
            所以，最后得到的这个解法。
            这里再贴一个迭代版（字符串处理可以改成list+join）：
            >>> res=''
            >>> while word1 and word2:
            >>>     if word1>word2:
            >>>         res+=word1[0]
            >>>         word1=word1[1:]
            >>>     else:
            >>>         res+=word2[0]
            >>>         word2=word2[1:]
            >>> res+=word1+word2
            >>> return res
        """
        if word1 == '': return word2
        if word2 == '': return word1
        if word1 > word2: return word1[0]+self.largestMerge(word1[1:], word2)
        else: return word2[0]+self.largestMerge(word1, word2[1:])

    def minAbsDifference(self, nums: List[int], goal: int) -> int:
        """ 1755. 最接近目标值的子序列和

        给你一个整数数组 nums 和一个目标值 goal 。
        你需要从 nums 中选出一个子序列，使子序列元素总和最接近 goal 。
        也就是说，如果子序列元素和为 sum ，你需要 最小化绝对差 abs(sum - goal) 。
        返回 abs(sum - goal) 可能的 最小值 。

        输入：nums = [7,-9,15,-2], goal = -5
        输出：1
        解释：选出子序列 [7,-9,-2] ，元素和为 -4 。
        绝对差为 abs(-4 - (-5)) = abs(1) = 1 ，是可能的最小值。

        Fail
            想到用动态规划，类似背包问题，记得《背包九讲》中，提到DFS和DP的选择，N的分界就是40
            但是比赛时候解法超时了。没有进一步优化匹配，代码如下：
            >>> res_min = abs(sum(nums)-goal)
            >>> k = len(nums)//2
            >>> aset = {0}
            >>> for i in nums[:k]:
            >>>     tmp = {i+j for j in aset}
            >>>     aset.update(tmp)
            >>> bset = {0}
            >>> for i in nums[k:]:
            >>>     tmp = {i+j for j in bset}
            >>>     bset.update(tmp)
            >>> # print(aset, bset)
            >>> for i in aset:
            >>>     for j in bset:
            >>>         tmp = abs(i+j-goal)
            >>>         if tmp < res_min: res_min = tmp
            >>> return res_min
            所以改进后半段搜索效率，如下所示，能够900+ms勉强过线

            特别推荐一个优解：
            neg 和 pos 的两个数组，是提前进行预知过滤，能够 50ms AC
            >>> def minAbsDifference(self, nums: List[int], goal: int) -> int:
            >>>     n = len(nums)
            >>>     nums.sort(key=lambda x: -abs(x)) # 这里这里的排序，例如[-100, 50, -10, 5, -1]
            >>>     neg = [0 for _ in range(n+1)] # 注意 +1 的哨兵作用
            >>>     pos = [0 for _ in range(n+1)]
            >>>     for i in range(n-1, -1, -1): # 注意这里的倒序[-111, 55, -11, 5, -1]
            >>>         if nums[i] < 0:
            >>>             neg[i] = neg[i+1] + nums[i]
            >>>             pos[i] = pos[i+1]
            >>>         else:
            >>>             pos[i] = pos[i+1] + nums[i]
            >>>             neg[i] = neg[i+1]
            >>>     ans = abs(goal)
            >>>     s = set([0])
            >>>     def check(a, b):
            >>>         if b < goal - ans or goal + ans < a:
            >>>             return False
            >>>         return True
            >>>     for i in range(n): # 这里却是正序的，
            >>>         sl = [x for x in s if check(x+neg[i], x+pos[i])] #体现出neg和pos的作用，筛选 x
            >>>         if len(sl) == 0:
            >>>             break
            >>>         s = set(sl)
            >>>         for x in sl:
            >>>             y = x + nums[i]
            >>>             if abs(y - goal) < ans:
            >>>                 ans = abs(y - goal)
            >>>             if ans == 0:
            >>>                 return 0
            >>>             s.add(y)  # 更新s
            >>>     return ans

        """

        def dfs(lst):
            res = {0}
            for i in lst:
                tmp = {i+j for j in res}
                res.update(tmp)
            return res

        k = len(nums)//2
        alst = list(sorted(dfs(nums[:k])))
        blst = list(sorted(dfs(nums[k:])))
        i, j = 0, len(blst)-1  # 仔细思考下，这样的起始条件是合理的。
        res = abs(0-goal)  # (0, 0)也包含在其中的
        while i < len(alst) and j >= 0:  # 两头夹逼，类似某道题二位矩阵双向递增，求最接近的数字。
            res = min(res, abs(alst[i]+blst[j]-goal))
            if alst[i]+blst[j] > goal:
                j -= 1
            else:
                i += 1
        return res
