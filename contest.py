# -*- coding: utf-8 -*-
# @Author  : mew

# 所有题目来自 leetcode，仅供学习用途
from typing import *
from collections import *
from itertools import *
from functools import *
import math
import bisect


def interesting(func):
    def wrap(*args, **kwargs):
        print('# this program is Interesting!')
        return func(*args, **kwargs)

    return wrap


class Contest_230:
    """ 全场做下来，比较偏重数学推导和计算 """

    def countMatches(self, items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
        # 统计匹配检索规则的物品数量
        idx = ['type', 'color', 'name'].index(ruleKey)
        return sum(i[idx] == ruleValue for i in items)

    @interesting
    def closestCost(self, baseCosts: List[int], toppingCosts: List[int], target: int) -> int:
        """ 最接近目标价格的甜点成本

        AC
            用 分组背包？ 但是数据量很小，可以用dfs，不过我还是用了枚举。
            对于必须选择一样，有点懵，回头看，其实就是一个大的分组背包。就是倒数遍历状态叠加。

        Tips:
            看到tops选手的一个bitmap状态压缩，想想python有没有这个数据结构？哈哈，int本身就是bitmap啊
            改写如下（寻找target那块代码效率比较低）：
            >>> f = 0
            >>> for i in baseCosts:
            >>>     f |= 1<<i
            >>> for i in toppingCosts:
            >>>     f |= (f<<i)|(f<<(i+i))
            >>> low, high = target, target
            >>> while 1:
            >>>     if low > 0 and f>>low&1: return low
            >>>     if f>>high&1: return high
            >>>     low, high = low-1, high+1
        """
        a = {0}
        for i in toppingCosts:
            tmp = a.copy()
            for j in a:
                tmp.add(j+i)
                tmp.add(j+i+i)
            a = tmp
        a = list(sorted(a))
        res_d = min(baseCosts)
        res_u = max(baseCosts)+sum(toppingCosts)*2
        for i in baseCosts:
            idx = bisect.bisect_left(a, target-i)
            if idx > 0 and res_d < i+a[idx-1]: res_d = i+a[idx-1]
            if idx < len(a) and res_u > i+a[idx]: res_d = i+a[idx]
        if res_u-target < target-res_d:
            return res_u
        else:
            return res_d

    def minOperations(self, nums1: List[int], nums2: List[int]) -> int:
        """ 通过最少操作次数使数组的和相等

        AC+error2
            贪心+模拟，思路细节比较多，不算非常清晰，编码对相同数值进行压缩处理，error了2次，比赛时简化也能AC
            可以在逻辑和存储结构上再优化下
        """
        if len(nums1) < len(nums2): nums1, nums2 = nums2, nums1
        if len(nums1) > len(nums2)*6: return -1
        a = [nums1.count(i) for i in range(7)]
        b = [nums2.count(i) for i in range(7)]
        tota = sum(i*a[i] for i in range(7))
        totb = sum(i*b[i] for i in range(7))
        if tota == totb: return 0
        if tota < totb:
            a, b = b, a
        diff = abs(tota-totb)
        res = 0
        for i in range(1, 6):
            if (a[7-i]+b[i])*(6-i) >= diff:
                return res+(diff-1)//(6-i)+1
            res += a[7-i]+b[i]
            diff -= (a[7-i]+b[i])*(6-i)
        return res

    def getCollisionTimes(self, cars: List[List[int]]) -> List[float]:
        res = [-1.0]
        stack = [cars[-1]+[float('inf')]]  # 位置，速度，时间（表示时间点，也可以用有效线段的宽度表达）
        # 这里用时间点更方便一些，这样不用修改位置
        for y, v in reversed(cars[:-1]):  # 保证起始位置倒序
            tt = float('inf')
            while stack:
                yp, vp, tp = stack[-1]  # 取上一个线段（截距，斜率，x轴投影宽度）
                if v > vp:  # 能追上（斜率大）
                    tt = (yp-y)/(v-vp)  # 计算追逐时间
                    if tt < tp:  # 如果能在有效时间内追上，
                        res.append(tt)
                        break
                    else:
                        stack.pop()
                else:
                    stack.pop()  # 追不上，那可能追上前面更慢的
            if tt == float('inf'): res.append(-1.0)
            stack.append([y, v, tt])
        return res[::-1]


class Contest_229:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        """ 5685. 交替合并字符串
        AC
        """
        res = []
        for i, j in zip(word1, word2):
            res.append(i)
            res.append(j)
        res.append(word1[len(word2):])
        res.append(word2[len(word1):])
        return ''.join(res)

    def minOperations(self, boxes: str) -> List[int]:
        """ 5686. 移动所有球到每个盒子所需的最小操作数

        AC
            一开始代码考虑滑动窗口，但是测试没写对，看到很多人提交了，就还是用暴力法试试了。
            回头考虑窗口的话，是需要记录i位置左边的1的数量，和右边的1的数量，然后计算一遍总数，之后差分。
        """
        a = [i for i in range(len(boxes)) if boxes[i] == '1']
        res = [0]*len(boxes)
        for i in range(len(boxes)):
            res[i] = sum(abs(i-j) for j in a)
        return res

    def maximumScore(self, nums: List[int], multipliers: List[int]) -> int:
        """ 5687. 执行乘法运算的最大分数

        AC + Error2
            dfs超时，所以重写dp，dp下标有点麻烦，但是测试案例过了，基本就没问题了。
        """
        # # dfs 算法，超时了
        # res = -100000000000
        # def dfs(lst, mul, score):
        #     nonlocal res
        #     if not mul:
        #         res = max(res, score)
        #     else:
        #         dfs(lst[1:], mul[1:], score+lst[0]*mul[0])
        #         dfs(lst[:-1], mul[1:], score+lst[-1]*mul[0])
        #
        # dfs(nums, multipliers, 0)
        # return res

        # # 动态规划的方法，还是写一下状态方程再编码比较好
        n, m = len(nums), len(multipliers)
        dp = [[0]*(m+1) for _ in range(m+1)]  # dp[i][j] 表示左边取i个，右边取j个
        for dep in range(1, m+1):
            dp[dep][0] = dp[dep-1][0]+multipliers[dep-1]*nums[dep-1]
            dp[0][dep] = dp[0][dep-1]+multipliers[dep-1]*nums[-dep]
            for i in range(1, dep):
                r = dep-i
                dp[i][r] = max(dp[i][r-1]+multipliers[dep-1]*nums[-r],
                               dp[i-1][r]+multipliers[dep-1]*nums[i-1])
        return max(dp[i][m-i] for i in range(m))

    def longestPalindrome(self, word1: str, word2: str) -> int:
        """ 5688. 由子序列构造的最长回文串的长度

        Fail
            想想用dp，但是比赛时放弃了。

        """
        n1, n2 = len(word1), len(word2)
        n = n1+n2
        word = word1+word2
        res = 0
        dp = [[0]*n for _ in range(n)]
        for i in range(n): dp[i][i] = 1
        for i in range(1, n):
            for j in range(i-1, -1, -1):
                if word[i] == word[j]:
                    dp[j][i] = dp[j+1][i-1]+2
                    if j < n1 <= i:
                        res = max(res, dp[j][i])
                else:
                    dp[j][i] = max(dp[j+1][i], dp[j][i-1])
        return res


class Contest_Weekly_46:
    def longestNiceSubstring(self, s: str) -> str:
        """ 5668. 最长的美好子字符串

        s = "YazaAay"
        输出："aAa"
        解释："aAa" 是一个美好字符串，因为这个子串中仅含一种字母，其小写形式 'a' 和大写形式 'A' 也同时出现了。
        "aAa" 是最长的美好子字符串。

        输入：s = "dDzeE"
        输出："dD"
        解释："dD" 和 "eE" 都是最长美好子字符串。由于有多个美好子字符串，返回 "dD" ，因为它出现得最早。

        AC
            用滑动窗口，但是比赛字符串长度不超过100，很小，就直接用遍历了

        """

        def _check(s):
            for i in s:
                if i.swapcase() not in s:
                    return False
            return True

        for i in range(len(s), 1, -1):
            for j in range(len(s)-i):
                if _check(s[j:j+i]): return s[j:j+i]
        return ''

    def canChoose(self, groups: List[List[int]], nums: List[int]) -> bool:
        """ 5669. 通过连接另一个数组的子数组得到一个数组

        AC
            数据量不大，直接模拟，python取数组是方便
        """
        idx, n = 0, len(nums)
        for i in range(len(groups)):
            while idx < len(nums):
                if groups[i] == nums[idx:idx+len(groups[i])]:
                    idx += len(groups[i])
                    break
                idx += 1
            else:
                return False
        return True

    def highestPeak(self, isWater: List[List[int]]) -> List[List[int]]:
        """ 5671. 地图中的最高点

        AC + Error 2
            思路清晰，用队列
            error1初始化弄错了，二位矩阵res初始化维度不对，测试数据都是方阵
            error2对要扩展的点先入队列，出队后赋值，会造成重复入队列，所以代码复杂，超时了。
            如果改成入队前赋值，则代码简单，队列变短，就ok了。

        """
        dd = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        m, n = len(isWater), len(isWater[0])
        res = [[-1]*n for _ in range(m)]
        q = deque()

        for i in range(m):
            for j in range(n):
                if isWater[i][j] == 1:
                    q.append([i, j])
                    res[i][j] = 0
        while q:
            i, j = q.popleft()
            for dx, dy in dd:
                ii, jj = i+dx, j+dy
                if 0 <= ii < m and 0 <= jj < n and res[ii][jj] == -1:
                    res[ii][jj] = res[i][j]+1
                    q.append([ii, jj])
        return res

    @interesting
    def getCoprimes(self, nums: List[int], edges: List[List[int]]) -> List[int]:
        """ 5670. 互质树

        无环连通无向图（n个节点n-1条边的树），求每个节点的互质的最大根节点。

        Fail
            用list记录每个节点的父亲，0是根节点，其他节点并不确定（error1没有建树）
            对于树的处理不熟练，建树用邻接矩阵才行（error2超时）
            虽然算法没问题了，但是n大到1e5，所以计算互质的根节点直接变例，复杂度接近N*N（error3超时）
                >>> def getCoprimes(self, nums: List[int], edges: List[List[int]]) -> List[int]:
                >>> @lru_cache
                >>> def gcd(a, b):
                >>>     return a if b == 0 else gcd(b, a%b)
                >>>
                >>> n = len(nums)
                >>> parent = [-2]*n
                >>> parent[0] = -1
                >>> dd = defaultdict(list)  # 建树
                >>> for i, j in edges:
                >>>     dd[i].append(j)
                >>>     dd[j].append(i)
                >>> que = deque([0])  # 检查父节点，构建拓扑结构
                >>> while que:
                >>>     x = que.popleft()
                >>>     for i in dd[x]:
                >>>         if parent[i] == -2:
                >>>             que.append(i)
                >>>             parent[i] = x
                >>>
                >>> ans = [-1]*n
                >>> for i in range(n):
                >>>     p = parent[i]
                >>>     while True:
                >>>         if p == -1 or gcd(nums[i], nums[p]) == 1: # 追溯祖先节点
                >>>             ans[i] = p
                >>>             break
                >>>         p = parent[p]
                >>> return ans

            参考解答：
                互质节点对的预处理（因为value在50以内），所以记录value下手，dfs算法记录每个value最近的祖先节点位置。
                这样，将祖先节点个数压缩成最多50个。

        """
        n = len(nums)
        dd = defaultdict(list)  # 关系
        for i, j in edges:
            dd[i].append(j)
            dd[j].append(i)

        coprime = [[] for _ in range(51)]
        for i in range(1, 51):
            for j in range(i, 51):
                if math.gcd(i, j) == 1:
                    coprime[i].append(j)
                    coprime[j].append(i)
        dep = [-1]*51  # dep[i]记录最近的数值为i的深度（配合栈stack使用），因为祖先最近，深度越大，但是编号不确定
        stack = []
        ans = [-1]*n  # ans[i]记录互质的最近的祖先编号

        def dfs(idx, val, parent):
            depth = max(dep[cop] for cop in coprime[val])  # 取深度最大的node对应的编号
            ans[idx] = stack[depth] if depth > -1 else -1  # 如果-1 表示没有互质的祖先
            tmp, dep[val] = dep[val], len(stack)  # 然后，更新最近祖先表dep
            stack.append(idx)
            for i in dd[idx]:
                if i != parent:
                    dfs(i, nums[i], idx)  # 树结构叶子节点就不会再扩展了
            dep[val] = tmp  # 回溯
            stack.pop()

        dfs(0, nums[0], -1)
        return ans


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

    @interesting
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


class Contest_Weekly_45:
    def sumOfUnique(self, nums: List[int]) -> int:
        """ 1748. 唯一元素的和

        给你一个整数数组 nums 。数组中唯一元素是那些只出现 恰好一次 的元素。
        请你返回 nums 中唯一元素的 和 。

        AC
        """
        return sum(i for i, v in Counter(nums).items() if v == 1)

    def maxAbsoluteSum(self, nums: List[int]) -> int:
        """ 1749. 任意子数组和的绝对值的最大值

        AC
            贪心，数学方法，觉得应该对的。赛后了解下`前缀和`，以及如何求没有绝对值的最大值和最小值。
            >>> max(accumulate(nums, initial=0)) - min(accumulate(nums, initial=0))
        """
        if len(nums) == 0: return 0
        a = [0]
        for i in nums:
            a.append(a[-1]+i)
        return abs(max(a)-min(a))

    def minimumLength(self, s: str) -> int:
        """ 1750. 删除字符串两端相同字符后的最短长度

        AC
            模拟题
        """
        if len(s) <= 1: return len(s)
        left, right = 0, len(s)-1
        while s[left] == s[right] and left < right:
            a = s[left]
            while s[left] == a:
                left += 1
                if left > right: return 0
            while s[right] == a:
                right -= 1
                if right < left: return 0
        # print(left, right)
        return right-left+1

    @interesting
    def maxValue(self, events: List[List[int]], k: int) -> int:
        """ 1751. 最多可以参加的会议数目 II

        给你一个 events 数组，其中 events[i] = [startDayi, endDayi, valuei] ，表示第 i 个会议在 startDayi 天开始，
        第 endDayi 天结束，如果你参加这个会议，你能得到价值 valuei 。同时给你一个整数 k 表示你能参加的最多会议数目。
        你同一时间只能参加一个会议。如果你选择参加某个会议，那么你必须 完整 地参加完这个会议。
        会议结束日期是包含在会议内的，也就是说你不能同时参加一个开始日期与另一个结束日期相同的两个会议。
        请你返回能得到的会议价值 最大和 。

        Fail
            考虑背包，根据结束日期，endday[i]描述第i天截止得到的价值最大和。但是10e6的背包量，超时了
            >>> n = len(events)
            >>> events.sort(key=lambda x:x[1])
            >>> res = 0
            >>> endday = {-1:[0]*(k+1)}
            >>> for st, en, val in events:
            >>>     if en in endday: tmp = endday[en]
            >>>     else: tmp = [0]*(k+1)
            >>>     for i in endday:
            >>>         if i < st:
            >>>             for j in range(1, k+1):
            >>>                 if endday[i][j-1]+val > tmp[j]:
            >>>                     tmp[j] = endday[i][j-1]+val
            >>>     endday[en] = tmp
            >>> return max(max(v) for v in endday.values())
            赛后来看，dp时只需要定位到最后一格i<st，执行一遍O(k)，就可以了。
            而定位，用遍历也不能AC，最好用二分查找。

        """
        n = len(events)
        events.sort(key=lambda x:x[1])
        # dp[i][j]表示第i天截至，参加k个项目的最大价值。dp[][j]是单调非降的。
        dp = [[0]*(k+1) for _ in range(n)]+[[0]+[-0xfffffff]*k]  # 哨兵
        endday = [i[1] for i in events]
        for i, (st, en, val) in enumerate(events):
            idx = bisect.bisect_left(endday, st)-1  # 比插入left位置小1，如果是0，则对应-1哨兵
            for j in range(k, 0, -1):  # 这里要倒序，因为是0-1背包
                dp[i][j] = max(dp[idx][j-1]+val, dp[i-1][j])
        # print(dp)
        return max(dp[-2])  # 倒数第二个，不超过k个


if __name__ == '__main__':
    res = Solution().getCoprimes(
            [5, 6, 10, 2, 3, 6, 15], [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]]
    )
    print(res)
