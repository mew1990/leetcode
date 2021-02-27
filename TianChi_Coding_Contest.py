# -*- coding: utf-8 -*-
# @Author  : mew

class Contest_58:
    """ 2020-02-27
        1.能否转换 简单 1星
        2.卡牌游戏 中等 2星
        3.子序列宽度之和 困难 3星
        4.圣杯咒语 简单 1星
    """

    def canConvert(self, s, t):
        # 字符串匹配
        if t == '': return True
        pos = 0
        for i in s:
            if i == t[pos]:
                pos += 1
                if pos == len(t):
                    return True
        return False

    def cardGame(self, cost, damage, totalMoney, totalDamage):
        # 01背包，数据情况未知，应该规模不大
        dp = [0]*(totalMoney+1)
        for i in range(len(cost)):
            for j in range(totalMoney, cost[i]-1, -1):
                if dp[j-cost[i]]+damage[i] > dp[j]:
                    dp[j] = dp[j-cost[i]]+damage[i]
        return max(dp) >= totalDamage

    def sumSubseqWidths(self, A):
        # 数学公式推导, width 忘记乘了，error1
        n = len(A)
        if n == 1: return 0
        A.sort()
        res = 0
        for i in range(1, n):
            width = A[i]-A[i-1]
            if width:
                res += ((1<<n)-(1<<i)-(1<<(n-i))+1)*width
                res %= 1000000007
        return res

    def holyGrailspell(self, Spell):
        # 字符串处理
        a = ''.join(sorted(set(Spell)))
        for i in reversed(a):
            if i.swapcase() in a:
                return i.upper()
