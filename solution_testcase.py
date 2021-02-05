# -*- coding: utf-8 -*-
# @Author  : mew

from .solution import Solution
from .my_defs import *
import unittest


class p20(unittest.TestCase):
    method = Solution().isValid

    def test_case(self):
        s = "()[]{}"
        self.assertEqual(self.method(s), True)
        s = "("
        self.assertEqual(self.method(s), False)


class p21(unittest.TestCase):
    method = Solution().mergeTwoLists

    def test_case(self):
        l1 = myList([1, 2, 4])
        l2 = myList([1, 3, 4])
        res = myList([1, 1, 2, 3, 4, 4])
        for i, j in zip(myList(self.method(l1.first_node, l2.first_node)), res):
            self.assertEqual(i.val, j.val)


class p19(unittest.TestCase):
    method = Solution().removeNthFromEnd

    def _assert_equal(self, inputs, outputs):
        head, n = inputs
        res = outputs
        process = self.method(myList(head).first_node, n)
        self.assertEqual(myList(process).to_list(), res)

    def test_case(self):
        self._assert_equal(([1, 2, 3, 4, 5], 2), [1, 2, 3, 5])
        self._assert_equal(([1], 1), [])
        self._assert_equal(([1, 2], 1), [1])
        self._assert_equal(([1, 2], 2), [2])


class p34(unittest.TestCase):
    method = Solution().searchRange

    def _assert_equal(self, inputs, outputs):
        self.assertEqual(self.method(*inputs), outputs)

    def test_case(self):
        self._assert_equal(([5, 7, 7, 8, 8, 10], 8), [3, 4])
        self._assert_equal(([5, 7, 7, 8, 8, 10], 6), [-1, -1])
        self._assert_equal(([], 0), [-1, -1])


class p37(unittest.TestCase):
    method = Solution().solveSudoku

    def _assert_equal(self, inputs, outputs):
        self.method(inputs)
        self.assertEqual(inputs, outputs)

    def test_case(self):
        a = [["5", "3", ".", ".", "7", ".", ".", ".", "."],
             ["6", ".", ".", "1", "9", "5", ".", ".", "."],
             [".", "9", "8", ".", ".", ".", ".", "6", "."],
             ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
             ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
             ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
             [".", "6", ".", ".", ".", ".", "2", "8", "."],
             [".", ".", ".", "4", "1", "9", ".", ".", "5"],
             [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
        b = [["5", "3", "4", "6", "7", "8", "9", "1", "2"],
             ["6", "7", "2", "1", "9", "5", "3", "4", "8"],
             ["1", "9", "8", "3", "4", "2", "5", "6", "7"],
             ["8", "5", "9", "7", "6", "1", "4", "2", "3"],
             ["4", "2", "6", "8", "5", "3", "7", "9", "1"],
             ["7", "1", "3", "9", "2", "4", "8", "5", "6"],
             ["9", "6", "1", "5", "3", "7", "2", "8", "4"],
             ["2", "8", "7", "4", "1", "9", "6", "3", "5"],
             ["3", "4", "5", "2", "8", "6", "1", "7", "9"]]
        self._assert_equal(a, b)


class p39(unittest.TestCase):
    method = Solution().combinationSum

    def _assert_equal(self, inputs, outputs):
        res = self.method(*inputs)
        self.assertEqual(res, outputs)

    def test_case(self):
        self._assert_equal(([2, 3, 6, 7], 7),
                           [[2, 2, 3], [7]])


class p40(unittest.TestCase):
    method = Solution().combinationSum2

    def _assert_equal(self, inputs, outputs):
        res = self.method(*inputs)
        res.sort()
        outputs.sort()
        self.assertEqual(res, outputs)

    def test_case(self):
        self._assert_equal(([10, 1, 2, 7, 6, 1, 5], 8),
                           [[1, 1, 6], [1, 2, 5], [1, 7], [2, 6]])


if __name__ == '__main__':
    p20().run()
    p21().run()
    p19().run()
