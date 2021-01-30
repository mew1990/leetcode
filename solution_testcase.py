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


if __name__ == '__main__':
    p20().run()
    p21().run()
    p19().run()
