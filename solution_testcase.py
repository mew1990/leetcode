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
        for i, j in zip(myList().__iter__(self.method(l1.first_node, l2.first_node)), res):
            self.assertEqual(i.val, j.val)


if __name__ == '__main__':
    p20().run()
    p21().run()
