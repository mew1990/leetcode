# -*- coding: utf-8 -*-
# @Author  : mew
from typing import *


# Definition for singly-linked list.
# ... from leetcode
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class myList:
    """ 自定义单向链表
    >>> a = myList([1, 2, 3, 4, 5])
    >>> a.to_list()
    [1, 2, 3, 4, 5]

    """

    def __init__(self, lst: list or ListNode = None):
        self._head = ListNode()
        self._tail = self._head

        if isinstance(lst, list):
            for i in lst:
                self._tail.next = ListNode(i)
                self._tail = self._tail.next
        if isinstance(lst, ListNode):
            self._tail.next = lst
            while self._tail.next: self._tail = self._tail.next

    @property
    def first_node(self):
        return self._head.next

    def __iter__(self, cur: ListNode = None) -> ListNode:
        if cur is None: cur = self._head.next
        while cur:
            yield cur
            cur = cur.next

    def to_list(self):
        return [i.val for i in self]


from functools import lru_cache
import collections


class LRUCache():
    """ 定义 LRU 缓存机制，基于OrderedDict
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.ordered_dict = collections.OrderedDict()

    def get(self, key: int) -> int:
        if key in self.ordered_dict:
            self.ordered_dict.move_to_end(key)
            return self.ordered_dict[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.ordered_dict:
            self.ordered_dict.move_to_end(key)
        self.ordered_dict[key] = value
        if len(self.ordered_dict) > self.capacity:
            self.ordered_dict.popitem(last=False)


class DLinkedNode:
    def __init__(self, key=0, val=0):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class myLRUCache():
    """ 注意，除了用字典，为什么用双向链表？因为要达到O(1)的效果，可以省去了查找！！！
    """

    def __init__(self, capacity: int):
        self.cache = {}  # 字典，存储key
        self.head = DLinkedNode()  # 双向链表
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            self.move_to_head(node)  # 这里能达到O(1)效果，是因为用了双向链表，直接定位了node的前后
            return node.val
        else:
            return -1

    def put(self, key: int, val: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.val = val
            self.move_to_head(node)
        else:
            node = DLinkedNode(key, val)
            self.cache[key] = node
            self.add_to_head(node)
            self.size += 1
            if self.size > self.capacity:
                removed = self.remove_tail()
                self.cache.pop(removed.key)
                self.size -= 1

    def add_to_head(self, node):
        self.head.next.prev, node.next = node, self.head.next
        self.head.next, node.prev = node, self.head

    def _remove(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def move_to_head(self, node):
        self._remove(node)
        self.add_to_head(node)

    def remove_tail(self):
        removed = self.tail.prev
        self._remove(removed)
        return removed
