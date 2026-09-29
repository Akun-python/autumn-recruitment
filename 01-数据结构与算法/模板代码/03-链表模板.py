# -*- coding: utf-8 -*-
"""03-链表模板.py —— P0 必背（链表题 = 指针操作 + 画图）
配套教学：../教学/03-链表.ipynb
核心技巧：哑节点 dummy 处理头/尾；快慢指针；指针重连顺序。
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def to_list(arr):
    head = cur = None
    for v in arr:
        if head is None:
            head = ListNode(v)
            cur = head
        else:
            cur.next = ListNode(v)
            cur = cur.next
    return head


def to_array(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out


# ============ 1. 反转链表（迭代 + 递归双版本） ============
def reverse_list(head):
    """迭代：三指针 prev/cur/nxt"""
    prev, cur = None, head
    while cur:
        nxt = cur.next                    # 先存后继
        cur.next = prev                   # 反转指向前驱
        prev, cur = cur, nxt
    return prev


def reverse_list_rec(head):
    """递归：返回新头；head.next 的 next 指向 head"""
    if head is None or head.next is None:
        return head
    new_head = reverse_list_rec(head.next)
    head.next.next = head
    head.next = None
    return new_head


# ============ 2. 反转区间 [left, right]（206 升级，92 题） ============
def reverse_between(head, left, right):
    dummy = ListNode(-1, head)
    pre = dummy
    for _ in range(left - 1):             # 走到 left 前一个
        pre = pre.next
    cur = pre.next
    for _ in range(right - left):         # 头插法逐个翻到前面
        nxt = cur.next
        cur.next = nxt.next
        nxt.next = pre.next
        pre.next = nxt
    return dummy.next


# ============ 3. 环检测：快慢指针（141）+ 找环入口（142） ============
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            return True
    return False


def detect_cycle(head):
    """相遇后：慢指针从头与快指针同速走，再次相遇即环入口"""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            p = head
            while p is not slow:
                p = p.next
                slow = slow.next
            return p
    return None


# ============ 4. 合并两个有序链表（迭代 + 递归） ============
def merge_two(l1, l2):
    dummy = cur = ListNode(-1)
    while l1 and l2:
        if l1.val <= l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 or l2
    return dummy.next


def merge_two_rec(l1, l2):
    if not l1:
        return l2
    if not l2:
        return l1
    if l1.val <= l2.val:
        l1.next = merge_two_rec(l1.next, l2)
        return l1
    l2.next = merge_two_rec(l1, l2.next)
    return l2


# ============ 5. 倒数第 k 个节点 / 删除倒数第 k ============
def remove_nth_from_end(head, k):
    dummy = ListNode(-1, head)
    fast = slow = dummy
    for _ in range(k + 1):                # fast 先走 k+1 步
        fast = fast.next
    while fast:                           # slow 停在待删节点前
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next


# ============ 6. 链表排序：归并（O(n log n)，O(1) 额外空间） ============
def sort_list(head):
    if head is None or head.next is None:
        return head
    slow = fast = head
    while fast.next and fast.next.next:   # 快慢找中点
        slow = slow.next
        fast = fast.next.next
    right = sort_list(slow.next)          # 递归排序右半
    slow.next = None
    left = sort_list(head)                # 递归排序左半
    return merge_two(left, right)


# ============ 测试 ============
if __name__ == '__main__':
    assert to_array(reverse_list(to_list([1, 2, 3, 4]))) == [4, 3, 2, 1]
    assert to_array(reverse_list_rec(to_list([1, 2, 3, 4]))) == [4, 3, 2, 1]
    assert to_array(reverse_between(to_list([1, 2, 3, 4, 5]), 2, 4)) == [1, 4, 3, 2, 5]
    assert to_array(merge_two(to_list([1, 3]), to_list([2, 4]))) == [1, 2, 3, 4]
    assert to_array(remove_nth_from_end(to_list([1, 2, 3, 4, 5]), 2)) == [1, 2, 3, 5]
    assert to_array(sort_list(to_list([4, 2, 1, 3]))) == [1, 2, 3, 4]
    print('✅ 链表全家桶测试通过')