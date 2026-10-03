# -*- coding: utf-8 -*-
"""03-链表模板.py —— P0 必背（链表题 = 指针操作 + 画图）
配套教学：../教学/03-链表.ipynb
核心技巧：哑节点 dummy 处理头/尾；快慢指针；指针重连顺序。

链表题最容易踩坑的是"指针顺序"：
  - 先保存后继（nxt），再改 next，否则会丢失后面的节点；
  - 需要动头部时用哑节点 dummy，让头部也能像普通节点一样处理。
"""


class ListNode:
    """单链表节点。

    变量说明：
      - val: 节点值
      - next: 指向下一个节点的引用（None 表示链表末尾）
    """
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def to_list(arr):
    """把 Python 列表 arr 转成链表，返回头节点（空列表返回 None）。

    变量说明：
      - head: 链表头节点；cur: 当前链表的尾节点
      - v: 当前要插入的值

    过程拆解：
      1. head 为 None 时创建第一个节点（同时 head 和 cur 都指向它）；
      2. 之后每次 cur.next 挂上新节点，并把 cur 移到新节点；
      3. 遍历结束返回头节点。

    循环终止条件：
      - 遍历完整个数组。

    复杂度：
      - 时间 O(n)，空间 O(n)（新建节点）。
    """
    head = cur = None                 # 头节点和尾节点初始为空
    for v in arr:                     # 逐个值建节点
        if head is None:              # 第一个节点
            head = ListNode(v)        # 它同时是头节点
            cur = head                # 尾节点也指向它
        else:                         # 之后的节点
            cur.next = ListNode(v)    # 挂到当前尾部后面
            cur = cur.next            # 尾节点后移
    return head


def to_array(head):
    """把链表转回 Python 列表（用于测试/打印）。

    变量说明：
      - head: 从头开始遍历的当前节点（遍历中不断后移）
      - out: 结果列表

    循环终止条件：
      - head 变为 None（走到链表末尾）。

    复杂度：
      - 时间 O(n)，空间 O(n)（存结果）。
    """
    out = []                          # 结果列表
    while head:                       # 当前节点非空就继续
        out.append(head.val)          # 收集节点值
        head = head.next              # 移动到下一个节点
    return out


# ============ 1. 反转链表（迭代 + 递归双版本） ============
def reverse_list(head):
    """迭代反转链表（LeetCode 206），返回新头节点。

    三指针 prev / cur / nxt：
      - prev: 已反转部分的头（即当前节点的前驱）
      - cur: 当前正在处理的节点
      - nxt: 暂存 cur 的后继（否则改完 next 就找不到了）

    过程拆解：
      1. cur 从 head 开始，prev 初始为 None（新链表的终点）；
      2. 每步：先存 nxt → 把 cur.next 指向前驱 prev（完成反转）→
         prev、cur 同时前移一步；
      3. 循环结束（cur 为 None）时 prev 正好停在原链表末尾 → 新链表头。

    循环终止条件：
      - cur 为 None（遍历完整个链表）。

    复杂度：
      - 时间 O(n)，空间 O(1)。
    """
    prev, cur = None, head            # prev 是反转后的前驱，cur 是当前节点
    while cur:                        # 还有节点要反转
        nxt = cur.next                # 先存后继，防止丢失
        cur.next = prev               # 当前节点指向前驱 → 完成一步反转
        prev, cur = cur, nxt          # 两个指针整体前移
    return prev                       # prev 现在是反转后的头


def reverse_list_rec(head):
    """递归反转链表（LeetCode 206 递归版），返回新头节点。

    思路：先反转 head.next 之后的部分，再把 head 接到新链表的末尾。
    递归返回的是"反转后的新头"。

    变量说明：
      - head: 当前子问题的头节点
      - new_head: 递归反转 head.next 后得到的新的头节点

    过程拆解：
      1. 递归基：head 为空或只剩一个节点，直接返回（无需反转）；
      2. 递归调用 reverse_list_rec(head.next)，拿到反转后的新头 new_head；
      3. 反转后 head 是链表的"最后一个节点"，此时 head.next 仍指向原来的
         下一个节点：让 head.next.next = head（把 head 接到它后面）；
      4. head.next = None 断掉原来的正向链接，防止成环；
      5. 返回 new_head（真正的头）。

    递归终止条件：
      - head 为 None 或 head.next 为 None（单个节点）时停止。

    复杂度：
      - 时间 O(n)，空间 O(n)（递归调用栈深度 n）。
    """
    if head is None or head.next is None:  # 空链表或单节点：无需反转
        return head
    new_head = reverse_list_rec(head.next)  # 先反转后面整段，拿到新头
    head.next.next = head             # 让原来下一个节点的 next 指回 head
    head.next = None                  # head 变为新链表的末尾，断开旧链接
    return new_head                   # 返回新头


# ============ 2. 反转区间 [left, right]（206 升级，92 题） ============
def reverse_between(head, left, right):
    """反转链表中从 left 到 right 的区间（1-based 下标），其余不变。

    思路：哑节点 + 头插法。
      - pre 停在 left 前一个节点（区间外部，永不移动）；
      - cur 是区间内当前节点；nxt 是 cur 的后继；
      - 每次把 nxt "头插"到 pre 后面，逐步把区间倒过来。

    变量说明：
      - dummy: 哑节点，next 指向 head（统一处理 left=1 的情况）
      - pre: left 的前一个节点
      - cur: 区间内当前节点
      - nxt: cur 的下一个节点（将被搬到区间最前面）

    过程拆解：
      1. pre 走 left-1 步到达 left 的前一个节点；
      2. cur = pre.next（区间第一个节点）；
      3. 执行 right-left 次头插：
         - nxt = cur.next；
         - cur.next = nxt.next（把 nxt 从中间摘出来）；
         - nxt.next = pre.next（nxt 插到区间最前）；
         - pre.next = nxt（pre 直接指向新的区间头）；
         —— 每做一次，区间头就往后退一个，恰好 right-left 次完成反转。

    循环终止条件：
      - 头插循环执行 right-left 次后结束。

    复杂度：
      - 时间 O(right)，空间 O(1)。
    """
    dummy = ListNode(-1, head)         # 哑节点：left=1 时也能统一处理
    pre = dummy                        # pre 最后停在 left 前一个
    for _ in range(left - 1):          # 走 left-1 步
        pre = pre.next                 # pre 前进到 left 前一个节点
    cur = pre.next                     # cur 指向区间第一个节点
    for _ in range(right - left):      # 头插法：逐个把节点搬到区间最前
        nxt = cur.next                 # 先保存后继
        cur.next = nxt.next            # cur 跳过 nxt 直接连到 nxt 后面
        nxt.next = pre.next            # nxt 指向区间原头 → 成为新头
        pre.next = nxt                 # pre 挂上 nxt
    return dummy.next                  # 返回真正的头（跳过哑节点）


# ============ 3. 环检测：快慢指针（141）+ 找环入口（142） ============
def has_cycle(head):
    """判断链表是否有环（LeetCode 141）。

    快慢指针：fast 每次走 2 步，slow 每次走 1 步。
    - 无环：fast 会先走到 None；
    - 有环：fast 每轮追上 slow 一步，必然相遇（快慢指针同向追击）。

    变量说明：
      - slow / fast: 慢指针（1 步）/ 快指针（2 步）

    循环终止条件：
      - fast 或 fast.next 为 None（无环，遍历到头）；
      - slow is fast（有环，相遇）。

    复杂度：
      - 时间 O(n)，空间 O(1)。
    """
    slow = fast = head                 # 快慢指针都从头出发
    while fast and fast.next:          # fast 能走两步才继续（无环会走到头）
        slow = slow.next               # 慢指针走 1 步
        fast = fast.next.next          # 快指针走 2 步
        if slow is fast:               # 相遇 → 有环
            return True
    return False                       # fast 走到链表尾部 → 无环


def detect_cycle(head):
    """返回环的入口节点；无环返回 None（LeetCode 142）。

    数学结论：设头到环入口距离 a，环长 b。快慢指针在环内相遇时，
    slow 已走 s 步，fast 走 2s 步，且 2s - s = kb（s 是环长的整数倍）。
    此时把 slow 放回头部，fast 停在相遇点，二者同速各走 1 步，
    再次相遇的位置正好是环入口（距离恰为 a）。

    变量说明：
      - slow / fast: 快慢指针
      - p: 相遇后从头部出发的指针

    循环终止条件：
      - 第一段：fast 走到头（无环）或两指针相遇（有环）；
      - 第二段：p 与 slow 再次相遇（即环入口）。

    复杂度：
      - 时间 O(n)，空间 O(1)。
    """
    slow = fast = head                 # 快慢指针从头出发
    while fast and fast.next:          # 快指针能走两步才继续
        slow = slow.next               # 慢走 1 步
        fast = fast.next.next          # 快走 2 步
        if slow is fast:               # 第一次相遇 → 有环
            p = head                   # 新指针从头部出发
            while p is not slow:       # 同速走到再次相遇
                p = p.next             # p 每步 1
                slow = slow.next       # slow 每步 1（此时仍在环内）
            return p                   # 相遇点就是环入口
    return None                        # 无环


# ============ 4. 合并两个有序链表（迭代 + 递归） ============
def merge_two(l1, l2):
    """合并两个升序链表为一个升序链表（LeetCode 21 迭代版）。

    哑节点技巧：cur 每次接上两个链表头部较小的节点，然后前进。

    变量说明：
      - l1 / l2: 两个有序链表的当前节点
      - dummy / cur: 哑节点 / 新链表尾指针

    过程拆解：
      1. 循环里比较 l1.val 与 l2.val，把较小者接到 cur.next 后面；
      2. 对应链表指针后移，cur 也后移；
      3. 一个链表取完后，把另一个剩余部分整体接上。

    循环终止条件：
      - l1 或 l2 为空（一个链表取完）。

    复杂度：
      - 时间 O(n+m)，空间 O(1)。
    """
    dummy = cur = ListNode(-1)         # 哑节点作为新链表头
    while l1 and l2:                   # 两个链表都还有节点
        if l1.val <= l2.val:           # l1 头更小 → 接 l1
            cur.next = l1              # 把 l1 当前节点挂到新链表
            l1 = l1.next               # l1 后移
        else:                          # l2 头更小 → 接 l2
            cur.next = l2
            l2 = l2.next
        cur = cur.next                 # 新链表尾指针后移
    cur.next = l1 or l2                # 把剩余那段整体接上（None 也没关系）
    return dummy.next                  # 返回真正的头


def merge_two_rec(l1, l2):
    """合并两个升序链表（递归版，LeetCode 21）。

    思路：取两个头中较小的作为结果头，它的 next 递归地等于
    "较小头之后的链表"与另一链表的合并结果。

    变量说明：
      - l1 / l2: 两个有序链表的当前头节点

    递归终止条件：
      - l1 为空 → 结果就是 l2；
      - l2 为空 → 结果就是 l1。

    复杂度：
      - 时间 O(n+m)，空间 O(n+m)（递归栈深度）。
    """
    if not l1:                         # l1 空 → 直接返回 l2 剩余部分
        return l2
    if not l2:                         # l2 空 → 返回 l1 剩余部分
        return l1
    if l1.val <= l2.val:               # l1 头更小 → l1 作为结果头
        l1.next = merge_two_rec(l1.next, l2)  # 递归合并 l1.next 与 l2
        return l1
    l2.next = merge_two_rec(l1, l2.next)      # 否则 l2 作为结果头
    return l2


# ============ 5. 倒数第 k 个节点 / 删除倒数第 k ============
def remove_nth_from_end(head, k):
    """删除链表倒数第 k 个节点，返回新头（LeetCode 19）。

    双指针：fast 先走 k+1 步（指向待删节点的后一个），slow 从头出发，
    二者保持 k+1 的间距同步走；fast 走到末尾（None）时，slow 正好停在
    待删节点的前一个 → slow.next = slow.next.next 完成删除。

    变量说明：
      - dummy: 哑节点（统一处理删除头节点的情况）
      - fast / slow: 快慢指针
      - k: 倒数第几个（1-based）

    循环终止条件：
      - fast 为 None（走到链表尾部）。

    复杂度：
      - 时间 O(n)，空间 O(1)。
    """
    dummy = ListNode(-1, head)         # 哑节点，指向原头
    fast = slow = dummy                # 双指针都从哑节点出发
    for _ in range(k + 1):             # fast 先走 k+1 步
        fast = fast.next               # 走到待删节点的后一个
    while fast:                        # 双指针同步前进
        fast = fast.next               # fast 走到末尾时
        slow = slow.next               # slow 停在待删节点前一个
    slow.next = slow.next.next         # 跳过待删节点 → 完成删除
    return dummy.next                  # 返回新头（删除头节点也能正确处理）


# ============ 6. 链表排序：归并（O(n log n)，O(1) 额外空间） ============
def sort_list(head):
    """链表归并排序（LeetCode 148）：O(n log n) 时间，O(1) 额外空间。

    思路：快慢指针找中点 → 拆成两半 → 递归各自排序 → 归并。
    （数组归并需要 O(n) 辅助空间，链表只需改指针，所以是 O(1) 额外空间。）

    变量说明：
      - head: 当前待排序链表的头
      - slow / fast: 快慢指针，fast 走两步、slow 走一步 → slow 停在中点
      - left / right: 递归排序后的左右两半

    过程拆解：
      1. 递归基：空链表或单节点链表已有序；
      2. 快慢指针找中点：while fast.next and fast.next.next，
         fast 走两步、slow 走一步，结束时 slow 在前半段的最后一个节点；
      3. right = 对 slow.next 递归排序；slow.next = None 截断左半；
         left = 对 head 递归排序；
      4. 用 merge_two 合并左右两半。

    递归终止条件：
      - head 为 None 或 head.next 为 None。

    复杂度：
      - 时间 O(n log n)，空间 O(log n)（递归栈）额外空间（不复制节点）。
    """
    if head is None or head.next is None:  # 空/单节点已有序
        return head
    slow = fast = head                 # 快慢指针找中点
    while fast.next and fast.next.next:  # fast 能走两步就继续
        slow = slow.next               # 慢指针走一步
        fast = fast.next.next          # 快指针走两步
    right = sort_list(slow.next)       # 递归排序右半（slow.next 起）
    slow.next = None                   # 截断：把左半与右半断开
    left = sort_list(head)             # 递归排序左半
    return merge_two(left, right)      # 合并两个有序链表


# ============ 测试 ============
if __name__ == '__main__':
    assert to_array(reverse_list(to_list([1, 2, 3, 4]))) == [4, 3, 2, 1]
    assert to_array(reverse_list_rec(to_list([1, 2, 3, 4]))) == [4, 3, 2, 1]
    assert to_array(reverse_between(to_list([1, 2, 3, 4, 5]), 2, 4)) == [1, 4, 3, 2, 5]
    assert to_array(merge_two(to_list([1, 3]), to_list([2, 4]))) == [1, 2, 3, 4]
    assert to_array(remove_nth_from_end(to_list([1, 2, 3, 4, 5]), 2)) == [1, 2, 3, 5]
    assert to_array(sort_list(to_list([4, 2, 1, 3]))) == [1, 2, 3, 4]
    print('✅ 链表全家桶测试通过')
