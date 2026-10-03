# -*- coding: utf-8 -*-
"""04-栈与队列模板.py —— P0/P1 必背（单调栈是面试重灾区）
配套教学：../教学/04-栈与队列.ipynb
思想：单调栈维护"候选"，把不可能成为答案的元素提前弹出 → 每个元素进出一次 O(n)。

单调栈要点：
  - 栈内存"下标"，栈内对应值保持单调（递增/递减）；
  - 新元素到来时，把栈顶"不如它"的元素全部弹出（它们失去了成为答案的机会）；
  - 这样每个元素最多入栈、出栈各一次，整体 O(n)。
"""


# ============ 1. 单调栈：下一个更大元素（739/496 原型） ============
def next_greater(nums):
    """返回每个元素右边第一个比它大的值，没有则 -1（LeetCode 496）。

    维护一个"值单调递减"的栈（存下标）：
      - 从左到右遍历，当前值 x 若比栈顶值大，则 x 就是栈顶的"下一个更大元素"；
      - 弹出栈顶记录答案后继续比，直到栈顶值 >= x 或栈空；
      - x 入栈（它可能成为后面元素的"下一个更大元素"）。

    变量说明：
      - n: 数组长度；out: 答案数组（初始 -1）
      - stack: 单调递减栈，存下标
      - i / x: 当前下标 / 当前值

    循环终止条件：
      - 外层 for 遍历完所有元素；内层 while 在栈空或栈顶值 >= x 时停止。

    复杂度：
      - 时间 O(n)（每个元素进出栈一次），空间 O(n)。
    """
    n = len(nums)                      # 数组长度
    out = [-1] * n                     # 答案默认 -1（右边没有更大的）
    stack = []                         # 单调递减栈，存下标，栈顶是当前最小
    for i, x in enumerate(nums):       # 从左到右遍历
        while stack and nums[stack[-1]] < x:  # 栈顶值小于 x
            out[stack.pop()] = x       # x 就是栈顶的下一个更大元素 → 记录
        stack.append(i)                # 当前下标入栈（等待被"更大值"解出）
    return out


# ============ 2. 单调栈：每日温度（739）等待天数 ============
def daily_temperatures(temps):
    """每天要等几天才有更高温度（LeetCode 739）。

    与 next_greater 思路一致，只是记录"下标差"而不是值。

    变量说明：
      - temps: 温度数组
      - out: 答案数组（等待天数，初始 0）
      - stack: 单调递减栈，存下标
      - i / t: 当前下标 / 当前温度；j: 被弹出的栈顶下标

    循环终止条件：
      - 外层 for 遍历完；内层 while 在栈空或栈顶温度 >= 当前温度时停止。

    复杂度：
      - 时间 O(n)，空间 O(n)。
    """
    n = len(temps)                     # 天数
    out = [0] * n                      # 等待天数，默认 0（之后没有更高温）
    stack = []                         # 单调递减栈，存下标
    for i, t in enumerate(temps):      # 遍历每一天
        while stack and temps[stack[-1]] < t:  # 当前温度高于栈顶那天的温度
            j = stack.pop()            # 栈顶那天的答案确定了
            out[j] = i - j             # 等待天数 = 下标差
        stack.append(i)                # 当前天入栈，等待更高温出现
    return out


# ============ 3. 单调栈：最大矩形（84，稍难，背结论） ============
def largest_rectangle(heights):
    """柱状图中最大的矩形（LeetCode 84）。

    核心：对每根柱子 h，向左右各找到"第一个比它矮"的柱子，它们之间
    就是 h 能作为高的最大宽度区间 → 面积 = h × (right-left-1)。

    - left[i]: i 左边第一个高度 < h 的下标（没有则为 -1）；
    - right[i]: i 右边第一个高度 < h 的下标（没有则为 n）。
    两趟单调栈即可求出 left 与 right。

    变量说明：
      - heights: 柱高数组；n: 柱子数量
      - left / right: 左右最近更矮柱子的下标数组
      - stack: 单调递增栈（严格递增，存下标）
      - i / h: 当前下标 / 当前高度

    过程拆解：
      1. 左扫：维护递增栈，弹出所有 >= h 的（它们高度 >= h，不能当"更矮"界），
         栈顶（若有）就是左边第一个更矮的；h 入栈；
      2. 右扫（从右往左）：同样维护递增栈，得到右边第一个更矮的；
      3. 对每根柱子计算 h * (right[i]-left[i]-1)，取最大。

    循环终止条件：
      - 两趟扫描各遍历一遍数组。

    复杂度：
      - 时间 O(n)，空间 O(n)。
    """
    n = len(heights)                   # 柱子数量
    left = [0] * n                     # left[i]: 左边第一个更矮的柱子下标
    stack = []                         # 单调递增栈（存下标）
    for i, h in enumerate(heights):    # 从左往右扫
        while stack and heights[stack[-1]] >= h:  # 栈顶高度 >= h → 不是"更矮"
            stack.pop()                # 弹出（它们不可能作为 h 的左边界）
        left[i] = stack[-1] if stack else -1  # 栈顶是左边第一个更矮的，没有则 -1
        stack.append(i)                # 当前柱子入栈
    right = [0] * n                    # right[i]: 右边第一个更矮的柱子下标
    stack = []                         # 清空栈再来一趟
    for i in range(n - 1, -1, -1):     # 从右往左扫
        while stack and heights[stack[-1]] >= heights[i]:  # 同样弹出不矮的
            stack.pop()
        right[i] = stack[-1] if stack else n  # 栈顶是右边第一个更矮的，没有则 n
        stack.append(i)
    # 对每根柱子：宽度 = right-left-1，面积 = 高 × 宽，取全局最大
    return max(h * (right[i] - left[i] - 1) for i, h in enumerate(heights)) if heights else 0


# ============ 4. 单调队列：滑动窗口最大值（双端队列） ============
from collections import deque


def max_in_window(nums, k):
    """每个长度为 k 的滑动窗口的最大值（LeetCode 239），O(n)。

    与 max_sliding_window 完全相同（不同文件名下的同一模板，便于检索）。
    双端队列 dq 存下标，对应值单调递减：队首永远是当前窗口最大值。

    变量说明：
      - dq: 单调递减双端队列（存下标）；out: 结果列表
      - i / x: 当前下标 / 当前值

    过程拆解：
      1. 新元素 x 入队前，弹出队尾所有 <= x 的下标（它们当不了最大值）；
      2. x 入队（可能在队首大值滑出后成为最大值）；
      3. 队首若已滑出窗口（dq[0] <= i-k）则弹出；
      4. 窗口满（i >= k-1）时队首就是当前窗口最大值。

    循环终止条件：
      - 遍历完所有元素。

    复杂度：
      - 时间 O(n)，空间 O(k)。
    """
    dq = deque()                       # 存下标，队首→队尾对应值单调递减
    out = []                           # 结果
    for i, x in enumerate(nums):       # 遍历每个元素
        while dq and nums[dq[-1]] <= x:  # 队尾比新值小 → 永远当不了最大
            dq.pop()                   # 淘汰队尾
        dq.append(i)                   # 新元素下标入队
        if dq[0] <= i - k:             # 队首下标滑出窗口
            dq.popleft()               # 弹出队首
        if i >= k - 1:                 # 窗口长度达到 k
            out.append(nums[dq[0]])    # 队首即最大值
    return out


# ============ 5. 栈实现队列 / 队列实现栈 ============
class MyQueue:
    """用两个栈实现队列（LeetCode 232），均摊 O(1)。

    双栈思路：
      - sin 是"入队栈"：push 直接压入 sin；
      - sout 是"出队栈"：pop 时若 sout 为空，把 sin 全部倒进 sout
        （栈顶变栈底，顺序正好反转 → 先进先出）。

    变量说明：
      - sin: 入队栈；sout: 出队栈
      - x: 入队元素

    复杂度：
      - push O(1)；pop 均摊 O(1)（每个元素最多被搬一次）。
    """
    def __init__(self):
        self.sin, self.sout = [], []   # 入队栈 / 出队栈

    def push(self, x):
        """入队：直接压入入队栈。"""
        self.sin.append(x)

    def pop(self):
        """出队：确保出队栈非空后弹出栈顶（即队首）。"""
        self._move()                   # 若 sout 空则搬运 sin → sout
        return self.sout.pop()

    def _move(self):
        """把 sin 的元素全部搬到 sout（顺序反转 → 先进先出）。"""
        if not self.sout:              # 只在 sout 为空时搬运
            while self.sin:            # 逐个弹出 sin 压入 sout
                self.sout.append(self.sin.pop())


class MyStack:
    """用单队列实现栈（LeetCode 225）。

    单队列思路：push 时把新元素放到队尾，再把前面 n-1 个元素循环挪到队尾，
    让新元素"插队"到队首 → pop 直接取队首就是后进先出。

    变量说明：
      - q: 存储队列（deque）
      - x: 入栈元素

    复杂度：
      - push O(n)，pop O(1)。
    """
    def __init__(self):
        self.q = deque()               # 存储队列

    def push(self, x):
        """入栈：新元素进队尾，然后把前面的元素循环移到队尾后面。"""
        self.q.append(x)               # 新元素先进队尾
        for _ in range(len(self.q) - 1):  # 循环搬前面 n-1 个
            self.q.append(self.q.popleft())  # 队首搬到队尾 → 新元素逐步到队首

    def pop(self):
        """出栈：直接取队首（它就是最后入栈的元素）。"""
        return self.q.popleft()


# ============ 6. 有效括号（20）栈经典 ============
def is_valid(s):
    """判断括号字符串是否合法（LeetCode 20）。

    栈匹配：遇到左括号入栈；遇到右括号时，栈顶必须是配对的左括号，
    否则不合法。全部扫描完栈必须为空。

    变量说明：
      - pair: 右括号 → 对应左括号 的映射
      - stack: 存放未匹配的左括号
      - ch: 当前字符

    过程拆解：
      1. ch 是右括号：栈空（没有左括号可配对）或栈顶不是对应左括号 → False；
         配对成功则弹出栈顶；
      2. ch 是左括号：入栈；
      3. 扫描结束，栈空才合法（否则有左括号没配对）。

    循环终止条件：
      - 遍历完整个字符串；或提前发现不匹配返回 False。

    复杂度：
      - 时间 O(n)，空间 O(n)。
    """
    pair = {')': '(', ']': '[', '}': '{'}  # 右括号 → 左括号
    stack = []                         # 未匹配的左括号栈
    for ch in s:                       # 逐个字符处理
        if ch in pair:                 # 遇到右括号
            if not stack or stack.pop() != pair[ch]:  # 栈空或不匹配
                return False          # 不合法
        else:                         # 遇到左括号
            stack.append(ch)          # 入栈等待配对
    return not stack                   # 栈空 → 所有括号都配对完成


# ============ 测试 ============
if __name__ == '__main__':
    assert next_greater([2, 1, 2, 4, 3]) == [4, 2, 4, -1, -1]
    assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert largest_rectangle([2, 1, 5, 6, 2, 3]) == 10        # 高度 5 的矩形最宽
    assert max_in_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert is_valid('()[]{}') and not is_valid('(]')          # 合法 / 不合法
    q = MyQueue(); q.push(1); q.push(2)                        # 双栈队列
    assert q.pop() == 1                                        # 先进先出
    print('✅ 栈与队列模板测试通过')
