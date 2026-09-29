# -*- coding: utf-8 -*-
"""04-栈与队列模板.py —— P0/P1 必背（单调栈是面试重灾区）
配套教学：../教学/04-栈与队列.ipynb
思想：单调栈维护"候选"，把不可能成为答案的元素提前弹出 → 每个元素进出一次 O(n)。
"""


# ============ 1. 单调栈：下一个更大元素（739/496 原型） ============
def next_greater(nums):
    """返回每个元素右边第一个比它大的值（没有则 -1）"""
    n = len(nums)
    out = [-1] * n
    stack = []                            # 存下标，栈内按下标对应值单调递减
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:
            out[stack.pop()] = x          # x 是栈顶的下一个更大元素
        stack.append(i)
    return out


# ============ 2. 单调栈：每日温度（739）等待天数 ============
def daily_temperatures(temps):
    n = len(temps)
    out = [0] * n
    stack = []
    for i, t in enumerate(temps):
        while stack and temps[stack[-1]] < t:
            j = stack.pop()
            out[j] = i - j                # 天数差
        stack.append(i)
    return out


# ============ 3. 单调栈：最大矩形（84，稍难，背结论） ============
def largest_rectangle(heights):
    """每个柱子向左右找第一个更矮的 → 左右边界 → 面积
    技巧：左右各一趟单调栈，或一趟内用下标差。"""
    n = len(heights)
    left = [0] * n                        # left[i]: 左边第一个更矮的
    stack = []
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] >= h:
            stack.pop()
        left[i] = stack[-1] if stack else -1
        stack.append(i)
    right = [0] * n                       # right[i]: 右边第一个更矮的
    stack = []
    for i in range(n - 1, -1, -1):
        while stack and heights[stack[-1]] >= heights[i]:
            stack.pop()
        right[i] = stack[-1] if stack else n
        stack.append(i)
    return max(h * (right[i] - left[i] - 1) for i, h in enumerate(heights)) if heights else 0


# ============ 4. 单调队列：滑动窗口最大值（双端队列） ============
from collections import deque


def max_in_window(nums, k):
    dq = deque()                          # 存下标，对应值单调递减
    out = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:   # 队尾小的淘汰
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:                # 队首出窗
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out


# ============ 5. 栈实现队列 / 队列实现栈 ============
class MyQueue:
    """双栈：入栈 + 出栈转移，均摊 O(1)"""
    def __init__(self):
        self.sin, self.sout = [], []

    def push(self, x):
        self.sin.append(x)

    def pop(self):
        self._move()
        return self.sout.pop()

    def _move(self):
        if not self.sout:
            while self.sin:
                self.sout.append(self.sin.pop())


class MyStack:
    """单队列：入队后把前面 n-1 个循环挪到队尾"""
    def __init__(self):
        self.q = deque()

    def push(self, x):
        self.q.append(x)
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self):
        return self.q.popleft()


# ============ 6. 有效括号（20）栈经典 ============
def is_valid(s):
    pair = {')': '(', ']': '[', '}': '{'}
    stack = []
    for ch in s:
        if ch in pair:
            if not stack or stack.pop() != pair[ch]:
                return False
        else:
            stack.append(ch)
    return not stack


# ============ 测试 ============
if __name__ == '__main__':
    assert next_greater([2, 1, 2, 4, 3]) == [4, 2, 4, -1, -1]
    assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert largest_rectangle([2, 1, 5, 6, 2, 3]) == 10
    assert max_in_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert is_valid('()[]{}') and not is_valid('(]')
    q = MyQueue(); q.push(1); q.push(2)
    assert q.pop() == 1
    print('✅ 栈与队列模板测试通过')