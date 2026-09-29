# -*- coding: utf-8 -*-
"""06-回溯模板.py —— P0 必背（全排列/组合/子集三件套 + 去重 + 剪枝）
配套教学：../教学/07-回溯.ipynb
统一框架：选择 → 递归 → 撤销（backtrack 三步曲）。状态树画法见教学笔记。
"""


# ============ 1. 子集（78）：每个元素选/不选，结果 2^n ============
def subsets(nums):
    out = []
    path = []

    def bt(start):
        out.append(path[:])               # 每层都记录（含空集）
        for i in range(start, len(nums)):
            path.append(nums[i])
            bt(i + 1)                     # 不回头 → 组合去重天然成立
            path.pop()                    # 撤销
    bt(0)
    return out


# ============ 2. 组合（77）：n 选 k ============
def combine(n, k):
    out = []
    path = []

    def bt(start):
        if len(path) == k:
            out.append(path[:])
            return
        for i in range(start, n + 1):
            path.append(i)
            bt(i + 1)
            path.pop()
    bt(1)
    return out


# ============ 3. 全排列（46）：每层选未用元素 ============
def permute(nums):
    out = []
    path = []
    used = [False] * len(nums)

    def bt():
        if len(path) == len(nums):
            out.append(path[:])
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            used[i] = True
            path.append(x)
            bt()
            path.pop()
            used[i] = False
    bt()
    return out


# ============ 4. 去重模板（90/47）：排序 + 同层跳过 ============
def subsets_with_dup(nums):
    nums.sort()                           # 去重前提：排序
    out = []
    path = []

    def bt(start):
        out.append(path[:])
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:
                continue                  # 同层去重：跳过重复值
            path.append(nums[i])
            bt(i + 1)
            path.pop()
    bt(0)
    return out


def permute_unique(nums):
    nums.sort()
    out = []
    path = []
    used = [False] * len(nums)

    def bt():
        if len(path) == len(nums):
            out.append(path[:])
            return
        for i, x in enumerate(nums):
            if used[i]:
                continue
            if i > 0 and x == nums[i - 1] and not used[i - 1]:
                continue                  # 同层去重（前一个相同且未用）
            used[i] = True
            path.append(x)
            bt()
            path.pop()
            used[i] = False
    bt()
    return out


# ============ 5. 组合总和（39）：可重复选，剪枝 ============
def combination_sum(cands, target):
    cands.sort()
    out = []
    path = []

    def bt(start, rest):
        if rest == 0:
            out.append(path[:])
            return
        for i in range(start, len(cands)):
            if cands[i] > rest:
                break                     # 排序 + 剪枝：大过余量直接停
            path.append(cands[i])
            bt(i, rest - cands[i])        # 可重复 → i 不减
            path.pop()
    bt(0, target)
    return out


# ============ 6. 棋盘类：N 皇后（51，必背之一） ============
def solve_n_queens(n):
    out = []
    cols = [0] * n                        # cols[r] = 皇后所在列

    def ok(r, c):
        for pr in range(r):               # 检查前面所有行
            pc = cols[pr]
            if pc == c or abs(pc - c) == r - pr:
                return False              # 同列 / 对角线（|Δc|==|Δr|）
        return True

    def bt(r):
        if r == n:
            out.append(['.' * c + 'Q' + '.' * (n - 1 - c) for c in cols])
            return
        for c in range(n):
            if ok(r, c):
                cols[r] = c
                bt(r + 1)
    bt(0)
    return out


# ============ 测试 ============
if __name__ == '__main__':
    assert sorted(subsets([1, 2])) == [[], [1], [1, 2], [2]]
    assert combine(4, 2) == [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    assert permute([1, 2, 3])[0] == [1, 2, 3] and len(permute([1, 2, 3])) == 6
    assert len(subsets_with_dup([1, 2, 2])) == 6
    assert len(permute_unique([1, 1, 2])) == 3
    assert sorted(combination_sum([2, 3, 6, 7], 7)) == [[2, 2, 3], [7]]
    assert len(solve_n_queens(4)) == 2
    print('✅ 回溯全家桶测试通过')