# -*- coding: utf-8 -*-
"""06-回溯模板.py —— P0 必背（全排列/组合/子集三件套 + 去重 + 剪枝）
配套教学：../教学/07-回溯.ipynb
统一框架：选择 → 递归 → 撤销（backtrack 三步曲）。状态树画法见教学笔记。

回溯三步曲（每个递归函数都遵循）：
  1. 选择：把当前候选加入 path；
  2. 递归：进入下一层（start 控制不回头 / used 控制不重复）；
  3. 撤销：把刚加入的从 path 弹出，恢复现场，继续尝试下一个候选。
终止条件：递归到满足"答案长度/和"或候选穷尽时记录并返回。
"""


# ============ 1. 子集（78）：每个元素选/不选，结果 2^n ============
def subsets(nums):
    """返回 nums 的所有子集（LeetCode 78），共 2^n 个。

    思路：每个节点都记录当前 path（即一个子集）；用 start 保证
    只往后选（不回头），天然避免重复组合。

    变量说明：
      - out: 结果集合列表；path: 当前已选的子集
      - bt(start): 从下标 start 开始选元素
      - i: 本轮尝试选择的元素下标

    过程拆解：
      1. 每进入一层 bt，先把当前 path 的副本记入 out（含空集）；
      2. 依次尝试把 i >= start 的元素加入 path，递归下一层 bt(i+1)；
      3. 递归返回后撤销（path.pop），尝试下一个元素。

    递归终止条件：
      - start 到达 len(nums)（候选穷尽）时 for 循环自然结束。

    复杂度：
      - 时间 O(n·2^n)（共 2^n 个子集，每个拷贝 O(n)）；空间 O(n)（path+栈）。
    """
    out = []                           # 所有子集
    path = []                          # 当前路径

    def bt(start):
        out.append(path[:])            # 每层都记录当前子集（含空集）
        for i in range(start, len(nums)):  # 从 start 开始，只往后选
            path.append(nums[i])       # 选择
            bt(i + 1)                  # 递归：下一个只能从 i+1 开始（不回头）
            path.pop()                 # 撤销选择
    bt(0)                              # 从下标 0 开始
    return out


# ============ 2. 组合（77）：n 选 k ============
def combine(n, k):
    """返回 1..n 中所有长度为 k 的组合（LeetCode 77）。

    与子集几乎一样，只是多了"长度等于 k 就记录并返回"的终止条件。

    变量说明：
      - n / k: 数字范围 / 组合长度
      - out / path: 结果 / 当前组合
      - bt(start): 从 start 开始选；i: 本轮尝试的数字

    过程拆解：
      1. path 长度达到 k → 记录副本并返回（剪枝：不再往下选）；
      2. 否则从 start 到 n 依次选数字，递归 bt(i+1)，返回后撤销。

    递归终止条件：
      - len(path) == k（凑够长度，记录返回）；
      - start 超过 n（候选穷尽）。

    复杂度：
      - 时间 O(C(n,k)·k)；空间 O(k)（path+栈）。
    """
    out = []                           # 所有组合
    path = []                          # 当前组合

    def bt(start):
        if len(path) == k:             # 凑够 k 个 → 记录并返回
            out.append(path[:])
            return
        for i in range(start, n + 1):  # 从 start 到 n 选数字
            path.append(i)             # 选择
            bt(i + 1)                  # 递归：下一个更大
            path.pop()                 # 撤销
    bt(1)                              # 从 1 开始
    return out


# ============ 3. 全排列（46）：每层选未用元素 ============
def permute(nums):
    """返回 nums 的所有全排列（LeetCode 46），共 n! 个。

    与组合不同：排列讲究顺序，每层都要从头选，所以用 used 数组
    标记"本路径上已用过的下标"，没用过的才能选。

    变量说明：
      - used: 布尔数组，used[i] 表示下标 i 的元素是否已选
      - path: 当前排列；out: 结果

    过程拆解：
      1. path 长度等于 n → 记录并返回；
      2. 否则遍历每个下标 i：
         - used[i] 为 True 跳过（本路径已选）；
         - 标记 used[i]=True、加入 path → 递归 → 撤销（弹出 + used[i]=False）。

    递归终止条件：
      - len(path) == len(nums)。

    复杂度：
      - 时间 O(n·n!)；空间 O(n)。
    """
    out = []                           # 所有排列
    path = []                          # 当前排列
    used = [False] * len(nums)         # 标记哪些下标已用

    def bt():
        if len(path) == len(nums):     # 所有元素都排好了
            out.append(path[:])
            return
        for i, x in enumerate(nums):   # 每层从头选（排列讲究顺序）
            if used[i]:                # 已在当前路径用过 → 跳过
                continue
            used[i] = True             # 标记已用
            path.append(x)             # 选择
            bt()                       # 递归下一层
            path.pop()                 # 撤销选择
            used[i] = False            # 撤销标记
    bt()
    return out


# ============ 4. 去重模板（90/47）：排序 + 同层跳过 ============
def subsets_with_dup(nums):
    """含重复元素数组的所有子集（LeetCode 90），结果不重复。

    去重技巧：先排序，让相同值相邻；在**同一层**循环里，如果当前值
    等于前一个值（i > start），就跳过——因为前一个已经枚举过相同情况。

    变量说明：
      - nums: 排序后的数组；out / path: 结果 / 当前子集
      - bt(start): 从 start 开始；i: 本轮下标

    过程拆解：
      1. 每层先记录当前 path；
      2. 遍历 i >= start：若 i > start 且 nums[i] == nums[i-1]（同层重复值）
         → continue 跳过；否则选择 → 递归 → 撤销。

    递归终止条件：
      - start 越界（候选穷尽）。

    复杂度：
      - 时间 O(n·2^n)；空间 O(n)。
    """
    nums.sort()                        # 去重前提：排序使相同值相邻
    out = []                           # 结果
    path = []                          # 当前子集

    def bt(start):
        out.append(path[:])            # 记录当前子集
        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i - 1]:  # 同层去重：跳过重复值
                continue
            path.append(nums[i])       # 选择
            bt(i + 1)                  # 递归
            path.pop()                 # 撤销
    bt(0)
    return out


def permute_unique(nums):
    """含重复元素数组的全排列（LeetCode 47），结果不重复。

    去重关键：`i > 0 and x == nums[i-1] and not used[i-1]`——
    当遇到与前一个相同的值，且前一个还没被本路径用过（说明是"同层"
    重复的第一次分支），跳过。它保证相同值的多个副本只在第一个副本
    的分支下展开一次。

    变量说明：
      - nums: 排序后的数组；used: 已用标记；path / out: 当前排列 / 结果

    递归终止条件：
      - len(path) == len(nums)。

    复杂度：
      - 时间 O(n·n!)（最坏）；空间 O(n)。
    """
    nums.sort()                        # 排序，让相同值相邻
    out = []                           # 结果
    path = []                          # 当前排列
    used = [False] * len(nums)         # 已用标记

    def bt():
        if len(path) == len(nums):     # 排列完整
            out.append(path[:])
            return
        for i, x in enumerate(nums):
            if used[i]:                # 已用 → 跳过
                continue
            if i > 0 and x == nums[i - 1] and not used[i - 1]:  # 同层去重
                continue               # 前一个相同值未用 → 本分支重复，跳过
            used[i] = True             # 标记
            path.append(x)             # 选择
            bt()                       # 递归
            path.pop()                 # 撤销
            used[i] = False
    bt()
    return out


# ============ 5. 组合总和（39）：可重复选，剪枝 ============
def combination_sum(cands, target):
    """找出和等于 target 的所有组合（LeetCode 39），每个元素可无限次使用。

    与 combine 的区别：递归时传 i 而不是 i+1（同一个元素还能再选）；
    配合排序后的剪枝：cands[i] > rest 时，后面的更大，直接 break。

    变量说明：
      - cands: 排序后的候选数组；target: 目标和
      - path: 当前组合；out: 结果
      - bt(start, rest): 从 start 开始选，还需要凑 rest 的和
      - i: 本轮尝试的下标

    过程拆解：
      1. rest == 0 → 正好凑够，记录并返回；
      2. 遍历 i >= start：
         - cands[i] > rest → 剪枝 break（排序后后面都更大）；
         - 选择 cands[i] → 递归 bt(i, rest-cands[i])（可重复 → i 不减）→ 撤销。

    递归终止条件：
      - rest == 0（凑够）返回；或候选大于剩余（剪枝跳出）。

    复杂度：
      - 时间取决于组合数量（指数级上限）；空间 O(target/最小元素)。
    """
    cands.sort()                       # 排序：为剪枝做准备
    out = []                           # 结果
    path = []                          # 当前组合

    def bt(start, rest):
        if rest == 0:                  # 剩余 0 → 恰好凑成 target
            out.append(path[:])
            return
        for i in range(start, len(cands)):
            if cands[i] > rest:        # 当前值已超过剩余 → 后面的更大
                break                  # 剪枝：直接终止本层循环
            path.append(cands[i])      # 选择
            bt(i, rest - cands[i])     # 可重复选 → 递归仍从 i 开始
            path.pop()                 # 撤销
    bt(0, target)                      # 从 0 号开始，剩余 target
    return out


# ============ 6. 棋盘类：N 皇后（51，必背之一） ============
def solve_n_queens(n):
    """N 皇后问题的所有解（LeetCode 51）：n×n 棋盘放 n 个皇后互不攻击。

    按行递归：每行只能放一个皇后；用 cols[r] 记录第 r 行的列号。
    放之前用 ok(r, c) 检查与之前所有行是否冲突（同列或同对角线）。

    变量说明：
      - n: 棋盘大小；out: 所有解（每解是字符串棋盘列表）
      - cols: cols[r] = 第 r 行皇后的列
      - ok(r, c): 判断 (r, c) 能否放皇后
      - bt(r): 处理第 r 行

    过程拆解：
      1. ok(r, c)：对之前每一行 pr，若皇后同列（pc == c）或
         在两条对角线上（|pc-c| == r-pr）→ 冲突；
      2. bt(r)：r == n → 所有行都放好，把 cols 转成棋盘字符串记录；
         否则尝试每一列 c，可行就 cols[r]=c 并递归下一行。

    递归终止条件：
      - r == n（所有行放置完毕）。

    复杂度：
      - 时间 O(n!)（回溯剪枝后远小于 n!）；空间 O(n)（cols+栈）。
    """
    out = []                           # 所有解
    cols = [0] * n                     # cols[r] = 皇后所在列

    def ok(r, c):
        """判断 (r, c) 与前面 r 行的皇后是否冲突。"""
        for pr in range(r):            # 检查前面所有行
            pc = cols[pr]              # 前面某行皇后的列
            if pc == c or abs(pc - c) == r - pr:  # 同列 或 同对角线
                return False          # 冲突
        return True                    # 不冲突

    def bt(r):
        if r == n:                     # 放完最后一行 → 找到一个解
            # 每行：'.' * c + 'Q' + '.' * (n-1-c)，即皇后在第 c 列
            out.append(['.' * c + 'Q' + '.' * (n - 1 - c) for c in cols])
            return
        for c in range(n):             # 尝试第 r 行的每一列
            if ok(r, c):               # 该位置不冲突
                cols[r] = c            # 选择：放皇后
                bt(r + 1)              # 递归下一行
    bt(0)                              # 从第 0 行开始
    return out


# ============ 测试 ============
if __name__ == '__main__':
    assert sorted(subsets([1, 2])) == [[], [1], [1, 2], [2]]   # 子集 2^2=4 个
    assert combine(4, 2) == [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    assert permute([1, 2, 3])[0] == [1, 2, 3] and len(permute([1, 2, 3])) == 6
    assert len(subsets_with_dup([1, 2, 2])) == 6               # 去重子集
    assert len(permute_unique([1, 1, 2])) == 3                 # 去重排列
    assert sorted(combination_sum([2, 3, 6, 7], 7)) == [[2, 2, 3], [7]]
    assert len(solve_n_queens(4)) == 2                         # 4 皇后 2 个解
    print('✅ 回溯全家桶测试通过')
