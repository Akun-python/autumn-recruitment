# -*- coding: utf-8 -*-
"""07-动态规划模板.py —— P0 必背（背包/LIS/LCS/区间/打家劫舍）
配套教学：../教学/08-动态规划.ipynb
四步法：①定义 dp 语义 ②递推方程 ③初始化 ④遍历顺序（背包先物后容、逆序！）。
"""


# ============ 1. 0-1 背包（物品只能取一次） ============
def knapsack_01(weights, values, capacity):
    """dp[c] = 容量 c 的最大价值；先物品后容量且容量**逆序**"""
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):                    # 逐物品
        for c in range(capacity, weights[i] - 1, -1):   # 逆序防复用
            dp[c] = max(dp[c], dp[c - weights[i]] + values[i])
    return dp[capacity]


# ============ 2. 完全背包（物品无限取） ============
def knapsack_complete(weights, values, capacity):
    """区别只在容量**正序**遍历"""
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for c in range(weights[i], capacity + 1):       # 正序允许重复
            dp[c] = max(dp[c], dp[c - weights[i]] + values[i])
    return dp[capacity]


# ============ 3. 最长上升子序列 LIS（n log n 二分优化版） ============
def length_of_lis(nums):
    """tails[k] = 长度为 k+1 的 LIS 最小尾值；二分找插入位"""
    import bisect
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)  # 第一个 >= x 的位置
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x                  # 替换，保持 tails 单调
    return len(tails)


# 朴素 O(n²) 版（也可手写，考 dp 定义）
def length_of_lis_n2(nums):
    n = len(nums)
    dp = [1] * n                          # dp[i] = 以 i 结尾的 LIS 长度
    for i in range(n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp) if n else 0


# ============ 4. 最长公共子序列 LCS ============
def longest_common_subsequence(a, b):
    """dp[i][j] = a[:i] 与 b[:j] 的 LCS 长度；二维可滚到两行"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# ============ 5. 打家劫舍（198，线性 DP 入门题） ============
def rob(nums):
    """dp[i] = 前 i 家最大收益；dp[i] = max(dp[i-1], dp[i-2]+nums[i])"""
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    for x in nums[2:]:
        prev2, prev1 = prev1, max(prev1, prev2 + x)
    return prev1


# ============ 6. 区间 DP：戳气球（312）/ 合并石子 ============
def max_coins(nums):
    """经典区间 DP：dp[i][j] = (i,j) 开区间内的最大收益"""
    arr = [1] + nums + [1]
    n = len(arr)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):            # 区间长度从小到大
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):     # 最后戳 k
                dp[i][j] = max(dp[i][j],
                               dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j])
    return dp[0][n - 1]


# ============ 7. 编辑距离（72，经典二维 DP） ============
def edit_distance(a, b):
    """插入/删除/替换三种操作把 a 变 b 的最小次数"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i                      # 删除 i 次
    for j in range(n + 1):
        dp[0][j] = j                      # 插入 j 次
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],     # 删 a[i-1]
                                   dp[i][j - 1],     # 插 b[j-1]
                                   dp[i - 1][j - 1])  # 替换
    return dp[m][n]


# ============ 测试 ============
if __name__ == '__main__':
    assert knapsack_01([2, 3, 4], [3, 4, 5], 5) == 7
    assert knapsack_complete([1, 3], [2, 5], 6) == 12
    assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert length_of_lis_n2([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert longest_common_subsequence('abcde', 'ace') == 3
    assert rob([2, 7, 9, 3, 1]) == 12
    assert max_coins([3, 1, 5, 8]) == 167
    assert edit_distance('horse', 'ros') == 3
    print('✅ 动态规划全家桶测试通过')