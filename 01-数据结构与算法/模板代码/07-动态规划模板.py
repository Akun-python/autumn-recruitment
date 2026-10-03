# -*- coding: utf-8 -*-
"""07-动态规划模板.py —— P0 必背（背包/LIS/LCS/区间/打家劫舍）
配套教学：../教学/08-动态规划.ipynb
四步法：①定义 dp 语义 ②递推方程 ③初始化 ④遍历顺序（背包先物后容、逆序！）。

解 DP 题的标准流程：
  1. 定义 dp 数组：明确 dp[i]（或 dp[i][j]）表示什么；
  2. 写递推方程：当前状态如何由前面的状态转移而来；
  3. 初始化：填好边界（dp[0]、dp[i][0]、dp[0][j] 等）；
  4. 确定遍历顺序：依赖前面的状态，必须保证被依赖项先算好。
"""


# ============ 1. 0-1 背包（物品只能取一次） ============
def knapsack_01(weights, values, capacity):
    """0-1 背包：每个物品只能选一次，容量 capacity 内最大价值。

    dp[c] = "容量为 c 时能获得的最大价值"（滚动数组，一维）。

    为什么容量要**逆序**遍历：dp[c] 依赖 dp[c-w]（w 是当前物品重量）。
    若正序，本轮已经用当前物品更新过的 dp[c-w] 会被再次使用 → 等于
    一个物品被重复取（变成了完全背包）。逆序保证 dp[c-w] 还是"上一轮
    未取当前物品"的状态。

    变量说明：
      - weights / values: 物品重量 / 价值数组；capacity: 背包容量
      - dp: 容量→最大价值的滚动数组
      - i: 当前物品；c: 当前容量；w: 物品 i 的重量

    过程拆解：
      1. 初始化 dp 全 0（容量 0 价值 0）；
      2. 逐个物品 i：容量 c 从 capacity 逆序降到 weights[i]：
         - 不选：dp[c] 保持不变；
         - 选：dp[c-w] + values[i]；
         - 取两者较大值。

    复杂度：
      - 时间 O(n·capacity)，空间 O(capacity)。
    """
    n = len(weights)                   # 物品数量
    dp = [0] * (capacity + 1)          # dp[c]: 容量 c 的最大价值
    for i in range(n):                 # 逐个物品处理
        for c in range(capacity, weights[i] - 1, -1):  # 容量逆序：防止物品被重复取
            # 选/不选当前物品取最大：不选 = dp[c]，选 = dp[c-w] + value
            dp[c] = max(dp[c], dp[c - weights[i]] + values[i])
    return dp[capacity]                # 满容量下的最大价值


# ============ 2. 完全背包（物品无限取） ============
def knapsack_complete(weights, values, capacity):
    """完全背包：每个物品可无限取，容量 capacity 内最大价值。

    与 0-1 背包的唯一区别：容量改为**正序**遍历。
    正序时，本轮更新过的 dp[c-w] 会被后面的 dp[c] 再次利用 → 同一个
    物品被"重复加进去"，恰好实现了"无限取"。

    变量说明：
      - weights / values: 重量 / 价值；capacity: 容量
      - dp: 容量→最大价值；i / c: 当前物品 / 当前容量

    过程拆解：
      1. dp 初始全 0；
      2. 逐物品 i，容量 c 从 weights[i] 正序到 capacity：
         dp[c] = max(dp[c], dp[c-w] + value)。

    复杂度：
      - 时间 O(n·capacity)，空间 O(capacity)。
    """
    dp = [0] * (capacity + 1)          # dp[c]: 容量 c 的最大价值
    for i in range(len(weights)):      # 逐个物品
        for c in range(weights[i], capacity + 1):  # 容量正序 → 允许重复取
            dp[c] = max(dp[c], dp[c - weights[i]] + values[i])
    return dp[capacity]


# ============ 3. 最长上升子序列 LIS（n log n 二分优化版） ============
def length_of_lis(nums):
    """最长严格上升子序列长度（LeetCode 300），O(n log n)。

    维护 tails：tails[k] = "长度为 k+1 的上升子序列的最小末尾值"。
    - tails 单调递增；
    - 对每个 x，二分找第一个 >= x 的位置 i：
      * i == len(tails) → x 能接在现有所有子序列后形成更长的 → 追加；
      * 否则用 x 替换 tails[i]（让长度为 i+1 的子序列末尾更小，更优）。

    变量说明：
      - nums: 原数组；tails: 各长度 LIS 的最小尾值（单调递增）
      - x: 当前元素；i: 二分找到的插入位置

    复杂度：
      - 时间 O(n log n)（每元素一次二分），空间 O(n)。
    """
    import bisect
    tails = []                         # tails[k]: 长度 k+1 的 LIS 最小末尾
    for x in nums:                     # 逐个处理元素
        i = bisect.bisect_left(tails, x)  # 第一个 >= x 的位置（二分）
        if i == len(tails):            # 比所有末尾都大 → 可形成更长序列
            tails.append(x)
        else:                          # 替换：使该长度子序列末尾更小
            tails[i] = x
    return len(tails)                  # tails 的长度即最长上升子序列长度


# 朴素 O(n²) 版（也可手写，考 dp 定义）
def length_of_lis_n2(nums):
    """LIS 朴素 O(n²) 版：dp[i] = 以 nums[i] 结尾的 LIS 长度。

    对每个 i，扫描它左边所有 j：若 nums[j] < nums[i]，就能把 i 接到
    以 j 结尾的序列后面 → dp[i] = max(dp[i], dp[j]+1)。

    变量说明：
      - n: 数组长度；dp: dp[i] 以 i 结尾的 LIS 长度（初始 1：自己一个）
      - i: 当前结尾下标；j: 左侧候选下标

    复杂度：
      - 时间 O(n²)，空间 O(n)。
    """
    n = len(nums)                      # 数组长度
    dp = [1] * n                       # 每个元素至少自成 1 个长度的序列
    for i in range(n):                 # 以 i 结尾
        for j in range(i):             # 看左侧所有 j
            if nums[j] < nums[i]:      # 能接上（严格上升）
                dp[i] = max(dp[i], dp[j] + 1)  # 更新
    return max(dp) if n else 0         # 全局最大（空数组返回 0）


# ============ 4. 最长公共子序列 LCS ============
def longest_common_subsequence(a, b):
    """最长公共子序列长度（LeetCode 1143）。

    dp[i][j] = a 的前 i 个字符与 b 的前 j 个字符的 LCS 长度。
    - a[i-1] == b[j-1]：dp[i][j] = dp[i-1][j-1] + 1（两边各消耗一个）；
    - 否则：dp[i][j] = max(dp[i-1][j], dp[i][j-1])（丢掉 a 或 b 的一个）。

    变量说明：
      - m / n: 两串长度；a / b: 两字符串
      - dp: (m+1)×(n+1) 表格，dp[i][j] 见上

    初始化：dp[0][*] = dp[*][0] = 0（空串 LCS 为 0）。

    复杂度：
      - 时间 O(m·n)，空间 O(m·n)（可滚到两行优化为 O(n)）。
    """
    m, n = len(a), len(b)              # 两串长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]  # dp[i][j]: 前缀 LCS 长度
    for i in range(1, m + 1):          # 遍历 a 的前缀
        for j in range(1, n + 1):      # 遍历 b 的前缀
            if a[i - 1] == b[j - 1]:   # 当前字符相同 → 各取一个
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:                      # 不同 → 取"丢 a 一个"或"丢 b 一个"的较大值
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]                    # 全串的 LCS


# ============ 5. 打家劫舍（198，线性 DP 入门题） ============
def rob(nums):
    """打家劫舍（LeetCode 198）：不能偷相邻两家，求最大收益。

    dp[i] = 前 i 家（含第 i 家）能偷到的最大收益。
    - 不偷第 i 家：dp[i-1]；
    - 偷第 i 家：dp[i-2] + nums[i]（i-1 家不能偷）。
    用滚动变量 prev2/prev1 代替整个 dp 数组（只依赖前两个状态）。

    变量说明：
      - nums: 各家的钱；x: 当前家的钱
      - prev2: dp[i-2]（前两家）；prev1: dp[i-1]（前一家）

    初始化：prev2 = nums[0]（一家），prev1 = max(前两家)。

    复杂度：
      - 时间 O(n)，空间 O(1)。
    """
    if not nums:                       # 没有房子
        return 0
    if len(nums) == 1:                 # 只有一家
        return nums[0]
    prev2, prev1 = nums[0], max(nums[0], nums[1])  # 前 1/2 家的最优
    for x in nums[2:]:                 # 从第 3 家开始
        # 新 prev1 = max(不偷这家=旧 prev1, 偷这家=旧 prev2 + x)
        prev2, prev1 = prev1, max(prev1, prev2 + x)
    return prev1                       # 偷完所有家的最优


# ============ 6. 区间 DP：戳气球（312）/ 合并石子 ============
def max_coins(nums):
    """戳气球最大得分（LeetCode 312）。

    经典区间 DP。在数组两端补 1（arr = [1] + nums + [1]），
    dp[i][j] = "戳破 (i, j) 开区间内所有气球的最大得分"（i、j 是
    两侧的边界气球，最后才被戳破）。

    转移：枚举区间内最后被戳破的气球 k：
      dp[i][j] = max(dp[i][k] + dp[k][j] + arr[i]*arr[k]*arr[j])
    —— 最后戳 k 时，它的邻居是 i 和 j（区间内其他气球已被戳完）。

    变量说明：
      - nums: 气球得分数组；arr: 补 1 后的数组
      - n: arr 长度；dp: (n)×(n) 表格
      - length: 区间长度；i / j: 区间左右端；k: 最后戳破的气球

    遍历顺序：区间长度从小到大（小区间先算，供大区间使用）。

    复杂度：
      - 时间 O(n³)，空间 O(n²)。
    """
    arr = [1] + nums + [1]             # 首尾补 1（虚拟边界气球）
    n = len(arr)                       # 新数组长度
    dp = [[0] * n for _ in range(n)]   # dp[i][j]: (i,j) 开区间内最大得分
    for length in range(2, n):         # 区间长度从小到大（依赖小区间）
        for i in range(n - length):    # 区间左端点
            j = i + length             # 区间右端点
            for k in range(i + 1, j):  # 枚举最后戳破的气球 k
                # 得分 = 左区间 + 右区间 + 戳 k 时三气球乘积
                dp[i][j] = max(dp[i][j],
                               dp[i][k] + dp[k][j] + arr[i] * arr[k] * arr[j])
    return dp[0][n - 1]                # 整个开区间 (0, n-1) 的得分


# ============ 7. 编辑距离（72，经典二维 DP） ============
def edit_distance(a, b):
    """把 a 变成 b 的最小编辑次数（插入/删除/替换各算 1 次，LeetCode 72）。

    dp[i][j] = a 的前 i 个字符变成 b 的前 j 个字符的最小操作数。
    - a[i-1] == b[j-1]：不用操作，dp[i][j] = dp[i-1][j-1]；
    - 不等：三种操作取最小 +1：
      * 删 a[i-1]：dp[i-1][j]；
      * 插 b[j-1]：dp[i][j-1]；
      * 替换 a[i-1] 为 b[j-1]：dp[i-1][j-1]。

    初始化：dp[i][0] = i（删除 i 次），dp[0][j] = j（插入 j 次）。

    变量说明：
      - m / n: 两串长度；a / b: 两字符串；dp: (m+1)×(n+1) 表格

    复杂度：
      - 时间 O(m·n)，空间 O(m·n)。
    """
    m, n = len(a), len(b)              # 两串长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]  # dp[i][j]: 编辑距离
    for i in range(m + 1):
        dp[i][0] = i                   # a 全删 → i 次
    for j in range(n + 1):
        dp[0][j] = j                   # 空串插入 b 全部 → j 次
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:   # 当前字符相同 → 直接继承左上
                dp[i][j] = dp[i - 1][j - 1]
            else:                      # 不同 → 三种操作取最小再加 1
                dp[i][j] = 1 + min(dp[i - 1][j],     # 删 a[i-1]
                                   dp[i][j - 1],     # 插 b[j-1]
                                   dp[i - 1][j - 1])  # 替换
    return dp[m][n]                    # 全串的编辑距离


# ============ 测试 ============
if __name__ == '__main__':
    assert knapsack_01([2, 3, 4], [3, 4, 5], 5) == 7        # 选物品0和1
    assert knapsack_complete([1, 3], [2, 5], 6) == 12       # 6 个重量 1
    assert length_of_lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4  # 2,5,7,101
    assert length_of_lis_n2([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert longest_common_subsequence('abcde', 'ace') == 3  # "ace"
    assert rob([2, 7, 9, 3, 1]) == 12                       # 2+9+1
    assert max_coins([3, 1, 5, 8]) == 167                   # 312 官方示例
    assert edit_distance('horse', 'ros') == 3               # 72 官方示例
    print('✅ 动态规划全家桶测试通过')
