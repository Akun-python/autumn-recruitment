# -*- coding: utf-8 -*-
"""01-二分查找模板.py —— P0 必背
配套教学：../教学/05-二分查找.ipynb
核心：while l < r 闭区间 vs 开区间两套写法都要会；边界题用「左闭右开」最稳。

本文件收录二分查找的"全家桶"：
  1. 标准二分：找 target 是否存在（闭区间 [l, r]）
  2. 左边界：第一个 >= target 的位置（lower_bound，左闭右开 [l, r)）
  3. 右边界：第一个 >  target 的位置（upper_bound）
  4. 答案二分：在值域上对单调谓词二分（二分答案，面试高频）
  5. 旋转数组：找最小值 / 搜索 target

共同要点：
  - 每次把搜索区间砍半，时间复杂度 O(log n)；
  - 用 mid = l + (r-l)//2 而非 (l+r)//2，防止两个大数相加溢出；
  - 分清"闭区间"与"左闭右开"两种写法，边界收窄规则不同，别混用。
"""


# ============ 1. 标准二分：找 target 是否存在 ============
# 闭区间 [l, r]：l、r 都可能是答案，循环条件用 l <= r。
def binary_search(nums, target):
    """在有序数组 nums 中查找 target，存在返回下标，不存在返回 -1。

    变量说明：
      - nums: 升序有序数组
      - target: 要查找的目标值
      - l / r: 当前搜索区间的左右端点，闭区间 [l, r]
      - mid: 区间中点下标

    过程拆解：
      1. 初始区间覆盖整个数组 [0, n-1]；
      2. 每次取中点 mid：
         - nums[mid] == target → 命中，直接返回；
         - nums[mid] <  target → target 只可能在右半边，收缩 l = mid+1；
         - nums[mid] >  target → target 只可能在左半边，收缩 r = mid-1；
      3. 区间为空（l > r）仍没找到 → 返回 -1。

    循环终止条件：
      - l > r（区间为空）时退出；每次迭代区间长度至少减半。

    复杂度：
      - 时间 O(log n)，空间 O(1)。
    """
    l, r = 0, len(nums) - 1          # 闭区间左右端点
    while l <= r:                    # 区间非空才继续
        mid = l + (r - l) // 2       # 中点；用差值折半防 (l+r) 溢出
        if nums[mid] == target:      # 命中目标
            return mid
        elif nums[mid] < target:     # 中点值偏小 → 答案在右半边
            l = mid + 1              # 排除 mid 及其左边全部
        else:                        # 中点值偏大 → 答案在左半边
            r = mid - 1              # 排除 mid 及其右边全部
    return -1                        # 区间耗尽仍未找到


# ============ 2. 左边界：第一个 >= target 的位置（lower_bound） ============
# 左闭右开 [l, r)：r 是"开"的，循环条件用 l < r，退出时 l == r 就是答案。
def lower_bound(nums, target):
    """返回第一个 >= target 的下标（若全部 < target 则返回 len(nums)）。

    变量说明：
      - nums: 升序有序数组
      - l / r: 搜索区间 [l, r)，左闭右开
      - mid: 中点下标

    过程拆解：
      1. 初始区间 [0, len(nums))，r 取 len(nums) 而不是 len(nums)-1，
         这样"答案在末尾之后"（即不存在）时也能用 l == r 表示；
      2. 取中点 mid：
         - nums[mid] < target → mid 及其左边都太小，答案在右边，l = mid+1；
         - nums[mid] >= target → mid 可能是答案（也可能是更左边的），
           把右边界收到 mid，即 r = mid（保留 mid 候选）；
      3. 循环结束时 l == r，该位置就是第一个 >= target 的下标。

    循环终止条件：
      - l == r（区间收缩为空）时退出；这就是插入点/第一个满足条件的位置。

    复杂度：
      - 时间 O(log n)，空间 O(1)。
    """
    l, r = 0, len(nums)              # 左闭右开 [l, r)
    while l < r:                     # 区间还有元素才继续
        mid = l + (r - l) // 2       # 中点
        if nums[mid] < target:       # 中点太小 → 不可能是答案
            l = mid + 1              # 右移：排除 mid 及其左边
        else:                        # 中点 >= target → 可能是答案
            r = mid                  # 左移右边界，保留 mid（答案在左边或就是 mid）
    return l                         # 返回第一个 >= target 的位置


# ============ 3. 右边界：第一个 > target 的位置（upper_bound） ============
def upper_bound(nums, target):
    """返回第一个 > target 的下标（若全部 <= target 则返回 len(nums)）。

    与 lower_bound 的唯一区别：把比较条件从 < target 改为 <= target，
    即"等于 target 的也不算答案"，从而把边界推到第一个严格大于的位置。

    变量说明：
      - nums / target / l / r / mid: 同 lower_bound

    过程拆解：
      1. nums[mid] <= target → mid 及其左边都不可能是答案，l = mid+1；
      2. nums[mid] >  target → mid 可能是答案，r = mid 收窄；
      3. 退出时 l 指向第一个严格大于 target 的下标。

    循环终止条件：
      - l == r 时退出。

    复杂度：
      - 时间 O(log n)，空间 O(1)。
    """
    l, r = 0, len(nums)              # 左闭右开 [l, r)
    while l < r:
        mid = l + (r - l) // 2       # 中点
        if nums[mid] <= target:      # 含等于：等于 target 的也不满足条件
            l = mid + 1              # 右移：排除 mid 及其左边
        else:                        # mid 严格大于 target → 可能是答案
            r = mid                  # 左移右边界，保留 mid
    return l                         # 第一个 > target 的位置


# ============ 4. 在值域上二分（答案二分）——高频考点 ============
# 例：sqrt 整数部分 / 最小化最大值 / 满足条件的最小值
# 模板：对"可行"单调的谓词 pred(x) 二分
def binary_search_answer(pred, lo, hi):
    """在 [lo, hi] 的整数范围上二分，返回第一个满足 pred 的整数。

    pred: int -> bool，且 pred 的取值随 x 单调（形如 [False]*k + [True]*...）。
    典型应用：sqrt(x) 整数部分、能塞下的最小容量、能过的最小等待时间等。

    变量说明：
      - pred: 单调布尔谓词，pred(x) 为 True 表示 x 可行
      - lo / hi: 值域的下界 / 上界（含，整数）
      - l / r: 搜索区间 [l, r)，右端点取 hi+1 使得 hi 本身也可被选中
      - mid: 中点

    过程拆解：
      1. 沿用 lower_bound 的左闭右开思路：可行与不可行在值域上单调分界；
      2. pred(mid) 为 True → mid 及更右都可行，收 r = mid 找更小的可行值；
      3. pred(mid) 为 False → mid 及更左都不可行，收 l = mid+1；
      4. 退出时若 l <= hi 说明找到了可行点，否则全值域都不可行返回 -1。

    循环终止条件：
      - l == r 时退出；答案即 l（若在值域范围内）。

    复杂度：
      - 时间 O(log(hi-lo)) × pred 的单次代价；空间 O(1)。
    """
    # 假设 pred 形如 [False]*k + [True]*...，用 lower_bound 思路
    l, r = lo, hi + 1                # 左闭右开 [lo, hi+1)，保证 hi 能被考虑
    while l < r:
        mid = l + (r - l) // 2       # 中点
        if pred(mid):                # mid 可行 → 答案可能是 mid 或更小
            r = mid                  # 收窄右边界，继续找最小可行值
        else:                        # mid 不可行 → 答案必然更大
            l = mid + 1              # 右移左边界
    return l if l <= hi else -1      # 在值域内返回答案，否则表示无解


# ============ 5. 旋转数组：有序数组中找最小值 / 搜索 target ============
def find_min_rotated(nums):
    """在旋转有序数组（如 [4,5,6,1,2,3]）中找最小值，无重复元素。

    核心观察：把 nums[mid] 与 nums[r] 比较——
      - nums[mid] > nums[r] → 旋转点（最小值）在右半边，l = mid+1；
      - nums[mid] < nums[r] → 最小值在左半边（含 mid），r = mid。
    等价于在"值比右端点大的都是左半段"这一单调性质上做 lower_bound。

    变量说明：
      - l / r: 闭区间 [l, r] 左右端点；mid: 中点

    循环终止条件：
      - l == r 时退出，该位置即最小值下标。

    复杂度：
      - 时间 O(log n)，空间 O(1)。
    """
    l, r = 0, len(nums) - 1          # 闭区间
    while l < r:                     # 区间还有多个元素才继续
        mid = l + (r - l) // 2       # 中点
        if nums[mid] > nums[r]:      # 中点比右端大 → 断点在右半边
            l = mid + 1              # 最小值必在 (mid, r] 内
        else:                        # 中点 <= 右端 → 右半边是递增的
            r = mid                  # 最小值在 [l, mid] 内
    return nums[l]                   # l == r，即最小值


def search_rotated(nums, target):
    """在旋转有序数组 nums 中搜索 target，返回下标，找不到返回 -1。

    思路：虽然整体被旋转，但任意时刻 [l, mid] 与 [mid, r] 中至少有一段
    是严格有序的，可以借助"有序段"判断 target 落在哪一侧。

    变量说明：
      - l / r: 闭区间左右端点；mid: 中点

    过程拆解：
      1. nums[mid] == target → 命中；
      2. nums[l] <= nums[mid] → 左半段 [l, mid] 有序：
         - target 落在该有序段内（nums[l] <= target < nums[mid]）→ 搜左边；
         - 否则 → 搜右边；
      3. 否则右半段 [mid, r] 有序：
         - target 落在该有序段内（nums[mid] < target <= nums[r]）→ 搜右边；
         - 否则 → 搜左边。

    循环终止条件：
      - 命中返回；或 l > r（区间为空）返回 -1。

    复杂度：
      - 时间 O(log n)，空间 O(1)。
    """
    l, r = 0, len(nums) - 1          # 闭区间
    while l <= r:                    # 区间非空才继续
        mid = l + (r - l) // 2       # 中点
        if nums[mid] == target:      # 命中
            return mid
        if nums[l] <= nums[mid]:     # 左半段有序（mid 落在左段递增区）
            if nums[l] <= target < nums[mid]:   # target 在左有序段内
                r = mid - 1          # 收缩到左半边
            else:                    # target 在左段之外 → 去右边找
                l = mid + 1
        else:                        # 右半段有序（mid 落在右段递增区）
            if nums[mid] < target <= nums[r]:   # target 在右有序段内
                l = mid + 1          # 收缩到右半边
            else:                    # target 在右段之外 → 去左边找
                r = mid - 1
    return -1                        # 区间耗尽，未找到


# ============ 测试 ============
if __name__ == '__main__':
    assert binary_search([1, 3, 5, 7, 9], 5) == 2              # 存在 → 返回下标
    assert binary_search([1, 3, 5, 7, 9], 6) == -1             # 不存在 → -1
    assert lower_bound([1, 2, 2, 2, 3], 2) == 1                # 第一个 >=2 是下标 1
    assert upper_bound([1, 2, 2, 2, 3], 2) == 4                # 第一个 >2 是下标 4
    assert find_min_rotated([4, 5, 6, 1, 2, 3]) == 1           # 旋转数组最小值
    assert search_rotated([4, 5, 6, 1, 2, 3], 5) == 1          # 旋转数组中找 5
    print('✅ 二分全家桶测试通过')
