# -*- coding: utf-8 -*-
"""01-二分查找模板.py —— P0 必背
配套教学：../教学/05-二分查找.ipynb
核心：while l < r 闭区间 vs 开区间两套写法都要会；边界题用「左闭右开」最稳。
"""


# ============ 1. 标准二分：找 target 是否存在 ============
# 闭区间 [l, r]
def binary_search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2            # 防 (l+r) 溢出
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            l = mid + 1
        else:
            r = mid - 1
    return -1


# ============ 2. 左边界：第一个 >= target 的位置（lower_bound） ============
# 左闭右开 [l, r)
def lower_bound(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] < target:
            l = mid + 1                   # 右侧收缩：mid 及其左都不够
        else:
            r = mid                       # 右侧包含 mid（可能是答案）
    return l                              # 返回插入点/第一个>=target


# ============ 3. 右边界：第一个 > target 的位置（upper_bound） ============
def upper_bound(nums, target):
    l, r = 0, len(nums)
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] <= target:
            l = mid + 1
        else:
            r = mid
    return l


# ============ 4. 在值域上二分（答案二分）——高频考点 ============
# 例：sqrt 整数部分 / 最小化最大值 / 满足条件的最小值
# 模板：对"可行"单调的谓词 pred(x) 二分
def binary_search_answer(pred, lo, hi):
    """pred: int -> bool，满足 pred(x) 的 x 单调（false..true 或 true..false）
    返回第一个满足条件的整数。"""
    # 假设 pred 形如 [False]*k + [True]*...，用 lower_bound 思路
    l, r = lo, hi + 1
    while l < r:
        mid = l + (r - l) // 2
        if pred(mid):
            r = mid
        else:
            l = mid + 1
    return l if l <= hi else -1


# ============ 5. 旋转数组：有序数组中找最小值 / 搜索 target ============
def find_min_rotated(nums):
    """[4,5,6,1,2,3] -> 1；核心：mid 与 r 比较（无重复时）"""
    l, r = 0, len(nums) - 1
    while l < r:
        mid = l + (r - l) // 2
        if nums[mid] > nums[r]:           # 最小值在右半边
            l = mid + 1
        else:
            r = mid
    return nums[l]


def search_rotated(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        mid = l + (r - l) // 2
        if nums[mid] == target:
            return mid
        if nums[l] <= nums[mid]:          # 左半有序
            if nums[l] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        else:                             # 右半有序
            if nums[mid] < target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
    return -1


# ============ 测试 ============
if __name__ == '__main__':
    assert binary_search([1, 3, 5, 7, 9], 5) == 2
    assert binary_search([1, 3, 5, 7, 9], 6) == -1
    assert lower_bound([1, 2, 2, 2, 3], 2) == 1
    assert upper_bound([1, 2, 2, 2, 3], 2) == 4
    assert find_min_rotated([4, 5, 6, 1, 2, 3]) == 1
    assert search_rotated([4, 5, 6, 1, 2, 3], 5) == 1
    print('✅ 二分全家桶测试通过')
