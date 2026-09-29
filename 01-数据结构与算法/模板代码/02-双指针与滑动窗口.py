# -*- coding: utf-8 -*-
"""02-双指针与滑动窗口模板.py —— P0 必背
配套教学：../教学/02-双指针与滑动窗口.ipynb
三大题型：①快慢指针（环/链） ②左右夹逼（有序数组） ③滑动窗口（连续子数组/串）。
"""


# ============ 1. 左右夹逼（对撞指针）：有序数组两数之和 ============
def two_sum_sorted(nums, target):
    l, r = 0, len(nums) - 1
    while l < r:
        s = nums[l] + nums[r]
        if s == target:
            return [l, r]
        elif s < target:
            l += 1                        # 和太小 → 左移大
        else:
            r -= 1                        # 和太大 → 右移小
    return [-1, -1]


# ============ 2. 快慢指针：链表中点 / 环检测（见 03 链表模板） ============
# ============ 3. 滑动窗口通用框架（求最短/最小窗口） ============
# 模板要点：右扩（加入窗口）+ 左缩（满足条件时收缩），维护窗口统计
def min_window(s, t):
    """最小覆盖子串：s 中最短的包含 t 全部字符的子串（LeetCode 76）
    返回子串（无则空串）。窗口扩展 → 满足时收缩找最小。"""
    from collections import Counter
    need = Counter(t)
    missing = len(t)                      # 还缺几个字符
    l = 0
    best = (0, float('inf'))
    for r, ch in enumerate(s):
        if need[ch] > 0:
            missing -= 1                  # 新增了需要的字符
        need[ch] -= 1
        while missing == 0:               # 窗口已覆盖 t → 尝试收缩
            if r - l < best[1] - best[0]:
                best = (l, r)
            c = s[l]
            if need[c] == 0:
                missing += 1              # 移除一个关键字符 → 不再覆盖
            need[c] += 1
            l += 1
    return s[best[0]:best[1] + 1] if best[1] != float('inf') else ''


# ============ 4. 滑动窗口求最大值（单调队列版见 04 模板） ============
def max_sliding_window(nums, k):
    """每个长度为 k 的窗口最大值；单调递减队列维护候选（O(n)）"""
    from collections import deque
    dq = deque()
    out = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:   # 队尾小的没机会当最大值
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:                # 队首滑出窗口
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out


# ============ 5. 双指针 + 原地操作：移除指定值 / 去重 ============
def remove_element(nums, val):
    """原地移除 val，返回新长度（快慢指针：慢指针写位置）"""
    i = 0
    for j, x in enumerate(nums):
        if x != val:
            nums[i] = x
            i += 1
    return i


# ============ 6. 贪心高频模板（区间类 / 覆盖类；配套教学 09-贪心） ============
def erase_overlap_intervals(intervals):
    """435 无重叠区间：按**右端点**排序，贪心保留最早结束的
    证明：最早结束的区间给后续留最大空间（交换论证）。"""
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])    # 关键：按右端点排序
    end = intervals[0][1]
    keep = 1
    for s, e in intervals[1:]:
        if s >= end:                      # 不重叠 → 保留
            keep += 1
            end = e
    return len(intervals) - keep          # 需要删除的数量


def merge_intervals(intervals):
    """56 合并区间：按左端点排序，能合并就合并"""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    out = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= out[-1][1]:               # 重叠 → 融合右端
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def jump_game(nums):
    """55 跳跃游戏：维护最远可达位置，遍历中实时更新"""
    far = 0
    for i, step in enumerate(nums):
        if i > far:
            return False                  # 够不到 i → 失败
        far = max(far, i + step)
    return True


def jump_min(nums):
    """45 跳跃游戏 II：最少跳跃次数（BFS 层思想，O(n)）"""
    jumps = cur_end = cur_far = 0
    for i in range(len(nums) - 1):
        cur_far = max(cur_far, i + nums[i])
        if i == cur_end:                  # 到达当前层的边界 → 必须跳一次
            jumps += 1
            cur_end = cur_far
    return jumps


# ============ 测试 ============
if __name__ == '__main__':
    assert two_sum_sorted([1, 2, 3, 6], 8) == [1, 3]
    assert min_window('ADOBECODEBANC', 'ABC') == 'BANC'
    assert max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    a = [3, 2, 2, 3]
    assert remove_element(a, 3) == 2
    assert erase_overlap_intervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert jump_game([2, 3, 1, 1, 4]) and not jump_game([3, 2, 1, 0, 4])
    assert jump_min([2, 3, 1, 1, 4]) == 2
    print('✅ 双指针/滑窗/贪心模板测试通过')
