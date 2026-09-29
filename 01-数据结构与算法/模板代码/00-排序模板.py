# -*- coding: utf-8 -*-
"""00-排序模板.py —— 背诵级手写排序（P0 必背：快排/归并/堆排/插入）
配套教学：../教学/00-排序算法.ipynb
全部手写，不调用内置排序（测试行除外）。"""


# ============ 1. 插入排序 O(n²) 稳定 原地 ============
# 思想：像抓扑克，当前元素插入前面有序区
# 对近乎有序数组 O(n)；稳定（用 > 不用 >=）
def insertion_sort(a):
    n = len(a)
    for i in range(1, n):
        cur = a[i]
        j = i - 1
        while j >= 0 and a[j] > cur:      # 只移严格大于的 → 稳定
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = cur
    return a


# ============ 2. 归并排序 O(n log n) 稳定 空间 O(n) ============
# 思想：分治 —— 分成两半分别排好，再合并两个有序数组
# 面试点：合并用 <= 保稳定；可扩展求逆序对（右先出时 cnt += len(L)-i）
def merge_sort(a):
    if len(a) <= 1:
        return a[:]
    mid = len(a) // 2
    L = merge_sort(a[:mid])
    R = merge_sort(a[mid:])
    i = j = 0
    out = []
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:                  # <= 保证稳定
            out.append(L[i]); i += 1
        else:
            out.append(R[j]); j += 1
    out.extend(L[i:]); out.extend(R[j:])
    return out


# ============ 3. 快速排序 O(n log n) 期望 不稳定 原地 ============
# 思想：partition（Lomuto，单指针）选 pivot 就位，两侧递归
# 面试点：随机化 pivot 防最坏 O(n²)；partition 返回 pivot 最终位置
def partition(a, lo, hi):
    pivot = a[hi]                          # Lomuto: 取末尾为 pivot
    i = lo                                 # i 指向"大于区"起点
    for j in range(lo, hi):
        if a[j] < pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]              # pivot 就位
    return i


def quick_sort(a, lo=0, hi=None):
    if hi is None:
        hi = len(a) - 1
        a = a[:]
    if lo >= hi:
        return a if (lo == 0 and hi == len(a) - 1) else None
    # 随机化：交换随机位置到末尾再 partition（防已排序最坏）
    # import random; r = random.randint(lo, hi); a[r], a[hi] = a[hi], a[r]
    p = partition(a, lo, hi)
    quick_sort(a, lo, p - 1)
    quick_sort(a, p + 1, hi)
    return a if lo == 0 and hi == len(a) - 1 else None


# ============ 4. 堆排序 O(n log n) 不稳定 原地 O(1) ============
# 思想：建大顶堆 → 反复"堆顶与末尾交换 + 缩小范围下沉"
# 面试点：sift_down 下标 2i+1/2i+2；建堆从 n//2-1 往前；end 是当前堆大小
def sift_down(a, i, n):
    while 2 * i + 1 < n:
        l, r = 2 * i + 1, 2 * i + 2
        big = l if r >= n or a[l] >= a[r] else r
        if a[i] >= a[big]:
            break
        a[i], a[big] = a[big], a[i]
        i = big


def heap_sort(a):
    a = a[:]
    n = len(a)
    for i in range(n // 2 - 1, -1, -1):    # 建堆：从最后一个非叶
        sift_down(a, i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]        # 最大值进有序区
        sift_down(a, 0, end)               # 堆大小 = end
    return a


# ============ 测试 ============
if __name__ == '__main__':
    import random
    rng = random.Random(0)
    arr = [rng.randint(0, 99) for _ in range(200)]
    assert insertion_sort(arr[:]) == sorted(arr)
    assert merge_sort(arr) == sorted(arr)
    assert quick_sort(arr) == sorted(arr)
    assert heap_sort(arr) == sorted(arr)
    print('✅ 四种排序全部通过（200 随机数 vs sorted）')
