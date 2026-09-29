# -*- coding: utf-8 -*-
"""09-堆与TopK模板.py —— P1 高频（手写堆 + 快选 + TopK 三解）
配套教学：../教学/11-堆与Top-K.ipynb
TopK 三种姿势：①堆 O(n log k) ②快选 O(n) 期望 ③排序 O(n log n)。
"""


# ============ 1. 手写最小堆（核心：sift_up / sift_down） ============
class MinHeap:
    """用数组存的完全二叉树；heap[0] 是最小值"""
    def __init__(self, arr=None):
        self.a = (arr or [])[:]
        for i in range(len(self.a) // 2 - 1, -1, -1):
            self._sift_down(i)            # 建堆 O(n)

    def _sift_up(self, i):
        while i > 0:
            p = (i - 1) // 2
            if self.a[p] <= self.a[i]:
                break
            self.a[p], self.a[i] = self.a[i], self.a[p]
            i = p

    def _sift_down(self, i):
        n = len(self.a)
        while 2 * i + 1 < n:
            l, r = 2 * i + 1, 2 * i + 2
            small = l if r >= n or self.a[l] <= self.a[r] else r
            if self.a[i] <= self.a[small]:
                break
            self.a[i], self.a[small] = self.a[small], self.a[i]
            i = small

    def push(self, x):
        self.a.append(x)
        self._sift_up(len(self.a) - 1)

    def pop(self):
        if not self.a:
            raise IndexError('empty heap')
        top = self.a[0]
        last = self.a.pop()
        if self.a:
            self.a[0] = last              # 末元素补顶再下沉
            self._sift_down(0)
        return top

    def top(self):
        return self.a[0]

    def __len__(self):
        return len(self.a)


# ============ 2. TopK 最小：维护大小为 k 的大顶堆 ============
import heapq


def topk_smallest(nums, k):
    """返回最小的 k 个（堆解法 O(n log k)）"""
    heap = [-x for x in nums[:k]]
    heapq.heapify(heap)                   # 大顶堆 = 存负数的小顶堆
    for x in nums[k:]:
        if x < -heap[0]:
            heapq.heapreplace(heap, -x)   # 淘汰堆顶（当前最大）
    return sorted(-x for x in heap)


# ============ 3. 快选 QuickSelect：第 k 大（期望 O(n)） ============
def partition(a, lo, hi):
    pv = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] < pv:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]
    return i


def quick_select(nums, k):
    """第 k 小（0-based）；第 k 大 = quick_select(nums, n-1-k)"""
    a = nums[:]
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        p = partition(a, lo, hi)
        if p == k:
            return a[p]
        elif p < k:
            lo = p + 1
        else:
            hi = p - 1
    return -1


# ============ 4. 数据流中位数（295，双堆） ============
class MedianFinder:
    """小顶堆(大的一半) + 大顶堆(小的一半)，中位数 = 两者堆顶之一"""
    def __init__(self):
        self.small = []                   # 大顶堆（存负数）
        self.large = []                   # 小顶堆

    def add_num(self, x):
        heapq.heappush(self.small, -x)
        # 保证 small 堆顶 <= large 堆顶，且 |small| <= |large| <= |small|+1
        if self.small and self.large and -self.small[0] > self.large[0]:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small) + 1:
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def find_median(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        if len(self.large) > len(self.small):
            return self.large[0]
        return (-self.small[0] + self.large[0]) / 2


# ============ 测试 ============
if __name__ == '__main__':
    import random
    rng = random.Random(0)
    arr = [rng.randint(0, 999) for _ in range(500)]
    h = MinHeap(arr)
    popped = [h.pop() for _ in range(len(arr))]
    assert popped == sorted(arr)
    assert sorted(topk_smallest(arr, 5)) == sorted(arr)[:5]
    assert quick_select(arr, 0) == min(arr)
    assert quick_select(arr, 499) == max(arr)
    mf = MedianFinder()
    for x in [5, 1, 3, 2, 4]:
        mf.add_num(x)
    assert mf.find_median() == 3.0
    print('✅ 堆与 TopK 全家桶测试通过')