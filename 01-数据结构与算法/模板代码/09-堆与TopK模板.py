# -*- coding: utf-8 -*-
"""09-堆与TopK模板.py —— P1 高频（手写堆 + 快选 + TopK 三解）
配套教学：../教学/11-堆与Top-K.ipynb
TopK 三种姿势：①堆 O(n log k) ②快选 O(n) 期望 ③排序 O(n log n)。

堆的两种经典实现方式：
  - 内置 heapq（小顶堆，面试首选，代码短）；
  - 手写 MinHeap（考底层，必须会 sift_up / sift_down）。
用数组存完全二叉树：节点 i 的左孩子 2i+1、右孩子 2i+2、父节点 (i-1)//2。
"""


# ============ 1. 手写最小堆（核心：sift_up / sift_down） ============
class MinHeap:
    """用数组实现的二叉最小堆。

    性质：a[0] 是最小值；任意节点 a[i] <= 它的两个孩子。
    数组下标即完全二叉树位置：左孩子 2i+1、右孩子 2i+2、父 (i-1)//2。

    变量说明：
      - a: 存储堆的数组
    """
    def __init__(self, arr=None):
        """建堆：把任意数组整理成最小堆（O(n) 下滤法）。

        从最后一个非叶节点 (n//2-1) 倒着向下调整：因为叶子节点本身
        就是合法堆，只需让每个内部节点下沉到合适位置。
        """
        self.a = (arr or [])[:]        # 拷贝输入（不修改外部数组）
        for i in range(len(self.a) // 2 - 1, -1, -1):  # 从最后一个非叶往前
            self._sift_down(i)         # 每个内部节点下沉 → 整体成堆

    def _sift_up(self, i):
        """上浮：把下标 i 的元素一路向上，直到满足堆性质（用于 push）。

        与父节点比较：小于父 → 交换并继续向上；不小于 → 停止。

        循环终止条件：i 到根（i==0）或 a[i] >= 父节点。

        复杂度：O(log n)。
        """
        while i > 0:                   # 还没到根
            p = (i - 1) // 2           # 父节点下标
            if self.a[p] <= self.a[i]: # 父 <= 自己 → 已满足堆性质
                break                  # 停止上浮
            self.a[p], self.a[i] = self.a[i], self.a[p]  # 交换父子
            i = p                      # 继续从父节点位置向上

    def _sift_down(self, i):
        """下沉：把下标 i 的元素一路向下，直到满足堆性质（用于 pop/建堆）。

        找两个孩子中较小的：若自己 <= 较小孩子 → 停止；否则交换并继续下移。

        变量说明：
          - n: 当前堆大小；l / r: 左/右孩子；small: 较小的孩子

        循环终止条件：无左孩子（叶子，2i+1 >= n）或 a[i] <= 较小孩子。

        复杂度：O(log n)。
        """
        n = len(self.a)                # 堆大小
        while 2 * i + 1 < n:           # 有左孩子才需继续
            l, r = 2 * i + 1, 2 * i + 2  # 左右孩子
            small = l if r >= n or self.a[l] <= self.a[r] else r  # 较小者
            if self.a[i] <= self.a[small]:  # 已 <= 较小孩子 → 堆性质满足
                break                  # 停止下沉
            self.a[i], self.a[small] = self.a[small], self.a[i]  # 交换
            i = small                  # 继续下沉到孩子位置

    def push(self, x):
        """插入：先放数组末尾，再上浮到正确位置。O(log n)。"""
        self.a.append(x)               # 末尾追加（叶子位置）
        self._sift_up(len(self.a) - 1) # 上浮恢复堆性质

    def pop(self):
        """弹出最小值（堆顶）。O(log n)。

        过程：记录堆顶 → 把末元素搬到堆顶 → 下沉恢复堆性质。
        """
        if not self.a:                 # 空堆
            raise IndexError('empty heap')
        top = self.a[0]                # 堆顶即最小值
        last = self.a.pop()            # 取出末尾元素并缩短数组
        if self.a:                     # 堆非空才需要调整
            self.a[0] = last           # 末元素补到堆顶
            self._sift_down(0)         # 下沉恢复堆性质
        return top                     # 返回最小值

    def top(self):
        """查看堆顶（最小值），不弹出。O(1)。"""
        return self.a[0]

    def __len__(self):
        """堆的大小（支持 len() 调用）。"""
        return len(self.a)


# ============ 2. TopK 最小：维护大小为 k 的大顶堆 ============
import heapq


def topk_smallest(nums, k):
    """返回 nums 中最小的 k 个（升序），堆解法 O(n log k)。

    思路：维护一个大小为 k 的**大顶堆**（Python 无内置大顶堆，
    用"存负数的小顶堆"模拟）。堆里始终放着"当前见过的最小 k 个"：
      - 新元素 x 比堆顶（当前第 k 小）还小 → 替换堆顶；
      - 否则 x 不可能进前 k，忽略。

    变量说明：
      - nums: 原数组；k: 要取几个
      - heap: 大顶堆（存负数）；x: 当前元素

    复杂度：
      - 时间 O(n log k)，空间 O(k)。
    """
    heap = [-x for x in nums[:k]]      # 前 k 个取负 → 小顶堆即"大顶堆"
    heapq.heapify(heap)                # 线性建堆
    for x in nums[k:]:                 # 处理剩余元素
        if x < -heap[0]:               # 比当前堆顶（前 k 小里的最大）还小
            heapq.heapreplace(heap, -x)  # 替换堆顶（自动恢复堆性质）
    return sorted(-x for x in heap)    # 取负还原并升序返回


# ============ 3. 快选 QuickSelect：第 k 大（期望 O(n)） ============
def partition(a, lo, hi):
    """Lomuto 分区（同快速排序）：以 a[hi] 为 pivot，返回其最终位置。

    变量说明：
      - a: 数组；lo / hi: 区间边界；pv: pivot 值
      - i: "小于 pivot 区"写入位置；j: 扫描游标

    循环终止条件：j 扫完 [lo, hi-1]。

    复杂度：O(hi-lo+1) 时间，O(1) 空间。
    """
    pv = a[hi]                         # 取末尾为 pivot
    i = lo                             # 小于区写入位置
    for j in range(lo, hi):            # 扫描
        if a[j] < pv:                  # 比 pivot 小 → 换到左侧
            a[i], a[j] = a[j], a[i]    # 交换
            i += 1                     # 左区边界右移
    a[i], a[hi] = a[hi], a[i]          # pivot 就位
    return i                           # 返回 pivot 最终位置


def quick_select(nums, k):
    """第 k 小元素（0-based k；第 k 大 = quick_select(nums, n-1-k)）。

    思路：每次 partition 后，pivot 落在最终位置 p：
      - p == k → a[p] 就是答案；
      - p <  k → 答案在右半，收缩 lo = p+1；
      - p >  k → 答案在左半，收缩 hi = p-1。
    平均每次只处理一半，期望 O(n)。

    变量说明：
      - a: 复制的数组；lo / hi: 当前区间；p: pivot 位置；k: 目标秩

    循环终止条件：
      - p == k 命中返回；或 lo > hi（理论上不会，k 必合法）。

    复杂度：
      - 期望 O(n)，最坏 O(n²)（可通过随机选 pivot 规避）；空间 O(n)（副本）。
    """
    a = nums[:]                        # 复制，避免修改外部数组
    lo, hi = 0, len(a) - 1             # 初始区间
    while lo <= hi:                    # 区间非空
        p = partition(a, lo, hi)       # 分区，pivot 就位
        if p == k:                     # 命中第 k 小
            return a[p]
        elif p < k:                    # 答案在右半
            lo = p + 1
        else:                          # 答案在左半
            hi = p - 1
    return -1                          # 兜底（k 合法时不会走到）


# ============ 4. 数据流中位数（295，双堆） ============
class MedianFinder:
    """数据流中位数（LeetCode 295）。

    双堆维护：把数分成"较小的一半"和"较大的一半"：
      - small: 大顶堆（存负数模拟）——存较小的一半；
      - large: 小顶堆——存较大的一半。
    维护两个堆大小相差不超过 1，且 small 堆顶 <= large 堆顶，
    则中位数就是两个堆顶之一（或平均值）。

    变量说明：
      - small: 大顶堆（元素为负数，堆顶 -small[0] 是较小半边的最大值）
      - large: 小顶堆（堆顶是较大半边的最小值）
      - x: 新加入的数字
    """
    def __init__(self):
        self.small = []                # 大顶堆（存负数）——较小的一半
        self.large = []                # 小顶堆——较大的一半

    def add_num(self, x):
        """插入一个数，并保持两个堆的大小与顺序约束。O(log n)。

        过程：
          1. 新数先进 small（取负 → small 变成大顶堆）；
          2. 若 small 堆顶 > large 堆顶（顺序颠倒）→ 把 small 堆顶移到 large；
          3. 平衡大小：|small| 最多比 |large| 大 1，反之亦然。
        """
        heapq.heappush(self.small, -x)  # 先放入较小半边
        # 保证 small 堆顶 <= large 堆顶（否则把 small 的最大值搬过去）
        if self.small and self.large and -self.small[0] > self.large[0]:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        # 保证 |small| <= |large| + 1（small 不能太大）
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        # 保证 |large| <= |small| + 1（large 也不能太大）
        if len(self.large) > len(self.small) + 1:
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def find_median(self):
        """返回当前中位数。O(1)。

        - 元素个数奇数 → 数量多的那一边的堆顶；
        - 偶数 → 两个堆顶的平均值。
        """
        if len(self.small) > len(self.large):      # 奇数 → small 多一个
            return -self.small[0]                  # 中位数是 small 堆顶
        if len(self.large) > len(self.small):      # 奇数 → large 多一个
            return self.large[0]                   # 中位数是 large 堆顶
        return (-self.small[0] + self.large[0]) / 2  # 偶数 → 两堆顶平均


# ============ 测试 ============
if __name__ == '__main__':
    import random
    rng = random.Random(0)                         # 固定种子
    arr = [rng.randint(0, 999) for _ in range(500)]  # 500 个随机数
    h = MinHeap(arr)                               # 手写堆
    popped = [h.pop() for _ in range(len(arr))]    # 依次弹出
    assert popped == sorted(arr)                   # 弹出顺序 = 升序
    assert sorted(topk_smallest(arr, 5)) == sorted(arr)[:5]  # 最小 5 个
    assert quick_select(arr, 0) == min(arr)        # 第 0 小 = 最小值
    assert quick_select(arr, 499) == max(arr)      # 第 499 小 = 最大值
    mf = MedianFinder()                            # 数据流中位数
    for x in [5, 1, 3, 2, 4]:
        mf.add_num(x)
    assert mf.find_median() == 3.0                 # 中位数 3
    print('✅ 堆与 TopK 全家桶测试通过')
