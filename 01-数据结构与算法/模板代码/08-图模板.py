# -*- coding: utf-8 -*-
"""08-图模板.py —— P0 必背（BFS/DFS/拓扑/Dijkstra）
配套教学：../教学/10-图.ipynb
图的三种存储：邻接表（默认，List[List[int]]）/ 邻接矩阵 / 边表。

本文件默认使用邻接表存储：
  - 无权图：g[u] 是 u 的邻居列表；
  - 带权图：g[u] 是 (邻居, 边权) 的列表。
"""

from collections import deque


# ============ 图的基本存储：邻接表 ============
# n 个节点，edges = [(u, v), ...]（无向图需双向加）
def build_adj(n, edges, directed=False):
    """根据边列表构建邻接表。

    变量说明：
      - n: 节点个数（编号 0..n-1）
      - edges: 边列表，每条边 (u, v) 表示 u → v
      - directed: True 表示有向图（只加一条），False 表示无向图（双向加）
      - g: 邻接表，g[u] 是 u 的所有邻居

    复杂度：
      - 时间 O(n + |E|)，空间 O(n + |E|)。
    """
    g = [[] for _ in range(n)]         # 初始化 n 个空邻接表
    for u, v in edges:                 # 逐条边加入
        g[u].append(v)                 # 正向：u → v
        if not directed:               # 无向图还要加反向
            g[v].append(u)             # 反向：v → u
    return g


# ============ 1. BFS（最短路径/层序，O(V+E)） ============
def bfs(g, start):
    """从 start 出发的 BFS，返回到各点的最短距离（无权图，LeetCode 通用）。

    队列 + visited（这里用 dist==-1 兼作访问标记）：
      - 出队一个点 u，遍历其邻居 v；
      - v 没访问过 → dist[v] = dist[u]+1 并入队（BFS 保证首次到达即最短）。

    变量说明：
      - g: 邻接表；start: 起点
      - n: 节点数；dist: 距离数组（-1 表示不可达/未访问）
      - q: 队列；u: 出队节点；v: u 的邻居

    循环终止条件：
      - 队列为空（所有可达节点都处理完）。

    复杂度：
      - 时间 O(V+E)，空间 O(V)。
    """
    n = len(g)                         # 节点数
    dist = [-1] * n                    # dist[v]: start 到 v 的最短距离，-1 未访问
    dist[start] = 0                    # 起点距离 0
    q = deque([start])                 # 起点入队
    while q:                           # 队列非空
        u = q.popleft()                # 出队当前节点
        for v in g[u]:                 # 遍历邻居
            if dist[v] == -1:          # 首次访问 → 距离最短
                dist[v] = dist[u] + 1  # 距离 = 父距离 + 1
                q.append(v)            # 入队待扩展
    return dist                        # 各点最短距离（-1 不可达）


# ============ 2. DFS（连通分量/回溯遍历） ============
def dfs(g, start):
    """从 start 出发的 DFS 遍历，返回访问顺序。

    递归 + seen 标记，防止重复访问和死循环（无向图尤其需要）。

    变量说明：
      - g: 邻接表；start: 起点
      - seen: 是否已访问；order: 访问顺序
      - u: 当前节点；v: 邻居

    递归终止条件：
      - 当前节点的所有邻居都已访问（或没有邻居）时自然返回。

    复杂度：
      - 时间 O(V+E)，空间 O(V)（栈 + seen）。
    """
    seen = [False] * len(g)            # 访问标记
    order = []                         # 访问顺序

    def rec(u):
        seen[u] = True                 # 标记当前节点已访问
        order.append(u)                # 记录访问顺序
        for v in g[u]:                 # 遍历邻居
            if not seen[v]:            # 邻居未访问才深入
                rec(v)                 # 递归
    rec(start)                         # 从起点开始
    return order


# ============ 3. 拓扑排序（207/210；BFS Kahn 算法） ============
def topo_sort(n, edges):
    """拓扑排序（Kahn 算法：BFS + 入度）；有环返回 []。

    思路：反复删除"入度为 0"的点，它们可以安全排在前面；
    每删除一个点，把它所有后继的入度减 1，减到 0 的后继入队。
    最终若输出节点数 == n 说明无环；否则有环（环上节点入度永不为 0）。

    变量说明：
      - n: 节点数；edges: 有向边 (u, v)，u 在 v 之前
      - g: 邻接表；indeg: 各点入度
      - q: 入度为 0 的节点队列；order: 拓扑序
      - u: 出队节点；v: u 的后继

    循环终止条件：
      - 队列为空。

    复杂度：
      - 时间 O(V+E)，空间 O(V+E)。
    """
    g = [[] for _ in range(n)]         # 邻接表
    indeg = [0] * n                    # 入度数组
    for u, v in edges:                 # 建图并统计入度
        g[u].append(v)                 # u → v
        indeg[v] += 1                  # v 的入度加 1
    q = deque([i for i in range(n) if indeg[i] == 0])  # 入度为 0 的先入队
    order = []                         # 拓扑序结果
    while q:                           # 队列非空
        u = q.popleft()                # 取出入度为 0 的节点
        order.append(u)                # 排进拓扑序
        for v in g[u]:                 # 删除 u 后更新其后继
            indeg[v] -= 1              # 后继入度减 1
            if indeg[v] == 0:          # 入度归零 → 可以排队
                q.append(v)
    return order if len(order) == n else []  # 数量不够 → 有环，返回空


# ============ 4. 图是否二分（染色 BFS，785） ============
def is_bipartite(g):
    """判断无向图是否二分图（LeetCode 785）：能否用 2 种颜色染所有点，
    使任意一条边的两端颜色不同。

    对每个连通分量做 BFS 染色：从某点染 0 开始，邻居必须染 1-color[u]；
    若邻居已染且颜色相同 → 冲突，不是二分图。

    变量说明：
      - g: 邻接表；color: 染色数组（-1 未染，0/1 两种颜色）
      - i: 连通分量起点；u: 出队节点；v: 邻居

    循环终止条件：
      - 所有连通分量处理完，或发现冲突提前返回 False。

    复杂度：
      - 时间 O(V+E)，空间 O(V)。
    """
    color = [-1] * len(g)              # -1 未染，0/1 两色
    for i in range(len(g)):            # 遍历每个节点（覆盖所有连通分量）
        if color[i] != -1:             # 已染色 → 属于处理过的分量
            continue
        color[i] = 0                   # 新分量起点染 0
        q = deque([i])                 # 入队
        while q:                       # BFS 染色
            u = q.popleft()
            for v in g[u]:             # 邻居必须与 u 颜色相反
                if color[v] == -1:     # 未染 → 染相反色
                    color[v] = 1 - color[u]
                    q.append(v)        # 入队继续
                elif color[v] == color[u]:  # 已染且与 u 相同 → 冲突
                    return False       # 不是二分图
    return True                        # 全部染色成功


# ============ 5. Dijkstra 最短路径（手写堆，O(E log V)；无负权） ============
def dijkstra(g, start):
    """Dijkstra 单源最短路（LeetCode 743 等；要求边权非负）。

    贪心 + 优先队列（小顶堆）：每次取出"当前距离最小的未确定点"，
    用它松弛所有邻居；dist 首次被更新即可能不是最终值，所以用
    "d > dist[u] 跳过"的过期条目处理法（懒删除）。

    变量说明：
      - g: 带权邻接表，g[u] = [(邻居, 权重), ...]
      - start: 起点；n: 节点数
      - dist: 起点到各点的最短距离（inf 表示不可达）
      - pq: 小顶堆，(当前距离, 节点)；d: 出堆距离；u: 出堆节点
      - v / w: 邻居 / 边权；nd: 经 u 到 v 的新距离

    循环终止条件：
      - 堆为空（所有可达点都已确定）。

    复杂度：
      - 时间 O(E log V)，空间 O(V+E)。
    """
    import heapq
    n = len(g)                         # 节点数
    dist = [float('inf')] * n          # 初始全不可达
    dist[start] = 0                    # 起点距离 0
    pq = [(0, start)]                  # 小顶堆 (距离, 节点)
    while pq:                          # 堆非空
        d, u = heapq.heappop(pq)       # 取当前距离最小的节点
        if d > dist[u]:                # 过期条目（已通过更短路径更新过）
            continue                   # 跳过
        for v, w in g[u]:              # 松弛所有邻居
            nd = d + w                 # 经 u 到 v 的距离
            if nd < dist[v]:           # 更短 → 更新并入堆
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist                        # 各点最短距离


# ============ 6. 多源 BFS / 0-1 BFS（边权 0/1 时用 deque） ============
def bfs_01(g0, g1, start):
    """0-1 BFS：边权只有 0 或 1 时的最短路（比 Dijkstra 更快，O(V+E)）。

    用双端队列替代优先队列：0 权边松弛成功后从**队首**插入
    （等价于同层，先处理），1 权边从**队尾**插入（下一层）。

    变量说明：
      - g0 / g1: 0 权 / 1 权 邻接表（同一编号节点分别记录两类出边）
      - start: 起点；n: 节点数
      - dist: 最短距离（inf 不可达）；dq: 双端队列
      - u: 出队节点；v: 邻居

    循环终止条件：
      - 队列为空。

    复杂度：
      - 时间 O(V+E)，空间 O(V)。
    """
    import math
    n = len(g0)                        # 节点数
    dist = [math.inf] * n              # 最短距离，初始不可达
    dist[start] = 0                    # 起点距离 0
    dq = deque([start])                # 双端队列
    while dq:                          # 队列非空
        u = dq.popleft()               # 取出队首
        for v in g0[u]:                # 0 权边
            if dist[v] > dist[u]:      # 能更短
                dist[v] = dist[u]      # 距离不变（0 权）
                dq.appendleft(v)       # 插到队首（同一层优先处理）
        for v in g1[u]:                # 1 权边
            if dist[v] > dist[u] + 1:  # 能更短
                dist[v] = dist[u] + 1  # 距离 +1
                dq.append(v)           # 插到队尾（下一层）
    return dist                        # 各点最短距离


# ============ 测试 ============
if __name__ == '__main__':
    g = build_adj(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
    assert bfs(g, 0) == [0, 1, 2, 1]      # 0→1/3 距离 1，2 距离 2
    assert dfs(g, 0) == [0, 1, 2, 3]
    assert topo_sort(4, [(0, 1), (1, 2), (3, 2)]) == [0, 3, 1, 2]
    assert is_bipartite([[1, 3], [0, 2], [1, 3], [0, 2]])     # 环四染色成功
    gw = [[(1, 5)], [(2, 3), (3, 1)], [(0, 2)], []]
    assert dijkstra(gw, 0) == [0, 5, 8, 6]    # 0→1:5, 0→2:5+3=8, 0→3:5+1=6
    print('✅ 图模板全家桶测试通过')
