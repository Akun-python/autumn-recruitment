# -*- coding: utf-8 -*-
"""08-图模板.py —— P0 必背（BFS/DFS/拓扑/Dijkstra）
配套教学：../教学/10-图.ipynb
图的三种存储：邻接表（默认，List[List[int]]）/ 邻接矩阵 / 边表。
"""

from collections import deque


# ============ 图的基本存储：邻接表 ============
# n 个节点，edges = [(u, v), ...]（无向图需双向加）
def build_adj(n, edges, directed=False):
    g = [[] for _ in range(n)]
    for u, v in edges:
        g[u].append(v)
        if not directed:
            g[v].append(u)
    return g


# ============ 1. BFS（最短路径/层序，O(V+E)） ============
def bfs(g, start):
    """返回从 start 到各点的最短距离（无权重）"""
    n = len(g)
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        for v in g[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


# ============ 2. DFS（连通分量/回溯遍历） ============
def dfs(g, start):
    seen = [False] * len(g)
    order = []

    def rec(u):
        seen[u] = True
        order.append(u)
        for v in g[u]:
            if not seen[v]:
                rec(v)
    rec(start)
    return order


# ============ 3. 拓扑排序（207/210；BFS Kahn 算法） ============
def topo_sort(n, edges):
    """返回拓扑序；若有环返回 []"""
    g = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        g[u].append(v)
        indeg[v] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else []


# ============ 4. 图是否二分（染色 BFS，785） ============
def is_bipartite(g):
    color = [-1] * len(g)                 # -1 未染，0/1 两色
    for i in range(len(g)):
        if color[i] != -1:
            continue
        color[i] = 0
        q = deque([i])
        while q:
            u = q.popleft()
            for v in g[u]:
                if color[v] == -1:
                    color[v] = 1 - color[u]
                    q.append(v)
                elif color[v] == color[u]:
                    return False
    return True


# ============ 5. Dijkstra 最短路径（手写堆，O(E log V)；无负权） ============
def dijkstra(g, start):
    """g: 邻接表存 (邻居, 权重)"""
    import heapq
    n = len(g)
    dist = [float('inf')] * n
    dist[start] = 0
    pq = [(0, start)]                     # (距离, 节点)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue                      # 过期条目
        for v, w in g[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# ============ 6. 多源 BFS / 0-1 BFS（边权 0/1 时用 deque） ============
def bfs_01(g0, g1, start):
    """g0/g1: 0 权/1 权邻接表；双端队列：0 权前插、1 权后插"""
    import math
    n = len(g0)
    dist = [math.inf] * n
    dist[start] = 0
    dq = deque([start])
    while dq:
        u = dq.popleft()
        for v in g0[u]:
            if dist[v] > dist[u]:
                dist[v] = dist[u]
                dq.appendleft(v)
        for v in g1[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


# ============ 测试 ============
if __name__ == '__main__':
    g = build_adj(4, [(0, 1), (1, 2), (2, 3), (0, 3)])
    assert bfs(g, 0) == [0, 1, 2, 1]      # 0→1/3 距离 1，2 距离 2
    assert dfs(g, 0) == [0, 1, 2, 3]
    assert topo_sort(4, [(0, 1), (1, 2), (3, 2)]) == [0, 3, 1, 2]
    assert is_bipartite([[1, 3], [0, 2], [1, 3], [0, 2]])
    gw = [[(1, 5)], [(2, 3), (3, 1)], [(0, 2)], []]
    assert dijkstra(gw, 0) == [0, 5, 8, 6]
    print('✅ 图模板全家桶测试通过')