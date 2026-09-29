# -*- coding: utf-8 -*-
"""05-二叉树模板.py —— P0 必背（遍历是所有树题的基础）
配套教学：../教学/06-二叉树.ipynb
要点：递归版背熟；迭代版掌握"前序+中序统一栈法"与"后序两栈/标记法"。
"""


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build(arr, i=0):
    """按层序数组建树（None 表示空位）"""
    if i >= len(arr) or arr[i] is None:
        return None
    node = TreeNode(arr[i])
    node.left = build(arr, 2 * i + 1)
    node.right = build(arr, 2 * i + 2)
    return node


# ============ 1. DFS 前中后序（递归三行版） ============
def preorder(root):
    return [root.val] + preorder(root.left) + preorder(root.right) if root else []


def inorder(root):
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []


def postorder(root):
    return postorder(root.left) + postorder(root.right) + [root.val] if root else []


# ============ 2. 前序 + 中序：统一迭代（标记法，背这个） ============
def preorder_iter(root):
    out, stack = [], [(root, False)]
    while stack:
        node, visited = stack.pop()
        if node is None:
            continue
        if visited:
            out.append(node.val)
        else:
            stack.append((node.right, False))
            stack.append((node.left, False))
            stack.append((node, True))    # 前序：标记放在最后压 → 先出
    return out


def inorder_iter(root):
    out, stack = [], [(root, False)]
    while stack:
        node, visited = stack.pop()
        if node is None:
            continue
        if visited:
            out.append(node.val)
        else:
            stack.append((node.right, False))
            stack.append((node, True))    # 中序：标记在中间
            stack.append((node.left, False))
    return out


def postorder_iter(root):
    out, stack = [], [(root, False)]
    while stack:
        node, visited = stack.pop()
        if node is None:
            continue
        if visited:
            out.append(node.val)
        else:
            stack.append((node, True))    # 后序：标记最先压 → 最后出
            stack.append((node.right, False))
            stack.append((node.left, False))
    return out


# ============ 3. BFS 层序遍历 ============
from collections import deque


def level_order(root):
    if not root:
        return []
    out, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):           # 按层取
            node = q.popleft()
            level.append(node.val)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        out.append(level)
    return out


# ============ 4. 二叉树的深度 / 是否平衡 ============
def max_depth(root):
    return 0 if not root else 1 + max(max_depth(root.left), max_depth(root.right))


def is_balanced(root):
    """后序返回 (高度, 是否平衡)，O(n)"""
    def dfs(node):
        if not node:
            return 0, True
        hl, bl = dfs(node.left)
        hr, br = dfs(node.right)
        return 1 + max(hl, hr), bl and br and abs(hl - hr) <= 1
    return dfs(root)[1]


# ============ 5. 最近公共祖先 LCA（236，必背） ============
def lowest_common_ancestor(root, p, q):
    if root is None or root is p or root is q:
        return root                       # 命中目标或空
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root                       # 分居两侧 → root 是 LCA
    return left or right


# ============ 6. 二叉搜索树：验证 / 第 k 小 / 插入删除 ============
def is_bst(root, lo=float('-inf'), hi=float('inf')):
    if not root:
        return True
    if not (lo < root.val < hi):
        return False
    return is_bst(root.left, lo, root.val) and is_bst(root.right, root.val, hi)


def kth_smallest(root, k):
    """中序第 k 个（迭代，O(H+k)）"""
    stack, cur = [], root
    while stack or cur:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        k -= 1
        if k == 0:
            return cur.val
        cur = cur.right
    return -1


# ============ 7. 路径问题：最大路径和（124） / 直径（543） ============
def max_path_sum(root):
    best = [float('-inf')]

    def dfs(node):
        if not node:
            return 0
        l = max(dfs(node.left), 0)        # 负贡献剪掉
        r = max(dfs(node.right), 0)
        best[0] = max(best[0], node.val + l + r)   # 过 node 的路径
        return node.val + max(l, r)                # 向上提供的最大单边
    dfs(root)
    return best[0]


# ============ 测试 ============
if __name__ == '__main__':
    root = build([3, 9, 20, None, None, 15, 7])
    assert preorder(root) == [3, 9, 20, 15, 7]
    assert inorder(root) == [9, 3, 15, 20, 7]
    assert postorder(root) == [9, 15, 7, 20, 3]
    assert preorder_iter(root) == preorder(root)
    assert inorder_iter(root) == inorder(root)
    assert postorder_iter(root) == postorder(root)
    assert level_order(root) == [[3], [9, 20], [15, 7]]
    assert max_depth(root) == 3
    assert is_balanced(root)
    bst = build([4, 2, 5, 1, 3])
    assert is_bst(bst) and kth_smallest(bst, 3) == 3
    assert max_path_sum(root) == 47          # 9+3+20+15（过根 3 的最大路径）
    assert max_path_sum(build([-10, 9, 20, None, None, 15, 7])) == 42   # 124 官方示例
    print('✅ 二叉树模板全家桶测试通过')