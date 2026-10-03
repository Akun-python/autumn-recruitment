# -*- coding: utf-8 -*-
"""05-二叉树模板.py —— P0 必背（遍历是所有树题的基础）
配套教学：../教学/06-二叉树.ipynb
要点：递归版背熟；迭代版掌握"前序+中序统一栈法"与"后序两栈/标记法"。

本文件包含：
  1. 递归前中后序遍历（三行版）
  2. 迭代前中后序遍历（统一"标记法"，一套代码改三处）
  3. BFS 层序遍历
  4. 深度 / 平衡判断
  5. 最近公共祖先 LCA
  6. 二叉搜索树：验证 / 第 k 小
  7. 路径问题：最大路径和
"""


class TreeNode:
    """二叉树节点。

    变量说明：
      - val: 节点值
      - left: 左孩子；right: 右孩子（None 表示无孩子）
    """
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build(arr, i=0):
    """按"层序数组"建树（LeetCode 风格：None 表示空位，下标 2i+1/2i+2）。

    变量说明：
      - arr: 层序数组，arr[i] 为 None 表示该位置没有节点
      - i: 当前节点在数组中的下标
      - node: 新建的当前节点

    过程拆解：
      1. 递归基：i 越界或 arr[i] 是 None → 该位置没有节点，返回 None；
      2. 建当前节点，左孩子在 2i+1、右孩子在 2i+2（完全二叉树下标规则）；
      3. 递归建左右子树。

    递归终止条件：
      - i >= len(arr) 或 arr[i] is None。

    复杂度：
      - 时间 O(n)，空间 O(n)（递归栈 + 节点）。
    """
    if i >= len(arr) or arr[i] is None:  # 越界或空位
        return None
    node = TreeNode(arr[i])            # 建当前节点
    node.left = build(arr, 2 * i + 1)  # 左孩子在 2i+1
    node.right = build(arr, 2 * i + 2) # 右孩子在 2i+2
    return node


# ============ 1. DFS 前中后序（递归三行版） ============
def preorder(root):
    """前序遍历：根 → 左 → 右（LeetCode 144）。

    递归基：root 为空返回空列表；否则先取根值，再拼左、右子树结果。

    递归终止条件：root 为 None。

    复杂度：时间 O(n)，空间 O(n)（最坏退化为链时递归栈 O(n)）。
    """
    return [root.val] + preorder(root.left) + preorder(root.right) if root else []


def inorder(root):
    """中序遍历：左 → 根 → 右（LeetCode 94），BST 中序即升序。

    递归终止条件：root 为 None。

    复杂度：时间 O(n)，空间 O(n)。
    """
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []


def postorder(root):
    """后序遍历：左 → 右 → 根（LeetCode 145）。

    递归终止条件：root 为 None。

    复杂度：时间 O(n)，空间 O(n)。
    """
    return postorder(root.left) + postorder(root.right) + [root.val] if root else []


# ============ 2. 前序 + 中序：统一迭代（标记法，背这个） ============
def preorder_iter(root):
    """迭代前序遍历（标记法）。

    标记法思路：栈里放 (节点, 是否已访问)。第一次遇到节点先压入其
    右、左孩子（因为栈是后进先出，先压右再压左 → 左先出），最后压
    (node, True) 标记"下次弹出时输出值"。
    - 前序：标记放最后压 → 最先被弹出输出（根 → 左 → 右）。

    变量说明：
      - out: 遍历结果；stack: 栈，元素是 (节点, 是否已访问)
      - node: 当前弹出节点；visited: 是否已访问过

    循环终止条件：
      - stack 为空。

    复杂度：
      - 时间 O(n)，空间 O(n)。
    """
    out, stack = [], [(root, False)]   # 栈中存 (节点, 是否已访问)
    while stack:                       # 栈非空继续
        node, visited = stack.pop()    # 弹出栈顶
        if node is None:               # 空节点跳过
            continue
        if visited:                    # 已访问过 → 输出值
            out.append(node.val)
        else:                          # 首次遇到 → 按"后进先出"压入子节点
            stack.append((node.right, False))  # 先压右
            stack.append((node.left, False))   # 再压左（左先出）
            stack.append((node, True))         # 前序：标记放最后压 → 先出
    return out


def inorder_iter(root):
    """迭代中序遍历（标记法）：左 → 根 → 右。

    与前序的区别只在压栈顺序：把 (node, True) 放到左孩子之后压入，
    使得"左孩子 → 根 → 右孩子"的出栈顺序 = 中序。

    循环终止条件：stack 为空。

    复杂度：时间 O(n)，空间 O(n)。
    """
    out, stack = [], [(root, False)]
    while stack:
        node, visited = stack.pop()
        if node is None:
            continue
        if visited:
            out.append(node.val)
        else:
            stack.append((node.right, False))  # 右最后处理
            stack.append((node, True))         # 中序：标记在中间 → 根在左之后输出
            stack.append((node.left, False))   # 左最先处理
    return out


def postorder_iter(root):
    """迭代后序遍历（标记法）：左 → 右 → 根。

    与前序的区别：把 (node, True) 最先压入（最后弹出输出），
    再压右、左孩子 → 输出顺序正好是 左 → 右 → 根。

    循环终止条件：stack 为空。

    复杂度：时间 O(n)，空间 O(n)。
    """
    out, stack = [], [(root, False)]
    while stack:
        node, visited = stack.pop()
        if node is None:
            continue
        if visited:
            out.append(node.val)
        else:
            stack.append((node, True))         # 后序：标记最先压 → 最后输出
            stack.append((node.right, False))  # 右孩子
            stack.append((node.left, False))   # 左孩子（先出）
    return out


# ============ 3. BFS 层序遍历 ============
from collections import deque


def level_order(root):
    """层序遍历，返回二维列表 [[第一层], [第二层], ...]（LeetCode 102）。

    BFS 队列：每次把"当前队列长度"个节点一次性取出，就是完整的一层。

    变量说明：
      - out: 结果二维列表；q: 队列（存下一层要处理的节点）
      - level: 当前层的节点值列表；node: 当前出队节点

    过程拆解：
      1. 根节点入队；
      2. 每轮先记 len(q)（本层节点数），循环取这么多个节点：
         - 收集节点值到 level；
         - 把左右孩子（若非空）入队，作为下一层；
      3. level 加入 out。

    循环终止条件：
      - q 为空（所有层都处理完）。

    复杂度：
      - 时间 O(n)，空间 O(n)（队列最多存一层的节点）。
    """
    if not root:                       # 空树
        return []
    out, q = [], deque([root])         # 结果列表 + 初始队列
    while q:                           # 队列非空继续
        level = []                     # 本层节点值
        for _ in range(len(q)):        # 只处理"当前层"的节点数
            node = q.popleft()         # 出队
            level.append(node.val)     # 收集本层值
            if node.left:              # 左孩子入队
                q.append(node.left)
            if node.right:             # 右孩子入队
                q.append(node.right)
        out.append(level)              # 本层结果加入
    return out


# ============ 4. 二叉树的深度 / 是否平衡 ============
def max_depth(root):
    """二叉树最大深度（LeetCode 104）。

    递归：空树深度 0；否则 1 + max(左子树深度, 右子树深度)。

    递归终止条件：root 为 None。

    复杂度：时间 O(n)，空间 O(n)（递归栈）。
    """
    return 0 if not root else 1 + max(max_depth(root.left), max_depth(root.right))


def is_balanced(root):
    """判断是否平衡二叉树（LeetCode 110）：任意节点左右子树高度差 <= 1。

    用后序递归同时返回 (子树高度, 子树是否平衡)，避免重复计算高度。

    变量说明：
      - dfs(node) 返回 (高度, 是否平衡)
      - hl / bl: 左子树 (高度, 平衡否)；hr / br: 右子树 (高度, 平衡否)

    递归终止条件：node 为 None → (0, True)（空树高 0 且平衡）。

    复杂度：时间 O(n)（每个节点只访问一次），空间 O(n)。
    """
    def dfs(node):
        if not node:                   # 空树：高度 0，平衡
            return 0, True
        hl, bl = dfs(node.left)        # 左子树高度与平衡性
        hr, br = dfs(node.right)       # 右子树高度与平衡性
        # 高度 = 1 + 最大子树高；平衡 = 左右都平衡 且 高度差 <= 1
        return 1 + max(hl, hr), bl and br and abs(hl - hr) <= 1
    return dfs(root)[1]                # 只取"是否平衡"


# ============ 5. 最近公共祖先 LCA（236，必背） ============
def lowest_common_ancestor(root, p, q):
    """最近公共祖先（LeetCode 236）：p、q 一定在树中。

    后序递归：每棵子树向上返回"找到了 p 或 q 的那个节点"（或 None）。
      - 若某节点左右子树各返回一个非 None → p、q 分居两侧 → 该节点是 LCA；
      - 若只有一侧非 None → 该侧找到的就是 LCA（祖先关系向上传递）。

    变量说明：
      - p / q: 要找祖先的两个节点
      - left / right: 左右子树递归返回的"命中节点或 None"

    递归终止条件：
      - root 为 None（探到底）；
      - root 是 p 或 q（命中目标，向上返回）。

    复杂度：时间 O(n)，空间 O(n)（递归栈）。
    """
    if root is None or root is p or root is q:  # 空 或 命中 p/q
        return root                    # 返回命中节点（或 None）
    left = lowest_common_ancestor(root.left, p, q)   # 在左子树找
    right = lowest_common_ancestor(root.right, p, q) # 在右子树找
    if left and right:                 # 两边都命中 → p、q 分居两侧
        return root                    # root 就是最近公共祖先
    return left or right               # 只有一侧命中 → 该侧结果向上传


# ============ 6. 二叉搜索树：验证 / 第 k 小 / 插入删除 ============
def is_bst(root, lo=float('-inf'), hi=float('inf')):
    """验证二叉搜索树（LeetCode 98）：左 < 根 < 右（严格）。

    递归下传上下界：左子树所有值必须在 (lo, root.val) 内，
    右子树所有值必须在 (root.val, hi) 内。

    变量说明：
      - lo / hi: 当前节点允许的取值范围（开区间）
      - root: 当前节点

    递归终止条件：
      - root 为 None → True；
      - 值不在 (lo, hi) 内 → False（提前剪枝）。

    复杂度：时间 O(n)，空间 O(n)。
    """
    if not root:                       # 空树是 BST
        return True
    if not (lo < root.val < hi):       # 超出允许范围
        return False
    # 左子树范围收紧为 (lo, root.val)，右子树收紧为 (root.val, hi)
    return is_bst(root.left, lo, root.val) and is_bst(root.right, root.val, hi)


def kth_smallest(root, k):
    """二叉搜索树中第 k 小的元素（LeetCode 230）。

    BST 中序遍历 = 升序 → 中序走到第 k 个就是答案。
    迭代版：用栈模拟中序（先一路压左，再弹出、计数、转向右）。

    变量说明：
      - stack: 模拟中序的栈；cur: 当前节点
      - k: 剩余需要找的数量（每输出一个减 1）

    循环终止条件：
      - stack 空且 cur 空（遍历完）或 k 减到 0（找到答案）。

    复杂度：时间 O(H+k)（H 为树高），空间 O(H)。
    """
    stack, cur = [], root              # 栈 + 当前节点
    while stack or cur:                # 还有节点可处理
        while cur:                     # 一路向左压栈
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()              # 弹出最左节点（当前最小）
        k -= 1                         # 计数减 1
        if k == 0:                     # 正好是第 k 个
            return cur.val
        cur = cur.right                # 转向右子树继续中序
    return -1                          # k 超过节点数（按题意不会发生）


# ============ 7. 路径问题：最大路径和（124） / 直径（543） ============
def max_path_sum(root):
    """二叉树中的最大路径和（LeetCode 124）。

    路径可以经过任意节点、任意起止（不一定经过根），但每条边最多走一次。
    后序递归，对每个节点计算"过该节点的最大路径和"并更新全局最优。

    变量说明：
      - best: 全局最大路径和（用单元素列表实现闭包修改）
      - dfs(node): 返回"从 node 向下延伸的最大单边和"（node 作为路径一端）
      - l / r: 左、右子树返回的最大单边和（负数剪成 0，不贡献）

    过程拆解：
      1. 空节点返回 0；
      2. l = max(左子树单边和, 0)：负数贡献直接剪掉（不选左子树）；
         r 同理；
      3. best = max(best, node.val + l + r)：过 node、左右都选的完整路径；
      4. 向上返回 node.val + max(l, r)：只能选单边向上延伸。

    递归终止条件：node 为 None → 返回 0。

    复杂度：时间 O(n)，空间 O(n)。
    """
    best = [float('-inf')]             # 全局最优（初始为负无穷）

    def dfs(node):
        if not node:                   # 空节点：贡献 0
            return 0
        l = max(dfs(node.left), 0)     # 左子树单边和，负数剪掉
        r = max(dfs(node.right), 0)    # 右子树单边和，负数剪掉
        best[0] = max(best[0], node.val + l + r)  # 过 node 的完整路径
        return node.val + max(l, r)    # 向上只提供"单边"最大贡献
    dfs(root)                          # 后序遍历整棵树
    return best[0]                     # 返回全局最优


# ============ 测试 ============
if __name__ == '__main__':
    root = build([3, 9, 20, None, None, 15, 7])
    assert preorder(root) == [3, 9, 20, 15, 7]
    assert inorder(root) == [9, 3, 15, 20, 7]
    assert postorder(root) == [9, 15, 7, 20, 3]
    assert preorder_iter(root) == preorder(root)     # 迭代与递归结果一致
    assert inorder_iter(root) == inorder(root)
    assert postorder_iter(root) == postorder(root)
    assert level_order(root) == [[3], [9, 20], [15, 7]]  # 层序遍历
    assert max_depth(root) == 3
    assert is_balanced(root)
    bst = build([4, 2, 5, 1, 3])
    assert is_bst(bst) and kth_smallest(bst, 3) == 3
    assert max_path_sum(root) == 47          # 9+3+20+15（过根 3 的最大路径）
    assert max_path_sum(build([-10, 9, 20, None, None, 15, 7])) == 42   # 124 官方示例
    print('✅ 二叉树模板全家桶测试通过')
