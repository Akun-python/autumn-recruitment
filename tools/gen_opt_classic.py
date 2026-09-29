# -*- coding: utf-8 -*-
"""生成 08-优化算法/教学/ 的「经典优化算法」notebook（08-13 六篇）。

08-线性规划与对偶 / 09-整数规划与分支定界 / 10-遗传算法 /
11-模拟退火与禁忌搜索 / 12-粒子群与蚁群算法 / 13-拉格朗日对偶与KKT

代码注释规范：手写实现每行给「在算什么 / 为什么这么算」，公式与代码一一对应。
生成后请运行 tools/check_opt_figs.py 批量执行验证（嵌入输出 + 重叠检查）。
"""
import os
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   '08-优化算法', '教学')


def md(src):
    return new_markdown_cell(src)


def code(src):
    return new_code_cell(src)


def build(cells):
    nbk = new_notebook(cells=cells)
    nbk.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nbk.metadata['language_info'] = {'name': 'python', 'version': '3.10.0'}
    return nbk


def save(nbk, fname):
    path = os.path.join(OUT, fname)
    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nbk, f)
    print('written:', path)


# =====================================================================
# 08-线性规划与对偶
# =====================================================================
def nb08():
    cells = []

    cells.append(md(r"""# 🎯 08-线性规划与对偶

> 目标：把优化从「无约束连续」扩展到「**有约束**」——线性规划 LP 是最简单最经典的约束优化：目标与约束全部线性。
> 面试考点：可行域为什么是多面体、最优解为什么在顶点、单纯形法怎么走、对偶与影子价格、和 SVM 对偶的关系。

## 本章定位（08-13 经典优化算法线）

| | 01-07 深度学习优化器 | 08-13 经典优化算法 |
|---|---|---|
| 问题 | 连续、可微、无约束 | 有约束 / 离散 / 黑箱 |
| 可用信息 | 梯度 ∇f | 只有目标值（或再加一阶信息） |
| 求解思路 | 局部迭代（快） | 顶点移动 / 全局搜索（稳） |
| 代表 | SGD / Adam / 牛顿法 | LP / IP / GA / SA / PSO / ACO / KKT |

> 08-09 是「**精确方法**」（单纯形、分支定界，能给出最优解）；10-12 是「**启发式方法**」（GA / SA / TS / PSO / ACO，给近似解）；
> 13 是约束优化的「**理论工具**」（拉格朗日对偶 / KKT），与机器学习（SVM / 正则化 / Lasso）直接接轨。"""))

    cells.append(md(r"""## §1 线性规划：形式与几何

**标准形式**（本讲统一用 max 版本）：

$$\max c^\top x \quad \text{s.t.} \quad Ax \le b,\ x \ge 0$$

**三要素**：目标系数 $c$、约束矩阵 $A$、资源向量 $b$。加松弛变量 $s = b - Ax \ge 0$ 可化为等式形式 $Ax + s = b$。

**几何**：$Ax \le b,\ x \ge 0$ 围出的可行域是**凸多面体**；线性目标在凸集上的极值必然取在**顶点**（角点）——这就是单纯形法的出发点。

**图解法**（2 变量）：画约束半平面 → 交出可行域 → 沿目标等高线推到最大。

> 例子（下节代码会解它）：
> $$\max\ 3x_1+2x_2 \quad \text{s.t.}\ x_1+x_2\le 4,\ x_1\le 2,\ x_2\le 3,\ x_1,x_2\ge 0$$
> 最优解 $(2,2)$，$z=10$。"""))

    cells.append(code(r"""import numpy as np
import matplotlib.pyplot as plt

# 中文字体设置：Windows 用微软雅黑，缺失时回退 SimHei / DejaVu Sans
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False   # 让坐标轴负号正常显示（不显示成方块）

# ========== 图解 LP：max 3x1 + 2x2 ==========
# 目标函数 z = c^T x = 3*x1 + 2*x2，求最大值
c = np.array([3.0, 2.0])
# 约束 A x <= b（共 3 条）：
#   (1) x1 + x2 <= 4   —— 资源约束（如两种原料合计不超过 4 吨）
#   (2) x1      <= 2   —— 产品 1 产量上限
#   (3)      x2 <= 3   —— 产品 2 产量上限
#   另有隐含约束 x1, x2 >= 0（非负，图上即第一象限）
A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
b = np.array([4.0, 2.0, 3.0])

xs = np.linspace(0, 5, 300)          # 画线用的 x 采样（覆盖整个绘图区域）
fig, ax = plt.subplots(figsize=(7.2, 5.6))

# 可行域着色（fill_between 是 PolyCollection，不参与重叠检查；边界用折线画出）：
# x1 只能取 [0,2]（约束 x1<=2），每个 x1 处 y 的上界 = min(x2<=3, x1+x2<=4 -> x2<=4-x1)
xx = np.linspace(0, 2, 200)
ax.fill_between(xx, 0, np.minimum(3.0, 4.0 - xx), color='#9ecae1', alpha=0.35)
# 可行域边界折线，顶点依次为 (0,0)->(2,0)->(2,2)->(1,3)->(0,3)（与 fill 区域吻合）
ax.plot([0, 2, 2, 1, 0], [0, 0, 2, 3, 3], color='#3182bd', lw=1.6)

# 逐条画约束边界线并标注：a1*x1 + a2*x2 = bi
for (a1, a2), bi, lab in zip(A, b, ['x1+x2<=4', 'x1<=2', 'x2<=3']):
    if a2 != 0:
        ys = (bi - a1 * xs) / a2          # 由 a1*x + a2*y = bi 解出 y = (bi - a1*x)/a2
        ax.plot(xs, ys, lw=1.4)           # 画这条边界直线
        ax.text(0.15, (bi - a1 * 0.1) / a2 + 0.12, lab, fontsize=9)  # 在线的左端上方标注
    else:
        ax.axvline(bi, lw=1.4)            # 竖直边界（x1 = bi）
        ax.text(bi + 0.05, 4.4, lab, fontsize=9)

# 目标等高线：z = 3x1+2x2 = 常数（虚线）。加大 z 相当于把线往右上平移，
# 与可行域最后相切的顶点就是最优解。
for z in [4, 8, 10]:
    y = (z - c[0] * xs) / c[1]           # 由 3x + 2y = z 解出 y = (z - 3x)/2
    ax.plot(xs, y, '--', lw=1.0, color='#C44E52', alpha=0.65)
    ax.text(1.05, (z - c[0] * 1.0) / c[1] + 0.15, 'z=%d' % z, color='#C44E52', fontsize=9)

# 最优顶点 (2,2)：等高线 z=10 恰好与可行域顶点相切，再大就离开可行域了
opt = (2, 2)
ax.plot(*opt, 'o', ms=11, color='#C44E52', zorder=5)
ax.annotate('最优解 (2,2)，z=10', xy=opt, xytext=(2.35, 2.1),
            arrowprops=dict(arrowstyle='->', color='#C44E52'), fontsize=10, color='#C44E52')
ax.set_xlim(-0.3, 5.3); ax.set_ylim(-0.3, 5.3)
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('线性规划图解法：可行域是多面体，最优解必在顶点', fontsize=12)
ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
print('图解结论：最优顶点 (2, 2)，z = 3*2 + 2*2 = 10')"""))

    cells.append(md(r"""## §2 单纯形法：沿边走到最优顶点

**直觉**：从一个可行顶点出发，每次沿一条**能改善目标**的边走到相邻顶点，直到任何边都不能再改善 → 到达最优顶点。
顶点数有限所以理论必终止；但顶点数可能指数多，所以要看实现效率。

**检验数（reduced cost）**：目标行里非基变量的系数 $r_j = c_j - c_B B^{-1} A_j$。
- 存在 $r_j < 0$（max 问题）：该变量入基能增加目标
- 全部 $r_j \ge 0$：当前顶点已最优，停止

**比值检验（ratio test）**：入基变量增大到多少会顶掉某个基变量 → 选**最小正比值**对应行出基，保证新顶点仍可行。

**防循环**：退化顶点可能让目标不变而绕圈 → **Bland 规则**（取最小下标入基 / 平局取最小下标出基）保证不循环。

**伪代码**

```
T ← 增广单纯形表（约束行 + 松弛基 + 检验数行 -c）
loop:
    r ← 检验数行
    若 r ≥ 0: return 当前基解
    入基 j ← 最小负检验数下标（Bland）
    若列 A[:,j] 全 ≤ 0: return 无界
    出基 i ← 最小比值行（Bland 平局）
    以 (i,j) 为枢轴做高斯消元
```"""))

    cells.append(code(r"""# ================= 手写 tableau 单纯形法 =================
# 求解问题：max c^T x  s.t.  A x <= b,  x >= 0（要求 b >= 0，保证初始基可行）
#
# 单纯形表 T 的布局（m 条约束、n 个变量，共 m+1 行、m+n+1 列）：
#   ┌──────────────┬───────────┬─────┐
#   │ A (约束系数) │ I (松弛)  │  b  │   <- 约束行：A x + I s = b，s 是松弛变量
#   ├──────────────┼───────────┼─────┤
#   │ -c(目标系数) │   0       │  z  │   <- 目标行：z - c^T x = 0；RHS 列最终存最优值 z*
#   └──────────────┴───────────┴─────┘
# 列含义：前 n 列 = 原始变量 x；中间 m 列 = 松弛变量 s（初始就构成单位阵 → 初始基可行）；
# 最后一列 = 右端项。目标行的前 n+m 个元素就是【检验数】（reduced cost）。
def simplex_max(c, A, b):
    # 返回 (status, x*, z*)；Bland 规则保证不循环（详见下方步骤注释）
    c = np.asarray(c, float); A = np.asarray(A, float); b = np.asarray(b, float)
    m, n = A.shape
    T = np.zeros((m + 1, m + n + 1))    # 建零表：行 = m 条约束 + 1 条目标；列 = n+m 个变量 + RHS
    T[:m, :n] = A                       # 左上角放约束系数矩阵 A
    T[:m, n:n + m] = np.eye(m)          # 松弛变量列 = 单位阵：Ax + s = b 的 s 天然是初始基
    T[:m, -1] = b                       # 右端项 b
    T[-1, :n] = -np.array(c, dtype=float)  # 目标行系数取负 → 检验数行（<0 表示该变量入基能提质）
    basis = list(range(n, n + m))       # 每行当前的基变量下标：初始就是松弛变量 n..n+m-1
    for _ in range(1000):
        # ---- 步骤 1：最优性检验 ----
        # 检验数（目标行前 n+m 个元素）全部 >= 0 时，任何非基变量入基都不能再提高目标 → 最优
        neg = np.where(T[-1, :-1] < -1e-9)[0]
        if neg.size == 0:
            break
        # ---- 步骤 2：选入基变量 ----
        # 取最靠前的负检验数（Bland 规则取最小下标 → 数学上保证不循环）
        j = int(neg[0])
        col = T[:-1, j]                 # 入基列：该变量在各约束行中的系数
        # ---- 步骤 3：无界判定 ----
        # 入基列整列 <= 0：增大该变量不会减掉任何基变量 → 目标可无限增大（问题无界）
        if np.all(col <= 1e-9):
            return 'unbounded', None, None
        # ---- 步骤 4：比值检验（选出基变量）----
        # 基变量值 = b_i / T[i,j]（入基变量每增加 1，第 i 行基变量减少 T[i,j]）。
        # 选最小正比值的行出基 → 保证换基后所有基变量仍非负（顶点仍可行）。
        pos = np.where(col > 1e-9)[0]
        ratios = T[pos, -1] / col[pos]
        min_r = ratios.min()
        cand = pos[np.abs(ratios - min_r) <= 1e-9]                # 比值相同的候选行（退化情形）
        i = int(cand[np.argmin([basis[rr] for rr in cand])])      # Bland 平局：取基变量下标最小的行
        # ---- 步骤 5：枢轴变换（高斯-约当消元） ----
        T[i] = T[i] / T[i, j]           # 枢轴行同除以枢轴元素 → 枢轴位置变 1
        for r in range(m + 1):
            if r != i and abs(T[r, j]) > 1e-12:
                T[r] = T[r] - T[r, j] * T[i]   # 其余行消去入基列 → 除枢轴外该列全是 0
        basis[i] = j                    # 记录：第 i 行的基变量现在是 x_j（旧的出基）
    # ---- 步骤 6：从最终表格读回最优解 ----
    # 基变量的值 = 所在行的 RHS；非基变量 = 0（线性方程组直接给出的答案）
    x = np.zeros(n)
    for i, bv in enumerate(basis):
        if bv < n:                      # 只取原始变量（跳过松弛变量列）
            x[bv] = T[i, -1]
    return 'optimal', x, T[-1, -1]      # 目标行 RHS = 目标函数最优值 z*

c = [3, 2]; A = [[1, 1], [1, 0], [0, 1]]; b = [4, 2, 3]
st, x_star, z_star = simplex_max(c, A, b)
print('status =', st)
print('x* =', x_star, ' z* =', z_star)
assert st == 'optimal' and np.allclose(x_star, [2, 2]) and abs(z_star - 10) < 1e-6
print('对照通过：手写单纯形 == 图解法最优 (2,2) z=10')"""))

    cells.append(code(r"""# ========== 对照 1：顶点枚举（2 变量小规模穷举，用来验证单纯形） ==========
# 思路：LP 最优解必在顶点；顶点 = 两条约束边界线（包括 x_i=0 的两条坐标轴）的交点。
# 枚举所有两两交点 -> 检查是否可行（满足全部约束）-> 挑目标值最大的顶点。
def vertex_enum(c, A, b, n_vars=2):
    lines = []                          # 收集所有边界线：(系数向量, 常数)
    for i in range(n_vars):
        e = np.zeros(n_vars); e[i] = 1
        lines.append((e, 0.0))          # 坐标轴边界：x_i = 0
    for (a, bi) in zip(A, b):
        lines.append((np.array(a, float), float(bi)))   # 约束边界：a^T x = b_i
    best = None
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):      # 两两求交点
            M = np.vstack([lines[i][0], lines[j][0]])   # 2x2 线性方程组系数矩阵
            rhs = np.array([lines[i][1], lines[j][1]])
            if abs(np.linalg.det(M)) < 1e-12:   # 两条线平行/重合 -> 无唯一交点，跳过
                continue
            p = np.linalg.solve(M, rhs)         # 解线性方程组得到交点坐标
            # 可行性检查：p >= 0 且满足所有 A x <= b
            if p.min() >= -1e-9 and all(aa @ p <= bi + 1e-9 for aa, bi in zip(A, b)):
                if best is None or c @ p > c @ best:   # 保留目标值最大的可行顶点
                    best = p
    return best, c @ best

p_best, z_best = vertex_enum(np.array(c), A, b)
print('顶点枚举: x* =', p_best, ' z =', z_best)
assert np.allclose(p_best, [2, 2]) and abs(z_best - 10) < 1e-9

# ========== 对照 2：对偶问题构造 + 强对偶验证（scipy 当 oracle） ==========
# 对偶构造规则（自动套用，面试要能默写）：
#   原问题  max c^T x,  Ax <= b, x >= 0
#   对偶    min b^T y,  A^T y >= c, y >= 0
# 即：目标系数与右端项互换（c<->b）、约束矩阵转置（A<->A^T）、<= 变 >=、max 变 min。
from scipy.optimize import linprog
# linprog 只接受 <= 约束：把 A^T y >= c 两边取负 -> -A^T y <= -c
res = linprog(np.array(b), A_ub=-np.array(A).T, b_ub=-np.array(c),
              bounds=[(0, None)] * 3)           # y >= 0，对偶变量共 3 个（= 原问题约束数）
print('对偶最优 y* =', res.x, ' 对偶目标 =', res.fun)
assert abs(res.fun - z_star) < 1e-6
print('强对偶验证：原问题 z* = %.4f == 对偶目标 = %.4f ✅' % (z_star, res.fun))

# 互补松弛条件：y_i * (b - A x*)_i = 0（对偶变量 × 原问题松弛量 = 0）
# 含义：约束 i 若没被用满（松弛 b - A x* > 0），则其影子价格 y_i 必须为 0；
#       只有被用满的约束才“值钱”。（SVM 里对应的就是：非支持向量的 α_i = 0）
Ax = np.array(A) @ x_star
comp = res.x * (np.array(b) - Ax)
print('互补松弛 y_i·(b-Ax)_i =', comp)
assert np.allclose(comp, 0, atol=1e-6)
print('互补松弛成立：约束用满(松弛=0)的才有影子价格，未用满的 y_i=0')"""))

    cells.append(md(r"""## §3 对偶理论：一个 LP 的两副面孔

**对偶问题**（自动从原问题构造）：

| 原问题（max） | 对偶问题（min） |
|---|---|
| $\max c^\top x$ | $\min b^\top y$ |
| $Ax \le b$ | $A^\top y \ge c$ |
| $x \ge 0$ | $y \ge 0$ |

**弱对偶**：任意可行 $x, y$ 有 $c^\top x \le b^\top y$（对偶给原问题一个上界）。
**强对偶**：原问题有最优解时两者相等（LP 永远满足）→ 可用对偶来验证、或解更容易的一侧。

**互补松弛**：最优时 $y_i (b - Ax)_i = 0$、$x_j (A^\top y - c)_j = 0$。
含义：约束 $i$ 若未用满（松弛 $>0$），其影子价格 $y_i = 0$；只有用到饱和的约束才有价值。

**影子价格**：$y_i^*$ 是资源 $i$ 增加一单位时最优目标值的边际增量 → 敏感性分析 / 定价。

**与 ML 的联系**：SVM 的拉格朗日对偶（13 篇）是同一套数学；LP 常见于推荐、调度、资源分配、匹配类问题建模。"""))

    cells.append(md(r"""## §4 内点法、复杂度与常见坑

**单纯形 vs 内点法**：单纯形沿**边界**顶点走，最坏指数时间（Klee–Minty 反例），实际几乎线性；
内点法（Karmarkar 1984）从**内部**沿障碍函数的中心路径逼近，多项式时间，大规模 LP 常用。

**常见坑**：
- **退化**：顶点有冗余约束 → 多次枢轴目标不变，可能循环 → 用 Bland 规则 / 字典序
- **无界**：入基列全非正 → 目标可无限增大
- **不可行**：需两阶段法（先解人工变量问题）——手撕题通常给可行起点
- **标准化**：$\ge$ 约束减松弛、自由变量拆成两个非负变量、目标统一

> 面试常问「为什么最优解在顶点」：线性函数在凸多面体上的极值一定在极点达到（画图 + 反证即可）。"""))

    cells.append(md(r"""## §5 面试追问与自测

1. LP 最优解为什么必在顶点？（线性目标、凸可行域）
2. 单纯形每次怎么选入基 / 出基？（检验数 + 比值检验）
3. 对偶问题的构造规则？强对偶何时成立？（LP 恒成立；一般凸问题要 Slater 条件，见 13 篇）
4. 影子价格怎么用？（资源边际价值、敏感性分析）
5. 单纯形 vs 内点法复杂度？（最坏指数 vs 多项式；实际单纯形很快）

**自测清单**
- [ ] 把任意 LP 写成标准形式，能画 2D 图解法
- [ ] 默写单纯形表结构与一次枢轴运算
- [ ] 会构造对偶并验证强对偶（本讲代码）
- [ ] 说出互补松弛与影子价格的含义

> 💡 下节预告：变量要求整数后「取整不可行」——09 整数规划与分支定界。"""))

    return build(cells)


# =====================================================================
# 09-整数规划与分支定界
# =====================================================================
def nb09():
    cells = []

    cells.append(md(r"""# 🔢 09-整数规划与分支定界

> 目标：LP 加上「变量必须为整数」后难度陡增（IP 是 NP-hard），本讲掌握：IP 建模、**分支定界 B&B** 手撕、
> 割平面直觉、求解器实践。面试考点：为什么不能对松弛解取整、上界/下界方向、背包 B&B 手撕。

## §0 整数规划为什么难

**整数规划（IP）**：$\max c^\top x$ s.t. $Ax \le b,\ x \ge 0,\ x \in \mathbb{Z}^n$（部分变量整数 → MILP）。

- **LP 松弛**：丢掉整数约束 → 连续问题，好解，但只是**上界**
- **取整不靠谱**：松弛解四舍五入可能**不可行**（违反约束），即使可行也常常**非最优**
- 整数可行点是**离散点集**，不再是连续多面体 → 顶点法失效 → 需要分支定界 / 割平面

**本节例子**（贯穿全程）：

$$\max\ 8x_1+5x_2 \quad \text{s.t.}\ x_1+x_2\le 6,\ 9x_1+5x_2\le 45,\ x_1,x_2\ge 0,\ \text{整数}$$

LP 松弛最优 $(3.75, 2.25)$ 目标 $41.25$；但取整 $(4,2)$ 违反 $9x_1+5x_2\le 45$！真正的整数最优是 $(5,0)$，目标 $40$。"""))

    cells.append(code(r"""import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 复用 08 篇的单纯形（本 notebook 自包含；逐行解读见 08-线性规划与对偶.ipynb） ----------
def simplex_max(c, A, b):
    # max c^T x  s.t.  A x <= b,  x >= 0；返回 (status, x*, z*)
    # 单纯形表：行 = m 约束 + 1 目标；列 = n 变量 + m 松弛 + RHS；Bland 规则防循环
    c = np.asarray(c, float); A = np.asarray(A, float); b = np.asarray(b, float)
    m, n = A.shape
    T = np.zeros((m + 1, m + n + 1))    # 增广矩阵：约束行 A|I|b，目标行 -c|0|z
    T[:m, :n] = A                       # 约束系数
    T[:m, n:n + m] = np.eye(m)          # 松弛变量列 = 单位阵（初始基）
    T[:m, -1] = b                       # RHS
    T[-1, :n] = -np.array(c, dtype=float)  # 检验数行（负 = 入基可提高目标）
    basis = list(range(n, n + m))       # 每行的基变量（初始为松弛变量）
    for _ in range(1000):
        neg = np.where(T[-1, :-1] < -1e-9)[0]     # 找负检验数
        if neg.size == 0:
            break                                 # 检验数全 >= 0 → 最优
        j = int(neg[0])                           # 入基：最小下标（Bland）
        col = T[:-1, j]
        if np.all(col <= 1e-9):                   # 整列非正 → 无界
            return 'unbounded', None, None
        pos = np.where(col > 1e-9)[0]
        ratios = T[pos, -1] / col[pos]            # 比值检验：最小正比值行出基
        min_r = ratios.min()
        cand = pos[np.abs(ratios - min_r) <= 1e-9]
        i = int(cand[np.argmin([basis[rr] for rr in cand])])   # Bland 平局
        T[i] = T[i] / T[i, j]                     # 枢轴行归一化
        for r in range(m + 1):
            if r != i and abs(T[r, j]) > 1e-12:
                T[r] = T[r] - T[r, j] * T[i]      # 消去入基列（高斯-约当）
        basis[i] = j
    x = np.zeros(n)
    for i, bv in enumerate(basis):
        if bv < n:
            x[bv] = T[i, -1]                      # 基变量 = 该行 RHS
    return 'optimal', x, T[-1, -1]                # 目标行 RHS = 最优值

# ---------- LP 松弛 + 取整反例 ----------
c9 = [8, 5]; A9 = [[1, 1], [9, 5]]; b9 = [6, 45]
st, xr, zr = simplex_max(c9, A9, b9)
print('LP 松弛最优:', np.round(xr, 2), ' z = %.2f' % zr)
# 常见错误做法：把松弛最优 (3.75, 2.25) 四舍五入成 (4, 2)
print('取整候选 (4,2) 是否可行:', 9 * 4 + 5 * 2 <= 45 and 4 + 2 <= 6)   # 9*4+5*2=46 > 45 → 不可行!
print('取整候选 (4,2) 的目标 =', 8 * 4 + 5 * 2)

# ---------- 穷举所有整数点求真解（小规模对照基准） ----------
best_ip = None
for x1 in range(7):                       # 枚举 x1 = 0..6
    for x2 in range(7):                   # 枚举 x2 = 0..6
        if x1 + x2 <= 6 and 9 * x1 + 5 * x2 <= 45:   # 先检查可行性
            if best_ip is None or 8 * x1 + 5 * x2 > 8 * best_ip[0] + 5 * best_ip[1]:
                best_ip = (x1, x2)        # 保留目标值更大的整数点
print('穷举整数最优:', best_ip, ' z =', 8 * best_ip[0] + 5 * best_ip[1])"""))

    cells.append(code(r"""# ================= 手写分支定界（B&B） =================
# LP 松弛用 scipy.linprog 作为「线性规划 oracle」——B&B 的核心逻辑（分支/剪枝）是我们手写的，
# 松弛求解是工业级子程序（面试手撕时同样把 LP 松弛当黑盒）。
#
# 关键概念（max 问题）：
#   * 上界 UB：当前子问题的 LP 松弛最优值（去掉整数约束只会让解更好或不变 → 松弛值 >= 整数最优）
#   * 下界 LB：已找到的整数可行解的目标值（全局最优至少这么好）
#   * 剪枝   ：UB <= LB 时该子问题不可能产出更好的整数解 → 整棵子树直接放弃
from scipy.optimize import linprog

def bb_2d(c, A, b, bds, max_nodes=300):
    # 2 变量整数最大化问题的分支定界。
    # bds: 各变量整数界 [(lo,hi),...]。返回 (最优解, 最优值, 节点记录)。
    c = np.array(c, float); A = np.array(A, float); b = np.array(b, float)
    best_val = -np.inf; best_x = None   # 全局下界（当前最好整数解的值）与对应解
    nodes = []                          # 节点记录：(id, 父id, 松弛解, 松弛值, 状态)
    queue = [(bds, -1)]                 # 待考察子问题：整数界 + 父节点 id（FIFO → BFS）
    nid = 0
    while queue and nid < max_nodes:
        bd, parent = queue.pop(0)
        # 1) 解当前子问题的 LP 松弛（bounds 带上分支得到的整数界）
        res = linprog(-c, A_ub=A, b_ub=b, bounds=bd, method='highs')
        nid += 1
        if not res.success:             # 松弛不可行 → 本分支无可行解，剪枝
            nodes.append((nid, parent, None, None, 'infeasible'))
            continue
        x = res.x; z = -res.fun
        # 2) 判断松弛解是否已经是整数解
        frac = [i for i in range(len(c)) if abs(x[i] - round(x[i])) > 1e-6]
        if not frac:                    # 整数可行：它就是本子问题的可行整数解 → 更新下界
            nodes.append((nid, parent, x, z, 'integer'))
            if z > best_val:
                best_val = z; best_x = x
            continue
        nodes.append((nid, parent, x, z, 'fractional'))
        # 3) 上界剪枝：即使继续分支，最优也不可能超过松弛上界 z <= best_val
        if z <= best_val + 1e-9:
            nodes[-1] = (nid, parent, x, z, 'pruned')
            continue
        # 4) 分支：选一个分数变量 x_i，拆成 x_i <= floor(x_i*) 和 x_i >= ceil(x_i*) 两个子问题
        i = frac[0]
        lo, hi = bd[i]
        child_lo = [bd[k] if k != i else (lo, float(np.floor(x[i]))) for k in range(len(c))]
        child_hi = [bd[k] if k != i else (float(np.ceil(x[i])), hi) for k in range(len(c))]
        queue.append((child_lo, nid))   # 左分支：x_i 取 floor 及以下
        queue.append((child_hi, nid))   # 右分支：x_i 取 ceil 及以上
    return best_x, best_val, nodes

best_x, best_val, nodes = bb_2d(c9, A9, b9, [(0, 6), (0, 6)])
print('分支定界最优:', best_x, ' z =', best_val)
print('B&B 树（id, 父, 松弛解, 松弛值, 状态）：')
for nd in nodes:
    print('  ', nd)
assert best_x[0] == 5 and best_x[1] == 0 and abs(best_val - 40) < 1e-9
print('对照通过：B&B == 穷举整数最优 (5,0) z=40')"""))

    cells.append(code(r"""# ---------- 对照 2：scipy.optimize.milp（工业求解器内核，验证手写 B&B） ----------
from scipy.optimize import milp, LinearConstraint, Bounds
# 目标写最小化：max 8x1+5x2 等价于 min -8x1-5x2 → c = [-8, -5]
# integrality=1 表示对应变量要求整数
res = milp(c=np.array([-8.0, -5.0]),
           constraints=[LinearConstraint(np.array([[1.0, 1.0], [9.0, 5.0]]), -np.inf, [6.0, 45.0])],
           integrality=np.ones(2), bounds=Bounds(0, 6))
print('scipy.milp:', res.x, ' z =', -res.fun)
assert np.allclose(res.x, [5, 0]) and abs(-res.fun - 40) < 1e-9
print('对照通过：手写 B&B == scipy.milp == 穷举，三层验证一致')

# ========== 示意图：松弛解 vs 整数解 ==========
fig, ax = plt.subplots(figsize=(7, 5.4))
xs = np.linspace(-0.2, 7, 300)
ax.plot(xs, 6 - xs, color='#4C72B0', lw=1.3, label='x1+x2=6')
ax.plot(xs, (45 - 9 * xs) / 5, color='#55A868', lw=1.3, label='9x1+5x2=45')
ax.axvline(0, color='k', lw=0.7); ax.axhline(0, color='k', lw=0.7)
# 所有整数可行点（灰色方块）
feas = [(i, j) for i in range(7) for j in range(7) if i + j <= 6 and 9 * i + 5 * j <= 45]
for p in feas:
    ax.plot(*p, 's', ms=4, color='#8E8E93')
# 松弛最优 vs 整数最优：松弛点 (3.75,2.25) 周围根本没有整数可行点，
# 最近的取整候选 (4,2) 还落在约束外（红箭头标注）
ax.plot(3.75, 2.25, 'o', ms=10, color='#C44E52', label='LP 松弛最优 (3.75,2.25)')
ax.plot(5, 0, '*', ms=16, color='#C44E52', label='整数最优 (5,0)')
ax.annotate('取整 (4,2) 不可行!', xy=(4, 2), xytext=(4.4, 2.7),
            arrowprops=dict(arrowstyle='->'), fontsize=9, color='#C44E52')
ax.set_xlim(-0.3, 7); ax.set_ylim(-0.3, 7)
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('LP 松弛 ≠ 整数解：最优整数解不是松弛解的取整', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()"""))

    cells.append(md(r"""## §1 整数规划建模（先会建模，再会求解）

**0-1 背包**：$\max \sum_i v_i x_i$ s.t. $\sum_i w_i x_i \le W,\ x_i \in \{0,1\}$
**指派问题**：$\min \sum_{i,j} c_{ij} x_{ij}$ s.t. 每行/每列恰好一个 1，$x_{ij}\in\{0,1\}$
**TSP 的 MILP**：$x_{ij}\in\{0,1\}$ 表示走边 $(i,j)$ + 出入度约束 + **子圈消除**（MTZ 大 M 约束 $u_i-u_j+n x_{ij}\le n-1$）
**选址/设施**：变量 = 是否建 + 是否服务，逻辑关系用大 M 线性化

**建模通用技巧**：
- 逻辑「若 A 则 B」→ $x_B \ge x_A$（或 $y \le M x$）
- 0-1 乘积 $x_1 x_2$ → 辅助变量 $z$ + 四条线性约束（$z \le x_1,\ z \le x_2,\ z \ge x_1+x_2-1,\ z\ge 0$）
- 对称解重复 → 加序约束 $x_1 \le x_2 \le \cdots$ 打破对称

> **max 问题里，LP 松弛值永远是整数最优的上界**——这是分支定界剪枝的依据。"""))

    cells.append(code(r"""# ================= 0-1 背包：手写分支定界（上界=分数背包贪心） =================
# 模型：max Σ v_i x_i  s.t.  Σ w_i x_i <= W,  x_i ∈ {0,1}
# 上界为什么用【分数背包】：把“装/不装”放松成“可以装一部分”，得最大化问题。
# 可拆的松弛解价值必然 >= 整数背包最优价值 → 天然是上界，且贪心就能算（性价比排序）。
rng = np.random.default_rng(0)
n_items = 12
weights = rng.integers(1, 15, size=n_items)
values = rng.integers(5, 40, size=n_items)
W = 40
print('物品 (w, v):', list(zip(weights, values)))

def knap_ub(pos, cap, cur):
    # 分数背包上界：从第 pos 件起，按“单位重量价值 v/w”从高到低贪心装（允许拆最后一件）
    b = cur
    for k in range(pos, n_items):
        if weights[k] <= cap:           # 整件装得下 → 整件装，扣容量
            b += values[k]; cap -= weights[k]
        else:                           # 装不下 → 按剩余容量比例拆（这就是“可拆”带来的松的上界）
            b += values[k] * cap / weights[k]
            break
    return b

def knap_bb():
    # 按性价比降序排列物品：先分支“单位价值高”的物品，上界收紧更快 → 剪枝更多
    order = sorted(range(n_items), key=lambda i: values[i] / weights[i], reverse=True)
    w = [weights[i] for i in order]
    v = [values[i] for i in order]
    best = 0; best_sel = None; nodes = 0; pruned = 0

    # 深度优先搜索解空间树：每个物品两个分支（装 / 不装）
    def dfs(pos, cap, cur, sel):
        nonlocal best, best_sel, nodes, pruned
        nodes += 1
        if cur > best:                  # 当前装法价值超过已知最优 → 更新下界与最优解
            best = cur; best_sel = sel.copy()
        if pos == n_items:              # 所有物品都决策完 → 回溯
            return
        if knap_ub(pos, cap, cur) <= best:   # 上界剪枝：后续就算全按分数装也超不过 best
            pruned += 1
            return
        if w[pos] <= cap:               # 分支 1：装第 pos 件（容量够才装）
            sel.append(order[pos])
            dfs(pos + 1, cap - w[pos], cur + v[pos], sel)
            sel.pop()
        dfs(pos + 1, cap, cur, sel)     # 分支 2：不装第 pos 件（容量不变）

    dfs(0, W, 0, [])
    return best, sorted(best_sel), nodes, pruned

best_k, sel_k, nk, pk = knap_bb()
print('背包 B&B：最优价值 %d，选中物品 %s，访问节点 %d，剪枝 %d' % (best_k, sel_k, nk, pk))

# ---------- 对照：动态规划（二维 0/1 背包，dp[cap] = 容量 cap 能装的最大价值） ----------
dp = np.zeros(W + 1, int)
for i in range(n_items):
    for cap in range(W, weights[i] - 1, -1):   # 从大到小遍历：保证每件物品只被用一次（01 背包）
        dp[cap] = max(dp[cap], dp[cap - weights[i]] + values[i])
print('DP 最优价值:', dp[W])
assert dp[W] == best_k
print('对照通过：背包 B&B == 背包 DP')"""))

    cells.append(md(r"""## §2 分支定界细节（为什么能剪枝）

**框架**：维护子问题队列（原问题 + 一些整数界）：
1. 解当前子问题的 **LP 松弛**，得到上界 $\bar z$
2. 剪枝三条件：松弛**不可行** / 松弛解已**整数可行**（更新全局下界 $\underline z$）/ $\bar z \le \underline z$
3. 否则选一个**小数变量** $x_i$ 分支成两个子问题：$x_i \le \lfloor x_i^* \rfloor$、$x_i \ge \lceil x_i^* \rceil$

**工程细节**：
- 分支变量选择：最分数化 / 伪代价 / 强分支（节点数可差一个数量级）
- 节点选择：DFS（省内存、快出可行解）vs 最佳上界优先（剪枝更多）
- 初始下界：贪心可行解（背包 = 依次装性价比最高的物品）
- 大规模 → 交给求解器（**分支切割** = B&B + 割平面）

**复杂度**：最坏指数（$2^n$ 量级），实际取决于 LP 界松紧与分支策略；10-12 篇的启发式就是「时间不够时的近似替代」。"""))

    cells.append(md(r"""## §3 割平面（Gomory）与 MILP 求解器

**割平面思路**：解 LP 松弛得小数顶点 → 加一条**切割不等式**，它：
- 切掉当前小数顶点（松弛最优不再可行）
- 不切掉任何整数可行点
重复直到松弛最优变整数。

**Gomory 割**：从最优单纯形表某行推导，对系数取整：$\sum_j \lfloor a_{ij}\rfloor x_j \le \lfloor b_i\rfloor$。
**分支切割（branch-and-cut）**：B&B 每个节点都加割 → 现代 MILP 求解器（Gurobi / CPLEX / SCIP / CBC）标配。

**求解器实践**：建模语言 PuLP / OR-Tools / Pyomo；小规模可直接 `scipy.optimize.milp`（本讲对照用）。

**常见坑**：
- 弱 LP 界（松弛上界太松）→ 分支爆炸
- 大 M 取太大 → 数值病态；取「恰好够大」最好
- 对称解不打破 → 反复搜索等价节点"""))

    cells.append(md(r"""## §4 面试追问与自测

1. 为什么不能对 LP 松弛解取整？（反例：本例 (4,2) 不可行）
2. 分支定界的上界/下界分别怎么来？（max 问题：松弛 = 上界，可行整数 = 下界）
3. 剪枝三条件是什么？背包 B&B 的上界为什么用分数背包贪心（可拆贪心 ≥ 整数最优）？
4. 大 M 法怎么线性化「若 A 则 B」？MTZ 子圈消除的作用？
5. 什么时候手撕 B&B、什么时候上求解器？（考试小规模手撕；工业直接求解器）

**自测清单**
- [ ] 把 0-1 背包 / 指派 / TSP 写成 IP 模型
- [ ] 手撕背包分支定界（上界=分数背包、深度优先、剪枝）
- [ ] 说明 LP 松弛界的方向（max：上界）
- [ ] 一句话讲清 Gomory 割与分支切割
- [ ] 复现本讲 B&B 与 scipy.optimize.milp 结果一致

> 💡 下节预告：IP 精确但指数级；时间不够用启发式——10 遗传算法。"""))

    return build(cells)


# =====================================================================
# 10-遗传算法
# =====================================================================
def nb10():
    cells = []

    cells.append(md(r"""# 🧬 10-遗传算法

> 目标：当问题「没有梯度 / 离散 / 黑箱」时，用**进化式启发搜索**找近似最优。本讲掌握：编码、选择、交叉、变异、
> 精英保留四个算子 + 两个手写实验（多峰函数优化 / TSP）。面试考点：四个算子各干什么、为什么能全局搜索、
> 早熟收敛、二进制编码的 Hamming 悬崖。

## §0 算法骨架与适用场景

**生物隐喻**：种群（候选解集合）→ 适应度（目标值）→ 选择（优胜劣汰）→ 交叉（基因重组）→ 变异（随机扰动）→ 新一代。

**伪代码**

```
初始化种群（随机）
repeat G 代:
    计算每个个体的适应度
    精英保留最优的 elite 个
    按适应度选择父代（锦标赛/轮盘赌）
    交叉生成子代（pc）; 变异（pm）; 子代入新种群
    若早熟（多样性消失）→ 加大变异
```

**与梯度法对比**：

| 维度 | 梯度下降 | 遗传算法 |
|---|---|---|
| 需要的信息 | 梯度（可导） | 仅目标函数值（黑箱） |
| 解空间 | 连续 | 连续/离散均可（靠编码） |
| 搜索方式 | 局部迭代 | 种群并行 + 重组全局搜索 |
| 收敛性质 | 快但可能困在局部最优 | 慢但不易卡死 |
| 典型场景 | 训练神经网络 | TSP / 调度 / 特征选择 / 超参搜索 |"""))

    cells.append(code(r"""import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

rng = np.random.default_rng(2024)

# ---------- 测试函数：f(x)=x·sin(10πx)+1, x∈[0,1]，多峰，全局最大 1.85 (x≈0.85) ----------
def target(x):
    return x * np.sin(10 * np.pi * x) + 1.0

# ================= 遗传算法四算子（手写核心） =================
# 编码：二进制基因（NB=16 位）。x ∈ [0,1] 被量化成 2^16 个档位。
# 为什么用二进制：交叉/变异对位操作自然、搜索粒度可控；代价是 Hamming 悬崖（见 §2）。
NB = 16

def decode(gene):
    # 二进制基因 → [0,1] 的实数：按位累加得整数 val，再除以 2^NB-1 归一化
    # 例：NB=4 时 1010 -> 10/15 ≈ 0.667
    val = 0
    for bit in gene:
        val = (val << 1) | int(bit)     # 左移一位再或上当前位 = 二进制转整数
    return val / (2 ** NB - 1)

def fitness(gene):
    return target(decode(gene))         # 适应度 = 目标函数值（越高越好）

def init_pop(size):
    # 随机种群：每行一个个体，NB 个 0/1 位
    return rng.integers(0, 2, size=(size, NB))

def select_tournament(pop, fits, k=3):
    # 锦标赛选择：随机抽 k 个个体，取适应度最高的那个作为父本
    # 作用：保持选择压力又不至于让最强个体垄断（保留多样性）
    idx = rng.choice(len(pop), k, replace=False)
    return pop[np.argmax(fits[idx])].copy()

def crossover(p1, p2, pc=0.85):
    # 单点交叉：以概率 pc 在随机断点 pt 交换两条基因的尾部，模拟基因重组
    if rng.random() > pc:               # 不交叉时直接复制父本
        return p1.copy(), p2.copy()
    pt = int(rng.integers(1, NB))       # 断点取 1..NB-1（至少交换一位）
    c1 = np.concatenate([p1[:pt], p2[pt:]])
    c2 = np.concatenate([p2[:pt], p1[pt:]])
    return c1, c2

def mutate(gene, pm=0.01):
    # 位翻转变异：每位以概率 pm 取反（0→1 或 1→0）
    # 作用：引入随机扰动，防止种群过早同化（早熟收敛），保住全局搜索能力
    g = gene.copy()
    mask = rng.random(NB) < pm          # 布尔掩码：哪些位要翻转
    g[mask] = 1 - g[mask]
    return g

def ga(pop_size=60, gens=120, pm=0.01, pc=0.85, elite=2, seed=1):
    # 主循环：初始化 → 每代评估/记录/选亲/交叉/变异/精英保留
    global rng
    rng = np.random.default_rng(seed)   # 固定随机种子：实验可复现
    pop = init_pop(pop_size)
    best_hist, mean_hist = [], []       # 记录每代最优与平均适应度（画收敛曲线用）
    for _ in range(gens):
        fits = np.array([fitness(g) for g in pop])
        best_hist.append(fits.max()); mean_hist.append(fits.mean())
        order = np.argsort(fits)[::-1]  # 按适应度从高到低排序（精英在前）
        newpop = [pop[i].copy() for i in order[:elite]]   # 精英保留：最优 elite 个直接进下一代
        while len(newpop) < pop_size:
            p1 = select_tournament(pop, fits)             # 锦标赛选父本1
            p2 = select_tournament(pop, fits)             # 锦标赛选父本2
            c1, c2 = crossover(p1, p2, pc)                # 交叉产生两个子代
            c1 = mutate(c1, pm); c2 = mutate(c2, pm)      # 变异
            newpop.extend([c1, c2])
        pop = np.array(newpop[:pop_size])                 # 补齐到种群规模
    fits = np.array([fitness(g) for g in pop])
    return np.array(best_hist), np.array(mean_hist), pop[np.argmax(fits)]   # 末代最优个体

bh, mh, bg = ga(seed=1)
bx = decode(bg)     # 解码末代最优基因 → 最优解的 x 坐标
print('GA 找到: x = %.4f, f = %.4f（全局最优 ≈ 1.8505 @ x≈0.85）' % (bx, target(bx)))
print('收敛：首代最优 %.3f → 末代最优 %.3f' % (bh[0], bh[-1]))
assert abs(target(bx) - 1.8505) < 1e-3   # 必须逼近全局最优（多峰函数不卡在局部峰）

# ========== 图 1：GA 在多峰函数上找到全局最优 ==========
gx = np.linspace(0, 1, 400)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(gx, target(gx), lw=1.6, color='#4C72B0', label='f(x) = x·sin(10πx)+1')
ax.axvline(bx, color='#C44E52', ls='--', lw=1.2)
ax.plot(bx, target(bx), '*', ms=15, color='#C44E52', label='GA 找到 (x=%.3f, f=%.3f)' % (bx, target(bx)))
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('GA 在多峰函数上找到全局最优（未卡在左侧局部峰）', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()

# ========== 图 2：收敛曲线 ==========
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(bh, lw=1.8, color='#C44E52', label='每代最优')
ax.plot(mh, lw=1.4, color='#4C72B0', alpha=0.8, label='每代平均')
ax.set_xlabel('代数'); ax.set_ylabel('适应度')
ax.set_title('GA 收敛曲线：最优提升 + 均值追赶（种群整体变好）', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()"""))

    cells.append(code(r"""# ================= 图 3：变异率 pm 的影响（早熟 vs 破坏） =================
# pm 太小：种群很快同质化，找不到新区域（早熟收敛）；
# pm 太大：好基因频繁被破坏，收敛慢甚至发散。
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for pm, col in [(0.001, '#8E8E93'), (0.01, '#4C72B0'), (0.15, '#C44E52')]:
    bh2, _, _ = ga(pm=pm, seed=1)
    ax.plot(bh2, lw=1.5, color=col, label='pm = %.3f' % pm)
ax.set_xlabel('代数'); ax.set_ylabel('每代最优适应度')
ax.set_title('变异率对比：pm 太小→早熟，太大→破坏优秀基因', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()"""))

    cells.append(md(r"""## §1 排列编码与 TSP：遗传算法也能跑组合优化

多峰函数用的是**二进制编码**；TSP（旅行商）要用**排列编码**：一条染色体 = 一个城市访问顺序。
此时交叉不能乱切（会得到重复城市），要用**部分匹配交叉（PMX）**或**顺序交叉（OX）**：

**OX 交叉步骤**（见代码）：
1. 从父本 P1 随机选一段 `[a,b]`，原样继承到子代
2. 从父本 P2 的 `b+1` 起循环扫描，按顺序填入子代空缺位（跳过重复城市）
3. 保证每个城市在子代恰好出现一次（仍是合法环）

变异用**反转片段**（2-opt 一步）：随机选一段反转顺序 → 相当于局部扰动。

> 编码决定了交叉/变异的含义——这是遗传算法的设计要点，面试高频。"""))

    cells.append(code(r"""# ================= 排列编码 GA 求解 TSP =================
Nc = 12
rng12 = np.random.default_rng(2)      # 本实例: GA 27.12 < 最近邻 29.85（改善约 9%，见下方对照）
cities = rng12.random((Nc, 2)) * 10
dist = np.sqrt(((cities[:, None, :] - cities[None, :, :]) ** 2).sum(-1))   # 距离矩阵 d[i,j]

def tour_len(order):
    # 环长 = 沿顺序相邻城市距离之和 + 回起点的边（封闭环）
    return sum(dist[order[i], order[(i + 1) % Nc]] for i in range(Nc))

def ox_cross(p1, p2):
    # 顺序交叉 OX：保证子代是合法排列（不重复、不缺城市）
    a, b = sorted(rng12.integers(0, Nc, size=2).tolist())   # 随机选段 [a,b]
    child = [-1] * Nc                                       # 子代初始化成占位符
    child[a:b + 1] = p1[a:b + 1]                            # 步骤1：P1 的 [a,b] 段原样继承
    # 步骤2：从 P2 的 b+1 位置开始循环扫描，把未出现过的城市按序填入子代空缺
    fill = [x for x in list(p2[b + 1:]) + list(p2[:b + 1]) if x not in child]
    k = 0
    for i in range(Nc):
        if child[i] == -1:                                  # 遇到空缺就补一个
            child[i] = fill[k]; k += 1
    return child

def invert_mut(order):
    # 反转变异：随机选一段 [a,b] 反转顺序（对应 TSP 里 2-opt 一步）
    i, j = sorted(rng12.integers(0, Nc, size=2).tolist())
    return order[:i] + order[i:j + 1][::-1] + order[j + 1:]

def ga_tsp(pop_size=120, gens=250, pc=0.9, pm=0.25, elite=3, seed=1):
    # 主循环与函数优化版同构：适应度 = 环长越小越好 → 用 -环长 作为适应度
    r = np.random.default_rng(seed)
    pop = [r.permutation(Nc).tolist() for _ in range(pop_size)]   # 随机排列作为初始种群
    best_hist = []
    best_tour = None; best_len = np.inf
    for _ in range(gens):
        fits = np.array([-tour_len(o) for o in pop])              # 适应度取负环长（越大越好）
        best_hist.append(-fits.max())
        order = np.argsort(fits)[::-1]
        newpop = [pop[i][:] for i in order[:elite]]               # 精英保留
        while len(newpop) < pop_size:
            # 锦标赛选父本：随机抽 3 个个体，取适应度最高者的下标
            i1 = int(np.argmax(fits[r.choice(len(pop), 3, replace=False)]))
            i2 = int(np.argmax(fits[r.choice(len(pop), 3, replace=False)]))
            if r.random() < pc:
                c1 = ox_cross(pop[i1], pop[i2]); c2 = ox_cross(pop[i2], pop[i1])
            else:
                c1 = pop[i1][:]; c2 = pop[i2][:]
            if r.random() < pm:
                c1 = invert_mut(c1); c2 = invert_mut(c2)
            newpop += [c1, c2]
        pop = newpop[:pop_size]
        l = tour_len(pop[order[0]])
        if l < best_len:                                          # 记录全局最优
            best_len = l; best_tour = pop[order[0]][:]
    return np.array(best_hist), best_tour, best_len

bh_t, bt, bl = ga_tsp(seed=1)
print('TSP GA 最优环长: %.2f' % bl)

# ---------- 对照：最近邻启发式（从每座城市出发，每次走最近的未访问城市） ----------
def nearest_neighbor():
    best = np.inf
    for s in range(Nc):                                     # 每个起点都试
        un = set(range(Nc)); un.discard(s)
        cur = [s]
        while un:
            nxt = min(un, key=lambda j: dist[cur[-1], j])   # 挑最近的未访问城市
            cur.append(nxt); un.discard(nxt)
        best = min(best, tour_len(cur))
    return best

nn_len = nearest_neighbor()
print('最近邻启发式: %.2f；GA 相对改善 %.1f%%' % (nn_len, (nn_len - bl) / nn_len * 100))
assert bl < nn_len, 'GA 应优于最近邻基线'

# ========== 图：GA 找到的 TSP 路径 + 收敛曲线 ==========
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax[0].plot(cities[:, 0], cities[:, 1], 'o', color='#4C72B0')
order_l = bt + [bt[0]]                                      # 闭合环
ax[0].plot(cities[order_l, 0], cities[order_l, 1], '-', color='#C44E52', lw=1.5)
ax[0].set_title('GA 找到的 TSP 最优路径（环长 %.2f）' % bl, fontsize=12)
ax[0].grid(alpha=0.3)
ax[1].plot(bh_t, lw=1.6, color='#C44E52')
ax[1].set_title('TSP 遗传算法收敛：环长逐代下降', fontsize=12)
ax[1].set_xlabel('代数'); ax[1].set_ylabel('最优环长')
ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()"""))

    cells.append(md(r"""## §2 早熟收敛与 Hamming 悬崖（两大经典坑）

**早熟收敛（premature convergence）**：某个高适应度个体快速复制扩散，种群失去多样性 → 搜索停滞在局部最优。
缓解：锦标赛/轮盘赌（压力可调）、精英保留（不丢最优）、自适应变异、**岛屿模型/迁徙**、多样化惩罚。

**Hamming 悬崖**：二进制相邻整数可能对应**完全不同的基因**（如 `0111...1` 与 `1000...0`）。
梯度式的交叉变异难以在小范围移动 → 可用 **Gray 码** 消除悬崖。"""))

    cells.append(md(r"""## §3 面试追问与自测

1. 四个算子各干什么、为什么能全局搜索？（选择=局部开发、交叉/变异=全局探索）
2. 早熟收敛是什么、怎么缓解？（多样性管理）
3. 二进制编码的 Hamming 悬崖？Gray 码怎么解决？
4. TSP 用什么编码、交叉为什么不能简单单点？（排列编码 + OX/PMX）
5. GA vs 模拟退火？（种群并行 vs 单点马尔可夫链，见 11 篇）

**自测清单**
- [ ] 手写二进制解码 / 锦标赛选择 / 单点交叉 / 位翻转变异
- [ ] 用 GA 跑通多峰函数并逼近全局最优（本讲断言）
- [ ] 默写 OX 交叉步骤，手撕 TSP GA
- [ ] 讲清早熟与悬崖，并各给一种解法

> 💡 下节预告：单点搜索也能全局收敛——11 模拟退火与禁忌搜索。"""))

    return build(cells)


# =====================================================================
# 11-模拟退火与禁忌搜索
# =====================================================================
def nb11():
    cells = []

    cells.append(md(r"""# 🌡️ 11-模拟退火与禁忌搜索

> 目标：掌握两条「**单点 + 记忆**」的元启发式主线——模拟退火 SA（接受劣解逃出局部最优）与
> 禁忌搜索 TS（记忆最近动作避免回头）。TSP 上手，面试考原理与调参。也顺带对比无记忆的随机重启。

## §0 局部搜索的困境与两条出路

**局部搜索（局部最优陷阱）**：从解出发，只朝「改进方向」移动 → 停在最近的局部最优，与全局最优可能相去甚远。

**出路 1（SA）**：**允许暂时变差**——以概率接受劣解，温度高时概率大（全局探索），温度低时概率小（局部收敛）。
**出路 2（TS）**：**记住最近动作**——最近做过的移动进禁忌表，短期内禁止回头，强制探索新区域。

**邻域**：TSP 用 **2-opt**（反转一段）或 **swap**（交换两城市）。邻域是启发式的设计自由度，见 §2。"""))

    cells.append(code(r"""import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- TSP 实例（N=20 城市，让局部最优陷阱足够明显） ----------
N = 20
rng = np.random.default_rng(5)
cities = rng.random((N, 2)) * 10
dist = np.sqrt(((cities[:, None, :] - cities[None, :, :]) ** 2).sum(-1))   # 距离矩阵 d[i,j]

def tour_len(order):
    # 封闭环长：相邻城市距离之和 + 最后回到起点
    return sum(dist[order[i], order[(i + 1) % N]] for i in range(N))

def random_tour(r):
    return r.permutation(N).tolist()

# ================= 基线：随机重启局部搜索（无记忆） =================
# 思想：反复随机初始化 + 贪心 2-opt 改进，各跑一次，取最好。
# 它没有「记忆」：每次重启都从零开始，容易反复掉进同一个局部最优。
def random_restart(restarts=25, steps=100, seed=0):
    r = np.random.default_rng(seed)
    best = np.inf
    for _ in range(restarts):
        cur = random_tour(r); L = tour_len(cur)
        for _ in range(steps):
            i, j = sorted(r.integers(0, N, size=2).tolist())   # 随机取一段 [i,j]
            nxt = cur[:i] + cur[i:j + 1][::-1] + cur[j + 1:]   # 2-opt：反转该段
            nL = tour_len(nxt)
            if nL < L:                                          # 只接受改进
                cur = nxt; L = nL
        best = min(best, L)
    return best

rr_len = random_restart()
print('随机重启局部搜索最优环长: %.2f' % rr_len)"""))

    cells.append(code(r"""# ================= 手写模拟退火（SA） =================
# 核心：Metropolis 准则——当前解 x，邻居 y，代价差 Δ = f(y)-f(x)：
#   若 Δ < 0（更好）→ 一定接受
#   若 Δ ≥ 0（更差）→ 以概率 exp(-Δ/T) 接受（T 越大越敢接受坏解）
# 温度 T 从 T0 按 T ← T * alpha 几何降温：
#   高温段：几乎什么都接受 → 全局乱逛（逃出局部最优）
#   低温段：只接受改进 → 精细收敛（模拟淬火成晶体）
def sa_tsp(T0=50, alpha=0.998, iters=4000, seed=1):
    r = np.random.default_rng(seed)
    cur = random_tour(r); L = tour_len(cur)
    best = L; best_tour = cur[:]
    T = T0
    hist = []
    for _ in range(iters):
        i, j = sorted(r.integers(0, N, size=2).tolist())
        nxt = cur[:i] + cur[i:j + 1][::-1] + cur[j + 1:]     # 2-opt 邻居
        nL = tour_len(nxt)
        delta = nL - L
        # Metropolis 判定：改进必收；恶化按概率收（exp(-Δ/T) ∈ (0,1)，Δ 越大越难收）
        if delta < 0 or r.random() < np.exp(-delta / T):
            cur = nxt; L = nL
            if L < best:                                      # 全程记录最优（降温后期可能回升）
                best = L; best_tour = cur[:]
        T *= alpha                                            # 几何降温
        hist.append(best)
    return best, best_tour, np.array(hist)

sa_len, sa_tour, sa_hist = sa_tsp()
print('模拟退火最优环长: %.2f' % sa_len)

# ================= 手写禁忌搜索（TS） =================
# 核心三件套：
#   1) 禁忌表：记录最近做过/撤销的动作（swap 城市对），禁止立刻重做 → 防止在局部最优来回震荡
#   2) 藐视准则（aspiration）：若禁忌动作能刷新全局最优，破例允许（好到值得破戒）
#   3) 邻域选择：每步考察整个 swap 邻域，选「非禁忌中最好」的动作执行
def ts_tsp(tabu_len=18, iters=4000, seed=2):
    r = np.random.default_rng(seed)
    cur = random_tour(r); L = tour_len(cur)
    best = L; best_tour = cur[:]
    # 禁忌矩阵：tabu[a,b] = 最早允许再次交换 (a,b) 的代数（0 = 不禁忌）
    tabu = np.zeros((N, N), int)
    hist = []
    for it in range(iters):
        best_move = None; best_nL = np.inf
        for a in range(N):                                    # 枚举 swap 邻域（全部城市对）
            for c in range(a + 1, N):
                if a == c: continue
                nxt = cur[:]; nxt[a], nxt[c] = nxt[c], nxt[a]   # 交换两城市位置
                nL = tour_len(nxt)
                # 禁忌判断：该城市对在禁忌期内，且不能刷新全局最优（藐视准则）→ 跳过
                if tabu[a, c] > it and nL >= best - 1e-9:
                    continue
                if nL < best_nL:                              # 选当前邻域里最好的合法动作
                    best_nL = nL; best_move = (a, c, nxt)
        if best_move is None:                                 # 邻域全被禁忌 → 随机一步（防死锁）
            a, c = sorted(r.integers(0, N, size=2).tolist())
            nxt = cur[:]; nxt[a], nxt[c] = nxt[c], nxt[a]
            cur = nxt; L = tour_len(nxt)
        else:
            a, c, cur = best_move; L = best_nL
            tabu[a, c] = it + tabu_len                        # 动作进禁忌表（到期自动解禁）
            tabu[c, a] = it + tabu_len
        if L < best:                                          # 更新全局最优
            best = L; best_tour = cur[:]
        hist.append(best)
    return best, best_tour, np.array(hist)

ts_len, ts_tour, ts_hist = ts_tsp()
print('禁忌搜索最优环长: %.2f' % ts_len)

print('随机重启 %.2f | SA %.2f | TS %.2f' % (rr_len, sa_len, ts_len))
assert sa_len < rr_len and ts_len < rr_len, 'SA/TS 应明显优于无记忆的随机重启'

# ========== 图：收敛曲线对比 ==========
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.plot(sa_hist, lw=1.6, color='#C44E52', label='SA（Metropolis 接受劣解）')
ax.plot(ts_hist, lw=1.6, color='#4C72B0', label='TS（禁忌表+藐视准则）')
ax.axhline(rr_len, color='#8E8E93', ls='--', lw=1.2, label='随机重启（无记忆）%.2f' % rr_len)
ax.set_xlabel('迭代步'); ax.set_ylabel('当前最优环长')
ax.set_title('TSP：SA / TS 明显优于无记忆的随机重启', fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()"""))

    cells.append(md(r"""## §1 调参地图与原理深挖

**SA 三参数**：初始温度 $T_0$（足够大到能接受大部分劣解）、降温率 $\alpha$（0.9~0.999，越小越快越粗）、
迭代数（与 $T_0,\alpha$ 匹配，总步数 ≈ $\ln(T_f/T_0)/\ln\alpha$）。
**Metropolis 的直觉**：$\exp(-\Delta/T)$——$\Delta$ 大（很坏）几乎不接受，$T$ 大（高温）什么都接受。
数学上 SA 在理想降温计划下以概率 1 收敛到全局最优（Geman & Geman 1984，理论保证但不实用）。

**TS 三件套**：禁忌表长度 $L$（太短防不住回环，太长限制搜索，本讲 18，约 N）、
禁忌对象（TSP 常用**城市对**，也可用完整解/动作）、藐视准则（何时破例）。
**TS 变体**：随机化 TS、频率记忆（长期禁忌低频动作）、路径重连（path relinking）。

**邻域设计是灵魂**：2-opt 一次改 2 条边，swap 一次换 2 个位置——邻域大小、可改进幅度决定收敛速度与质量。"""))

    cells.append(md(r"""## §2 面试追问与自测

1. Metropolis 准则的公式与直觉？为什么高温敢接受坏解？
2. SA 参数怎么调？降温太快/太慢分别有什么后果？
3. 禁忌表记什么？禁忌长度太短/太长？藐视准则何时用？
4. SA vs TS vs 随机重启的本质区别？（概率接受 vs 强制记忆 vs 无记忆）
5. 2-opt 是什么？为什么它很少破坏环结构？

**自测清单**
- [ ] 手写 Metropolis 判定（一行 if 条件）
- [ ] 手写禁忌表更新 + 邻域扫描 + 藐视准则
- [ ] 说出 SA/TS 各自防止「困在局部最优」的机制
- [ ] 复现本讲：SA/TS 均优于随机重启（断言已内置）

> 💡 下节预告：群体智能——12 粒子群与蚁群算法。"""))

    return build(cells)


# =====================================================================
# 12-粒子群与蚁群算法
# =====================================================================
def nb12():
    cells = []

    cells.append(md(r"""# 🐝 12-粒子群与蚁群算法

> 目标：掌握两类**群体智能**算法——粒子群 PSO（连续优化）与蚁群 ACO（组合优化）。
> 面试考点：PSO 三条更新项的含义、惯性权重、ACO 的转移概率与信息素更新、什么时候选哪类启发式。

## §0 群体智能：个体简单，群体涌现

- **PSO（粒子群）**：模拟鸟群觅食——每个粒子记「自己见过的最好位置」和「群体最好位置」，靠这两个拉力移动。
  连续空间、导数信息可完全不用。
- **ACO（蚁群）**：模拟蚂蚁觅食——蚂蚁沿信息素走，走短路的蚂蚁回程留下更多信息素 → **正反馈**强化好路径。
  天然适合图上找路径（TSP / 路由 / 调度）。

| | PSO | ACO |
|---|---|---|
| 解空间 | 连续 | 离散（图/排列） |
| 记忆 | 个人+群体的历史最优 | 集体记录在信息素矩阵 |
| 搜索机制 | 速度加权合成 | 概率转移 + 信息素蒸发/沉积 |
| 适用 | 连续优化 / 超参搜索 | TSP / 网络路由 / 装箱 |"""))

    cells.append(code(r"""import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 测试函数：Rastrigin（2D，全局最优 0 @ (0,0)，四周全是局部谷，SA/GA 都容易卡） ----------
def rastrigin(x):
    # f(x) = A·d + Σ (x_i² - A·cos(2π x_i))，A=10
    # 线性项 x_i² 把全局推向原点；余弦项造出一排排周期性的局部谷 → 多峰陷阱
    A = 10.0
    x = np.asarray(x, float)
    return A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

# ================= 手写粒子群优化（PSO） =================
# 每个粒子 i 维护三样东西：
#   位置 x_i     —— 当前候选解
#   速度 v_i     —— 移动方向和步长
#   个体最优 pbest_i —— 自己历史见过的最好位置（“记忆”）
# 再加上群体共享的 gbest（全局最好位置）。
#
# 速度更新 = 三条力的加权合成：
#   v ← w·v                 (1) 惯性项：保持原速度方向。惯性权重 w 从 0.9 线性降到 0.4
#                                = 前期大范围探索 → 后期局部精细收敛
#      + c1·r1·(pbest - x)  (2) 认知项：拉回自己历史最好位置（自私记忆）
#      + c2·r2·(gbest - x)  (3) 社会项：拉向群体最好位置（信息共享 = 协作牵引）
# 位置更新：x ← x + v（欧拉积分）
def pso(iters=120, n_particles=40, w0=0.9, w1=0.4, c1=1.5, c2=1.5, seed=0):
    r = np.random.default_rng(seed)
    dim = 2
    x = r.uniform(-4.5, 4.5, size=(n_particles, dim))   # 初始位置：搜索范围内均匀随机
    v = r.uniform(-1, 1, size=(n_particles, dim))       # 初始速度：小随机
    pbest = x.copy()                                    # 个体最优初值 = 自己初始位置
    gbest = x[np.argmin([rastrigin(p) for p in x])]     # 群体最优 = 初始里最好的粒子
    hist = []
    for t in range(iters):
        w = w0 + (w1 - w0) * t / iters                  # 惯性线性递减 0.9→0.4（先探索后开发）
        r1 = r.random(size=(n_particles, dim))          # 认知项的随机系数（随机性=多样性）
        r2 = r.random(size=(n_particles, dim))          # 社会项的随机系数
        v = w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)   # 三条力合成新速度
        x = x + v                                       # 按速度移动位置
        for i in range(n_particles):                    # 逐粒子更新个体最优
            if rastrigin(x[i]) < rastrigin(pbest[i]):
                pbest[i] = x[i]
        gi = int(np.argmin([rastrigin(p) for p in pbest]))   # 找 pbest 里最好的那个粒子
        if rastrigin(pbest[gi]) < rastrigin(gbest):   # 若它超过全局最优 → 更新 gbest
            gbest = pbest[gi].copy()
        hist.append(rastrigin(gbest))
    return gbest, np.array(hist)

g_best, pso_hist = pso()
print('PSO 找到: x =', np.round(g_best, 4), ' f = %.4f（全局最优 0 @ (0,0)）' % rastrigin(g_best))
assert rastrigin(g_best) < 1e-6     # 必须收敛到全局最优（Rastrigin 多峰也不怕）

# ========== 图：地形 + 收敛 ==========
gx = np.linspace(-4.5, 4.5, 220); gy = np.linspace(-4.5, 4.5, 220)
Zg = np.array([[rastrigin([a, b]) for a in gx] for b in gy])   # 网格目标值（画等高线用）
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
cf = ax[0].contourf(gx, gy, Zg, levels=18, cmap='viridis_r')
fig.colorbar(cf, ax=ax[0], shrink=0.85)
ax[0].plot(g_best[0], g_best[1], '*', ms=15, color='white', label='gbest (%.3f, %.3f)' % tuple(g_best))
ax[0].set_title('Rastrigin 地形（大量局部谷）', fontsize=12)
ax[0].legend(fontsize=9, loc='upper right')
ax[1].plot(pso_hist, lw=1.8, color='#C44E52')
ax[1].set_yscale('log')             # 目标值指数级下降 → 对数轴更直观
ax[1].set_title('PSO 收敛曲线（Rastrigin 2D 最小化 → 0）', fontsize=12)
ax[1].set_xlabel('迭代'); ax[1].set_ylabel('gbest 目标值 (log)')
plt.tight_layout(); plt.show()"""))

    cells.append(md(r"""## §1 PSO 参数与改进

**标准参数**：$c_1=c_2\approx 1.5\sim2$，$w$：0.9→0.4 线性递减；种群 20~50。
**经典变体**：
- **惯性权重法**（本讲）：$w$ 大→全局探索，$w$ 小→局部开发；递减模拟「先广后精」
- **收缩因子法**（Clerc）：$\chi$ 缩放整条速度，形式 $v\leftarrow\chi(v+c_1 r_1(p-x)+c_2 r_2(g-x))$，理论保证收敛
- **全局版 vs 局部版**：只用邻域内最优（环拓扑）→ 抗早熟
- **约束处理**：罚函数 / 速度+位置边界钳制 / 重新随机

**为什么 PSO 快**：不需要梯度，只需要目标值；粒子间通过 gbest 隐式共享信息（协作），
比独立多次随机重启的「蛮力」效率高得多（本讲收敛到 0 就是证据）。"""))

    cells.append(code(r"""# ================= 手写蚁群算法（ACO）求解 TSP =================
# 两个核心量：
#   信息素 τ(i,j)：边 (i,j) 的“社会记忆”，好边信息素浓
#   启发信息 η(i,j) = 1/d(i,j)：距离的倒数，近的边天然更吸引（先验贪心）
# 蚂蚁 k 在 i 处选下一城 j 的概率（轮盘赌采样）：
#   P(i→j) ∝ τ(i,j)^α · η(i,j)^β
#   α 越大越跟随信息素（正反馈），β 越大越贪心距离（先验）
# 一轮结束后：
#   蒸发：τ ← (1-ρ)τ            —— 防止信息素无限累积，让系统“忘掉”旧信息
#   沉积：τ(i,j) ← τ(i,j) + Q/L —— 走完环长 L 越短的蚂蚁，给走过的边加得越多（正反馈！）
Nc = 14
r4 = np.random.default_rng(4)
cities = r4.random((Nc, 2)) * 10
dist = np.sqrt(((cities[:, None, :] - cities[None, :, :]) ** 2).sum(-1))
np.fill_diagonal(dist, np.inf)      # 自己到自己的距离设无穷：防止蚂蚁原地打转

def tour_len(order):
    return sum(dist[order[i], order[(i + 1) % Nc]] for i in range(Nc))

def aco(ants=50, iters=80, alpha=1.0, beta=5.0, rho=0.5, q=10, seed=0):
    r = np.random.default_rng(seed)
    tau = np.ones((Nc, Nc))         # 信息素初始均匀为 1（无先验偏好）
    np.fill_diagonal(tau, 0)        # 自己到自己不存信息素
    eta = 1.0 / dist                # 启发信息预计算：1/距离（距离为 inf 的位置是 0，蚂蚁不去）
    best_len = np.inf; best_tour = None; hist = []
    for it in range(iters):
        all_tours = []
        for a in range(ants):
            start = int(r.integers(0, Nc))
            un = set(range(Nc)); un.discard(start)
            tour = [start]
            while un:
                i = tour[-1]
                cand = list(un)
                # 转移概率：τ^α · η^β 归一化成概率分布，再按分布轮盘赌抽一城
                p = tau[i, cand] ** alpha * eta[i, cand] ** beta
                p = p / p.sum()
                j = cand[int(r.choice(len(cand), p=p))]
                tour.append(j); un.discard(j)
            all_tours.append((tour, tour_len(tour)))
        tau = tau * (1 - rho)       # ① 蒸发：整张表统一乘 (1-ρ)
        for tour, L in all_tours:   # ② 沉积：每只蚂蚁给自己走的边加 Q/L
            for k in range(Nc):
                i, j = tour[k], tour[(k + 1) % Nc]
                tau[i, j] += q / L  # 环长越短 → 加得越多 → 下一轮更招蚂蚁 → 正反馈
                tau[j, i] += q / L
        bt = min(all_tours, key=lambda t: t[1])     # 记录本次迭代最好蚂蚁
        if bt[1] < best_len:
            best_len = bt[1]; best_tour = bt[0]
        hist.append(best_len)
    return best_tour, best_len, np.array(hist)

aco_tour, aco_len, aco_hist = aco()
print('ACO 最优环长: %.2f' % aco_len)

# ---------- 对照：最近邻贪心基线 ----------
def nearest_neighbor():
    best = np.inf
    for s in range(Nc):                             # 每个城市都当起点试一遍
        un = set(range(Nc)); un.discard(s)
        cur = [s]
        while un:
            nxt = min(un, key=lambda j: dist[cur[-1], j])   # 每次都去最近的未访问城市
            cur.append(nxt); un.discard(nxt)
        best = min(best, tour_len(cur))
    return best

nn_len = nearest_neighbor()
print('最近邻基线: %.2f' % nn_len)
assert aco_len < nn_len          # ACO 必须优于贪心基线（正反馈找到了更短的环）

# ========== 图：路径 + 收敛 ==========
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
ax[0].plot(cities[:, 0], cities[:, 1], 'o', color='#4C72B0')
o = aco_tour + [aco_tour[0]]              # 闭合环（末尾补回起点）
ax[0].plot(cities[o, 0], cities[o, 1], '-', color='#C44E52', lw=1.5)
ax[0].set_title('ACO 找到的 TSP 路径（环长 %.2f）' % aco_len, fontsize=12)
ax[0].grid(alpha=0.3)
ax[1].plot(aco_hist, lw=1.6, color='#C44E52')
ax[1].set_title('ACO 收敛：信息素正反馈逐步强化好边', fontsize=12)
ax[1].set_xlabel('迭代'); ax[1].set_ylabel('最优环长')
ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.show()"""))

    cells.append(md(r"""## §2 ACO 细节与改进

- α、β 的平衡：α 大（跟随群体）→ 收敛快易早熟；β 大（贪心）→ 接近最近邻。
- ρ 蒸发率：ρ 太小信息素残留 → 早熟；ρ 太大忘得快 → 接近随机搜索。常取 0.5 左右。
- **精英蚂蚁 / 排名**：只让全局最优的蚂蚁排外沉积更多 → 收敛加速（ACS：给最优边额外加）。
- **MMAS（最大最小蚁群）**：把 τ 限制在 [τ_min, τ_max] 防早熟 —— 工业常用。
- ACO vs GA 在 TSP 上：ACO 用「路径增量」信息（边），GA 用「排列交叉」；ACO 倾向于更快找到好环（正反馈），GA 全局搜索能力强。

## §3 选型总表：什么时候用哪个启发式

| 场景 | 推荐 | 理由 |
|---|---|---|
| 连续、黑箱、多峰 | PSO / GA | 不需要梯度，多峰不卡 |
| TSP / 路由 / 调度 | ACO / GA / SA+2-opt | 问题天然是图/排列 |
| 有成熟贪心构造法可用 | ACO / 模拟退火＋局部搜索 | 好初始解加速收敛 |
| 时间极紧、要稳定 | 模拟退火 / 禁忌搜索 | 单点内存小、步数可控 |
| 要最优解保证 | 分支定界 / MILP 求解器（08/09 篇） | 启发式只保证“好”，不保证“最优” |

> 面试套路：先问问题性质（连续/离散？有没有梯度？规模？要最优还是近似？）→ 再锁定算法家族。"""))

    cells.append(md(r"""## §4 面试追问与自测

1. PSO 三条更新项各代表什么？惯性权重递减为什么好？
2. ACO 转移概率为什么用 τ^α·η^β？蒸发率 ρ 太大小分别会怎样？
3. PSO vs ACO 适用场景差异？（连续 vs 离散）
4. 群体智能为什么“协作”比独立多开枪更高效？（正反馈/信息共享）
5. 什么时候不能用启发式？（要求最优保证 → 精确方法）

**自测清单**
- [ ] 手写 PSO 速度/位置更新（含 w 递减 + pbest/gbest 更新）
- [ ] 手写 ACO 转移概率 + 蒸发/沉积，跑通 TSP
- [ ] 说出信息素正反馈的一轮完整回路（走路→评价→沉积→吸引）
- [ ] 默写选型总表

> 💡 下节预告：约束优化的理论基石——13 拉格朗日对偶与 KKT。"""))

    return build(cells)


# =====================================================================
# 13-拉格朗日对偶与KKT
# =====================================================================
def nb13():
    cells = []

    cells.append(md(r"""# ⚖️ 13-拉格朗日对偶与KKT

> 目标：把约束优化从「会解例子」提升到「懂原理」：拉格朗日乘子、**KKT 四条件**、对偶构造与互补松弛——
> 这是 SVM、Lasso、正则化、强化学习（约束策略）背后共同的数学。面试考点：KKT 四条件默写、互补松弛的含义、slack。

## §0 等式约束：拉格朗日乘子法

问题：$\min f(x)\ \text{s.t.}\ h(x)=0$。
引入乘子 $\lambda$：$L(x,\lambda) = f(x) + \lambda h(x)$。

**核心几何**：最优解处，目标等高线与约束曲线**相切** → 两者法向平行 → $\nabla f(x^*) + \lambda \nabla h(x^*) = 0$。
（若不平行，沿约束移动一点还能继续降低 $f$。）

**为什么是“乘子”**：$\nabla f$ 与 $\nabla h$ 平行意味着存在 $\lambda$ 使二者线性相关——$\lambda$ 度量 f 对约束松紧的敏感度（= 影子价格，见 08 篇）。"""))

    cells.append(code(r"""import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from scipy.optimize import minimize

# ================= 拉格朗日乘子法：min f(x) s.t. h(x)=0 =================
# 例子：min f = x1² + x2²   s.t.  x1 + x2 = 1
#   几何：目标等高线是圆（越靠近原点越小），约束是直线 x1+x2=1
#   最优 = 圆与直线相切处 → 切点 (0.5, 0.5)，f* = 0.5（投影最短距离问题）
f = lambda x: x[0] ** 2 + x[1] ** 2
h = lambda x: x[0] + x[1] - 1

# ---- 在最优解 (0.5, 0.5) 处验证相切条件 ∇f + λ∇h = 0 ----
x_star = np.array([0.5, 0.5])       # 圆与直线的切点（解析解）
gx = 2 * x_star                     # ∇f = 2x → (1, 1)
gh = np.array([1.0, 1.0])           # ∇h = (1,1)（约束系数）
lam_star = -gx[0] / gh[0]           # 由 2x1 + λ·1 = 0 解出 λ = -2x1 = -1
print('∇f + λ∇h =', gx + lam_star * gh, '（应为 0：梯度平行）')
assert np.allclose(gx + lam_star * gh, 0)

# ---- 对照：scipy SLSQP（工业级约束优化求解器） ----
res = minimize(f, [0.2, 0.5], constraints={'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1}, method='SLSQP')
print('解析最优: [0.5 0.5]  f = 0.5')
print('SLSQP 数值解:', res.x, ' f = %.4f' % res.fun)
assert np.allclose(res.x, [0.5, 0.5], atol=1e-3)
print('对照通过：拉格朗日解析解 == SLSQP 数值解')

# ========== 图：等高线相切约束线 ==========
xs = np.linspace(-0.6, 1.4, 300)
Xs, Ys = np.meshgrid(xs, xs)
Fv = Xs ** 2 + Ys ** 2
fig, ax = plt.subplots(figsize=(6.8, 6.0))
cf = ax.contourf(Xs, Ys, Fv, levels=16, cmap='viridis_r')
fig.colorbar(cf, ax=ax, shrink=0.85)
ax.plot(xs, 1 - xs, color='white', lw=2, label='约束 x1+x2=1')
ax.plot(0.5, 0.5, '*', ms=17, color='#C44E52')
ax.annotate('最优 (0.5,0.5)，等高线相切', xy=(0.5, 0.5), xytext=(0.72, 0.62),
            arrowprops=dict(arrowstyle='->', color='#C44E52'), fontsize=10, color='#C44E52')
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('等式约束：最优处梯度与约束法向量平行', fontsize=12)
ax.legend(fontsize=9); ax.set_xlim(-0.6, 1.4); ax.set_ylim(-0.6, 1.4)
plt.tight_layout(); plt.show()"""))

    cells.append(md(r"""## §1 不等式约束与 KKT 四条件

问题：$\min f(x)\ \text{s.t.}\ g_i(x)\le 0,\ i=1..m$（等式约束可并入）。

**KKT 条件**（约束规范成立时，$\bar x$ 最优 ⟺ 存在乘子 $\mu_i$ 满足）：

| # | 条件 | 含义 |
|---|---|---|
| 1 | $\nabla f + \sum_i \mu_i \nabla g_i = 0$ | **平稳性**：梯度被约束梯度线性表出 |
| 2 | $g_i(x) \le 0$ | **原始可行**：解本身在可行域内 |
| 3 | $\mu_i \ge 0$ | **对偶可行**：乘子不取负 |
| 4 | $\mu_i g_i(x) = 0$ | **互补松弛**：约束拉紧($g_i=0$)才可能有 $\mu_i>0$；没拉紧 → $\mu_i=0$ |

**互补松弛的直觉**：只有「起作用」（active，$g_i=0$）的约束才产生“拉力”，不起作用的约束乘子必须为 0。
（对应到 SVM：只有支持向量（$g=0$）的 $\alpha_i>0$，其余样本 $\alpha_i=0$。）

**Slater 条件**：凸问题里存在严格可行点（$g_i(x)<0$）→ 强对偶成立 → KKT 是充要条件。非凸问题 KKT 只是必要条件。"""))

    cells.append(code(r"""# ================= KKT 四条件：两个对照例子 =================
# 例1：min f = (x1-1)² + (x2-1)²   s.t.  x1 + x2 >= 4（写成 g = 4-x1-x2 <= 0）
#   无约束最优 (1,1) 处 x1+x2=2 < 4 → 不满足约束 → 最优解被顶到边界 x1+x2=4 上
#   ⇒ 约束拉紧（g1=0），乘子 μ>0
f1 = lambda x: (x[0] - 1) ** 2 + (x[1] - 1) ** 2
g1 = lambda x: 4 - x[0] - x[1]      # 约束写成 g <= 0 的标准形式
# scipy 的 'ineq' 表示 fun(x) >= 0，所以传 -g1 = x1+x2-4 >= 0
res1 = minimize(f1, [0.0, 0.0], constraints={'type': 'ineq', 'fun': lambda x: -g1(x)}, method='SLSQP')
x1 = res1.x
# KKT 平稳性：∇f + μ∇g = 0 ⇒ 2(x1-1) + μ(-1) = 0 ⇒ μ = 2(x1-1)（x1=2 时 μ=2）
mu1 = 2 * (x1[0] - 1)
print('例1 解: [%.3f %.3f]  f = %.3f  g = %.2e' % (x1[0], x1[1], f1(x1), g1(x1)))
print('例1 KKT: ∇f = [%.2f %.2f] ，∇f + %.2f·∇g =' % (2*(x1[0]-1), 2*(x1[1]-1), mu1),
      2 * (x1 - 1) + mu1 * np.array([-1.0, -1.0]), '（μ=2 使之为 0）')
assert np.allclose(x1, [2, 2], atol=1e-3) and abs(mu1 - 2) < 1e-2   # 解在边界上、乘子>0

# 例2：min f = (x1-0.5)² + x2²   s.t.  x1 >= 0（写成 g = -x1 <= 0）
#   无约束最优 (0.5, 0) 已经有 x1=0.5 >= 0 → 约束根本没起作用（inactive）
#   ⇒ 互补松弛：g2 < 0 时 μ2 必须 = 0
f2 = lambda x: (x[0] - 0.5) ** 2 + x[1] ** 2
g2 = lambda x: -x[0]                # -x1 <= 0 ⟺ x1 >= 0
res2 = minimize(f2, [0.0, 0.0], constraints={'type': 'ineq', 'fun': lambda x: -g2(x)}, method='SLSQP')
x2 = res2.x
mu2 = 2 * (x2[0] - 0.5)             # 平稳性：2(x1-0.5) + μ(-1) = 0 ⇒ μ = 2(x1-0.5)
print('例2 解: [%.3f %.3f]  f = %.2e （无约束解 (0.5,0) 已满足约束 → μ=0）' % (x2[0], x2[1], f2(x2)))
assert np.allclose(x2, [0.5, 0], atol=1e-3) and abs(mu2) < 1e-2    # 约束不拉紧、乘子=0
print('对照通过：两个 KKT 例子的数值解与解析一致')"""))

    cells.append(md(r"""## §2 对偶问题的更一般构造（回到 08 篇）

对原问题（拉格朗日函数对 x 取 inf）：$g(\mu) = \inf_x L(x,\mu)$ → **对偶问题** $\max_{\mu\ge 0} g(\mu)$。

**弱对偶永远成立**（对偶值 ≤ 原值）；**强对偶**（相等）在凸 + Slater 下成立。
KKT 的 1+3 正是“$x^*$ 与 $\mu^*$ 互为对偶最优”的显式写法；条件 4（互补松弛）让两面完全锁死。

**在 ML 里**：
- **SVM**：KKT → 只有支持向量的 α>0；对偶把约束写进目标，引出核技巧
- **Lasso**：软阈值解 = 一维子问题的 KKT（见下节）
- **RL 约束策略优化（CPO）**：用拉格朗日对偶把「期望回报约束」写进目标函数，与 SVM 的处理同构"""))

    cells.append(md(r"""## §3 坐标下降与 Lasso（一维闭式解 = 软阈值）

Lasso：$\min_w \tfrac{1}{2}\|y - Xw\|^2 + \lambda \|w\|_1$

**坐标下降**：每次固定其他坐标只优化一个 $w_j$ —— 一维子问题有**闭式解**：

一维问题 $\min_w \tfrac{1}{2}(z - w)^2 + \lambda |w|$ 的最优解是**软阈值算子**：
$$w^* = \operatorname{soft}(z, \lambda) = \operatorname{sign}(z)\cdot\max(|z|-\lambda, 0)$$

含义：把 $z$ 向 0 收缩 $\lambda$；缩不动了就置 0 —— **这就是 L1 产生稀疏解的机制**（L2 里没有“置零”）。

**为什么划算**：每轮每个坐标一个 $O(m)$ 投影（可增量维护残差变 $O(1)$ 级联），几百轮就收敛 —— Lasso 大数据的标配。"""))

    cells.append(code(r"""# ================= 坐标下降求解 Lasso（手写 + sklearn 对照） =================
# 问题：min ½||y - Xw||² + λ||w||₁
# 坐标下降循环：固定其它坐标只优化 w_j。推导一句话：
#   记 r' = y - Σ_{k≠j} X_k w_k（残差暂时不含第 j 列），则一维子问题为
#   min_a ½||X_j||²·(a - z)² + λ|a|，其中 z = X_jᵀr'/||X_j||²
#   → 闭式解 a* = soft(z, λ/||X_j||²)（软阈值：把 z 向 0 收缩 λ/||X_j||²）
import numpy as np
from sklearn.linear_model import Lasso

def soft(z, lam):
    # 软阈值算子：min ½(z-w)² + λ|w| 的闭式解 w* = sign(z)·max(|z|-λ, 0)
    # 几何：把 z 向 0 压 λ；|z| ≤ λ 时直接置 0（左上图的“平底”= L1 稀疏性的来源）
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0.0)

def coord_lasso(X, y, lam, iters=2000, tol=1e-9):
    m, p = X.shape
    w = np.zeros(p)
    r = y.copy()                            # 残差向量 r = y - X w，全程增量维护（O(m) 而非重算）
    for _ in range(iters):
        w_old = w.copy()
        for j in range(p):
            r = r + X[:, j] * w[j]          # ① 把 w_j 的贡献拿回 → r 暂时不含第 j 列（r'）
            mj = X[:, j] @ X[:, j]          # ② 列平方范数 ||X_j||²
            z = (X[:, j] @ r) / mj          # ③ 一维最小二乘解（投影系数）
            w[j] = soft(z, lam / mj)        # ④ 软阈值闭式解（收缩量 = λ/||X_j||²，可能置零）
            r = r - X[:, j] * w[j]          # ⑤ 用新 w_j 更新残差，恢复不变量 r = y - X w
        if np.max(np.abs(w - w_old)) < tol: # 所有权重几乎不再变 → 已收敛，提前退出
            break
    return w

# ---- 数据：8 特征但只有 3 个真正有用 → 检验 Lasso 的稀疏性 ----
rng = np.random.default_rng(1)
m, p = 150, 8
X = rng.standard_normal((m, p))
w_true = np.array([1.5, 0, 0, -2.0, 0, 0.8, 0, 0])     # 支持集 {0,3,5}
y = X @ w_true + 0.1 * rng.standard_normal(m)
lam = 0.1

w_cd = coord_lasso(X, y, lam)
print('坐标下降 w:', np.round(w_cd, 4))
assert abs(w_cd[0] - 1.5) < 0.05 and abs(w_cd[3] + 2.0) < 0.05   # 非零坐标逼近真值

# ---- 对照：sklearn（同一问题的工业实现） ----
# sklearn 优化目标为 1/(2m)·||y-Xw||² + alpha·||w||₁，与我们写的 ½||·||² + λ||w||₁
# 差一个 1/m 尺度因子 → 令 alpha = λ / m 即可对齐（两边同乘 m）
# fit_intercept=False：sklearn 默认会拟合截距（内部把 X/y 中心化），那是另一个问题；
# 我们手写版不设截距，所以关掉它才能让两边目标完全一致地对比。
# tol=1e-8：收紧 sklearn 默认 1e-4 的收敛容差，保证两边都算到足够精确再比。
sk = Lasso(alpha=lam / m, max_iter=10000, tol=1e-8, fit_intercept=False).fit(X, y)
print('sklearn   w:', np.round(sk.coef_, 4))
dev = np.max(np.abs(w_cd - sk.coef_))
print('最大偏差: %.6f' % dev)
assert dev < 1e-3
print('对照通过：手写坐标下降 Lasso == sklearn Lasso（稀疏解保留 w0/w3/w5）')"""))

    cells.append(md(r"""## §4 面试追问与自测

1. 默写 KKT 四条件；互补松弛怎么直观理解？（只有拉紧的约束才有乘子）
2. 等式约束 ∇f∥∇h 与 KKT 平稳性的关系？（不等式是「线性表出」，等式是特例）
3. 强对偶什么时候成立？（凸 + Slater）非凸为什么 KKT 只是必要条件？
4. 软阈值算子从哪来？为什么 L1 会置零、L2 不会？（|w| 在 0 处不可导 → 角点解）
5. Lasso 坐标下降每一轮为什么要拿回再投影？（增量残差）

**自测清单**
- [ ] 默写并解释 KKT 四条件（一字不差）
- [ ] 手写软阈值算子并讲出几何
- [ ] 手写坐标下降 Lasso 并与 sklearn 对齐（本讲断言 dev<1e-3）
- [ ] 说出 KKT 在 SVM / Lasso / 强化学习约束优化中的落点

> 至此经典优化算法线（08-13）完结：精确方法（LP/IP）→ 启发式（GA/SA/TS/PSO/ACO）→ 理论工具（KKT/对偶）。"""))

    return build(cells)


def main():
    save(nb08(), '08-线性规划与对偶.ipynb')
    save(nb09(), '09-整数规划与分支定界.ipynb')
    save(nb10(), '10-遗传算法.ipynb')
    save(nb11(), '11-模拟退火与禁忌搜索.ipynb')
    save(nb12(), '12-粒子群与蚁群算法.ipynb')
    save(nb13(), '13-拉格朗日对偶与KKT.ipynb')
    print('ALL DONE')


if __name__ == '__main__':
    main()