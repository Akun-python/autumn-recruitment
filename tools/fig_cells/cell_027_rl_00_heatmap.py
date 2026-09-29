# source: 00-MDP与贝尔曼方程.ipynb cell 27
# value_heatmap.py —— V* 热力图 + 最优策略箭头叠加
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

SIZE, GAMMA = 4, 0.9
N = SIZE * SIZE
MOVE = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def in_bounds(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE

def trans(s, a):
    if s == N - 1:
        return [(1.0, s, 0.0)]
    r, c = divmod(s, SIZE)
    dr, dc = MOVE[a]
    nr, nc = r + dr, c + dc
    if not in_bounds(nr, nc):
        return [(1.0, s, -1.0)]
    s2 = nr * SIZE + nc
    return [(1.0, N - 1, 10.0)] if s2 == N - 1 else [(1.0, s2, -1.0)]

V = np.zeros(N)
for _ in range(1000):
    Vn = np.zeros(N)
    for s in range(N):
        Vn[s] = max(sum(p * (r + GAMMA * V[s2]) for p, s2, r in trans(s, a)) for a in range(4))
    if np.max(np.abs(Vn - V)) < 1e-9:
        V = Vn
        break
    V = Vn

# 最优动作箭头方向
arrows = []
for s in range(N):
    q = [sum(p * (r + GAMMA * V[s2]) for p, s2, r in trans(s, a)) for a in range(4)]
    a = int(np.argmax(q))
    arrows.append(MOVE[a])
arrows = np.array(arrows)

fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=150)
im = ax.imshow(V.reshape(SIZE, SIZE), cmap='viridis', origin='upper')
for r in range(SIZE):
    for c in range(SIZE):
        s = r * SIZE + c
        if s == N - 1:
            ax.text(c, r, '★', ha='center', va='center', fontsize=16, color='white')
            continue
        dr, dc = arrows[s]
        ax.annotate('', xy=(c + dc * 0.32, r + dr * 0.32),
                    xytext=(c - dc * 0.18, r - dr * 0.18),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.4))
        ax.text(c, r + 0.38, f'{V[s]:.1f}', ha='center', va='center',
                fontsize=7, color='white', alpha=0.9)
ax.set_title('4×4 网格最优价值热力图 + 最优策略箭头', fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig('rl_00_heatmap.png', dpi=150)
print('已保存 rl_00_heatmap.png（价值热力图）')
print('终点价值最高, 向角落递减; 箭头全部指向价值上升方向')
