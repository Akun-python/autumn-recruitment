# source: 00-MDP与贝尔曼方程.ipynb cell 44
# fig_contraction.py —— 过程图②: 贝尔曼算子收缩(不同初值收敛到同一 V*)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

SIZE, GAMMA = 4, 0.9
N = SIZE * SIZE
MOVE = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def in_b(r, c):
    return 0 <= r < SIZE and 0 <= c < SIZE

def trans(s, a):
    if s == N - 1:
        return [(1.0, s, 0.0)]
    r, c = divmod(s, SIZE)
    dr, dc = MOVE[a]
    nr, nc = r + dr, c + dc
    if not in_b(nr, nc):
        return [(1.0, s, -1.0)]
    s2 = nr * SIZE + nc
    return [(1.0, N - 1, 10.0)] if s2 == N - 1 else [(1.0, s2, -1.0)]

def bellman_op(V):
    Vn = np.zeros(N)
    for s in range(N):
        Vn[s] = max(sum(p * (r + GAMMA * V[s2]) for p, s2, r in trans(s, a))
                    for a in range(4))
    return Vn

rng = np.random.RandomState(0)
V_a, V_b = rng.rand(N) * 10, np.zeros(N)
trace_a, trace_b = [], []
for _ in range(40):
    trace_a.append(V_a[0]); trace_b.append(V_b[0])
    V_a, V_b = bellman_op(V_a), bellman_op(V_b)

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
ax.plot(trace_a, 'o-', ms=3, label='初值A(随机)', color='purple')
ax.plot(trace_b, 's-', ms=3, label='初值B(全0)', color='teal')
ax.set_xlabel('迭代轮数 k'); ax.set_ylabel('V_k(状态0)')
ax.set_title('贝尔曼最优算子是 γ-收缩映射：不同初值收敛到同一 V*')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'bellman_contraction.png'), dpi=150)
print('已保存 images/bellman_contraction.png')
plt.close(fig)
plt.show()
