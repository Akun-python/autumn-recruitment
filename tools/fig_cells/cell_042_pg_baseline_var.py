# source: 05-策略梯度与Actor-Critic.ipynb cell 42
# fig_pg_baseline.py —— 过程图②: 三种基线方差对比(网格, 短训练)
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

def step(s, a):
    if s == N - 1:
        return N - 1, 0.0
    r, c = divmod(s, SIZE)
    dr, dc = MOVE[a]
    nr, nc = r + dr, c + dc
    if not in_b(nr, nc):
        return s, -1.0
    s2 = nr * SIZE + nc
    return (N - 1, 10.0) if s2 == N - 1 else (s2, -1.0)

def run(mode, episodes=100, alpha=0.05, seed=0):
    np.random.seed(seed)
    W = np.random.RandomState(seed).randn(4, N) * 0.1
    w_v = np.zeros(N)
    returns = []
    for _ in range(episodes):
        traj = []
        s = 0
        while s != N - 1 and len(traj) < 300:
            x = np.zeros(N); x[s] = 1.0
            logits = W @ x
            logits -= logits.max()
            p = np.exp(logits); p = p / p.sum()
            a = int(np.random.choice(4, p=p))
            s2, r = step(s, a)
            traj.append((x, a, r))
            s = s2
        G = 0.0
        for x, a, r in reversed(traj):
            G = r + GAMMA * G
            p = np.exp(W @ x)
            p = p / p.sum()
            if mode == 'value':
                b = w_v @ x
                w_v += 0.05 * (G - b) * x
            elif mode == 'const':
                b = np.mean([t[2] for t in traj])
            else:
                b = 0.0
            W += alpha * (G - b) * np.outer(np.eye(4)[a] - p, x)
        returns.append(sum(t[2] for t in traj))
    return np.array(returns)

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
names = ['朴素(无基线)', '常数基线', '价值基线']
colors = ['gray', 'orange', 'green']
for name, mode, c in zip(names, ['none', 'const', 'value'], colors):
    ret = run(mode, seed=3)
    # 汇报: 后50回合均值 + 方差
    m, v = ret[-50:].mean(), ret[-50:].var()
    ax.bar(name, v, color=c, alpha=0.8, width=0.5)
    ax.text(name, v, f'均值 {m:.1f}\n方差 {v:.1f}', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('后 50 回合奖励方差(越小越稳)')
ax.set_title('策略梯度基线消融: 价值基线方差最低(AC 动机)')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'pg_baseline_var.png'), dpi=150)
print('已保存 images/pg_baseline_var.png')
plt.close(fig)
plt.show()
