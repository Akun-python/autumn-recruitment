# source: 02-蒙特卡洛与时序差分.ipynb cell 42
# fig_mctd_curves.py —— 过程图①: 随机游走 TD vs MC 学习曲线(RMS)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

N_STATES = 19
TRUE_V = np.array([i / 20 for i in range(1, N_STATES + 1)])
GAMMA = 1.0

def run_walk(start=9, max_len=200):
    s, traj = start, []
    for _ in range(max_len):
        ns = s + (1 if np.random.random() < 0.5 else -1)
        if ns < 0:
            traj.append((s, 0.0)); break
        if ns > N_STATES - 1:
            traj.append((s, 1.0)); break
        traj.append((s, 0.0)); s = ns
    return traj

def curves(method, episodes=100, alpha=0.1, seed=0):
    np.random.seed(seed)
    V = np.full(N_STATES, 0.5)
    out = []
    for _ in range(episodes):
        if method == 'mc':
            G, seen = 0.0, set()
            for s, r in reversed(run_walk()):
                G = r + GAMMA * G
                if s not in seen:
                    seen.add(s)
                    V[s] += alpha * (G - V[s])
        else:
            for s, r in run_walk():
                V[s] += alpha * (r - V[s])
        out.append(np.sqrt(np.mean((V - TRUE_V) ** 2)))
    return np.array(out)

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
for m, c, lab in [('td', 'green', 'TD(0)'), ('mc', 'blue', 'MC(首访)')]:
    ax.plot(curves(m, seed=1), lw=2, color=c, label=lab)
ax.set_xlabel('回合数'); ax.set_ylabel('RMS 误差')
ax.set_title('随机游走: TD vs MC 的学习曲线（TD 更快触底=样本效率高）')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'mc_td_curves.png'), dpi=150)
print('已保存 images/mc_td_curves.png')
plt.close(fig)
plt.show()
