# source: 02-蒙特卡洛与时序差分.ipynb cell 43
# fig_lambda_scan.py —— 过程图②: TD(λ) 资格迹 λ 扫描
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

def td_lambda(lam, alpha=0.1, episodes=100, seed=0):
    np.random.seed(seed)
    V = np.full(N_STATES, 0.5)
    for _ in range(episodes):
        E = np.zeros(N_STATES)
        for s, r in run_walk():
            E *= GAMMA * lam
            E[s] += 1.0
            delta = r - V[s]
            V += alpha * delta * E
    return np.sqrt(np.mean((V - TRUE_V) ** 2))

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
lams = np.linspace(0, 1, 21)
errs = [td_lambda(lam, seed=2) for lam in lams]
ax.plot(lams, errs, 'o-', ms=4, color='darkorange')
ax.axvline(0.6, color='gray', ls='--', lw=1)
ax.text(0.62, max(errs), 'λ≈0.6 最优', color='gray')
ax.set_xlabel('λ（资格迹强度）'); ax.set_ylabel('最终 RMS 误差')
ax.set_title('TD(λ) λ 扫描: 偏差(λ小)-方差(λ大)折中的鞍底')
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'td_lambda_scan.png'), dpi=150)
print('已保存 images/td_lambda_scan.png')
plt.close(fig)
plt.show()
