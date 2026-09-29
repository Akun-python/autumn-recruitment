# source: 04-深度Q网络DQN.ipynb cell 43
# fig_ddqn_bias.py —— 过程图②: DDQN vs DQN 起点 Q 估值对比(悬崖, 噪声奖励)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROWS, COLS = 4, 11
START, GOAL = (3, 0), (3, 10)
CLIFF = [(3, c) for c in range(1, 10)]
NS, NA = ROWS * COLS, 4
MOVE = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def step_idx(s, a, noise=0.5):
    r, c = divmod(s, COLS)
    dr, dc = MOVE[a]
    nr, nc = r + dr, c + dc
    nr, nc = max(0, min(ROWS - 1, nr)), max(0, min(COLS - 1, nc))
    s2 = nr * COLS + nc
    extra = noise * np.random.randn()
    if (nr, nc) in CLIFF:
        return START[0] * COLS + START[1], -100.0 + extra, False
    if (nr, nc) == GOAL:
        return s2, 0.0, True
    return s2, -1.0 + extra, False

def onehot(s):
    x = np.zeros(NS); x[s] = 1.0
    return x

class LinQ:
    def __init__(self, seed=0):
        self.W = np.random.RandomState(seed).randn(NS, NA) * 0.05
    def q(self, s):
        return onehot(s) @ self.W
    def copy_from(self, other):
        self.W = other.W.copy()

def run(double, seed=0, episodes=300):
    rng = np.random.RandomState(seed)
    online, target = LinQ(seed), LinQ(seed + 1)
    target.copy_from(online)
    D, CAP, B, alpha, gamma = [], 2000, 32, 0.02, 0.9
    step_count = 0
    for ep in range(episodes):
        s = START[0] * COLS + START[1]
        eps = max(0.05, 0.3 * (1 - ep / episodes))
        steps = 0
        while steps < 200:
            a = rng.randint(NA) if rng.random() < eps else int(np.argmax(online.q(s)))
            s2, r, done = step_idx(s, a)
            D.append((s, a, r, s2, done))
            if len(D) > CAP:
                D.pop(0)
            if len(D) >= B:
                idx = rng.randint(0, len(D), B)
                for i in idx:
                    s_i, a_i, r_i, s2_i, done_i = D[i]
                    if done_i:
                        y = r_i
                    elif double:
                        a_star = int(np.argmax(online.q(s2_i)))
                        y = r_i + gamma * target.q(s2_i)[a_star]
                    else:
                        y = r_i + gamma * np.max(target.q(s2_i))
                    pred = online.q(s_i)[a_i]
                    online.W[:, a_i] += alpha * (y - pred) * onehot(s_i)
            s = s2
            steps += 1
            step_count += 1
            if step_count % 20 == 0:
                target.copy_from(online)
            if done:
                break
    return online.q(START[0] * COLS + START[1]).max()

vals = {'DQN': [run(False, seed=s) for s in range(4)],
        'DDQN': [run(True, seed=s) for s in range(4)]}
means = {k: np.mean(v) for k, v in vals.items()}
stds = {k: np.std(v) for k, v in vals.items()}

fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=150)
b = ax.bar(list(means.keys()), list(means.values()), yerr=list(stds.values()),
           color=['crimson', 'royalblue'], capsize=6, alpha=0.85)
ax.set_ylabel('起点 Q 估计(越贴近真实越低)')
ax.set_title('DDQN 抑制 Q 值虚高(噪声奖励悬崖环境)')
for rect, v in zip(b, means.values()):
    ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
            f'{v:.2f}', ha='center', va='bottom')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'ddqn_bias.png'), dpi=150)
print('已保存 images/ddqn_bias.png')
plt.close(fig)
plt.show()
