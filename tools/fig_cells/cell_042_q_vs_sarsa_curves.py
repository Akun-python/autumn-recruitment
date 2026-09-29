# source: 03-Q学习与SARSA.ipynb cell 42
# fig_qvssarsa.py —— 过程图①: 悬崖漫步 Q-learning vs SARSA 学习曲线
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
N_ACTIONS = 4
MOVE = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def step(state, a):
    r, c = state
    dr, dc = MOVE[a]
    nr, nc = r + dr, c + dc
    nr, nc = max(0, min(ROWS - 1, nr)), max(0, min(COLS - 1, nc))
    if (nr, nc) in CLIFF:
        return START, -100.0
    if (nr, nc) == GOAL:
        return GOAL, 0.0
    return (nr, nc), -1.0

def learn(method, episodes=200, alpha=0.5, gamma=0.9, eps=0.1, seed=0):
    np.random.seed(seed)
    Q = np.zeros((ROWS, COLS, N_ACTIONS))
    ret = []
    for _ in range(episodes):
        s = START
        a = np.random.randint(4) if np.random.random() < eps else int(np.argmax(Q[s[0], s[1]]))
        total, steps = 0.0, 0
        while s != GOAL and steps < 300:
            s2, r = step(s, a)
            if method == 'q':
                target = r + gamma * np.max(Q[s2[0], s2[1]])
            else:
                a2 = np.random.randint(4) if np.random.random() < eps else int(np.argmax(Q[s2[0], s2[1]]))
                target = r + gamma * Q[s2[0], s2[1], a2]
            Q[s[0], s[1], a] += alpha * (target - Q[s[0], s[1], a])
            s, a = s2, (a2 if method == 'sarsa' else (np.random.randint(4) if np.random.random() < eps
                                                     else int(np.argmax(Q[s2[0], s2[1]]))))
            total += r
            steps += 1
        ret.append(total)
    return np.array(ret)

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
window = np.ones(20) / 20
q = np.convolve(learn('q', seed=1), window, mode='valid')
s = np.convolve(learn('sarsa', seed=1), window, mode='valid')
ax.plot(q, lw=2, color='crimson', label='Q-learning（激进，早期掉崖）')
ax.plot(s, lw=2, color='royalblue', label='SARSA（保守，绕路安全）')
ax.set_xlabel('回合数（20 步滑动平均）'); ax.set_ylabel('累计奖励')
ax.set_title('悬崖漫步: on/off-policy 行为差异（Sutton 例 6.6 复现）')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'q_vs_sarsa_curves.png'), dpi=150)
print('已保存 images/q_vs_sarsa_curves.png')
plt.close(fig)
plt.show()
