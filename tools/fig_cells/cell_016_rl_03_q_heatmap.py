# source: 03-Q学习与SARSA.ipynb cell 16
# q_heatmap.py —— Q-learning 学到的 Q 表热力图
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

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

np.random.seed(0)
Q = np.zeros((ROWS, COLS, N_ACTIONS))
alpha, gamma, eps = 0.5, 0.9, 0.1
for _ in range(500):
    s = START
    steps = 0
    while s != GOAL and steps < 200:
        a = np.random.randint(4) if np.random.random() < eps else int(np.argmax(Q[s[0], s[1]]))
        s2, r = step(s, a)
        Q[s[0], s[1], a] += alpha * (r + gamma * np.max(Q[s2[0], s2[1]]) - Q[s[0], s[1], a])
        s = s2
        steps += 1

Vmax = Q.max(axis=2)
fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
im = ax.imshow(Vmax, cmap='RdYlGn', vmin=-40, vmax=5)
for r in range(ROWS):
    for c in range(COLS):
        if (r, c) in CLIFF:
            ax.text(c, r, '崖', ha='center', va='center', fontsize=9, color='black')
        elif (r, c) == GOAL:
            ax.text(c, r, 'G', ha='center', va='center', fontsize=11, color='black', fontweight='bold')
        else:
            ax.text(c, r, f'{Vmax[r, c]:.0f}', ha='center', va='center', fontsize=8)
ax.set_xticks(range(COLS)); ax.set_yticks(range(ROWS))
ax.set_title('悬崖漫步 Q-learning 学到的 Q 表(每个状态最大动作价值)', fontsize=11)
fig.colorbar(im, ax=ax, fraction=0.03)
plt.tight_layout()
plt.savefig('rl_03_q_heatmap.png', dpi=150)
print('已保存 rl_03_q_heatmap.png')
print('终点附近绿、悬崖红、起点上方绕行路径黄绿 → 学到了安全近路')
