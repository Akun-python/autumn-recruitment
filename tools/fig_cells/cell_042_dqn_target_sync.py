# source: 04-深度Q网络DQN.ipynb cell 42
# fig_target_sync.py —— 过程图①: 目标网络"冻结-同步"机制示意
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

steps = np.arange(0, 100)
online = 2.0 + 1.5 * np.sin(steps / 6) + 0.2 * steps / 100     # 在线网络缓慢爬升
target = np.array([online[min(st // 25 * 25, 99)] for st in steps])  # 每25步硬拷贝

fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
ax.plot(steps, online, lw=2, color='crimson', label='online 网络(每步更新)')
ax.plot(steps, target, lw=2, ls='--', color='royalblue', label='target 网络(冻结→同步)')
ax.annotate('硬拷贝 θ⁻ ← θ', xy=(25, target[25]), xytext=(30, 2.6),
            arrowprops=dict(arrowstyle='->'), fontsize=9)
ax.set_xlabel('训练步数'); ax.set_ylabel('Q 估计')
ax.set_title('目标网络机制: 中间冻结住监督信号, 每 C 步同步一次')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'dqn_target_sync.png'), dpi=150)
print('已保存 images/dqn_target_sync.png')
plt.close(fig)
plt.show()
