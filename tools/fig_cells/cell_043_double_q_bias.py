# source: 03-Q学习与SARSA.ipynb cell 43
# fig_doubleq_bias.py —— 过程图②: 最大化偏差数值演示(E[max]>max E)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

rng = np.random.RandomState(0)
true_q = np.array([0.0, 0.0])            # 两个动作真值都是 0
K = 2000
est_max = np.array([max(true_q + rng.randn(2) * 1.0) for _ in range(K)])
est_double = np.array([0.5 * (max(true_q + rng.randn(2) * 1.0)
                              + max(true_q + rng.randn(2) * 1.0)) for _ in range(K)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=150)
for ax, data, color, ttl in [
    (ax1, est_max, 'crimson', '单表 max（正偏）'),
    (ax2, est_double, 'royalblue', '双表交替（居中）')]:
    ax.hist(data, bins=40, color=color, alpha=0.7)
    ax.axvline(0, color='black', ls='--', lw=1, label='真值 0')
    ax.set_xlabel('Q 估计'); ax.set_ylabel('频数')
    ax.set_title(ttl, fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle('最大化偏差: max 会把噪声取偏(红); 双表解耦(蓝)居中', fontsize=11)
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'double_q_bias.png'), dpi=150)
print('已保存 images/double_q_bias.png')
plt.close(fig)
plt.show()
