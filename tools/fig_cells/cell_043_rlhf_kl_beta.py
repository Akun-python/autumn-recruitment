# source: 06-PPO与GRPO.ipynb cell 43
# fig_kl_beta.py —— 过程图②: KL 系数 β 的权衡(奖励 vs 偏移)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 模拟: β 越大 → KL 偏移越小, 但奖励增益也小
betas = np.array([0.0, 0.01, 0.05, 0.1, 0.2, 0.5])
kl = np.array([0.42, 0.33, 0.22, 0.15, 0.09, 0.04])        # KL(θ‖ref) 示意
reward_gain = np.array([6.0, 5.2, 4.1, 3.0, 1.8, 0.6])     # RM 得分增量示意

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
ax2 = ax.twinx()
ax.plot(betas, kl, 'o-', color='royalblue', label='KL(θ‖ref)（左轴）')
ax2.plot(betas, reward_gain, 's-', color='darkorange', label='奖励增益（右轴）')
ax.set_xlabel('KL 系数 β'); ax.set_ylabel('KL 偏移', color='royalblue')
ax2.set_ylabel('奖励模型得分增量', color='darkorange')
ax.set_title('β 权衡: 惩罚太松→漂移大(RM被hack); 太紧→不进步')
ax.legend(loc='upper right'); ax2.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'rlhf_kl_beta.png'), dpi=150)
print('已保存 images/rlhf_kl_beta.png')
plt.close(fig)
plt.show()
