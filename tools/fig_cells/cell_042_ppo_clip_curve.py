# source: 06-PPO与GRPO.ipynb cell 42
# fig_ppo_clip.py —— 过程图①: PPO clip 目标曲面(ratio vs 优势方向)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

r = np.linspace(0.0, 2.0, 200)
eps = 0.2

def l_clip(r, A, eps=0.2):
    return np.minimum(r * A, np.clip(r, 1 - eps, 1 + eps) * A)

fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=150)
for ax, A in zip(axes, [1.0, -1.0]):
    ax.plot(r, r * A, ls='--', color='gray', label='无 clip（线性）')
    ax.plot(r, l_clip(r, A, eps), lw=2.2, color='crimson', label=f'clip(1±{eps})')
    ax.axvspan(1 - eps, 1 + eps, color='gold', alpha=0.25, label='信任区域')
    ax.axvline(1.0, color='black', lw=1, ls=':')
    ax.set_title(f'优势 A={A:+.0f}', fontsize=11)
    ax.set_xlabel('概率比 r_t(θ)'); ax.set_ylabel('代理目标')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle('PPO clip: 信任区域外目标被压平 → 梯度受限', y=1.02)
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'ppo_clip_curve.png'), dpi=150)
print('已保存 images/ppo_clip_curve.png')
plt.close(fig)
plt.show()
