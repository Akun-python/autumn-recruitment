# source: 00-MDP与贝尔曼方程.ipynb cell 43
# fig_discount.py —— 过程图①: 折扣回报的权重衰减(不同 γ)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
import os

steps = np.arange(0, 20)
fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
for gamma, c in [(0.0, 'gray'), (0.5, 'orange'), (0.9, 'green'), (0.99, 'blue')]:
    ax.plot(steps, gamma ** steps, 'o-', ms=4, label=f'γ={gamma}', color=c)
ax.set_xlabel('未来时刻 t'); ax.set_ylabel('γ^t（奖励权重）')
ax.set_title('折扣因子 γ 对远期奖励权重的衰减（推导 G_t 的几何权重）')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
import os as _os
out = _os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if _os.path.basename(_os.getcwd()) == '教学':
    out = _os.path.join(_os.getcwd(), 'images')
_os.makedirs(out, exist_ok=True)
fig.savefig(_os.path.join(out, 'bellman_discount.png'), dpi=150)
print('已保存 images/bellman_discount.png')
plt.close(fig)
plt.show()
