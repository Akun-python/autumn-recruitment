# source: 01-动态规划与表格方法.ipynb cell 43
# fig_dp_compare.py —— 过程图②: 策略迭代 vs 值迭代"评估工作量"对比
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 教学示意数据: 每轮"底层扫描量"(评估扫描 × 动作) —— 曲线下面积近似总成本
pi_rounds = np.array([500, 2, 500, 2, 500, 2, 2])     # 策略迭代: 评估很长, 改进很短, 少轮
vi_rounds = np.full(24, 30.0)                          # 值迭代: 每轮都轻, 但轮多

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=150)
axes[0].bar(range(len(pi_rounds)), pi_rounds, color='steelblue', width=0.6)
axes[0].set_title('策略迭代（少数几轮，每轮评估扫满）', fontsize=10)
axes[0].set_xlabel('轮次'); axes[0].set_ylabel('单轮扫描量')
axes[1].bar(range(len(vi_rounds)), vi_rounds, color='seagreen', width=0.6)
axes[1].set_title('值迭代（很多轮，每轮只扫一遍）', fontsize=10)
axes[1].set_xlabel('轮次'); axes[1].set_ylabel('单轮扫描量')
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'dp_iters_compare.png'), dpi=150)
print('已保存 images/dp_iters_compare.png')
plt.close(fig)
plt.show()
