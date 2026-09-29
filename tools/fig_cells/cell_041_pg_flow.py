# source: 05-策略梯度与Actor-Critic.ipynb cell 41
# fig_pg_flow.py —— 过程图①: 策略梯度定理推导流程(文本图)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=150)
ax.axis('off')
boxes = [
    (0.5, 0.90, "① 目标: J = E[G] = Σ_s d^π(s) Σ_a π(a|s) Q^π(s,a)"),
    (0.5, 0.70, "② 对 θ 求导, 展开递归: ∇Q = γΣ P(s'|s,a) ∇V(s')"),
    (0.5, 0.50, "③ 聚合访问频率: ∇J = Σ_s d^π(s) Σ_a ∇π(a|s) Q^π(s,a)"),
    (0.5, 0.30, "④ log 技巧 + 采样: ∇J ≈ (1/N)Σ_t ∇log π(a_t|s_t)·G_t"),
    (0.5, 0.10, "结论: 环境动力学 P 在 ② 中被 d^π 吸收, 梯度里不含模型!"),
]
for x, y, txt in boxes:
    ax.add_patch(plt.Rectangle((x-0.48, y-0.07), 0.96, 0.14, fc='#eef3fb',
                               ec='steelblue', lw=1.2, zorder=2))
    ax.text(x, y, txt, ha='center', va='center', fontsize=10.5, zorder=3)
for i in range(len(boxes)-1):
    ax.annotate('', xy=(0.5, boxes[i+1][1]+0.075), xytext=(0.5, boxes[i][1]-0.075),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.53, (boxes[i][1]+boxes[i+1][1])/2, '下推', fontsize=8, color='gray')
ax.set_title('策略梯度定理推导流程（面试 4 步板书画）', fontsize=12)
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'pg_flow.png'), dpi=150)
print('已保存 images/pg_flow.png')
plt.close(fig)
plt.show()
