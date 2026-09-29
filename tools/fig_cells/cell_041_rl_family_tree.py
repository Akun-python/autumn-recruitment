# source: 07-强化学习面试八股与高频题.ipynb cell 41
# fig_family_tree.py —— 过程图: RL 算法家族谱系树(文本树状图)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=150)
ax.axis('off')
nodes = {
    'root': (0.5, 0.94, 'MDP <S,A,P,R,γ>'),
    'dp': (0.14, 0.74, 'DP(有模型)\n策略/值迭代'),
    'mc': (0.40, 0.74, 'MC(无模型)\n首访/每访'),
    'td': (0.66, 0.74, 'TD(自举)\nTD(0)/n步/TD(λ)'),
    'pf': (0.92, 0.74, '策略梯度\nREINFORCE'),
    'ql': (0.14, 0.50, 'Q-family\nQ/SARSA/期望/Double'),
    'dqn': (0.40, 0.50, 'DQN 家族\nDQN/DDQN/Dueling/PER'),
    'ac': (0.66, 0.50, 'AC 家族\nAC/A2C/GAE'),
    'ppo': (0.92, 0.50, 'PPO/GRPO\n+RLHF/DPO'),
    'top': (0.45, 0.26, '面试考点: 五条对比线 + 手推模板'),
}
for (x, y, t) in nodes.values():
    ax.add_patch(plt.Rectangle((x-0.11, y-0.05), 0.22, 0.13, fc='#eef3fb',
                               ec='steelblue', lw=1.1, zorder=2,
                               transform=ax.transData))
    ax.text(x, y, t, ha='center', va='center', fontsize=8.6, zorder=3)
for child in ['dp', 'mc', 'td', 'pf']:
    ax.annotate('', xy=(nodes[child][0], nodes[child][1] + 0.07),
                xytext=(nodes['root'][0], nodes['root'][1] - 0.07),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
for parent, child in [('dp', 'ql'), ('mc', 'ql'), ('td', 'dqn'), ('td', 'ac'),
                      ('pf', 'ac'), ('ql', 'dqn'), ('ac', 'ppo'), ('pf', 'ppo')]:
    ax.annotate('', xy=(nodes[child][0], nodes[child][1] + 0.07),
                xytext=(nodes[parent][0], nodes[parent][1] - 0.07),
                arrowprops=dict(arrowstyle='->', color='lightgray', lw=1.0))
ax.set_title('强化学习算法家族谱系（07 篇总览）', fontsize=13)
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'rl_family_tree.png'), dpi=150)
print('已保存 images/rl_family_tree.png')
plt.close(fig)
plt.show()
