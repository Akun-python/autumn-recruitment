# source: 01-动态规划与表格方法.ipynb cell 42
# fig_dp_converge.py —— 过程图①: 值迭代误差收敛(不同 γ, log 尺度)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def vi_errors(gamma, size=4, iters=60):
    N = size * size
    MOVE = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    def trans(s, a):
        if s == N - 1:
            return [(1.0, s, 0.0)]
        r, c = divmod(s, size)
        dr, dc = MOVE[a]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < size and 0 <= nc < size):
            return [(1.0, s, -1.0)]
        s2 = nr * size + nc
        return [(1.0, N - 1, 10.0)] if s2 == N - 1 else [(1.0, s2, -1.0)]
    V = np.zeros(N)
    errs = []
    for _ in range(iters):
        Vn = np.zeros(N)
        for s in range(N):
            Vn[s] = max(sum(p * (r + gamma * V[s2]) for p, s2, r in trans(s, a)) for a in range(4))
        err = np.max(np.abs(Vn - V))
        errs.append(err)
        V = Vn
    return np.array(errs)

fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
for gamma in [0.5, 0.8, 0.9, 0.95]:
    e = vi_errors(gamma) + 1e-12
    ax.semilogy(e, 'o-', ms=3, label=f'γ={gamma}')
ax.axhline(1e-3, color='red', ls='--', lw=1)
ax.text(1, 1e-3, 'tol=1e-3', color='red', va='bottom')
ax.set_xlabel('迭代轮数 k'); ax.set_ylabel('max|V_{k+1}-V_k| (log)')
ax.set_title('值迭代收敛: 误差按 γ^k 收缩(斜率=log γ)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
out = os.path.join('autumn-recruit-algo', '07-强化学习', '教学', 'images')
if os.path.basename(os.getcwd()) == '教学':
    out = os.path.join(os.getcwd(), 'images')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'dp_convergence.png'), dpi=150)
print('已保存 images/dp_convergence.png')
plt.close(fig)
plt.show()
