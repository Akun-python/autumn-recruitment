# -*- coding: utf-8 -*-
"""生成 10-自然语言处理/教学/04-Attention与Transformer.ipynb（nbformat 4）"""
import os

import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell, new_notebook

OUT = '10-自然语言处理/教学'
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, 'images'), exist_ok=True)

META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
}

cells = []


def _lines(src):
    return [l + "\n" for l in src.rstrip().split("\n")]


def md(src):
    cells.append(new_markdown_cell(_lines(src)))


def code(src):
    cells.append(new_code_cell(_lines(src)))


# =====================================================================
# 10-04 Attention 与 Transformer
# =====================================================================
md(r"""# 🎯 10-04 · Attention 与 Transformer

> 目标：理解注意力机制与 Transformer 结构。手写 **QKV 自注意力 + 缩放点积** 并对照 torch，
> 手撕 **Softmax 稳定版**、**多头拼接**，最后用 mini-Transformer 完成一个「倒序」小任务。

> 🧩 **生活化类比**：注意力 = 阅读时划重点——每个词都回头问别的词「你跟我有什么关系?」，
> 关系近的（Q·K 大）重点吸收。Transformer = 一堆这样的"重点划书人"（多头）同时工作，再配一个前馈网络做深度加工。""")

md(r"""## 1. 从 Attention 讲起

**注意力机制**（Attention Is All You Need, 2017）：

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^\\top}{\\sqrt{d_k}}\\right) V$$

- **Q（查询）**：我要找什么；**K（键）**：我是什么；**V（值）**：我给什么
- $QK^\\top$：所有查询与键的相似度打分
- **除以 $\\sqrt{d_k}$**：防止点积过大把 softmax 推入饱和区（梯度消失）
- softmax 归一化成权重 → 加权求和 V

**为什么是缩放点积而不是直接点积**：$Q,K$ 若方差为 1，点积方差为 $d_k$；除以 $\\sqrt{d_k}$ 让方差回到 1。

**批量实现**（B 批 × H 头可以合并）：`(B, T, D) @ (B, D, T) -> (B, T, T)`，再 `@ V`。""")

code(r"""# ---------- 实验 1：手写缩放点积注意力 vs torch.nn.functional.scaled_dot_product_attention ----------
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
rng = np.random.default_rng(0)

T, D = 6, 8
Q = rng.normal(size=(1, T, D))   # (B,T,D)
K = rng.normal(size=(1, T, D))
V = rng.normal(size=(1, T, D))

def scaled_dot_attention(Q, K, V):
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(K.shape[-1])     # (B,T,T)
    p = np.exp(scores - scores.max(axis=-1, keepdims=True))      # 稳定 softmax
    p = p / p.sum(axis=-1, keepdims=True)
    return p @ V, p

out_my, attn = scaled_dot_attention(Q, K, V)
import torch
with torch.no_grad():
    out_t = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(Q), torch.from_numpy(K), torch.from_numpy(V)).numpy()
print('手写 vs torch 最大误差: %.2e' % np.abs(out_my - out_t).max())
assert np.allclose(out_my, out_t, atol=1e-5)
print('注意力权重（第 0 个 token 对其它 token）:', np.round(attn[0, 0], 3))
print('要点：注意力权重和为 1，模型自动学会「关注谁」；QKV 都来自输入时叫自注意力')""")

md(r"""## 2. Self-Attention 与多头

- **自注意力**：Q = K = V = 输入的线性投影 → 每个词参考整句（含自己）
- **多头**：并行 h 组 QKV 投影（每组 $d_k = D/h$），结果拼接再过线性层
  - 每头可关注不同关系（语法、指代、词义）
  - 多头 = 特征子空间的专家分工

**复杂度**：自注意力 $O(T^2 d)$——T 是序列长度，长序列是瓶颈（这也是 flash-attention / 线性注意力的动机）""")

code(r"""# ---------- 实验 2：手写多头注意力 vs torch.nn.MultiheadAttention 对照 ----------
import torch
import torch.nn as nn

torch.manual_seed(0)
B, T_, D_, H_ = 2, 5, 16, 4      # 4 头，每头 d_k = 4
mha = nn.MultiheadAttention(D_, H_, batch_first=True, bias=False)

# 把 torch 参数整理成手写可用的矩阵（4D x D 每头一份）
wq = mha.in_proj_weight[:D_].detach().numpy().reshape(H_, D_ // H_, D_)
wk = mha.in_proj_weight[D_:2 * D_].detach().numpy().reshape(H_, D_ // H_, D_)
wv = mha.in_proj_weight[2 * D_:].detach().numpy().reshape(H_, D_ // H_, D_)
wo = mha.out_proj.weight.detach().numpy()          # (D, D)

def mha_forward(x, wq, wk, wv, wo, H):
    B, T, D = x.shape
    dk = D // H
    heads = []
    for h in range(H):
        Qh = x @ wq[h].T                            # (B,T,dk)
        Kh = x @ wk[h].T
        Vh = x @ wv[h].T
        sc = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(dk)
        p = np.exp(sc - sc.max(-1, keepdims=True))
        p = p / p.sum(-1, keepdims=True)
        heads.append(p @ Vh)
    cat = np.concatenate(heads, axis=-1)            # (B,T,D)
    return cat @ wo.T

x4 = torch.randn(B, T_, D_)
with torch.no_grad():
    out_t4, _ = mha(x4, x4, x4)
out_my4 = mha_forward(x4.numpy(), wq, wk, wv, wo, H_)
err = np.abs(out_my4 - out_t4.detach().numpy()).max()
print('手写多头 vs torch.MultiheadAttention 最大误差: %.2e' % err)
assert err < 1e-4
print('一致 ✓ | h 个头 -> 拼接 -> 线性投影 out_proj，与论文公式完全一致')""")

md(r"""## 3. Transformer 总装：Encoder / Decoder

**Encoder 层**：多头自注意力 → 残差 + LayerNorm → 前馈 FFN（两层 MLP + ReLU）→ 残差 + LayerNorm
**Decoder 层**：掩码自注意力（保证只能看左侧）→ 交叉注意力（K/V 来自 Encoder）→ FFN
**加性细节**
- **位置编码**（正弦/可学习）：注意力本身无序，必须注入位置信息
- **LayerNorm**：对每个 token 的特征做归一化（不受序列长度影响）
- **残差连接**：稳定深层训练

**Token 化**：词 → 字（BPE 子词）→ embedding；输出端投影到词表 + softmax。""")

code(r"""# ---------- 实验 3：mini-Transformer 学「倒序」（教学版双向注意力 + FFN） ----------
torch.manual_seed(0)
import torch.nn as nn

# 任务：输入 0~6 的序列，输出它的倒序（如 [1,3,2] -> [2,3,1]）。
# 用「双向自注意力」（Encoder 风格，类似 BERT）：模型靠位置编码学会"位置重排"；
# 若用 Decoder 的因果掩码（只看左侧），倒序任务在结构上不可学——体会"结构即能力"。
T_in, V_in = 5, 8                       # 长度 5，词表含 0..6 和填充符 7
rng = np.random.default_rng(0)
class MiniTF(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V_in, 32)
        self.pos = nn.Parameter(torch.randn(1, T_in, 32) * 0.05)
        self.Wq = nn.Linear(32, 32); self.Wk = nn.Linear(32, 32); self.Wv = nn.Linear(32, 32)
        self.ffn = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
        self.ln1 = nn.LayerNorm(32); self.ln2 = nn.LayerNorm(32)
        self.head = nn.Linear(32, V_in)
    def forward(self, x):
        e = self.emb(x) + self.pos
        Q = self.Wq(e); K = self.Wk(e); V = self.Wv(e)
        sc = Q @ K.transpose(-1, -2) / np.sqrt(32)
        p = torch.softmax(sc, dim=-1)               # 双向：Encoder 风格
        a = self.ln1(e + p @ V)
        a = self.ln2(a + self.ffn(a))
        return self.head(a)

def gen_batch(n=96):
    xs, ys = [], []
    for _ in range(n):
        L = int(rng.integers(2, 6))                       # 随机长度 2..5
        seq = rng.integers(0, 7, size=L).tolist()
        xs.append([7] * (T_in - L) + seq)               # 前补填充符 7
        ys.append(list(reversed(seq)) + [7] * (T_in - L))
    return torch.tensor(xs), torch.tensor(ys)

model = MiniTF()
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
for ep in range(300):
    xb, yb = gen_batch()
    logits = model(xb)
    loss = nn.functional.cross_entropy(logits.permute(0, 2, 1), yb)
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 100 == 0:
        print('ep %3d loss %.3f' % (ep, loss.item()))

with torch.no_grad():
    xb, yb = gen_batch(300)
    pred = model(xb).argmax(-1)
    # 只看非填充位置
    ok = ((pred == yb) | (yb == 7)).float().mean().item()
print('倒序任务 token 级准确率（含填充位置）: %.3f' % ok)
assert ok > 0.95
# 演示一例
x0, y0 = gen_batch(1)
with torch.no_grad():
    p0 = model(x0)[0].argmax(-1).tolist()
L0 = (x0[0] != 7).sum().item()
print('示例  输入:', x0[0][-L0:].tolist(), ' 期望:', y0[0][-L0:].tolist(), ' 预测:', p0[-L0:])""")

md(r"""## 4. 数字敏感度与易错点（背诵）

**数字**
- 缩放因子 $1/\\sqrt{d_k}$；多头常用 h=8、$d_k=64$、$d_{model}=512$
- Encoder 层 FFN 中间维度通常 4×d_model（512→2048）
- 复杂度：自注意力 $O(T^2 d)$；FFN $O(T d^2)$
- GPT-3 175B 参数、96 层；BERT-base 12 层 110M

**易错点**
1. softmax 前要**减最大值**防溢出（实验 1 已演示）
2. 注意力 mask 用 `-inf`（softmax 里 e^-inf = 0），不是 0
3. Decoder 的掩码是**下三角**：只能看当前位置及左侧
4. 位置编码在**加 embedding 之后**、Attention 之前
5. LayerNorm 的归一化轴是**特征维**（最后一维），不是序列维""")

md(r"""## 5. 面试速答（30 秒背诵版）

- **注意力公式**：$\\text{softmax}(QK^\\top/\\sqrt{d_k})V$；$\\sqrt{d_k}$ 防饱和
- **多头**：h 组 QKV 线性投影 → 拼接 → out_proj；每头关注不同关系
- **自注意力**：Q=K=V；复杂度 $O(T^2 d)$
- **Transformer 层**：MHA → 残差+LN → FFN → 残差+LN（Decoder 再嵌掩码与交叉注意力）
- **位置编码**：注意力无序 → 正弦/可学习位置注入
- **主流演进**：BERT（Encoder 双向）、GPT（Decoder 自回归）、flash-attention（省显存）""")

md(r"""## 6. 自测清单

- [ ] 默写注意力公式，说出除以 sqrt(d_k) 的原因
- [ ] 手写稳定版 softmax（减 max）与缩放点积注意力
- [ ] 手写多头（h 组投影 + 拼接 + out_proj）并与 torch 对照
- [ ] 画出 Encoder 单层结构图（MHA → Add+LN → FFN → Add+LN）
- [ ] 说清 Decoder 掩码与交叉注意力的区别
- [ ] 跑通 mini-Transformer 倒序任务（准确率 > 0.95）
- [ ] 说出自注意力复杂度与 Transformer 相对 RNN 的两大优势（并行、长依赖）

> 💡 下一篇 `05-预训练模型` 把 Transformer 放大成 BERT/GPT 并讲清"预训练-微调"范式。""")

nb = new_notebook(cells=cells, metadata=META)
path = os.path.join(OUT, '04-Attention与Transformer.ipynb')
nbformat.write(nb, path)
print('written:', path, '| cells:', len(cells))