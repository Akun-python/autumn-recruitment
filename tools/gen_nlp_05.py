# -*- coding: utf-8 -*-
"""生成 10-自然语言处理/教学/05-预训练模型与微调.ipynb（nbformat 4）"""
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
# 10-05 预训练模型与微调
# =====================================================================
md(r"""# 🧠 10-05 · 预训练模型与微调（BERT / GPT）

> 目标：理解「预训练 + 微调」范式。手写 **mini-BERT 的 MLM 预训练** 与 **mini-GPT 的自回归生成**，
> 亲眼看到「掩码填空」和「续写」两种训练目标如何塑造表示。

> 🧩 **生活化类比**：预训练 = 通识教育（大量阅读，学语言规律）；微调 = 岗前培训（少量专业资料，
> 学会具体任务）。BERT 是"完形填空型学霸"，GPT 是"接话型作家"。""")

md(r"""## 1. 范式：预训练 → 微调

**为什么需要预训练**：标注数据贵、无标注语料海量。先在无标注语料上学"通用语言能力"（预训练），
再在少量标注数据上学"任务能力"（微调）。

| 模型 | 架构 | 预训练任务 | 特点 |
|------|------|-----------|------|
| **ELMo** | 双向 LSTM | 前向+后向 LM | 早期上下文表示 |
| **BERT** | Transformer Encoder | MLM + NSP | 双向，编码器专用 |
| **GPT** | Transformer Decoder | 自回归 LM | 单向，生成/对话 |
| **T5** | Encoder-Decoder | 文本到文本 | 所有任务统一成生成 |

**上下文表示 vs 静态词向量**：Word2Vec 一个词一个向量（多义词无法区分）；BERT 里「苹果」在
「苹果公司」和「吃苹果」中向量不同——**表示随上下文变化**。""")

md(r"""## 2. BERT：双向 + MLM

**预训练任务**
- **MLM（掩码语言建模）**：随机 mask 15% 的词，用**左右两侧**上下文预测
  - 15% 中：80% 换 `[MASK]`、10% 换随机词、10% 保留（缓解预训练/微调不一致）
- **NSP（下一句预测）**：判断 B 句是否是 A 句的下一句（BERT 原始版本；RoBERTa 证明可去掉）

**输入表示**：`[CLS]` + 句子 A + `[SEP]` + 句子 B + `[SEP]`；token 级 = token embedding + 位置 embedding + 段 embedding。
`[CLS]` 的输出向量常被用作整句表示（分类任务）。

**微调**：加一个分类头/标注头，全模型梯度更新，2~5 epoch 即可。""")

code(r"""# ---------- 实验 1：mini-BERT —— 手写掩码语言模型（MLM） ----------
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)
sents = ['我 爱 北京 天安门', '北京 是 首都', '天安门 在 北京',
         '我 爱 学习 自然语言', '自然语言 很 有趣']
toks = [s.split() for s in sents]
vocab = ['[MASK]', '[PAD]'] + sorted({w for s in toks for w in s})
w2i = {w: i for i, w in enumerate(vocab)}; i2w = {i: w for w, i in w2i.items()}
V, L = len(vocab), 5
MASK_ID = w2i['[MASK]']; PAD_ID = w2i['[PAD]']
rng = np.random.default_rng(0)

def batch_of(n=6, mask_prob=0.15):
    xs, labels = [], []
    for _ in range(n):
        k = rng.integers(0, len(toks))
        seq = toks[k][:L]
        x = [w2i[w] for w in seq] + [PAD_ID] * (L - len(seq))
        lab = [-100] * L                       # -100 不参与 CE
        for p in range(len(seq)):
            if rng.random() < mask_prob:
                lab[p] = x[p]
                x[p] = MASK_ID
        if all(l == -100 for l in lab):        # 至少 mask 一个
            p = rng.integers(0, len(seq)); lab[p] = x[p]; x[p] = MASK_ID
        xs.append(x); labels.append(lab)
    return torch.tensor(xs), torch.tensor(labels)

class MiniBERT(nn.Module):
    def __init__(self, vocab, d=32, seq=5):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, seq, d) * 0.05)
        self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d); self.Wv = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, d))
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
    def forward(self, x):
        e = self.emb(x) + self.pos
        Q, K, V = self.Wq(e), self.Wk(e), self.Wv(e)
        sc = Q @ K.transpose(-1, -2) / np.sqrt(e.shape[-1])
        p = torch.softmax(sc, dim=-1)          # 双向：BERT 可以看到左右两侧
        a = self.ln1(e + p @ V)
        a = self.ln2(a + self.ffn(a))
        return self.head(a)                    # (B, L, V) 每个位置预测词

model = MiniBERT(V)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
ls = []
for ep in range(250):
    xb, yb = batch_of()
    logits = model(xb)
    loss = nn.functional.cross_entropy(logits.permute(0, 2, 1), yb, ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
    ls.append(loss.item())
    if ep % 80 == 0:
        print('ep %3d mlm loss %.3f' % (ep, loss.item()))

# 评估：逐位置 cloze（一次 mask 一个词，看能否靠上下文填回）——与训练分布一致
def eval_mlm():
    hits = tot = 0
    for s in toks:
        for j in range(len(s)):
            x = [w2i[w] for w in s[:L]] + [PAD_ID] * (L - len(s))
            x[j] = MASK_ID
            with torch.no_grad():
                pred = model(torch.tensor([x]))[0, j].argmax().item()
            hits += (pred == w2i[s[j]]); tot += 1
    return hits / tot

acc = eval_mlm()
print('MLM 自填准确率（mask 全部真词后靠上下文恢复）: %.3f' % acc)
assert acc > 0.85
# 演示：mask「北京」猜「首 都」
xb = torch.tensor([[w2i[w] for w in ['天安门', '在', '[MASK]']] + [PAD_ID, PAD_ID]])
with torch.no_grad():
    pred = model(xb)[0, 2].argmax().item()
print('「天安门 在 [MASK]」 -> 模型填:', i2w[pred])
assert pred == w2i['北京']""")

md(r"""## 3. GPT：单向自回归

- **训练目标**：给定前文预测下一个 token（自回归），只允许看左侧 → 因果掩码（下三角）
- **生成**：贪心 / 采样 / beam search；temperature 控制随机性
- **缩放法则**：模型越大、数据越多 → 能力越强（GPT-3 175B、ChatGPT 系列沿此路线）
- 与 BERT 对比：BERT 编码器双向（适合理解任务），GPT 解码器单向（适合生成任务）""")

code(r"""# ---------- 实验 2：mini-GPT —— 手写因果注意力 + 自回归生成 ----------
text = 'hello '                                   # 6 字符周期串
chars = sorted(set(text)); c2i = {c: i for i, c in enumerate(chars)}; i2c = {i: c for c, i in c2i.items()}
V2, T2 = len(chars), 6

class MiniGPT(nn.Module):
    def __init__(self, vocab, d=32, seq=6):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(1, seq, d) * 0.05)
        self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d); self.Wv = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, d))
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
    def forward(self, x, mask):
        e = self.emb(x) + self.pos
        Q, K, V = self.Wq(e), self.Wk(e), self.Wv(e)
        sc = Q @ K.transpose(-1, -2) / np.sqrt(e.shape[-1])
        sc = sc.masked_fill(mask, float('-inf'))   # 因果掩码：只看左侧
        p = torch.softmax(sc, dim=-1)
        a = self.ln1(e + p @ V)
        a = self.ln2(a + self.ffn(a))
        return self.head(a)

causal = torch.triu(torch.ones(T2, T2, dtype=torch.bool), diagonal=1)[None]
def gpt_batch(n=64):
    xs, ys = [], []
    for _ in range(n):
        s = rng.integers(0, len(text))
        x = [c2i[text[(s + k) % len(text)]] for k in range(T2)]
        y = [c2i[text[(s + k + 1) % len(text)]] for k in range(T2)]
        xs.append(x); ys.append(y)
    return torch.tensor(xs), torch.tensor(ys)

model2 = MiniGPT(V2)
opt2 = torch.optim.Adam(model2.parameters(), lr=5e-3)
for ep in range(150):
    xb, yb = gpt_batch()
    logits = model2(xb, causal)
    loss = nn.functional.cross_entropy(logits.permute(0, 2, 1), yb)
    opt2.zero_grad(); loss.backward(); opt2.step()
    if ep % 50 == 0:
        print('ep %3d lm loss %.3f' % (ep, loss.item()))

with torch.no_grad():
    xb, yb = gpt_batch(300)
    acc2 = (model2(xb, causal).argmax(-1) == yb).float().mean().item()
print('next-char 准确率: %.3f' % acc2)
assert acc2 > 0.9

# 自回归生成：从 'h' 开始，用自己预测的字符续写
with torch.no_grad():
    x = torch.tensor([[c2i['h']]])
    gen = ['h']
    for _ in range(11):
        if x.shape[1] < T2:
            xp = torch.cat([torch.full((1, T2 - x.shape[1]), c2i['h']), x], dim=1)
        else:
            xp = x[:, -T2:]
        m = torch.triu(torch.ones(xp.shape[1], xp.shape[1], dtype=torch.bool), diagonal=1)[None]
        nxt = model2(xp, m)[0, -1].argmax().item()
        gen.append(i2c[nxt])
        x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
print('mini-GPT 生成:', ''.join(gen))
assert ''.join(gen).startswith('hello')""")

md(r"""## 4. 微调与 Prompt

**微调（fine-tuning）**：在预训练权重上，用任务数据继续训练（分类头 + 少量 epoch）
**Prompt / 指令微调**：把任务写成自然语言指令，让模型生成答案（不换头，只换"问法"）
**RLHF**：人类反馈强化学习，让生成对齐人类偏好（GPT-3.5/4 的关键一步）

> 面试主线：**预训练学通用语言 → 微调学任务 → 对齐学偏好**。""")

md(r"""## 5. 数字敏感度与易错点（背诵）

**数字**
- BERT-base：12 层、768 维、12 头、110M 参数；BERT-large：340M
- MLM mask 比例 15%（80/10/10 拆分）；微调典型 2~5 epoch
- GPT-3：175B 参数、96 层、batch 巨大（3.2M tokens）
- 上下文窗口：BERT 512 tokens；现代 LLM 8K~1M+

**易错点**
1. BERT 是**双向**（无因果掩码），GPT 是**单向**（因果掩码）——结构即任务
2. MLM 的 80/10/10 是为缓解「预训练见 [MASK]、微调不见 [MASK]」的不一致
3. NSP 已被证伪（RoBERTa 移除后更强）；现在主流是「纯 MLM」或「纯 LM」
4. 微调 ≠ 从头训：学习率要小（1e-5~5e-5），防灾难性遗忘""")

md(r"""## 6. 面试速答（30 秒背诵版）

- **范式**：预训练（无标注语料学通用能力）+ 微调（少量标注学任务）
- **BERT**：Encoder 双向；MLM（mask 15%，80/10/10）+ NSP
- **GPT**：Decoder 单向；自回归 next-token；因果掩码
- **上下文表示**：一词多义靠上下文区分，Word2Vec 做不到
- **微调技巧**：小学习率、加分类头、防遗忘
- **对齐**：RLHF 让生成符合人类偏好""")

md(r"""## 7. 自测清单

- [ ] 说出 ELMo/BERT/GPT/T5 的架构与预训练任务
- [ ] 默写 MLM 的 80/10/10 规则与原因
- [ ] 手写 mini-BERT MLM 训练（mask 预测准确率 > 0.85）
- [ ] 画出生成时的因果掩码（下三角）并手写 mini-GPT 生成
- [ ] 说清 BERT 与 GPT 的结构差异如何决定任务分工
- [ ] 说出微调的三个实践要点（学习率/epoch/防遗忘）
- [ ] 解释 RLHF 在预训练-微调链条中的位置

> 💡 下一篇 `06-文本分类与序列标注` 用预训练/传统方法落到两个真实任务。""")

nb = new_notebook(cells=cells, metadata=META)
path = os.path.join(OUT, '05-预训练模型与微调.ipynb')
nbformat.write(nb, path)
print('written:', path, '| cells:', len(cells))