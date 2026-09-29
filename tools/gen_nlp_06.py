# -*- coding: utf-8 -*-
"""生成 10-自然语言处理/教学/06-文本分类与序列标注.ipynb（nbformat 4）"""
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
# 10-06 文本分类与序列标注
# =====================================================================
md(r"""# 🏷️ 10-06 · 文本分类与序列标注

> 目标：两大经典 NLP 任务。手写 **朴素贝叶斯文本分类**（与 sklearn 对照）、**BiLSTM 序列标注**，
> 手写 **Precision/Recall/F1/混淆矩阵**，建立"任务 → 特征/模型 → 评估"的完整闭环。

> 🧩 **生活化类比**：文本分类 = 给整篇文章贴标签（"这封邮件是垃圾吗？"）；
> 序列标注 = 给每个词贴标签（"张三/人名 昨天/时间 去了/动词 北京/地点"）。""")

md(r"""## 1. 文本分类：从 TF-IDF + 分类器说起

**管线**：分词 → TF-IDF/词袋向量 → 分类器（朴素贝叶斯 / 逻辑回归 / SVM / 树模型）

**朴素贝叶斯**（生成式）：
$$P(y|x) \\propto P(y) \\prod_{w \\in x} P(w|y)$$
- 条件独立假设：词之间独立（虽然不成立，但小文本上常够用）
- 训练：统计每类文档中每个词的出现（加一平滑）
- 预测：选 $\\arg\\max_y P(y) \\prod_w P(w|y)$（对数化防下溢）""")

code(r"""# ---------- 实验 1：手写朴素贝叶斯文本分类（加一平滑） vs sklearn ----------
import numpy as np
from collections import Counter

np.random.seed(0); rng = np.random.default_rng(0)

# 合成两类语料：体育（球/队/比赛/冠军） vs 科技（算法/模型/数据/智能）
def make_docs(n_per=60, seed=0):
    r = np.random.default_rng(seed)
    sport = ['球', '队', '比赛', '冠军', '球员', '联赛']
    tech = ['算法', '模型', '数据', '智能', '网络', '训练']
    docs, labels = [], []
    for _ in range(n_per):
        docs.append(' '.join(r.choice(sport, size=r.integers(4, 9)))); labels.append(0)
    for _ in range(n_per):
        docs.append(' '.join(r.choice(tech, size=r.integers(4, 9)))); labels.append(1)
    return docs, np.array(labels)

docs, labels = make_docs(60)
docs_te, labels_te = make_docs(30, seed=7)

# 手写：词表 + 类条件概率（加一平滑）
vocab = sorted({w for d in docs for w in d.split()})
v2i = {w: i for i, w in enumerate(vocab)}
cls_docs = {0: [d for d, y in zip(docs, labels) if y == 0],
            1: [d for d, y in zip(docs, labels) if y == 1]}
log_pw = {}
for c in (0, 1):
    cnt = Counter(w for d in cls_docs[c] for w in d.split())
    tot = sum(cnt.values()) + len(vocab)                    # 加一平滑分母
    log_pw[c] = {w: np.log((cnt[w] + 1) / tot) for w in vocab}
log_prior = {0: np.log(len(cls_docs[0]) / len(docs)), 1: np.log(len(cls_docs[1]) / len(docs))}

def predict_nb(d):
    scores = {}
    for c in (0, 1):
        s = log_prior[c]
        for w in d.split():
            s += log_pw[c].get(w, np.log(1 / (sum(len(v) for v in log_pw[c]) + 1)))  # 未见词兜底
        scores[c] = s
    return max(scores, key=scores.get)

preds = [predict_nb(d) for d in docs_te]
acc_nb = np.mean(np.array(preds) == labels_te)
print('手写朴素贝叶斯测试准确率: %.3f' % acc_nb)
assert acc_nb > 0.9

# sklearn 对照：CountVectorizer + MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
cv = CountVectorizer(token_pattern='[\\u4e00-\\u9fa5]+')
Xtr = cv.fit_transform(docs); Xte = cv.transform(docs_te)
nb_sk = MultinomialNB(alpha=1.0).fit(Xtr, labels)
acc_sk = nb_sk.score(Xte, labels_te)
print('sklearn MultinomialNB 测试准确率: %.3f' % acc_sk)
print('结论：手写与 sklearn 殊途同归（都 > 0.9）——逻辑一致、实现有别')""")

md(r"""## 2. 序列标注：BIO 与 BiLSTM+CRF

**BIO 标注**：B（Begin）实体开始、I（Inside）实体内部、O（Outside）实体外。
例：`张三 昨天 去 北京` → `B-PER O O B-LOC`

**BiLSTM-CRF**（SOTA 经典结构）
- BiLSTM 编码每个词的双向上下文 → 发射分数（每个位置对各标签的得分）
- CRF 层建模标签间转移约束（如 B 后不能直接跟 I）
- 只用 BiLSTM 时相邻标签独立性差 → CRF 补转移

> 教学版：本实验用 BiLSTM 做词性标注（简化版，无 CRF），体会序列建模与逐位置评估。""")

code(r"""# ---------- 实验 2：BiLSTM 词性序列标注（torch） ----------
import torch
import torch.nn as nn

torch.manual_seed(0)
# 简化词性：N=名词 V=动词 A=形容词；句法模板随机生成
templates = [['N', 'V', 'N'], ['N', 'V', 'N', 'N'], ['A', 'N', 'V', 'N'], ['N', 'A', 'V', 'A', 'N']]
lexicon = {'N': ['苹果', '北京', '学生', '老师', '天气'], 'V': ['吃', '去', '学习', '喜欢'],
           'A': ['好', '大', '聪明', '晴朗']}
all_tokens = sorted({w for ws in lexicon.values() for w in ws})
t2i = {w: i for i, w in enumerate(all_tokens)}
tag_ids = {'N': 0, 'V': 1, 'A': 2}

def gen_pos_data(n=500, seed=0):
    r = np.random.default_rng(seed)
    Xs, Ys = [], []
    for _ in range(n):
        tmpl = templates[r.integers(0, len(templates))]
        seq = [r.choice(lexicon[t]) for t in tmpl]
        Xs.append([t2i[w] for w in seq])
        Ys.append([tag_ids[t] for t in tmpl])
    return Xs, Ys

Xs_tr, Ys_tr = gen_pos_data(400); Xs_te, Ys_te = gen_pos_data(100, seed=9)
V = len(all_tokens); TAG = 3; MAXL = 5

def pad_batch(Xs, Ys):
    xs = torch.zeros(len(Xs), MAXL, dtype=torch.long)
    ys = torch.full((len(Xs), MAXL), -100, dtype=torch.long)   # -100 忽略填充
    for i, (x, y) in enumerate(zip(Xs, Ys)):
        xs[i, :len(x)] = torch.tensor(x); ys[i, :len(y)] = torch.tensor(y)
    return xs, ys

class BiTagger(nn.Module):
    def __init__(self, vocab, d=24, tag=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.lstm = nn.LSTM(d, d, bidirectional=True, batch_first=True)
        self.head = nn.Linear(2 * d, tag)
    def forward(self, x):
        h = self.lstm(self.emb(x))[0]
        return self.head(h)

model = BiTagger(V)
opt = torch.optim.Adam(model.parameters(), lr=5e-3)
for ep in range(60):
    idx = np.random.default_rng(ep).permutation(len(Xs_tr))[:160]
    xs, ys = pad_batch([Xs_tr[i] for i in idx], [Ys_tr[i] for i in idx])
    logits = model(xs)
    loss = nn.functional.cross_entropy(logits.permute(0, 2, 1), ys, ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()

xs_te, ys_te = pad_batch(Xs_te, Ys_te)
with torch.no_grad():
    pred = model(xs_te).argmax(-1)
mask = ys_te != -100
acc_seq = ((pred == ys_te) & mask).sum().item() / mask.sum().item()
print('BiLSTM 词性标注 token 准确率: %.3f' % acc_seq)
assert acc_seq > 0.95
# 演示：训练集第一条句子
demo_words = [all_tokens[i] for i in Xs_tr[0]]
with torch.no_grad():
    xs_d = torch.zeros(1, MAXL, dtype=torch.long)
    xs_d[0, :len(Xs_tr[0])] = torch.tensor(Xs_tr[0])
    pd = model(xs_d)[0, :len(Xs_tr[0])].argmax(-1).tolist()
tag_name = {0: 'N', 1: 'V', 2: 'A'}
print('示例: %s' % ' '.join(demo_words))
print('真实: %s' % ' '.join(tag_name[t] for t in Ys_tr[0]))
print('预测: %s' % ' '.join(tag_name[t] for t in pd))""")

md(r"""## 3. 评估指标：Precision / Recall / F1 / 混淆矩阵

- 混淆矩阵：TP / FP / FN / TN
- $P = \\dfrac{TP}{TP+FP}$（查准：预测为正的里面有多少对的）
- $R = \\dfrac{TP}{TP+FN}$（查全：真正的正例里抓到多少）
- $F1 = \\dfrac{2PR}{P+R}$（调和平均，兼顾两者）
- 多分类：macro-F1（每类算再平均）/ micro-F1（全局 TP/FP/FN 汇总）""")

code(r"""# ---------- 实验 3：手写指标 vs sklearn 对照 ----------
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

y_true = np.array([0, 1, 1, 0, 2, 1, 0, 2, 2, 1])
y_pred = np.array([0, 1, 0, 0, 2, 2, 0, 2, 1, 1])

def my_confusion(y_true, y_pred, n=3):
    C = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        C[t, p] += 1
    return C

C = my_confusion(y_true, y_pred)
print('手写混淆矩阵:\n', C)
assert (C == confusion_matrix(y_true, y_pred)).all()

def my_prf(y_true, y_pred, n=3):
    P, R, F1 = [], [], []
    for c in range(n):
        tp = ((y_true == c) & (y_pred == c)).sum()
        fp = ((y_true != c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        P.append(tp / max(tp + fp, 1)); R.append(tp / max(tp + fn, 1))
        F1.append(2 * P[-1] * R[-1] / max(P[-1] + R[-1], 1e-9))
    macro = float(np.mean(F1))
    tp = int((y_true == y_pred).sum())                 # micro：全局汇总
    fp = int((y_pred != y_true).sum()); fn = fp
    micro = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    return P, R, F1, macro, micro

P, R, F1, macro, micro = my_prf(y_true, y_pred)
P2, R2, F2, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2])
print('手写 P:', np.round(P, 3), ' | sklearn P:', np.round(P2, 3))
assert np.allclose(P, P2, atol=1e-9) and np.allclose(F1, F2, atol=1e-9)
_, _, F_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro', labels=[0, 1, 2])
print('macro-F1 手写=%.3f | sklearn=%.3f' % (macro, np.mean(F2)))
print('micro-F1 手写=%.3f | sklearn=%.3f' % (micro, F_micro))
assert abs(macro - np.mean(F2)) < 1e-9 and abs(micro - F_micro) < 1e-9
print('要点：指标全部一致——评估代码必须能逐位对上 sklearn，才算"会了"')""")

md(r"""## 4. 指标敏感性（背诵）

| 场景 | 该看哪个指标 | 原因 |
|------|-------------|------|
| 垃圾邮件（正类极少） | Recall / F1 | 漏判危害大，Accuracy 虚高 |
| 广告审核（宁错勿漏） | Precision | 误杀成本高 |
| 类别均衡 | Accuracy | 简单直观 |
| 多标签/多类 | macro-F1 | 平等对待每个类 |

**不平衡处理**：重采样（过采样少数类）、类别权重、Focal loss（难样本加权）。""")

md(r"""## 5. 面试速答（30 秒背诵版）

- **文本分类管线**：分词 → TF-IDF/词袋 → 分类器；深度学习直接微调预训练模型
- **朴素贝叶斯**：$P(y)\\prod P(w|y)$，条件独立假设 + 加一平滑；小文本够用
- **序列标注**：BIO 方案；BiLSTM 双向编码 + CRF 约束转移
- **指标**：P=TP/(TP+FP)、R=TP/(TP+FN)、F1=2PR/(P+R)；macro 逐类平均 vs micro 全局汇总
- **不平衡**：重采样/类权重/Focal loss
- **现代方案**：BERT 微调做分类与标注（05 篇）——数据足时碾压传统管线""")

md(r"""## 6. 自测清单

- [ ] 手写朴素贝叶斯分类（平滑 + 对数防下溢），准确率 > 0.9
- [ ] 手写混淆矩阵与 P/R/F1，与 sklearn 逐位一致
- [ ] 跑通 BiLSTM 序列标注（token 准确率 > 0.95）
- [ ] 说出 BIO 三个标记与示例
- [ ] 口算：TP=90 FP=10 FN=5 时的 P/R/F1
- [ ] 说出三类不平衡处理手段
- [ ] 比较传统管线与预训练微调管线的适用场景

> 💡 下一篇 `07-NLP 面试八股` 把全部 0~6 篇考点串成高频问答。""")

nb = new_notebook(cells=cells, metadata=META)
path = os.path.join(OUT, '06-文本分类与序列标注.ipynb')
nbformat.write(nb, path)
print('written:', path, '| cells:', len(cells))