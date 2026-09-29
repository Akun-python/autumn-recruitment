# -*- coding: utf-8 -*-
"""生成 10-自然语言处理/教学/01-n-gram与统计语言模型.ipynb（nbformat 4）"""
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
# 10-01 n-gram 与统计语言模型
# =====================================================================
md(r"""# 📊 10-01 · n-gram 与统计语言模型

> 目标：回答「一句话出现的概率是多少」。手写**n-gram 计数、加一/加 k 平滑、Perplexity 评估**，
> 在小语料上对比 unigram/bigram，理解马尔可夫假设如何把指数级组合变成可统计的计数。

> 🧩 **生活化类比**：语言模型 = 一个「接话大师」。你说"今天天气真"，它能接"好"而不是"红烧肉"——
> 靠的是统计语料里「天气真→好」出现的频率（n-gram），或让神经网络自己学（后面的 RNN/Transformer）。""")

md(r"""## 1. 语言模型与链式法则

句子概率按链式法则展开：

$$P(w_1 \\ldots w_n) = P(w_1) P(w_2|w_1) P(w_3|w_1 w_2) \\cdots P(w_n|w_1 \\ldots w_{n-1})$$

- 条件概率的上下文随 n 指数增长 → 无法直接统计
- **马尔可夫假设**：只依赖最近 $n{-}1$ 个词 → **n-gram 模型**
  - unigram：$P(w_i)$
  - bigram：$P(w_i | w_{i-1})$
  - trigram：$P(w_i | w_{i-2}, w_{i-1})$
- 记法：$\\langle s \\rangle$（句首）与 $\\langle /s \\rangle$（句尾）标记

**用途**：生成文本、拼写/输入法纠错、语音识别候选排序、机器翻译重排——也是预训练模型的根基。""")

code(r"""# ---------- 实验 1：小语料 unigram/bigram 统计 + 生成 ----------
import numpy as np
from collections import Counter

corpus = ['我 喜欢 学习 自然语言处理',
          '我 喜欢 深度 学习',
          '他 喜欢 学习 数学',
          '我 不 喜欢 下雨',
          '他 喜欢 下雨 天']
tokens = [sent.split() for sent in corpus]

def make_ngrams(sents, n):
    grams = []
    for s in sents:
        seq = ['<s>'] * (n - 1) + s + ['</s>']
        grams.append([tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)])
    return grams

uni = make_ngrams(tokens, 1)
bi = make_ngrams(tokens, 2)

counts1 = Counter(g[0] for s in uni for g in s)
counts2 = Counter(g for s in bi for g in s)
total1 = sum(counts1.values())

def p_unigram(w):
    return counts1[w] / total1

def p_bigram(w2, w1):
    return counts2[(w1, w2)] / counts1[w1]

counts1['<s>'] = len(tokens)          # unigram 无句首标记，补上供生成使用

print('unigram P(喜欢) = %.3f' % p_unigram('喜欢'))
print('bigram  P(学习|喜欢) = %.3f | P(下雨|喜欢) = %.3f' %
      (p_bigram('学习', '喜欢'), p_bigram('下雨', '喜欢')))
print('bigram  P(天|下雨) = %.3f | P(喜欢|下雨) = %.3f' %
      (p_bigram('天', '下雨'), p_bigram('喜欢', '下雨')))
assert p_bigram('学习', '喜欢') > 0      # 「喜欢学习」在语料中出现 2 次
assert p_bigram('天', '下雨') > 0

# 用 bigram 生成一句话
def generate(counts2, counts1, max_len=8):
    out = ['<s>']
    for _ in range(max_len):
        w1 = out[-1]
        cands = [w2 for (a, w2) in counts2 if a == w1]
        if not cands:
            break
        probs = np.array([counts2[(w1, w2)] / counts1[w1] for w2 in cands])
        w = np.random.choice(cands, p=probs)
        if w == '</s>':
            break
        out.append(w)
    return ' '.join(out[1:])

np.random.seed(1)
for _ in range(3):
    print('生成:', generate(counts2, counts1))
print('要点：n-gram 生成只靠计数，但语料小 -> 生成空洞且重复')""")

md(r"""## 2. 数据稀疏与平滑（重点）

**问题**：测试语料里总有训练时没见过的词/词组 → 概率为 0 → 整句概率为 0（灾难性）

**平滑方法**
- **加一平滑（Laplace）**：$P(w_i | w_{i-1}) = \\dfrac{\\text{count}(w_{i-1}, w_i) + 1}{\\text{count}(w_{i-1}) + V}$
- **加 k 平滑**：$+k$（k 可调超参）
- **回退（backoff）**：bigram 没数据就退到 unigram
- **插值（interpolation）**：$\\lambda_1 P_{uni} + \\lambda_2 P_{bi} + \\lambda_3 P_{tri}$
- **Good-Turing / Kneser-Ney**：更精细的频次重分配

> 面试点：**平滑的本质是"把概率质量从高频事件分一点给低频/未见过事件"**；
> 加一平滑对词汇表大的场景偏保守（把太多概率给了未见词）。""")

code(r"""# ---------- 实验 2：平滑对比（未见词组不再为 0） ----------
V = len(counts1)   # 词汇表大小

def p_bi_laplace(w2, w1, k=1.0):
    return (counts2[(w1, w2)] + k) / (counts1[w1] + k * V)

print('未见词组 P(学习|下雨) 原始=0 | 加一平滑=%.4f' % p_bi_laplace('学习', '下雨'))
print('常见词组 P(学习|喜欢) 原始=%.3f | 加一平滑=%.3f' %
      (p_bigram('学习', '喜欢'), p_bi_laplace('学习', '喜欢')))
print('概率总和（给定 w1=喜欢 的所有候选）:', round(sum(p_bi_laplace(w2, '喜欢') for w2 in counts1), 6), '（应为 1）')
assert abs(sum(p_bi_laplace(w2, '喜欢') for w2 in counts1) - 1.0) < 1e-6

# 平滑前后生成质量的直观对比：平滑让"不搭"的词组也有概率但概率很低
print('要点：平滑保留"可能性"，罕见组合概率小但不为 0 —— 语言模型不能给任何合法句 0 概率')""")

md(r"""## 3. Perplexity：语言模型的考试分数

$$\\text{PPL} = 2^{-\\frac{1}{N} \\sum_{i=1}^{N} \\log_2 P(w_i | \\text{context})}
= \\exp\\left(-\\frac{1}{N} \\sum_i \\log P(w_i|\\text{context})\\right)$$

- 平均对数概率的指数形式：**PPL 越小越好**（=每个词的平均"候选数"）
- PPL=10：每个词平均要从 10 个候选里猜
- 性质：均匀分布时 PPL = V；好模型（GPT-3）PPL 可 < 10
- 比较不同模型必须在**同一语料/同一切分**下，否则不可比""")

code(r"""# ---------- 实验 3：手写 Perplexity，对比 unigram vs bigram ----------
# 严格留一：只用前 4 句训练，第 5 句的部分片段做测试
train5 = tokens[:4]
uni5 = make_ngrams(train5, 1); bi5 = make_ngrams(train5, 2)
c1 = Counter(g[0] for s in uni5 for g in s)
c1['<s>'] = len(train5)                    # unigram 无句首标记，补上供 bigram 条件概率用
c2 = Counter(g for s in bi5 for g in s)
t1 = sum(c1.values())
p1 = lambda w: c1[w] / t1
p2 = lambda w2, w1: c2[(w1, w2)] / c1[w1]

def ppl(sents, mode):
    logp, cnt = 0.0, 0
    for s in sents:
        seq = ['<s>'] + s + ['</s>']
        for i in range(1, len(seq)):
            p = p1(seq[i]) if mode == 'uni' else p2(seq[i], seq[i - 1])
            if p == 0:
                return float('inf')           # 未平滑=0：整句概率为 0，PPL=inf（演示稀疏后果）
            logp += np.log(p); cnt += 1
    return float(np.exp(-logp / cnt))

# 测试 1：训练完全覆盖的句子，bigram 应比 unigram 更准
test_covered = [['我', '喜欢', '学习']]
print('覆盖句 PPL: unigram=%.2f | bigram=%.2f（bigram 用上上下文 -> 更准）' %
      (ppl(test_covered, 'uni'), ppl(test_covered, 'bi')))
assert ppl(test_covered, 'bi') < ppl(test_covered, 'uni')

# 测试 2：词组「数学 自然语言处理」在训练里从未连用 -> bigram 概率 0 -> PPL=inf
test_unseen = [['学习', '数学', '自然语言处理']]
print('未见词组句 PPL: unigram=%.2f | 未平滑 bigram=%s（「数学|自然语言处理」没出现过 -> 0 概率）' %
      (ppl(test_unseen, 'uni'), ppl(test_unseen, 'bi')))
assert ppl(test_unseen, 'bi') == float('inf')
print('这正是必须做平滑的原因（见实验 4）')""")

md(r"""## 4. 平滑 + Perplexity 的完整闭环（合并版）

完整评估：**训练集计数 → 平滑（处理稀疏）→ 测试集算 PPL（衡量预测质量）**。
注意区分：平滑在**训练侧**做；PPL 在**测试侧**算；测试集必须与训练集**词表一致**（未见词按未登录词处理）。""")

code(r"""# ---------- 实验 4：平滑后 bigram 的 PPL（与未平滑对比） ----------
def ppl_laplace(sents, k=1.0):
    logp, cnt = 0.0, 0
    for s in sents:
        seq = ['<s>'] + s + ['</s>']
        for i in range(1, len(seq)):
            p = (counts2[(seq[i - 1], seq[i])] + k) / (counts1[seq[i - 1]] + k * V)
            logp += np.log(p); cnt += 1
    return float(np.exp(-logp / cnt))

ppl_sm = ppl_laplace(test_unseen)
print('同一未见句「他 喜欢 下雨 天」: 未平滑 bigram PPL=%s -> 加一平滑后 PPL=%.2f' %
      (ppl(test_unseen, 'bi'), ppl_sm))
assert ppl_sm < float('inf')
assert ppl_sm < ppl(test_unseen, 'uni')
print('要点：平滑让所有组合概率 >0，PPL 恢复为有限值；罕见组合概率小但不为 0')""")

md(r"""## 5. n-gram 的局限与演进

| 局限 | 说明 | 演进 |
|------|------|------|
| 长依赖 | 窗口有限，管不到长距离关系 | RNN/LSTM（03 篇） |
| 稀疏 | 数据稀疏，平滑只是缓解 | 词向量 + 神经网络概率化（02 篇） |
| 无语义 | 纯计数，不编码相似性 | Word2Vec / BERT（02/05 篇） |
| 存储 | 高阶 n-gram 组合爆炸 | 神经语言模型共享参数 |

**数字速记**：V=1e5 时 trigram 理论组合 $V^2=10^{10}$；实际语料能覆盖的极少 → 稀疏必然。""")

md(r"""## 6. 面试速答（30 秒背诵版）

- **链式法则 + 马尔可夫假设**：n-gram 把条件概率截断到 n-1 个上文词
- **平滑为什么必要**：稀疏导致 0 概率 → 整句概率为 0；平滑=从高频分概率给低频
- **加一平滑公式**：$(c + 1)/(N + V)$；加 k、插值、回退是进阶
- **PPL 公式与含义**：$2^{-\\frac1N \\sum \\log_2 P}$，越小越好，均匀分布时 = V
- **n-gram vs 神经网络 LM**：计数 vs 参数共享 + 语义泛化
- **生成**：n-gram 采样空洞重复，神经网络生成更流畅""")

md(r"""## 7. 自测清单

- [ ] 默写链式法则与 bigram 概率公式（含句首句尾标记）
- [ ] 手写 unigram/bigram 计数与条件概率
- [ ] 默写加一平滑公式，口算：count=0、V=10 时 P=1/10
- [ ] 手写 Perplexity，说清与平均负对数概率的关系
- [ ] 说出平滑的三种方法与各自适用场景
- [ ] 口算：均匀分布 100 词时 PPL=100；PPL=10 的含义
- [ ] 说清 n-gram 三大局限与神经网络 LM 的对应改进

> 💡 下一篇 `02-Word2Vec 与 GloVe` 从「计数概率」跨到「分布式词向量」。""")

nb = new_notebook(cells=cells, metadata=META)
path = os.path.join(OUT, '01-n-gram与统计语言模型.ipynb')
nbformat.write(nb, path)
print('written:', path, '| cells:', len(cells))