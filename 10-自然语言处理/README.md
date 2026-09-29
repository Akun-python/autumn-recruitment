# 📝 10-自然语言处理

> 目标：把「自然语言处理」讲成一条主线——**文本表示（00）→ 统计语言模型（01）→ 词向量（02）→ 序列模型（03）→ Attention/Transformer（04）→ 预训练模型（05）→ 任务与评估（06）→ 面试八股（07）**。
> 面试考察的不是背模型名，而是 **每代方法解决什么问题、手写哪些核心算法、数字怎么记、在什么场景选谁**。

## 复习策略

1. **先建表示层（00-02）**：分词/One-hot/TF-IDF → n-gram 语言模型与平滑（PPL 是什么）→ Word2Vec 两种结构与负采样 → GloVe 共现矩阵分解，理解"词从符号到向量"
2. **再上序列层（03-04）**：RNN/BPTT/梯度消失 → LSTM 门控（手撕前向对照 torch）→ 注意力公式 → 多头 → mini-Transformer 学倒序（结构即能力：双向 vs 因果）
3. **冲到预训练（05）**：MLM（mini-BERT 手写）→ 自回归生成（mini-GPT）→ 微调 vs Prompt → RLHF
4. **落到任务（06）**：朴素贝叶斯分类 → BiLSTM 序列标注 → 手写混淆矩阵/P/R/F1/macro/micro
5. **手推优先**：TF-IDF 公式、n-gram 平滑、PPL、负采样梯度、RNN/BPTT、LSTM 四门、注意力/多头、MLM 训练目标、贝叶斯分类概率、P/R/F1 定义

## 自然语言处理笔记索引（`教学/`，8 篇，全部含推导 + 手写实现 + 对照验证 + 自测清单）

| 篇目 | 核心内容 | 对照验证 |
|------|----------|----------|
| 00-文本表示与TF-IDF | 中文词典分词/One-hot/BOW/TF-IDF/余弦相似度 | 手写 BOW·TF-IDF vs sklearn Count/TfidfVectorizer 一致 |
| 01-n-gram与统计语言模型 | 链式法则/马尔可夫/加一·拉普拉斯平滑/PPL/稀疏-未平滑 inf 演示 | 手写 PPL 与公式手算一致、覆盖 vs 未见句 PPL 对比 |
| 02-Word2Vec与GloVe | CBOW vs Skip-gram/负采样/3/4 幂次/共现矩阵/截断 SVD | 手写余弦相似度检索、logM 低秩分解重构误差 |
| 03-RNN与LSTM | RNN 前向/BPTT/数值梯度校验/LSTM 四门控/CharRNN 生成 | 手写 RNN·LSTM vs 数值梯度 <1e-6、LSTM vs torch <1e-5、toy 生成 acc>0.9 |
| 04-Attention与Transformer | 缩放点积注意力/softmax/多头拼接/mini-Transformer 倒序任务 | 手写注意力 vs torch `scaled_dot_product_attention` <1e-5、倒序 acc>0.95 |
| 05-预训练模型与微调 | MLM 概率掩码/mini-BERT 双向注意力/mini-GPT 因果掩码自回归/微调 | MLM cloze acc>0.85、GPT 生成 `hello` 前缀、掩码「北京」演示 |
| 06-文本分类与序列标注 | 朴素贝叶斯（加一平滑+对数）/BiLSTM 词性标注/混淆矩阵·P/R/F1·macro/micro | 手写贝叶斯 vs sklearn MultinomialNB acc>0.9、BiLSTM token acc>0.95、指标逐位一致 |
| 07-NLP面试八股 | 90 连问（6 组×15）+ TF-IDF/平滑 PPL/负采样/注意力/谱系图手撕 | 手撕题目全部可执行 |

## 本目录规划

```
10-自然语言处理/
├── README.md          # 本文件
├── 高频面试题.md      # 分类面经清单（带 TODO 打勾）
└── 教学/              # ✅ 8 篇 Jupyter notebook（推导 + 代码 + 对照 + 自测）
    └── images/        # 部分 notebook 落盘的插图
```

## 面试 30 秒速答（背诵骨架）

**表示与统计**
- **TF-IDF**：词在文档越常见（TF）、全库越稀有（IDF）越重要；IDF = log(N/df)
- **n-gram 平滑**：加一 `(c+k)/(c_prev+V·k)`；避免 0 概率 → PPL 从 inf 变有限
- **PPL**：exp(平均负对数似然)，同一语料内越低越好
- **Skip-gram vs CBOW**：中心→上下文（对低频友好）vs 上下文→中心（快）
- **负采样**：正样本 log σ + 负样本 log σ(-)，采样分布 3/4 次幂提稀有词

**序列与注意力**
- **BPTT**：沿时间展开反传，梯度 ∝ Whh^T 连乘 → 消失/爆炸；裁剪治爆炸
- **LSTM**：f/i/o 三门 + 细胞 c 加性更新（梯度高速公路）；遗忘门偏置初始≈1
- **注意力**：`softmax(QK^T/√d)V`；除 √d 防 softmax 饱和；多头=多子空间
- **因果掩码**：下三角 -inf → softmax 变 0；Decoder 只看左侧
- **Transformer vs RNN**：全局并行 vs 串行递推；KV cache 免重复计算

**预训练与任务**
- **BERT**：MLM（15% 掩码，80/10/10）+ 双向 → 理解；**GPT**：因果自回归 → 生成
- **微调要点**：小学习率（1e-5~5e-5）防破坏表示；RLHF = SFT → 奖励模型 → PPO
- **文本分类**：Acc 不够看 macro-F1；**NER**：实体级 P/R/F1（边界错算错）
- **BLEU**：候选 vs 参考 1-4 gram 重合 × 长度惩罚；ROUGE 看参考词覆盖（摘要）

## 验收标准

- [x] `高频面试题.md` 覆盖 分词/表示 / 语言模型 / 词向量 / RNN / Transformer / 预训练 / 任务与评估
- [x] 自然语言处理笔记 8 篇，全部含推导 + 手写实现 + 对照验证 + 自测清单
- [x] 能手推：TF-IDF、平滑与 PPL、负采样梯度、BPTT、LSTM 四门、注意力/多头、MLM 目标、贝叶斯概率、P/R/F1
- [x] 能口算：多头拼接维度、自注意力复杂度 O(T²d)、KV cache 省多少、BERT 三 embedding 相加
- [x] 全部 notebook 代码经批量执行验证（numpy/sklearn/torch 对照断言通过）

> 💡 手推优先级建议：TF-IDF → n-gram 平滑 → PPL → 负采样一步梯度 → BPTT → LSTM 前向 → 注意力公式 → 多头拼接 → MLM 80/10/10 → 混淆矩阵与 macro-F1。