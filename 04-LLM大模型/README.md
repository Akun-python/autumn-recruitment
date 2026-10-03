# 🚀 04-LLM大模型（重点模块）

> 🖼️ 配图：25 篇教学 notebook 均配有配图（Transformer 架构/位置编码/注意力/缓存层级/RAG/知识蒸馏等 Wikimedia Commons 示意图见 `教学/images/web_*`，标注出处与许可；LoRA、解码采样为本地教学示意图 `web_*_local.png`）。

> 大模型岗/算法岗的差异化竞争力所在。你有 **MiniMind 从零构建 LLM** 的动手实践（预训练/SFT/RL），这是别的候选人很难复制的优势——面试时要把"做过什么 + 踩过什么坑 + 怎么改进"讲透。

## 知识地图（面试考察的六大块）

```
LLM 考点
├── 1. 模型架构     Transformer 细节 / 位置编码 / GQA / MoE / 激活函数
├── 2. 分词与数据   BPE / 词表设计 / 数据配比 / 清洗
├── 3. 训练范式     Pretrain → SFT → RLHF(PPO/DPO/GRPO) → 损失函数对比
├── 4. 推理优化     KV Cache / 量化 / 投机解码 / vLLM / 长上下文
├── 5. 应用与生态   RAG / Agent / 多模态 / Prompt / CoT / 评估
└── 6. 前沿模型     LLaMA 家族 / GPT 家族 / DeepSeek / Qwen（结构+亮点+局限）
```

## 复习策略

1. **先精读 Transformer 原文**（Attention Is All You Need），其他一切由此展开
2. **公式级理解**：RoPE、GQA、RMSNorm、损失函数（CE/DPO/PPO/GRPO）都要能手写核心公式
3. **对比记忆**：DPO vs PPO、GQA vs MHA、标准注意力 vs FlashAttention、RAG vs 长上下文
4. **结合项目**：在 MiniMind 里实际调过的东西（训练曲线、数据、超参）是面试最佳素材
5. **论文分层**：精读（必背细节）/ 略读（记住贡献与结论）/ 泛读（知道名字）

## 本目录规划

```
04-LLM大模型/
├── README.md          # 本文件：知识地图 + 复习策略
├── 高频面试题.md      # 六大块面经清单（带 TODO 打勾）
├── 经典论文清单.md    # 精读/略读分类 + 必背要点
├── 专题笔记.md        # 📋 手推教程迁移清单（25 讲全量索引）
└── 教学/              # 📚 Transformer 从零手推教程（25 个 notebook，已全量迁移）
```

## 验收标准

- [ ] `高频面试题.md` 全部打勾
- [ ] `经典论文清单.md` 标注"精读"的论文 ≥ 10 篇读完
- [ ] 专题笔记 ≥ 6 篇
- [ ] 能 5 分钟讲清 MiniMind 项目：做什么、怎么训练、效果、瓶颈、改进
- [x] Transformer 手推系列 25 讲全量迁移到 `教学/`（覆盖架构/训练/推理/RL 全链路）

**使用方式**：直接在 `04-LLM大模型/教学/` 用 Jupyter 打开对应 notebook 复习；完整索引与面试考点对照见 `专题笔记.md`。