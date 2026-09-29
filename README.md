# 🎯 Autumn-Recruit Algo Notes（秋招算法学习仓库）

> **一个可执行的算法 / 机器学习 / 大模型 / 强化学习面试备战体系** —— 8 大模块 · **75 篇中文教学 Notebook** ·
> 核心算法**全部手写实现**并附 sklearn / torch 数值对照，从公式推导到面试八股一站式闭环。

<p align="center">
  <img src="assets/badges/license.svg" alt="license MIT"/>
  <img src="assets/badges/python.svg" alt="python 3.9+"/>
  <img src="assets/badges/modules.svg" alt="8 modules"/>
  <img src="assets/badges/notebooks.svg" alt="75 notebooks"/>
  <img src="assets/badges/numpy.svg" alt="from scratch"/>
  <img src="assets/badges/zh.svg" alt="中文教学"/>
  <img src="assets/badges/verified.svg" alt="nbclient 75/75"/>
</p>

---

## ✨ 项目特色

- 🧮 **手写优先**：排序 / DP / LR / SVM / PCA / GBDT / CNN / LSTM / DQN / PPO / Transformer 全部从零实现，
  **附 sklearn / torch 数值对照**（梯度一致、输出逐位一致），"看一眼就知道公式怎么变成代码"；
- 📐 **推导 + 配图**：每篇含**变种模型谱系表、逐步推导（面试手推模板）、matplotlib 过程图**，
  打开 notebook 即可直接看到推导过程与结果图；
- 🎯 **面试向**：高频题单 + 90 连问速答 + **手写三件套默写** + 面试复盘闭环；
- 🧪 **可验证**：全仓 **68 本 notebook 已全部通过 nbclient 全量执行验证**（含 sklearn / torch 对照断言；无需 GPU、无需联网）；
- 🗺️ **进度可追踪**：ROADMAP 分阶段 + 每个模块自测清单，学一章勾一章。

## 📊 数据一览

| 指标 | 数值 |
|---|---|
| 模块 | 8（算法 / ML / DL / LLM / 八股工程 / 复盘 / RL / 优化算法 + 路线） |
| 教学 Notebook | **75 篇**（算法 14 · ML 7 · DL 7 · LLM 24 · 八股 8 · RL 8 · 优化 7） |
| 背诵级手写模板 | 11 份（`01-数据结构与算法/模板代码/`） |
| 推导过程图 | 15 张（`07-强化学习/教学/images/`）+ 16 张（`05-八股与工程/images/`） |
| 执行验证 | **75 / 75 本 nbclient PASS（全仓）** |
| 依赖 | `numpy` / `matplotlib` / `sklearn` / `torch`（02-04 对照验证用；无需 GPU、无需联网） |

## 🚀 快速开始

```bash
git clone <your-repo-url> autumn-recruit-algo
cd autumn-recruit-algo

# 仅需基础科学计算环境，无需 GPU
pip install numpy matplotlib jupyter nbclient

# 打开任意教学 notebook，从上到下逐 cell 运行
jupyter notebook "07-强化学习/教学/00-MDP与贝尔曼方程.ipynb"
```

> 每个 notebook 均自包含（不依赖自定义包、不访问网络、不用 gym 封装；02-04 的 sklearn/torch 仅用于
> 对照验证），环境自写、随机种子固定，可复现全部数值结果。

## 🧩 模块总览

| 模块 | 内容 | 教学 Notebook | 状态 |
|---|---|---|---|
| 01-数据结构与算法 | 排序/哈希/双指针/链表/二分/二叉树/回溯/DP/贪心/图/堆/位运算/字符串 | 14 篇教学 + 11 份模板 | ✅ |
| 02-机器学习 | 线性回归/逻辑回归/NB与K-Means/SVM/PCA/GBDT/决策树随机森林 | 7 篇推导笔记 | ✅ |
| 03-深度学习 | BP/CNN/优化器/RNN-LSTM/BN/激活初始化/正则化 | 7 篇模型笔记 | ✅ |
| 04-LLM大模型 | Transformer 从零手推 24 讲（架构/分词/训练/推理/RL 对齐） | 24 篇教学 notebook | ✅ |
| 05-八股与工程 | Python/OS/网络/数据库/Redis/分布式/工程工具/高频手撕 | 8 篇八股 notebook | ✅ |
| 06-面试复盘 | 复盘方法论 + 每场面试记录模板 | — | ✅ |
| 07-强化学习 | MDP/DP/MC-TD/Q学习/DQN/策略梯度/PPO-GRPO/面试八股 | 8 篇教学 notebook（含 15 张推导图） | ✅ |
| 08-优化算法 | SGD/动量/Adam/AdamW/学习率调度/二阶优化/LLM 训练（FP16·BF16·ZeRO） | 7 篇优化笔记（推导+对照+自测） | ✅ |

## 📁 目录结构

```
autumn-recruit-algo/
├── README.md                  # 本文件：仓库总览
├── ROADMAP.md                 # 学习路线：分阶段时间线 + 产出物
├── LICENSE                    # MIT
├── assets/
│   └── badges/                # 📛 自绘 SVG 徽章（无外部依赖）
│
├── 01-数据结构与算法/          # ✍️ 算法刷题模块
│   ├── README.md              # 刷题策略与顺序
│   ├── 高频题单.md            # 分类高频题（LeetCode 题号）
│   ├── 复杂度速查.md          # 数据结构/排序复杂度速查表
│   ├── 教学/                  # ✅ 14 篇手写教学 notebook
│   └── 模板代码/              # ✅ 11 份背诵级手写模板
│
├── 02-机器学习/               # 📐 ML 推导模块
│   ├── README.md
│   ├── 高频面试题.md
│   └── 推导笔记/              # ✅ 7 篇手写教学 notebook
│
├── 03-深度学习/               # 🧠 DL 模型模块
│   ├── README.md
│   ├── 高频面试题.md
│   └── 模型笔记/              # ✅ 7 篇手写教学 notebook
│
├── 04-LLM大模型/              # 🚀 LLM 重点模块
│   ├── README.md              # 知识地图（六大块）+ 复习策略
│   ├── 高频面试题.md
│   ├── 经典论文清单.md
│   ├── 专题笔记.md            # 24 讲全量索引
│   └── 教学/                  # ✅ 24 篇 Transformer 从零手推 notebook
│
├── 05-八股与工程/             # 🏗️ 基础八股 + 工程能力
│   ├── README.md
│   ├── 速查清单.md
│   └── 教学/                  # ✅ 8 篇八股 notebook
│
├── 06-面试复盘/               # 🔁 复盘闭环
│   ├── README.md
│   └── 复盘模板.md
│
├── 07-强化学习/               # 🤖 强化学习（含 LLM 对齐 RLHF/GRPO）
│   ├── README.md
│   ├── 高频面试题.md
│   └── 教学/                  # ✅ 8 篇手写教学 notebook
│       └── images/            # 🖼️ 15 张推导过程图
│
└── 08-优化算法/               # ⚙️ 优化算法（SGD→动量→Adam→LLM 训练实践）
    ├── README.md
    ├── 高频面试题.md
    └── 教学/                  # ✅ 7 篇手写教学 notebook（推导+代码+对照+自测）
```

## 📚 教学 Notebook 索引（手写实现 · 对照验证）

> 中文教学 + 公式推导 + **从零手写实现**（02-04 附 sklearn / torch 数值对照），含可运行测试用例与中间过程输出，附自测清单。

### 01-数据结构与算法/教学/（14 篇，已完成 ✅）

00-排序算法 · 数组与哈希 · 双指针与滑动窗口 · 链表 · 栈与队列 · 二分查找 · 二叉树 · 回溯 ·
动态规划 · 贪心 · 图 · 堆与 Top-K · 位运算 · 字符串

> 另有 `01-数据结构与算法/模板代码/`：**11 份背诵级手写模板**（排序/二分/双指针/链表/栈队列/二叉树/回溯/DP/图/堆/KMP，每份带复杂度注释与自检测试）。

### 02-机器学习/推导笔记/（7 篇，已完成 ✅）

| # | 主题 | 手写核心 |
|---|------|----------|
| 1 | 线性回归 | 正规方程 + 梯度下降 + 岭/Lasso（次梯度）+ 多项式过拟合演示 |
| 2 | 逻辑回归 | sigmoid/softmax + 交叉熵梯度 $X^T(p-y)$ + 多分类手写 |
| 3 | 朴素贝叶斯与K-Means | 高斯/多项式 NB + 拉普拉斯平滑 + K-Means(E/M) + 轮廓系数 |
| 4 | SVM | 对偶推导 + 核矩阵 + 手写 SMO（含收敛中间输出） |
| 5 | PCA | 协方差/特征分解/SVD 三视角 + 方差解释率 + 图像压缩 |
| 6 | GBDT | 手写回归树 + 残差拟合 + 学习率实验 + 特征重要性 |
| 7 | 决策树与随机森林 | 熵/信息增益 + ID3 建树 + Bagging/投票 + OOB/置换重要性 |

### 03-深度学习/模型笔记/（7 篇，已完成 ✅）

| # | 主题 | 手写核心 |
|---|------|----------|
| 1 | 多层感知机与 BP | 全梯度 BP + 梯度检查 + XOR 手写 |
| 2 | 卷积神经网络 | 卷积/池化前向 + im2col-GEMM 对照 + 尺寸/参数量 |
| 3 | 优化器 | SGD/Momentum/AdaGrad/RMSProp/Adam 手写 + 峡谷/MLP 对比 |
| 4 | RNN 与 LSTM | BPTT + 梯度消失分析 + sin 预测 + LSTM 门控前向 |
| 5 | 批量归一化 BN | 前向/反向三项式 + 梯度检查 + 训练/推理差异 |
| 6 | 激活函数与初始化 | 激活库 + 梯度消失实验 + Xavier/He 方差推导 |
| 7 | 正则化 | L1/L2/Dropout/早停手写 + 过拟合对比实验 |

### 04-LLM大模型/教学/（24 篇 Transformer 从零手推，已完成 ✅）

| 分组 | 讲次 | 主题 |
|---|---|---|
| 架构 | 00-09 | 神经网络基础 · 自注意力 · 多头注意力 · 位置编码与 **RoPE** · 前馈与激活 · **LayerNorm** · 编码器/解码器 · 输出层与损失 · **MQA/GQA** · MiniMind 整体架构 |
| 训练与对齐 | 10-13 | **BPE 分词器** · 预训练循环 · **LoRA 与 SFT** · **DPO 与 RLHF** |
| 推理与前沿 | 14-23 | 量化入门 PTQ/QAT · RAG 与 Agent · MoE · 长上下文 · 解码采样 · **KV 缓存** · PEFT 变种 · **PPO/GRPO 手推** · GPTQ/AWQ · 知识蒸馏 |

> 衔接 `07-强化学习/`：RL 底座学完后，21-PPO与GRPO手推 / 13-DPO与RLHF 即为 LLM 对齐的完整推导闭环。

### 05-八股与工程/（8 篇，已完成 ✅）

Python 语言 · 操作系统 · 计算机网络 · 数据库 · Redis · 分布式 · 工程工具 · 高频手撕代码
（每篇 70+ cells，含**速查表 + 模拟问答 + 手撕代码自测**，nbclient 全量验证）

### 07-强化学习/教学/（8 篇，已完成 ✅ · 44-46 cells/篇）

| # | 主题 | 手写核心 |
|---|------|----------|
| 00 | MDP 与贝尔曼方程 | 网格世界值迭代 + 最优策略可视化 + 贝尔曼算子收缩图 |
| 01 | 动态规划与表格方法 | 策略评估/策略迭代/价值迭代 + 收敛成本对比 + 推导配图 |
| 02 | 蒙特卡洛与时序差分 | MC 首访/每访 vs TD(0) 无模型价值估计 + TD(λ) λ 扫描图 |
| 03 | Q学习与SARSA | 悬崖漫步 off/on-policy 轨迹对比 + 最大化偏差直方图 |
| 04 | 深度Q网络DQN | numpy 迷你 DQN + 回放/目标网络/DDQN/Dueling + 目标网络同步图 |
| 05 | 策略梯度与Actor-Critic | REINFORCE/最简 AC 学习曲线 + GAE + 推导流程图/基线方差图 |
| 06 | PPO与GRPO | clip 数值实验 + RLHF 三阶段 + R1 组内优势 + clip 曲面/KL 权衡图 |
| 07 | 强化学习面试八股 | 90 连问速答 + 老虎机探索实验 + 手写模板 + 算法谱系树 |

> 每篇含：**变种模型谱系表格 + 逐步推导（面试手推模板）+ matplotlib 过程图 2 张**（`教学/images/`，共 15 张；`RL 算法家族谱系树` 见下图示例）。

### 08-优化算法/教学/（7 篇，已完成 ✅）

| # | 主题 | 手写核心 |
|---|------|----------|
| 01 | 梯度下降与迭代优化入门 | 学习率三态（一步到位/发散/蜗牛爬）+ BGD·SGD·Mini-batch 噪声对比 |
| 02 | 动量族与加速方法 | 峡谷地形轨迹 SGD vs Momentum vs NAG + EMA 推导 + 鞍点穿越 |
| 03 | 自适应学习率与 Adam | AdaGrad 停摆 / RMSProp 修复 / Adam 偏差校正 + 手写 vs `torch.optim.Adam` 对照 |
| 04 | Adam 变体与解耦权重衰减 | AdamW 解耦推导 + 手写 vs `torch.optim.AdamW` 对照 + LAMB/Lion/AMSGrad |
| 05 | 学习率调度与训练策略 | warmup/cosine/step/exp 曲线 + 梯度裁剪 + 梯度累积等价性 + EMA |
| 06 | 二阶优化方法 | 牛顿法二次收敛 vs GD 步数对比 + BFGS/L-BFGS + K-FAC 直觉 |
| 07 | 大模型训练优化实践 | FP16/BF16/FP32 数值域 + loss scaling + Adam 状态内存（7B≈84GB）+ ZeRO 分片曲线 |

> 衔接 `03-深度学习/模型笔记/03-优化器.ipynb`：基础优化器已讲，本章 01-04 加深推导、05-07 直达 LLM 训练实战。

## 💼 使用案例（真实运行输出）

### Case 1 · 排序：计数排序的**稳定性**与基数排序（摘自 `01/教学/00-排序算法.ipynb`）

```python
arr = [4, 2, 2, 8, 3, 3, 1, 0, 9]
print('counting_sort(稳定) :', counting_sort(arr, K=10))
print('radix_sort           :', radix_sort([170, 45, 75, 90, 2, 24, 802, 66]))
```

```text
counting_sort(稳定) : [0, 1, 2, 2, 3, 3, 4, 8, 9]
radix_sort           : [2, 24, 45, 66, 75, 90, 170, 802]
```

### Case 2 · RL：Q-learning 一行更新 → 立刻验证收敛（摘自 `07/教学/07-...面试八股` 手写题①）

```python
# 30 秒默写：Q-learning 更新（单步 TD 目标）
Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
```

```text
Q 表(收敛后):
[[ 8.     6.2  ]
 [10.     6.197]
 [ 0.     0.   ]]
最优策略: 状态0 -> 动作0 | 状态1 -> 动作0
→ 正确解: 全程选动作0(走向目标); 面试时默写这一行即可
```

### Case 3 · RL：UCB 一行实现 → 自动锁定最优臂（摘自 `07/教学/07-...面试八股` 手写题④）

```python
# 15 秒默写：UCB1 选择规则（探索-利用平衡）
arm = np.argmax(mean + c * np.sqrt(np.log(t + 1) / (count + 1e-9)))
```

```text
UCB(c=2) 总奖励: 800 / 1000  (理论最优≈900)
各臂采样分布: [0.026 0.039 0.064 0.166 0.705]
→ 最优臂(0.9)被采了最多 → UCB 自动"花最少代价锁定最优"
```

> 更多真实输出见各 notebook：`02/推导笔记/06-GBDT.ipynb`（残差拟合过程）、
> `07/教学/06-PPO与GRPO.ipynb`（clip 四象限数值）、`07/教学/02-...时序差分.ipynb`（TD vs MC 学习曲线）等。

## 🗺️ 学习路线

- **先读 [`ROADMAP.md`](ROADMAP.md)**：分阶段时间线 + 产出物，不要跳阶段；
- **算法模块**：按 `01/README.md` 顺序分类刷题，每类先看模板再做题；
- **ML/DL 模块**：按上文索引逐篇打开 notebook，从上到下运行；每篇末尾自测清单打勾后再进入下一篇；
- **RL 模块**：3 轮路线 —— 打底(00-03) → 进阶(04-06) → 冲刺(07 手推三件套默写)；
- **每场面试后**：用 `06/面试复盘/复盘模板.md` 做 24 小时复盘。

## 🤝 贡献指南

- 内容以**教学正确性优先**：推导步骤完整、代码可复现、随机种子固定；
- 新增 notebook 建议遵循既有结构：观念讲解 → 公式推导（含谱系/变种）→ 手写实现 → 实验/配图 → 自测清单；
- 提交前请确保 notebook 可通过 `nbclient` 全量执行（参考 `07-强化学习` 模块的验证方式）；
- 欢迎通过 Issue/PR 补充高频面试题、修正推导、完善模板。

## ✅ 维护计划

- [x] Phase 1：算法教学 notebook（14 篇）+ 11 份手写模板
- [x] Phase 2：ML 推导笔记（7 篇：LR/SVM/PCA/GBDT/NB/K-Means/树）
- [x] Phase 3：DL 模型笔记（7 篇：BP/CNN/优化器/RNN-LSTM/BN/激活/正则）
- [x] Phase 4：RL 教学 notebook（8 篇：MDP/DP/MC-TD/Q学习/DQN/PG-AC/PPO-GRPO/面试八股，含推导与配图）
- [x] Phase 5：优化算法教学 notebook（7 篇：SGD→动量→Adam/AdamW→调度→二阶→LLM 训练 FP16/ZeRO，含 torch 对照）
- [ ] Phase 6：面经复盘归档（05/06/07/08 模块内容按自有资料持续补齐）

## 📄 License

[MIT](LICENSE) © 2024–2025 Autumn-Recruit Algo Notes 作者

## ⚠️ 声明

题单与面经整理自公开高频考点，仅供个人学习使用；面试题答案请结合官方文档与论文原文自行核对。