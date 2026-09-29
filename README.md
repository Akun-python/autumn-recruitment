# 🎯 Autumn-Recruit Algo Notes（秋招算法学习仓库）

> **一个可执行的算法 / 机器学习 / 大模型 / 强化学习 / 计算机视觉 / NLP / 时间序列面试备战体系** —— 11 大模块 · **114 篇中文教学 Notebook** ·
> 核心算法**全部手写实现**并附 sklearn / torch 数值对照，从公式推导到面试八股一站式闭环。

<p align="center">
  <img src="assets/badges/license.svg" alt="license MIT"/>
  <img src="assets/badges/python.svg" alt="python 3.9+"/>
  <img src="assets/badges/modules.svg" alt="11 modules"/>
  <img src="assets/badges/notebooks.svg" alt="114 notebooks"/>
  <img src="assets/badges/numpy.svg" alt="from scratch"/>
  <img src="assets/badges/zh.svg" alt="中文教学"/>
  <img src="assets/badges/verified.svg" alt="nbclient 114/114"/>
</p>

---

## ✨ 项目特色

- 🧮 **手写优先**：排序 / DP / LR / SVM / PCA / GBDT / CNN / LSTM / DQN / PPO / Transformer 全部从零实现，
  **附 sklearn / torch 数值对照**（梯度一致、输出逐位一致），"看一眼就知道公式怎么变成代码"；
- 📐 **推导 + 配图**：每篇含**变种模型谱系表、逐步推导（面试手推模板）、matplotlib 过程图**，
  打开 notebook 即可直接看到推导过程与结果图；
- 🎯 **面试向**：高频题单 + 90 连问速答 + **手写三件套默写** + 面试复盘闭环；
- 🧪 **可验证**：全仓 **114 本 notebook 已全部通过 nbclient 全量执行验证**（含 sklearn / torch 对照断言；无需 GPU、无需联网）；
- 🗺️ **进度可追踪**：ROADMAP 分阶段 + 每个模块自测清单，学一章勾一章。

## 📊 数据一览

| 指标 | 数值 |
|---|---|
| 模块 | 11（算法 / ML / DL / LLM / 八股工程 / 复盘 / RL / 优化算法 / 计算机视觉 / NLP / 时间序列 + 路线） |
| 教学 Notebook | **114 篇**（算法 14 · ML 13 · DL 7 · LLM 24 · 八股 8 · RL 8 · 优化 13 · CV 11 · NLP 8 · 时间序列 8） |
| 背诵级手写模板 | 11 份（`01-数据结构与算法/模板代码/`） |
| 推导过程图 | 15 张（`07-强化学习/教学/images/`）+ 16 张（`05-八股与工程/images/`）+ 各章 `教学/images/` |
| 执行验证 | **114 / 114 本 nbclient PASS（全仓）** |
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
| 02-机器学习 | 线性回归/逻辑回归/NB与K-Means/SVM/PCA/GBDT/决策树随机森林/KNN/AdaBoost/神经网络/HMM/CRF/特征工程 | 13 篇推导笔记 | ✅ |
| 03-深度学习 | BP/CNN/优化器/RNN-LSTM/BN/激活初始化/正则化 | 7 篇模型笔记 | ✅ |
| 04-LLM大模型 | Transformer 从零手推 24 讲（架构/分词/训练/推理/RL 对齐） | 24 篇教学 notebook | ✅ |
| 05-八股与工程 | Python/OS/网络/数据库/Redis/分布式/工程工具/高频手撕 | 8 篇八股 notebook | ✅ |
| 06-面试复盘 | 复盘方法论 + 每场面试记录模板 | — | ✅ |
| 07-强化学习 | MDP/DP/MC-TD/Q学习/DQN/策略梯度/PPO-GRPO/面试八股 | 8 篇教学 notebook（含 15 张推导图） | ✅ |
| 08-优化算法 | SGD/动量/Adam/AdamW/学习率调度/二阶优化/LLM 训练（FP16·BF16·ZeRO）+ 经典优化（LP/IP/GA/SA/TS/PSO/ACO/KKT） | 13 篇优化笔记（推导+对照+自测） | ✅ |
| 09-计算机视觉 | 图像基础/边缘特征/CNN/经典架构/目标检测（R-CNN·**YOLO v1→v11 全系列**）/分割（FCN·**U-Net 全家族**·DeepLab）/度量学习/**SAM 基础模型** | 11 篇视觉笔记（推导+对照+自测） | ✅ |
| 10-自然语言处理 | 文本表示/TF-IDF/n-gram 语言模型/词向量（Word2Vec·GloVe）/RNN·LSTM/Attention·Transformer/预训练（mini-BERT·mini-GPT）/分类与标注 | 8 篇 NLP 笔记（推导+对照+自测） | ✅ |
| 11-时间序列 | 平稳性/ADF/ACF·PACF/ARMA·ARIMA/指数平滑/分解/特征工程·ML/深度学习（LSTM·递归 vs 直接）/评估与异常检测 | 8 篇时序笔记（推导+对照+自测） | ✅ |

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
│   └── 推导笔记/              # ✅ 13 篇手写教学 notebook
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
├── 08-优化算法/               # ⚙️ 优化算法（训练优化器 + 经典优化算法）
│   ├── README.md
│   ├── 高频面试题.md
│   └── 教学/                  # ✅ 13 篇手写教学 notebook（推导+代码+对照+自测）
│
└── 09-计算机视觉/             # 🖼️ 计算机视觉（经典处理 + 检测/分割 + 基础模型）
    ├── README.md
    ├── 高频面试题.md
    └── 教学/                  # ✅ 11 篇手写教学 notebook（推导+代码+对照+自测）
        └── images/            # 🖼️ 部分 notebook 落盘的插图

├── 10-自然语言处理/           # 📝 自然语言处理（表示 → 统计LM → 词向量 → 序列 → 预训练）
│   ├── README.md
│   ├── 高频面试题.md
│   └── 教学/                  # ✅ 8 篇手写教学 notebook（推导+代码+对照+自测）
│       └── images/            # 🖼️ 部分 notebook 落盘的插图
│
└── 11-时间序列/               # 📈 时间序列（平稳性 → ARMA → 平滑/分解 → ML/DL → 异常检测）
    ├── README.md
    ├── 高频面试题.md
    └── 教学/                  # ✅ 8 篇手写教学 notebook（推导+代码+对照+自测）
        └── images/            # 🖼️ 部分 notebook 落盘的插图
```

## 📚 教学 Notebook 索引（手写实现 · 对照验证）

> 中文教学 + 公式推导 + **从零手写实现**（02-04 附 sklearn / torch 数值对照），含可运行测试用例与中间过程输出，附自测清单。

### 01-数据结构与算法/教学/（14 篇，已完成 ✅）

00-排序算法 · 数组与哈希 · 双指针与滑动窗口 · 链表 · 栈与队列 · 二分查找 · 二叉树 · 回溯 ·
动态规划 · 贪心 · 图 · 堆与 Top-K · 位运算 · 字符串

> 另有 `01-数据结构与算法/模板代码/`：**11 份背诵级手写模板**（排序/二分/双指针/链表/栈队列/二叉树/回溯/DP/图/堆/KMP，每份带复杂度注释与自检测试）。

### 02-机器学习/推导笔记/（13 篇，已完成 ✅）

| # | 主题 | 手写核心 |
|---|------|----------|
| 1 | 线性回归 | 正规方程 + 梯度下降 + 岭/Lasso（次梯度）+ 多项式过拟合演示 |
| 2 | 逻辑回归 | sigmoid/softmax + 交叉熵梯度 $X^T(p-y)$ + 多分类手写 |
| 3 | 朴素贝叶斯与K-Means | 高斯/多项式 NB + 拉普拉斯平滑 + K-Means(E/M) + 轮廓系数 |
| 4 | SVM | 对偶推导 + 核矩阵 + 手写 SMO（含收敛中间输出） |
| 5 | PCA | 协方差/特征分解/SVD 三视角 + 方差解释率 + 图像压缩 |
| 6 | GBDT | 手写回归树 + 残差拟合 + 学习率实验 + 特征重要性 |
| 7 | 决策树与随机森林 | 熵/信息增益 + ID3 建树 + Bagging/投票 + OOB/置换重要性 |
| 8 | KNN | 距离度量 + 手写 KNN + KD 树（建树/剪枝查询）+ 维度灾难 |
| 9 | AdaBoost | 前向分步 + 指数损失推导 + 手写树桩 boosting + 误差上界 |
| 10 | 神经网络与反向传播 | 两层 MLP 手写 + BP 链式法则手推 + XOR/月牙分类 |
| 11 | HMM 隐马尔可夫模型 | 前向/后向手写 + Viterbi 手写 + Baum-Welch(EM) 手写 |
| 12 | CRF 与最大熵模型 | 最大熵原理 + 线性链 CRF + 手写梯度上升学特征权重 + 标注偏置 |
| 13 | 特征工程与模型评估 | 编码/缩放/缺失值 + 手写 AUC(秩和) + 手写 KFold + 类别不平衡 |

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

### 08-优化算法/教学/（13 篇，已完成 ✅）

**训练优化器（01-07）**

| # | 主题 | 手写核心 |
|---|------|----------|
| 01 | 梯度下降与迭代优化入门 | 学习率三态（一步到位/发散/蜗牛爬）+ BGD·SGD·Mini-batch 噪声对比 |
| 02 | 动量族与加速方法 | 峡谷地形轨迹 SGD vs Momentum vs NAG + EMA 推导 + 鞍点穿越 |
| 03 | 自适应学习率与 Adam | AdaGrad 停摆 / RMSProp 修复 / Adam 偏差校正 + 手写 vs `torch.optim.Adam` 对照 |
| 04 | Adam 变体与解耦权重衰减 | AdamW 解耦推导 + 手写 vs `torch.optim.AdamW` 对照 + LAMB/Lion/AMSGrad |
| 05 | 学习率调度与训练策略 | warmup/cosine/step/exp 曲线 + 梯度裁剪 + 梯度累积等价性 + EMA |
| 06 | 二阶优化方法 | 牛顿法二次收敛 vs GD 步数对比 + BFGS/L-BFGS + K-FAC 直觉 |
| 07 | 大模型训练优化实践 | FP16/BF16/FP32 数值域 + loss scaling + Adam 状态内存（7B≈84GB）+ ZeRO 分片曲线 |

**经典优化算法（08-13）**

| # | 主题 | 手写核心 |
|---|------|----------|
| 08 | 线性规划与对偶 | 图解法 + 手写单纯形 tableau vs 顶点枚举 + 对偶/影子价格/强对偶验证（scipy 对照） |
| 09 | 整数规划与分支定界 | 取整反例 + 手写 B&B vs 穷举 vs `scipy.milp` + 背包 B&B vs DP |
| 10 | 遗传算法 | 选择/交叉/变异/精英 手写 + 多峰函数全局最优 + TSP（OX 交叉）vs 最近邻 |
| 11 | 模拟退火与禁忌搜索 | Metropolis 准则 + 2-opt TSP + 禁忌表/藐视准则 + SA·TS vs 随机重启对比 |
| 12 | 粒子群与蚁群算法 | PSO 速度更新 + Rastrigin 收敛 + ACO 信息素更新 + TSP vs 最近邻 + 选型总表 |
| 13 | 拉格朗日对偶与 KKT | 乘子法/KKT 四条件 vs SLSQP + 互补松弛 + 坐标下降 Lasso vs sklearn |

> 衔接 `03-深度学习/模型笔记/03-优化器.ipynb`：基础优化器已讲，本章 01-04 加深推导、05-07 直达 LLM 训练实战；
> 08-13 补充经典运筹学/组合优化（精确方法 + 启发式 + 理论工具），面试「算法岗交叉考点」全覆盖。

### 09-计算机视觉/教学/（11 篇，已完成 ✅）

| # | 主题 | 手写核心 |
|---|------|----------|
| 00 | 图像基础与图像处理 | 灰度化/直方图均衡/HSV 往返/Otsu/高斯·中值滤波/几何变换（手写 vs matplotlib） |
| 01 | 边缘检测与特征提取 | Sobel/Canny 滞后/Harris 角点 4/9/SIFT 尺度空间/HOG 可分性/NCC 匹配 |
| 02 | 卷积神经网络深入 | im2col+GEMM 加速 249×/池化反向/卷积 dX·dW·db 反向/感受野/TinyCNN 100% |
| 03 | 经典CNN架构与残差网络 | LeNet/VGG 参数量手算/1×1 卷积/深度残差 8 块 vs 无残差对比/depthwise/SE |
| 04 | 目标检测 | IoU·GIoU 手写/NMS/Soft-NMS/anchor/YOLOv1 编解码/mAP（PR 曲线） |
| 05 | 图像分割 | 双线性上采样/转置卷积/FCN/Mini U-Net mIoU 0.92/Dice 不平衡实验/ROI Align |
| 06 | 度量学习与人脸识别 | 对比/三元组损失/难样本挖掘/ArcFace 边际/检索 Recall@K·mAP |
| 07 | 视觉面试八股 | 90 连问 + 谱系树 + 手撕默写清单 |
| 08 | **YOLO 系列目标检测详解** | v1→v11 全演进/损失五项/聚类 anchor/编解码 round-trip/CIoU/TaskAlignedAssigner |
| 09 | **U-Net 系列图像分割详解** | 家族全表/跳连 concat/Attention Gate（torch <1e-5）/U-Net++ 稠密跳连/V-Net 3D |
| 10 | **SAM 与视觉基础模型** | patch embedding/cross-attention（torch <1e-5）/点提示编码/MAE 掩码/SAM2/自动标注 |

> 主线：**经典处理（00-01）→ CNN（02-03）→ 检测（04, 08）→ 分割（05, 09）→ 度量（06）→ 基础模型（10）**；
> 08/09/10 三篇为现代检测与分割的「全系列 + 前沿」详解（YOLO v1→v11、U-Net 全家族、SAM）。

### 10-自然语言处理/教学/（8 篇，已完成 ✅）

| # | 主题 | 手写核心 |
|---|------|----------|
| 00 | 文本表示与 TF-IDF | 词典分词/One-hot/BOW/TF-IDF 手写 vs sklearn、余弦检索 |
| 01 | n-gram 与统计语言模型 | 链式法则/加一·拉普拉斯平滑/PPL 手写（覆盖 vs 未见句、未平滑 inf） |
| 02 | Word2Vec 与 GloVe | CBOW vs Skip-gram/负采样/共现矩阵 + 截断 SVD 低秩分解 |
| 03 | RNN 与 LSTM | RNN 前向+BPTT 数值梯度 <1e-6、LSTM 四门 vs torch <1e-5、CharRNN 生成 |
| 04 | Attention 与 Transformer | 缩放点积注意力 vs torch <1e-5、多头拼接、mini-Transformer 倒序 acc>0.95 |
| 05 | 预训练模型与微调 | mini-BERT MLM（掩码 15%/cloze acc>0.85）、mini-GPT 因果自回归生成 |
| 06 | 文本分类与序列标注 | 朴素贝叶斯 vs sklearn acc>0.9、BiLSTM 词性标注 acc>0.95、指标手写逐位一致 |
| 07 | NLP 面试八股 | 90 连问 + TF-IDF/平滑 PPL/负采样梯度/因果注意力手撕 + 谱系图 |

> 主线：**表示（00-02）→ 序列建模（03-04）→ 预训练（05）→ 任务与评估（06）→ 八股冲刺（07）**；每篇含推导 + 手写 + sklearn/torch 对照 + 自测清单。

### 11-时间序列/教学/（8 篇，已完成 ✅）

| # | 主题 | 手写核心 |
|---|------|----------|
| 00 | 基础与平稳性 | ACF/PACF/差分/简化 ADF/Ljung-Box 手写 + 随机游走 vs 平稳化对比 |
| 01 | ARMA 与 ARIMA | AR 的 OLS 估计/MA 网格搜索/AIC 定阶/ACF·PACF 识别/多步预测 |
| 02 | 指数平滑 | SES/Holt/Holt-Winters 手写 + 网格选参 + 95% 区间 ±1.96σ√h |
| 03 | 时间序列分解 | 中心化移动平均/季节指数/残差白噪声诊断 + STL 对比 |
| 04 | 特征工程与机器学习 | 滞后/滚动统计/时间编码 + Ridge vs 昨值重复 + ExpandingWindow CV |
| 05 | 深度学习时序预测 | LSTM 滑窗 vs 基线 + 递归 vs 直接多步（误差累积对比） |
| 06 | 评估与异常检测 | 指标手写 vs sklearn + z-score/IQR/滚动窗口 + P/R 阈值 |
| 07 | 时间序列面试八股 | 90 连问 + ACF/AR-OLS/SES/ExpandingCV/LSTM 手撕 + 谱系 |

> 主线：**平稳性（00）→ ARMA（01）→ 平滑/分解（02-03）→ 特征 ML（04）→ 深度学习（05）→ 评估/异常（06）→ 八股冲刺（07）**；每篇含推导 + 手写 + sklearn/torch 对照 + 自测清单。

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
- [x] Phase 2：ML 推导笔记（13 篇：LR/SVM/PCA/GBDT/NB/K-Means/树 + KNN/AdaBoost/MLP/HMM/CRF/特征工程）
- [x] Phase 3：DL 模型笔记（7 篇：BP/CNN/优化器/RNN-LSTM/BN/激活/正则）
- [x] Phase 4：RL 教学 notebook（8 篇：MDP/DP/MC-TD/Q学习/DQN/PG-AC/PPO-GRPO/面试八股，含推导与配图）
- [x] Phase 5：优化算法教学 notebook（13 篇：SGD→动量→Adam/AdamW→调度→二阶→LLM 训练 FP16/ZeRO + 经典优化 LP/IP/GA/SA/TS/PSO/ACO/KKT，含 torch/scipy/sklearn 对照）
- [x] Phase 6：计算机视觉教学 notebook（11 篇：图像基础/特征提取/CNN 深入/经典架构/目标检测（R-CNN·**YOLO v1→v11 全系列**）/分割（FCN·**U-Net 全家族**）/度量学习/**SAM 基础模型**，含 torch/sklearn 对照）
- [x] Phase 7：自然语言处理教学 notebook（8 篇：文本表示/TF-IDF/n-gram 语言模型/词向量/RNN·LSTM/Attention·Transformer/预训练（mini-BERT·mini-GPT）/分类与标注/NLP 八股，含 sklearn/torch 对照）
- [x] Phase 8：时间序列教学 notebook（8 篇：平稳性/ARMA·ARIMA/指数平滑/分解/特征工程·ML/LSTM 深度学习/评估与异常检测/时序八股，含 sklearn/torch 对照）
- [ ] Phase 9：面经复盘归档（05/06/07/08/09/10/11 模块内容按自有资料持续补齐）

## 📄 License

[MIT](LICENSE) © 2024–2025 Autumn-Recruit Algo Notes 作者

## ⚠️ 声明

题单与面经整理自公开高频考点，仅供个人学习使用；面试题答案请结合官方文档与论文原文自行核对。