# 🧠 03-深度学习

> 🖼️ 配图：15 篇教学 notebook 均配有 Wikimedia Commons 网络示意图（感知机/CNN/梯度下降/LSTM/Attention/LeNet/SSM/GAN/VAE 等，见 `模型笔记/images/web_*`，标注出处与许可）。

> 目标：网络结构理解 + 训练机制吃透。深度学习面试的考察重点是 **为什么**（为什么用 ReLU、为什么 BN 有效、为什么 Adam 要偏差校正）。

## 复习策略

1. **从反向传播做起**：把 MLP 的 BP 手推一遍，全图就通了
2. **每一层三问**：做什么 / 为什么有效 / 有什么问题（如 BN 在推理时的行为）
3. **经典网络按谱系记**：CNN 演进线（LeNet→AlexNet→VGG→ResNet→...）、注意力线（RNN→LSTM→Attention→Transformer）
4. **公式会背**：激活函数、优化器更新公式、BN 训练/推理公式、LSTM 门控公式
5. **训练经验积累**：学习率、初始化、正则化、数据增强，面试官喜欢问"你踩过哪些坑"

## 模型笔记索引（`模型笔记/`，15 篇，全部含推导 + 可运行代码 + torch 对照 + 自测清单）

### 训练视角（01–07）

| 篇目 | 核心内容 | 对照验证 |
|------|----------|----------|
| 01-多层感知机与反向传播 | 全连接 BP 推导 / softmax+交叉熵灵魂公式 ∂L/∂z=p−onehot / 梯度检查 / 全 0 初始化对称破坏 | **torch.autograd 梯度一致（误差 ~1e-16）** |
| 02-卷积神经网络CNN | 互相关 / im2col / 池化 / 微型 CNN / 模板匹配直觉 / 空洞·深度可分离·ResNet | im2col 与循环版输出一致 |
| 03-优化器 | SGD/Momentum/AdaGrad/RMSProp/Adam 五类手写 + 峡谷轨迹 + Adam 偏差校正 | 训练 loss 对照表（Adam 综合最优） |
| 04-RNN与LSTM | BPTT 手推 / 梯度消失 / LSTM 反向推导 / 传送带类比 | **nn.LSTM 前向一致（误差 ~1e-8）+ 数值梯度检查** |
| 05-批量归一化BN | 白化直觉 / 前向反向 / 训练 vs 推理统计量 / **无 BN 大 lr 发散对照组** | torch BatchNorm1d 一致 + 数值梯度 |
| 06-激活函数与初始化 | ReLU/LeakyReLU/Swish/GELU 对比 / 死亡 ReLU 实验 / Xavier·He 推导 / **全 0 初始化对称演示** | 梯度传播消失/爆炸对照、He 保持方差 O(1) |
| 07-正则化 | L1·L2·Dropout·早停 / **inverted dropout 训练推理期望一致性** / 过拟合 U 形曲线 | torch.nn.Dropout 输出逐位一致 |

### 架构视角（08–15）

| 篇目 | 核心内容 | 对照验证 |
|------|----------|----------|
| 08-神经网络架构全景图 | 谱系三族 / 三视角选型 / **参数量·FLOPs 手算**（Conv·LSTM·MHA）/ 六类 mini 骨架 / O(T²) vs O(T) 量级图 | 手算 vs torch.numel() 逐项一致 |
| 09-CNN经典架构与PyTorch实现 | LeNet→VGG→ResNet→EfficientNet 演进 / **手写 2D 卷积** / 感受野递推 / 1×1·Depthwise / 残差 vs 无残差收敛对比 | 手写卷积 vs nn.Conv2d（~1e-15） |
| 10-RNN与LSTM深入GRU双向 | **torch 手写 RNN/LSTM/GRU cell** / 门控公式与权重排布 / 梯度消失实测 / 双向拼接 / LSTM 学周期 | 手写 vs nn.LSTM/nn.GRU 逐位一致（~1e-7） |
| 11-Attention与Transformer从零实现 | 缩放点积推导（√d）/ **手写多头 vs nn.MultiheadAttention** / 正弦 PE / Encoder 块 / mini Encoder 倒序任务 acc>0.95 | SDPA/MHA 逐位对照 + 训练收敛断言 |
| 12-Mamba与状态空间模型 | 连续 SSM → **ZOH 离散化推导** / **递推≡卷积数值验证** / 选择性演示 / toy Mamba 块 / SSM 学累计和 | 递推 vs 卷积输出一致（~1e-12） |
| 13-GAN生成对抗网络 | minimax 目标 / **最优判别器 D\*=p/(p+p) 推导** / toy GAN 8 高斯环 / 模式坍缩 / **WGAN-GP** 对比 | 生成分布 vs 真实分布散点对比 |
| 14-VAE与扩散模型 | **ELBO 两项推导** / 重参数化 / toy VAE / **DDPM 加噪→预测噪声→反向采样** | VAE/DDPM 采样覆盖环全部 mode |
| 15-深度学习架构面试八股 | 全架构对比表 / **可执行默写自检**（参数量·门控·注意力·ZOH）/ 40 连问 / 手撕清单 10 条 / 踩坑经验 | 默写三连跑通 = 公式过关 |

## 本目录规划

```
03-深度学习/
├── README.md          # 本文件
├── 高频面试题.md      # 分类面经清单（带 TODO 打勾）
└── 模型笔记/          # ✅ 15 篇 Jupyter notebook（推导 + 代码 + torch 对照 + 自测）
                       #    01–07 训练视角：BP/优化器/BN/正则化
                       #    08–15 架构视角：CNN/RNN·LSTM·GRU/Transformer/Mamba/GAN/VAE·扩散/八股
```

## 验收标准

- [x] `高频面试题.md` 覆盖 BP / BN / LSTM / 优化器 / 注意力 / 架构对比高频考点
- [x] 模型笔记 15 篇（BP、BN、LSTM、优化器对比、Transformer、Mamba、GAN、VAE/扩散必含）
- [x] 能手推：BP、BN 前向/反向、Adam 更新公式、LSTM 门控梯度、注意力公式、ZOH 离散化
- [x] 全部 notebook 代码经批量执行验证（torch 对照断言通过）

> 💡 手推优先级建议：MLP BP（四组参数的梯度）→ 优化器更新公式（含偏差校正）→ BN 训练/推理公式 → LSTM 三组门 + 细胞状态反向 → 注意力 softmax(QK^T/√d)V → ZOH 离散化。
> 架构篇学习顺序：08 总览 → 09 CNN → 10 RNN/LSTM/GRU → 11 Transformer → 12 Mamba → 13 GAN → 14 VAE/扩散 → 15 八股总复习。