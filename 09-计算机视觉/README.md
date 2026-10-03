# 🖼️ 09-计算机视觉

> 目标：把「计算机视觉」讲成一条主线——**经典图像处理（00-01）→ 卷积与 CNN 架构（02-03）→ 目标检测（04, 08）→ 图像分割（05, 09）→ 度量学习（06）→ 视觉基础模型（10）→ 视觉面试八股（07）**。
> 面试考察的不是背模型名，而是 **每代算法解决什么问题、手写哪些核心算子、数字怎么记、在什么场景选谁**。

## 复习策略

1. **先建直觉层（00-01）**：图像 = 数字矩阵 → 滤波/直方图/HSV/Otsu → Canny/Harris/SIFT/HOG，理解"特征从哪来"
2. **再上表示层（02-03）**：卷积为什么适合图像（局部+平移等变）→ im2col 加速 → 卷积反向三条梯度（手撕主菜）→ LeNet→ResNet→EfficientNet 演进
3. **任务三件套**：
   - **检测（04 + 08）**：IoU 系损失（IoU/GIoU/DIoU/CIoU）→ NMS → anchor 编解码 → mAP → R-CNN 两阶段 → **YOLO v1→v11 全演进**（损失五项/K-Means anchor/CIoU/TaskAlignedAssigner）
   - **分割（05 + 09）**：语义/实例/全景 → 上采样与转置卷积 → FCN → **U-Net 全家族**（Attention/U-Net++/3+/TransUNet/Swin-UNet）→ DeepLab/Mask R-CNN → Dice/mIoU
   - **度量（06）**：对比损失 → 三元组 → 难样本挖掘 → ArcFace → 检索 Recall@K/mAP
4. **冲到前沿（10）**：SAM 三组件（ViT 编码器 / 提示编码器 / 掩码解码器）、MAE 预训练、SAM2 流式记忆、Grounding DINO+SAM+CLIP 自动标注
5. **手推优先**：灰度化公式、双线性上采样、转置卷积输出尺寸、卷积 dX/dW/db、IoU/GIoU/DIoU/CIoU、YOLOv1 损失五项、anchor 编解码、Dice/mIoU、Attention Gate、cross-attention、patch embedding、ArcFace logit

## 视觉笔记索引（`教学/`，12 篇，全部含推导 + 手写实现 + 对照验证 + 自测清单）

| 篇目 | 核心内容 | 对照验证 |
|------|----------|----------|
| 00-图像基础与图像处理 | 像素/灰度化/直方图均衡/HSV/Otsu/滤波/几何变换（双线性） | 手写 HSV vs matplotlib、直方图均衡单调性、双线性插值 |
| 01-边缘检测与特征提取 | Sobel/Canny/Harris/SIFT/HOG/特征匹配/NMS | 手写 Harris 角点值、HOG 与 skimage 对照 |
| 02-卷积神经网络深入 | 多通道卷积/im2col+GEMM/池化反向/卷积反向三条梯度/TinyCNN 训练 | 手写 vs torch.conv2d、数值梯度检查、测试准确率 |
| 03-经典CNN架构与残差网络 | LeNet/AlexNet/VGG/GoogLeNet/ResNet 残差动机/DenseNet/EfficientNet | 手写 ResBlock 恒等 vs 非恒等收敛曲线 |
| 04-目标检测 | IoU 系损失/NMS/Soft-NMS/anchor/R-CNN→Faster/YOLOv1 编解码/mAP | 手写 NMS/mAP 与 sklearn 式对照、PR 曲线 |
| 05-图像分割 | 三类分割/上采样/FCN/基础 U-Net/Dice/ROI Align/Mask R-CNN | Mini U-Net 合成分割 mIoU > 0.9、手写 vs torch 上采样 |
| 06-度量学习与人脸识别 | 对比损失/三元组/难样本挖掘/ArcFace/检索评估 | 手写 vs torch、测试检索 Recall@1 > 0.85 |
| 07-视觉面试八股 | 90 连问（6 组×15）+ 谱系树 + 手撕默写清单 | 手撕题目全部可执行 |
| **08-YOLO系列目标检测详解** | **故事线**：三次思想跃迁 → v1 网格回归（损失五项/开根号）→ v2 anchor（IoU 距离聚类/BN/**偏移回归编解码**）→ v3 多尺度 FPN（anchor 分配/BCE/**FPN+PANet 融合**）→ v4/v5 训练配方（Mosaic/CIoU/**Mish+CSP/Focus**）→ v6/v7 解耦头/E-ELAN/**RepConv 重参数化融合** → v8 anchor-free（四边距离+DFL+TaskAlignedAssigner）→ v9/v10/v11（**PGI 梯度模拟/PSA/C2PSA**/Mamba）+ 演进时间线，**每个版本均为「原理 + 实现」对偶** | 手写 v1 损失 vs 手算、v1 推理流水线（阈值+NMS）、anchor round-trip、BN 手写 vs torch、Mosaic 拼接、v8 decode+DFL、对齐度量 top-k、RepConv 融合数值对照、Focus 信息无损 |
| **09-U-Net系列图像分割详解** | **故事线**：像素级之难 → FCN 遗产 → U-Net 跳连（concat 形状链）→ Residual/V-Net（Dice 梯度）→ Attention U-Net（AG 门控）→ U-Net++（稠密跳连+深监督加权）→ U-Net 3+（全尺度）→ TransUNet（Transformer 瓶颈形状链）→ Swin-UNet（patch merge/expand）→ nnU-Net（自动决策演示）+ 家族对比表 | AG 手写 vs torch < 1e-5、Dice 数值梯度、U-Net++ 形状断言、patch merge 语义验证、V-Net 3D 推演 |
| **10-SAM与视觉基础模型** | **故事线**：分割三重困局 → promptable 思想 → 三组件（眼睛/耳朵/手）→ ViT 位置编码 → 提示编码（点/框/掩码降采样）→ 交叉注意力 → 歧义消解（3 候选+IoU 打分）→ 数据引擎飞轮（11 亿掩码）→ MAE 预训练 → SAM2 视频 → Grounding DINO+CLIP 生态 | patch embedding 与 cross-attn 手写 vs torch < 1e-5、位置编码近/远相似度、掩码降采样形状、重建基线 MSE |
| **11-3D视觉：点云与NeRF与3DGS** | **故事线**：2D 之壁 → 点云表示与处理（体素化/最远点采样/PointNet 对称聚合）→ 神经渲染 NeRF（体渲染积分/位置编码/分层采样）→ 3DGS（3D 高斯/投影/α 混合）→ 三大范式对比（显式 vs 隐式 vs 混合） | FPS 采样手写 vs 朴素、体渲染数值积分 vs 解析、NeRF 小样本重建 MSE、3DGS 光栅化模拟、投影梯度数值检查 |

> 🧭 **知识链闭环**：04 的 IoU/NMS/anchor 是 08 的零件库；05 的上采样/Dice 是 09 的零件库；
> 08/09 的提示驱动思想在 10 汇合到 SAM；08 的 FPN/深监督与 09 的跳连/深监督同源（多尺度信息流动）。

## 本目录规划

```
09-计算机视觉/
├── README.md          # 本文件
├── 高频面试题.md      # 分类面经清单（带 TODO 打勾）
└── 教学/              # ✅ 12 篇 Jupyter notebook（推导 + 代码 + 对照 + 自测）
    └── images/        # 部分 notebook 落盘的插图
```

## 面试 30 秒速答（背诵骨架）

**经典与表示**
- **灰度化**：Luma `0.299R+0.587G+0.114B`（人眼对绿最敏感）；直方图均衡 = CDF 映射提升对比度
- **Otsu**：类间方差最大化的阈值；**Canny 五步**：高斯模糊 → Sobel → NMS → 双阈值滞后 → 连接
- **Harris**：M 矩阵两个特征值都大 → 角点；**SIFT**：DoG 尺度空间 + 128 维描述子，尺度/旋转不变
- **卷积为什么适合图像**：局部感受野 + 权重共享（平移等变）→ 参数远少于全连接
- **卷积反向**：`dW = 输入窗口 ⊗ dout`；`dX = 翻转核 ⊗ dout`；`db = dout 求和`
- **im2col**：把卷积变成 GEMM（行=窗口、列=通道×核）→ GPU 矩阵库加速
- **ResNet 为什么有效**：恒等捷径保梯度（深层不退化）+ 残差学习"差值"更容易
- **感受野**：`r_new = r + (k-1)·stride_累积`；两层 3×3 = 一层 5×5 感受野、参数省 28%

**检测 / 分割 / 度量**
- **YOLOv1**：S=7 网格回归、w/h 开根号（相对误差均匀化）、λ_coord=5、λ_noobj=0.5
- **演进主线**：网格回归 → anchor → 多尺度 FPN(v3) → 配方(Mosaic/CIoU/PANet, v4/5) → 解耦头(v6) → anchor-free+TaskAlignedAssigner(v8) → PGI(v9) → PSA(v10) → C2PSA(v11)
- **CIoU**：IoU − 中心距惩罚 ρ²/c² − 宽高比惩罚 αv（分离时也有梯度）；**NMS/Soft-NMS**：降序去重 / 衰减分数
- **mAP@[0.5:0.95]**：IoU 0.5~0.95 每 0.05 算 AP 再平均（定位要求高）
- **U-Net 为什么强**：编码-解码 + 跳连保留高分辨率细节（医学小样本）；U-Net 家族：门控 → ++/3+ → 换骨干 → nnU-Net
- **Dice 为什么不平衡友好**：集合级度量不被背景像素淹没；与 CE 组合用
- **ROI Align vs ROI Pooling**：浮点双线性采样 vs 量化取整（亚像素精度）
- **ArcFace**：`logit = s·cos(θ+m)`，角度边际使嵌入更紧凑；难样本挖掘加速训练
- **SAM**：ViT 编码器 + 提示编码器 + 掩码解码器，零样本、类别无关；MAE 掩码 75% 逼出语义

## 验收标准

- [x] `高频面试题.md` 覆盖 图像基础 / CNN / 检测 / 分割 / 度量 / 基础模型 / 手撕默写
- [x] 视觉笔记 12 篇，全部含推导 + 手写实现 + 对照验证 + 自测清单（08/09/10/11 为故事版深讲）
- [x] 能手推：灰度化与均衡化、双线性上采样、卷积 dX/dW/db、IoU 系损失、NMS、YOLOv1 损失五项、anchor 编解码、Dice/mIoU、三元组、ArcFace
- [x] 能口算：卷积输出尺寸、感受野递推、416 图 stride 32 → 13×13 网格（255 通道）
- [x] 全部 notebook 代码经批量执行验证（numpy/sklearn/torch 对照断言通过）

> 💡 手推优先级建议：双线性上采样 → 卷积输出尺寸 → 卷积反向三条梯度 → IoU/GIoU/DIoU/CIoU → YOLOv1 损失五项 → anchor 编解码 → Dice/mIoU → 三元组 → ArcFace logit → NMS。