# -*- coding: utf-8 -*-
"""生成 09-计算机视觉/教学/07-视觉面试八股.ipynb（nbformat 4）"""
import os

import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell, new_notebook

OUT = '09-计算机视觉/教学'
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
# 07-视觉面试八股
# =====================================================================
md(r"""# 🎯 07 · 视觉面试八股（90 连问 + 手撕默写）

> 目标：把 00-06 全部压缩成**面试弹药**。90 个高频问题分 6 组速览，
> 4 个必考手撕代码现场默写（NMS / 卷积反向 / 双线性+ROI Align / mAP），附模型谱系树与调参清单。

> 📌 **使用方法**：先闭卷自答 → 再对答案；手撕部分**只看题目手写代码**，写完再跑对照。""")

md(r"""## 1. 组 A：图像基础与处理（15 连问）

1. 灰度化公式？（$Y = 0.299R+0.587G+0.114B$）
2. RGB 与 HSV 转换公式？H 的取值范围？
3. 直方图均衡化原理与公式？（CDF 映射）
4. 为什么均衡化能提升对比度？（拉伸累积分布）
5. 卷积与相关（correlation）的区别？（核是否翻转）
6. 卷积核半径与 O(k²) 复杂度的加速思路？（分离核/FFT/积分图）
7. 高斯核为什么可分离？（二维高斯 = 两个一维高斯外积）
8. 均值 vs 中值滤波各擅长什么？（高斯噪声 vs 椒盐噪声）
9. 边缘检测算子：Sobel/Laplacian/Canny 的区别？
10. Canny 四步？（高斯模糊→梯度+方向→非极大值抑制→双阈值滞后）
11. 双线性插值为什么比最近邻平滑？（4 近邻加权）
12. 图像旋转为什么用逆映射？（避免空洞/混叠）
13. Otsu 阈值原理？（最大化类间方差）
14. 图像金字塔/多尺度为什么有用？（尺度不变）
15. 数据增强常用手段与注意事项？（翻转/旋转/颜色抖动；注意不破坏语义）""")

md(r"""## 2. 组 B：特征与经典方法（15 连问）

16. SIFT 为什么尺度不变、旋转不变？（DoG 尺度空间 + 主方向）
17. SIFT 特征怎么匹配？（最近邻 + 比值测试）
18. HOG 特征步骤与思想？（梯度直方图 → 块归一化）
19. HOG 为什么对光照鲁棒？（局部对比度归一化）
20. 角点是什么？Harris 响应的定义？（$R = det(M) - k\\, tr^2(M)$）
21. 模板匹配怎么评估？（NCC 归一化互相关）
22. 特征点检测 vs 描述子 vs 匹配的关系？
23. K-Means 做视觉词袋（BoVW）的流程？
24. PCA 降维在图像特征上的作用？（去相关+压缩）
25. 传统方法 vs 深度学习做视觉任务的本质差异？（手工特征 vs 端到端学习）
26. 仿射变换 vs 透视变换？（6 参数 vs 8 参数）
27. 相机标定要标什么？（内参/外参/畸变）
28. 立体视觉怎么算深度？（视差 → 深度公式 $Z = fB/d$）
29. 光流是什么？LK 光流假设？（亮度恒定+小运动+空间一致）
30. 背景建模（帧差/混合高斯）的适用场景？（固定相机监控）""")

md(r"""## 3. 组 C：CNN 基础与架构（15 连问）

31. 卷积参数量与计算量公式？（$C_{in}k^2C_{out}$；$2\\times$ 乘加）
32. 感受野怎么算？（逐层叠加 $r_{l} = r_{l-1} + (k{-}1)\\prod s$）
33. 1x1 卷积有什么用？（通道混合/升降维/廉价非线性）
34. 池化的作用？（降采样+平移不变性；max 保留纹理、avg 保留背景）
35. 为什么用 ReLU？（缓解梯度消失/稀疏激活/计算快）
36. 梯度消失/爆炸在深层网络的成因与对策？（初始化/BN/残差）
37. BN 放在激活前还是后？训练/推理行为差异？（用 moving mean/var）
38. dropout 为什么能防过拟合？（集成训练子网络）；推理要缩放
39. 卷积反向传播三件套？（dX、dW、db 各自怎么算）
40. im2col 为什么快？（矩阵化 GEMM，显存换速度）
41. 空洞卷积（dilated）有什么用？（不降分辨率扩大感受野）
42. 深度可分离卷积的计算量节省？（≈ 1/9 + 1/C 比例）
43. LeNet/AlexNet/VGG 时代演进的关键点？
44. ResNet 为什么能训很深？（恒等捷径，梯度直通）
45. MobileNet 与 EfficientNet 的设计哲学？（计算-精度权衡的缩放）""")

md(r"""## 4. 组 D：检测与分割（15 连问）

46. IoU/GIoU/DIoU/CIoU 公式与各自的改进点？
47. NMS 流程？Soft-NMS 怎么降权？（$e^{-IoU^2/\\sigma}$）
48. 为什么不用 NMS 剪枝代替所有框？（需要排序+重复抑制，一阶段尾处理）
49. anchor 是什么？怎么设计（scale/ratio/K-Means）？
50. R-CNN → Fast → Faster 各自解决了什么？
51. RPN 的网络结构与训练标签怎么定？（前后景 IoU 阈值）
52. YOLO v1 输出张量结构与损失组成？（S×S×(B×5+C)）
53. YOLO v1 的 x/y 与 w/h 编码方式？
54. FPN 解决什么问题？（多尺度小目标）
55. RetinaNet 为什么用 Focal Loss？（难易样本不平衡）
56. 转置卷积输出尺寸公式？与插值上采样的区别？
57. 语义分割 U-Net 的跳连为什么有效？
58. Dice loss 为什么对类别不平衡友好？
59. mIoU / mAP 怎么算？空类怎么办？（跳过）
60. ROI Align vs ROI Pooling 的采样差异？""")

md(r"""## 5. 组 E：度量学习、生成与其他（15 连问）

61. 对比损失公式与梯度行为？（同类拉近到 0，异类推到 margin 外）
62. Triplet 损失为什么必须难样本挖掘？
63. Center Loss / CosFace / ArcFace 的区别？（中心 vs 余弦 vs 角度边际）
64. 人脸识别 1:1 与 1:N 的差异？（阈值验证 vs Top-K 检索）
65. 检索评估为什么必须屏蔽自身？
66. 人脸识别全流程？（检测→对齐→嵌入→比对）
67. GAN 的博弈目标？（minmax V(D,G)）
68. GAN 训练不稳定的原因与技巧？（模式坍塌/谱范数/梯度惩罚）
69. VAE 的 ELBO 与重参数化技巧？
70. Diffusion 模型的两条路径？（前向加噪 vs 反向去噪）
71. 自监督学习（SimCLR/MoCo）的核心思想？（对比正负样本）
72. 迁移学习什么时候冻结骨干？
73. 知识蒸馏的软标签与温度 T？
74. 什么是"开集识别"/"闭集识别"？度量学习为何适合开放集？
75. 视频理解与图像的差异？（时间建模：3D 卷积/光流/时间 Transformer）""")

md(r"""## 6. 组 F：工程与训练（15 连问）

76. 小数据集怎么做？（迁移学习/数据增强/正则/预训练微调）
77. 训练不收敛排查顺序？（梯度检查→loss 曲线→lr→数据）
78. 学习率策略？（warmup / cosine / 阶梯下降 / OneCycle）
79. 过拟合判断与应对？（train/test gap → 正则/增强/早停）
80. 类别不平衡怎么办？（重采样/加权损失/Focal）
81. BN vs LayerNorm 各适合什么？（CV 用 BN、NLP/小 batch 用 LN）
82. 混合精度训练（FP16）要点？（loss scaling，主权重 FP32）
83. 分布式训练：DataParallel vs DDP？（DDP 梯度同步更优）
84. 推理加速手段？（量化/剪枝/蒸馏/算子融合/TensorRT）
85. 如何评测模型不光看指标？（PR 曲线/坏例分析/混淆矩阵）
86. 目标检测里小目标为什么难？（下采样信息丢失/anchor 不匹配）
87. 如何 debug 一个 NaN loss？（数据→梯度→学习率→数值溢出）
88. 数据标注质量控制？（一致性/审核/主动学习挑难例）
89. 模型部署时输入预处理要一致？（推理前必须同样归一化）
90. 面试反问环节该问什么？（数据规模/评测标准/分工）""")

md(r"""## 7. 手撕 ①：NMS（3 分钟内默写）

```python
def nms(boxes, scores, iou_thr=0.5):
    idx = np.argsort(-scores)
    keep = []
    while len(idx):
        i = idx[0]; keep.append(i)
        rest = idx[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        idx = rest[ious <= iou_thr]
    return keep
```
加分项：**矩阵化 NMS**（向量化 IoU 矩阵，避免 Python 循环）与 Soft-NMS（衰减分数而非删除）。""")

code(r"""# ---------- 手撕 ①：NMS 默写 + 矩阵化加速对照 ----------
import numpy as np
np.random.seed(0)

def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)

def nms_naive(boxes, scores, iou_thr=0.5):
    idx = np.argsort(-scores)
    keep = []
    while len(idx):
        i = idx[0]; keep.append(i)
        rest = idx[1:]
        ious = np.array([iou(boxes[i], boxes[j]) for j in rest])
        idx = rest[ious <= iou_thr]
    return np.array(keep)

def nms_vec(boxes, scores, iou_thr=0.5):
    # 矩阵化：一次算全部 IoU，无需 Python 内层循环
    N = len(boxes)
    order = np.argsort(-scores)
    b = boxes[order]                                     # 降序框
    inter = np.maximum(0, np.minimum(b[:, None, 2:], b[None, :, 2:])
                       - np.maximum(b[:, None, :2], b[None, :, :2])).prod(-1)
    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    iou_mat = inter / (areas[:, None] + areas[None, :] - inter + 1e-9)
    keep = np.ones(N, dtype=bool)                        # 初始全部候选
    for i in range(N):
        if keep[i]:                                      # 只让「被选中」的框去抑制后续
            keep[i + 1:] &= iou_mat[i, i + 1:] <= iou_thr
    return order[keep]

# 随机 30 个重叠框验证两种实现一致
boxes = np.random.uniform(0, 100, (30, 4))
boxes[:, 2:] += boxes[:, :2]                              # 保证 x2>x1, y2>y1
scores = np.random.uniform(0.3, 1.0, 30)
k1, k2 = nms_naive(boxes, scores, 0.5), nms_vec(boxes, scores, 0.5)
print('naive 保留 %d 个，向量化保留 %d 个，结果一致: %s' % (len(k1), len(k2), np.array_equal(k1, k2)))
assert np.array_equal(k1, k2)

# Soft-NMS：不删除，只衰减分数
def soft_nms(boxes, scores, iou_thr=0.5, sigma=0.5):
    s = scores.copy()
    order = np.argsort(-s)
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        rest = order[1:]
        for j in rest:
            v = iou(boxes[i], boxes[j])
            if v >= iou_thr:
                s[j] *= np.exp(-(v ** 2) / sigma)          # 高斯衰减
        order = rest[np.argsort(-s[rest])]
    return np.array(keep), s[keep]

ks, ss = soft_nms(boxes, scores)
print('Soft-NMS 保留 %d 个（分数被衰减而非清零），适合密集人群场景' % len(ks))""")

md(r"""## 8. 手撕 ②：卷积反向传播（dW / dX / db）

对第 $f$ 个卷积核，记 $z^{n,f} = x^n \\star w^f + b^f$（有效区域），则：

- $dW[f] = \\sum_n \\text{corr}(x^n, dout^{n,f})$：输入与输出梯度做相关
- $db[f] = \\sum_n \\sum_{i,j} dout^{n,f}_{i,j}$
- $dX[n] = \\sum_f \\text{full conv}(dout^{n,f} \\text{(pad)}, w^f \\text{翻转})$

必考细节：**dX 是 dout 与翻转核的 full 卷积**；数值梯度校验是标准做法。""")

code(r"""# ---------- 手撕 ②：卷积反向传播默写 + 数值梯度校验 ----------
def conv_forward(x, w, b, stride=1, pad=1):
    # x:(N,C,H,W) w:(F,C,kh,kw) -> out:(N,F,H',W')
    N, C, H, W = x.shape
    F, _, kh, kw = w.shape
    Hp = (H + 2 * pad - kh) // stride + 1
    Wp = (W + 2 * pad - kw) // stride + 1
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((N, F, Hp, Wp))
    for n in range(N):
        for f in range(F):
            for i in range(Hp):
                for j in range(Wp):
                    out[n, f, i, j] = np.sum(
                        xp[n, :, i * stride:i * stride + kh, j * stride:j * stride + kw] * w[f]) + b[f]
    return out

def conv_backward(dout, x, w, b, stride=1, pad=1):
    N, C, H, W = x.shape
    F, _, kh, kw = w.shape
    Hp, Wp = dout.shape[2], dout.shape[3]
    dw = np.zeros_like(w); db = np.zeros_like(b)
    dx = np.zeros_like(x)
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    dxp = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    for n in range(N):
        for f in range(F):
            for i in range(Hp):
                for j in range(Wp):
                    g = dout[n, f, i, j]
                    dw[f] += g * xp[n, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
                    db[f] += g
                    dxp[n, :, i * stride:i * stride + kh, j * stride:j * stride + kw] += g * w[f]
    if pad > 0:
        dx = dxp[:, :, pad:-pad, pad:-pad]
    else:
        dx = dxp
    return dx, dw, db

rng = np.random.default_rng(42)
x = rng.normal(size=(2, 3, 8, 8)).astype(np.float64)
w = rng.normal(size=(4, 3, 3, 3)).astype(np.float64)
b = rng.normal(size=(4,)).astype(np.float64)
out = conv_forward(x, w, b, stride=1, pad=1)
dout = rng.normal(size=out.shape)
dx, dw, db = conv_backward(dout, x, w, b, stride=1, pad=1)

# 数值校验（对 w 的偏导数）
eps = 1e-6
dw_num = np.zeros_like(w)
for f in range(w.shape[0]):
    for c in range(w.shape[1]):
        for p in range(w.shape[2]):
            for q in range(w.shape[3]):
                wp, wm = w.copy(), w.copy()
                wp[f, c, p, q] += eps; wm[f, c, p, q] -= eps
                fp = conv_forward(x, wp, b); fm = conv_forward(x, wm, b)
                dw_num[f, c, p, q] = np.sum((fp - fm) / (2 * eps) * dout)
print('dW 手写 vs 数值梯度 最大误差: %.2e' % np.abs(dw - dw_num).max())
assert np.allclose(dw, dw_num, atol=1e-8)
print('dX 形状一致:', dx.shape == x.shape, '| db 形状一致:', db.shape == b.shape)
print('口述要点：dW = 输入与 dout 相关；db = dout 求和；dX = dout 与翻转核 full 卷积')""")

md(r"""## 9. 手撕 ③：双线性插值 + ROI Align（默写）

双线性：4 邻域加权；ROI Align：bin 内采样点 $(k{+}0.5)/k$ 双线性取特征再池化。
三个考点：**坐标对齐（-0.5 + 0.5/s）、边界 clamp、浮点不量化**。""")

code(r"""# ---------- 手撕 ③：双线性 + ROI Align 默写对照 ----------
def bilinear_sample(feat, ys, xs):
    H, W = feat.shape
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 2); y1 = y0 + 1
    x0 = np.clip(np.floor(xs).astype(int), 0, W - 2); x1 = x0 + 1
    fy = (ys - y0); fx = (xs - x0)
    top = feat[y0, x0] * (1 - fx) + feat[y0, x1] * fx
    bot = feat[y1, x0] * (1 - fx) + feat[y1, x1] * fx
    return top * (1 - fy) + bot * fy

def roi_align(feat, roi, out_size=2, pool_size=2):
    x1, y1, x2, y2 = roi
    out = np.zeros((out_size, out_size))
    for i in range(out_size):
        for j in range(out_size):
            by0 = y1 + i * (y2 - y1) / out_size; by1 = y1 + (i + 1) * (y2 - y1) / out_size
            bx0 = x1 + j * (x2 - x1) / out_size; bx1 = x1 + (j + 1) * (x2 - x1) / out_size
            ys = by0 + (np.arange(pool_size) + 0.5) * (by1 - by0) / pool_size
            xs = bx0 + (np.arange(pool_size) + 0.5) * (bx1 - bx0) / pool_size
            vals = bilinear_sample(feat, np.repeat(ys, pool_size), np.tile(xs, pool_size))
            out[i, j] = vals.max()
    return out

# 亮度阶梯特征图：值 = 行号，语义可手动验证
feat = np.tile(np.arange(10, dtype=float)[:, None], (1, 10))
out = roi_align(feat, (1.5, 1.5, 5.5, 5.5))
print('ROI Align（特征=行号，应随 ROI 纵向位置逐步变大）:\n', np.round(out, 2))
assert out[0, 0] < out[1, 0]                              # 靠下 bin 采样到更大的行号
assert np.allclose(out[0, 0], out[0, 1], atol=1e-10)      # 同一行内列不变（值只由行决定）
assert np.allclose(out[1, 0], out[1, 1], atol=1e-10)
print('口述要点：1) 每 bin 内 2x2 采样点；2) 双线性插值；3) max 池化——全程浮点，无取整')""")

md(r"""## 10. 手撕 ④：mAP 评估（默写）

1. 预测按置信度降序；每个预测找**未匹配** GT 中 IoU 最大的（>阈值 → TP，否则 FP）
2. 累计 TP/FP → Precision/Recall 序列
3. AP = PR 曲线下面积（11 点插值或积分）；mAP = 各类平均

易错：**一个 GT 只匹配一次**；AP 对排序质量敏感。""")

code(r"""# ---------- 手撕 ④：mAP 默写 + 自检 ----------
def iou_b(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter + 1e-9)

def compute_map(dets, gts, iou_thr=0.5):
    dets = sorted(dets, key=lambda d: -d[2])
    tp, fp = np.zeros(len(dets)), np.zeros(len(dets))
    matched = np.zeros(len(gts), dtype=bool)
    for k, (x1, y1, x2, y2, s) in enumerate(dets):
        best_i, best_iou = -1, 0.0
        for i, g in enumerate(gts):
            v = iou_b((x1, y1, x2, y2), g)
            if v > best_iou and not matched[i]:
                best_i, best_iou = i, v
        if best_i >= 0 and best_iou >= iou_thr:
            tp[k] = 1; matched[best_i] = True
        else:
            fp[k] = 1
    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    rec = cum_tp / len(gts)
    prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)
    ap = 0.0
    for t in np.arange(0, 1.0001, 0.1):            # 11 点插值
        m = rec >= t
        ap += max(prec[m]) if m.any() else 0.0
    return ap / 11, rec, prec

# 已知正确结果的手工小例：2 GT + 3 预测（全命中）
gts = [(10, 10, 40, 40), (60, 10, 90, 40)]
dets = [(11, 11, 39, 39, 0.9), (61, 11, 89, 39, 0.8), (13, 13, 37, 37, 0.6)]
ap, rec, prec = compute_map(dets, gts)
print('手工例 AP = %.3f（3 个预测全正确且按序排列 -> AP 高）' % ap)
assert ap > 0.9
print('口述要点：一个 GT 只匹配一次；AP 对"把对的排在前面"的排序质量敏感')""")

md(r"""## 11. 训练与调参速查

**定位问题三步**
1. 先查数据：标签对不对？归一化是否一致？
2. 再查梯度：小 batch 数值梯度 vs 手写反向（见手撕 ②）
3. 最后调参：lr（1e-3 起）→ batch → 优化器（AdamW）→ schedule

**过拟合清单**：数据增强↑ / dropout / weight decay / 早停 / 预训练微调 / 减少模型容量
**欠拟合清单**：加大模型 / 降低正则 / 更长训练 / 提高 lr / 去掉 BN 层位置错误

**小目标与不平衡**
- 多尺度训练 + FPN 特征融合
- 加权 CE / Dice / Focal loss
- 调大输入分辨率（代价：显存）

**数字速记**：ImageNet 224×224；COCO 80 类；Pascal VOC 20 类；
ResNet-50 约 25M 参数；YOLOv5s 约 7M；标准训练 lr=0.1×batch/256（线性缩放）""")

md(r"""## 12. 模型谱系树（绘图）

把 00-06 的模型按「分类 / 检测 / 分割 / 生成」四条主线画成谱系，面试时能顺着时间线讲发展逻辑：
**越新的模型 = 在解决前一代的痛点**。""")

code(r"""# ---------- 谱系树可视化（matplotlib 手绘） ----------
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.axis('off')

branches = [
    ('分类', 0, ['LeNet(98)', 'AlexNet(12)', 'VGG(14)', 'GoogLeNet(14)',
                'ResNet(15)', 'DenseNet(17)', 'EfficientNet(19)']),
    ('检测', 1, ['R-CNN(14)', 'Fast R-CNN(15)', 'Faster R-CNN(15)', 'SSD(16)',
                'YOLO v1(16)', 'YOLO v3(18)', 'YOLO v5(20)', 'YOLO v8(23)']),
    ('分割', 2, ['FCN(15)', 'U-Net(15)', 'DeepLab v3(17)', 'Mask R-CNN(17)',
                'PSPNet(17)', 'SegFormer(21)']),
    ('生成', 3, ['VAE(13)', 'GAN(14)', 'StyleGAN(19)', 'DDPM(20)',
                'Stable Diffusion(22)', 'Sora(24)']),
]
for name, c, items in branches:
    ax.text(0.02, 0.97 - c * 0.24, '▍' + name, fontsize=13, weight='bold',
            transform=ax.transAxes, color='#C44E52')
    for k, it in enumerate(items):
        ax.text(0.14 + 0.15 * k, 0.935 - c * 0.24, it, fontsize=8.5,
                transform=ax.transAxes, rotation=28, ha='left', va='top')
ax.text(0.5, 0.02, '四个谱系 × 时间轴：每一代都在解决上一代的痛点（感受野→深度→尺度→生成质量）',
        ha='center', fontsize=10, transform=ax.transAxes, color='#333333')
plt.tight_layout(); plt.savefig('images/cv07_tree.png', dpi=110, bbox_inches='tight'); plt.show()""")

md(r"""## 13. 30 条高频追问速答（一句话版）

1. **ResNet 为什么不用 BN 放残差前？** 都可以，先 BN 再 ReLU 是常见推荐
2. **为什么 VGG 用 3x3 堆叠？** 两层 3x3 ≈ 5x5 感受野，参数更少
3. **1x1 卷积的本质？** 通道维的全连接 + 空间共享
4. **max vs avg pool 梯度？** max 只回传赢家；avg 均分
5. **BN 推理为什么用统计量？** 测试时不依赖 batch，要确定性的归一化
6. **ReLU 死区怎么办？** LeakyReLU / 初始化调优
7. **NMS 阈值高低的取舍？** 高→多框冗余；低→漏检
8. **RPN 为什么快？** 共享骨干 feature map 一次前向
9. **YOLO v1 一格一物？** 是，重叠物体表现差 → v2 anchor 缓解
10. **FPN 自顶向下怎么融合？** 上采样 + 1x1 逐元素相加
11. **Focal Loss 的 γ？** 默认 2；样本难易调节焦点
12. **U-Net 和 FCN 最大区别？** 跳连（细节保留）
13. **Dice 梯度的坑？** 接近 0/1 时梯度可能不稳定 → 平滑项/组合 CE
14. **mAP@0.5:0.95 为什么苛刻？** 10 个 IoU 阈值平均，重定位精度
15. **Triplet margin 大小影响？** 大→类间距更大但训练更难
16. **ArcFace 的 s 是什么？** 特征缩放因子，默认 64
17. **1:N 库里人很多怎么加速？** 向量检索（近似最近邻/倒排）
18. **对比学习负样本数量？** SimCLR 靠大 batch；MoCo 用队列
19. **GAN 模式坍塌？** 生成器只生成少数模式，判别器太强/多样性损失
20. **DDPM 训练目标？** 预测噪声 ε，MSE
21. **迁移学习什么时候微调全模型？** 数据量大、域差距大时
22. **蒸馏为什么效果好？** 软标签带类间关系信息
23. **warmup 为什么必要？** 避免初期大更新破坏预训练权重/Adam 二阶矩噪声
24. **data augmentation 一致性？** 训练用随机增强，测试必须禁用
25. **混合精度为什么不掉点？** 主权重 FP32 + loss scaling
26. **DDP 与 DP 差异？** DDP 每卡独立 forward + 梯度 AllReduce，通信更少
27. **量化 INT8 掉点如何补偿？** 校准集 + 混合量化（敏感层 FP16）
28. **模型评估只看 accuracy？** 要 PR/AUC/混淆矩阵/失败案例
29. **小目标为什么难检？** 下采样后特征太少，anchor 与 GT 匹配少
30. **面试反问问什么？** 数据集与指标 / 失败案例 / 团队分工与迭代节奏""")

md(r"""## 14. 自测清单

- [ ] 90 连问能闭卷答出 ≥ 70 题（组 A-F 各抽查 2-3 题）
- [ ] 3 分钟内默写 NMS（能升级 Soft-NMS / 矩阵化）
- [ ] 默写卷积反向三件套并能跑通数值校验
- [ ] 默写 ROI Align 并说清与 ROI Pooling 的差异
- [ ] 默写 mAP（11 点插值），说清 GT 只匹配一次
- [ ] 按谱系树讲清 4 条主线的时间线演进逻辑
- [ ] 30 条速答任意抽 10 条能流利作答
- [ ] 调参查问题三步法（数据→梯度→调参）熟记

> 💡 全章验收：`09-计算机视觉/README.md` 复习策略逐条打勾 →
> 「80% 内容能脱离笔记讲出」即达标；高频题见 `09-计算机视觉/高频面试题.md`。""")

nb = new_notebook(cells=cells, metadata=META)
path = os.path.join(OUT, '07-视觉面试八股.ipynb')
nbformat.write(nb, path)
print('written:', path, '| cells:', len(cells))