# -*- coding: utf-8 -*-
"""生成 10-SAM与视觉基础模型.ipynb（深度故事版）。

叙事链：分割的「封闭世界」困局（痛点：每个数据集训一个模型/标注贵）→
promptable 思想 → SAM 三组件（眼睛/耳朵/手）→ 数据引擎飞轮 → MAE 预训练 →
SAM2（照片→视频）→ 生态组合拳（Grounding DINO+CLIP）。篇末知识链指向 04/05/08/09。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_cv_common import md, code, build, save


def nb10():
    cells = []

    # ---------------- 开场：知识链定位 ----------------
    cells.append(md(r"""# ✂️ 10 · SAM 与视觉基础模型（Segment Anything · 故事版）

> 前八篇你学会的是「给每个数据集训一个分割模型」（05 基础 + 09 U-Net 家族）。
> 这一篇回答一个颠覆性问题：**能不能做一个「什么都能分割」的模型，用户给个点/框就出掩码？**
>
> 阅读方式：先理解**困局**（为什么以前做不到），再拆 **SAM 三组件**（眼睛/耳朵/手），
> 然后看**数据引擎**（11 亿掩码怎么来）、**MAE 预训练**（眼睛怎么开）、**SAM2**（照片→视频）。
>
> 知识链定位：这是 04/05/08/09 的**终点站**——检测和分割都被「提示」驱动，
> 而 SAM 把分割做成了零样本对话。它与 LLM 的 CLIP/Grounding DINO 一起构成视觉基础模型生态。

> 🧩 **一句话主线**：分割太贵（标注像素级）→ 能不能让用户「点一下」就分？
> SAM 用**三组件**回答：图像编码器（懂图）＋提示编码器（懂你）＋掩码解码器（会画）。"""))

    cells.append(md(r"""## §0 前 SAM 时代：分割的三重困局（为什么"什么都能分"是件大事）

1. **封闭世界**：传统分割模型学的是「训练时的类别集合」——换个数据集/类别就要重训
2. **标注太贵**：像素级标注是框标注的 10 倍成本；一张医学图标注要医生数小时
3. **交互式分割的老办法弱**：GrabCut/传统交互工具靠低级图像线索（颜色/边缘），
   遇到语义歧义（"我要的是车，不是车里的狗"）就无能为力

**转折思想（promptable segmentation）**：
> 分割任务的本质不是「分类每个像素」，而是**「听用户的提示，把用户要的那个东西画出来」**。
> 提示 = 点（前景/背景）/ 框 / 掩码 / 文字 —— 让分割像「对话」一样可交互。
> 这需要模型**见过海量数据 + 无数提示方式**——于是有了 11 亿掩码的预训练（数据引擎，见第 5 幕）。"""))

    cells.append(md(r"""## 第 1 幕 · 架构总览：眼睛、耳朵、手

SAM（2023，Meta）由三个模块组成，各司其职：

| 组件 | 比喻 | 输入 | 输出 |
|---|---|---|---|
| **图像编码器** | 眼睛 | 1024×1024 图 | 64×64=4096 个图像 token（256 维） |
| **提示编码器** | 耳朵 | 点/框/掩码 | 提示 token（稀疏：点/框 → 几个向量） |
| **掩码解码器** | 手 | 图像 token + 提示 token | 掩码 + IoU 分数 |

- 图像编码器是**预训练好的 ViT**（不懂图就进不去——MAE 预训练故事见第 6 幕），**权重冻结不参与提示**；
- 提示编码器与掩码解码器**每次推理都工作**（轻量，几百毫秒）；
- 推理两阶段：**眼睛先看一遍图**（可缓存）→ **耳朵听懂提示** → **手画掩码**（可反复改提示，眼睛不用重看）。

> 面试必答：**为什么图像编码器要解耦？** 因为提示是"轻量交互"，如果每次改提示都要重跑整张图的
> 编码器，交互就卡死了。解耦 = 图像特征算一次、提示随便改。"""))

    cells.append(code(r"""import numpy as np
import torch
import torch.nn as nn

# ================= 手写 ViT Patch Embedding（SAM 的眼睛第一步） + torch 对照 =================
# 图像 (N,C,H,W) -> 切成 (H/P)*(W/P) 个 patch -> 每个 patch 展平成向量 -> 线性投影到 D 维
# 关键洞察：这就是一个 kernel=stride=P 的卷积！（卷积天然做了「局部窗口 + 线性投影」）
def patch_embed_np(x, P):
    N, C, H, W = x.shape
    G = H // P
    # ① 切 patch：内存顺序 (N,C,H,W) -> reshape (N,C,G,P,G,P)（H=G·P 拆成 G、P 两维）
    #    再 transpose 成 (N,G,G,C,P,P)：patch 网格 (G,G)、通道 C、块内 (P,P)
    xp = x.reshape(N, C, G, P, G, P).transpose(0, 2, 4, 1, 3, 5)
    # ② 每个 patch 按「通道→行→列」展平（与卷积核 w[c,u,v] 展平顺序一致）
    flat = xp.reshape(N, G * G, C * P * P)
    return flat

torch.manual_seed(0)
xt = torch.randn(1, 3, 16, 16)
proj = nn.Conv2d(3, 64, kernel_size=4, stride=4)              # patch 4x4、投影到 64 维
with torch.no_grad():
    tok_t = proj(xt).flatten(2).transpose(1, 2)               # (1,16,64)：16 个 token
x = xt.numpy(); P = 4
flat = patch_embed_np(x, P)
W = proj.weight.detach().numpy().reshape(64, 3 * P * P); b = proj.bias.detach().numpy()
tok_np = flat.reshape(16, 3 * P * P) @ W.T + b                 # 每个 patch 线性投影 = Conv2d 权重
err = np.abs(tok_np - tok_t.numpy()[0]).max()
print('token 张量形状: %s（16 个 token，每个 64 维）' % (tok_t.shape,))
print('手写 patch embedding vs torch Conv2d 最大误差: %.2e' % err)
assert err < 1e-5
print('对照通过：patch 切分 + 线性投影 == Conv2d(stride=P)（SAM 里 P=16，token 数 = 64×64×64×64/16² = 4096）')"""))

    cells.append(md(r"""## 第 2 幕 · 眼睛的细节：ViT 与位置编码

图像编码器把 1024×1024 切成 16×16 的 patch → 4096 个 token，每个 token 线性投影 + **位置编码**。

**为什么需要位置编码**：Transformer 没有"顺序"概念（注意力是集合操作），
不告诉它"token 3 在 token 1 的右下"，它就把图当成一袋子碎片。

**正弦位置编码**（Transformer 原创，ViT/SAM 沿用）：
$$\text{PE}(pos, 2i) = \sin\!\Big(pos / 10000^{2i/D}\Big), \qquad \text{PE}(pos, 2i+1) = \cos\!\Big(pos / 10000^{2i/D}\Big)$$
频率从低到高：低维编码"位置在哪"，高维编码"位置有多细"——不同维度分工，且**相对距离信息**被编码进内积里。"""))

    cells.append(code(r"""# ================= 手写正弦位置编码 + 可视化 =================
def pos_embed_sin_1d(pos, D=16):
    # pos: (L,1) 位置索引；D 维编码，偶数维 sin、奇数维 cos
    freqs = 1.0 / (10000 ** (np.arange(0, D, 2) / D))       # 频率从 1 到 1/10000 递减
    emb = np.zeros((len(pos), D))
    emb[:, 0::2] = np.sin(pos * freqs[None, :])
    emb[:, 1::2] = np.cos(pos * freqs[None, :])
    return emb

L = 64
pe = pos_embed_sin_1d(np.arange(L)[:, None], D=16)
dots = np.array([pe[0] @ pe[s] for s in range(L)])           # 位置 0 与位置 s 编码的内积
near = dots[1:4].mean(); far = dots[30:34].mean()            # 附近 vs 远处的平均相似度
print('附近位置(s=1..3)相似度均值 %.3f vs 远处(s=30..33) %.3f' % (near, far))
print('近处明显高于远处 -> 模型能从内积里读出「距离多远」（相对位置信息）')
assert near > far

# ========== 图：64 个位置 × 16 维的编码热图（低频/高频分工可见） ==========
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(7.6, 4.6))
im = ax.imshow(pe.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xlabel('位置 pos'); ax.set_ylabel('编码维度（低维=低频粗位置，高维=高频细位置）')
ax.set_title('正弦位置编码：低维慢变（管"在哪"）、高维快变（管"多细"）', fontsize=12)
plt.colorbar(im, ax=ax, fraction=0.046); plt.tight_layout(); plt.show()"""))

    cells.append(md(r"""## 第 3 幕 · 耳朵的细节：提示编码器

提示分**稀疏**（点/框）与**稠密**（掩码）两类，编码方式完全不同：

| 提示 | 编码方式 | 一句话原理 |
|---|---|---|
| **点** | 位置（正弦编码）+ 类别 token（前景/背景可学习向量）相加 | "这里有个东西"或"这里不是东西" |
| **框** | 左上角点 + 右下角点 各自走一遍点编码，再加一个可学习 **box token** | 框 = 两个角点 + "这是个框"的语义 |
| **掩码** | 卷积下采样到 64×64 → 1×1 卷积提通道 → 与图像 token 相加 | 稠密提示直接"画"进特征 |
| 文字 | SAM 1/2 **不支持**（由 Grounding DINO 补，见第 8 幕） | — |

**为什么点要分前景/背景**：同样"点一下"，说"这是猫"和"这不是猫"含义相反——
可学习的类别 token 让网络把这两种指令区分开。"""))

    cells.append(code(r"""# ================= 点提示编码 + 框提示（手写） =================
def point_encode(pos_norm, fg=1, D=8):
    # pos_norm: (K,2) 归一化坐标 [0,1]；正弦编码 + 前后景可学习 token 相加
    pe = np.stack([pos_embed_sin_1d(np.round(pos_norm[:, i] * 63)[:, None].astype(int), D)[:, 0]
                   for i in range(2)], axis=1)              # 简化：两轴各自 8 维拼成 (K,16)
    fg_token = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]) * (1 if fg else -1)  # 可学习类别 token
    # 最终编码 = 位置编码(投影到 D 维后) + 类别 token（SAM 实际维度 256，这里演示原理用 8 维）
    pos_part = np.tile(np.array([0.6, 0.4, 0.8, 0.2, 0.7, 0.3, 0.9, 0.1]), (len(pos_norm), 1))
    return pos_part * (1 + fg_token * 0.25)                  # 位置信息 × 类别偏置（示意）

# 演示：3 个点（2 前景 1 背景）+ 1 个全局可学习 token -> (4,8) 编码矩阵
pts = np.array([[0.2, 0.2], [0.8, 0.8], [0.5, 0.5]])
fgs = [1, 1, 0]
codes = np.stack([point_encode(pts[i:i + 1], fgs[i])[0] for i in range(3)])
global_token = np.ones((1, 8))                               # 全局 token：给解码器"读题"的起点
enc = np.vstack([codes, global_token])
print('点 prompt 编码后形状:', enc.shape, '（3 个点 + 1 个全局 token）')
print('编码矩阵（含位置信息 + 前后景偏置）:\n', np.round(enc, 3))
assert enc.shape == (4, 8)

# ---- 框提示：左上角 + 右下角 + box token ----
tl = point_encode(pts[0:1], 1)[0]; br = point_encode(pts[1:2], 0)[0]
box_token = np.full(8, 0.9)                                   # 可学习 box token（示意）
box_emb = np.stack([tl, br, box_token])
print('框提示编码: 左上角', np.round(tl, 2), ' 右下角', np.round(br, 2), ' box token', np.round(box_token, 2))
assert box_emb.shape == (3, 8)
print('SAM 解码器拿到 prompt token 后，与图像 token 做交叉注意力 → 见下一节')"""))

    cells.append(code(r"""# ================= 掩码提示（稠密）：卷积降采样到 64x64（手写） =================
def mask_downsample(mask, out_hw=64, ch=256, seed=0):
    # mask: (1,1,1024,1024) 二值；平均池化到 out_hw×out_hw（每块均值 = 提示"覆盖比例"）
    #        再 1x1 卷积升到 ch 通道（SAM 实际用 4 个 stride2 卷积，这里演示等价形状）
    N, C, H, W = mask.shape
    s = H // out_hw
    pooled = mask.reshape(N, C, out_hw, s, out_hw, s).mean(axis=(3, 5))   # (1,1,64,64)
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.05, size=(ch, C, 1, 1))
    conv = np.einsum('nchw,fc->nfhw', pooled, w1[:, :, 0, 0])            # (1,256,64,64)
    return pooled, conv

mask1024 = (np.arange(1024)[:, None] < 600) & (np.arange(1024)[None, :] < 800)  # 假掩码：一块矩形
pooled, conv = mask_downsample(mask1024[None, None].astype(float))
print('掩码提示: 1024x1024 -> 平均池化 %s -> 1x1 卷积 %s（与图像 token 同尺寸可相加）' % (pooled.shape, conv.shape))
assert pooled.shape == (1, 1, 64, 64) and conv.shape == (1, 256, 64, 64)
print('要点：稠密提示以"掩码特征"形式直接参与解码，比点/框信息量更大')"""))

    cells.append(md(r"""## 第 4 幕 · 手的细节：掩码解码器

掩码解码器 = **双向 Transformer + 两输出头**（结构小，只有 ~4M 参数）：
- **提示 token** 作为 Query，**图像 token** 作为 Key/Value → 交叉注意力：
  $$\text{Attn}(Q, K, V) = \text{softmax}\!\Big(\frac{QK^T}{\sqrt{d}}\Big) V$$
  解码器被提示"牵着"去图上找对应区域（下文手写交叉注意力 + torch 对照）
- 输出端：**掩码头**（token 与图像特征逐 token 解码出掩码）+ **IoU 头**（预测掩码质量分数）
- **为什么一次给 3 个掩码**：一个提示可能对应多个合理物体（"这个点"可能是整只猫、猫头、猫耳朵）——
  SAM 输出 3 个歧义候选，用 IoU 头挑最优（第 5 幕演示）"""))

    cells.append(code(r"""# ================= 手写交叉注意力（解码器核心）+ torch 对照 =================
def cross_attn_np(q, k, v):
    # q: (T,D) 提示 token；k,v: (N,D) 图像 token；输出 (T,D)
    d = q.shape[1]
    scores = q @ k.T / np.sqrt(d)              # 缩放点积：除以 √d 防止维度大时 softmax 饱和
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p = p / p.sum(axis=1, keepdims=True)       # softmax：每个提示 token 关注图像的位置权重
    return p @ v, p

rng = np.random.default_rng(0)
T, N, D = 4, 16, 8
q = rng.normal(size=(T, D)); k = rng.normal(size=(N, D)); v = rng.normal(size=(N, D))
out_np, attn_np = cross_attn_np(q, k, v)

import torch.nn.functional as F
q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
k_t = torch.tensor(k, dtype=torch.float32).unsqueeze(0)
v_t = torch.tensor(v, dtype=torch.float32).unsqueeze(0)
out_t = F.scaled_dot_product_attention(q_t, k_t, v_t)     # torch 官方实现（同公式）
err = np.abs(out_np - out_t.numpy()[0]).max()
print('手写 cross-attn vs torch 最大误差: %.2e' % err)
assert err < 1e-5
print('注意力矩阵形状: %s（行 = 提示，列 = 被关注的图像 token，行和为 1）' % (attn_np.shape,))
assert np.allclose(attn_np.sum(axis=1), 1.0)

# ========== 图：注意力热图（提示 token 0 关注了哪些图像位置） ==========
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
axes[0].imshow(attn_np, cmap='hot', aspect='auto')
axes[0].set_title('交叉注意力矩阵（4 提示 × 16 图像 token）', fontsize=12)
axes[0].set_xlabel('图像 token'); axes[0].set_ylabel('提示 token')
axes[1].imshow(attn_np[0].reshape(4, 4), cmap='hot')
axes[1].set_title('提示 0 的注意力（图像 token 重排回网格）', fontsize=12); axes[1].axis('off')
plt.tight_layout(); plt.show()
print('每行注意力都集中到部分图像 token —— 提示告诉了解码器"该看哪里"')"""))

    cells.append(md(r"""## 第 5 幕 · 歧义消解：为什么一次给 3 个掩码

"点一下"是**有歧义**的指令：一个点既可能指整只猫，也可能指猫头、猫耳朵。
如果模型只给一个掩码，猜错了用户还得再补点——交互体验差。

**SAM 的解**：解码器一次输出 **3 个掩码候选** + 每个候选的 **IoU 预测分**，
推理时按分数自动挑最优（也可以把 3 个都返回给用户选）。
- 3 个候选本质是"对提示歧义的不同解读"（从小到大或不同形状）
- IoU 头在训练时学"我画的这个掩码跟真实掩码重合多少" → 推理时用预测分排序"""))

    cells.append(code(r"""# ================= 歧义消解演示：3 候选掩码 + IoU 打分 + 自动挑选 =================
yy, xx = np.mgrid[0:64, 0:64]; c = 32
dist = np.sqrt((xx - c) ** 2 + (yy - c) ** 2)
truth = dist < 18                                   # "真值"：用户想要的实心圆 r=18
cands = [dist < r for r in (14, 18, 22)]            # 3 个候选：实心圆 r=14/18/22（提示"点一下"的歧义解读）

def iou_bin(a, b):
    inter = (a & b).sum(); union = (a | b).sum()
    return inter / max(union, 1)

scores = np.array([iou_bin(cand, truth) for cand in cands])
best = int(np.argmax(scores))
print('3 个候选掩码的 IoU 分数: %s（与真值圆的重合度）' % np.round(scores, 3))
print('自动选择候选 %d（分数最高）→ 这就是推理时 IoU 头的作用（挑最像真的那个）' % best)
assert best == 1 and scores[1] == 1.0

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.9))
for ax, (name, m, s) in zip(axes[:3], [('候选0 r=14', cands[0], scores[0]), ('候选1 r=18', cands[1], scores[1]), ('候选2 r=22', cands[2], scores[2])]):
    ax.imshow(m, cmap='gray'); ax.set_title('%s\nIoU=%.3f' % (name, s)); ax.axis('off')
axes[3].imshow(truth, cmap='gray'); axes[3].set_title('真值（r=18 圆）'); axes[3].axis('off')
plt.tight_layout(); plt.show()
print('歧义消解逻辑：一个提示 → 3 个解读 → IoU 头打分 → 自动/交给用户挑最优')"""))

    cells.append(md(r"""## 第 6 幕 · 数据引擎：11 亿掩码怎么来的（飞轮故事）

SAM 要"什么都能分割"，前提是**见过海量掩码**——但全手工标注 11 亿掩码不可能。
Meta 设计了**数据引擎三阶段飞轮**：

| 阶段 | 谁在标 | 怎么标 | 产出 |
|---|---|---|---|
| ① 人工辅助 | 专业标注员 | 用 SAM 雏形（框+点）辅助标注 | 120k 图 |
| ② 半自动 | SAM + 标注员修正 | 模型预测 → 人只挑错补点 | 550k 图 |
| ③ 全自动 | SAM 自己 | 网格点提示自动生成 + 阈值过滤高质量掩码 | **1100 万图 · 11 亿掩码** |

> 关键技巧：**模型越强 → 标注越便宜 → 数据越多 → 模型更强**（飞轮）。
> 1100 万图是 LAION 的 11%；11 亿掩码（Mask R-CNN 数据量的 300 倍）→ SA-1B 数据集。

**为什么训练 SAM 用模拟提示**：从真实掩码随机"翻出来"点/框（点一个在物体内、框一个包住物体），
模型学"任何提示都能复原掩码"——这就是 promptable 的监督来源。"""))

    cells.append(code(r"""# ================= MAE 预训练可视化 + 掩码重建基线演示 =================
# MAE（2022）思想：把图像 mask 掉 75% 的 patch，让模型从剩下的 25% 重建整图。
# 为什么 mask 这么狠：遮挡越多，"猜"越需要全局语义（学会物体结构而不是抄邻近像素）。
rng = np.random.default_rng(0)
G = 8                                          # 8x8 个 patch
mask = rng.permutation(G * G) < int(0.75 * G * G)   # 随机 mask 75%
mask = mask.reshape(G, G)
mask_pix = np.kron(mask, np.ones((8, 8))).astype(bool)   # 升采样到像素级（64x64）
img = np.zeros((64, 64))
for i in range(G):
    for j in range(G):
        img[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = 0.2 + 0.6 * (i / G) + 0.4 * (j / G)  # 平滑渐变图
visible = (G * G - mask.sum())
print('可见 patch 数: %d / %d；MAE 解码器要从这 %d 个 patch 重建整图' % (visible, G * G, visible))

# ---- 重建基线：用可见 patch 的全局均值填充被 mask 的位置（比随机填充好多少） ----
vis_mean = img[~mask_pix].mean()
rec_mean = img.copy(); rec_mean[mask_pix] = vis_mean
rec_rand = img.copy(); rec_rand[mask_pix] = rng.uniform(img.min(), img.max(), size=mask_pix.sum())
mse_mean = ((rec_mean[mask_pix] - img[mask_pix]) ** 2).mean()
mse_rand = ((rec_rand[mask_pix] - img[mask_pix]) ** 2).mean()
print('重建 MSE：均值填充 %.4f vs 随机填充 %.4f（均值填充靠"图是平滑的"先验）' % (mse_mean, mse_rand))
assert mse_mean < mse_rand

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))
axes[0].imshow(img, cmap='viridis'); axes[0].set_title('原图'); axes[0].axis('off')
axes[1].imshow(np.where(mask_pix, np.nan, img), cmap='viridis'); axes[1].set_title('mask 75% 后的输入'); axes[1].axis('off')
axes[2].imshow(rec_mean, cmap='viridis'); axes[2].set_title('可见 patch 均值重建（基线）'); axes[2].axis('off')
plt.tight_layout(); plt.show()
print('MAE 学的比"均值填充"强得多：它用可见 patch 学出物体结构，把缺失部分"画"出来')"""))

    cells.append(md(r"""## 第 7 幕 · SAM 2：从照片到电影（2024）

**痛点**：SAM 只处理单张图；视频分割（跟着一个物体连续切）需要**时间一致性**。

**方案（SAM 2）**：
- **流式记忆（Memory Bank）**：把前面帧的掩码/特征编码成"记忆 token"，当前帧解码时作为额外提示
  —— "上一帧我框的这只鸟，这一帧还在吗？还在就继续切"
- **可提示修正**：任何一帧点一下/框一下，错误立刻修正并沿时间轴传播
- **SA-V 数据集**：视频掩码标注，配合数据引擎同样三阶段自动标注

**SAM 2 与 SAM 1 的核心差异**：加了一层"记忆"，让掩码在时间轴上**连续**。
> 版本线：SAM → SAM 2 → SAM 2.1（数据更多/微调技巧）——主线没变，数据与记忆在变。"""))

    cells.append(md(r"""## 第 8 幕 · 生态组合拳：Grounding DINO + SAM + CLIP

SAM 不会"说"类别，但组合起来就是完整的开放词汇分割管线：

| 角色 | 模型 | 干什么 |
|---|---|---|
| 指挥官 | **Grounding DINO** | 文字 → 框（开放词汇检测：把"那只猫"变成框） |
| 画师 | **SAM** | 框 → 掩码（像素级轮廓） |
| 评委 | **CLIP** | 掩码裁剪 → 图文匹配打分（验证/筛选） |

**自动标注管线**（一份提示驱动全流程）：
`文字 → Grounding DINO 出框 → SAM 出掩码 → CLIP 打分筛选 → 产出训练数据`
这就是现代数据飞轮的标配——**检测/分割模型互相喂数据**。

> 面试题：**SAM 能直接用于检测吗？** 不能——它没有类别语义，只做"你要的东西"。
> 类别信息来自 Grounding DINO / CLIP 那一侧。"""))

    cells.append(md(r"""## §9 数字敏感度与易错点（背诵）

**数字**
- 图像：1024×1024；patch 16×16 → 4096 个图像 token；token 维度 256
- 掩码解码器：双向 Transformer、~4M 参数；3 个掩码候选 + IoU 头
- 数据：SA-1B = 1100 万图 · 11 亿掩码（Mask R-CNN 数据的 ~300 倍）
- 预训练：MAE mask 75%；SAM 2：记忆 token + SA-V 视频数据集
- 生态：Grounding DINO（文字→框）、CLIP（图文匹配）

**易错点**
1. SAM 是**提示驱动**的掩码分割，**没有类别**——"类别无关"
2. 图像编码器**权重冻结**（推理时可复用），提示编码器/解码器每次工作
3. 3 个掩码是**歧义消解**，不是 3 个类别
4. MAE mask 75% 是为了逼出语义；mask 太少退化成"抄邻近像素"
5. SAM 小目标/同物体被切碎/边缘粘连是已知局限
6. SAM 不支持文字提示（别背错）"""))

    cells.append(md(r"""## §10 面试速答（30 秒背诵版 · 按故事线）

- **SAM 是什么**：提示驱动的分割基础模型，点/框/掩码 → 掩码，类别无关
- **为什么强**：11 亿掩码预训练（数据引擎）+ 提示式设计（零样本泛化）
- **三组件**：图像编码器（ViT，冻结）/提示编码器（点框掩码）/掩码解码器（双向 Transformer + 掩码/IoU 头）
- **为什么 3 个掩码**：提示有歧义（整猫/猫头/猫耳），IoU 头打分自动挑
- **MAE 与 SAM 关系**：MAE 预训练图像编码器（先学图像结构，再学分割）
- **数据引擎是什么**：三阶段飞轮（人工辅助→半自动→全自动），模型越强数据越便宜
- **SAM 2 改进**：流式记忆 → 视频分割 + 时间一致性
- **SAM 的局限**：无类别/小目标一般/可切碎/无法本（Grounding DINO 补）
- **组合管线**：Grounding DINO（文字→框）+ SAM（框→掩码）+ CLIP（筛选）

## §11 知识链：本篇 → 全章终点

- **向前**：04/08（检测：框是提示的一种）、05/09（分割：U-Net 训专用模型 vs SAM 通用零样本）
- **再向前**：LLM 的 CLIP/多模态 —— SAM 与它们同属"视觉基础模型"生态（04 章 Transformer 是底层）
- **自测清单**
  - [ ] 手写 patch embedding 与 Conv2d(stride=P) 对照 < 1e-5
  - [ ] 手写正弦位置编码并说出低/高频分工
  - [ ] 手写 cross-attention 与 torch 对照 < 1e-6
  - [ ] 手写掩码降采样形状（1024→64→256 通道）
  - [ ] 复述数据引擎三阶段与数字（11 亿掩码/1100 万图/75% mask）
  - [ ] 讲清 SAM vs U-Net 的本质差异（提示驱动 vs 训练专用）"""))

    return build(cells)


if __name__ == '__main__':
    save(nb10(), '10-SAM与视觉基础模型.ipynb')
