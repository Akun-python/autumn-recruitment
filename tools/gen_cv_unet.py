# -*- coding: utf-8 -*-
"""生成 09-U-Net系列图像分割详解.ipynb（深度故事版）。

叙事链：分割的像素级之难 → FCN 遗产（痛点：上采样粗糙/细节丢失）→
U-Net 跳连（痛点：一条链不够/全带进来=噪音）→ Residual/V-Net（痛点：注意力浪费）→
Attention U-Net（痛点：还是逐级拼接）→ U-Net++/3+（痛点：CNN 感受野有限）→
TransUNet（痛点：Transformer 贵）→ Swin-UNet（痛点：调参繁琐）→ nnU-Net。
每节 = 痛点 → 方案 → 新痛点；手写实现逐行注释；篇末知识链指向 05/08/10。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_cv_common import md, code, build, save


def nb09():
    cells = []

    # ---------------- 开场：知识链定位 ----------------
    cells.append(md(r"""# 🧬 09 · U-Net 系列图像分割详解（U-Net → U-Net++ → Swin-UNet 全家族 · 故事版）

> 上一篇 `05-图像分割` 给你备好了**基础件**（上采样/Dice/mIoU/基础 U-Net）。
> 这一篇讲 U-Net 家族的 **9 幕进化史**：从一条跳连，到给跳连装门、织网、换骨干、全自动——
> 每一代解决上一个痛点，又埋下下一个伏笔。
>
> 阅读方式：按「**痛点 → 方案 → 新痛点**」的故事线走，先记动机再背结构与通道数。
> 下一篇 `10-SAM` 是它的终极形态：分割从「训练一个专用模型」变成「提示即可分割」。

> 🧩 **一句话主线**：跳连 = 把「编码器记住的细节」直接递给解码器。
> 第一代只拼 1 条（U-Net）→ 第二代给跳连**装门**（Attention U-Net）→ 第三代把跳连**织成网**
> （U-Net++ / U-Net 3+）→ 第四代换骨干（TransUNet / Swin-UNet）→ 工程派（nnU-Net）全自动。"""))

    cells.append(md(r"""## §0 分割为什么难？——「上下文」与「细节」的矛盾（故事起点）

分割要回答每个像素的类别：**边界细节**要精确（像素级），**物体语义**要正确（知道是"什么"）。
这两件事互相拉扯：
- 只看局部 → 边界清楚，但分不清"这是猫还是狗的一部分"
- 只看全局 → 语义对，但边界糊成一团

**编码-解码范式**就是来和解这个矛盾的：
- **编码器**：逐级下采样 → 压缩出**语义**（知道"什么"），代价是丢失细节
- **解码器**：逐级上采样 → 恢复分辨率（画回"在哪"），但细节已经丢了

**全部 U-Net 家族的故事，都围绕一件事：怎么把编码器丢掉的细节找回来。**"""))

    cells.append(md(r"""## 第 1 幕 · 遗产：FCN 与它留下的问题

FCN（2015）第一次把分割做成全卷积：分类网络去掉全连接 → 输出热图 → 上采样回原尺寸。
但它有三个问题：

1. **上采样粗糙**：转置卷积/插值学不回高频细节 → 边界糊
2. **细节丢失**：编码器逐级下采样时小物体/细边被丢掉，且**没有回收机制**
3. **感受野与分辨率两难**：网络越深语义越强，但特征图越小、细节越少

> 🔄 **转折点**：能不能把「编码器每一级还活着的细节」直接送到解码器对应层？
> —— 这就是 U-Net 的答案：**跳连**。"""))

    cells.append(code(r"""import numpy as np

# ================= 微型 U-Net 前向（numpy）：跳连 concat 的形状推演 =================
# 手写三个组件：3x3 卷积（stride=1 pad=1 尺寸不变）、2x2 最大池化、最近邻上采样 ×2
def conv3x3(x, w, b):
    # x: (N,C,H,W)；w: (F,C,3,3)；b: (F,)；逐位置点乘求和（教学版循环，形状直观）
    N, C, H, W = x.shape
    F, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))     # pad=1 保持尺寸
    out = np.zeros((N, F, H, W))
    for n in range(N):
        for f in range(F):
            for i in range(H):
                for j in range(W):
                    out[n, f, i, j] = np.sum(xp[n, :, i:i + kh, j:j + kw] * w[f]) + b[f]
    return out

def maxpool2(x):
    # (N,C,H,W) -> (N,C,H//2,W//2)：2x2 窗口取最大（reshape 成 6 维再对窗口轴 max）
    N, C, H, W = x.shape
    return x.reshape(N, C, H // 2, 2, W // 2, 2).max(axis=(3, 5))

def nearest_up2(x):
    # (N,C,H,W) -> (N,C,2H,2W)：每个像素复制成 2x2（解码器最朴素的上采样）
    N, C, H, W = x.shape
    return np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)

rng = np.random.default_rng(0)
x = rng.normal(size=(2, 1, 16, 16))                       # 2 张 16x16 单通道图
w_e1 = rng.normal(0, 0.1, size=(8, 1, 3, 3)); b_e1 = np.zeros(8)    # 编码器级 1
w_e2 = rng.normal(0, 0.1, size=(16, 8, 3, 3)); b_e2 = np.zeros(16)  # 瓶颈级
w_d1 = rng.normal(0, 0.1, size=(16, 24, 3, 3)); b_d1 = np.zeros(16) # 解码器（ch=8+16=24）
w_out = rng.normal(0, 0.1, size=(4, 16, 3, 3)); b_out = np.zeros(4) # 4 类分割头

def relu(z): return np.maximum(z, 0)

e1 = relu(conv3x3(x, w_e1, b_e1))                           # (2,8,16,16) 编码器第 1 级（留给跳连）
e2 = relu(conv3x3(maxpool2(e1), w_e2, b_e2))                # (2,16,8,8)  瓶颈（语义最浓）
u = nearest_up2(e2)                                         # (2,16,16,16) 上采样回原分辨率
cat = np.concatenate([u, e1], axis=1)                       # ★ 跳连：通道 16+8=24 沿通道维拼接
d1 = relu(conv3x3(cat, w_d1, b_d1))                         # (2,16,16,16) 融合细节与语义
logits = conv3x3(d1, w_out, b_out)                          # (2,4,16,16)  逐像素 4 类 logits
print('形状链: x%s → e1%s → 池化 → e2%s → 上采样%s → 拼接%s → d1%s → logits%s' % (
    x.shape, e1.shape, e2.shape, u.shape, cat.shape, d1.shape, logits.shape))
assert e1.shape == (2, 8, 16, 16) and e2.shape == (2, 16, 8, 8)
assert cat.shape == (2, 24, 16, 16) and logits.shape == (2, 4, 16, 16)
print('跳连要点：解码器通道 = 上采样语义(16) + 同级细节(8) = 24；两条信息流在通道维汇合')

# ---- 感受野对照：为什么「跳连」能补细节 ----
# 编码器 1 级输出（16x16）感受野只有 3x3 卷积叠 2 层 = 5x5（细节多但语义弱）；
# 瓶颈（8x8）感受野 9x9（语义强但细节稀）。跳连 = 让解码器同时拿到「5x5 的细节」与「9x9 的语义」。
print(f'感受野递推：2 层 3x3 = {1 + 2*2} → 3 层 = {1 + 3*2} —— 深层语义强但细节少，浅层反之（跳连各取所长）')"""))

    cells.append(md(r"""## 第 2 幕 · U-Net（2015）：让细节「乘电梯」直达

**方案**：编码-解码 + **跳连 concat**——把每一级编码特征直接拼给同级解码器。
- 深度 4 级（下采样到 1/16），用 concat（解码器自己学怎么融合）
- **医学小样本也能训**：裁剪策略（overlap-tile）+ 镜像 padding（边缘无像素也照样切）+ 数据增强
- 经典配图：U 形结构（编码 U 左臂、解码 U 右臂、跳连是横梁）

**为什么它特别适合医学**：医学数据少（几百张）、目标结构固定（器官形状相近）、边界细节要求高
——跳连让网络不浪费任何像素信息，小数据也能学出锋利边界。"""))

    cells.append(md(r"""## 第 3 幕 · Residual U-Net / V-Net：让网络更深、让损失更稳

**痛点**：U-Net 直接加深会遇到两个问题——① 深层难训（梯度消失）；② 小器官（几个像素）被
背景淹没，**交叉熵损失被背景主导**。

**方案 1（Residual U-Net, 2018）**：编码/解码块换成**残差块**（$y = F(x) + x$）：
恒等捷径保梯度直通 → 可以堆更深（残差思想来自 ResNet，见 03 篇）。

**方案 2（V-Net, 2016）**：
- **3D 卷积**（核 3×3×3、池 2×2×2）——器官是体积，2D 切片割裂空间连续性
- **跳连用加法**（不是 concat）——通道不翻倍、参数量小、残差风格
- **首创 Dice loss**：集合级度量，目标像素再少也有梯度（下面手写 + 数值梯度检查）"""))

    cells.append(code(r"""# ================= Dice loss 手写 + 数值梯度检查 + 不平衡对比 =================
def dice_loss(p, y, smooth=1.0):
    # Dice = 2|P∩Y|/(|P|+|Y|)，损失 = 1 − Dice。p/y 都是 0~1（概率/标签）
    # 为什么抗不平衡：分母同时含 p 和 y 的总量，目标再小也有非零梯度；CE 则被背景像素淹没
    num = 2 * (p * y).sum() + smooth
    den = p.sum() + y.sum() + smooth
    return 1.0 - num / den

def numeric_grad(f, arr, eps=1e-6):
    g = np.zeros_like(arr)
    for i in range(arr.size):
        idx = np.unravel_index(i, arr.shape)
        old = arr[idx]
        arr[idx] = old + eps; fp = f(arr)
        arr[idx] = old - eps; fm = f(arr)
        arr[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
    return g

rng = np.random.default_rng(0)
p = rng.uniform(0.01, 0.9, 8)                 # 8 个像素的预测概率
y = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])      # 标签：只有 1 个正像素（严重不平衡）
g_num = numeric_grad(lambda v: dice_loss(v, y), p.copy())
print('Dice 数值梯度 max |g| = %.3f（正像素处有梯度 → 小器官也能学）' % np.abs(g_num).max())
assert np.isfinite(g_num).all() and np.abs(g_num).max() > 1e-3

# ---- 与 CE 对比：同一组预测，看谁「看见」了正像素 ----
import torch
import torch.nn.functional as F
def ce_loss(p, y):
    # 二元 CE（数值稳定写法）
    return -(y * np.log(p + 1e-7) + (1 - y) * np.log(1 - p + 1e-7)).mean()

p_weak = np.full(8, 0.5)                       # 预测全 0.5（没学会）
p_strong = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # 预测对了正像素
print('损失绝对值：Dice 弱 %.3f → 强 %.3f | CE 弱 %.3f → 强 %.3f' % (
    dice_loss(p_weak, y), dice_loss(p_strong, y), ce_loss(p_weak, y), ce_loss(p_strong, y)))

# ---- 关键对比：模型已基本学会时，「正像素」上的梯度谁更大（谁还能继续推动小器官） ----
p_good = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])   # 背景几乎全对，正像素 0.9
g_ce = -(1.0 / (p_good[0] * len(p_good)))                             # CE 对正像素梯度 = -1/(N·p)
g_dice = numeric_grad(lambda v: dice_loss(v, y), p_good.copy())[0]    # Dice 对正像素梯度（数值）
print('正像素梯度：CE = %+.4f | Dice = %+.4f（Dice 大 %d 倍）' % (
    g_ce, g_dice, abs(g_dice) / abs(g_ce)))
assert abs(g_dice) > abs(g_ce)
print('结论：正像素占比 1/8 时，Dice 对正像素的推动力是 CE 的数倍 → 医学小器官首选（常与 CE 加权组合）')"""))

    cells.append(code(r"""# ================= V-Net（3D U-Net）形状推演 =================
def vnet_shapes(xshape=(1, 1, 64, 64, 64), depth=3):
    # 模拟 V-Net：每级 2 个 3D 卷积（通道翻倍）+ 2x2x2 池化（空间减半）
    c = 16
    rows = [('输入', xshape)]
    d, h, w = xshape[2], xshape[3], xshape[4]
    for lvl in range(depth):
        c *= 2
        rows.append(('级 %d 卷积后' % lvl, (xshape[0], c, d, h, w)))
        d, h, w = d // 2, h // 2, w // 2
        rows.append(('级 %d 池化后' % lvl, (xshape[0], c, d, h, w)))
    for name, s in rows:
        print('  %-14s %s' % (name, s))
    return d

print('V-Net（3D U-Net）形状推演（输入 (1,1,64,64,64)）:')
bottom = vnet_shapes()
assert bottom == 8     # 3 级池化后 64->32->16->8
print('跳连用加法：解码器第 i 级 = 上采样(深层) + 编码器同级特征（通道不变，参数省一半）')"""))

    cells.append(md(r"""## 第 4 幕 · Attention U-Net（2018）：给跳连装个「安检门」

**痛点**：U-Net 把编码器细节**全部**拼进解码器——胰腺这类弱边界器官，周围一大片
无关区域也被带进去，模型分不清重点。

**方案（Attention Gate, AG）**：解码器每一级先用来自**更深层**的门控信号 $g$ 决定
"当前该看编码器特征的哪里"：
$$\alpha = \sigma\Big(W_{att}\cdot\text{ReLU}\big(W_g(g) + W_x(x)\big)\Big), \qquad \text{输出} = x\odot\alpha$$

- $W_g, W_x$ 是 **1×1 卷积**（把 g 和 x 都投影到同一通道数，相加）
- $\alpha\in[0,1]$ 逐空间位置的门控：**背景压 0、目标留 1**
- 参数量只多几个 1×1 卷积，比 U-Net++ 轻得多

> 深层 g 为什么能做门控信号：深层语义已经知道"目标大概在哪"，用它给浅层细节"指路"。"""))

    cells.append(code(r"""import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= Attention Gate 手写（numpy）+ torch 数值对照 =================
def att_gate_np(x, g, wg, wx, wa, ba, Fg=8):
    # x: (N,Cx,H,W) 编码器特征；g: (N,Cg,H,W) 深层门控（语义更强）
    # 1x1 卷积等价于「通道维加权和」（kernel=1 没有空间混合）：einsum (N,C,H,W)x(F,C)->(N,F,H,W)
    g_proj = np.einsum('nchw,fc->nfhw', g, wg[:, :, 0, 0])   # (N,Fg,H,W)
    x_proj = np.einsum('nchw,fc->nfhw', x, wx[:, :, 0, 0])   # (N,Fg,H,W)
    h = np.maximum(g_proj + x_proj, 0)             # ReLU：相加后逐元素非线性
    alpha = 1.0 / (1.0 + np.exp(-(np.sum(h * wa, axis=1, keepdims=True) + ba)))  # σ（wa (1,Fg,1,1) 广播）
    return x * alpha, alpha

rng = np.random.default_rng(0)
N, Cx, Cg, H, W, Fg = 2, 16, 8, 32, 32, 8
x = rng.normal(size=(N, Cx, H, W)); g = rng.normal(size=(N, Cg, H, W))
wg = rng.normal(0, 0.1, size=(Fg, Cg, 1, 1)); wx = rng.normal(0, 0.1, size=(Fg, Cx, 1, 1))
wa = rng.normal(0, 0.1, size=(1, Fg, 1, 1)); ba = np.zeros(1)
out_np, alpha_np = att_gate_np(x, g, wg, wx, wa, ba, Fg)

# torch 对照：同样权重放进 1x1 Conv2d
torch.manual_seed(0)
wg_t = nn.Conv2d(Cg, Fg, 1); wx_t = nn.Conv2d(Cx, Fg, 1); wa_t = nn.Conv2d(Fg, 1, 1)
with torch.no_grad():
    wg_t.weight.copy_(torch.tensor(wg)); wg_t.bias.zero_()
    wx_t.weight.copy_(torch.tensor(wx)); wx_t.bias.zero_()
    wa_t.weight.copy_(torch.tensor(wa)); wa_t.bias.copy_(torch.tensor(ba))
    h = F.relu(wg_t(torch.tensor(g, dtype=torch.float32)) + wx_t(torch.tensor(x, dtype=torch.float32)))
    alpha_t = torch.sigmoid(wa_t(h))
    out_t = torch.tensor(x, dtype=torch.float32) * alpha_t
err = np.abs(out_np - out_t.numpy()).max()
print('AG 手写 vs torch 最大误差: %.2e' % err)
assert err < 1e-5
print('对照通过：α 是逐空间位置 ∈[0,1] 的门控，输出 = 编码器特征 × α')

# ========== 图：门控 α 的效果（合成特征：中间亮斑是"器官"） ==========
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
yy, xx = np.mgrid[0:32, 0:32]
blob = np.exp(-((xx - 18) ** 2 + (yy - 14) ** 2) / 90.0)     # 假 "器官"
bg1 = np.exp(-((xx - 6) ** 2 + (yy - 24) ** 2) / 40.0)       # 干扰背景亮斑
g_sig = blob + 0.35 * bg1                                    # 深层门控信号（强化器官）
alpha_demo = 1.0 / (1.0 + np.exp(-6 * (g_sig - 0.5)))        # sigmoid 化的门控
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
axes[0].imshow(blob, cmap='viridis'); axes[0].set_title('编码器特征 x（器官+杂讯）'); axes[0].axis('off')
axes[1].imshow(g_sig, cmap='viridis'); axes[1].set_title('深层门控信号 g'); axes[1].axis('off')
axes[2].imshow(blob * alpha_demo, cmap='viridis'); axes[2].set_title('输出 = x·α（器官保留）'); axes[2].axis('off')
plt.tight_layout(); plt.show()
print('AG 应用后：与器官重叠区域 α≈1 被保留，背景亮斑 α≈0 被压掉')"""))

    cells.append(md(r"""## 第 5 幕 · U-Net++（2018）：把跳连「织成网」

**痛点**：一条跳连只连接同级；但器官结构有**多尺度**（大肿瘤和小肿瘤差几十倍），
且浅层节点没有直接梯度（只能靠深层反传）。

**方案（嵌套稠密跳连）**：
- 每个节点 $X^{i,j}$（第 $i$ 层第 $j$ 列）吃「**本行左侧所有节点**（上采样到同级）+ **深一层同列节点**」
- 结果：每个解码节点都见过**全部尺度**的信息；浅层也有直接梯度路径
- **深监督**：$X^{0,1..3}$ 每个都接分割头，总损失 = 加权和（浅层头学粗结构、深层头学细边界）
- **推理可剪枝**：训完可以裁掉部分节点（精度几乎不掉、速度提升）

> 故事里的一句话：**U-Net 是一条链，U-Net++ 是一张网。**"""))

    cells.append(code(r"""# ================= U-Net++ 稠密跳连：小规模 torch 形状推演（level=3） =================
# 规则（论文记法）：X^{i,j}（j≥1）的输入 =
#   上采样( X^{i+1,j-1} ) 与 本行左侧 X^{i,0..j-1} 沿通道维拼接，再过一个卷积
class UpBlock(nn.Module):
    def __init__(self, ch_sum):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(ch_sum, ch_sum, 3, padding=1)   # 拼接后通道数 = ch_sum
    def forward(self, deep, *left):
        return self.conv(torch.cat([self.up(deep)] + list(left), dim=1))

torch.manual_seed(0)
enc = [torch.randn(1, c, h, h) for c, h in [(8, 64), (16, 32), (32, 16), (64, 8)]]  # 编码器 4 级
X = {(i, 0): enc[i] for i in range(4)}
# 每个节点的输入通道 = 深层节点上采样后通道 + 本行左侧各节点通道（逐节点手算拼接和）
X[(2, 1)] = UpBlock(64 + 32)(X[(3, 0)], X[(2, 0)])                # up(X^{3,0})64 + X^{2,0}32 = 96
X[(1, 1)] = UpBlock(32 + 16)(X[(2, 0)], X[(1, 0)])                # 32+16 = 48
X[(1, 2)] = UpBlock(96 + 16 + 48)(X[(2, 1)], X[(1, 0)], X[(1, 1)])  # 96+16+48 = 160
X[(0, 1)] = UpBlock(16 + 8)(X[(1, 0)], X[(0, 0)])                 # 16+8 = 24
X[(0, 2)] = UpBlock(48 + 8 + 24)(X[(1, 1)], X[(0, 0)], X[(0, 1)]) # 48+8+24 = 80
X[(0, 3)] = UpBlock(160 + 8 + 24 + 80)(X[(1, 2)], X[(0, 0)], X[(0, 1)], X[(0, 2)])  # 272

print('U-Net++ 节点张量形状（行 = 分辨率层，列 = 节点序号）:')
for i in range(4):
    cols = [X[(i, j)] for j in range(max(1, 4 - i))]
    print('  layer %d: %s' % (i, '   '.join(str(t.shape) for t in cols)))
assert X[(1, 2)].shape == (1, 160, 32, 32) and X[(0, 3)].shape == (1, 272, 64, 64)
print('深监督：X^{0,1}/X^{0,2}/X^{0,3} 各接一个 1x1 分割头 → 多损失加权（下文演示权重模式）')"""))

    cells.append(code(r"""# ================= 深监督的加权方式：3 种权重模式对比 =================
import numpy as np
# 模拟训练过程：浅层头（head1）先收敛、深层头（head3）后收敛（loss 指数下降）
epochs = 30
t = np.linspace(0, 1, epochs)
losses = {
    'head1(浅)': 2.0 * np.exp(-3.0 * t) + 0.2,    # 浅层头：结构粗、收敛快
    'head2(中)': 2.0 * np.exp(-2.5 * t) + 0.2,
    'head3(深)': 2.0 * np.exp(-2.0 * t) + 0.2,   # 深层头：细节细、收敛慢
}
print('深监督总损失 = Σ w_j·L_j（每 epoch 各头独立算损失再加权求和）:')
for name, w in [('均匀', [1/3, 1/3, 1/3]),
                ('递增(越深越重)', [0.2, 0.3, 0.5]),
                ('递减(浅层为主)', [0.5, 0.3, 0.2])]:
    total = sum(w[i] * list(losses.values())[i] for i in range(3))
    print('  %-16s 首 epoch %.3f → 末 epoch %.3f（收敛后总损失）' % (name, total[0], total[-1]))
# 结论：深监督的价值不在权重模式而在「每个头都给梯度」——
# 浅层头让早期训练就有直接梯度，深层头收尾时精修边界。
print('U-Net++ 论文默认：深监督权重随深度递减（浅层头占主导），训练完推理只用最深分支（可剪枝）')"""))

    cells.append(md(r"""## 第 6 幕 · U-Net 3+（2020）：全尺度狂欢

**痛点**：U-Net++ 的节点仍只吃「同一级 + 深一级」，**跨尺度**信息要靠逐级传递——大结构信息传到最浅层时已被稀释。

**方案（全尺度跳连）**：每个解码器节点把**编码器全部 4 级**都拼进来（不同尺度用池化/上采样对齐分辨率），
并加**边界监督**（辅助 loss 学边缘）：
- 大尺度（浅层）特征 → 管边界细节；小尺度（深层）特征 → 管全局语义
- 一个节点同时拿到「4 种尺度」→ 多尺度结构一目了然
- 代价：显存大（每层都拼 4 份）、训练慢

> 故事对比：**U-Net++ 是"网"，U-Net 3+ 是"全连接"**——跳连密度顶到上限。"""))

    cells.append(md(r"""## 第 7 幕 · TransUNet（2021）：给瓶颈换颗「大脑袋」

**痛点**：所有 CNN 跳连方案都受一个硬限制——**CNN 感受野有限**。
一个 3×3 卷积只看到 3×3，器官的**全局结构**（比如整个心脏的形状）要靠堆叠才能勉强"看见"。

**方案**：U 形骨架不变，但**最深一层**不再是纯卷积，而是：
- CNN 编码到 1/16 → 特征拉成 token 序列 → **12 层 Transformer**（全局自注意力：每两个位置都能互相看到）
- Transformer 输出 reshape 回 2D → 逐级上采样 + 跳连（解码器还是 CNN）

**为什么只在最深一层换 Transformer**：全局注意力是 $O(n^2)$，只在 1/16 分辨率算，量级可控；
浅层保持卷积（高效 + 保留细节）。**CNN 给细节、Transformer 给全局**，各取所长。"""))

    cells.append(code(r"""# ================= TransUNet 形状推演（numpy：CNN → Transformer → 上采样） =================
N, C, H, W = 2, 512, 16, 16
feat = np.zeros((N, C, H, W))                 # CNN 编码到 1/16 的深层特征
tokens = feat.reshape(N, H * W, C)            # ① 拉成 token 序列：16x16=256 个 token，每 token 512 维
trans = np.zeros((N, H * W, C))               # ② Transformer 前向（自注意力形状不变：256×512）
back = trans.reshape(N, C, H, W)              # ③ reshape 回 2D 特征图
# ④ 逐级上采样：1/16 → 1/8，并与编码器 1/8 级跳连 concat（解码器仍是 CNN）
up = np.zeros((N, 256, 32, 32)); enc8 = np.zeros((N, 256, 32, 32))
cat = np.concatenate([up, enc8], axis=1)
print('TransUNet 形状链: CNN特征%s → token序列%s → Transformer输出%s → 回2D%s → 上采样+跳连%s' % (
    feat.shape, tokens.shape, trans.shape, back.shape, cat.shape))
assert tokens.shape == (2, 256, 512) and cat.shape == (2, 512, 32, 32)
print('要点：Transformer 的 Q/K/V 都是 256 个 token 之间两两交互 → 全局上下文；CNN 解码器保留边界细节')
print('注意力复杂度：256×256 = 65k 次点积（1/16 分辨率可接受）；若在 1/4 分辨率做 = 4096×4096，贵 256 倍')"""))

    cells.append(md(r"""## 第 8 幕 · Swin-UNet（2022）：全 Transformer 的 U

**痛点**：TransUNet 只换了最深一层——能不能**整个 U 都是 Transformer**？

**方案（Swin-UNet）**：
- **Swin block**：窗口内自注意力（8×8 窗口）+ 窗口平移（跨窗信息流动）——比全局注意力线性省算
- **Patch Merge 下采样**：2×2 邻域拼成 1 个 token（4 通道合成 1 个）→ 分辨率减半、通道翻倍（下文手写）
- **Patch Expand 上采样**：每 token 展开成 2×2（分辨率翻倍、通道减半）
- 跳连仍是 **concat**：把编码器每级输出拼给解码器

> 与 TransUNet 的血缘对比：**TransUNet = CNN 编码 + Transformer 瓶颈；Swin-UNet = 全程 Transformer**。"""))

    cells.append(code(r"""# ================= Patch Merge / Patch Expand 手写（Swin 的下/上采样） =================
def patch_merge(x):
    # (B,C,H,W) -> (B,4C,H/2,W/2)：把 2x2 邻域的 4 个像素沿「通道」拼接（Swin 下采样）
    B, C, H, W = x.shape
    return (x.reshape(B, C, H // 2, 2, W // 2, 2)      # (B,C,gh,ph,gw,pw)
              .transpose(0, 2, 4, 1, 3, 5)            # (B,gh,gw,C,ph,pw)
              .reshape(B, H // 2, W // 2, 4 * C)       # 邻域 4 像素进通道
              .transpose(0, 3, 1, 2))                 # (B,4C,H/2,W/2)

def patch_expand(x):
    # (B,C,H,W) -> (B,C,2H,2W)：每个像素展开成 2x2 块（Swin-UNet 上采样，channel 投影另行线性层）
    B, C, H, W = x.shape
    return np.repeat(np.repeat(x, 2, axis=2), 2, axis=3)

rng = np.random.default_rng(0)
x = rng.normal(size=(2, 8, 16, 16))
m = patch_merge(x); e = patch_expand(x)
print('patch_merge: (2,8,16,16) -> %s（分辨率减半、通道×4）' % (m.shape,))
print('patch_expand: (2,8,16,16) -> %s（分辨率×2，配合线性层降通道）' % (e.shape,))
assert m.shape == (2, 32, 8, 8) and e.shape == (2, 8, 32, 32)
# 验证 merge 语义：取 x[0, 0, 0, 0] 位置的 2x2 邻域应出现在 m 的 4 个通道里
crop = x[0, :, 0:2, 0:2]                              # (8,2,2) 每通道的左上 2x2
merged = m[0, :, 0, 0]                                 # (32,) 左上 token 的 32 通道
assert np.allclose(merged.reshape(8, 4)[:, 0], crop[:, 0, 0])  # 通道序：邻域像素按 (ph*2+pw) 展开
print('merge 语义验证：2x2 邻域像素按通道展开，顺序与 reshape 布局一致 ✓')"""))

    cells.append(md(r"""## 第 9 幕 · nnU-Net（2021）：把调参变成「自动化」

**痛点**：以上每个变体都有超参（patch 多大、batch 几个、重采样怎么选、增强开不开）——
3D 医学分割数据形状千奇百怪（CT 1mm 各向同性 vs MRI 0.5×0.5×5mm），手工调参要命的慢。

**方案（nnU-Net 三步决策，不是网络而是管线）**：
1. **固定规则**：从不失败的默认（resample 到中位间距、全卷积推理滑窗）
2. **经验法则**：按数据 shape 定 patch 大小（3D 常用 128³）与 batch（按显存预算）
3. **实验**：同一数据跑 3 个配置（U-Net 3D 不同深度），选验证集最好的

**为什么它是开箱即用 SOTA**：把「数据形状 → 配置」的映射自动化，任何新数据集都自动得到合理配置。"""))

    cells.append(code(r"""# ================= nnU-Net 的决策演示：按数据形状自动定配置 =================
import numpy as np

def nnunet_plan(shape_3d, gpu_mb=8000, spacing=(1.0, 1.0, 1.0)):
    # shape_3d: (D,H,W)；spacing: 各轴体素间距（mm）
    D, H, W = shape_3d
    # ① 重采样：把非各向同性轴补到中位间距（MRI z 轴 5mm 就先把 z 补成 1mm）
    target = np.median(spacing)
    resampled = [int(s * sp / target) for s, sp in zip(shape_3d, spacing)]
    # ② patch 大小：经验法则「不超过中位 shape、且 ≤ 目标尺寸」
    patch = tuple(min(s, 128) for s in resampled)
    # ③ batch：显存预算 = 每样本占用 × batch（经验系数）
    elems = np.prod(patch) * 1.0          # 每样本体素数（简化，不含通道）
    batch = max(1, int(gpu_mb * 1024 / (elems * 1e3)))   # 粗略按 1KB/体素估
    return target, resampled, patch, batch

for shape, sp in [((128, 128, 96), (1.0, 1.0, 1.0)),   # CT：各向同性
                  ((96, 96, 30), (0.5, 0.5, 5.0))]:    # MRI：z 轴严重非各向同性
    target, resampled, patch, batch = nnunet_plan(shape, spacing=sp)
    print('数据 %s (间距 %s) -> 目标间距 %.2fmm → 重采样后 %s → patch %s, batch %d' %
          (shape, sp, target, resampled, patch, batch))
# 验证：MRI 的 z 轴从 30 层被重采样到 5 倍 ≈ 150（各向同性后网络更好学）
_, _, resampled_mri, _ = nnunet_plan((96, 96, 30), spacing=(0.5, 0.5, 5.0))
assert resampled_mri[2] > 100
print('结论：同一套规则自动适配不同模态——这就是 nnU-Net「零手动调参」的来源')"""))

    cells.append(md(r"""## §10 家族横向对比（背诵级）

| 家族成员 | 跳连方式 | 额外机制 | 解决的问题 | 代价 |
|---|---|---|---|---|
| U-Net | 同级 1 条 concat | — | 细节丢失 | 弱边界带杂讯 |
| Residual U-Net | concat | 块内残差 | 深层难训 | — |
| V-Net | 加法 | 3D + Dice | 空间连续性/不平衡 | 3D 显存 |
| Attention U-Net | concat + α 门控 | AG | 弱边界/杂讯 | 参数略增 |
| U-Net++ | 稠密嵌套 | 深监督+剪枝 | 多尺度/浅层无梯度 | 参数暴涨 |
| U-Net 3+ | 全尺度 | 边界监督 | 跨尺度稀释 | 显存大 |
| TransUNet | concat | 最深一层 Transformer | CNN 感受野有限 | 注意力 O(n²) |
| Swin-UNet | concat | 窗口注意力+patch merge | 全局+效率 | 窗口粒度 |
| nnU-Net | — | 管线自动化 | 调参繁琐 | 工程量大 |

> 选型口诀：**医学小数据 U-Net；弱边界 Attention；多尺度结构 ++/3+；全局结构 TransUNet；3D 体积 V-Net；开箱即用 nnU-Net**。"""))

    cells.append(code(r"""# ========== 图：U-Net 家族「最浅解码块输入通道数」对比（复杂度直观） ==========
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

names = ['U-Net', 'Attention U-Net', 'U-Net++', 'U-Net 3+', 'TransUNet', 'Swin-UNet']
chans = [128, 128, 960, 960, 320, 320]     # 编码器 4 级通道 [64,128,256,512] 下最浅解码块输入
fig, ax = plt.subplots(figsize=(8.4, 4.4))
bars = ax.bar(names, chans, color=['#9ecae1', '#a6d854', '#fdae61', '#d9d9d9', '#bc80bd', '#bc80bd'])
for b, v in zip(bars, chans):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, str(v), ha='center', fontsize=9)
ax.set_ylabel('最浅解码块输入通道数（粗略）')
ax.set_title('跳连策略越密，解码块看到的特征越多（≈复杂度越高）', fontsize=12)
ax.grid(axis='y', alpha=0.3); plt.tight_layout(); plt.show()"""))

    cells.append(md(r"""## §11 数字敏感度与易错点（背诵）

**数字**
- U-Net：4 级编码（1/2→1/16）、跳连 concat、医学小样本（几百张可训）
- V-Net：3D 核 3³、池 2³、跳连加法、Dice（smooth 常用 1）
- U-Net++：节点 X^{i,j}、深监督 3 头、推理可剪枝
- U-Net 3+：全尺度 4 级全拼 + 边界监督
- TransUNet：最深 1/16 拉 256 token × 12 层 Transformer
- Swin-UNet：窗口 8×8、patch merge/expand 2×2
- nnU-Net：3 步（固定规则/经验法则/实验）

**易错点**
1. 转置卷积 ≠ 反卷积（deconv 在信号处理是解卷积）
2. Dice 分母加 smooth 防除零；空类跳过（mIoU 计算时）
3. U-Net 跳连是 concat 不是 add（通道翻倍）；V-Net 才是 add
4. 深监督的「浅层头」输出分辨率更大（越浅越细）——权重一般递减
5. TransUNet 的 Transformer 只在最深一层（全局注意力贵，不能到处用）
6. nnU-Net 是管线不是网络——面试别答成「一种 U-Net」"""))

    cells.append(md(r"""## §12 面试速答（30 秒背诵版 · 按故事线）

- **U-Net 为什么强**：编码-解码 + 跳连 concat 保留高分辨率细节；医学小样本友好
- **跳连为什么 concat 不是 add**：concat 保留两条流的独立性让网络学怎么融合；add 是残差风格（V-Net），省参数
- **Attention Gate 解决什么**：跳连全带进来 → 背景杂讯也进来；α 门控只保留与当前语义相关的区域
- **U-Net++ vs U-Net**：一条链 → 稠密嵌套网 + 深监督（浅层也有梯度），推理可剪枝
- **Dice 为什么对不平衡好**：集合级度量不被背景像素淹没；与 CE 组合用
- **TransUNet vs Swin-UNet**：CNN 编码 + Transformer 最深一层 vs 全 Transformer
- **Swin 为什么快**：窗口内自注意力（8×8）而不是全局 O(n²)，窗口平移交换信息
- **nnU-Net 是什么**：自动配置管线（重采样/patch/batch/增强/损失自适应），不是网络
- **深监督是什么**：浅层也接输出头，多损失加权——梯度不再只从最深层回传

## §13 知识链：本篇 → 哪里去？

- **向后**：`10-SAM` —— U-Net 家族在"训练专用模型"，SAM 直接"提示即分割"（零样本 + 通用）
- **向前**：05 篇的双线性上采样/转置卷积/Dice 是零件库；08 篇的 FPN/深监督与这里同源
- **自测清单**
  - [ ] 手写 U-Net 跳连形状链，说清 concat 后通道数（16+8=24）
  - [ ] 手写 Dice 并做数值梯度检查；讲清为什么抗不平衡（与 CE 对比）
  - [ ] 手写 Attention Gate 与 torch 对照 < 1e-5
  - [ ] torch 推演 U-Net++ 稠密跳连（X^{1,2}=160 通道怎么来）
  - [ ] 手写 patch merge/expand 并验证语义
  - [ ] 默写 9 个家族成员的一句话贡献 + 选型口诀
  - [ ] 口算 V-Net 3 级池化形状（64³→32³→16³→8³）"""))

    return build(cells)


if __name__ == '__main__':
    save(nb09(), '09-U-Net系列图像分割详解.ipynb')
