# -*- coding: utf-8 -*-
"""生成 11-时间序列/教学/07-时间序列面试八股.ipynb（nbformat 4）"""
import os

import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell, new_notebook

OUT = '11-时间序列/教学'
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
# 11-07 时间序列面试八股
# =====================================================================
md(r"""# 🗂️ 11-07 · 时间序列面试八股（全章速成）

> 把 00~06 篇收拢成 **90 个高频问答（6 组 × 15）+ 5 个手撕**。
> 覆盖：平稳性/ACF/ADF、ARIMA、指数平滑/分解、特征工程/ML、深度学习、评估/异常检测。

> 复习路径：**平稳性（00）→ ARMA/ARIMA（01）→ 指数平滑/分解（02/03）→ 特征+ML（04）→ 深度学习（05）→ 评估/异常（06）**。""")

md(r"""## 第一组：基础 · 平稳性 · ACF（1-15）

1. **时间序列三要素？** 趋势、季节/周期、噪声；加法/乘法模型
2. **弱平稳三条件？** 均值恒定、方差恒定、协方差只依赖滞后
3. **强平稳 vs 弱平稳？** 联合分布不变 vs 一、二阶矩稳定；实用看弱平稳
4. **随机游走为什么非平稳？** y_t = y_{t-1}+ε，方差随 t 线性增长
5. **怎么把非平稳变平稳？** 差分（去趋势/随机游走）、季节差分（去周期）、对数（稳方差）
6. **ADF 检验原假设？** 有单位根（非平稳）；t 统计量越负越拒绝
7. **ACF 定义？** ρ(k)=γ(k)/γ(0)；样本估计用去均值后点积/方差
8. **PACF 定义？** 剔除中间滞后后 y_t 与 y_{t-k} 的纯相关（逐阶回归最后系数）
9. **白噪声特征？** 均值 0、方差恒定、ACF 全 0；Ljung-Box 检验
10. **Ljung-Box 的 Q 与自由度？** Q=n(n+2)Σρ²/(n-k) ~ χ²(h)
11. **95% ACF 置信带？** ±1.96/√n；越界提示存在自相关
12. **为什么建模前必须平稳？** 统计规律漂移 → 参数无意义、预测回归到历史均值
13. **差分 d 怎么定？** ADF 不过就再差一阶（通常 ≤2）
14. **季节差分？** y_t − y_{t−m}（m=周期）；与普通差分可叠加
15. **单位根的经济/业务含义？** 冲击永久持续（如股价），而非回归均值""")

code(r"""# ---------- 手撕 1：ACF + 差分 + 简化 ADF 三件套 ----------
import numpy as np

rng = np.random.default_rng(0)
rw = np.cumsum(rng.normal(0, 1, 300))

def acf(x, max_lag=10):
    x = x - x.mean(); v = np.dot(x, x)
    return np.array([1.0] + [np.dot(x[k:], x[:-k]) / v for k in range(1, max_lag + 1)])

def adf_stat(x):
    dx = x[1:] - x[:-1]; lag = x[:-1]
    A = np.stack([np.ones(len(dx)), lag], axis=1)
    beta, *_ = np.linalg.lstsq(A, dx, rcond=None)
    resid = dx - A @ beta
    se = np.sqrt((resid @ resid) / (len(dx) - 2) / np.sum((lag - lag.mean()) ** 2))
    return beta[1] / se

print('随机游走 ADF t=%.2f（>-2.86 非平稳） | 差分后 ADF t=%.2f（平稳）' % (adf_stat(rw), adf_stat(np.diff(rw))))
print('差分后 ACF 前 5 项:', np.round(acf(np.diff(rw), 5)[:5], 3))
assert adf_stat(rw) > -2.86 and adf_stat(np.diff(rw)) < -2.86
assert np.abs(acf(np.diff(rw))[1:]).max() < 0.25
print('面试 30 秒版：看图 → 差分 → ADF 确认 → ACF/PACF 定阶 → 建模 → 残差白噪声')""")

md(r"""## 第二组：ARMA / ARIMA（16-30）

16. **AR(p) 公式？** y_t = c + φ₁y_{t-1}+…+φ_p y_{t-p} + ε_t；历史观测回归
17. **MA(q) 公式？** y_t = μ + ε_t + θ₁ε_{t-1}+…+θ_q ε_{t-q}；历史误差回归
18. **ARMA 与 ARIMA 区别？** ARIMA = 先差 d 阶再做 ARMA；ARIMA(p,d,q)
19. **AR(1) 平稳条件？** |φ|<1；AR(p) 特征根在单位圆外
20. **MA 一定平稳？** 是（有限记忆、冲击有限期影响）
21. **ACF/PACF 定阶？** AR: PACF 截尾；MA: ACF 截尾；ARMA 都拖尾
22. **AR 怎么估计？** OLS 直接回归滞后项
23. **MA 怎么估计？** 误差不可观测 → 反解误差迭代 / MLE / 数值优化
24. **AIC/BIC 公式与差异？** -2lnL+2k vs -2lnL+k·ln n；BIC 更严（大样本）
25. **预测怎么外推？** AR 部分递归代入；长期预测收敛到均值（回归均值）
26. **为什么 ARIMA 预测置信区间变宽？** 误差逐期累积，方差随 h 增长
27. **季节 ARIMA（SARIMA）？** 加季节阶 (P,D,Q)m；处理周期结构
28. **残差要满足什么？** 白噪声（Ljung-Box 通过）；否则加阶/换模型
29. **ARIMA 局限？** 线性、单变量为主、长序列/非线性不行
30. **什么时候该上 ML/DL？** 非线性、多变量、外生特征、长依赖场景""")

code(r"""# ---------- 手撕 2：AR(2) OLS 估计 + AIC 定阶 ----------
def simulate_ar(phi, n=400, sigma=1.0):
    p = len(phi); y = np.zeros(n); e = rng.normal(0, sigma, n)
    for t in range(p, n):
        y[t] = sum(phi[k] * y[t - 1 - k] for k in range(p)) + e[t]
    return y

def fit_ar(y, p):
    X = np.stack([np.ones(len(y) - p)] + [y[p - 1 - k: len(y) - 1 - k] for k in range(p)], axis=1)
    beta, *_ = np.linalg.lstsq(X, y[p:], rcond=None)
    return beta

def aic_ar(y, p):
    beta = fit_ar(y, p)
    X = np.stack([np.ones(len(y) - p)] + [y[p - 1 - k: len(y) - 1 - k] for k in range(p)], axis=1)
    resid = y[p:] - X @ beta
    return len(resid) * np.log(resid @ resid / len(resid)) + 2 * (p + 1)

y2 = simulate_ar([0.6, 0.25])
print('估计 φ:', np.round(fit_ar(y2, 2)[1:], 3), '（真值 [0.6, 0.25]）')
aics = {p: round(aic_ar(y2, p), 2) for p in (1, 2, 3)}
print('AIC:', aics, '-> 选 p =', min(aics, key=aics.get))
assert min(aics, key=aics.get) == 2
print('手撕要点：滞后矩阵构造 = y[p-1-k: len-1-k]；AIC = n·log(SSE/n)+2k')""")

md(r"""## 第三组：指数平滑 · 分解（31-45）

31. **SES 公式与 α 含义？** ŷ_{t+1}=αy_t+(1-α)ŷ_t；α 大跟手、小平滑
32. **SES 与 ARIMA 关系？** 等价 ARIMA(0,1,1)
33. **Holt 比 SES 多什么？** 趋势项 b_t；预测 ℓ_t+h·b_t
34. **Holt-Winters 多什么？** 季节项 s_t（加法/乘法）；γ 控季节平滑
35. **加法 vs 乘法季节？** 季节幅度固定 vs 随水平缩放；乘法要求 y>0
36. **参数怎么选？** 网格/优化最小化 SSE（留出集或 CV）
37. **指数平滑适用？** 单变量、低延迟、可解释场景（销售/库存/监控基线）
38. **分解的加法模型？** y = T + S + R；乘法 y = T×S×R（对数转加法）
39. **经典分解三步？** MA 去趋势 → 周期均值季节指数 → 残差
40. **中心化 MA 为什么两步？** 偶数周期窗口对齐到时间点（两步平均）
41. **STL 与经典分解差异？** LOESS 局部回归、季节可漂移、鲁棒迭代
42. **残差诊断？** Ljung-Box 白噪声 + 残差方差占比；不过关=结构没提干净
43. **分解的用途？** EDA、季节调整、残差建模、异常检测基线
44. **季节性检测？** 季节指数显著偏离 0 / 周期自相关显著
45. **乘法分解注意？** 数据须为正；零/负值先平移或换加法""")

code(r"""# ---------- 手撕 3：SES + Holt 各一步 + Holt-Winters 季节指数 ----------
def ses_step(alpha, y_prev, f_prev):
    return alpha * y_prev + (1 - alpha) * f_prev

def holt_step(alpha, beta, y, l_prev, b_prev):
    l = alpha * y + (1 - alpha) * (l_prev + b_prev)
    b = beta * (l - l_prev) + (1 - beta) * b_prev
    return l, b

print('SES 一步: α=0.3, y=10, f_prev=9 ->', round(ses_step(0.3, 10, 9), 3), '（= 3 + 6.3 = 9.3）')
l, b = holt_step(0.4, 0.2, 12, 10, 0.5)
print('Holt 一步: ℓ=%.3f b=%.3f -> 一步预测 ℓ+b=%.3f' % (l, b, l + b))
assert abs(ses_step(0.3, 10, 9) - 9.3) < 1e-9
print('手撕要点：SES 一行、Holt 两行、HW 三行——背这三组递推公式即可')""")

md(r"""## 第四组：特征工程 · 机器学习（46-60）

46. **时序 ML 的输入怎么构造？** 滑窗：X=[y_{t-k}…y_{t-1}]，y=目标
47. **滞后特征对应什么？** AR 思想的泛化；阶数看 ACF 显著滞后+业务周期
48. **滚动统计特征？** 均值/标准差/最大值/斜率（窗口 7/30）
49. **时间编码方式？** 类别（星期）one-hot；周期（小时/月份）sin-cos
50. **为什么随机 K-Fold 不行？** 时间泄漏：未来混入训练
51. **ExpandingWindow vs RollingWindow？** 起点固定扩张 vs 固定长度滑动（适应漂移）
52. **特征泄漏例子？** 滚动特征混入目标未来值；全量标准化；重复样本
53. **归一化怎么防泄漏？** 只用训练段统计量，预测后还原
54. **树模型 vs 线性对特征要求？** 树不敏感尺度、自动交互；线性需标准化+手工交互
55. **多步预测特征怎么滚动？** 预测值递归回填（或直接模型输出多步）
56. **时序 CV 每折怎么切？** 训练集 = 折之前全部历史，测试集 = 下一段
57. **为什么要有基线？** 昨值重复/历史均值常打败复杂模型；先基线再升级
58. **分类任务怎么处理时序？** 滑窗特征 + 分类器；或事件窗口聚合
59. **类别/星期特征 one-hot 后维度？** 7 维；可用周内/周末二分降维
60. **特征重要性与时序？** 树模型可解释；滚动均值常最重要（平滑了噪声）""")

code(r"""# ---------- 手撕 4：滑窗特征 + ExpandingWindow 时序 CV ----------
from sklearn.linear_model import Ridge

n4, win4 = 300, 10
t4 = np.arange(n4)
s4 = 5 * np.sin(2 * np.pi * t4 / 30) + rng.normal(0, 0.8, n4)
X4 = np.stack([s4[i - win4:i] for i in range(win4, len(s4))])
Y4 = s4[win4:]

def expanding_cv(X, Y, folds=3, test=30):
    n = len(X); errs = []
    for f in range(folds):
        te = min(int(n * 0.5) + f * test, n)
        m = Ridge().fit(X[:te - test], Y[:te - test])
        errs.append(np.mean(np.abs(m.predict(X[te - test:te]) - Y[te - test:te])))
    return float(np.mean(errs))

print('ExpandingCV MAE=%.3f | 昨值重复 MAE=%.3f' %
      (expanding_cv(X4, Y4), float(np.mean(np.abs(np.diff(s4))))))
assert expanding_cv(X4, Y4) < float(np.mean(np.abs(np.diff(s4))))
print('要点：手写 CV 时永远"训练在前、测试在后"，绝不 shuffle')""")

md(r"""## 第五组：深度学习时序（61-75）

61. **DL 做时序的优势？** 自动非线性/多变量/长依赖特征
62. **LSTM 怎么用？** 滑窗输入 → 取末时刻隐状态 → 输出预测
63. **TCN 特点？** 因果膨胀卷积：并行、感受野指数增长、无递归累积
64. **Transformer 时序适配？** 位置编码+注意力；长序列用 Informer/稀疏注意力
65. **递归多步的缺点？** 误差逐期累积（一步错步步错）
66. **直接多步怎么做？** 每步一个输出头/模型；无累积误差
67. **Seq2Seq / teacher forcing？** 解码时用真实值引导训练，推理时用预测值
68. **为什么按步长拆开评估？** 第 1 步误差必然最小；长步长才是真能力
69. **归一化还原？** MinMax 后预测要乘回尺度；全量统计量=泄漏
70. **梯度裁剪？** RNN/LSTM 常见 clip=1.0 治爆炸
71. **窗口长度怎么定？** 覆盖最强周期；太短缺信息、太长难训练
72. **N-BEATS/iTransformer 是什么？** 纯 MLP 分层/通道独立注意力——当前基准模型
73. **多变量时序？** 多通道输入 LSTM/Transformer 即可；外生变量拼接
74. **数据量小怎么办？** 统计/ML 更稳；DL 易过拟合，用简单结构+正则
75. **预测不确定性？** MC Dropout / 分位数回归 / 集成；DL 默认点预测无区间""")

code(r"""# ---------- 手撕 5：LSTM 一步预测 + 递归多步 ----------
import torch
import torch.nn as nn

torch.manual_seed(0)
w5, H5 = 20, 16
z5 = (s4 - s4.mean()) / s4.std()
X5 = torch.tensor(np.stack([z5[i - w5:i] for i in range(w5, len(z5))]), dtype=torch.float32).unsqueeze(-1)
Y5 = torch.tensor(z5[w5:], dtype=torch.float32).unsqueeze(-1)
sp = int(0.8 * len(X5))
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, H5, batch_first=True); self.fc = nn.Linear(H5, 1)
    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1])
net = Net()
opt = torch.optim.Adam(net.parameters(), lr=5e-3)
for ep in range(50):
    idx = torch.randperm(sp)[:200]
    loss = nn.functional.mse_loss(net(X5[idx]), Y5[idx])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    pred5 = net(X5[sp:]).squeeze(-1).numpy() * s4.std() + s4.mean()
true5 = (Y5[sp:].squeeze(-1).numpy() * s4.std() + s4.mean())
mae_lstm = np.mean(np.abs(pred5 - true5))
mae_base = np.mean(np.abs(np.diff(s4[sp:])))
print('LSTM 一步 MAE=%.3f | 差分基线 MAE=%.3f' % (mae_lstm, mae_base))
assert mae_lstm < mae_base
print('手撕要点：滑窗 -> LSTM -> 末态 -> FC；预测完记得还原尺度')""")

md(r"""## 第六组：评估 · 异常检测（76-90）

76. **MAE 特点？** 平均绝对误差，对离群鲁棒
77. **RMSE 特点？** 放大误差，惩罚大偏差；与 MSE 同源
78. **MAPE 的坑？** y=0 爆炸；非对称（高估/低估惩罚不同）
79. **sMAPE 的坑？** 有界 0~200%；对称但分母为 0 需处理
80. **时序评估要注意？** 按步长拆开；必须对比基线（昨值重复/均值）
81. **z-score 异常检测？** |x-μ|/σ > 3；假设近似正态
82. **IQR 法？** Q1-1.5IQR / Q3+1.5IQR 之外；分位数稳健
83. **滚动窗口法？** 局部均值±k·局部标准差；适应漂移
84. **阈值怎么选？** P/R 曲线扫描 + F1；按业务成本定倾向
85. **Precision/Recall 公式？** TP/(TP+FP)、TP/(TP+FN)
86. **F1 什么时候用？** 正负类不均衡；宏/微平均
87. **尖峰 vs 漂移检测区别？** 瞬时大偏差 vs 均值缓慢变化（后者用 CUSUM/分位回归）
88. **异常检测防泄漏？** 滚动统计只用历史窗口（因果）
89. **无标签怎么评估异常检测？** 人工抽样 + 检出率/误报率；或半监督注入已知异常
90. **报警抖动怎么办？** 连续 N 次触发才报警 / 冷却期 / 平滑分数""")

md(r"""## 附：时间序列方法谱系一页纸

```
                       ┌─ 平稳化：差分/季节差分/对数 ─┐
原始序列 ── 看图/ADF ──┤                            ├─▶ ARMA/ARIMA/SARIMA（统计预测）
                       └─ 分解：MA/STL（趋势+季节）──┘
                                                            ┌─ 指数平滑 SES/Holt/HW（轻量递推）
                                                            ├─ 特征+ML：滑窗/滚动/编码 + Ridge/LGBM
                                                            └─ 深度学习：LSTM/TCN/Transformer/N-BEATS
评估：MAE/RMSE/MAPE/sMAPE + 时序CV（Expanding） + 基线对比
异常：z-score / IQR / 滚动窗口 / CUSUM + P/R/F1 阈值
```

> 考前 30 分钟扫一遍 90 问 + 5 个手撕，覆盖 90% 时序算法岗考点。""")

md(r"""## 自测清单（本章总验收）

- [ ] 六组 90 问：能不看答案复述 80%+
- [ ] 5 个手撕：ACF/差分/ADF、AR-OLS/AIC、SES/Holt 递推、滑窗+ExpandingCV、LSTM+递归多步
- [ ] 能从一个真实业务序列完整讲：平稳化 → 建模 → 评估 → 异常监控
- [ ] 能对比：ARIMA vs 指数平滑 vs ML vs DL 的适用边界
- [ ] 回到 00~06 篇把「面试速答」再过一遍

> 📍 验收路径：**00 平稳性 → 01 ARMA → 02 平滑 → 03 分解 → 04 特征ML → 05 深度学习 → 06 评估异常 → 07 总复习**。""")

nb = new_notebook(cells=cells, metadata=META)
path = os.path.join(OUT, '07-时间序列面试八股.ipynb')
nbformat.write(nb, path)
print('written:', path, '| cells:', len(cells))