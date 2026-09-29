# 📈 11-时间序列

> 目标：把「时间序列」讲成一条主线——**平稳性（00）→ ARMA/ARIMA（01）→ 指数平滑（02）→ 分解（03）→ 特征工程+机器学习（04）→ 深度学习（05）→ 评估与异常检测（06）→ 面试八股（07）**。
> 面试考察的不是背公式，而是 **怎么判断平稳、怎么选模型、怎么评估、预测或监控时踩过哪些坑**。

## 复习策略

1. **先建平稳性直觉（00）**：趋势/季节/噪声三要素 → 弱平稳三条件 → 差分/ADF → ACF/PACF → 白噪声/Ljung-Box，理解"建模前先平稳化"
2. **统计模型两条线（01-03）**：
   - **ARMA/ARIMA**：AR=历史观测回归、MA=历史误差回归；ACF/PACF 定阶、AIC/BIC 选阶、OLS/网格估计
   - **轻量递推**：SES（等价 ARIMA(0,1,1)）→ Holt 加趋势 → Holt-Winters 加季节 → 经典分解（MA+季节指数）与 STL
3. **机器学习路线（04-05）**：滑窗特征（滞后/滚动/时间编码）→ 时序 CV（Expanding，防泄漏）→ Ridge/LGBM → LSTM/TCN/Transformer → 递归 vs 直接多步
4. **评估与监控（06）**：MAE/MSE/RMSE/MAPE/sMAPE 手写对照 → z-score/IQR/滚动窗口异常检测 → P/R 阈值
5. **手推优先**：ACF/PACF、ADF t 统计量、AR 的 OLS、AIC、SES/Holt/HW 递推、滑窗特征、ExpandingCV、指标公式

## 时间序列笔记索引（`教学/`，8 篇，全部含推导 + 手写实现 + 对照验证 + 自测清单）

| 篇目 | 核心内容 | 对照验证 |
|------|----------|----------|
| 00-基础与平稳性 | 三要素/弱平稳三条件/差分/ADF/ACF·PACF/Ljung-Box | 随机游走 vs 差分 ADF 符号、白噪声 ACF≈0 断言 |
| 01-ARMA与ARIMA | AR OLS 估计/MA 网格搜索/ACF·PACF 定阶/AIC 选阶/多步预测 | AR(1)/AR(2) 系数估计误差 <0.1、AIC 选出 p=2 |
| 02-指数平滑 | SES/Holt/Holt-Winters 手写/网格选参/95% 区间 ±1.96σ√h | 留出集 MAE 阈值、区间覆盖率 >0.8 |
| 03-时间序列分解 | 中心化移动平均/季节指数/残差诊断/Ljung-Box/STL 对比 | 季节幅度≈10、残差白噪声 p>0.05、方差占比 <10% |
| 04-特征工程与机器学习 | 滞后/滚动统计/时间编码/ExpandingWindow/泄漏陷阱 | Ridge MAE < 昨值重复×0.8、时序 CV 对比 |
| 05-深度学习时序预测 | 滑窗数据集/LSTM vs 昨值重复/递归 vs 直接多步/归一化防泄漏 | LSTM MSE < 基线×0.8、递归尾部误差累积断言 |
| 06-评估与异常检测 | MAE/MSE/RMSE/MAPE/sMAPE 手写 vs sklearn/z-score·IQR·滚动窗口/P-R | 指标逐位一致、三类检测抓尖峰、F1>0.8 |
| 07-时间序列面试八股 | 90 连问（6 组×15）+ ACF/AR-OLS/SES/ExpandingCV/LSTM 手撕 + 谱系 | 手撕题目全部可执行 |

## 本目录规划

```
11-时间序列/
├── README.md          # 本文件
├── 高频面试题.md      # 分类面经清单（带 TODO 打勾）
└── 教学/              # ✅ 8 篇 Jupyter notebook（推导 + 代码 + 对照 + 自测）
    └── images/        # 部分 notebook 落盘的插图
```

## 面试 30 秒速答（背诵骨架）

**平稳与识别**
- **弱平稳三条件**：均值/方差恒定、协方差只依赖滞后；随机游走非平稳（方差随 t 增长）
- **ADF**：`Δy = α + βt + γy_{t-1} + …`，检验 γ=0；t 越负越拒绝；差分 d≤2
- **ACF/PACF 定阶**：AR→PACF 截尾、MA→ACF 截尾、ARMA 都拖尾；置信带 ±1.96/√n
- **白噪声**：ACF 全 0；Ljung-Box Q 大 → 有残留自相关

**模型**
- **AR(p)**：OLS 直接估；**MA(q)**：反解误差迭代/网格；**AIC/BIC**：-2lnL+2k vs -2lnL+k·ln n
- **SES**：`ŷ=αy+(1-α)ŷ`，等价 ARIMA(0,1,1)；**Holt** 加趋势 ℓₜ+hbₜ；**HW** 加季节 sₜ（加法/乘法）
- **分解**：MA 去趋势 → 周期均值季节指数 → 残差；中心化 MA 偶数周期两步平均；STL=LOESS 迭代
- **特征+ML**：滞后（AR 泛化）/滚动统计/时间编码；随机 K-Fold 泄漏 → ExpandingWindow
- **多步**：递归（误差累积）vs 直接（各步独立）vs Seq2Seq；按步长拆开评估
- **评估与异常**：MAPE 怕 0、sMAPE 对称有界；z-score/IQR/滚动窗口；P/R-F1 挑阈值

## 验收标准

- [x] `高频面试题.md` 覆盖 平稳性 / ARMA / 平滑·分解 / 特征工程 / 深度学习 / 评估·异常
- [x] 时间序列笔记 8 篇，全部含推导 + 手写实现 + 对照验证 + 自测清单
- [x] 能手推：ACF/PACF、ADF、AR 的 OLS、AIC、SES/Holt/HW 递推、滑窗特征、ExpandingCV、指标公式
- [x] 能口算：±1.96/√n 置信带、ARIMA(0,1,1) 与 SES 等价、σ√h 区间扩张、多步递归误差累积
- [x] 全部 notebook 代码经批量执行验证（numpy/sklearn/torch 对照断言通过）

> 💡 手推优先级建议：ACF 公式 → ADF t 统计量 → AR 的 OLS → AIC → SES/Holt 两步 → HW 季节项 → 滑窗特征 → ExpandingCV → z-score 异常检测 → sMAPE。