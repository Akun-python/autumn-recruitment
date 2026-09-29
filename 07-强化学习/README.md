# 🤖 07-强化学习

> 目的：算法 / 大模型（LLM）岗面试的 **强化学习** 主线，从 MDP 到 DQN、PPO、GRPO（DeepSeek-R1），
> 全部为**手写 numpy 教学 notebook**（不调 gym / torch 封装，环境自写，纯标准库可运行）。
> 与 `04-LLM大模型/教学/21-PPO与GRPO手推.ipynb` 前后衔接（本篇给 RL 底座，那篇给 LLM 推导细节）。

## 📓 教学 Notebook 索引（按序学习，逐 cell 运行）

> 每篇均为 **44–46 cells**，包含：变种模型谱系表格、逐步推导（面试手推模板）、
> 可运行 numpy 实验、matplotlib 过程图 2 张（见 [📊 推导过程与配图](#-推导过程与配图)）。

| # | Notebook | 核心内容 | 手写/实验 | 优先级 |
|---|----------|----------|-----------|--------|
| 00 | [00-MDP与贝尔曼方程.ipynb](教学/00-MDP与贝尔曼方程.ipynb) | MDP/MRP/POMDP 谱系 / 回报与折扣 γ / V·Q / 贝尔曼方程逐步推导 / 收缩性 | 4×4 网格世界 + 值迭代 + 收缩收敛图 | 🔴 必背 |
| 01 | [01-动态规划与表格方法.ipynb](教学/01-动态规划与表格方法.ipynb) | DP 变种谱系(同步/异步/就地) / 策略评估 / 策略迭代 / 值迭代 / 改进定理推导 | 策略迭代收敛 + 两种 DP 成本结构图 | 🔴 必背 |
| 02 | [02-蒙特卡洛与时序差分.ipynb](教学/02-蒙特卡洛与时序差分.ipynb) | MC/TD 变种谱系 / TD(0) / n步 / TD(λ) 前后向等价推导 / 资格迹 | 随机游走 TD vs MC + λ 扫描图 | 🔴 必背 |
| 03 | [03-Q学习与SARSA.ipynb](教学/03-Q学习与SARSA.ipynb) | 控制变种谱系 / Q(off) / SARSA(on) / 期望 SARSA / Double Q(E[max]≥maxE 推导) / Dyna-Q | 悬崖漫步学习曲线 + 最大化偏差直方图 | 🔴 必背 |
| 04 | [04-深度Q网络DQN.ipynb](教学/04-深度Q网络DQN.ipynb) | DQN 家族变种目标推导 / 回放 / 目标网络 / DDQN / Dueling 可辨识性 / PER / Rainbow | numpy 迷你 DQN + 目标同步/降高估图 | 🔴 必背 |
| 05 | [05-策略梯度与Actor-Critic.ipynb](教学/05-策略梯度与Actor-Critic.ipynb) | 策略方法变种谱系 / 策略梯度定理四步推导 / REINFORCE+基线 / AC / GAE / A2C | 基线方差消融 + 推导流程图 | 🔴 必背 |
| 06 | [06-PPO与GRPO.ipynb](教学/06-PPO与GRPO.ipynb) | 对齐变种谱系 / TRPO→PPO 代理目标推导 / clip / RLHF 三阶段 / GRPO(R1) / DPO | clip 曲面图 + β-KL 权衡图 | 🔴 必背 |
| 07 | [07-强化学习面试八股与高频题.ipynb](教学/07-强化学习面试八股与高频题.ipynb) | 90 连问速答 / 变种谱系总表 / 手推三件套 / 开放题框架 / 易错点 | 老虎机 UCB 等实验 + 算法家族谱系树 | 🔴 必背 |

## 📊 推导过程与配图（教学/images/，共 15 张）

| 图 | 所在篇 | 讲什么 |
|---|---|---|
| bellman_discount.png | 00 | 折扣 γ 对远期奖励权重的衰减 |
| bellman_contraction.png | 00 | 贝尔曼算子收缩，不同初值收敛同一 V* |
| dp_convergence.png | 01 | 值迭代误差按 γ^k 收缩（log 斜率） |
| dp_iters_compare.png | 01 | 策略迭代 vs 值迭代的成本结构 |
| mc_td_curves.png | 02 | 随机游走 TD vs MC 学习曲线 |
| td_lambda_scan.png | 02 | TD(λ) λ 扫描的偏差-方差鞍底 |
| q_vs_sarsa_curves.png | 03 | 悬崖漫步 on/off-policy 行为差异 |
| double_q_bias.png | 03 | 最大化偏差 E[max]≥maxE 直方图 |
| dqn_target_sync.png | 04 | 目标网络"冻结-同步"机制 |
| ddqn_bias.png | 04 | DDQN 抑制 Q 值虚高 |
| pg_flow.png | 05 | 策略梯度定理 4 步推导流程 |
| pg_baseline_var.png | 05 | 三种基线方差对比（AC 动机） |
| ppo_clip_curve.png | 06 | PPO clip 目标曲面与信任区域 |
| rlhf_kl_beta.png | 06 | KL 系数 β 的奖励-漂移权衡 |
| rl_family_tree.png | 07 | 全模块算法家族谱系树 |

> 复习方式：**看到图 → 讲出对应公式 + 推导要点 + 面试问法** 三件套。

## 🗺️ 学习路线（3 轮）

- **第 1 轮 · 打底（2 天）**：00 → 01 → 02 → 03，逐 cell 运行，重点看**推导过程**与**谱系表格**
- **第 2 轮 · 进阶（2 天）**：04 → 05 → 06，结合 `images/` 过程图复述原理，跑通手写实现
- **第 3 轮 · 冲刺（面试前 1 天）**：07 全部八股速答 1 分钟 + 手推三件套默写 3 遍 + 看图复述

## 🔗 关联模块

- LLM 对齐推导：`../04-LLM大模型/教学/21-PPO与GRPO手推.ipynb`（PPO/GRPO 在语言模型上的完整手推）
- 基础概率/优化：`../02-机器学习/推导笔记/`（梯度下降、正则与 RL 的优化直觉相通）
- 面试复盘：`../06-面试复盘/复盘模板.md`

## ✅ 自测标准

- [ ] 能默写：贝尔曼期望/最优方程、Q-learning 更新、策略梯度定理、PPO clip 目标、GRPO 优势公式
- [ ] 能对比：MC vs TD、Q-learning vs SARSA、PPO vs TRPO、PPO vs GRPO
- [ ] 能在 30 秒内完成"手推三件套"：贝尔曼方程、策略梯度定理、PPO clip 目标
- [ ] 能手写：4×4 网格值迭代、悬崖漫步 Q-learning、多臂老虎机 UCB（各 15 分钟内）
- [ ] 能讲清：为什么 DQN 要回放+目标网络；为什么 GRPO 去 Critic；RLHF 的 KL 项防什么
- [ ] 8 本 notebook 全部可从上到下运行通过（nbclient 已全量验证）
