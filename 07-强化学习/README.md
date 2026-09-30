# 🤖 07-强化学习

> 目的：算法 / 大模型（LLM）岗面试的 **强化学习** 主线，从 MDP 到 DQN、PPO、GRPO（DeepSeek-R1），
> 再到**连续动作（DDPG/TD3/SAC）、多智能体（MARL）、基于模型（MCTS/AlphaGo）、离线 RL、模仿学习** 的前沿全谱系；
> 全部为**手写 numpy 教学 notebook**（不调 gym / torch 封装，环境自写，纯标准库可运行）。
> 每本均已按**书籍章节级**扩写：每个核心知识点一节完整讲解（**本节导读 → 直觉引入 → 形式定义 →
> 逐步推导 → 数字例子 → 易错点 → 面试问法**），共 **80+ 节"讲透"内容**；
> 并嵌入**网络配图 13 张 + 自绘过程图 15 张**（见 [📊 推导过程与配图](#-推导过程与配图)）。
> 与 `04-LLM大模型/教学/21-PPO与GRPO手推.ipynb` 前后衔接（本篇给 RL 底座，那篇给 LLM 推导细节）。

## 📓 教学 Notebook 索引（按序学习，逐 cell 运行）

> 每篇均为 **10–46 cells**，包含：变种模型谱系表格、逐步推导（面试手推模板）、
> 可运行 numpy 实验、matplotlib 过程图（见 [📊 推导过程与配图](#-推导过程与配图)）。
> **08–12 为"故事版"新五篇**：每个算法开头都有**完整介绍 + 0 基础通俗例子 + 公式推导 + 手写实现**。

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
| 08 | [08-连续动作控制DDPG-TD3-SAC详解.ipynb](教学/08-连续动作控制DDPG-TD3-SAC详解.ipynb)（故事版） | 连续动作为什么难 / **DPG 定理四步推导+梯度检查** / DDPG 四网络+软更新 / OU 探索 / **TD3 三改进** / **SAC 熵正则+温度自适应** | DPG 解析 vs 数值梯度、软更新跟随、双 Q min 偏差、α 单调性 | 🔴 必背 |
| 09 | [09-多智能体强化学习MARL详解.ipynb](教学/09-多智能体强化学习MARL详解.ipynb)（故事版） | 多智能体是什么 / 随机博弈建模 / **三大挑战(非平稳/信用分配/扩展性)** / IQL / **CTDE** / **VDN→QMIX 单调性推导** / MADDPG / **COMA 反事实基线** | 囚徒困境 NE 枚举、独立学习震荡演示、QMIX 单调验证、COMA 归因 | 🔴 必背 |
| 10 | [10-基于模型强化学习与规划详解.ipynb](教学/10-基于模型强化学习与规划详解.ipynb)（故事版） | model-free vs model-based / 动力学模型拟合 / **Dyna-Q** / **MCTS 四阶段+UCT 公式推导** / AlphaGo 三件套 / MuZero / **模型误差累积** | 最小二乘学模型、Dyna 样本效率对比、手写 UCB 选择、误差爆炸演示 | 🔴 必背 |
| 11 | [11-离线强化学习详解.ipynb](教学/11-离线强化学习详解.ipynb)（故事版） | 离线 RL 是什么(看录像学开车) / **分布偏移+外推误差** / **BCQ ε-球约束** / **CQL 保守目标推导** / **IQL expectile 回归** | 外推误差拟合演示、BCQ 候选过滤、CQL 压 OOD、expectile 分位数 | 🟡 掌握 |
| 12 | [12-模仿学习与逆强化学习详解.ipynb](教学/12-模仿学习与逆强化学习详解.ipynb)（故事版） | 模仿学习是什么(教练示范) / **BC 行为克隆+复合误差推导** / **DAgger 数据集聚合** / **IRL 逆强化学习(特征匹配/MaxEnt 梯度)** / **GAIL 对抗模仿(min-max)** / 与 RLHF 衔接 | 复合误差放大演示、DAgger 覆盖对比、IRL 特征匹配收敛、GAIL 1D 对抗(判别器≈0.5) | 🔴 必背 |

## 📊 推导过程与配图（教学/images/，共 28 张 = 15 张自绘过程图 + 13 张网络配图）

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
| （新图：软更新跟随 / OU vs iid） | 08 | 目标网络慢跟随、探索噪声平滑性 |
| （新图：非平稳震荡 / QMIX 单调） | 09 | 独立学习不收敛、单调混合曲线 |
| （新图：Dyna 样本效率 / UCB 探索利用） | 10 | 模型复用收益、多臂选择分布 |
| （新图：外推误差 / expectile 分位数） | 11 | OOD 外推飞走、τ 敏感性 |
| （新图：复合误差 / GAIL 对抗） | 12 | BC 自回归漂移、判别器逼策略靠近专家 |

### 🖼️ 网络配图（按知识点嵌入 notebook，图源已注明出处）

| 图 | 所在篇 | 讲什么 | 图源 |
|---|---|---|---|
| rl_agent_env_loop.png | 00 | 强化学习交互回路（Agent-Environment Loop） | Wikimedia（CC BY-SA） |
| mdp_bellman_diagram.png | 00 | MDP 状态转移图 → 贝尔曼方程"读图" | Wikimedia / Sutton&Barto 图 3.4（CC BY-SA） |
| mdp_example.png | 01 | 完整 MDP 例子（手算策略评估题源） | Wikimedia（CC BY-SA） |
| 02_td_learning.png | 02 | 时序差分在状态链上"只前看一步" | Wikimedia（CC BY-SA） |
| 03_qlearning.png | 03 | Q 表初始化 vs 训练后的矩阵对比 | Wikimedia（CC BY-SA） |
| dqn_target_sync.png | 04 | 目标网络"冻结-同步"机制（自绘） | 本章自绘 |
| pg_flow.png | 05 | 策略梯度定理 4 步推导流程（自绘） | 本章自绘 |
| ppo_clip_curve.png | 06 | PPO clip 目标曲面与信任区域（自绘） | 本章自绘 |
| 07_rl_algorithms.png | 07 | 强化学习算法全景分类树 | OpenAI Spinning Up（MIT） |
| 08_robot_standup.png | 08 | 连续控制（机器人起身）真实场景 | OpenAI Spinning Up（MIT） |
| 09_prisoner.png | 09 | 囚徒困境矩阵（MARL 出发点） | Wikimedia（CC BY-SA） |
| 10_mcts.png | 10 | MCTS 四阶段"选择-扩展-模拟-回传" | Wikimedia（CC BY-SA） |
| 10_alphago.png | 10 | AlphaGo 三件套架构（策略/价值/MCTS） | OpenAI Spinning Up（MIT） |
| 12_imitation.png | 12 | 模仿学习方法训练曲线对比（BC 复合误差） | Wikimedia（CC BY-SA） |

> 网络图均按各源许可使用（Wikimedia CC BY-SA / OpenAI Spinning Up MIT），
> notebook 内图注处已注明出处；剩余 15 张为本章自绘过程图。
> 复习方式：**看到图 → 讲出对应公式 + 推导要点 + 面试问法** 三件套。

## 🗺️ 学习路线（3 轮）

- **第 1 轮 · 打底（2 天）**：00 → 01 → 02 → 03，逐 cell 运行，重点看**推导过程**与**谱系表格**
- **第 2 轮 · 进阶（3 天）**：04 → 05 → 06 → 08 → 09，结合 `images/` 过程图复述原理，跑通手写实现
- **第 3 轮 · 前沿+冲刺（2 天）**：10 → 11 → 12（前沿补齐）→ 07 全部八股速答 1 分钟 + 手推三件套默写 3 遍 + 看图复述

> 快速浏览线：08-12 每篇开头都有**0 基础通俗例子**，30 分钟内能建立全谱系直觉；
> 面试冲刺优先 08（连续动作）+ 09（多智能体）+ 12（模仿学习），这三篇是近年高频增量考点。

## 🔗 关联模块

- LLM 对齐推导：`../04-LLM大模型/教学/21-PPO与GRPO手推.ipynb`（PPO/GRPO 在语言模型上的完整手推）
- 多智能体时代：`../10-自然语言处理` 的 Agent/多 Agent 协作（与 09 篇 MARL 思想互通）
- 基础概率/优化：`../02-机器学习/推导笔记/`（梯度下降、正则与 RL 的优化直觉相通）
- 面试复盘：`../06-面试复盘/复盘模板.md`

## ✅ 自测标准

- [ ] 能默写：贝尔曼期望/最优方程、Q-learning 更新、策略梯度定理、PPO clip 目标、GRPO 优势公式
- [ ] 能默写（前沿）：DPG 定理、QMIX 单调约束、UCT 公式、CQL 保守目标、IQL expectile 损失、GAIL min-max 目标
- [ ] 能对比：MC vs TD、Q-learning vs SARSA、PPO vs TRPO、PPO vs GRPO、DDPG vs TD3 vs SAC
- [ ] 能对比（前沿）：VDN vs QMIX、MADDPG vs DDPG、BCQ vs CQL vs IQL、model-free vs model-based、BC vs DAgger vs IRL vs GAIL
- [ ] 能在 30 秒内完成"手推三件套"：贝尔曼方程、策略梯度定理、PPO clip 目标
- [ ] 能手写：4×4 网格值迭代、悬崖漫步 Q-learning、多臂老虎机 UCB（各 15 分钟内）
- [ ] 能讲清：为什么 DQN 要回放+目标网络；为什么 GRPO 去 Critic；RLHF 的 KL 项防什么
- [ ] 能讲清（前沿）：MARL 三大挑战；外推误差为什么致命；MCTS 四阶段；BC 复合误差；IRL 与 GAIL 对偶
- [ ] 13 本 notebook 全部可从上到下运行通过（nbclient 已全量验证）
