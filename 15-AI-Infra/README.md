# ⚙️ 15-AI-Infra：AI 基础设施（推理 / 训练 / 性能优化）

> AI Infra（AI 基础设施）岗位 = **让大模型"训得快、跑得省"的系统工程师**：
> 训练侧（分布式并行、混合精度、显存优化）+ 推理侧（KV Cache、量化、批处理、推理引擎）+
> 算子层（CUDA/GEMM 优化、算子融合）+ 性能分析（profiler、roofline、吞吐/延迟调优）。
>
> 面向秋招画像：**AI Infra / 大模型推理优化 / 训练平台 / 高性能计算（HPC）/ 异构计算 / CUDA 工程师** 岗位。
> 与 04-LLM大模型（模型结构）、08-优化算法（训练优化器/FP16-ZeRO）、13-编程语言（C++/Linux）、05-八股与工程（OS/网络/分布式）直接衔接。

> 🖼️ 配图与深化：16 篇教学 notebook 均含手写实现 + 数值对照（numpy/torch，无需 GPU 可执行）；
> 模块目录会随产出持续扩充，内容以当前目录和下方索引为准。

## 知识地图（面试考察的十大块）

```
AI Infra 考点
├── 1. 系统基础        CPU/GPU 架构差异 · 内存层次 · 带宽 vs 算力 · roofline 模型
├── 2. 算子层          GEMM 优化（tiling/register blocking）· 算子融合 · CUDA 线程模型
├── 3. 训练框架        计算图 · 自动微分（反向传播实现）· PyTorch 内核 · torch.compile（图模式）
├── 4. 混合精度        FP16/BF16/FP32 · loss scaling · 显存账本（参数/梯度/优化器状态/激活）
├── 5. 分布式训练      数据并行 DP/FSDP · 张量并行 TP · 流水线并行 PP · ZeRO · 梯度同步
├── 6. 通信            NCCL · 集合通信（allreduce/broadcast）· Ring AllReduce 通信量 · 拓扑
├── 7. 推理优化        KV Cache · 连续批处理（vLLM/PagedAttention）· 投机解码 · 并行采样
├── 8. 量化压缩        PTQ/QAT · INT8/INT4 · GPTQ/AWQ/SmoothQuant · 误差分析与校准
├── 9. 部署与监控      QPS/TTOT/TPOT 指标 · 推理服务架构 · profiler 定位瓶颈 · 成本估算
└── 🔟 稀疏、卸载与分层  MoE 专家分页（Expert Paging）· 专家卸载（Expert Offloading / CPU-GPU 流水）· 预测式专家预取（Predictive Expert Prefetch）· KV Cache 分层与页级调度（Paged / Tiered / Prefetched）· FP4/NVFP4 量化与 2:4 结构化稀疏
```

## 学习路线（推荐顺序）

1. **打底（00-01）**：先懂"系统怎么算"——计算图与自动微分引擎（手写 autograd），GEMM 与算子优化（手写分块乘法 + roofline）✅
2. **训练侧（02-04）**：混合精度与显存账本 → 分布式并行四件套（DP/FSDP/TP/PP + ZeRO）→ 集合通信与 NCCL（手写 Ring AllReduce）✅
3. **推理侧（05-07）**：KV Cache 与内存计算 → 推理引擎 vLLM 连续批处理/投机解码 → 量化与压缩（PTQ/GPTQ/AWQ/SmoothQuant）✅
4. **工程与冲刺（08-10）**：性能分析与优化（profiler/算子融合/成本估算）→ AI-Infra 面试八股速查（30 连问 + 手算题）→ FlashAttention 与 CUDA 线程模型 ✅
5. **MoE 推理系统（11-13）**：专家分页（页表/替换策略/命中率账本）→ 专家卸载与 CPU-GPU 双缓冲流水（重叠率/stall/tokens-s）→ 预测式专家预取（上层/马尔可夫/logistic 三种预测器 + 命中率-浪费率权衡）✅
6. **分层与低比特（14-15）**：KV Cache 的内存分层与页级调度（GQA vs MLA 账本 / 页表 + LRU + 前缀共享 COW / 预测式 KV 预取 / 压缩换容量）→ FP4/NVFP4 量化与 2:4 结构化稀疏（E2M1 编解码 · block scale 误差三分量 · 稀疏补偿取舍 · 字节账本）✅

## 教学 Notebook 索引

| # | 主题 | 核心手写/演示 | 状态 |
|---|------|--------------|------|
| 00 | [深度学习推理引擎：从计算图到自动微分](教学/00-深度学习推理引擎-计算图与自动微分.ipynb) | 手写 mini autograd（Value 类 + 拓扑序反向）数值梯度对照 <1e-8 / 引擎训练 XOR 100% / 7B 激活显存估算 / 算子融合实测 | ✅ |
| 01 | [矩阵乘法与算子级优化：GEMM 工程解剖](教学/01-矩阵乘法与算子级优化GEMM.ipynb) | naive vs 分块(tiled) vs 寄存器分块实测 / FLOPs·算力·带宽 roofline / 块大小甜点区扫描 | ✅ |
| 02 | [混合精度与显存账本](教学/02-混合精度与显存账本.ipynb) | FP16 上溢/下溢演示 / BF16 模拟截断 / loss scaling 救回梯度曲线 / 7B 显存四件套 + ZeRO-1/2/3 分片堆叠图 | ✅ |
| 03 | [分布式并行训练：DP/FSDP/TP/PP/ZeRO](教学/03-分布式并行训练DP-FSDP-TP-PP与ZeRO.ipynb) | 4 卡数据并行梯度 allreduce = 串行验证 / 张量并行行切·列切等价 + 通信量 / PP bubble 曲线 / 70B 显存-通信权衡双轴图 | ✅ |
| 04 | [集合通信与 NCCL：手写 Ring AllReduce](教学/04-集合通信与NCCL.ipynb) | 原语全家桶速验 / **手写 Ring AllReduce 两阶段**（P=4/8 数值验证）/ ring vs tree 通信量曲线 / 梯度桶化演示 | ✅ |
| 05 | [KV Cache 与推理内存](教学/05-KV Cache与推理内存.ipynb) | 缓存 vs 无缓存工作量曲线 / KV 内存公式手算与增长曲线 / 手写带缓存解码一致验证 / PagedAttention 分页 vs 连续预分配 | ✅ |
| 06 | [推理引擎：vLLM 连续批处理与投机解码](教学/06-推理引擎vLLM连续批处理与投机解码.ipynb) | 静态 vs 连续批处理调度模拟（活跃度曲线）/ 投机解码加速比公式+模拟 / decode 吞吐-延迟权衡曲线 | ✅ |
| 07 | [量化与模型压缩 PTQ/GPTQ/AWQ/SmoothQuant](教学/07-量化与模型压缩PTQ-GPTQ-AWQ-SmoothQuant.ipynb) | per-tensor vs per-channel 误差实测 / 输出域误差补偿机制（最小二乘投影验证）/ SmoothQuant 激活迁移数学等价验证 / 位宽-精度-显存权衡 | ✅ |
| 08 | [性能分析与优化](教学/08-性能分析与优化.ipynb) | 手写简易 profiler 时间拆解 / roofline 实测判定算力-带宽受限 / 算子融合收益实测 / 6ND 训练成本与推理成本估算 | ✅ |
| 09 | [AI-Infra 面试八股与高频题](教学/09-AI-Infra面试八股与高频题.ipynb) | 八大块知识地图 + 30 连问（30 秒答案）+ 5 道手算题自动判分 + 一句话串联全章 | ✅ |
| 10 | [FlashAttention 与 CUDA 线程模型](教学/10-FlashAttention与CUDA线程模型.ipynb) | 显存层级与 roofline 回顾 / 标准注意力 IO 瓶颈定量 / 手写分块 softmax（online softmax）数值对照 / flash attn 前向模拟（tiling 循环）/ CUDA 线程模型（grid-block-thread）与 block reduce 演示 / 面试考点 | ✅ |
| 11 | [专家分页（Expert Paging）与显存池化](教学/11-专家分页与显存池化.ipynb) | Mixtral 8x7B 专家参数账本（352 MB/专家 @BF16 · 45.1 G 参数 ≈90 GB · PCIe 单专家 ≈14 ms）/ 手写 ExpertPager（逻辑页表 → GPU 槽位 · LRU/LFU/FIFO 淘汰 · 命中率与 stall 统计）/ Zipf 热度轨迹模拟 / 槽位数与命中率扫描 / 与 PagedAttention 的页生命周期对照 | ✅ |
| 12 | [专家卸载（Expert Offloading）与 CPU-GPU 协同](教学/12-专家卸载与CPU-GPU协同.ipynb) | 卸载比例 `r = 1 - (B-P_dense)/P_exp` 手算（24 GB 卡跑 45 G 参数专家 · 专家 FP8 口径，BF16 则 ×2）/ 手写双缓冲流水模拟器（compute 与 transfer 重叠）/ 逐层同步 vs 静态预取 vs 动态触发三调度对比 / 带宽-驻留比例扫描（tokens/s、stall 占比）/ Fiddler·DeepSpeed-MoE·KTransformers 工程对照 | ✅ |
| 13 | [预测式专家预取（Predictive Expert Prefetch）](教学/13-预测式专家预取.ipynb) | 为什么必须预测（router 数据相关）/ 三种预测器手写：上层直取 · 马尔可夫转移矩阵 · numpy logistic regression / precision@k、recall@k、浪费率、有效加速比定义与实测 / 强/弱层间相关两组轨迹对比 / 预取提前层数 k 的命中率-浪费率权衡曲线 | ✅ |
| 14 | [KV Cache 的内存分层与页级调度](教学/14-KV Cache分层与页级调度.ipynb) | GQA vs MLA 的 KV 账本手算（327,680 B/token vs 70,272 B/token，MLA 仅 0.214×；128K×batch8 = 344 GB vs 74 GB）/ 手写 `KVTierManager`（逻辑页表 → HBM 槽位 → DRAM · LRU 淘汰 · 异步预取）/ 四种访问轨迹（长解码滑窗 · 多轮前缀复用 · agent 长尾扫描 · 混合）/ 页大小×容量扫描（差一页即命中率崩塌的相变）/ 前缀共享与 COW（0→75% 共享：命中率 0.600→0.862）/ 预测式 KV 预取（停顿 215→34.5 ms，浪费 3.1%）/ 压缩换容量（FP8/INT4：停顿 832→193 ms）/ TripleKV·Twindex·Aggron·TideCache·AscMLA 对照 | ✅ |
| 15 | [FP4 量化与结构化稀疏](教学/15-FP4量化与结构化稀疏.ipynb) | 手写 FP4（E2M1）可表示值集合与编解码 / NVFP4（block16 + FP8-E4M3 scale）vs INT4-per-channel vs FP8 的误差三分量（块粒度 ↑ 至 92~96%、scale 量化仅 ~1%）/ block 大小 8→128 扫描 / 2:4 结构化稀疏与补偿变体（保能量 vs 保逐点精度的取舍）/ MLA KV 的 FP4 账本（32K 上下文 BF16 2.30 GB → FP4 0.65 GB）/ 字节账本（BF16→FP4+2:4 共 7.11×）/ SmoothQuant 式通道均衡（离群场景 MSE −75.4%）/ 随机舍入偏差 / VR-SparseQuant·AscMLA·QMoE 对照 | ✅ |

## 🖼️ 图库：自绘高清流程图 + 官方高清配图

### 自绘高清流程图（11 张 · 300dpi · `教学/images/`，原创可放大打印）

| 图 | 说明 | 用于 Notebook |
|---|------|------|
| [`ai_infra_roadmap.png`](教学/images/ai_infra_roadmap.png) | AI-Infra 四大阶段知识地图（打底→训练侧→推理侧→工程冲刺） | 09 |
| [`ring_allreduce.png`](教学/images/ring_allreduce.png) | Ring AllReduce 两阶段（scatter-reduce + all-gather）4 卡示意 | 04 |
| [`batch_scheduling.png`](教学/images/batch_scheduling.png) | 静态 vs 连续批处理甘特图（长尾浪费 vs 完成即补位） | 06 |
| [`parallel_3d.png`](教学/images/parallel_3d.png) | DP×TP×PP 三维并行网格（64 卡 = 2×4×8） | 03 |
| [`memory_zeRO.png`](教学/images/memory_zeRO.png) | 训练显存四件套（BF16 7B：权重14+梯度14+Adam84≈112GB）→ ZeRO-3 分片（P=8 每卡≈14GB） | 02 |
| [`prefill_decode.png`](教学/images/prefill_decode.png) | prefill（算力受限）vs decode（带宽受限）两阶段画像 | 06 |
| [`quantization_scales.png`](教学/images/quantization_scales.png) | per-tensor vs per-channel 量化 scale（离群通道效应） | 07 |
| [`speculative_decoding.png`](教学/images/speculative_decoding.png) | 投机解码流程（草稿→并行校验→拒绝采样） | 06 |
| [`kv_cache_mem.png`](教学/images/kv_cache_mem.png) | 有/无 KV Cache 的注意力机制（O(T²)→O(T)） | 05 |
| [`paged_vs_contig.png`](教学/images/paged_vs_contig.png) | 连续预分配（20-40% 利用率）vs PagedAttention 分页 | 05 |
| [`roofline_annotated.png`](教学/images/roofline_annotated.png) | roofline 拐点标注（带宽受限 vs 算力受限） | 01 / 08 |

### 自绘机制图（SVG · 矢量 · 公式可缩放）

> 面向"过程可视化"教学：把推导中间步骤、调度时间线、内存/通信/量化迁移画成矢量图，放大不糊、公式清晰，随 Notebook 内嵌使用。

| 图 | 说明 | 用于 Notebook |
|---|------|------|
| [`00_autograd_backward.svg`](教学/images/00_autograd_backward.svg) | 前向计算图 → 反向梯度流（链式法则逐边传播） | 00 |
| [`02_precision_memory_tradeoff.svg`](教学/images/02_precision_memory_tradeoff.svg) | FP32/BF16/FP16/INT4 位宽-精度-显存权衡 | 02 |
| [`03_pipeline_1f1b_schedule.svg`](教学/images/03_pipeline_1f1b_schedule.svg) | 流水线并行 1F1B 调度时间线与气泡公式 | 03 |
| [`04_ring_tree_topology.svg`](教学/images/04_ring_tree_topology.svg) | Ring/Tree 拓扑连边与 α-β 时间模型交叉 | 04 |
| [`07_smoothquant_migration.svg`](教学/images/07_smoothquant_migration.svg) | SmoothQuant 激活-权重离群迁移（X′=X·diag(s)⁻¹） | 07 |
| [`08_profiler_fusion_timeline.svg`](教学/images/08_profiler_fusion_timeline.svg) | 融合 vs 未融合 kernel 时间线与 HBM 流量 | 08 |
| [`10_flashattention_tiling_hbm_sram.svg`](教学/images/10_flashattention_tiling_hbm_sram.svg) | 标准 vs tiled FlashAttention（HBM/SRAM/在线 softmax） | 10 |
| [`10_cuda_grid_block_thread.svg`](教学/images/10_cuda_grid_block_thread.svg) | Grid→Block→Warp→Thread 层级与 GEMM 索引 | 10 |
| [`11_expert_paging.svg`](教学/images/11_expert_paging.svg) | 逻辑专家页表 → GPU 槽位（LRU 淘汰）→ 缺页经 PCIe 从 CPU/DRAM 搬入 | 11 |
| [`12_offload_pipeline.svg`](教学/images/12_offload_pipeline.svg) | Layer i 计算 ‖ Layer i+1 专家搬运：双缓冲时间线上的 compute/transfer/stall | 12 |
| [`13_prefetch_timeline.svg`](教学/images/13_prefetch_timeline.svg) | router 输出 → 预测 → 后台 DMA 搬运 → 命中/未命中，"预取了但没用"的浪费区间 | 13 |
| [`14_kv_tier_paging.svg`](教学/images/14_kv_tier_paging.svg) | 逻辑 KV 页表 → HBM 槽位池 → DRAM/NVMe 分级；下方两条轨道：同步缺页搬运 vs 异步预取 | 14 |
| [`15_fp4_sparse.svg`](教学/images/15_fp4_sparse.svg) | FP4 位布局（1+2+1）与 8 格刻度尺 · 16 权重共用一把 block scale · 2:4 稀疏 + Tensor Core 2× 通路 | 15 |

### 官方高清配图（论文/官方文档原图，点击直达）

| 主题 | 官方来源 | 图 | 对照阅读 |
|------|---------|-----|---------|
| 3D 并行（DP×TP×PP 组合与 trade-off） | [Megatron-LM](https://arxiv.org/abs/2104.04473) | 图 1：并行配置示意 | 03 |
| PagedAttention 分页管理 KV | [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180) | 图 2/3：KV 块与页表 | 05 |
| FlashAttention IO 感知分块 | [FlashAttention](https://arxiv.org/abs/2205.14135) | 图 1：HBM/SRAM 分块 | 00/08 |
| SmoothQuant 激活-权重迁移 | [SmoothQuant](https://arxiv.org/abs/2211.10438) | 图 2：离群迁移示意 | 07 |
| GPTQ 量化重建与误差补偿 | [GPTQ](https://arxiv.org/abs/2210.17323) | 图 1/2：逐列重建 | 07 |
| ZeRO 的 P/G/O 三级分片 | [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054) | 图 1：显存分片 | 02/03 |
| Ring AllReduce 两阶段流水 | [Bringing HPC Techniques to DL](https://arxiv.org/abs/1707.03928) | 图 2：ring 收发轮转 | 04 |
| Transformer 架构总览 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 图 1 | 00 |
| KV Cache 机制图解 | [HuggingFace KV Cache 指南](https://huggingface.co/docs/transformers/kv_cache) | 官方图示 | 05 |
| NCCL 集合通信原语/拓扑 | [NCCL 用户指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/) | 原语与拓扑说明 | 04 |

> 说明：自绘图包含两类——11 张门面流程图（PNG 300dpi，可放大打印）与 13 张机制图（SVG 矢量，公式可缩放），均已内嵌到对应 Notebook 的「深入原理」处；官方链接来自论文与官方文档（稳定可达），可作为高清原图对照学习，不建议直接嵌入 Notebook（保持离线自包含）。

## 与其他章节衔接

> 本 README 位于 `15-AI-Infra/`，跨模块链接统一使用 `../模块名/路径`，方便从 GitHub 页面直接跳转。

| 本模块主题 | 前置章节 | 衔接说明 |
|---|---|---|
| 计算图 / 自动微分 | [03-深度学习 01-BP](../03-深度学习/模型笔记/01-多层感知机与反向传播.ipynb) | 从"手推梯度"到"引擎自动反向" |
| 混合精度 / 显存 / ZeRO | [08-优化算法 07-大模型训练优化实践](../08-优化算法/教学/07-大模型训练优化实践.ipynb) | 那边讲数值与内存，这边补系统实现 |
| 量化 | [04-LLM大模型 14/22](../04-LLM大模型/教学/14-量化入门PTQ与QAT.ipynb) | 04 讲算法，本模块补工程部署视角 |
| KV Cache / 推理 | [04-LLM大模型 19-KV缓存](../04-LLM大模型/教学/19-KV缓存.ipynb) | 04 讲原理与内存账本，本模块补引擎调度 |
| MoE 专家分页 / 卸载 / 预取 | [04-LLM大模型 16-MoE混合专家](../04-LLM大模型/教学/16-MoE混合专家.ipynb) | 04 讲路由、top-k 与负载均衡，本模块补"专家放不下"之后的显存墙、页表、卸载调度与预取预测 |
| KV Cache 分层 / 低比特量化 | [04-LLM大模型 19-KV缓存](../04-LLM大模型/教学/19-KV缓存.ipynb) · [07-量化与模型压缩](教学/07-量化与模型压缩PTQ-GPTQ-AWQ-SmoothQuant.ipynb) | 04 讲 KV 原理与内存账本，本模块 14 补页表分层/前缀共享/KV 预取；07 讲 PTQ 算法，本模块 15 补 FP4 block scale 与硬件结构化稀疏 |
| C++ / Linux / 分布式 | [13-编程语言](../13-编程语言/README.md) · [05-八股与工程](../05-八股与工程/README.md) | 工程表达与 OS/网络基础 |

## 面试高频题型（详见 [高频面试题.md](高频面试题.md)）

- **公式计算类**：FLOPs/参数量/显存/通信量的手算（RoPE 时 FLOPs、7B 训练显存、Ring AllReduce 通信量）
- **对比选型类**：DP vs TP vs PP vs FSDP；PTQ vs QAT；连续批处理 vs 静态批处理；TF32 vs FP16 vs BF16
- **手撕代码类**：autograd 反向、分块 GEMM、Ring AllReduce、softmax 数值稳定改写
- **排查调优类**：显存 OOM 定位、吞吐瓶颈（算力受限 vs 带宽受限）、量化掉点排查
- **前沿选型类**：KV 分页 vs 专家分页的同构性、前缀共享 COW、NVFP4 vs INT4-per-channel、何时该退回 FP8、2:4 稀疏为什么能加速

## 验收标准

- [x] 全模块 16 篇 notebook 设计为可执行：00 计算图与自动微分 / 01 GEMM 算子优化 / 02 混合精度与显存 / 03 分布式并行 / 04 集合通信与 NCCL / 05 KV Cache 与推理内存 / 06 推理引擎 vLLM / 07 量化与压缩 / 08 性能分析 / 09 面试八股 / 10 FlashAttention 与 CUDA 线程模型 / 11 专家分页 / 12 专家卸载与 CPU-GPU 协同 / 13 预测式专家预取 / 14 KV Cache 分层与页级调度 / 15 FP4 量化与结构化稀疏（全部 numpy/torch 自包含，无需 GPU、无需联网；11-15 已用脚本顺序执行全部 code cell 验证通过，其余各篇已在本地按单元手工核对数值与公式一致性，建议在 Jupyter 环境跑一遍验证）
- [x] 每篇含：问题表 → 直觉 → 公式/原理 → 手写实现 → 数值对照 → 面试考点 → 自测清单
- [x] 全部代码 numpy/torch 自包含，随机种子固定，可复现
- [ ] 可进一步扩充：训练/推理端到端案例（如从零训一个 1B 的显存与耗时预算表）、框架源码导读（vLLM/FSDP 关键类）