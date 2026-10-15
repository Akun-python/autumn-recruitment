# 🕸️ 22-图神经网络与知识图谱：GCN / GAT / GraphSAGE / TransE / 图表示

> 图与知识是**风控、社交、搜索、大模型（GraphRAG）**的重要方向。
> 本模块覆盖**图基础与图表示（随机游走/Node2Vec）、GCN（消息传递）、GAT 与 GraphSAGE、
> 链路预测与图对比学习、知识图谱构建（三元组/sqlite3 实战）、TransE 推理、面试八股，
> 以及进阶图结构：异构图（RGCN/元路径）、超图（HGNN）、动态图（时间编码）**，
> 全部可执行、含断言与可视化；**GNN 模型（GCN/GAT/GraphSAGE/TransE/InfoNCE/RGCN/HGNN）全部用 torch 手搓**
> （`torch.nn.Module` 自定义网络 + autograd 训练，非调库），图结构/存储部分用 networkx + sqlite3。
>
> 面向秋招画像：**图算法、风控算法、推荐算法、NLP/知识图谱、大模型 RAG** 岗位。
> 前置衔接：02-机器学习（向量/相似度）、03-深度学习（注意力/MLP）、
> 19-推荐系统（链路预测/embedding 检索）、04-LLM 大模型（GraphRAG）。

> 🖼️ 教学 notebook 基于 torch / numpy / pandas / matplotlib / networkx / sqlite3，自包含可执行，
> 含断言与数值对照；绘图统一中文字体并显式 `plt.show()`。

## 知识地图（面试考察的七大块）

```
图神经网络与知识图谱考点
├── 1. 图基础与表示  networkx · 度/最短路/连通分量/聚类系数 · 随机游走 · Node2Vec(p/q)
├── 2. 消息传递      GCN（对称归一化/半监督）· GAT（注意力加权）· GraphSAGE（采样聚合）
├── 3. 图任务       链路预测（CN/AA/Hits@K）· 图对比学习（扰动+InfoNCE）
├── 4. 知识图谱构建  三元组/RDF · sqlite3 存储查询 · 实体对齐 · 构建五步
├── 5. 知识推理      TransE（h+r≈t）· margin 损失 · KG 补全 · RotatE 等变体
├── 6. 进阶图结构   异构图（RGCN 关系矩阵 · 元路径）· 超图（关联矩阵 · HGNN）· 动态图（快照 · 时间编码）
└── 7. 八股与手撕   直推/归纳 · 过平滑 · 图大怎么训 · BFS/GCN 单层手撕
```

## 学习路线（推荐顺序）

1. **打基础（00）**：networkx 建图 + 图性质 + 随机游走/Node2Vec ✅
2. **核心模型（01-02）**：手写 GCN（归一化+半监督）→ GAT 注意力 + GraphSAGE 采样 ✅
3. **任务实战（03）**：链路预测（CN/AA + AUC）+ 图对比学习 ✅
4. **知识图谱（04-05）**：三元组存储查询（sqlite3）→ TransE 手写推理 ✅
5. **进阶结构（07）**：异构图（RGCN/元路径）→ 超图（HGNN）→ 动态图（时间编码）✅
6. **冲刺（06）**：18 个高频速答 + 手撕（BFS 最短路/单层 GCN 消息传递）✅

## 教学 Notebook 索引（8 篇，全部通过 nbclient 执行验证）

| 笔记 | 核心手写/演示 | 状态 |
|------|--------------|------|
| 00-图基础与图表示学习 | networkx 建图可视化、邻接矩阵、度/最短路/连通分量/割边、随机游走、Node2Vec p/q、与词向量对比 | ✅ |
| 01-图卷积网络GCN | 加自环+对称归一化、两层 GCN 前向、半监督节点分类（准确率>70%）、感受野分析、过平滑 | ✅ |
| 02-图注意力与采样邻居 | 注意力系数手写（softmax 归一化）、GCN vs GAT 权重可视化、GraphSAGE 邻居采样聚合、聚合器对比 | ✅ |
| 03-链路预测与对比学习 | 共同邻居/Adamic-Adar 手写、留出法 AUC 评估、图对比学习（扰动+InfoNCE）、工业应用 | ✅ |
| 04-知识图谱构建 | 三元组、sqlite3 存储与模式匹配查询、networkx 图可视化（带关系标签）、实体对齐、构建五步 | ✅ |
| 05-知识图谱推理TransE | h+r≈t 距离、margin 损失+负采样训练、Hits@K 评估（>70%）、推理补全、局限与变体 | ✅ |
| 06-图神经网络与知识图谱面试八股 | 30 秒框架、18 个高频速答、BFS 最短路手撕、单层 GCN 消息传递手撕 | ✅ |
| 07-进阶图结构：异构图、超图与动态图 | RGCN 关系矩阵手写、元路径实例计数、HGNN 超图卷积手写（关联矩阵）、时间编码 ψ(Δt) 与时间衰减聚合 | ✅ |

## 🖼️ 图库：自绘高清流程图（300dpi · `教学/images/`）

| 图 | 说明 | 用于 Notebook |
|---|------|------|
| [`gnn_message_passing.png`](教学/images/gnn_message_passing.png) | GCN 消息传递聚合（邻居特征→聚合→更新） | 01 |
| [`transe_triplet.png`](教学/images/transe_triplet.png) | TransE 三元组向量平移（h+r≈t） | 05 |
| [`gnn_interview_map.png`](教学/images/gnn_interview_map.png) | 图神经网络面试知识地图 | 06 |
| [`hetero_metapath.png`](教学/images/hetero_metapath.png) | 异构图（networkx 专业绘制：用户-商品-店铺三类型 + 元路径 U-P-S 红色高亮） | 07 |
| [`hypergraph_hgcn.png`](教学/images/hypergraph_hgcn.png) | 超图 vs 普通图对比（超边圈 ≥3 节点） | 07 |
| [`dynamic_snapshots.png`](教学/images/dynamic_snapshots.png) | 动态图快照序列（t1→t2→t3 边演化，新边橙色高亮） | 07 |

**📷 真实配图（Wikimedia Commons，CC BY-SA 4.0，已插入对应 notebook）**

| 图 | 说明 | 用于 Notebook |
|---|------|------|
| [`web_transformer_arch.png`](教学/images/web_transformer_arch.png) | Transformer 完整架构（注意力=图上消息传递） | 00 |
| [`web_node2vec.svg`](教学/images/web_node2vec.svg) | Node2Vec 原论文图解（Grover & Leskovec 2016，BFS/DFS 游走与 p/q 参数） | 00 |
| [`web_mpnn_message.png`](教学/images/web_mpnn_message.png) | 消息传递神经网络 MPNN 图解（节点聚合邻居更新） | 01 |
| [`web_gat_attention.png`](教学/images/web_gat_attention.png) | GAT 注意力机制图解（一作 Petar Veličković 绘制） | 02 |
| [`web_knowledge_graph.png`](教学/images/web_knowledge_graph.png) | Wikidata 知识图谱可视化（三元组真实样子） | 04 |
| [`web_kge_embedding.png`](教学/images/web_kge_embedding.png) | 知识图谱嵌入示意（实体/关系→向量） | 04, 05 |
| [`web_bipartite_k32.svg`](教学/images/web_bipartite_k32.svg) | 完全二部图 K₃,₂（用户-物品构图） | 03 |
| [`web_contrastive_pairs.png`](教学/images/web_contrastive_pairs.png) | 对比学习正/负样本对（拉近正对、推远负对） | 03 |
| [`web_hypergraph.svg`](教学/images/web_hypergraph.svg) | 超图示意图（超边圈 ≥3 节点，标准记号） | 07 |
| [`web_directed_hypergraph.svg`](教学/images/web_directed_hypergraph.svg) | 有向超图示例（多对一/一对多超边） | 07 |
| [`web_dynamic_evolution.png`](教学/images/web_dynamic_evolution.png) | 动态网络中社区演化（社区分裂/转移） | 07 |

**📄 经典论文原图（arXiv 论文原图，已插入对应 notebook 并标注出处）**

| 图 | 论文出处 | 用于 Notebook |
|---|------|------|
| [`paper_gcn_karate.png`](教学/images/paper_gcn_karate.png) | Kipf & Welling, GCN, ICLR 2017（arXiv:1609.02907）图1：空手道俱乐部网络 | 01 |
| [`paper_gcn_karate_emb.png`](教学/images/paper_gcn_karate_emb.png) | Kipf & Welling, GCN, ICLR 2017 图2：GCN 嵌入可视化 | 01 |
| [`paper_mpnn_fig1.png`](教学/images/paper_mpnn_fig1.png) | Gilmer et al., MPNN, ICML 2017（arXiv:1704.01212）图1：分子消息传递 | 01 |
| [`paper_gat_attn.png`](教学/images/paper_gat_attn.png) | Veličković et al., GAT, ICLR 2018（arXiv:1710.10903）图1：注意力系数与多头 | 02 |
| [`paper_node2vec_bfsdfs.png`](教学/images/paper_node2vec_bfsdfs.png) | Grover & Leskovec, node2vec, KDD 2016（arXiv:1607.00653）图1：BFS vs DFS | 00 |
| [`paper_node2vec_walk.png`](教学/images/paper_node2vec_walk.png) | Grover & Leskovec, node2vec, KDD 2016 图2：p/q 游走概率 | 00 |
| [`paper_node2vec_homo.png`](教学/images/paper_node2vec_homo.png) | Grover & Leskovec, node2vec, KDD 2016 图4/5：同质 vs 结构等价 | 00 |
| [`paper_rgcn_update.png`](教学/images/paper_rgcn_update.png) | Schlichtkrull et al., R-GCN, ESWC 2018（arXiv:1703.06103）图2：关系分组聚合 | 07 |
| [`paper_hgnn_conv.png`](教学/images/paper_hgnn_conv.png) | Feng et al., HGNN, AAAI 2019（arXiv:1809.09401）：超图卷积层 | 07 |
| [`paper_hgnn_pipeline.png`](教学/images/paper_hgnn_pipeline.png) | Feng et al., HGNN, AAAI 2019 图1：HGNN 整体框架 | 07 |
| [`paper_hgnn_hyperedges.png`](教学/images/paper_hgnn_hyperedges.png) | Feng et al., HGNN, AAAI 2019：超边构造 | 07 |
| [`paper_tgn_fig1.png`](教学/images/paper_tgn_fig1.png) | Rossi et al., TGN, 2020（arXiv:2006.10637）图1：批处理计算流 | 07 |
| [`paper_tgat_arch.png`](教学/images/paper_tgat_arch.png) | Xu et al., TGAT, ICLR 2020（arXiv:2002.07962）图2：TGAT 架构 | 07 |
| [`paper_tgat_tep.png`](教学/images/paper_tgat_tep.png) | Xu et al., TGAT, ICLR 2020 图1：时间编码函数 | 07 |

## 与其他章节衔接

| 本模块主题 | 前置章节 | 衔接说明 |
|---|---|---|
| 向量/相似度/聚类 | [02-机器学习](../02-机器学习/README.md) | 图表示与链路预测的数值基础 |
| 注意力机制 / MLP | [03-深度学习](../03-深度学习/README.md) | GAT 注意力与 GNN 网络的训练基础 |
| 链路预测 / 双塔召回 | [19-推荐系统](../19-推荐系统/README.md) | 用户-物品图与向量召回同构 |
| GraphRAG / 检索增强 | [04-LLM大模型 07-RAG](../04-LLM大模型/README.md) | 知识图谱进 RAG 的落地方向 |
| 关系数据 / SQL | [17-数据科学](../17-数据科学/README.md) | 三元组存储与图查询的 SQL 基础 |

## 面试高频题型（详见 [高频面试题.md](高频面试题.md)）

- **GNN 基础类**：为什么用 GNN、消息传递框架、GCN 公式与归一化、感受野与过平滑
- **模型对比类**：GCN vs GAT vs GraphSAGE、直推式 vs 归纳式、大图训练（邻居采样/Cluster-GCN）
- **图任务类**：节点分类/链路预测/图分类、链路预测指标（AUC/Hits@K/MRR）、图对比学习
- **进阶结构类**：异构图 vs 同构图、RGCN 关系矩阵、元路径、超图关联矩阵与 HGNN、动态图建模（快照/时间编码）
- **知识图谱类**：三元组表示、构建五步、图数据库 vs 关系库、TransE 思想与变体、KG 补全评估
- **落地类**：GraphRAG、风控团伙识别、好友推荐、推荐二部图召回

## 验收标准

- [ ] 能用 networkx 建图并算度/最短路/连通分量/聚类系数/割边
- [ ] 能写随机游走并解释 Node2Vec 的 p/q 语义
- [ ] 能手写 GCN 层公式、对称归一化与半监督训练
- [ ] 能解释 GAT 注意力系数与 GraphSAGE 采样聚合
- [ ] 能手写共同邻居/Adamic-Adar 并用 AUC 评估链路预测
- [ ] 能用 sqlite3 构建三元组表并做模式匹配查询
- [ ] 能手写 TransE（margin 损失+负采样）并用 Hits@K 评估
- [ ] 能对比 GCN/GAT/GraphSAGE 的选择依据
- [ ] 能解释 RGCN 的多关系矩阵与元路径的语义约束
- [ ] 能用关联矩阵 H 表示超图并说清 HGNN 的"聚组-广播"传播
- [ ] 能对比离散快照 vs 连续时间两种动态图建模
- [ ] 能答 18 个高频速答题并手撕 BFS 最短路与单层 GCN