# 🧬 25-生物信息学：序列 / 基因 / 宏基因组 / 数据处理

> 生物信息学 = **用计算处理生物数据**：读序列 → 比对 → 注释 → 定量 → 群落分析。
> 本模块从「FASTA/FASTQ 数据格式」起步，覆盖序列比对（动态规划手写）、
> 基因结构（ORF/密码子）、基因表达（RNA-seq/TPM/差异表达）、
> 变异（VCF/SNP）、系统发育（进化树）、以及**宏基因组**
> （OTU/α-β 多样性/功能注释）的常见数据分析全流程。
>
> 面向画像：**生物信息学 / 计算生物学 / 基因组学 / 宏基因组分析 / 生信工程师** 岗位。
> 00-11 篇基于 numpy/pandas/scipy/matplotlib 自包含实现（可离线运行）；
> **12-14 实战篇使用真实 NCBI 基因组 + 专业包**（Biopython / pyrodigal / scikit-bio），
> 数据在 `data/` 目录，完整走一遍宏基因组「质控 → 组装 → 基因预测 →
> 功能注释 → 物种分类 → 多样性分析」流程。

> 🖼️ 配图与深化：教学 notebook 全部自包含可执行，含断言与数值对照
> （手写 vs 库实现）；绘图统一中文字体并显式 `plt.show()`。
> 实战篇需专业包：`pip install biopython pyrodigal scikit-bio`（详见各篇首格）。

## 知识地图（八大块 + 实战流程）

```
生物信息学考点
├── 1. 序列与格式   FASTA/FASTQ/GFF/VCF · 碱基/氨基酸 · 质量值 Phred · 序列统计
├── 2. 序列比对     Needleman-Wunsch(全局) · Smith-Waterman(局部) · 打分矩阵 · 回溯
├── 3. 搜索与索引   k-mer · seed-extend · BLAST 思想 · E-value
├── 4. 基因结构     ORF/6框翻译 · 密码子表 · 基因注释(GFF) · 启动子/外显子内含子
├── 5. 表达定量     RNA-seq(计数/TPM/FPKM) · 归一化 · 差异表达(t检验/火山图)
├── 6. 变异分析     SNP/Indel · VCF 解析 · 等位基因频率 · 注释
├── 7. 系统发育     距离矩阵 · UPGMA/邻接法 · 进化树解读
├── 8. 宏基因组     OTU/ASV · α多样性(Shannon/Simpson/Chao1) · β多样性(PCoA/PERMANOVA) · 功能注释
└── 9. 实战全流程(真实数据+专业包)  测序模拟/质控 → de Bruijn/consensus 组装 → pyrodigal
    基因预测 → 同源功能注释 → k-mer 物种分类 → skbio α/β 多样性与 PERMANOVA
```

## 学习路线（推荐顺序）

1. **打底（00-03）**：序列格式与统计 → 全局/局部比对（DP 手写）→ 共识与保守性 → BLAST 思想 ✅
2. **基因与表达（04-06）**：ORF 检测与注释 → RNA-seq 归一化与差异表达 → VCF 变异分析 ✅
3. **进化与群落（07-09）**：系统发育树 → 宏基因组 OTU 与 α 多样性 → β 多样性与群落比较 ✅
4. **冲刺（10-11）**：宏基因组功能注释 → 面试八股 60 连问 + 手撕 ✅
5. **实战（12-14，真实数据 + 专业包）**：质控与组装 → 基因预测与功能注释 → 物种分类与多样性 ✅

## 教学 Notebook 索引（15 篇，全部通过 nbclient 执行验证）

| 笔记 | 核心手写/演示 | 状态 |
|------|--------------|------|
| 00-生物数据基础与序列表示 | FASTA/FASTQ 解析、Phred 质量值、GC 含量、k-mer 计数 | ✅ |
| 01-序列比对入门 | Needleman-Wunsch 全局比对 DP 手写、回溯、Smith-Waterman 局部比对 | ✅ |
| 02-多重序列比对与保守性 | 一致性序列、保守性打分、位点熵、蛋白家族概念 | ✅ |
| 03-BLAST与序列搜索 | k-mer 索引、seed-extend、同源命中、E-value 直觉 | ✅ |
| 04-基因结构与注释 | 6 框翻译、ORF 检测手写、密码子表、GFF 注释解析 | ✅ |
| 05-基因表达数据分析 | RNA-seq 计数、TPM/RPKM/FPKM 手写、差异表达 t 检验、火山图 | ✅ |
| 06-基因组变异与VCF | SNP/Indel、VCF 解析、等位基因频率、变异类型 | ✅ |
| 07-系统发育树入门 | 距离矩阵、UPGMA 手写、NJ 直觉、进化树解读与 bootstrap | ✅ |
| 08-宏基因组OTU与α多样性 | OTU/ASV 聚类、丰度矩阵、Shannon/Simpson/Chao1 手写、稀疏曲线 | ✅ |
| 09-宏基因组β多样性与群落比较 | Bray-Curtis/Jaccard 手写、PCoA、NMDS 直觉、PERMANOVA 模拟 | ✅ |
| 10-宏基因组功能注释与物种组成 | 物种级联分类、功能注释(KEGG/COG 概念)、相对丰度堆叠图 | ✅ |
| 11-生物信息学面试八股与高频题 | 60 连问速答 + 手撕（比对/ORF/TPM/Shannon/UPGMA/距离） | ✅ |
| **12-宏基因组实战·质控与组装** | **真实 E.coli 基因组 + Biopython**：模拟测序、FASTQ 解析、滑动窗口质控、de Bruijn 图、consensus 组装、N50 | ✅ |
| **13-宏基因组实战·基因预测与功能注释** | **pyrodigal（真实 Prodigal）**：真实基因组预测 4000+ 基因、起始密码子偏好、GC3、Biopython 同源功能注释 | ✅ |
| **14-宏基因组实战·物种分类与多样性** | **scikit-bio**：3 真实基因组 k-mer 指纹分类、α/β 多样性、PCoA、PERMANOVA（24 样本两组显著差异） | ✅ |

> **数据与依赖**：`data/` 存放 NCBI RefSeq 真实基因组（E. coli K-12 `NC_000913.3`、
> B. subtilis 168 `NC_000964.3`、S. aureus NCTC 8325 `NC_007795.1`，gzip 共约 3.4MB）
> 与演示 FASTQ；12-14 需要 `pip install biopython pyrodigal scikit-bio`（Anaconda 环境已验证）。

> **📖 论文配图（`figures/`，均开放获取可再分发）**：12-14 实战篇嵌入 8 幅真实论文插图
> 并与教学内容对应讲解——
> ① fastp 双端处理工作流（Chen et al., 2018, Bioinformatics）；
> ② fastp 质控前后碱基含量曲线（同上）；
> ②⁺ fastp 重复序列估计（同上，配 §4 去重）；
> ③ Prodigal 算法伪代码与动态规划连接示意（Hyatt et al., 2010, BMC Bioinformatics）；
> ④ Kraken k-mer→LCA 分类算法（Wood & Salzberg, 2014, Genome Biology）；
> ④⁺ Kraken 唾液宏基因组分类分布（同上，配 §1 物种组成）；
> ⑤ metaSPAdes 混合菌株 de Bruijn 图（Nurk et al., 2017, Genome Research）。
> 每幅图配中文"怎么读这张图"图注，并标注论文出处与开放许可（CC-BY 等）。

## 与其他章节衔接

| 本模块主题 | 前置章节 | 衔接说明 |
|---|---|---|
| 动态规划（比对） | [01-数据结构与算法](../01-数据结构与算法/README.md) | Needleman-Wunsch 是经典 DP；LCS 思想复用 |
| 统计检验（差异表达） | [23-统计学](../23-统计学/README.md) | t 检验 / FDR 多重比较在差异表达中的应用 |
| 聚类与降维（群落） | [02-机器学习](../02-机器学习/README.md) | PCoA/NMDS 是降维；PERMANOVA 是置换检验 |
| 概率与分布 | [00-数学基础](../00-数学基础/README.md) | 泊松/负二项在测序深度建模中的应用 |

## 面试高频题型（详见 [高频面试题.md](高频面试题.md)）

- **格式类**：FASTA vs FASTQ、Phred 质量值、GFF/VCF 字段
- **比对类**：NW vs SW、打分矩阵、空位罚分、比对的应用
- **搜索类**：BLAST 流程、k-mer、E-value 含义
- **基因类**：ORF 怎么找、6 框翻译、基因注释
- **表达类**：RNA-seq 流程、TPM vs FPKM、归一化、差异表达
- **变异类**：SNP/Indel、VCF、等位基因频率、注释
- **进化类**：UPGMA vs NJ、bootstrap、树的解读
- **宏基因组类**：OTU vs ASV、α/β 多样性、Chao1、PCoA、PERMANOVA

## 验收标准

- [ ] 能手写 FASTA/FASTQ 解析与 Phred 质量值换算
- [ ] 能手写 Needleman-Wunsch 与 Smith-Waterman 并回溯路径
- [ ] 能解释 BLAST 的 seed-extend 流程与 E-value
- [ ] 能完成 6 框翻译与 ORF 检测
- [ ] 能手写 TPM/RPKM 归一化并解释区别
- [ ] 能解析 VCF 并计算等位基因频率
- [ ] 能手写 UPGMA 构建进化树
- [ ] 能手写 Shannon/Simpson/Chao1 多样性指数
- [ ] 能手写 Bray-Curtis 距离并完成 PCoA
- [ ] 能解释 PERMANOVA 的置换检验原理
- [ ] 高频题 60+ 题可现场作答，手撕清单可现场完成