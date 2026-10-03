# 🎙️ 16-语音：语音识别与合成（ASR / TTS）

> 语音算法岗 = **让机器"听得懂、说得出"的算法工程师**：
> 识别侧（语音信号处理、特征提取、传统 GMM-HMM → 端到端 CTC/Attention/RNN-T 的 ASR 全链路）+
> 合成侧（声学模型、声码器、端到端 TTS：Tacotron / FastSpeech / VITS）+
> 工程侧（VAD/降噪/回声消除、流式 vs 非流式、CER/WER/MOS 评估与上线打磨）+
> 前沿侧（语音增强、声纹识别、语音转换 SVC、语音大模型与智能体）。
>
> 面向秋招画像：**语音算法 / ASR / TTS / 音频算法 / 语音信号处理 / 语音多模态** 岗位。
> 与 00-数学基础（FFT/概率）、02-机器学习（GMM-EM/HMM/CRF）、03-深度学习（BPTT/Attention/Transformer）、04-LLM大模型（多模态/语音大模型）、12-智能体开发（06-语音）直接衔接。

## 知识地图（面试考察的九大块）

```
语音算法考点
├── 1. 信号与特征     采样定理 · 分帧加窗 · STFT/语谱图 · Mel 刻度 · MFCC/Fbank · VAD/端点检测
├── 2. 传统识别        DTW · GMM-EM · HMM 三大问题（前向/维特比/前向后向）· 三音素 · 发音词典 · WFST 直觉
├── 3. 端到端 ASR      CTC（前向后向/对齐）· seq2seq+Attention（LAS）· RNN-T · 贪心 vs beam search · 流式 vs 非流式
├── 4. 语音合成 TTS    TTS 管线 · 声学模型 vs 声码器 · Tacotron 注意力 · 时长/韵律建模 · Griffin-Lim · VITS（flow/VAE）
├── 5. 评估与工程      CER/WER · MOS · 降噪/AEC · VAD 工程 · ASR 错误类型与纠错 · 流式服务上线
├── 6. 语音增强       谱减/维纳/掩码(IRM/cIRM) · 神经增强 · MCRA 噪声跟踪 · SI-SDR/PESQ/STOI · 混响/多通道
├── 7. 声纹识别        x-vector/ECAPA · EER/DET · PLDA/AHC 日志 · EEND/PIT · 防伪 anti-spoofing
├── 8. 语音转换 SVC    VQ-VAE/AutoVC/StarGAN-VC · HuBERT 内容特征 · F0 提取与搬移 · So-VITS-SVC 管线
└── 9. 语音大模型      Whisper · 音频 token(RVQ/EnCodec) · RNN-T 流式 · 级联 vs 端到端 · 语音智能体
```

## 学习路线（推荐顺序）

1. **打底（00）**：先懂"声音怎么变成特征"——采样/量化、分帧加窗、STFT 与语谱图、Mel 刻度、MFCC·Fbank 从零手写、VAD ✅
2. **传统路线（01）**：DTW 动态规划手写、GMM-EM 建模、HMM 三大问题（前向/维特比/前向后向）手写、孤立词识别实验 ✅
3. **端到端 ASR（02）**：CTC 前向-后向推导 + 手写、贪心解码、seq2seq + 注意力、CER/WER 评估 ✅
4. **TTS 合成（03）**：注意力 TTS（Tacotron 思路）、Griffin-Lim 相位重建手写、flow·VAE 直觉、MOS 评估 ✅
5. **八股冲刺（04）**：70 连问速答 + 手撕（MFCC/CTC/DTW/Griffin-Lim）+ 全章谱系图 ✅
6. **前端增强（05）**：谱减/维纳/IRM·cIRM 掩码从零实现、MLP 神经掩码（numpy 反向传播）、MCRA 噪声跟踪、SI-SDR/PESQ/STOI 评测、混响与多通道 ✅
7. **身份与日志（06）**：EER/DET、i-vector、x-vector/ECAPA、AAM-Softmax、PLDA、AHC 聚类、EEND/PIT、防伪与部署 ✅
8. **音色转换（07）**：F0 提取与搬移手写、VQ-VAE 三项损失、AutoVC 信息瓶颈、StarGAN 循环一致性、HuBERT 内容特征、So-VITS-SVC 管线拆解 ✅
9. **大模型收尾（08）**：Whisper 架构与多任务前缀、音频 token 化（RVQ）、RNN-T 流式、级联 vs 端到端延迟对比、语音智能体工程 ✅

## 教学 Notebook 索引

| 笔记 | 核心手写/演示 | 状态 |
|------|--------------|------|
| [00-语音信号基础与特征提取](教学/00-语音信号基础与特征提取.ipynb) | 分帧加窗/STFT/语谱图/MFCC·Fbank 从零手写/VAD | ✅ |
| [01-传统语音识别GMM-HMM与DTW](教学/01-传统语音识别GMM-HMM与DTW.ipynb) | DTW 动态规划手写/HMM 前向·维特比手写/GMM-EM/孤立词识别实验 | ✅ |
| [02-端到端ASR-CTC与Attention](教学/02-端到端ASR-CTC与Attention.ipynb) | CTC 前向-后向推导+手写/贪心解码/seq2seq+注意力/CER·WER | ✅ |
| [03-语音合成TTS-Tacotron与VITS](教学/03-语音合成TTS-Tacotron与VITS.ipynb) | 注意力 TTS/Griffin-Lim 相位重建手写/flow·VAE 直觉/MOS 评估 | ✅ |
| [04-语音面试八股与高频题](教学/04-语音面试八股与高频题.ipynb) | 70 连问速答+手撕（MFCC/CTC/DTW/Griffin-Lim）+谱系图 | ✅ |
| [05-语音增强与降噪](教学/05-语音增强与降噪.ipynb) | 谱减/维纳滤波/IRM·cIRM 掩码/MLP 神经掩码（numpy 反向传播）/MCRA/SI-SDR/混响·多通道 | ✅ |
| [06-声纹识别与说话人日志](教学/06-声纹识别与说话人日志.ipynb) | EER/DET 曲线/i-vector 演示/x-vector·ECAPA/AAM-Softmax/PLDA/AHC 聚类/EEND·PIT | ✅ |
| [07-语音转换与歌声转换SVC](教学/07-语音转换与歌声转换SVC.ipynb) | F0 提取与中值平滑/共振峰搬移可视化/VQ-VAE 量化演示/AutoVC·StarGAN·HuBERT/So-VITS-SVC | ✅ |
| [08-语音大模型与语音智能体](教学/08-语音大模型与语音智能体.ipynb) | Whisper 自回归解码演示/RVQ 残差量化/RNN-T 对齐计数/WER 手算/RTF·延迟对比 | ✅ |

> 说明：全部教学 notebook 含手写实现 + 数值对照（numpy/scipy，无需 GPU 可执行，音频用合成/内置数据离线自包含）；`05-08` 为后续扩充的增强/声纹/转换/大模型四本，与 `00-04` 构成完整九本体系。

## 与其他章节衔接

| 本模块主题 | 前置章节 | 衔接说明 |
|---|---|---|
| MFCC/STFT 的 FFT 与窗函数 | [00-数学基础](00-数学基础/README.md) | 傅里叶变换/线性代数打底，这边落到信号处理与特征 |
| GMM-EM / HMM 三大问题 | [02-机器学习 11-HMM](02-机器学习/推导笔记/11-HMM隐马尔可夫模型.ipynb) | 那边讲概率模型与动态规划算法，这边套到语音声学建模 |
| CTC 序列对齐 / 梯度反传 | [02-ML 12-CRF](02-机器学习/推导笔记/12-CRF与最大熵模型.ipynb) · [03-DL BP](03-深度学习/模型笔记/01-多层感知机与反向传播.ipynb) | 序列对齐/前向后向思想与 BPTT 手推复用 |
| Attention / Transformer 编码器 | [03-深度学习 11-Transformer](03-深度学习/模型笔记/11-Attention与Transformer从零实现.ipynb) | 端到端 ASR 与 attention TTS 的骨架直接复用 |
| 神经增强 / 掩码网络 | [03-深度学习 MLP/反向传播](03-深度学习/模型笔记/01-多层感知机与反向传播.ipynb) | 05 的 MLP 神经掩码用 numpy 全反传，与 DL 笔记互证 |
| 说话人 embedding / 度量学习 | [03-深度学习 对比学习](03-深度学习/模型笔记) | 06 的 cosine 相似度与 AAM-Softmax 与度量学习直接相关 |
| VQ / 离散表示 | [04-LLM大模型 token 化](04-LLM大模型/README.md) | 07 的 VQ-VAE、08 的 RVQ 音频 token 与 LLM 分词思想同源 |
| 语音多模态 / 语音 Agent | [04-LLM大模型](04-LLM大模型/README.md) 多模态 · [12-智能体开发 06-语音](12-智能体开发/教学/06-异步事件语音与Computer-Use.ipynb) | 大模型时代语音入口与多模态对齐的落点 |
| 语言模型 / 文本处理 | [10-自然语言处理](10-自然语言处理/README.md) | 声学模型之外的语言模型（LM）与解码融合 |

## 面试高频题型（详见 [高频面试题.md](高频面试题.md)）

- **公式推导类**：MFCC 全流程、CTC 前向-后向、HMM 前向/维特比 DP、DTW 递推、CER/WER、谱减/维纳公式、VQ-VAE 三项损失、循环一致性损失
- **对比选型类**：MFCC vs Fbank；CTC vs Attention vs RNN-T；Tacotron vs FastSpeech vs VITS；谱减 vs 维纳 vs 掩码；VC vs SVC；级联 vs 端到端语音大模型
- **手撕代码类**：分帧加窗 + STFT、MFCC、DTW（动态规划）、CTC 前向后向、Griffin-Lim、CER/WER、F0 自相关提取、VQ 量化、RVQ 残差量化
- **工程排查类**：语音识别错误类型分析、VAD 参数调优、流式 vs 非流式延迟控制、增强因果性约束、声纹防伪、上线掉点定位

## 验收标准

- [ ] 全模块 9 篇 notebook 通过 nbclient 全量执行（无需 GPU、无需联网）：00 信号与特征 / 01 GMM-HMM 与 DTW / 02 端到端 ASR / 03 TTS / 04 面试八股 / 05 增强 / 06 声纹 / 07 转换 / 08 语音大模型
- [ ] 每篇含：问题表 → 直觉 → 公式/原理 → 手写实现 → 数值对照 → 面试考点 → 自测清单
- [ ] 全部代码 numpy/scipy 自包含，随机种子固定，音频示例用合成波形/可复现数据，不依赖外部下载
- [ ] 绘图统一带 `plt.show()`（Jupyter 打开即显示过程图）
