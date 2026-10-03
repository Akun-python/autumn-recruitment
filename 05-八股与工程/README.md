# 🔧 05-八股与工程

> 目的：算法岗面试里"基础不牢"是最大的减分项。本目录采用 **ipynb 教学 notebook（精读）+ md 速查（盲测）** 双轨制：
> - 📓 `*.ipynb`：可运行的交互式教学讲义 —— 讲解 + 可运行代码 + **示意图配图**，Jupyter 打开逐 cell 运行
> - 📄 `*.md`：纯文本速查版 —— 面试前快速过 + 打勾

## 📓 教学 Notebook 索引（精读主路径，打开 Jupyter/VSCode 逐 cell 运行）

| Notebook | 主题 | 配图 | cells | 优先级 |
|----------|------|------|-------|--------|
| [01-Python语言.ipynb](01-Python语言.ipynb) | GIL / 装饰器 / 生成器 / GC / asyncio / 内存管理 | GIL 执行示意 | 94 | 🔴 必背 |
| [02-操作系统.ipynb](02-操作系统.ipynb) | 进程线程协程 / 死锁 / IPC / 虚拟内存 / 调度 | 进程五状态、分页示意 | 76 | 🔴 必背 |
| [03-计算机网络.ipynb](03-计算机网络.ipynb) | TCP 握手挥手 / 可靠传输 / HTTP 演进 / TLS / DNS | 握手时序、挥手时序、拥塞控制曲线、HTTP2 多路复用 | 42 | 🔴 必背 |
| [04-数据库.ipynb](04-数据库.ipynb) | B+ 树 / 聚簇索引 / ACID / 隔离级别 / MVCC / 锁 / **SQL 语言实战** | B+ 树结构、MVCC 版本链 | 65 | 🔴 必背 |
| [05-Redis.ipynb](05-Redis.ipynb) | 数据结构 / 穿透击穿雪崩 / 一致性 / 持久化 / 分布式锁 | 缓存三问题、布隆过滤器 | 41 | 🔴 必背 |
| [06-分布式.ipynb](06-分布式.ipynb) | CAP / 一致性 / MQ / 分布式事务 / 限流熔断 | CAP 三角、一致性哈希环、令牌桶 vs 漏桶 | 64 | 🟡 掌握 |
| [07-工程工具.ipynb](07-工程工具.ipynb) | Git / Linux / pytest / 设计模式 / 性能分析 | Git 分支合并示意 | 59 | 🟡 掌握 |
| [08-高频手撕代码.ipynb](08-高频手撕代码.ipynb) | 装饰器 / 生成器 / 单例 / 线程进程 / LRU / 排序 / Trie / 并查集 / 拓扑 / Dijkstra | LRU 结构示意 | 76 | 🔴 必背 |

> 💡 **配图说明**：`images/` 目录下 16 张示意图均由本仓库用 matplotlib 自绘（中文标注、可复现），避免版权问题；notebook 用相对路径 `images/xxx.png` 引用。
> ⚠️ 互联网图源（Wikimedia/CDN）在当前网络环境不可达，已用自绘示意替代；如需补充真实截图/照片，可自行添加至 `images/` 并在对应 cell 引用。

## 📄 Markdown 速查索引（冲刺盲测用）

| 文件 | 内容 |
|------|------|
| [速查清单.md](速查清单.md) | 打勾盲测清单：🔴 🟡 🟢 分级 + 每项链到 notebook 章节 |
| [01-Python语言.md](01-Python语言.md) | Python 主题速查（notebook 的纯文本版） |
| [02-操作系统.md](02-操作系统.md) | 操作系统主题速查 |
| [03-计算机网络.md](03-计算机网络.md) | 计算机网络主题速查 |
| [04-数据库.md](04-数据库.md) | 数据库主题速查 |
| [05-Redis.md](05-Redis.md) | Redis 主题速查 |
| [06-分布式.md](06-分布式.md) | 分布式主题速查 |
| [07-工程工具.md](07-工程工具.md) | 工程工具主题速查 |
| [08-高频手撕代码.md](08-高频手撕代码.md) | 手撕代码纯文本版 |

## 🖼️ 配图目录（images/）

| 图 | 对应主题 | 图 | 对应主题 |
|----|----------|----|----------|
| python_gil.png | GIL 多线程 | dist_cap.png | CAP 三角 |
| os_process_states.png | 进程五状态 | dist_consistent_hash.png | 一致性哈希环 |
| os_paging.png | 虚拟内存分页 | dist_token_bucket.png | 令牌桶 vs 漏桶 |
| net_tcp_handshake.png | TCP 三次握手 | git_branch.png | Git 分支合并 |
| net_tcp_fourwave.png | TCP 四次挥手 | hand_lru.png | LRU 结构 |
| net_congestion.png | 拥塞控制 cwnd | redis_cache.png | 缓存穿透/击穿/雪崩 |
| net_http2.png | HTTP/2 多路复用 | redis_bloom.png | 布隆过滤器 |
| db_bplus.png | B+ 树 | db_mvcc.png | MVCC 版本链 |

## 复习节奏建议

- **精读轮（第 1 周）**：每天 1 个 notebook，Jupyter 里逐 cell 运行，边跑边画核心图（TCP 状态机、B+ 树、MVCC 版本链）
- **盲测轮（第 2 周）**：用 `速查清单.md` 或 md 速查版逐项盲讲 1 分钟，讲不清的返回 notebook 对应章节标 `⏳` 再攻
- **冲刺轮（面试前 3 天）**：只刷 `08-高频手撕代码.ipynb`，全部代码 5 分钟内默写

## 自测标准

- [ ] 所有 notebook 能从上到下**全部运行通过**（无报错）
- [ ] 清单里 🔴 必背条目能不看资料讲 1 分钟
- [ ] 能手写：装饰器、生成器、单例、LRU、生产者-消费者、快速排序、Trie、并查集
- [ ] 能画出：TCP 三次握手状态迁移、B+ 树结构与回表路径、MVCC ReadView 判定流程
- [ ] 每份 notebook 文末「面试追问」都能答

## 关联模块

- 算法部分见 `../01-数据结构与算法/`（14 篇教学 notebook + 11 份模板）
- 机器学习见 `../02-机器学习/`、深度学习见 `../03-深度学习/`
- LLM 见 `../04-LLM大模型/`（重点模块：Transformer 从零手推 24 讲）
- 面试复盘见 `../06-面试复盘/`