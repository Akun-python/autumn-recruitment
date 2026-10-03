# Redis 八股与工程（秋招速通版）

## 核心概念

### (a) 🗂️ Redis 八股全景：从数据结构到高可用的一条主线

Redis 面试题有一条清晰的进阶路线：先懂「为什么快」与五种数据结构，再进入缓存场景（穿透 / 击穿 / 雪崩、一致性），然后是持久化与淘汰策略，最后是分布式锁与高可用（主从 / 哨兵 / 集群）。下图把九大主题串成一张地图。

（该图在 GitHub/VSCode + mermaid 插件中可渲染）

```mermaid
flowchart TB
    Redis["🚀 Redis 八股全景"]
    Redis --> DS["📦 五种数据结构 + 单线程为何快"]
    Redis --> Cache["🧩 缓存三大问题<br/>穿透 / 击穿 / 雪崩"]
    Cache --> Bloom["🌸 布隆过滤器"]
    Redis --> Cons["🔄 缓存与数据库一致性"]
    Redis --> Pers["💾 持久化：RDB vs AOF"]
    Redis --> Exp["⏰ 过期删除与内存淘汰"]
    Redis --> Lock["🔒 分布式锁（SETNX + Lua）"]
    Redis --> HA["🏥 高可用与治理<br/>主从 / Sentinel / Cluster / 大 Key"]
```

- **📦 五种数据结构 + 单线程为何快**：String（计数器 / 缓存）、List（消息队列）、Hash（对象）、Set（去重 / 共同关注）、ZSet（排行榜）各有底层实现。单线程快四问：纯内存访问、高效数据结构、IO 多路复用、避免上下文切换与锁竞争——但单线程也意味着一个慢命令会阻塞全局。
- **🧩 缓存三大问题**：穿透（查不存在的 key 打穿到 DB）用空值缓存 / 布隆过滤器挡；击穿（热点 key 过期瞬间并发打 DB）用互斥锁 / 逻辑过期；雪崩（大量 key 同时过期）用随机过期时间 / 集群分散。三者的定义与方案对比表要能默写。
- **🌸 布隆过滤器**：用多个哈希函数把 key 映射到位数组，能判断「一定不存在 / 可能存在」，以极小内存挡住穿透。要会讲误判率公式与「不能删除」的局限（可升级为计数布隆过滤器）。
- **🔄 缓存与数据库一致性**：主流是 Cache-Aside（旁路缓存）：读先查缓存、写先更 DB 再删缓存。难点在并发时序（先更 DB 还是先删缓存），常用延迟双删 / 订阅 binlog 异步更新兜底，接受「最终一致」而非强一致。
- **💾 持久化：RDB vs AOF**：RDB 是定期全量快照、恢复快但可能丢数据；AOF 记写命令，appendfsync 三种策略在性能与安全间取舍，重写（Rewrite）压缩体积。Redis 4.0+ 混合持久化 = RDB 快照 + 增量 AOF，兼顾两者。
- **⏰ 过期删除与内存淘汰**：删除策略是惰性删除 + 定期删除（主动被动结合）；内存淘汰 8 种策略里重点背 noeviction / allkeys-lru / volatile-lru / allkeys-random。注意 Redis 的 LRU 是采样近似实现，不是严格 LRU。
- **🔒 分布式锁（SETNX + Lua）**：分布式锁要满足互斥、防死锁（过期时间）、防误删（value 带唯一标识，Lua 脚本原子校验再删）、可重入 / 续期（看门狗）。Redlock 是官方多节点方案但学界有争议，能讲出利弊即可。
- **🏥 高可用与治理**：主从复制保证数据冗余（全量 + 增量同步）；Sentinel 哨兵负责监控、选主、通知，实现故障自动切换；Cluster 用 16384 个槽位分片，节点增减走 MOVED / ASK 重定向。还要会答大 Key / 热点 Key 的危害与 SCAN 渐进处理，以及 Redis vs Memcached 的选型对比（多数据结构、持久化、集群是 Redis 的三张王牌）。

> 面向**算法 / 后端秋招**的 Redis 高频考点整理。每个主题包含：**定义与原理讲解**、
> **公式或伪代码**、**可直接运行的 Python 3 示例**、**对比表格**、**面试追问与标准回答**、
> **易错点 / 面试点评**。涉及 redis-py 的示例需本机装有 Redis 服务，纯算法示例（布隆过滤器、
> LRU 等）可零依赖直接运行。

**本文目录**
1. Redis 基础：单线程为什么快 + 五种基本数据结构
2. 缓存穿透 / 击穿 / 雪崩（含布隆过滤器 Python 实现）
3. 缓存与数据库一致性（延时双删 / CDC / Cache-Aside）
4. 持久化：RDB vs AOF vs 混合持久化
5. 过期策略与淘汰策略（8 种 + LRU 近似实现）
6. 分布式锁（SETNX + Lua + Redlock + 看门狗）
7. 高可用：主从复制 / Sentinel / Cluster
8. 大 Key 与热点 Key 问题
9. Redis vs Memcached 对比

---

## 1. Redis 基础：单线程为什么快 + 五种基本数据结构

### 1.1 定义

**Redis（Remote Dictionary Server）** 是一个基于内存的 **Key-Value 数据库**，支持丰富的数据结构、持久化、主从复制、哨兵与集群，广泛应用于**缓存、分布式锁、排行榜、消息队列**等场景。

### 1.2 单线程为什么快（经典四问）

| 原因 | 说明 |
| --- | --- |
| **纯内存操作** | 读写内存（纳秒级），没有磁盘 IO 随机访问的瓶颈 |
| **IO 多路复用** | 用 **epoll** 单线程同时监听海量 socket，事件就绪才回调处理，无阻塞等待 |
| **无锁、无上下文切换** | 单线程执行命令，天然避免**加锁/解锁、线程切换、CPU 缓存失效**的开销 |
| **高效数据结构** | 底层用 SDS、跳表、压缩列表等精心设计的数据结构，命令复杂度多为 O(1)/O(log N) |

> 注意：Redis 的「单线程」指**执行命令的线程只有一个**（6.0 前整个网络模型单线程；6.0 后引入多线程 IO 读写，但**命令执行仍单线程**），持久化、过期淘汰等由后台线程完成。

### 1.3 五种基本数据结构对照表

| 结构 | 命令示例 | 典型应用 | 底层实现 |
| --- | --- | --- | --- |
| **String** | `SET / GET / INCR / SETNX / MSET` | 缓存、计数器、分布式锁、限流 | **SDS（简单动态字符串）**、int/embstr/raw 编码 |
| **List** | `LPUSH / RPOP / LRANGE / BLPOP` | 消息队列、最新列表、任务队列 | **quicklist**（ziplist 节点链表） |
| **Hash** | `HSET / HGET / HINCRBY / HGETALL` | 对象缓存（用户信息、购物车） | **listpack/ziplist** 或 **dict（哈希表）** |
| **Set** | `SADD / SISMEMBER / SPOP / SUNION` | 去重、共同关注、随机抽奖 | **intset** 或 **dict（值为 NULL 的哈希表）** |
| **ZSet** | `ZADD / ZRANGE / ZSCORE / ZINCRBY` | 排行榜、延时队列、滑动窗口限流 | **skiplist + dict** 或 **listpack** |

### 1.4 底层实现要点

- **SDS**：C 字符串的改进版——**记录长度、自动扩容、二进制安全（可存 \0）**，避免 `strlen` O(n) 与缓冲区溢出。
- **quicklist**：双向链表，每个节点是一个 **ziplist/listpack**（压缩列表），兼顾内存紧凑与首尾操作高效。
- **skiplist + dict**：跳表维护**有序**，dict 维护 **member→score** 映射，保证 `ZSCORE` 也是 O(1)。
- **编码转换（以小哈希为例）**：元素少且值小时用 **listpack（ziplist）**，超过阈值（如 512 个或 64 字节）自动转为 **哈希表**。

### 1.5 Python 3 可运行示例（redis-py 基本操作）

```python
"""redis-py 基本操作示例。需先启动 redis-server（默认 6379 端口）。
pip install redis 后可直接运行。"""
try:
    import redis
except ImportError:
    print("请先安装: pip install redis")
    raise SystemExit

r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

# String：计数器（INCR 原子自增）
r.set('counter', 0)
print('INCR ->', r.incr('counter'), r.incr('counter'))

# List：消息队列（LPUSH + BRPOP）
r.rpush('mq', 'job1', 'job2')
print('LPOP ->', r.lpop('mq'))

# Hash：对象缓存
r.hset('user:1001', mapping={'name': 'alice', 'age': 18})
print('HGET ->', r.hget('user:1001', 'name'))

# Set：去重 / 共同关注
r.sadd('follow:a', 'u1', 'u2'); r.sadd('follow:b', 'u2', 'u3')
print('共同关注 ->', r.sinter('follow:a', 'follow:b'))

# ZSet：排行榜（分数从高到低）
r.zadd('rank', {'alice': 90, 'bob': 80, 'carol': 95})
print('Top2 ->', r.zrevrange('rank', 0, 1, withscores=True))
```

### 1.6 面试追问

**追问 1：Redis 6.0 引入多线程 IO，还是单线程吗？**
**标准回答：** 是的——**命令执行依然是单线程**（保证原子性与无锁）；多线程只用于**socket 读写的编解码**，把耗时 IO 从主线程剥离，网络吞吐提升，但命令语义不变。

**追问 2：单线程执行命令会不会太慢？**
**标准回答：** 单命令都是微秒级，瓶颈通常在**网络 IO 和内存**；真正要防的是 **O(N) 危险命令**（`KEYS *`、大 Key 的 `HGETALL`、`SMEMBERS`），它们会阻塞主线程（「慢查询」）。

**追问 3：String 和 Hash 存对象怎么选？**
**标准回答：** 整体读写用 String（JSON 序列化）简单高效；需要**频繁改单个字段**（如用户积分、购物车数量）用 **Hash**（`HINCRBY` 原子且只传一个字段，省带宽省内存）。

> ⚠️ **易错点**：**ZSet 底层是跳表 + dict 两套结构**；「跳表为什么不用红黑树」是高频追问——跳表**实现简单、支持范围查询、区间插入删除 O(logN)**，且与 dict 共享 member 无冗余内存问题。

---

## 2. 缓存穿透 / 击穿 / 雪崩

### 2.1 定义

| 问题 | 定义 | 特征 |
| --- | --- | --- |
| **缓存穿透** | 查询**根本不存在**的 key，缓存和数据库都没有，每次请求都打到 DB | 恶意攻击 / 非法参数，DB 压力大且缓存永远防不住 |
| **缓存击穿** | 一个**热点 key 恰好过期**，瞬间大量请求同时穿透到 DB | 单个 key 过期瞬间，DB 被打爆 |
| **缓存雪崩** | **大量 key 同时过期**（或 Redis 宕机），海量请求直击 DB | 整片缓存失效，DB 雪崩 |

**区别一句话**：穿透 = **查不到**（缓存永远 Miss）；击穿 = **热点 key 过期**（单个）；雪崩 = **大量 key 同时过期 / 服务不可用**（批量）。

### 2.2 解决方案对比表

| 问题 | 方案 | 说明 |
| --- | --- | --- |
| 穿透 | **布隆过滤器** | 请求前先判「可能存在」，过滤掉不存在 key |
| 穿透 | 缓存空值 + 短过期 | 不存在也缓存 `null`（TTL 短，如 5 分钟），并加业务兜底 |
| 穿透 | 参数校验 / 限流 | 非法参数直接拒绝（如 id < 0） |
| 击穿 | **互斥锁（Mutex）** | 只有一个线程去查 DB 重建缓存，其余等待/降级 |
| 击穿 | **逻辑过期** | key 永不过期，逻辑时间过期后异步重建，请求先返回旧值 |
| 雪崩 | **随机过期时间** | 过期时间加随机抖动（如 ±随机数），错峰失效 |
| 雪崩 | 多级缓存 / 熔断降级 | 本地缓存兜底、限流、开关降级 |

### 2.3 布隆过滤器原理与公式

- 原理：一个 **m 位的位数组** + **k 个哈希函数**。插入时把元素经 k 个哈希映射到 k 个位并置 1；查询时若 **k 个位全为 1 则「可能存在」**，否则**必然不存在**。
- 特点：**不会漏报（假阴性为 0），但可能误报（假阳性）**；不可删除（可用计数布隆过滤器）。
- 公式（给定期望元素数 n 与误判率 p）：

```text
位数组长度 m = - n·ln(p) / (ln2)^2
哈希函数个数 k = (m/n) · ln2
实际误判率 p' = (1 - e^(-k·n/m))^k
```

### 2.4 Python 3 可运行示例（布隆过滤器实现）

```python
"""纯 Python 布隆过滤器（标准库，可直接运行）。"""
import hashlib, math

class BloomFilter:
    def __init__(self, n: int, p: float = 0.01):
        self.m = int(-n * math.log(p) / (math.log(2) ** 2))   # 位数组长度
        self.k = max(1, int(self.m / n * math.log(2)))        # 哈希函数个数
        self.bits = [0] * self.m

    def _hashes(self, item: str):
        # 用 md5/sha1 双摘要派生 k 个相互独立的哈希值
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        return [(h1 + i * h2 + i * i) % self.m for i in range(self.k)]

    def add(self, item: str):
        for idx in self._hashes(item):
            self.bits[idx] = 1

    def contains(self, item: str) -> bool:
        return all(self.bits[idx] for idx in self._hashes(item))

if __name__ == '__main__':
    bf = BloomFilter(n=1000, p=0.01)
    for i in range(1000):
        bf.add(f"user:{i}")
    print("已插入 1000 个 key")
    print("user:500 存在? ->", bf.contains("user:500"))    # True（必然）
    print("user:9999 存在? ->", bf.contains("user:9999"))  # False（大概率）
    # 统计误判率
    false_positive = sum(bf.contains(f"miss:{i}") for i in range(1000))
    print(f"1000 个不存在 key 的误判数 ≈ {false_positive}（理论 ~1%）")
```

### 2.5 互斥锁（击穿）与逻辑过期（伪代码）

```python
# —— 方案 A：互斥锁（SETNX 抢锁，只有一个线程回源重建）——
# key = 热点缓存key; lock_key = key + ':lock'
"""
value = GET(key)
if value is None:                      # 缓存未命中（过期）
    if SETNX(lock_key, 1, NX=True, EX=10):   # 抢到锁
        try:
            value = query_db(key)      # 只有一个线程查 DB
            SETEX(key, 300, value)     # 重建缓存
        finally:
            DEL(lock_key)
    else:
        sleep(20ms); return GET(key)   # 没抢到：短暂等待后重试/降级
"""
# —— 方案 B：逻辑过期（key 永不过期，异步重建）——
"""
SET(key, value)  # 永不过期；value 中内嵌逻辑过期时间
read 时：
    判断 value.logic_expire < now ？
        是：返回旧 value，同时后台线程抢锁重建（防止击穿）
        否：直接返回 value
"""
```

### 2.6 面试追问

**追问 1：布隆过滤器能删除元素吗？**
**标准回答：** 标准布隆过滤器**不能删除**（删除一个位可能影响其他元素）；可用**计数布隆过滤器（Counting Bloom Filter）**，但计数溢出会退化。Redis 官方 **RedisBloom** 模块的 `BF.INSERT / BF.EXISTS` 可直接使用。

**追问 2：互斥锁方案中，没抢到锁的请求怎么办？**
**标准回答：** 短暂自旋重试读缓存，或**返回旧数据/降级数据**（逻辑过期方案天然支持返回旧值），避免大量请求空转打爆 DB。

**追问 3：穿透和击穿如何同时防？**
**标准回答：** 入口布隆过滤器**防穿透**（过滤不存在 key）+ 热点 key 互斥锁/逻辑过期**防击穿** + 过期时间随机化**防雪崩**，三层组合。

> ⚠️ **易错点**：**缓存空值**只能防「反复查询同一个不存在 key」的穿透；攻击者每次用**不同随机 key** 时，空值缓存会占满内存，必须配合布隆过滤器。

---

## 3. 缓存与数据库一致性

### 3.1 问题定义

缓存与数据库是**两套存储**，更新时序不同会产生**不一致**：缓存是旧值、DB 是新值，或反之。核心矛盾：**先更新谁、谁先失效/生效**。

### 3.2 Cache-Aside（旁路缓存）模式

- **读**：先查缓存，命中直接返回；未命中查 DB，回填缓存。
- **写**：**先更新 DB，再删除缓存（del）**；下一次读未命中再回填。

> 为什么写用「删缓存」而不是「更新缓存」？更新缓存需要知道新值且易与并发写冲突；**删除**是惰性的——下次读时回填，实现简单且最终一致。

### 3.3 时序问题与对策

| 方案 | 做法 | 评价 |
| --- | --- | --- |
| **先删缓存，再更新 DB** | 删除 → 写库 | 删除后、写库前有读会回填旧值 → **不一致**，已基本弃用 |
| **先更新 DB，再删缓存** | 写库 → 删除 | 主流方案；仍有「读旧值回填」窗口，用**延时双删**缓解 |
| **延时双删（Delayed Double Delete）** | 写库 → 删缓存 → **延时 500ms~1s 再删一次** | 消除回填旧值的窗口；代价是短暂不一致可接受 |
| **binlog / CDC 订阅** | 订阅 DB binlog（Canal），异步删缓存/更新缓存 | **解耦、可靠**，删缓存失败可重试；最终一致性强 |
| **消息队列 + 重试** | 删除失败写入 MQ 重试 | 保证「删除」不丢 |

**延时双删（Python 伪代码 + redis-py）：**

```python
"""延时双删：先更新 DB，再删缓存；短暂延时后二次删除。"""
import redis, time

r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

def update_user(user_id, new_data):
    # 1. 更新数据库（示意）
    update_db(user_id, new_data)            # UPDATE user SET ... WHERE id=?
    # 2. 第一次删缓存
    r.delete(f"user:{user_id}")
    # 3. 延时（等可能的「读旧值→回填」完成）
    time.sleep(0.5)
    # 4. 第二次删缓存（兜底）
    r.delete(f"user:{user_id}")

def update_db(user_id, data):
    print(f"[DB] UPDATE user {user_id} -> {data}")
```

### 3.4 CAP 权衡说明

- **CAP**：分布式系统在**一致性（C）/ 可用性（A）/ 分区容错性（P）** 中最多满足两项；网络分区不可避免 → **必须选 P**，再在 C 与 A 之间权衡。
- 缓存与 DB 是**两副本**，天然存在**最终一致性（Eventually Consistent）**而非强一致：
  - 追求强一致：**读 DB 前先失效缓存 + 串行化**，成本高、性能差，缓存场景通常不选；
  - 缓存场景的默认选择：**AP（可用性优先）+ 最终一致**——短暂读到旧值可接受（如 500ms 内）；
  - 需要近似强一致时用 **Cache-Aside + 延时双删 / binlog 订阅 + 失败重试**，把不一致窗口压到最小。

### 3.5 面试追问

**追问 1：为什么「先删缓存再更新库」不好？**
**标准回答：** 删缓存后、写库前的空窗期内，若有读请求命中 DB 旧值并回填缓存，缓存又变成旧值且可能长期不更新（直到下次过期），不一致窗口大且难以收敛。

**追问 2：延时双删的「延时」怎么选？**
**标准回答：** 取「读请求从 DB 读旧值到回填缓存」的最坏耗时（一般数百毫秒到 1s），太大则短暂不一致过久；工程上通常 500ms 并配合**第二次删除失败重试**。

**追问 3：为什么不用「先更新缓存」？**
**标准回答：** 并发写时「更新缓存」的顺序与写库顺序可能不一致（写库乱序导致缓存覆盖成旧值），且每次写都写缓存浪费 IO；**删除缓存 + 惰性回填**天然规避这两个问题。

> ⚠️ **易错点**：**删除缓存可能失败**（网络抖动），这是不一致的主要来源——务必用 **binlog 订阅 / MQ 重试**兜底；「延时双删」不是强一致方案，它只是把不一致窗口缩小到工程可接受。

---

## 4. 持久化：RDB vs AOF

### 4.1 定义

- **RDB（快照）**：把**某一时刻的全量数据**以二进制格式写入 `.rdb` 文件，恢复快、体积小。
- **AOF（追加日志）**：把**每一条写命令**追加到 `.aof` 文件，恢复慢但**丢失数据更少**（可配 fsync）。

### 4.2 RDB 触发方式

- **手动**：`SAVE`（阻塞主线程，生产禁用）/ `BGSAVE`（fork 子进程写快照，不阻塞）；
- **自动**：配置文件 `save <seconds> <changes>`，如 `save 900 1`（900 秒内 ≥1 次修改）、`save 300 10`、`save 60 10000`；
- **关闭时**：正常 `SHUTDOWN` 会生成 RDB；`FLUSHALL` 等不影响已有文件。

### 4.3 AOF 三种 fsync 策略

| 策略 | 行为 | 数据安全 | 性能 |
| --- | --- | --- | --- |
| `always` | **每条写命令**都 fsync | 最多丢 1 条命令 | 最慢 |
| `everysec`（默认） | 每秒 fsync 一次 | 最多丢 **1 秒**数据 | 折中 |
| `no` | 交给操作系统刷盘 | 可能丢较多数据 | 最快 |

### 4.4 AOF 重写（Rewrite）

- 问题：AOF 无限增长，恢复慢、占磁盘。
- 方案：**AOF 重写**——根据**当前内存数据集**生成最小命令集写入新 AOF（如 `INCR` 100 次合并成 `SET 100`），由**子进程**执行，父进程把重写期间的增量命令缓冲并追加到新文件，保证不丢数据。
- 触发：手动 `BGREWRITEAOF`；自动按 `auto-aof-rewrite-percentage` / `auto-aof-rewrite-min-size`。

### 4.5 RDB vs AOF 对比表

| 维度 | **RDB** | **AOF** |
| --- | --- | --- |
| 数据格式 | 二进制快照 | 追加写命令（文本/二进制协议） |
| 数据丢失 | 上次快照后的修改全丢（取决于 save 策略） | everysec 最多丢 1s；always 不丢 |
| 恢复速度 | **快**（直接加载） | 慢（重放所有命令） |
| 文件大小 | 小（压缩快照） | 大（可重写控制） |
| 对性能影响 | BGSAVE fork 有短暂内存/CPU 开销 | 写放大，always 策略影响大 |
| 适用场景 | 冷备、快速恢复、灾备 | 数据安全要求高的缓存 |

### 4.6 混合持久化（Redis 4.0+）

- **方案**：AOF 重写时生成 **RDB 格式的「基础快照」** + 快照之后的新命令追加到 AOF 尾部；
- **收益**：加载时**先秒级加载 RDB**，再重放少量增量命令——兼顾**恢复速度**与**数据安全**；
- 配置：`aof-use-rdb-preamble yes`。

### 4.7 面试追问

**追问 1：BGSAVE 会阻塞主线程吗？**
**标准回答：** 磁盘写入由**子进程**完成不阻塞；但 **fork 本身**可能短暂阻塞（COW 复制页表，内存越大耗时越长），且 **COW 写时复制**会带来额外内存开销——大内存实例要关注。

**追问 2：RDB 和 AOF 能同时开吗？**
**标准回答：** 可以（生产常用组合）。**加载优先级 AOF 高于 RDB**（AOF 数据更全）；开启混合持久化后加载更快。

**追问 3：为什么 Redis 重启后优先用 AOF 恢复？**
**标准回答：** 因为 AOF 记录的命令比 RDB 快照更**接近最新状态**，数据丢失更少；加载完成后再由 RDB/混合文件提供性能。

> ⚠️ **易错点**：`SAVE`（同步阻塞）与 `BGSAVE`（异步）别混淆；**fork 子进程是「写时复制」**——子进程快照期间主进程的修改不会进 RDB，而是依赖 AOF/下次快照。

---

## 5. 过期策略与淘汰策略

### 5.1 过期删除策略（删「已过期」的 key）

| 策略 | 原理 | 缺点 |
| --- | --- | --- |
| **惰性删除** | 访问 key 时才检查是否过期，过期则删 | 过期 key 可能长期占用内存（不被访问就一直在） |
| **定期删除** | 后台定时**随机抽一批**带过期时间的 key 检查删除 | 抽样不全面，仍有残留 |
| 组合（Redis 实际采用） | **惰性删除 + 定期删除**：被访问时惰性删；后台周期抽样删 | 两者都漏掉的过期 key 靠**内存淘汰**兜底 |

### 5.2 内存淘汰策略（内存不够时「驱逐」key，8 种）

| 策略 | 作用域 | 行为 |
| --- | --- | --- |
| `noeviction`（默认） | — | 不淘汰，写命令直接报错 |
| `allkeys-lru` | **所有 key** | 淘汰最近最少使用 |
| `volatile-lru` | **仅设置了过期时间的 key** | 淘汰最近最少使用 |
| `allkeys-lfu` | 所有 key | 淘汰最不经常使用（频次） |
| `volatile-lfu` | 仅有过期时间的 key | 淘汰最不经常使用 |
| `allkeys-random` | 所有 key | 随机淘汰 |
| `volatile-random` | 仅有过期时间的 key | 随机淘汰 |
| `volatile-ttl` | 仅有过期时间的 key | 淘汰**剩余 TTL 最小**的 key |

> 口诀：**LRU/LFU/Random/TTL × (allkeys/volatile) + noeviction = 8 种**；volatile 系策略若没有过期 key 则退化为 noeviction。

### 5.3 LRU 近似实现原理

- 真实 LRU 需要双向链表 + 哈希表，代价高；
- Redis 的 **近似 LRU**：每个 key 记录 **24 位 LRU 时钟（lru 字段）**，淘汰时**随机采样 N 个（默认 5）**，驱逐其中 **lru 时间最早**的 key，并维护一个**候选池**（pool）提升淘汰质量；
- 改进：**LFU**（Redis 4.0+）用对数衰减的访问频次计数器，抵抗「偶尔访问的热点被 LRU 误淘汰」。

### 5.4 Python 3 可运行示例（LRU 缓存实现 + 近似 LRU 采样）

```python
"""标准库实现 LRU 缓存 + 近似 LRU 采样（可直接运行）。"""
from collections import OrderedDict
import random

class LRUCache:
    """O(1) 的精确 LRU：OrderedDict 移动元素到末尾实现。"""
    def __init__(self, capacity: int):
        self.cap = capacity
        self.d = OrderedDict()

    def get(self, key):
        if key not in self.d:
            return -1
        self.d.move_to_end(key)          # 最近使用 -> 末尾
        return self.d[key]

    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        self.d[key] = value
        if len(self.d) > self.cap:
            self.d.popitem(last=False)   # 淘汰最久未使用（头部）

def redis_approx_lru(sample_size=5):
    """模拟 Redis 近似 LRU：随机采样 + 淘汰最早访问的 key。"""
    pool = {f"k{i}": random.randint(0, 100) for i in range(50)}  # lru 时钟越小越旧
    sample = random.sample(list(pool.items()), sample_size)
    victim = min(sample, key=lambda kv: kv[1])
    print(f"采样 {sample_size} 个: {sorted(sample, key=lambda kv: kv[1])[:3]} ...")
    print(f"淘汰最久未使用: {victim[0]} (lru={victim[1]})")
    return victim[0]

if __name__ == '__main__':
    c = LRUCache(2)
    c.put('a', 1); c.put('b', 2); c.get('a'); c.put('c', 3)
    print("精确 LRU 淘汰 ->", c.d)      # {'b':2, 'c':3}，a 保留
    print("近似 LRU 采样淘汰 ->", redis_approx_lru())
```

### 5.5 面试追问

**追问 1：为什么 Redis 不用精确 LRU？**
**标准回答：** 精确 LRU 需要维护**双向链表 + 哈希表**且每次访问都要移动节点（指针操作多、内存占用大）；**近似 LRU + 采样**在内存与淘汰质量之间取得平衡，实测与精确 LRU 差距很小。

**追问 2：volatile-lru 和 allkeys-lru 怎么选？**
**标准回答：** 业务 key 都设过期时间用 volatile-lru（保留长期驻留的关键数据）；缓存场景（都可淘汰）用 **allkeys-lru**；注意 volatile 系在「无过期 key」时退化为 noeviction，可能触发写报错。

**追问 3：LFU 相比 LRU 好在哪里？**
**标准回答：** LFU 统计**访问频次**，不会被「一次偶然访问」误提升热度；适合**低频热点常驻**场景（如接口限流计数）。代价是维护频次计数器，实现更复杂。

> ⚠️ **易错点**：**过期删除 ≠ 内存淘汰**——前者删「已过期」的 key（惰性+定期），后者在**内存满**时驱逐「未过期」的 key（8 种策略）；两个机制**同时存在**，面试常混淆。

---

## 6. 分布式锁

### 6.1 需求与 SETNX 原子性问题

- 场景：多实例/多进程互斥访问共享资源，单机锁（`threading.Lock`）无法跨进程。
- 朴素方案：`SETNX lock_key value` 抢锁 + `EXPIRE lock_key 30` 设过期。

```text
问题 1（非原子）：
SETNX 成功 → 进程崩溃 → 没设 EXPIRE → 锁永不释放（死锁）
问题 2（误删）：
A 持锁超时释放后 B 拿到锁；A 的延迟 DEL 把 B 的锁删掉
```

### 6.2 官方推荐方案（原子 Lua 脚本）

- **加锁**：`SET lock value NX EX 30`（Redis 2.6.12+ 一条命令原子完成「NX + EX」）；
- **释放锁**：用 **Lua 脚本**「先校验 value（唯一标识，防误删），再 DEL」，保证检查与删除的原子性：

```lua
-- 释放分布式锁（官方推荐写法）
-- KEYS[1] = 锁的 key, ARGV[1] = 持有者唯一标识（如 UUID）
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
```

### 6.3 Python 3 可运行示例（redis-py + Lua 释放锁）

```python
"""分布式锁完整示例（需 redis-server，可直接运行）。
加锁: SET key value NX EX；释放: Lua 校验唯一标识后 DEL。"""
import redis, uuid, time

r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

UNLOCK_LUA = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
else
    return 0
end
"""

def acquire(lock_key, token, ttl=10):
    return r.set(lock_key, token, nx=True, ex=ttl)   # NX + EX 原子

def release(lock_key, token):
    return r.eval(UNLOCK_LUA, 1, lock_key, token) == 1

if __name__ == '__main__':
    lock_key, token = 'order:123:lock', str(uuid.uuid4())
    if acquire(lock_key, token):
        try:
            print("抢锁成功，执行临界区业务...")
            time.sleep(0.5)
        finally:
            ok = release(lock_key, token)      # 校验 token 后删除
            print("释放锁成功" if ok else "锁已被他人持有/过期")
    else:
        print("抢锁失败，其他线程持有锁")
```

### 6.4 看门狗（Watchdog）续期

- 问题：业务执行时间超过锁 TTL → 锁提前过期，其他线程抢锁 → **并发执行**。
- 方案：**看门狗**——持有锁期间后台线程周期性（如每 TTL/3）对锁 **续期**，业务结束释放时取消续期；类比 **Redisson** 的 `watchdog`（默认锁 30s，每 10s 续期）。
- 注意：看门狗续期也是「续命」，**进程宕机后锁仍会过期**（避免死锁）——这是分布式锁「自动失效」特性的价值。

### 6.5 Redlock 思想与争议

- **思想**（Redis 作者提出）：在 **N 个独立 Redis 节点**（通常 5）上都尝试加锁，**大多数（N/2+1）成功且总耗时 < TTL** 才算加锁成功；释放时对所有节点释放。
- **目的**：避免单点 master 故障丢锁后锁失效。
- **争议**（Martin Kleppmann 批评）：
  - 依赖**时钟**：节点 GC 停顿/时钟跳变会导致锁提前失效或被误判；
  - 加锁过程本身**不是原子的跨节点事务**，存在「加了一半锁进程暂停」的窗口；
  - 业界共识：**单 Redis + Lua + 看门狗**对绝大多数场景足够；追求强一致应改用 **ZooKeeper（临时节点 + 顺序节点）/ etcd（租约）**。

### 6.6 面试追问

**追问 1：为什么释放锁要用 Lua？**
**标准回答：** 「get 校验 + del 删除」两步若分开执行，可能在校验后、删除前锁被其他线程替换，导致**误删他人锁**；Lua 脚本在 Redis 中**原子执行**，杜绝竞态。

**追问 2：锁过期但业务没执行完怎么办？**
**标准回答：** 用**看门狗续期**（Redisson 默认每 10s 续 30s 锁），业务结束主动释放；进程崩溃则锁自然过期，不会死锁——「续期」与「自动过期」互补。

**追问 3：Redlock 可靠吗？生产怎么选？**
**标准回答：** Redlock 在**时钟漂移、GC 停顿**场景有理论缺陷，且实现重；绝大多数缓存场景**单实例 + Lua + 看门狗**足够；**金融/强一致**场景选 **ZooKeeper / etcd** 的临时顺序节点实现。

> ⚠️ **易错点**：**value 必须是唯一标识（UUID/线程号）**，删除前必须校验——否则会误删他人的锁；「SETNX + EXPIRE 两条命令」是经典**非原子**反例，面试必考。

---

## 7. 高可用：主从复制 / Sentinel / Cluster

### 7.1 主从复制（Replication）

- **全量复制**：从节点首次连接 → 主节点 `BGSAVE` 生成 RDB → 传输 RDB → 从节点加载；期间增量命令写入 **repl backlog（复制积压缓冲区）** 后补发。
- **增量复制**：断线重连后，若断点仍在 **repl_backlog** 中 → 只补发断点后的命令（`PSYNC` 续传），否则退化为全量复制。
- 角色：一主多从，**从节点只读**（默认），写命令只在主节点执行。

### 7.2 哨兵（Sentinel）

- 作用：**监控**（心跳检测主从）、**自动故障转移**（主节点客观下线 → 选举新主并让从节点切换）、**通知**（向客户端提供当前主节点地址）。
- 判定：主观下线（单哨兵认为失联）→ **客观下线**（quorum 个哨兵都认为失联）→ **选举新主**（Raft 选 Leader 哨兵执行切换）。
- 部署：**哨兵至少 3 个**（奇数，避免脑裂），哨兵之间用 Raft 选主。

### 7.3 Cluster 集群与槽位

- 概念：**16384 个哈希槽（slot）**，分布在 N 个节点上（如 3 主 3 从，每主约 5461 槽）。
- 键分配公式：`slot = CRC16(key) % 16384`；带 `{}` 的 **hash tag** 可把多个 key 归入同一槽（如 `{user:1}:info`、`{user:1}:cart`）。
- 重定向：
  - **MOVED**：key 的槽在**其他节点** → 客户端收到 `MOVED slot ip:port` 后**更新路由缓存**并跳转；
  - **ASK**：槽在**迁移中** → 临时重定向，客户端**不更新缓存**（迁移完成前仍需回原节点确认）。
- 容灾：每个主节点配从节点，主挂从顶上（选举提升）。

### 7.4 Python 3 可运行示例（槽位计算与 MOVED/ASK 处理逻辑）

```python
"""CRC16 槽位计算 + MOVED/ASK 重定向处理逻辑（标准库，可直接运行）。"""
def crc16(data: bytes) -> int:
    poly = 0x1021
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc

def slot_of(key: str) -> int:
    # 支持 hash tag: {user:1} 部分参与 CRC16
    if '{' in key and '}' in key:
        tag = key[key.index('{') + 1: key.index('}')]
        return crc16(tag.encode()) % 16384
    return crc16(key.encode()) % 16384

def handle_reply(reply, client):
    """简化模拟: 处理 MOVED / ASK / OK。"""
    if reply.startswith('MOVED'):
        _, slot, addr = reply.split()
        print(f"槽 {slot} 已迁移到 {addr}，更新本地路由缓存并重试")
        return True
    if reply.startswith('ASK'):
        _, slot, addr = reply.split()
        print(f"槽 {slot} 正在迁移，向 {addr} 发 ASKING 后重试（不更新缓存）")
        return True
    print("命令执行成功:", reply)
    return False

if __name__ == '__main__':
    for key in ['user:1001', 'cart:2002', '{user:1}:info', '{user:1}:cart']:
        print(f"key={key:<16} slot={slot_of(key)}")
    handle_reply('MOVED 1234 10.0.0.2:6379', None)
    handle_reply('ASK 1234 10.0.0.3:6379', None)
```

### 7.5 面试追问

**追问 1：全量复制为什么用 RDB 而不是 AOF？**
**标准回答：** RDB 是**压缩二进制快照**，传输体积小、从节点加载快；AOF 重放命令慢且文件大。同步期间的增量命令由 **repl_backlog** 缓冲补发。

**追问 2：哨兵怎么防止脑裂？**
**标准回答：** 客观下线需要 **quorum**（多数哨兵）确认；新主选举由哨兵 Leader 用 **Raft** 完成；同时主节点可通过 `min-replicas-to-write` 等配置限制「与多数失联的旧主」继续写，减少双写脑裂。

**追问 3：为什么是 16384 个槽，不是 65536？**
**标准回答：** 16384 槽的 **CRC16 取模分布足够均匀**；槽信息以**位图（bitmap）**在节点间传播（2KB 大小），槽再多会增大 Gossip 心跳包；且 Redis 集群节点规模通常在千级以内，16384 粒度已满足（官方设计取舍）。

**追问 4：MOVED 和 ASK 的区别？**
**标准回答：** **MOVED**：槽**已经**属于目标节点，客户端要**更新路由缓存**，后续直接访问新节点；**ASK**：槽**正在迁移**（仅本次），客户端发 **ASKING** 后去目标节点查一次，**不更新缓存**。

> ⚠️ **易错点**：**Cluster 不支持跨槽的多 key 操作**（`MGET` 多个不同槽的 key 报错），除非用 hash tag；**从节点默认不可写**，读写分离要配 `readonly`。

---

## 8. 大 Key 与热点 Key 问题

### 8.1 定义

- **大 Key（Big Key）**：单个 key 的 value 很大——如 **String 超过 10KB~100KB**、集合类 key 元素超过 **万级**（如 10 万个元素的 Hash/Set/ZSet）。
- **热点 Key（Hot Key）**：**单位时间被超高频率访问**的 key（如秒杀商品、明星热搜），单节点 CPU/网卡打满。

### 8.2 危害

| 问题 | 危害 |
| --- | --- |
| **大 Key** | ① 读取/删除/迁移时**阻塞主线程**（O(N) 命令）；② 内存倾斜（集群单节点压力大）；③ 网络传输大（带宽占用）；④ 持久化 fork/COW 开销大 |
| **热点 Key** | ① 单节点 CPU/网卡打满，集群其他节点空闲（**数据倾斜 + 访问倾斜**）；② 缓存击穿时放大 DB 压力；③ 慢查询拖累整体 |

### 8.3 处理方案

**大 Key：**
- **拆分**：把大 Hash/Set 按业务维度拆成多个小 key（`user:1001:info`、`user:1001:tags`）；
- **压缩**：String 存压缩后的数据（如 gzip 的 JSON）；
- **渐进式操作**：删除用 `UNLINK`（异步释放内存，4.0+）或 `SCAN + HDEL/SREM` 分批删；
- **禁用危险命令**：`KEYS *`、大 key 的 `HGETALL`/`SMEMBERS` 改 `HSCAN`/`SSCAN` 分批取。

**热点 Key：**
- **本地缓存（多级缓存）**：JVM/进程内缓存热点数据，先查本地再查 Redis，把热点流量挡在应用侧；
- **热点复制（key 打散）**：把热点 key 复制 N 份（`hot:1`…`hot:N`），请求随机访问副本，分散单节点压力；
- **读写分离 + 集群**：把热点节点上的副本分散到多节点；
- **监控预警**：`redis-cli --hotkeys`（LFU 统计）、info 命令监控，及时拆分。

### 8.4 Python 3 可运行示例（SCAN 渐进删除大 Key）

```python
"""用 SCAN 分批删除大 Hash 的字段（标准库 + redis-py，可直接运行）。"""
import redis

r = redis.Redis(host='127.0.0.1', port=6379, decode_responses=True)

def clear_big_hash(key, batch=1000):
    """分批 HDEL，避免一次性 HGETALL/DEL 阻塞主线程。"""
    cursor, deleted = 0, 0
    while True:
        cursor, fields = r.hscan(key, cursor=cursor, count=batch)
        if fields:
            deleted += r.hdel(key, *fields.keys())
        if cursor == 0:
            break
    print(f"共删除 {deleted} 个字段")
    r.delete(key)          # 空壳 key 直接删
    print("大 Key 清理完成")

if __name__ == '__main__':
    # 构造一个 10000 字段的测试 Hash
    test_key = 'big:hash'
    pipe = r.pipeline()
    for i in range(10000):
        pipe.hset(test_key, f'f{i}', f'v{i}')
    pipe.execute()
    clear_big_hash(test_key)
```

### 8.5 面试追问

**追问 1：为什么大 Key 删除会阻塞？**
**标准回答：** `DEL` 对集合类型需要**逐个释放元素内存**（O(N)），大 Key 会长时间占用主线程；4.0+ 用 **UNLINK 异步释放**，或 SCAN 分批删。

**追问 2：热点 key 在集群里为什么是灾难？**
**标准回答：** 集群按**槽**分布数据，热点 key 只落在**一个槽/一个节点**上——所有请求都打到同一节点，集群水平扩展失效，还伴随**CPU/网卡打满**与**单点故障风险**。

**追问 3：热点 key 怎么提前发现？**
**标准回答：** 业务埋点统计、`redis-cli --hotkeys`（需开 LFU）、监控 QPS 突增告警、代理层（Codis/Twemproxy）统计；发现后**本地缓存 + key 打散**双管齐下。

> ⚠️ **易错点**：**热点 key ≠ 大 key**——前者是**访问频率**问题，后者是**体积**问题；面试要分别给出方案，别混为一谈。

---

## 9. Redis vs Memcached 对比

### 9.1 对比表

| 维度 | **Redis** | **Memcached** |
| --- | --- | --- |
| 数据结构 | **String / List / Hash / Set / ZSet** + 模块 | 仅 **String（value 为字节数组）** |
| 持久化 | **RDB / AOF / 混合** | 无持久化（纯缓存） |
| 线程模型 | 单线程执行命令（IO 多路复用） | **多线程**（每个核多线程处理） |
| 内存淘汰 | 8 种策略（LRU/LFU/Random/TTL） | 仅 **LRU** |
| 原子操作 | INCR/DECR/事务/Lua/发布订阅 | CAS（比较并交换）+ 简单命令 |
| 集群 | Cluster 槽位 + 哨兵 + 主从 | 客户端分片（一致性哈希），无内置集群 |
| 典型场景 | 缓存 + 分布式锁 + 排行榜 + 队列 | 纯 KV 缓存、超大并发读 |
| 内存效率 | 多种编码（int/embstr/ziplist） | 纯字节存储，简单场景内存略省 |

### 9.2 选型建议

- 需要**数据结构丰富、持久化、分布式能力** → **Redis**（绝大多数场景）；
- 只需要**极简 KV 缓存、追求多线程吞吐** → Memcached（如纯对象缓存）；
- 现代实践：**几乎统一用 Redis**（Memcached 已边缘化），但面试仍会考差异点。

### 9.3 面试追问

**追问 1：为什么 Memcached 多线程却不如 Redis 全能？**
**标准回答：** 多线程只提升**网络读写吞吐**，但牺牲了**简单性与原子性**（需要锁保护共享结构）；Redis 单线程换来**命令天然原子、无锁、实现简单**，配合 IO 多路复用在大并发下依然够用，且功能远多于 Memcached。

**追问 2：两者内存满了行为有什么不同？**
**标准回答：** Memcached 只有 **LRU 淘汰**（且可关，关掉后写失败）；Redis 默认 **noeviction**（写报错），可配 8 种策略——**语义更可控**。

**追问 3：什么时候坚持用 Memcached？**
**标准回答：** 纯字符串缓存、对**多线程吞吐极致**要求且**完全不需要持久化/复杂结构**的极简场景；现实中大多被 Redis 替代。

> ⚠️ **易错点**：Memcached **没有内置集群与持久化**——「Memcached 用一致性哈希做分布式」是**客户端分片**，不是服务端能力；Redis 的集群是**服务端原生**支持。

---

## 附：高频速记表（考前 5 分钟）

| 考点 | 一句话答案 |
| --- | --- |
| 单线程为何快 | 内存 + epoll 多路复用 + 无锁 + 高效数据结构 |
| ZSet 底层 | **skiplist + dict**（有序 + 按 member 查 score） |
| 穿透/击穿/雪崩 | 查不到 / 热点过期 / 批量过期，分别用布隆/互斥锁/随机 TTL |
| 布隆过滤器 | m 位数组 + k 哈希，**不漏报、可误报**，m = -n·ln(p)/(ln2)² |
| 一致性 | Cache-Aside + 延时双删 + binlog 订阅，最终一致 |
| RDB vs AOF | 快照快但丢数据多；AOF 慢但最多丢 1s（everysec） |
| 淘汰 8 策略 | LRU/LFU/Random/TTL × allkeys/volatile + noeviction |
| 分布式锁 | **SET key val NX EX + Lua 校验删除 + 看门狗续期** |
| Cluster | **slot = CRC16(key) % 16384**；MOVED 更新缓存，ASK 不更新 |
| 大 Key vs 热点 | 体积大 vs 访问频；拆 key / UNLINK，本地缓存 + 副本打散 |

> 祝顺利上岸！本笔记配套仓库：`autumn-recruit-algo/05-八股与工程/`。
