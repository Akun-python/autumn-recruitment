# 🐍🔧 13-编程语言

> 目标：把「编程语言」讲成一条主线——**Python 语言与数据模型（00）→ C++ 核心语法与 STL（01）→ 机器人操作系统 ROS（02）→ Docker（03）→ Git（04）→ SQL（05）→ Linux 系统（06）→ Shell 脚本（07）→ Go 语言与并发（08）→ Java 核心与 JVM（09）**。
> 面向算法 / 机器人 / 自动驾驶岗位面试与工程实践：不止语法，还讲**语言特性为什么存在、内存模型、惯用法、面试高频追问**。
> 每篇均含「## 核心概念」知识结构图（mermaid）+ 通俗详解 + 面试速答骨架；Linux/Shell/Go/Java 四篇另有「问题表 → 故事引入 → 深入原理 → 面试考点 → 小结」完整结构与示意图配图（`教学/images/`，12 张）。

## 复习策略

1. **Python 线（00）**：对象模型（可变/不可变、深浅拷贝）→ 容器与推导式 → 函数式（lambda/闭包/装饰器）→ 迭代器/生成器 → 类与魔术方法 → GIL/并发选型 → 性能优化（内置函数、内存）
2. **C++ 线（01）**：指针/引用/值类别 → 内存模型（栈/堆/RAII/智能指针）→ 类与多态（虚函数/vtable）→ 模板/STL 容器与算法 → 移动语义/右值引用 → 现代 C++（11/14/17/20）高频特性
3. **ROS 线（02）**：节点/话题/服务/动作 → 工作空间与 catkin/colcon → 消息定义与收发 → TF/URDF → 机器人仿真（Gazebo）→ ROS2 对比
4. **工程工具线（03-07）**：Docker（镜像/容器/卷/网络/Compose/K8s）→ Git（三区模型/分支/冲突/回滚/协作）→ SQL（CRUD/联表/聚合/窗口函数/索引/事务）→ Linux（体系结构/权限/进程/三剑客/网络排查）→ Shell（变量引号/条件循环/管道重定向/健壮脚本）
5. **后端语言线（08-09）**：Go（类型/切片/map/接口/goroutine/channel/GMP）→ Java（集合/HashMap/线程/JVM 数据区/类加载/GC/调优）
6. **面试四件套**：直觉（一句话）→ 语法/示例（能默写）→ 易错点 → 与另一语言对比

## 教学索引（`教学/`，持续扩充）

| 篇目 | 核心内容 | 对照验证 |
|------|----------|----------|
| 00-Python核心语法与数据模型 | 对象模型/容器/推导式/函数式/迭代器生成器/类魔术方法/并发/性能 | 手写 vs 内置对照、深浅拷贝断言 |
| 01-C++核心语法与STL | 指针引用/内存模型/RAII/多态/模板/STL/移动语义/现代C++ | 手写 vs STL 对照、内存/性能实验 |
| 02-ROS入门与话题通信 | 节点/话题/服务/动作/工作空间/消息/TF/URDF/仿真 | 手写最小发布订阅 vs ROS 真实行为对照 |
| 03-Docker容器化与部署 | 容器 vs VM/镜像/可写层/Dockerfile/卷/网络/Compose/K8s | 手写 Dockerfile 与指令拆解对照 |
| 04-Git版本控制与协作 | 三区模型/分支/merge-rebase/冲突/reset-revert/远端协作 | 命令语义模拟与状态流转对照 |
| 05-SQL数据库基础 | 关系模型/四分类/JOIN/聚合/窗口函数/索引/ACID/隔离级别 | 手写查询语义 vs 标准 SQL 对照 |
| 06-Linux系统基础与常用命令 | 体系结构/文件系统/权限/命令大全/三剑客/进程与信号/管道/shell | 权限位与 umask 计算、管道语义模拟 |
| 07-Shell脚本与自动化 | 变量引号/条件循环/函数/管道重定向/通配符 vs 正则/数组/set 健壮性 | 引号展开、管道流水线、脚本模式模拟 |
| 08-Go语言基础与并发 | 类型/切片陷阱/map/接口/goroutine/channel/GMP/WaitGroup/Mutex | slice 共享数组、生产者-消费者、并发计数模拟 |
| 09-Java核心与JVM入门 | 集合/HashMap/线程同步/JVM 数据区/双亲委派/GC 分代/调优参数 | 栈帧与栈溢出、可达性 GC、HashMap 扩容模拟 |

## 本目录规划

```
13-编程语言/
├── README.md          # 本文件
└── 教学/              # Jupyter notebook（推导 + 代码 + 对照 + 自测）
    └── images/        # ✅ 12 张示意图（三层结构/权限/管道/引号/slice/GMP/channel/JVM/GC/HashMap…）
```

## 面试 30 秒速答（背诵骨架）

**Python**
- **可变 vs 不可变**：list/dict/set 可变、str/tuple/frozenset 不可变；函数默认参数别用可变对象
- **深/浅拷贝**：`copy.copy` 浅（只复制顶层）、`copy.deepcopy` 深（递归）；`a = b` 只是绑定
- **迭代器 vs 生成器**：迭代器实现 `__iter__`/`__next__`；生成器用 `yield`，惰性、一次消费
- **装饰器**：`functools.wraps` 保元信息；`@property`/`@staticmethod`/`@classmethod` 区别
- **GIL**：CPython 同一时刻只有一个线程跑字节码；I/O 密集用线程/协程，CPU 密集用进程
- **`is` vs `==`**：`is` 比身份（id）、`==` 比值；小整数/短字符串有缓存
- **垃圾回收**：引用计数为主 + 标记-清除/分代处理循环引用

**C++**
- **引用 vs 指针**：引用必须初始化、不能为空、不能改绑；指针可为空、可重绑
- **RAII**：资源在构造时获取、析构时释放（智能指针/锁/文件），异常安全基石
- **智能指针**：`unique_ptr`（独占）/`shared_ptr`（计数）/`weak_ptr`（弱引用破环）
- **虚函数**：`virtual` + `override`；vtable 指针实现多态；析构函数要 virtual
- **左值 vs 右值**：右值引用 `&&` + `std::move`/完美转发，实现移动语义省拷贝
- **STL 容器选型**：vector 连续/随机访问；list 中间插入；deque 双端；unordered_* 哈希；map 红黑树有序
- **模板**：编译期多态；`auto`/范围 for/`constexpr` 是 C++11 后高频特性

**ROS（1 vs 2）**
- **节点**：最小可执行单元，`rospy`/`roscpp`；`roslaunch` 启动
- **话题**：异步发布/订阅，`Publisher`/`Subscriber` + 消息类型（msg）
- **服务**：同步请求/响应（srv）；**动作**：长任务反馈/取消（action）
- **ROS2**：DDS 通信、去中心化、`colcon` 构建、`rclpy`/`rclcpp`、生命周期节点

**工程工具（03-07）**
- **Docker**：镜像=只读模板、容器=可写实例；Namespace/Cgroups 隔离；`docker build/run/exec/compose`
- **Git**：工作区→暂存区→版本库三区；`git reset` 移头、`git revert` 新增提交、`reflog` 救回
- **SQL**：SELECT 执行顺序 FROM→WHERE→GROUP BY→HAVING→ORDER BY→LIMIT；JOIN 四类；事务 ACID + 四种隔离级别
- **Linux**：r/w/x=4/2/1；SIGTERM 优雅/SIGKILL 强杀；三剑客 grep 搜、sed 改、awk 算；`ps|grep|awk` 查 PID
- **Shell**：双引号展开/单引号字面；`[` 是 test 命令要空格；管道 `|` 接 stdout→stdin；`set -euo pipefail` 三件套

**后端语言（08-09）**
- **Go**：编译型单二进制 + GC + goroutine（2KB 栈、GMP 调度）；channel 通信而非共享内存；slice 共享底层数组、append 超 cap 换数组；map 并发读写 panic
- **Java**：JVM 数据区（堆/方法区/栈/PC）；可达性分析判垃圾；分代收集（新生代复制、老年代 Full GC）；双亲委派防冒充核心类；HashMap = 数组+链表+红黑树、负载因子 0.75

## 验收标准

- [x] `README.md` 覆盖 Python / C++ / ROS + 工程工具（Docker/Git/SQL/Linux/Shell）+ 后端语言（Go/Java）的复习策略与背诵骨架
- [x] 10 篇教学 notebook 全部含「核心概念」知识结构图（mermaid）+ 逐节点通俗解释
- [x] 06-09 四篇（Linux/Shell/Go/Java）已补全「问题表 + 故事引入 + 深入原理 + 面试考点 + 小结」并配 12 张示意图
- [ ] 教学 notebook 全部含「讲解 + 手写 + 对照 + 自测清单」
- [x] 06-09 四篇代码可执行验证（nbclient err=0）
- [ ] 高频追问按语言逐个补齐

> 💡 学习优先级建议：Python 00 → C++ 01 → Docker 03 → Git 04 → SQL 05 → Linux 06 → Shell 07 → Go 08 → Java 09 → ROS 02（按岗位选）；面向机器人/自动驾驶岗位建议 Python + C++ + ROS 三条线全过，后端岗加 Go/Java，纯算法岗可过 Python + Docker/Git/SQL/Linux。
