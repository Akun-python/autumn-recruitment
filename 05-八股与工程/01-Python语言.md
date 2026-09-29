# 01 Python 语言核心八股（秋招算法 / 后端通用）

> 覆盖范围：GIL、装饰器、生成器与迭代器、深浅拷贝、函数传参、垃圾回收、并发选型与 asyncio、内存管理、高频速览题。
> 使用方式：每节先看**原理**，再跑**代码**，用**对比表**速记差异，最后自测**面试追问**。代码均为 Python 3 可直接运行。

---

## 1. GIL 全局解释器锁

### 1.1 是什么：为什么存在

**GIL（Global Interpreter Lock，全局解释器锁）** 是 CPython 解释器中的一个互斥锁，保证**同一时刻只有一个线程能执行 Python 字节码**。它不是 Python 语言规范的一部分，而是 CPython 的实现细节（Jython / PyPy 等实现没有 GIL）。

**为什么存在 GIL？**

- **内存安全与简单性**：CPython 的对象内存管理（引用计数）不是线程安全的。如果多个线程同时修改一个对象的引用计数，会产生竞争条件导致内存损坏。GIL 让解释器内部状态天然串行，**避免了对每个对象加锁的复杂度**。
- **历史原因**：早期 CPU 单核为主，多线程主要用于 I/O；GIL 的实现简单且性能足够。
- **第三方 C 扩展**：大量 C 扩展（如早期的 NumPy 部分路径）假设 GIL 存在，移除此锁会破坏 ABI 兼容。

**代价**：多线程无法利用多核并行执行纯 CPU 计算（**并行度受限，但并发性仍在**）。注意区分：**并发（concurrency）≠ 并行（parallelism）**，GIL 杀死的是后者。

> ⚠️ **易错点**：不要答「Python 不支持多线程」。Python 多线程对 I/O 密集型任务完全有效；受限的只是**多线程并行执行纯 Python 的 CPU 密集任务**。

### 1.2 对多线程 / 多进程的影响

| 维度 | 多线程 threading | 多进程 multiprocessing |
| --- | --- | --- |
| 内存空间 | **共享**同一进程地址空间 | 各自独立地址空间（默认） |
| GIL 影响 | 受 GIL 约束，CPU 密集几乎无法加速 | 每个进程有独立解释器与 GIL，**可多核并行** |
| 数据传递 | 直接读写共享对象（需加锁） | 通过 `Queue` / `Pipe` / 共享内存（IPC） |
| 开销 | 创建快、切换轻 | 创建重（fork/spawn 复制解释器）、IPC 有序列化成本 |
| 适用 | I/O 密集（网络、磁盘、爬虫） | CPU 密集（计算、训练数据预处理） |
| 崩坏风险 | 一个线程崩溃（段错误）可能带崩进程 | 单进程崩溃不影响其他进程 |

```python
# gil_demo.py —— 验证 GIL：CPU 密集场景下多线程无加速
import threading
import time

def count_down(n):
    while n > 0:
        n -= 1

N = 60_000_000

t0 = time.perf_counter()
count_down(N)
print(f"单线程耗时: {time.perf_counter() - t0:.2f}s")

def run_threads(k):
    t0 = time.perf_counter()
    ts = [threading.Thread(target=count_down, args=(N // k,)) for _ in range(k)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print(f"{k} 线程耗时: {time.perf_counter() - t0:.2f}s")

run_threads(2)   # 实测 ≈ 单线程的 1.5~2 倍时间（反而更慢），而不是减半
run_threads(4)
```

```python
# 多进程版：CPU 密集任务真正并行
import multiprocessing as mp
import time

def count_down(n):
    while n > 0:
        n -= 1
    return n

if __name__ == "__main__":
    N = 60_000_000
    t0 = time.perf_counter()
    with mp.Pool(2) as pool:
        pool.map(count_down, [N // 2, N // 2])
    print(f"2 进程耗时: {time.perf_counter() - t0:.2f}s")   # 接近单线程的一半
```

### 1.3 三种绕开 GIL 的方式

**① 多进程（multiprocessing）**：每个进程独立解释器 + 独立 GIL，真正并行，代价是 IPC 与内存翻倍。

**② 借助会释放 GIL 的库（NumPy 等）**：NumPy 的向量化运算在 C 层实现，执行期间会**主动释放 GIL**，因此 NumPy 计算可以在多线程下并行加速。

```python
# numpy 在大数组运算期间释放 GIL，多线程可并行
import numpy as np
import threading
import time

def heavy_dot():
    a = np.random.rand(3000, 3000)
    return np.linalg.eigvals(a)

t0 = time.perf_counter()
ts = [threading.Thread(target=heavy_dot) for _ in range(4)]
for t in ts:
    t.start()
for t in ts:
    t.join()
print(f"4 线程跑 numpy 耗时: {time.perf_counter() - t0:.2f}s")  # 明显加速
```

**③ C 扩展中主动释放 GIL**：编写 CPython C 扩展时，用 `Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS` 宏包裹不操作 Python 对象的长计算段，临时释放 GIL：

```c
/* c_ext.c —— 示意：C 扩展释放 GIL 做重计算 */
static PyObject* heavy(PyObject* self, PyObject* args) {
    Py_BEGIN_ALLOW_THREADS
    /* 纯 C 计算，不触碰 Python 对象 —— 此时其他线程可执行 */
    Py_END_ALLOW_THREADS
    Py_RETURN_NONE;
}
```

| 绕开方式 | 是否真并行 | 工程成本 | 典型场景 |
| --- | --- | --- | --- |
| multiprocessing | ✅ 是 | 中（IPC、序列化） | 纯 Python CPU 密集 |
| 依赖释放 GIL 的库 | ✅ 是 | 低（改用 NumPy/C 库） | 数值计算、图像处理 |
| C 扩展 + ALLOW_THREADS | ✅ 是 | 高（C 编程、编译） | 高性能库、绑定期 |

### 1.4 面试追问

**Q：Python 3.12 / 3.13 有 GIL 吗？可以关掉吗？**
A：有。3.13 引入了**实验性的自由线程模式（free-threaded build，无 GIL）**，通过 `--disable-gil` 编译选项开启，但生态兼容性仍在演进；生产环境默认仍是 GIL。Python 3.12 只是把 GIL 调度做了优化（如 `stop-the-world` 改进），并未移除。

**Q：GIL 什么时候释放？**
A：每执行固定数量的字节码指令（`sys.getswitchinterval()` 默认 5ms 切换一次）；遇到阻塞 I/O、`time.sleep`、C 扩展主动释放时会立即让出。线程调度由 OS 完成，不是公平轮询。

**Q：既然 GIL 存在，为什么爬虫多线程还有用？**
A：I/O 等待（网络 recv、磁盘 read）期间线程阻塞在系统调用上，**GIL 已被释放**，其他线程可以运行。因此 I/O 密集型任务多线程收益巨大。

> 💡 **面试点评**：标准三连「是什么 → 为什么（引用计数线程安全 + 历史 + C 扩展生态）→ 影响与绕开（多进程/numpy/C 扩展）」，再补一句「GIL 是 CPython 实现细节，非语言规范」即满分。

---

## 2. 装饰器

### 2.1 原理：闭包

**装饰器（Decorator）** 是一个**接收函数（或类）并返回新函数（或类）的可调用对象**，本质是「函数即对象」+ **闭包（Closure）**。闭包 = 内层函数 + 引用的外层自由变量，即使外层函数已返回，内层函数仍能访问这些变量。

```python
# 闭包：内层函数记住外层变量
def make_adder(n):
    def add(x):
        return x + n        # n 是自由变量，被内层记住
    return add

add_5 = make_adder(5)
print(add_5(10))            # 15
print(add_5.__closure__)    # 可以看到闭包绑定的 n=5
```

**装饰器语法糖的本质**：

```python
@timer
def f(): ...

# 等价于
def f(): ...
f = timer(f)
```

### 2.2 无参装饰器

```python
import time
import functools

def timer(func):
    @functools.wraps(func)          # 保留原函数名/docstring，见 2.4
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 耗时 {time.perf_counter() - t0:.4f}s")
        return result
    return wrapper

@timer
def slow_add(a, b):
    time.sleep(0.1)
    return a + b

print(slow_add(1, 2))
```

### 2.3 带参装饰器（装饰器工厂）

带参装饰器 = **再包一层**：外层函数接收参数并返回真正的装饰器。

```python
def repeat(times):
    """带参装饰器：把被装饰函数重复执行 times 次"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(times):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def say(msg):
    print(f"say: {msg}")

say("hi")   # 打印 3 次
```

**记忆口诀：`@decorator` 执行 `decorator(func)`；`@decorator(args)` 执行 `decorator(args)(func)`。**

### 2.4 functools.wraps 的作用

不包 `wraps` 时，被装饰后的函数 `__name__`、`__doc__` 会被 wrapper 覆盖，破坏内省与调试。`functools.wraps(func)` 本质是把 `func.__name__`、`__doc__`、`__module__` 等复制到 wrapper 上，并更新 `__wrapped__` 指向原函数。

```python
import functools

def bad_decorator(func):
    def wrapper(*a, **kw):
        return func(*a, **kw)
    return wrapper

@bad_decorator
def important():
    """重要文档"""
    pass

print(important.__name__)   # wrapper —— 丢失了原函数名！

@functools.wraps
def ok_decorator(func):     # 错误姿势：@functools.wraps 得到的不是可用的装饰器（partial）
    def wrapper(*a, **kw):
        return func(*a, **kw)
    return wrapper

# 正确姿势：wraps(func) 返回的装饰器再装饰 wrapper，逐行对比如下 ↓

def ok_decorator2(func):     # ✅ 正确写法
    @functools.wraps(func)   # wraps(func) 才是「返回装饰器」的正确调用
    def wrapper(*a, **kw):
        return func(*a, **kw)
    return wrapper

@ok_decorator2
def important2():
    """重要文档2"""
    pass

print(important2.__name__)  # important2 —— 被正确保留
```

### 2.5 类装饰器

类装饰器有两种：① 用类充当装饰器（实现 `__call__`）；② 装饰器作用于类（类工厂）。两者都高频出现。

```python
import time

class Timer:
    """用类实现装饰器：实例化时接收函数，__call__ 包装调用"""
    def __init__(self, func):
        functools.wraps(func)(self)          # 把函数元数据复制到实例
        self.func = func

    def __call__(self, *args, **kwargs):
        t0 = time.perf_counter()
        result = self.func(*args, **kwargs)
        print(f"[类装饰器] {self.func.__name__} 耗时 {time.perf_counter()-t0:.3f}s")
        return result

@Timer
def work():
    time.sleep(0.05)
    return "done"

print(work())
```

```python
# 装饰器作用于类：给类批量注册/增加方法
def add_methods(cls):
    def greet(self):
        return f"Hello, {self.name}"
    cls.greet = greet
    return cls

@add_methods
class Person:
    def __init__(self, name):
        self.name = name

p = Person("Alice")
print(p.greet())   # Hello, Alice
```

### 2.6 functools.lru_cache 的演进

`lru_cache(maxsize)` 为函数添加 **LRU（Least Recently Used，最近最少使用）缓存**：以参数为 key 记忆返回值。**演进脉络**：

- Python 3.2 引入 `functools.lru_cache`（用 `OrderedDict` 实现，maxsize 必须为整数）。
- 3.8 引入 `functools.cached_property`（只读属性只计算一次）与 `lru_cache` 的 `typed` 参数（区分 `1` 与 `1.0`）。
- **3.9 新增 `functools.cache`**：`lru_cache(maxsize=None)` 的**无上限简写**，适合参数域有限且确定性的函数。
- 3.12 起 `lru_cache` 底层改为更高效的实现，并支持 `user_function` 直接作为参数。

```python
import functools
import time

@functools.lru_cache(maxsize=128)          # 经典 LRU
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)

@functools.cache                             # Python 3.9+：等价 maxsize=None
def fib2(n):
    return n if n < 2 else fib2(n - 1) + fib2(n - 2)

@functools.cached_property                   # 3.8+：只算一次且不可写
class Config:
    def __init__(self, path):
        self.path = path
    @cached_property
    def data(self):
        time.sleep(0.2)                      # 模拟昂贵加载
        return {"path": self.path}

t0 = time.perf_counter()
print(fib(30), f"耗时 {time.perf_counter()-t0:.2f}s")   # 秒出，未缓存会指数爆炸

c = Config("/etc/app.conf")
t0 = time.perf_counter(); print(c.data); print(c.data)
print(f"二次访问耗时 {time.perf_counter()-t0:.5f}s")     # 第二次不重新加载
```

**递归 + lru_cache 后复杂度从 O(2ⁿ) 降到 O(n)**，这是面试最常见的应用。

> ⚠️ **易错点**：被 `lru_cache` 装饰的函数参数必须**可哈希**（list/dict 会报 TypeError）；它缓存的是**引用**，内部对象被修改不会自动失效。

### 2.7 装饰器对比表

| 类型 | 写法 | 返回 | 典型用途 |
| --- | --- | --- | --- |
| 无参装饰器 | `def timer(func)` | 新函数 | 计时、日志、鉴权 |
| 带参装饰器 | `def repeat(n)` → `def decorator(func)` | 新函数 | 重试、限流阈值 |
| 类装饰器(函数) | 类实现 `__call__` | 可调用实例 | 需保存状态的装饰 |
| 类装饰器(类) | `def add_methods(cls)` | 新类 | 批量注入方法、注册 |
| 内置缓存 | `lru_cache` / `cache` | 新函数 | 记忆化、DP 优化 |

### 2.8 面试追问

**Q：多个装饰器的执行顺序？**
A：**自下而上装饰，自上而下执行**。`@a @b def f` 等价于 `f = a(b(f))`：先执行 `b(f)` 再执行 `a(...)`；调用时先进入最外层 `a` 的 wrapper，再进 `b` 的 wrapper，最后才是 `f` 本体。

**Q：装饰器如何携带并修改被装饰函数的参数？**
A：在 wrapper 内通过 `*args, **kwargs` 捕获并透传，可读取/修改后再调用原函数；也可在调用前做参数校验（参数校验装饰器是经典考题）。

**Q：闭包与装饰器的关系？**
A：装饰器的 wrapper 是一个闭包——它引用了外层 `func` 这个自由变量；离开闭包机制，装饰器无法在被装饰函数定义处之外保留对它的引用。

> 💡 **面试点评**：手写装饰器时务必带 `functools.wraps`，并在结尾解释「没写它会导致 `__name__` 变成 wrapper」——这是面试官最容易设坑的点。

---

## 3. 生成器与迭代器

### 3.1 迭代器协议

**迭代器（Iterator）** 是实现了迭代器协议的对象：必须提供 `__iter__()`（返回自身）和 `__next__()`（返回下一项，耗尽时抛 `StopIteration`）。**可迭代对象（Iterable）** 只要求实现 `__iter__()`，`for` 循环会先调用 `iter(obj)` 得到迭代器，再反复 `next()`。

```python
# 手写迭代器：range 的简化版
class MyRange:
    def __init__(self, start, end):
        self.cur, self.end = start, end

    def __iter__(self):            # 返回迭代器（自身）
        return self

    def __next__(self):
        if self.cur >= self.end:
            raise StopIteration    # 耗尽信号
        self.cur += 1
        return self.cur - 1

for x in MyRange(1, 4):
    print(x)                       # 1 2 3

it = iter(MyRange(1, 4))
print(next(it), next(it))          # 1 2
```

**迭代器是「惰性 + 一次性」的**：数据边取边算、游标不可回退；list 是可迭代对象但**不是**迭代器（`iter([1,2,3])` 才产生迭代器）。

### 3.2 yield 机制

**生成器函数**：函数体内含 `yield`，调用时**不执行函数体**，而是返回一个生成器对象；每次 `next()` 执行到下一个 `yield` 并**暂停**，同时保存整个局部状态（局部变量、指令指针）。

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a                # 产出后暂停，保存 a/b
        a, b = b, a + b

g = fibonacci(5)
print(list(g))                 # [0, 1, 1, 2, 3]
# print(list(g))              # 第二次为空：一次性
```

**生成器与普通函数的区别（对比表）**：

| 维度 | 普通函数 | 生成器函数 |
| --- | --- | --- |
| 关键字 | `return` | `yield`（可同时有 return 作为结束值） |
| 执行时机 | 调用即执行完 | 调用只创建对象，`next()` 才推进 |
| 内存 | 一次性构造全部结果 | 逐个产出，**O(1) 内存** |
| 状态 | 无 | 自动保存局部变量与执行位置 |
| 可重复遍历 | 每次调用重新执行 | 对象一次性，遍历完需重建 |

### 3.3 生成器表达式

**生成器表达式（Generator Expression）** 是「圆括号版推导式」，返回生成器而非 list：

```python
n = 10_000_000
gen = (i * i for i in range(n))          # O(1) 内存
lst = [i * i for i in range(n)]          # O(n) 内存，10M 个 int

print(sum(i for i in range(10)))         # 45，生成器直接作为参数
print(sum([i for i in range(10)]))       # 等价但更耗内存
```

**惰性求值陷阱**：生成器表达式中的变量在**迭代时**才求值，而不是创建时——经典的闭包陷阱同样适用。

```python
# 经典陷阱：生成器表达式延迟求值
g = (x for x in range(3))
n = 10                      # 此刻 range(3) 已固定，与 n 无关，安全
# 但如果写成 (x + n for x in range(3)) 且 n 之后变化，会在 next 时用最新的 n
```

### 3.4 send / throw / close

生成器对象还有三个方法，用于**双向通信**与终止控制：

- `send(value)`：把 value 作为上一个 `yield` 的**表达式结果**注入，并推进到下一个 yield。首个调用必须 `next(g)`（或 `send(None)`）。
- `throw(exc)`：在暂停点抛出异常，可在生成器内用 try/except 捕获。
- `close()`：在暂停点抛出 `GeneratorExit` 终止生成器（一般不能捕获后继续 yield）。

```python
def echo():
    print("生成器启动")
    try:
        while True:
            msg = yield                # send 的值会赋给 msg
            print(f"生成器收到: {msg}")
    except GeneratorExit:
        print("生成器被关闭")

g = echo()
next(g)                 # 启动到第一个 yield
g.send("hello")         # 生成器收到: hello
g.send(42)
g.close()               # 生成器被关闭
```

```python
# 用 throw 在生成器内处理异常
def safe_div_gen():
    for x in [10, 20, 0, 30]:
        try:
            yield 100 // x
        except ZeroDivisionError:
            yield -1

print(list(safe_div_gen()))     # [10, 5, -1, 3]
```

> ⚠️ **易错点**：`for` 循环会自动处理 `StopIteration`；**不要手动捕获 StopIteration 后再 `next()`**——3.7+ 中生成器内部 `return value` 会把 value 放进 `StopIteration.value`，可用于「协程返回值」。

### 3.5 生成器 vs 迭代器 vs 协程

| 概念 | 定义 | 关系 |
| --- | --- | --- |
| 可迭代对象 | 有 `__iter__` | 迭代器的上游概念 |
| 迭代器 | 有 `__iter__` + `__next__` | 可迭代对象的一种 |
| 生成器 | 含 yield 的函数/表达式 | 是迭代器的**便捷实现方式** |
| 协程 | 可挂起/恢复、可双向传值的函数 | 生成器是协程的基础（PEP 342 扩展了 send/throw/close） |

### 3.6 面试追问

**Q：生成器与迭代器的区别？**
A：迭代器是协议（`__iter__` + `__next__`），生成器是**利用 yield 自动实现该协议的工具**——手写迭代器要维护游标状态，生成器由解释器代劳。

**Q：为什么大文件读取要用生成器？**
A：`for line in f` 本身按行惰性读取，但若 `f.readlines()` 会一次性载入全文件进内存；生成器保证**任一时刻内存里只有一行**，O(1) 内存处理 GB 级文件。

**Q：yield 与 return 能共存吗？**
A：能。`return value` 在生成器中表示**结束并携带返回值**（放进 `StopIteration.value`），但 for 循环取不到它，需用 `next(g)` 捕获 `StopIteration.value` 才能读到。

**Q：协程与生成器的关系？**
A：生成器是**协程的底层机制**——PEP 342 给生成器加了 `send/throw/close` 后它就能双向通信；asyncio 的协程（`async def`）则是基于此抽象的更高级形态。

> 💡 **面试点评**：提到「惰性求值 + O(1) 内存」后，主动补充「send 让生成器从单向管道升级为双向通信」，能立刻区分开背答案的候选人。

---

## 4. 深拷贝 vs 浅拷贝

### 4.1 概念与 copy.copy / copy.deepcopy

**浅拷贝（shallow copy）**：创建新容器对象，但**嵌套子对象仍共享引用**；**深拷贝（deepcopy）**：递归复制所有嵌套对象，得到完全独立的一棵对象树。

```python
import copy

a = [1, 2, [3, 4]]
b = copy.copy(a)          # 浅拷贝
c = copy.deepcopy(a)      # 深拷贝

a[2].append(5)
print(a)   # [1, 2, [3, 4, 5]]
print(b)   # [1, 2, [3, 4, 5]]   ← 共享内层 list，被连带修改
print(c)   # [1, 2, [3, 4]]      ← 完全独立

a[0] = 99
print(b)   # [1, 2, ...]   ← 外层元素独立（切片式赋值），浅拷贝只共享嵌套层
```

**理解要点**：浅拷贝**不共享顶层**（`b[0] = x` 不影响 a），但**共享嵌套可变对象**。`list.copy()`、`dict.copy()`、切片 `a[:]` 都是浅拷贝。

### 4.2 可变 / 不可变对象

**不可变对象（int/str/tuple）**：任何"修改"都是**重新绑定新对象**，不存在原地修改，因此拷贝与否几乎无差别（deepcopy 对不可变对象会直接复用）。**可变对象（list/dict/set/自定义对象）** 才存在共享与独立之争。

```python
import copy
t = (1, [2, 3])                     # tuple 含可变元素
s = copy.copy(t)
d = copy.deepcopy(t)
t[1].append(4)
print(s)   # (1, [2, 3, 4])         ← 浅拷贝共享内层 list
print(d)   # (1, [2, 3])            ← 深拷贝彻底隔离
```

**易错点**：说「tuple 不可变所以安全」是错的——**不可变是容器本身的约束，不约束其内含的可变元素**。

### 4.3 嵌套结构陷阱

**陷阱一：循环引用**。deepcopy 内部维护 `memo` 字典，遇到已复制对象直接复用，因此能正确处理自引用/互引用结构（浅拷贝只拷贝一层，无此问题）。

```python
import copy
a = []
a.append(a)                    # 自引用
b = copy.deepcopy(a)
print(b is b[0])               # True：deepcopy 用 memo 正确重建循环
```

**陷阱二：重复引用**。同一对象被引用两次时，deepcopy 保证复制后仍是**同一个对象**（id 相同），这是 memo 的另一个作用。

**陷阱三：不可复制的资源**。文件句柄、线程、socket 不能复制；deepcopy 对它们会尝试复制或报错，此时需要自定义 `__deepcopy__`。

### 4.4 自定义 __deepcopy__ / __copy__

实现 `__copy__(self, memo)` 与 `__deepcopy__(self, memo)` 可控制拷贝行为——例如数据库连接、线程池等「昂贵且应共享」的资源。

```python
import copy

class ConnectionPool:
    def __init__(self, dsn, size=5):
        self.dsn = dsn
        self.size = size
        self._conns = [f"conn-{i}" for i in range(size)]

    def __copy__(self, memo=None):
        """浅拷贝：共享连接池本体，只复制参数壳"""
        return ConnectionPool(self.dsn, self.size)

    def __deepcopy__(self, memo):
        """深拷贝：连接池应全局唯一，直接复用自身"""
        memo[id(self)] = self
        return self

p = ConnectionPool("postgres://db")
q = copy.deepcopy(p)
print(q is p, q.dsn)                 # True —— 池被共享，不被复制
```

### 4.5 对比表

| 维度 | 浅拷贝 | 深拷贝 |
| --- | --- | --- |
| 顶层容器 | 新建 | 新建 |
| 嵌套可变对象 | **共享引用** | 递归复制，完全独立 |
| 不可变元素 | 复用 | 复用（无差异） |
| 循环/重复引用 | 无需处理 | memo 去重，正确处理 |
| 性能 | 快，O(1)~O(1层) | 慢，O(全部对象) |
| API | `copy.copy` / `list[:]` | `copy.deepcopy` |

### 4.6 面试追问

**Q：什么情况下必须用深拷贝？**
A：嵌套结构会被多处修改且需要隔离，或要把「当前快照」保存下来（如状态机快照、撤销栈）。纯读不改的场景浅拷贝就够。

**Q：`copy.deepcopy` 为什么慢？**
A：要递归遍历整棵对象图，且每访问一个对象都查/写 memo 字典，开销与对象总数线性相关；相比之下浅拷贝只复制顶层。

**Q：可变默认参数问题与拷贝有关吗？**
A：有关但不同——`def f(x=[])` 的 `[]` 是**函数对象的属性**，每次调用共享同一个 list；修复用 `def f(x=None)` 再 `x = x if x is not None else []`，或调用侧浅拷贝隔离。

> 💡 **面试点评**：能画出「浅拷贝共享嵌套层、深拷贝独立整棵树」的对象图，并提到 deepcopy 的 memo 处理循环引用，本题就是加分项。

---

## 5. 函数传参：值传递还是引用传递？

### 5.1 真相：对象引用传递（传对象引用）

Python 既不完全是值传递，也不完全是引用传递——标准说法是**「传对象引用」（call-by-object-reference）**，等价于 **「传共享对象」（call-by-sharing）**：

- 参数变量保存的是**对象的引用（指针）**；
- 函数内部对参数**重新赋值**（`x = ...`），只改变**局部变量**的指向，**不影响外部变量**；
- 对参数所指**对象做原地修改**（`x.append(...)`），外部**能看到**变化（因为指向同一对象）。

```python
def demo(x, lst):
    x = x + 1                # 重新绑定局部 x，外部 a 不受影响
    lst.append(1)            # 原地修改对象，外部 b 受影响
    lst = [100]              # 重新绑定局部 lst，外部 b 不受影响

a, b = 1, []
demo(a, b)
print(a)   # 1
print(b)   # [1]
```

**一句话总结**：**「变量是名字，不是盒子」**——赋值永远是「把名字绑定到对象」，传参也是「把名字绑定到传入对象」。所以更准确的模型是：**Python 传的是引用，但按「引用赋值」的语义**，外部与内部**共享对象但不共享变量**。

### 5.2 可变对象的坑

**坑一：函数内原地修改污染调用方。**

```python
def add_user(users, name):
    users.append(name)        # 原地修改

db = ["alice"]
add_user(db, "bob")
print(db)                     # ['alice', 'bob'] —— 调用方被污染
```

修复：内部先拷贝 `users = list(users)`，或返回新列表保持纯函数风格。

**坑二：可变默认参数（经典面试题）。**

```python
def bad(x=[]):
    x.append(1)
    return x

print(bad())   # [1]
print(bad())   # [1, 1]     ← 同一个列表被反复追加！
print(bad.__defaults__)     # ([1, 1, 1],)  默认值对象在函数定义时创建且共享

def good(x=None):
    x = x if x is not None else []
    x.append(1)
    return x

print(good(), good())       # [1] [1]  每次新建
```

**坑三：可变对象作为 dict 的 key / set 的元素**会因哈希变化导致查找错乱（需不可变）。

### 5.3 对比表

| 语言 | 传递模型 | 重新赋值影响外部？ | 原地修改影响外部？ |
| --- | --- | --- | --- |
| C 值传递 | 复制值 | 否 | 否（副本） |
| C 指针/引用 | 传地址 | 视写法 | 是 |
| Java 基本类型 | 值传递 | 否 | — |
| Java 对象 | 传引用副本 | 否（换引用不影响） | **是** |
| **Python** | **传对象引用** | **否** | **是** |
| Ruby/Python 类 | call-by-sharing | 否 | 是 |

### 5.4 面试追问

**Q：Python 到底是值传递还是引用传递？**
A：两者都不准确。准确表述是**传对象引用**：函数拿到的是对象引用，重新绑定参数名不影响外部，但通过该引用做原地修改会影响外部。这是 Python 面试最高频的「扣分题」，务必把两个 case 分开答。

**Q：如何让函数不修改传入的 list？**
A：内部拷贝（`list(users)` / `users.copy()` / `copy.deepcopy` 按需），或改写为返回新对象的纯函数；也可用 `tuple` 包裹不可变接口。

**Q：`a += 1` 与 `a = a + 1` 有区别吗？**
A：对不可变对象无本质区别；对**可变对象**，`list += other` 是原地 `__iadd__`（`extend`），`list = list + other` 是新建对象——这就是「+= 原地修改 vs + 新建」的经典陷阱。

> 💡 **面试点评**：本题的价值在**表述精确**。说「Python 是引用传递」会被追问「那为什么 `x = x + 1` 不影响外部」，说「值传递」又被 `lst.append` 打脸。背熟「变量是名字不是盒子 + 两个 case」即可。

---

## 6. 垃圾回收

### 6.1 引用计数（Reference Counting）

**CPython 的 GC 分两层**：以**引用计数**为主，以**标记-清除 + 分代回收**为兜底。每个对象头部有一个 `ob_refcnt`，引用增删时更新：

```
创建/绑定/入容器   → refcnt += 1
del/重新绑定/出容器 → refcnt -= 1
refcnt 归 0        → 立即回收对象内存（确定性，无停顿）
```

```python
import sys

a = [1, 2, 3]
print(sys.getrefcount(a))   # 2（getrefcount 自身临时 +1）
b = a
print(sys.getrefcount(a))   # 3
del b
print(sys.getrefcount(a))   # 2
del a                       # 归 0 → 立即回收
```

**优点**：即时、简单、可预测（`__del__` 确定性调用）；**缺点**：无法处理**循环引用**（互相引用导致 refcnt 永不归零）——这是标记-清除存在的原因。

> ⚠️ **易错点**：`sys.getrefcount` 返回结果总比直觉多 1（它自己临时引用了一次）；`-5~256` 小整数被全局缓存，引用数虚高是正常的。

### 6.2 标记-清除（Mark-and-Sweep）

**循环引用（Cyclic Reference / Reference Cycle）**：`a.b = b; b.a = a` 后双方 refcnt ≥ 1，即使外部无引用也不被回收。CPython 用**标记-清除**处理：

1. **根集合**：从全局变量、栈、GC 内部可达对象出发；
2. **标记**：DFS 遍历所有可达对象，打上 mark；
3. **清除**：未标记的**不可达对象**被回收（含整个环）。

```python
import gc

class Node:
    def __init__(self, name):
        self.name = name
        self.next = None
    def __del__(self):
        print(f"回收 {self.name}")

gc.collect()                    # 先清空历史垃圾
a, b = Node("a"), Node("b")
a.next, b.next = b, a           # 循环引用
del a, b                        # refcnt 各剩 1（互指），不会立即回收
print("触发回收:", gc.collect())  # 分代回收清除该环，打印 回收 a / 回收 b
```

### 6.3 分代回收（Generational GC）

**经验法则**：「对象活得越久，越不可能死」。CPython 把容器对象分到 **3 个代（generation）**：新建对象入 **第 0 代**；**每存活过一次回收**就升一代；**0 代回收最频繁（阈值最小），2 代最不频繁**。阈值可用 `gc.get_threshold()` 查看（默认 `(700, 10, 10)`：每新增 700 个对象触发 0 代回收，0 代回收 10 次触发 1 代，1 代 10 次触发 2 代）。

```
回收频率：gen0 >>> gen1 > gen2
触发条件：新对象数量累计达阈值 → 只扫 0 代；0 代回收次数达阈值 → 连扫 0+1 代……
```

```python
import gc

print(gc.get_threshold())            # (700, 10, 10)
print(gc.get_count())                # 当前三代各自累积的对象数
gc.collect(0)                        # 只做 0 代回收
gc.collect()                         # 全代回收
```

**关键点**：分代回收只追踪**容器对象**（list/dict/自定义实例等），非容器对象（int/str 等）不参与，靠引用计数即可；**小对象、短命对象**被快速回收，所以 0 代扫描成本低。

### 6.4 gc 模块常用接口

| 接口 | 作用 |
| --- | --- |
| `gc.collect(gen?)` | 手动触发回收，返回回收对象数 |
| `gc.get_threshold()` / `gc.set_threshold()` | 读写各代阈值 |
| `gc.get_count()` | 各代累积对象数 |
| `gc.disable()` / `gc.enable()` | 关闭/开启分代回收（**危险**，需自行兜底） |
| `gc.garbage` | 无法被 `__del__` 正常处理的回收对象列表 |
| `gc.is_tracked(obj)` | 判断对象是否被分代回收追踪 |

**循环引用 + `__del__` 陷阱**：若环中对象定义了 `__del__`，Python 无法确定清理顺序，会把它们放进 `gc.garbage` 而不自动回收（3.4+ 大部分场景已能安全处理，但最佳实践仍是**避免在 `__del__` 中访问环内其他对象**）。

### 6.5 对比表

| 机制 | 是否主用 | 处理对象 | 能否破循环引用 | 特点 |
| --- | --- | --- | --- | --- |
| 引用计数 | ✅ 主 | 所有对象 | ❌ | 即时、确定性 |
| 标记-清除 | 兜底 | 容器对象 | ✅ | 遍历有停顿 |
| 分代回收 | 兜底 | 容器对象 | ✅（配合标记清除） | 用「代」分摊扫描成本 |

### 6.6 面试追问

**Q：Python 的 GC 能关掉吗？**
A：能 `gc.disable()` 关分代回收，但**引用计数无法关闭**；只要代码不制造循环引用，关掉分代回收也基本不泄漏。

**Q：为什么循环引用不直接用「解除引用计数归零」解决？**
A：循环引用时双方 refcnt 都 ≥1，永远达不到 0，引用计数无法自行判定「环整体不可达」；必须从根出发做可达性分析（标记-清除），这是引入分代 GC 的原因。

**Q：大对象与 GC 的关系？**
A：超大对象（如 >256KB 的缓冲区）由**系统内存分配器直接分配**，不经过 PyMalloc 内存池；它们是否进分代回收取决于类型与 `gc` 追踪状态。

> 💡 **面试点评**：完整答出「引用计数为主 + 标记清除/分代兜底」的双层结构，再用 `del a,b` + `gc.collect()` 现场演示循环引用回收，即高分。

---

## 7. 多线程 / 多进程 / 协程选型与 asyncio

### 7.1 三者的本质区别

| 维度 | 多线程 threading | 多进程 multiprocessing | 协程 asyncio |
| --- | --- | --- | --- |
| 单位 | 内核线程 | 独立进程 | 用户态协程 |
| 并行 | 受 GIL 限制（CPU 密集无效） | ✅ 真并行 | 单线程内并发 |
| 切换成本 | 内核态切换，微秒级 | 进程间切换最重 | **纳秒~微秒级**，纯用户态 |
| 内存 | 共享 | 独立 + IPC | 共享（单线程） |
| 适用 | I/O 密集（适中并发） | CPU 密集 / 隔离需求 | **高并发 I/O**（万级连接） |
| 心智负担 | 锁、竞态 | IPC、序列化 | 事件循环、不可阻塞 |

**选型口诀**：**CPU 密集 → 多进程；高并发 I/O → 协程；混合 → 多进程/线程 + 协程嵌套**（如 `ProcessPoolExecutor` 里跑 asyncio）。

### 7.2 asyncio 事件循环原理

**asyncio = 单线程事件循环（Event Loop）驱动的协作式多任务**。核心机制：

1. **事件循环**维护一个**任务队列（就绪队列）** 与**等待表**（socket/定时器等）；
2. **协程通过 `await` 主动挂起**，把控制权交还事件循环（**协作式**，非抢占式）；
3. 当 I/O 就绪（`epoll`/`kqueue`/`select` 检测到）或定时器到期，事件循环**恢复对应协程**继续执行。

```
协程A await read() ──挂起──► 事件循环调度 ──► 协程B 执行
                                   ▲                  │
                           I/O 就绪回调 ◄──epoll 就绪事件
```

**async/await 的本质**：`async def` 定义的函数返回**协程对象**（coroutine）；`await` 等待可等待对象（coroutine / Future / Task），**遇到阻塞就挂起**。协程**不能**在未运行事件循环时执行，必须 `asyncio.run()` 或 `loop.run_until_complete()`。

**协程切换开销**：远小于线程切换——无系统调用、无内核态切换、无栈切换开销，仅保存/恢复 Python 栈帧，**一次切换约百纳秒级**；而线程切换要经过内核调度，微秒级且可能触发上下文切换/缓存失效。

### 7.3 asyncio 并发示例

```python
# asyncio_demo.py —— 3 个 IO 任务并发，总耗时 ≈ 最大单个任务耗时
import asyncio
import time

async def fetch(url, delay):
    await asyncio.sleep(delay)          # 模拟网络 IO（sleep 会真正让出）
    return f"{url} done in {delay}s"

async def main():
    t0 = time.perf_counter()
    results = await asyncio.gather(      # 并发调度 3 个协程
        fetch("http://a.com", 1),
        fetch("http://b.com", 2),
        fetch("http://c.com", 3),
    )
    for r in results:
        print(r)
    print(f"总耗时: {time.perf_counter() - t0:.2f}s")   # ≈ 3s 而非 6s

asyncio.run(main())
```

```python
# 用任务(Task)实现「火速发射」+ 回调
async def tick(name, n):
    for i in range(n):
        await asyncio.sleep(0.1)
        print(name, i)

async def main2():
    tasks = [asyncio.create_task(tick(f"T{i}", 3)) for i in range(3)]  # 立即排队
    await asyncio.gather(*tasks)        # 等待全部完成

asyncio.run(main2())
```

> ⚠️ **易错点**：
> 1. 协程内**禁止**调用阻塞函数（`time.sleep`/同步 `requests.get`），否则会**阻塞整个事件循环**；应使用 `asyncio.sleep` / `aiohttp`，或把阻塞代码丢进 `loop.run_in_executor`。
> 2. 只 `asyncio.run(fetch(...))` 且 fetch 内部无 await 点时会「同步跑完」，不要误以为自动并发。
> 3. 别忘 `await`：`asyncio.gather` 忘 await 会拿到 coroutine 对象。

```python
# 阻塞代码的正确姿势：扔进线程池执行器
import asyncio, time

def blocking_io():
    time.sleep(1)                        # 同步阻塞（示例）
    return "blocking done"

async def main3():
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, blocking_io)  # 不阻塞事件循环
    print(result)

asyncio.run(main3())
```

### 7.4 面试追问

**Q：协程为什么比线程轻量？**
A：线程是内核对象，切换要经过内核、涉及栈和 CPU 上下文保存；协程是**用户态程序控制**的挂起/恢复，仅保存解释器栈帧，且**无锁竞争问题**（单线程内不会同时执行两段代码）。因此协程可支撑十万级并发连接，线程则困难得多。

**Q：事件循环里出现 CPU 密集代码会怎样？**
A：会**卡死事件循环**——所有协程都无法推进，表现为「假死」。解决：`run_in_executor` 扔到线程池，或用多进程。

**Q：gather 与 TaskGroup 的区别？**
A：`asyncio.gather` 在任一任务异常时默认直接抛错（可用 `return_exceptions=True` 控制）；Python 3.11+ 的 `asyncio.TaskGroup` 更严格：**一个任务失败会取消其余任务**并聚合异常（结构化并发，类似 Trio）。

**Q：asyncio 与多线程能混用吗？**
A：能，但注意——`asyncio.run` 会**创建新事件循环**；在线程内跑 asyncio 需自行管理 loop；跨线程往 loop 提交任务用 `loop.call_soon_threadsafe`。

> 💡 **面试点评**：把「协作式调度」与「切换开销量级对比（内核态 vs 用户态）」讲清楚，再补一个 `gather` 并发示例，本题稳过。

---

## 8. 内存管理

### 8.1 小整数池（Small Int Cache）

CPython 在解释器启动时**预创建 -5 ~ 256 的整数对象**（`small_ints`），所有引用同一整数的代码都指向**同一个对象**：

```python
a, b = 256, 256
print(a is b)      # True  —— 池内
c, d = 257, 257
print(c is d)      # False —— 池外，各自新建（交互式环境可能因编译优化为 True）
```

**原因**：小整数极高频使用，缓存避免反复 malloc；代价是这些对象**永不回收**（常驻内存极小，可忽略）。`is` 比较整数不可靠，**整数比较一律用 `==`**。

### 8.2 字符串 intern

短字符串（标识符、编译期字面量）会被 **intern**：相同内容的字符串共享同一对象，比较可用 `is` 加速（CPython 对「看起来像标识符」的字符串自动 intern）。

```python
s1 = "hello_world"      # 编译期字面量，被 intern
s2 = "hello_world"
print(s1 is s2)         # True（同文件内通常成立）

t1 = "hello world!"     # 含空格等非标识符字符，可能不 intern
t2 = "hello world!"
print(t1 is t2)         # False（依赖实现细节！）

u = "".join(["a", "b"]) # 运行时拼接，不 intern
print(u is "ab")        # False
print(u == "ab")        # True —— 永远用 ==
```

> ⚠️ **易错点**：**intern 是 CPython 实现细节，不是语言规范**。面试答「字符串用 `is` 比较」即错；正确姿势：**比较内容永远用 `==`，`is` 只比较身份**。

### 8.3 内存池（PyMalloc）

**CPython 对象内存不直接交给系统 malloc**，而是经 **PyMalloc 分配器**分层管理：

```
对象大小阈值：<=512 字节 → PyMalloc 内存池；>512 字节 → 系统 malloc
PyMalloc：arena(256KB) → pool(4KB) → block(8/16/32/.../512B 按 8 字节对齐分级)
小对象频繁创建/释放 → 池内快速复用，减少系统调用与碎片
```

**结果**：大量小对象（dict/list 节点、字符串等）的分配释放非常快，且**内存被池持有不立即还给 OS**——这就是「Python 进程 RSS 居高不下」的常见原因之一（`ps` 看到内存占用大，不代表泄漏，可能是池在复用）。

### 8.4 字典底层：哈希表

**CPython 3.6+ 的 dict 是「紧凑哈希表」（compact dict）**：哈希索引表（稀疏） + 条目数组（稠密）分离。核心概念：

- **哈希函数**：`hash(key)` 取低位作为槽位起点；`hash()` 对 str/bytes 有随机化种子（防 HashDoS）。
- **开放寻址（Open Addressing）**：冲突时**探测下一个槽位**，而非链表。CPython 探测序列为**分段的二次探测**（perturb 策略），保证均匀散布：

```
伪代码（开放寻址插入）：
    i = hash(key) & mask
    while 槽位 i 非空:
        i = (i*5 + perturb + 1) & mask     # 二次探测：perturb 每次右移
        perturb >>= PERTURB_SHIFT
    写入槽位 i

伪代码（查找）：
    从 i 出发按同一序列探测，遇 EMPTY 说明不存在；遇 key 相等返回
```

- **rehash（扩容/缩容）**：当**装载因子（used / mask+1）超过 2/3**，容量扩为约 2 倍并**重新分配索引表、重新计算每个 key 的槽位**；缩容类似。因此插入/删除**均摊 O(1)**，但扩容那一刻是 O(n)（会触发全量 rehash）。
- **退化风险**：若 key 的哈希分布差或哈希冲突多（恶意构造），dict 操作退化到接近 O(n)——随机化种子正是为此。

```python
d = {}
for i in range(8):          # 观察容量变化（实现细节，展示 rehash 行为）
    d[i] = i
# 装载因子超过阈值时会扩容；可配合 sys.getsizeof(d) 观察内存
import sys
print(sys.getsizeof(d))     # 容量决定对象大小
```

### 8.5 面试追问

**Q：`-5~256` 为什么是 `is` 安全的？**
A：这是**小整数池**的实现选择：启动时预创建这些对象并全局复用。超出范围的对象每次新建，`is` 结果不确定，所以**别用 `is` 比数值**。

**Q：dict 为什么快？哈希冲突怎么处理？**
A：O(1) 平均复杂度来自哈希 + 开放寻址；冲突时按探测序列找空位，探测序列设计（perturb 二次探测）是为了让冲突对象散开。恶意输入可让哈希全冲突，退化为 O(n)。

**Q：Python 内存为什么「只涨不降」？**
A：三个原因：① PyMalloc 池复用内存不还给 OS；② 小整数池/常量缓存常驻；③ 碎片化。判断是否泄漏要用 `tracemalloc` 对比快照，而不是看 RSS。

> 💡 **面试点评**：把「小整数池 / intern / PyMalloc / dict 哈希表」串成一条「Python 如何管理内存」的完整链路，是这一节的提分关键。

---

## 9. 高频题速览

### 9.1 两个快速排序写法对比

**写法一：原地 Lomuto 分区（面试标准）**——内存 O(log n)（递归栈），原地交换：

```python
import random

def quicksort_lomuto(arr, lo=0, hi=None):
    if hi is None:
        hi = len(arr) - 1
    if lo >= hi:
        return arr
    p = random.randint(lo, hi)          # 随机基准防退化
    arr[lo], arr[p] = arr[p], arr[lo]
    pivot = arr[lo]
    i = lo                              # i 左边都是 < pivot
    for j in range(lo + 1, hi + 1):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[lo], arr[i] = arr[i], arr[lo]   # pivot 归位
    quicksort_lomuto(arr, lo, i - 1)    # 递归左区间 [lo, i-1]
    quicksort_lomuto(arr, i + 1, hi)    # 递归右区间 [i+1, hi]
    return arr
```

**写法二：简洁递归（额外空间）**——清晰但每次分区新建 list，额外 O(n) 内存：

```python
def quicksort_simple(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort_simple(left) + mid + quicksort_simple(right)
```

| 维度 | 写法一（原地 Lomuto） | 写法二（新建分区） |
| --- | --- | --- |
| 额外内存 | O(log n) 递归栈 | **O(n)**（每层新建 list） |
| 稳定性 | 不稳定 | 不稳定 |
| 最坏复杂度 | O(n²)（随机基准可规避） | O(n²) |
| 工程适用 | ✅ 生产/竞赛 | 教学/演示 |

### 9.2 列表推导式 vs 生成器表达式

| 维度 | 列表推导式 `[x for x in ...]` | 生成器 `(x for x in ...)` |
| --- | --- | --- |
| 结果 | 立即构建 list | 惰性生成器对象 |
| 内存 | O(n) | **O(1)** |
| 可复用 | 可反复遍历 | 一次性 |
| 适用 | 结果集小、需索引/多次遍历 | 大数据流、链式惰性管道 |

```python
# 大文件行过滤：生成器管道 O(1) 内存
def big_log_filter(path, keyword):
    with open(path) as f:
        return (line for line in f if keyword in line)   # 惰性
```

### 9.3 `__init__` vs `__new__`

**`__new__` 负责「创建对象」（分配内存），`__init__` 负责「初始化对象」（设置属性）**。`__new__` 是**类方法**（第一个参数是 cls），在 `__init__` 之前被调用，且必须返回实例（否则 `__init__` 不会被调用）；`__init__` 是实例方法，返回必须为 None。

```python
class Point:
    def __new__(cls, x, y):
        print("__new__ 先执行")
        inst = super().__new__(cls)      # 分配内存
        inst._created = True             # 可在 new 里先挂标记
        return inst

    def __init__(self, x, y):
        print("__init__ 后执行")
        self.x, self.y = x, y

p = Point(1, 2)
# 输出顺序：__new__ 先执行 → __init__ 后执行
```

**典型应用**：单例（在 `__new__` 中返回缓存实例）、不可变对象定制（namedtuple 风格）。

```python
class Singleton:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### 9.4 `is` 与 `==`

**`==` 比较值（调用 `__eq__`），`is` 比较身份（id 是否相同，即 `is` 等价 `id(a) == id(b)`）**。`is` 不走 `__eq__`，更快；`None` 比较用 `is None`。

```python
a = [1, 2, 3]
b = [1, 2, 3]
print(a == b)    # True  值相等
print(a is b)    # False 不同对象
print(a is a)    # True
print(None is None)          # True —— 判断 None 永远用 is
print(0 is 0)    # True（小整数池）但不要依赖
```

**数值比较**：`==` 可能被重载出奇怪结果（如 numpy 数组返回布尔数组，需 `.all()`），`is` 不受影响。

### 9.5 速览对比总表

| 主题 | 一句话答案 |
| --- | --- |
| GIL 绕开 | 多进程 / 用释放 GIL 的库 / C 扩展 ALLOW_THREADS |
| 装饰器本质 | 闭包 + 函数即对象，`f = deco(f)` |
| 生成器内存 | 惰性 + O(1) 内存 |
| 深拷贝 | 递归复制整棵对象树，memo 处理循环引用 |
| 传参 | 传对象引用：重绑定不影响外部，原地修改影响外部 |
| GC 三层 | 引用计数（主）+ 标记清除 + 分代回收（兜底） |
| 高并发 IO | asyncio 事件循环，用户态协作切换 |
| 小整数池 | -5~256 全局缓存，`is` 不可靠，数值用 `==` |
| dict 冲突 | 开放寻址 + perturb 二次探测，装载因子 2/3 触发 rehash |

### 9.6 面试追问（速览）

**Q：快排写错的常见点？**
A：基准选择（固定首元素在有序输入退化 O(n²)，需随机化）；分区后递归区间边界（`i-1` / `i+1` 别写成 `i`）；`mid` 处理相等元素避免无限递归。

**Q：`x == y` 与 `x is y` 性能？**
A：`is` 只比较指针，O(1) 且不触发 `__eq__`；`==` 可能触发用户定义逻辑甚至重计算。判 `None`、判同一对象用 `is`。

**Q：`__new__` 里不返回实例会怎样？**
A：`__init__` **不会**被调用，`Point(...)` 返回 `__new__` 的返回值——这正是单例与不可变类型的实现手段。

> 💡 **面试点评**：本节是「背完即得分」的送分区，但**代码细节**（快排边界、推导式 vs 生成器）才是区分度所在，建议每题都动手跑一遍再上考场。

---

## 附录：本文件速记卡（一句话 × 9 主题）

1. **GIL**：CPython 全局互斥锁，保证字节码单线程执行；为引用计数线程安全 + C 扩展生态；CPU 密集用多进程。
2. **装饰器**：闭包实现 `f = deco(f)`；带参 = 三层嵌套；务必 `functools.wraps`。
3. **生成器**：yield 惰性求值、O(1) 内存、send/throw/close 双向通信。
4. **深浅拷贝**：浅拷共享嵌套层，深拷递归复制，deepcopy 用 memo 解循环引用。
5. **传参**：传对象引用——重绑定不传染、原地修改传染。
6. **GC**：引用计数即时回收，标记-清除破循环引用，分代回收摊成本。
7. **并发选型**：CPU 密集→多进程；IO 密集高并发→协程；线程居中。
8. **内存管理**：小整数池 -5~256、短字符串 intern、PyMalloc 池复用、dict 开放寻址 + rehash。
9. **高频题**：快排随机基准、推导式 vs 生成器内存、`__new__` 建对象 / `__init__` 初始化、`is` 比身份 / `==` 比值。
