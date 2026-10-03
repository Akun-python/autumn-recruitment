# -*- coding: utf-8 -*-
"""10-位运算与字符串模板.py —— P2 加分（位技巧 + KMP + 滚动哈希）
配套教学：../教学/12-位运算.ipynb & 13-字符串.ipynb

本文件包含：
  1. 位运算高频技巧（lowbit、2 的幂、popcount）
  2. 异或系列：只出现一次 / 缺失数字 / 两数相加
  3. 反转二进制位
  4. KMP 字符串匹配（O(n+m)）
  5. 滚动哈希 Rabin-Karp（O(n+m)）
"""
import math


# ============ 1. 位运算高频技巧（背注释） ============
def bit_tricks():
    """演示三个位运算技巧（返回结果供测试）。

    变量说明：
      - x / y / n: 演示用的数值
      - lowbit: n 的最低位的 1（n & -n）
      - is_pow2: n 是否为 2 的幂
      - count_1 / c: 二进制中 1 的个数

    各技巧原理：
      - n & (-n)：-n 是 n 的补码（取反+1），与 n 按位与会保留最低位的 1，
        例 6(110) & -6(010) = 2(010)；
      - n & (n-1)：n-1 会把最低位的 1 变成 0、其后全变 1，与 n 相与后
        最低位的 1 被消掉 → 结果为 0 当且仅当 n 只有这一个 1（2 的幂）；
      - while n: n &= n-1：每轮消掉一个最低位的 1 → 循环次数 = 1 的个数。
    """
    x, y, n = 8, 12, 6
    lowbit = n & (-n)                     # 取最低位 1：6(110)→2(010)
    is_pow2 = n > 0 and (n & (n - 1)) == 0   # 2 的幂只有 1 个 1
    count_1 = bin(n).count('1')           # 手写版：while n: n &= n-1; cnt+=1
    # 手写 popcount（不用内置 bin）：
    c = 0                                 # 计数器
    while n:                              # n 不为 0 就继续（还有 1 没消掉）
        n &= n - 1                        # 消掉最低位的 1
        c += 1                            # 计数加 1
    return lowbit, is_pow2, c


def hamming_weight(n):
    """统计二进制中 1 的个数（LeetCode 191）。

    原理：n & (n-1) 每次把最低位的 1 变成 0，其余位不变 →
    每执行一次少一个 1，执行次数就是 1 的个数。

    变量说明：
      - n: 输入整数（循环中被不断消减）
      - cnt: 1 的个数计数

    循环终止条件：
      - n 变为 0（所有 1 都消掉）。

    复杂度：
      - 时间 O(位数中 1 的个数)，空间 O(1)。
    """
    cnt = 0                               # 计数器
    while n:                              # 还有 1 没消掉
        n &= n - 1                        # 消掉最低位的 1
        cnt += 1                          # 计数
    return cnt


# ============ 2. 只出现一次的数字（136，异或消消乐） ============
def single_number(nums):
    """找出只出现一次的数字（其他都出现两次，LeetCode 136）。

    异或三条性质：
      - a ^ a = 0（自己异或自己得 0）；
      - a ^ 0 = a；
      - 异或满足交换律、结合律。
    所以把所有数全部异或：成对的都消成 0，剩下的就是那个单身数字。

    变量说明：
      - nums: 输入数组；res: 累计异或结果；x: 当前元素

    循环终止条件：
      - 遍历完所有元素。

    复杂度：
      - 时间 O(n)，空间 O(1)。
    """
    res = 0                               # 异或累计值（0 是单位元）
    for x in nums:                        # 逐个异或
        res ^= x                          # 成对消掉，剩下唯一的
    return res


# ============ 3. 缺失数字（268：xor 补齐下标） ============
def missing_number(nums):
    """找出 [0, n] 中缺失的那个数（数组有 n 个数，LeetCode 268）。

    思路：把数组所有元素 与 全部下标 0..n 异或。
    完整集合 {0,1,...,n} 的异或 与 数组元素的异或 抵消后，
    剩下的就是缺失的那个数字。

    变量说明：
      - nums: 长度为 n 的数组（缺失了一个 0..n 中的数）
      - res: 异或累计值；i / x: 当前下标 / 当前元素

    循环终止条件：
      - 遍历完所有元素。

    复杂度：
      - 时间 O(n)，空间 O(1)。
    """
    res = len(nums)                       # 先异或 n（完整集合多出的一个）
    for i, x in enumerate(nums):          # i 是下标，x 是元素
        res ^= i ^ x                      # 下标与元素异或（成对抵消）
    return res                            # 剩下缺失的数


# ============ 4. 异或求两数之和（不用 +） ============
def get_sum(a, b):
    """两个整数相加，但不允许用 + / -（LeetCode 371）。

    原理：
      - a ^ b：无进位相加的结果；
      - (a & b) << 1：进位（同为 1 的位要进位，左移一位放到位上）。
    反复把"无进位和"与"进位"相加，直到进位为 0。

    变量说明：
      - a / b: 两个加数（b 在循环中不断变成进位）
      - carry: 本轮进位

    循环终止条件：
      - b（进位）变为 0 → 无进位可加，a 即最终和。

    复杂度：
      - 时间 O(位数)（常数 32/64），空间 O(1)。
    """
    while b:                              # 进位不为 0 就继续
        carry = (a & b) << 1              # 算出进位：同为 1 的位左移
        a = a ^ b                         # 无进位相加
        b = carry                         # 下一轮把进位加上
    return a                              # 进位为 0 → a 就是结果


# ============ 5. 反转二进制位（190） ============
def reverse_bits(n):
    """反转 32 位无符号整数的二进制位（LeetCode 190）。

    思路：逐位取出 n 的最低位，累加到 res 的"低位→高位"，
    相当于把 n 从低位到高位逐位搬到 res 的高位到低位。

    变量说明：
      - n: 输入整数（循环中不断右移）
      - res: 结果（不断左移腾位 + 或上取出的位）

    循环终止条件：
      - 处理完 32 位。

    复杂度：
      - 时间 O(32)，空间 O(1)。
    """
    res = 0                               # 结果
    for _ in range(32):                   # 32 位逐个处理
        res = (res << 1) | (n & 1)        # 结果左移一位，再放入 n 的最低位
        n >>= 1                           # n 右移，处理下一位
    return res


# ============ 6. KMP 字符串匹配（O(n+m)） ============
def kmp_next(pattern):
    """构建 KMP 的 next 数组（失配跳转表）。

    next[i] = pattern[:i+1]（前缀）的"最长相等真前后缀"长度，
    即失配时已匹配前缀可以安全跳转的位置。

    变量说明：
      - pattern: 模式串；m: 模式串长度
      - nxt: next 数组；j: 当前已匹配的相等前后缀长度
      - i: 正在计算的模式串下标

    过程拆解：
      1. j 表示"已经匹配了多少个前后缀字符"；
      2. 若 pattern[i] != pattern[j] 且 j > 0 → 用 nxt[j-1] 回退
         （跳到次长的相等前后缀），直到相等或 j == 0；
      3. 若 pattern[i] == pattern[j] → j 加 1；
      4. nxt[i] = j。

    循环终止条件：
      - 内层 while：j == 0 或字符相等时停止；
      - 外层 for：算完整个模式串。

    复杂度：
      - 时间 O(m)，空间 O(m)。
    """
    m = len(pattern)                      # 模式串长度
    nxt = [0] * m                         # next 数组（初始全 0）
    j = 0                                 # 已匹配的相等前后缀长度
    for i in range(1, m):                 # i 从 1 开始（nxt[0] 恒为 0）
        while j > 0 and pattern[i] != pattern[j]:  # 失配
            j = nxt[j - 1]                # 回退到次长的相等前后缀
        if pattern[i] == pattern[j]:      # 匹配上
            j += 1                        # 前后缀长度加 1
        nxt[i] = j                        # 记录
    return nxt


def kmp_search(text, pattern):
    """用 KMP 在 text 中查找 pattern 的所有起始下标。

    与构建 next 的逻辑一致：主串只前进不回退，模式串失配时按
    next 跳转 → 整体 O(n+m)。

    变量说明：
      - text / pattern: 主串 / 模式串
      - nxt: next 数组；res: 命中位置列表
      - i / ch: 主串下标 / 字符；j: 模式串当前匹配长度

    过程拆解：
      1. ch 与 pattern[j] 失配且 j > 0 → j = nxt[j-1] 回退；
      2. 匹配上 → j 加 1；
      3. j == len(pattern) → 命中：记录起点 i-j+1，然后 j = nxt[j-1]
         继续寻找重叠的后续匹配。

    循环终止条件：
      - 遍历完整个主串。

    复杂度：
      - 时间 O(n+m)，空间 O(m)。
    """
    if not pattern:                       # 空模式串：无处可匹配
        return []
    nxt = kmp_next(pattern)               # 构建 next 表
    res = []                              # 命中位置
    j = 0                                 # 模式串已匹配长度
    for i, ch in enumerate(text):         # 主串逐个字符（不回退）
        while j > 0 and ch != pattern[j]: # 失配
            j = nxt[j - 1]                # 按 next 跳转
        if ch == pattern[j]:              # 匹配上
            j += 1                        # 已匹配长度 +1
        if j == len(pattern):             # 完整匹配一个
            res.append(i - j + 1)         # 起始下标 = i - len + 1
            j = nxt[j - 1]                # 跳转，继续找重叠匹配
    return res


# ============ 7. 滚动哈希（Rabin-Karp，O(n+m)） ============
def rabin_karp(text, pattern):
    """用滚动哈希（Rabin-Karp）在 text 中查找 pattern，返回第一个下标。

    思路：把字符串看成 base 进制的数（模 mod 防溢出）。哈希相等时
    再逐字符验证（防哈希碰撞）。

    变量说明：
      - text / pattern: 主串 / 模式串；m: 模式串长度
      - base / mod: 进制基数 / 取模数
      - hp: 模式串哈希；power: base^(m-1)（用于滚动删除最左字符）
      - cur: 当前窗口哈希；i / ch: 主串下标 / 当前字符

    过程拆解：
      1. 计算模式串哈希 hp；
      2. 从左到右维护等长窗口的哈希 cur：
         - 窗口满后，每加一个新字符，先减去最左字符的贡献
           (ord(text[i-m]) * power)，再乘 base 加新字符；
      3. cur == hp 时逐字符比对，相同则返回起始下标。

    循环终止条件：
      - 遍历完主串；命中则提前返回。

    复杂度：
      - 期望 O(n+m)，最坏 O(n·m)（哈希全碰撞）；空间 O(1)。
    """
    m = len(pattern)                      # 模式串长度
    if m > len(text) or not pattern:      # 模式串比主串长 或 空 → 无解
        return []
    base, mod = 131, 10**9 + 7            # 基数与大质数（降低碰撞概率）
    hp = 0                                # 模式串哈希
    for ch in pattern:                    # 逐字符算哈希
        hp = (hp * base + ord(ch)) % mod  # 多项式哈希
    power = pow(base, m - 1, mod)         # base^(m-1) 模 mod（滚动删除用）
    cur = 0                               # 当前窗口哈希
    for i, ch in enumerate(text):         # 逐字符滚动
        if i >= m:                        # 窗口已满 → 需要移除最左字符
            # 减去最左字符贡献，+mod 再 %mod 保证非负
            cur = ((cur - ord(text[i - m]) * power) % mod + mod) % mod
        cur = (cur * base + ord(ch)) % mod  # 加入新字符（乘 base 再加）
        if i >= m - 1 and cur == hp and text[i - m + 1:i + 1] == pattern:
            return i - m + 1              # 哈希相同且逐字符验证通过 → 命中
    return -1                             # 没找到


# ============ 测试 ============
if __name__ == '__main__':
    assert hamming_weight(6) == 2                          # 6(110) 有 2 个 1
    assert single_number([4, 1, 2, 1, 2]) == 4             # 4 只出现一次
    assert missing_number([3, 0, 1]) == 2                  # 缺失 2
    assert get_sum(5, 7) == 12                             # 5+7 不用加法
    assert reverse_bits(1) == 2**31                        # 1 → 最高位 1
    assert kmp_search('abababa', 'aba') == [0, 2, 4]       # 重叠匹配 3 处
    assert rabin_karp('hello world', 'world') == 6         # 起始下标 6
    print('✅ 位运算与字符串模板测试通过')
