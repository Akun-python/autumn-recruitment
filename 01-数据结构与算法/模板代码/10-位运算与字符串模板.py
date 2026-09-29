# -*- coding: utf-8 -*-
"""10-位运算与字符串模板.py —— P2 加分（位技巧 + KMP + 滚动哈希）
配套教学：../教学/12-位运算.ipynb & 13-字符串.ipynb
"""
import math


# ============ 1. 位运算高频技巧（背注释） ============
def bit_tricks():
    x, y, n = 8, 12, 6
    lowbit = n & (-n)                     # 取最低位 1：6(110)→2(010)
    is_pow2 = n > 0 and (n & (n - 1)) == 0   # 2 的幂只有 1 个 1
    count_1 = bin(n).count('1')           # 手写版：while n: n &= n-1; cnt+=1
    # 手写 popcount：
    c = 0
    while n:
        n &= n - 1
        c += 1
    return lowbit, is_pow2, c


def hamming_weight(n):
    """191：统计二进制 1 的个数（n & (n-1) 每次消掉最低位 1）"""
    cnt = 0
    while n:
        n &= n - 1
        cnt += 1
    return cnt


# ============ 2. 只出现一次的数字（136，异或消消乐） ============
def single_number(nums):
    """a^a=0, a^0=a，异或满足交换结合 → 成对的消掉"""
    res = 0
    for x in nums:
        res ^= x
    return res


# ============ 3. 缺失数字（268：xor 补齐下标） ============
def missing_number(nums):
    res = len(nums)
    for i, x in enumerate(nums):
        res ^= i ^ x
    return res


# ============ 4. 异或求两数之和（不用 +） ============
def get_sum(a, b):
    """a^b = 无进位和；(a&b)<<1 = 进位；循环到无进位"""
    while b:
        carry = (a & b) << 1
        a = a ^ b
        b = carry
    return a


# ============ 5. 反转二进制位（190） ============
def reverse_bits(n):
    res = 0
    for _ in range(32):
        res = (res << 1) | (n & 1)
        n >>= 1
    return res


# ============ 6. KMP 字符串匹配（O(n+m)） ============
def kmp_next(pattern):
    """next[i] = pattern[:i] 的最长相等前后缀长度（经典失配跳转表）"""
    m = len(pattern)
    nxt = [0] * m
    j = 0                                 # 已匹配前缀长度
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = nxt[j - 1]                # 回退到次长前后缀
        if pattern[i] == pattern[j]:
            j += 1
        nxt[i] = j
    return nxt


def kmp_search(text, pattern):
    """返回 pattern 在 text 中的所有起始下标"""
    if not pattern:
        return []
    nxt = kmp_next(pattern)
    res = []
    j = 0
    for i, ch in enumerate(text):
        while j > 0 and ch != pattern[j]:
            j = nxt[j - 1]
        if ch == pattern[j]:
            j += 1
        if j == len(pattern):
            res.append(i - j + 1)
            j = nxt[j - 1]                # 继续找重叠匹配
    return res


# ============ 7. 滚动哈希（Rabin-Karp，O(n+m)） ============
def rabin_karp(text, pattern):
    """哈希相等再逐字符验证（防碰撞）"""
    m = len(pattern)
    if m > len(text) or not pattern:
        return []
    base, mod = 131, 10**9 + 7
    hp = 0
    for ch in pattern:
        hp = (hp * base + ord(ch)) % mod
    power = pow(base, m - 1, mod)
    cur = 0
    for i, ch in enumerate(text):
        if i >= m:
            cur = ((cur - ord(text[i - m]) * power) % mod + mod) % mod
        cur = (cur * base + ord(ch)) % mod
        if i >= m - 1 and cur == hp and text[i - m + 1:i + 1] == pattern:
            return i - m + 1
    return -1


# ============ 测试 ============
if __name__ == '__main__':
    assert hamming_weight(6) == 2
    assert single_number([4, 1, 2, 1, 2]) == 4
    assert missing_number([3, 0, 1]) == 2
    assert get_sum(5, 7) == 12
    assert reverse_bits(1) == 2**31
    assert kmp_search('abababa', 'aba') == [0, 2, 4]
    assert rabin_karp('hello world', 'world') == 6
    print('✅ 位运算与字符串模板测试通过')