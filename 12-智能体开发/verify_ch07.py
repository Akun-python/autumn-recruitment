# -*- coding: utf-8 -*-
"""07 章代码输出验证脚本（写完正文后删除）"""
import math
import random

# ---------- 1) Pass@k 无偏估计 ----------
def pass_at_k(n, c, k):
    if n - c < k:          # 失败样本不够 k 个，抽 k 个不可能全军覆没
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)

print("=== 1) Pass@k 无偏估计（n=20, k=5）===")
for n, c in [(20, 4), (20, 2), (20, 8), (20, 1), (20, 5)]:
    naive = 1 - (1 - c / n) ** 5
    unbi = pass_at_k(n, c, 5)
    print(f"n={n} c={c}  Pass@1点估计={c/n:.4f}  朴素估算={naive:.4f}  无偏Pass@5={unbi:.4f}")

# ---------- 2) Pass^k 修复循环模拟 ----------
def pass_k_with_retries(n=20, p=0.5, learn=0.2, max_k=5, seed=42):
    rng = random.Random(seed)
    passed = {k: 0 for k in range(1, max_k + 1)}
    for _ in range(n):
        score = rng.random()          # 任务初始“能力分” 0~1
        for attempt in range(1, max_k + 1):
            if score >= p:            # 能力分 >= 0.5 判通过
                for k in range(attempt, max_k + 1):
                    passed[k] += 1
                break
            score += learn            # 修复一次，能力分 +0.2
    return {k: passed[k] / n for k in range(1, max_k + 1)}

print("\n=== 2) Pass^k（score>=p 判通过, learn=0.2）===")
res = pass_k_with_retries()
for k, v in res.items():
    print(f"Pass^{k} = {v:.3f}")

# ---------- 3) bootstrap 置信区间 ----------
def bootstrap_ci(samples, n_boot=2000, seed=7, alpha=0.05):
    rng = random.Random(seed)
    n = len(samples)
    means = []
    for _ in range(n_boot):
        bag = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(bag) / n)
    means.sort()
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return lo, hi, means[0], means[-1]

print("\n=== 3) bootstrap（20 题, B=2000, seed=7）===")
samples = [1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]  # 13 过 7 挂
lo, hi, mn, mx = bootstrap_ci(samples)
print(f"观测均值={sum(samples)/len(samples):.3f}  bootstrap均值范围=[{mn:.3f},{mx:.3f}]  95%CI=[{lo:.3f},{hi:.3f}]")

# ---------- 4) 换一种 Pass^k 设计试探 ----------
def pass_k_variant(n=20, p=0.5, learn=0.2, max_k=5, seed=42):
    rng = random.Random(seed)
    passed = {k: 0 for k in range(1, max_k + 1)}
    for _ in range(n):
        base = rng.random()
        for attempt in range(1, max_k + 1):
            prob = min(1.0, base + learn * (attempt - 1))
            if rng.random() < prob:
                for k in range(attempt, max_k + 1):
                    passed[k] += 1
                break
    return {k: passed[k] / n for k in range(1, max_k + 1)}

print("\n=== 4) Pass^k variant（base随机+每次随机判定）===")
for k, v in pass_k_variant().items():
    print(f"Pass^{k} = {v:.3f}")

# ---------- 5) 按任务难度 + 阈值判定的设计 ----------
def pass_k_variant2(n=20, p=0.5, learn=0.2, max_k=5, seed=42):
    rng = random.Random(seed)
    passed = {k: 0 for k in range(1, max_k + 1)}
    for _ in range(n):
        diff = rng.random()          # 难度 0~1
        for attempt in range(1, max_k + 1):
            score = p + learn * (attempt - 1) - diff * 0.6
            if score >= p:
                for k in range(attempt, max_k + 1):
                    passed[k] += 1
                break
    return {k: passed[k] / n for k in range(1, max_k + 1)}

print("\n=== 5) Pass^k variant2（难度抵扣）===")
for k, v in pass_k_variant2().items():
    print(f"Pass^{k} = {v:.3f}")
