# -*- coding: utf-8 -*-
"""07 章 Pass^k 模拟设计筛选"""
import random

def design_threshold(p, learn=0.2, n=20, max_k=5, seed=42):
    """能力分达门槛判通过；每重试一次能力+learn"""
    rng = random.Random(seed)
    passed = {k: 0 for k in range(1, max_k + 1)}
    for _ in range(n):
        score = rng.random()
        for attempt in range(1, max_k + 1):
            if score >= p:
                for k in range(attempt, max_k + 1):
                    passed[k] += 1
                break
            score += learn
    return {k: passed[k] / n for k in range(1, max_k + 1)}

def design_attempt_prob(p, learn=0.2, n=20, max_k=5, seed=42):
    """每次尝试的成功概率 = p + learn*(attempt-1)，每次新抽随机数判定"""
    rng = random.Random(seed)
    passed = {k: 0 for k in range(1, max_k + 1)}
    for _ in range(n):
        for attempt in range(1, max_k + 1):
            prob = min(1.0, p + learn * (attempt - 1))
            if rng.random() < prob:
                for k in range(attempt, max_k + 1):
                    passed[k] += 1
                break
    return {k: passed[k] / n for k in range(1, max_k + 1)}

print("threshold p=0.4:", {k: f"{v:.3f}" for k, v in design_threshold(0.4).items()})
print("threshold p=0.5:", {k: f"{v:.3f}" for k, v in design_threshold(0.5).items()})
print("attempt_prob p=0.5:", {k: f"{v:.3f}" for k, v in design_attempt_prob(0.5).items()})

# threshold p=0.4 逐任务明细，用于正文讲解
rng = random.Random(42)
print("\nthreshold p=0.4 明细:")
count = 0
for i in range(20):
    score = rng.random()
    for attempt in range(1, 6):
        if score >= 0.4:
            break
        score += 0.2
    print(f"任务{i+1:2d}: 首抽={score - 0.2*(attempt-1):.4f}  第{attempt}次尝试通过")