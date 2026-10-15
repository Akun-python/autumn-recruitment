# -*- coding: utf-8 -*-
"""搜索满足 Pass^1=0.600, Pass^3=1.000, Pass^5=1.000 的设计"""
import random

TARGET = {1: 0.6, 3: 1.0, 5: 1.0}

def design_prob(seed=42, n=20, p=0.5, learn=0.2, max_k=5, ge=False, shift=0):
    """每次尝试重新抽 x；pass if x <(=) p+learn*(attempt-1)"""
    rng = random.Random(seed)
    for _ in range(shift):
        rng.random()
    passed = {k: 0 for k in range(1, max_k + 1)}
    for _ in range(n):
        for attempt in range(1, max_k + 1):
            x = rng.random()
            prob = min(1.0, p + learn * (attempt - 1))
            ok = (x >= prob) if ge else (x < prob)
            if ok:
                for k in range(attempt, max_k + 1):
                    passed[k] += 1
                break
    return {k: round(passed[k] / n, 3) for k in range(1, max_k + 1)}

def design_diff(seed=42, n=20, p=0.5, learn=0.2, max_k=5, mode="thresh", scale=1.0, shift=0):
    """每任务抽一次难度 d；attempt 通过条件按 mode 变"""
    rng = random.Random(seed)
    for _ in range(shift):
        rng.random()
    passed = {k: 0 for k in range(1, max_k + 1)}
    for _ in range(n):
        d = rng.random() * scale
        for attempt in range(1, max_k + 1):
            if mode == "thresh":
                ok = d < p + learn * (attempt - 1)
            elif mode == "prob":  # 每次尝试还抽 x
                x = rng.random()
                ok = x < p + learn * (attempt - 1) - d
            else:
                raise ValueError
            if ok:
                for k in range(attempt, max_k + 1):
                    passed[k] += 1
                break
    return {k: round(passed[k] / n, 3) for k in range(1, max_k + 1)}

def design_score(seed=42, n=20, p=0.5, learn=0.2, max_k=5, scale=1.0, shift=0):
    """每任务抽一次能力分 s；pass if s >= p；否则 s += learn"""
    rng = random.Random(seed)
    for _ in range(shift):
        rng.random()
    passed = {k: 0 for k in range(1, max_k + 1)}
    for _ in range(n):
        s = rng.random() * scale
        for attempt in range(1, max_k + 1):
            if s >= p:
                for k in range(attempt, max_k + 1):
                    passed[k] += 1
                break
            s += learn
    return {k: round(passed[k] / n, 3) for k in range(1, max_k + 1)}

found = []
for p in [round(0.1 + 0.05*i, 2) for i in range(19)]:
    for ge in [False, True]:
        for shift in range(3):
            r = design_prob(p=p, ge=ge, shift=shift)
            if all(abs(r[k] - v) < 1e-6 for k, v in TARGET.items()):
                found.append((f"prob p={p} ge={ge} shift={shift}", r))
    for mode in ["thresh", "prob"]:
        for scale in [1.0, 0.8, 0.6, 0.5]:
            for shift in range(3):
                r = design_diff(p=p, mode=mode, scale=scale, shift=shift)
                if all(abs(r[k] - v) < 1e-6 for k, v in TARGET.items()):
                    found.append((f"diff p={p} mode={mode} scale={scale} shift={shift}", r))
    for scale in [1.0, 1.25]:
        for shift in range(3):
            r = design_score(p=p, scale=scale, shift=shift)
            if all(abs(r[k] - v) < 1e-6 for k, v in TARGET.items()):
                found.append((f"score p={p} scale={scale} shift={shift}", r))

print("found:", len(found))
for name, r in found[:20]:
    print(name, r)

if not found:
    print("none matched; print nearest candidates")
    best = []
    for p in [0.5]:
        for ge in [False, True]:
            for shift in range(2):
                r = design_prob(p=p, ge=ge, shift=shift)
                print(f"prob p={p} ge={ge} shift={shift}:", r)
