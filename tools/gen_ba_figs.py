# -*- coding: utf-8 -*-
"""Regenerate all 16 matplotlib figures in 05-八股与工程/images/ with
non-overlapping layouts, and verify each with the bbox overlap checker.

Usage: python tools/gen_ba_figs.py [fig1 fig2 ...]   (no args = all)
"""
import os, sys, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (Rectangle, FancyArrowPatch, Circle,
                                FancyBboxPatch, Polygon)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlap_check import report

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '05-八股与工程', 'images')
os.makedirs(OUT, exist_ok=True)
OUT = os.path.abspath(OUT)

YELLOW = '#fff2cc'; BLUE = '#dae8fc'; GREEN = '#d5e8d4'; ORANGE = '#ffe6cc'
RED = '#f8cecc'; GRAY = '#f5f5f5'; PURPLE = '#e1d5e7'

def new_ax(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.axis('off')
    return fig, ax

def box(ax, x, y, w, h, fc, ec='#555555', lw=1.2, zorder=2, rounded=False):
    if rounded:
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                           fc=fc, ec=ec, lw=lw, zorder=zorder)
    else:
        p = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(p)
    return p

def txt(ax, x, y, s, fs=10, ha='center', va='center', color='#222222',
        zorder=5, weight='normal', rotation=0):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, zorder=zorder,
            fontweight=weight, rotation=rotation)

def arrow(ax, p1, p2, color='#8b5a2b', lw=1.6, style='-|>', ms=14, ls='-',
          zorder=4, shrinkA=0, shrinkB=0, rad=0.0):
    conn = None if rad == 0 else f'arc3,rad={rad}'
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, linestyle=ls, zorder=zorder,
                        shrinkA=shrinkA, shrinkB=shrinkB,
                        connectionstyle=conn)
    ax.add_patch(a)
    return a

def title(ax, s, y=1.0, fs=14):
    txt(ax, 0.5, y, s, fs=fs, weight='bold')

# ---------------------------------------------------------------- 1 GIL
def fig_gil():
    fig, ax = new_ax((9.5, 5.2), (0, 10), (0, 5.6))
    title(ax, 'GIL：同一时刻只有一个线程能执行 Python 字节码', y=5.35)
    txt(ax, 0.7, 4.55, '线程 A', fs=11, weight='bold')
    txt(ax, 0.7, 2.05, '线程 B', fs=11, weight='bold')
    # execution blocks
    segs = [(1.5, 3.6, 'A 持有 GIL 运行', GREEN), (3.9, 1.7, 'B 持有 GIL 运行', GREEN),
            (6.3, 3.6, 'A 持有 GIL 运行', GREEN), (8.7, 1.7, 'B 持有 GIL 运行', GREEN)]
    for x, y, lab, c in segs:
        box(ax, x, y, 2.0, 0.75, c, ec='#4a7c59')
        txt(ax, x + 1.0, y + 0.375, lab, fs=9.5)
    for x, y in [(3.9, 3.6), (8.7, 3.6), (1.5, 1.7), (6.3, 1.7)]:
        box(ax, x, y, 2.0, 0.75, RED, ec='#b85450')
        txt(ax, x + 1.0, y + 0.375, '等待 GIL', fs=9.5, color='#922b21')
    # switches
    for x in [3.5, 5.9, 8.3]:
        ax.plot([x, x], [0.9, 4.95], color='#888888', lw=1.0, ls=':', zorder=1)
    # GIL lock icon center
    for x in [3.5, 5.9]:
        c = Circle((x, 2.75), 0.22, fc='#f8cecc', ec='#c0392b', lw=1.4, zorder=4)
        ax.add_patch(c)
        txt(ax, x, 2.75, 'GIL', fs=7, weight='bold', color='#922b21', zorder=5)
    txt(ax, 6.2, 5.1, '时间 →', fs=9, color='#666666')
    box(ax, 1.3, 0.3, 7.4, 0.8, BLUE)
    txt(ax, 5.0, 0.7, 'GIL 是 CPython 互斥锁：多线程可并发（I/O 友好）但无法并行执行字节码', fs=10)
    return fig

# ---------------------------------------------------------------- 2 process states
def fig_process_states():
    fig, ax = new_ax((10.0, 5.6), (0, 10.4), (0, 5.9))
    title(ax, '进程五状态模型（经典三态 + 新建 + 终止）', y=5.6)
    W, H = 1.7, 0.8
    # 新建 left-bottom, 就绪 left-top, 运行 center-top, 阻塞 center-bottom, 终止 right
    pos = {'新建': (0.4, 2.6), '就绪': (2.9, 4.1), '运行': (5.6, 4.1),
           '阻塞': (5.6, 1.5), '终止': (8.3, 3.3)}
    colors = {'新建': ORANGE, '就绪': YELLOW, '运行': GREEN, '阻塞': RED, '终止': GRAY}
    for k, (x, y) in pos.items():
        box(ax, x, y, W, H, colors[k], ec='#555555', rounded=True)
        txt(ax, x + W/2, y + H/2, k, fs=13, weight='bold')
    cx, cy = {}, {}
    for k, (x, y) in pos.items():
        cx[k] = x + W/2; cy[k] = y + H/2
    # 新建 -> 就绪 (提交)
    arrow(ax, (cx['新建'], cy['新建'] + H/2), (pos['就绪'][0], cy['就绪'] - H/2 + 0.1),
          color='#555555', rad=0.25)
    txt(ax, 1.95, 3.62, '提交', fs=9.5)
    # 就绪 -> 运行 (调度)
    arrow(ax, (pos['就绪'][0] + W, cy['就绪']), (pos['运行'][0], cy['运行']),
          color='#555555')
    txt(ax, 5.1, 4.9, '调度', fs=9.5)
    # 运行 -> 就绪 (时间片用完)  curved below
    arrow(ax, (pos['运行'][0] + 0.25, pos['运行'][1]),
          (pos['就绪'][0] + W - 0.25, pos['就绪'][1]), color='#555555', rad=-0.6)
    txt(ax, 4.0, 3.35, '时间片用完', fs=9.5)
    # 运行 -> 阻塞 (等待事件)
    arrow(ax, (cx['运行'], pos['运行'][1]), (cx['阻塞'], pos['阻塞'][1] + H),
          color='#555555')
    txt(ax, cx['运行'] + 0.65, 2.6, '等待事件', fs=9.5)
    # 阻塞 -> 就绪 (事件完成)  curved right then up
    arrow(ax, (pos['阻塞'][0] + W - 0.2, pos['阻塞'][1] + H / 2),
          (pos['运行'][0] - 0.1, cy['就绪'] - 0.1), color='#555555', rad=0.4)
    txt(ax, 5.1, 3.0, '事件完成', fs=9.5)
    # 运行 -> 终止
    arrow(ax, (pos['运行'][0] + W, cy['运行'] - 0.1), (pos['终止'][0], cy['终止'] + 0.15),
          color='#555555', rad=-0.3)
    txt(ax, 7.85, 4.15, '完成/退出', fs=9.5)
    txt(ax, 5.2, 0.55, '挂起态（就绪挂起/阻塞挂起）：内存不足时进程换出到磁盘（对调）', fs=9.5, color='#555555')
    return fig

# ---------------------------------------------------------------- 3 paging
def fig_paging():
    fig, ax = new_ax((11.5, 6.0), (0, 12), (0, 6.2))
    title(ax, '虚拟内存分页：虚拟页 → 页表 → 物理帧', y=5.9)
    row_y = {i: 4.35 - i * 0.66 for i in range(6)}   # VP0..VP5
    row_h = 0.52
    # ---- 虚拟地址空间
    txt(ax, 1.0, 5.35, '虚拟地址空间', fs=11, weight='bold')
    names = ['VP0', 'VP1', 'VP2', 'VP3', 'VP4(缺页)', 'VP5']
    for i, nm in enumerate(names):
        y = row_y[i]
        fc = RED if i == 4 else YELLOW
        box(ax, 0.3, y, 1.5, row_h, fc, ec='#8a7b3f' if i != 4 else '#b85450')
        txt(ax, 1.05, y + row_h/2, nm, fs=10, weight='bold',
            color='#922b21' if i == 4 else '#222222')
    # ---- 页表
    txt(ax, 4.0, 5.35, '页表（虚拟页号 → 物理帧号）', fs=11, weight='bold')
    box(ax, 2.6, 4.9, 2.8, 0.32, '#e8e8e8', ec='#999999')
    txt(ax, 3.35, 5.06, '有效位', fs=9)
    txt(ax, 4.4, 5.06, '帧号', fs=9)
    mapping = [('VP0', 'F12'), ('VP1', 'F0'), ('VP2', 'F5'), ('VP3', 'F19'),
               ('VP4', '缺页'), ('VP5', 'F46')]
    row = {}
    for i, (v, f) in enumerate(mapping):
        y = row_y[i]
        row[v] = y
        valid = '1' if f != '缺页' else '0'
        fc = '#ffe9e9' if f == '缺页' else '#ffffff'
        box(ax, 2.6, y, 2.8, row_h, fc, ec='#888888')
        txt(ax, 3.35, y + row_h/2, valid, fs=10)
        txt(ax, 4.4, y + row_h/2, f, fs=10, weight='bold',
            color='#c0392b' if f == '缺页' else '#222222')
    # ---- 物理内存
    txt(ax, 8.3, 5.35, '物理内存（帧）', fs=11, weight='bold')
    frames = {'F12': row_y[0], 'F0': row_y[1], 'F5': row_y[2], 'F19': row_y[3],
              'F7(空闲)': row_y[4], 'F46': row_y[5]}
    for f, y in frames.items():
        fc = '#f0f0f0' if '空闲' in f else GREEN
        box(ax, 7.0, y, 2.6, row_h, fc, ec='#4a7c59' if '空闲' not in f else '#999999')
        txt(ax, 8.3, y + row_h/2, f, fs=10, weight='bold',
            color='#666666' if '空闲' in f else '#1d5c33')
    # arrows: valid pages -> frames
    pairs = [('VP0', 'F12'), ('VP1', 'F0'), ('VP2', 'F5'), ('VP3', 'F19'), ('VP5', 'F46')]
    for v, f in pairs:
        arrow(ax, (5.4, row[v] + row_h/2), (7.0, frames[f] + row_h/2),
              color='#8b5a2b', lw=1.6, shrinkB=4)
    # 缺页: dashed red from page table down to note
    arrow(ax, (5.9, row['VP4'] + row_h/2), (5.9, 1.0), color='#c0392b', lw=1.8,
          style='-|>', ls=(0, (5, 3)), shrinkB=4)
    txt(ax, 6.2, 1.98, '缺页中断', fs=9, color='#c0392b', ha='left')
    # bottom note
    box(ax, 0.4, 0.25, 11.1, 0.62, '#fdf6e3', ec='#cccccc')
    txt(ax, 5.95, 0.56,
        '缺页流程：CPU 访问不在内存的页 → 缺页中断 → 调页入帧 → 更新页表 → 恢复指令执行', fs=9.5)
    return fig

# ---------------------------------------------------------------- 4 TCP handshake
def fig_tcp_handshake():
    fig, ax = new_ax((9.5, 5.6), (0, 10), (0, 5.85))
    title(ax, 'TCP 三次握手', y=5.55)
    XL, XR = 1.9, 8.1          # lifelines
    box_w, box_h = 1.7, 0.42
    # lifelines
    ax.plot([XL, XL], [0.75, 5.1], color='#666666', lw=1.4, zorder=1)
    ax.plot([XR, XR], [0.75, 5.1], color='#666666', lw=1.4, zorder=1)
    txt(ax, XL, 5.28, '客户端', fs=12, weight='bold')
    txt(ax, XR, 5.28, '服务端', fs=12, weight='bold')
    # state boxes LEFT of client lifeline / RIGHT of server lifeline
    def states(side, items):
        for name, y in items:
            if side == 'L':
                box(ax, XL - 0.15 - box_w, y, box_w, box_h, BLUE, rounded=True)
                txt(ax, XL - 0.15 - box_w/2, y + box_h/2, name, fs=9)
            else:
                box(ax, XR + 0.15, y, box_w, box_h, BLUE, rounded=True)
                txt(ax, XR + 0.15 + box_w/2, y + box_h/2, name, fs=9)
    # y bands: arrows at 4.35, 3.20, 2.05
    states('L', [('CLOSED', 4.72), ('SYN_SENT', 3.62), ('ESTABLISHED', 1.30)])
    states('R', [('LISTEN', 4.72), ('SYN_RCVD', 3.62), ('ESTABLISHED', 1.30)])
    # ① SYN
    arrow(ax, (XL, 4.35), (XR, 4.35), color='#c0392b', lw=2.0)
    txt(ax, 5.0, 4.52, '① SYN(seq=x)', fs=11, weight='bold', color='#c0392b')
    # ② SYN+ACK
    arrow(ax, (XR, 3.20), (XL, 3.20), color='#2e75b6', lw=2.0)
    txt(ax, 5.0, 3.37, '② SYN+ACK(seq=y, ack=x+1)', fs=11, weight='bold', color='#2e75b6')
    # ③ ACK
    arrow(ax, (XL, 2.05), (XR, 2.05), color='#2e75b6', lw=2.0)
    txt(ax, 5.0, 2.22, '③ ACK(seq=x+1, ack=y+1)', fs=11, weight='bold', color='#2e75b6')
    txt(ax, 5.0, 0.3, '为什么 3 次：确认双方收发能力 + 防止历史迟到 SYN 建立无效连接', fs=9.5, color='#555555')
    return fig

# ---------------------------------------------------------------- 5 TCP fourwave
def fig_tcp_fourwave():
    fig, ax = new_ax((9.5, 5.9), (0, 10), (0, 6.15))
    title(ax, 'TCP 四次挥手（全双工 → 双向独立关闭）', y=5.85)
    XL, XR = 1.9, 8.1
    box_w, box_h = 1.75, 0.4
    ax.plot([XL, XL], [0.7, 5.3], color='#666666', lw=1.4, zorder=1)
    ax.plot([XR, XR], [0.7, 5.3], color='#666666', lw=1.4, zorder=1)
    txt(ax, XL, 5.55, '主动关闭方 A', fs=12, weight='bold')
    txt(ax, XR, 5.55, '被动关闭方 B', fs=12, weight='bold')
    def states(side, items):
        for name, y in items:
            if side == 'L':
                box(ax, XL - 0.15 - box_w, y, box_w, box_h, BLUE, rounded=True)
                txt(ax, XL - 0.15 - box_w/2, y + box_h/2, name, fs=8.3)
            else:
                box(ax, XR + 0.15, y, box_w, box_h, BLUE, rounded=True)
                txt(ax, XR + 0.15 + box_w/2, y + box_h/2, name, fs=8.3)
    # arrows at y: 4.75, 3.95, 3.15, 2.35  -- states sit between arrows
    states('L', [('ESTABLISHED', 4.95), ('FIN_WAIT_1', 4.15), ('FIN_WAIT_2', 3.35),
                 ('TIME_WAIT(2MSL)', 1.1)])
    states('R', [('ESTABLISHED', 4.95), ('CLOSE_WAIT', 4.15), ('LAST_ACK', 2.55)])
    txt(ax, XR + 0.15 + box_w/2, 1.1 + box_h/2, 'CLOSED', fs=8.3)
    box(ax, XR + 0.15, 1.1, box_w, box_h, GRAY, rounded=True)
    # ① FIN
    arrow(ax, (XL, 4.75), (XR, 4.75), color='#c0392b', lw=2.0)
    txt(ax, 5.0, 4.92, '① FIN(seq=u)', fs=11, weight='bold', color='#c0392b')
    # ② ACK
    arrow(ax, (XR, 3.95), (XL, 3.95), color='#2e75b6', lw=2.0)
    txt(ax, 5.0, 4.12, '② ACK(ack=u+1)', fs=11, weight='bold', color='#2e75b6')
    # ③ FIN
    arrow(ax, (XR, 3.15), (XL, 3.15), color='#c0392b', lw=2.0)
    txt(ax, 5.0, 3.32, '③ FIN(seq=v)', fs=11, weight='bold', color='#c0392b')
    # ④ ACK
    arrow(ax, (XL, 2.35), (XR, 2.35), color='#2e75b6', lw=2.0)
    txt(ax, 5.0, 2.52, '④ ACK(ack=v+1)', fs=11, weight='bold', color='#2e75b6')
    txt(ax, 5.0, 0.3, 'TIME_WAIT 作用：① 响应对方重传的 FIN；② 等旧报文消散，防止新连接串扰', fs=9.5,
        color='#555555')
    return fig

# ---------------------------------------------------------------- 6 congestion
def fig_congestion():
    rounds = list(range(1, 15))
    cwnd = [1, 2, 4, 8, 16, 17, 18, 19, 20, 11, 12, 13, 14, 15]
    ssthresh = 16
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    segs = [('慢启动', (0, 4), '#2e75b6'), ('拥塞避免', (4, 8), '#c55a11'),
            ('丢包 → 快恢复', (8, 9), '#c0392b'), ('拥塞避免', (9, 14), '#c55a11')]
    for name, (a, b), c in segs:
        ax.plot(rounds[a:b], cwnd[a:b], 'o-', color=c, lw=2.2, ms=4, label=name, zorder=3)
    ax.axhline(ssthresh, color='gray', ls='--', lw=1.2, zorder=2)
    txt(ax, 1.2, ssthresh + 1.2, f'ssthresh = {ssthresh}', fs=9, color='gray', ha='center')
    ax.annotate('3 个冗余 ACK → ssthresh = cwnd/2, cwnd = ssthresh+3',
                xy=(9, cwnd[8]), xytext=(4.6, 4.6),
                arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.3),
                fontsize=9, color='#c0392b')
    ax.set_xlabel('RTT 轮次'); ax.set_ylabel('cwnd（拥塞窗口大小）')
    ax.set_title('TCP 拥塞控制：慢启动 → 拥塞避免 → 快恢复（Reno）')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=9)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------- 7 HTTP/2
def fig_http2():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
    # left: HTTP/1.1
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    txt(ax, 5, 5.6, 'HTTP/1.1：请求串行（队头阻塞）', fs=11, weight='bold')
    box(ax, 0.6, 0.55, 8.8, 0.5, '#bbbbbb', ec='#999999')
    txt(ax, 5, 0.8, 'TCP 连接', fs=9, color='#444444')
    labels = [('请求 1', 4.9, 4.75, '#dae8fc'), ('响应 1', 4.9, 3.95, '#d5e8d4'),
              ('请求 2', 4.9, 3.15, '#dae8fc'), ('响应 2', 4.9, 2.35, '#d5e8d4')]
    for s, x, y, c in labels:
        box(ax, x - 1.3, y, 2.6, 0.5, c, ec='#666666')
        txt(ax, x, y + 0.25, s, fs=9.5)
    # interleaving arrows at x=3.2 (left of boxes)
    for y1, y2 in [(4.75, 4.45), (3.95, 3.65), (3.15, 2.85)]:
        arrow(ax, (2.9, y1 + 0.25), (2.9, y2 + 0.25), color='#666666', ms=9)
    txt(ax, 8.2, 3.95, '必须等前一个\n完成', fs=8.5, color='#922b21', ha='left')
    # right: HTTP/2
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    txt(ax, 5, 5.6, 'HTTP/2：一条连接多流交错（多路复用）', fs=11, weight='bold')
    box(ax, 0.6, 0.55, 8.8, 0.5, '#bbbbbb', ec='#999999')
    txt(ax, 5, 0.8, 'TCP 连接', fs=9, color='#444444')
    items = [('流1', '#dae8fc'), ('流2', '#d5e8d4'), ('流3', '#ffe6cc'),
             ('流1', '#dae8fc'), ('流2', '#d5e8d4'), ('流3', '#ffe6cc'),
             ('流1', '#dae8fc'), ('流2', '#d5e8d4'), ('流3', '#ffe6cc'),
             ('流1', '#dae8fc'), ('流2', '#d5e8d4')]
    x = 1.0
    for s, c in items:
        box(ax, x, 3.3, 0.8, 1.0, c, ec='#666666')
        txt(ax, x + 0.4, 3.8, s, fs=8.5)
        x += 0.87
    txt(ax, 5, 4.75, '帧交错传输，互不等待', fs=9.5, color='#555555')
    for i in range(3):
        box(ax, 0.4 + i * 3.15, 1.8, 2.9, 0.5, '#ffffff', ec='#999999', lw=1.1)
        txt(ax, 1.85 + i * 3.15, 2.05, f'请求流 {i+1}', fs=8.5)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------- 8 B+ tree
def fig_bplus():
    fig, ax = new_ax((11.5, 5.8), (0, 12.6), (0, 6.0))
    title(ax, 'B+ 树：非叶子只存 key，叶子存数据并用链表串联', y=5.72)
    # root
    box(ax, 5.2, 4.8, 2.2, 0.6, BLUE, ec='#333333')
    txt(ax, 6.3, 5.1, '17 | 35', fs=11, weight='bold')
    txt(ax, 4.75, 5.1, '根（索引）', fs=8.5, color='#555555', ha='right')
    # internal nodes
    ints = [(1.5, 3.4, '5'), (5.5, 3.4, '28'), (9.5, 3.4, '65')]
    for x, y, k in ints:
        box(ax, x, y, 1.5, 0.6, BLUE, ec='#333333')
        txt(ax, x + 0.75, y + 0.3, k, fs=10, weight='bold')
    # leaves
    leaves = [(0.4, 1.6, '3 | 5'), (3.0, 1.6, '8 | 11 | 17'), (5.6, 1.6, '20 | 28'),
              (8.2, 1.6, '31 | 35'), (10.6, 1.6, '41 | 65')]
    for x, y, k in leaves:
        box(ax, x, y, 2.0, 0.6, GREEN, ec='#333333')
        txt(ax, x + 1.0, y + 0.3, k, fs=9.5, weight='bold')
    # edges
    for x in [1.5, 5.5, 9.5]:
        ax.plot([6.3, x + 0.75], [4.8, 4.0], color='#888888', lw=1.4, zorder=1)
    for ix, lx in [(1.5, 0.4), (1.5, 3.0), (5.5, 3.0), (5.5, 5.6), (9.5, 8.2), (9.5, 10.6)]:
        ax.plot([ix + 0.75, lx + 1.0], [3.4, 2.2], color='#888888', lw=1.2, zorder=1)
    # leaf chain
    for x in [2.4, 5.0, 7.6, 10.2]:
        arrow(ax, (x, 1.9), (x + 0.6, 1.9), color='#c0392b', lw=1.8, ms=12)
    txt(ax, 5.0, 1.05, '叶子节点双向链表 → 范围查询 / 排序只需顺序扫描', fs=9.5,
        color='#c0392b', weight='bold')
    txt(ax, 2.75, 4.15, '内部节点（仅 key）', fs=8.5, color='#555555')
    txt(ax, 8.9, 2.55, '叶子（key + 数据）', fs=8.5, color='#555555')
    return fig

# ---------------------------------------------------------------- 9 MVCC
def fig_mvcc():
    fig, ax = new_ax((11.5, 5.2), (0, 12), (0, 5.4))
    title(ax, 'MVCC 版本链：row + roll_pointer → undo log 旧版本', y=5.15)
    # current row
    box(ax, 0.5, 3.7, 3.6, 0.9, GREEN, ec='#333333')
    txt(ax, 2.3, 4.35, '行数据（最新版）', fs=10.5, weight='bold')
    txt(ax, 2.3, 3.95, 'trx_id = 120', fs=9.5)
    # undo versions
    vx = [5.6, 8.0, 10.4]
    for x, t1, t2 in zip(vx, ['版本 1', '版本 2', '版本 3'], ['trx_id = 110', 'trx_id = 100', 'trx_id = 90']):
        box(ax, x - 1.0, 3.7, 2.0, 0.9, '#fdf2cc', ec='#8a7b3f')
        txt(ax, x, 4.35, t1, fs=10, weight='bold')
        txt(ax, x, 3.95, t2, fs=9.5)
    # roll_pointer arrows between boxes
    for x1, x2 in [(4.1, 4.6), (6.6, 7.0), (9.0, 9.4)]:
        arrow(ax, (x1, 4.15), (x2, 4.15), color='#8a7b3f', lw=2.0, ms=14)
    txt(ax, 4.35, 3.3, 'roll_pointer', fs=8, color='#8a7b3f')
    txt(ax, 6.8, 3.3, 'roll_pointer', fs=8, color='#8a7b3f')
    txt(ax, 9.2, 3.3, 'roll_pointer', fs=8, color='#8a7b3f')
    # ReadView note
    box(ax, 0.5, 0.6, 11.0, 1.6, '#eef3fb', ec='#888888')
    txt(ax, 6.0, 1.9, '快照读可见性判断（ReadView）', fs=10.5, weight='bold')
    txt(ax, 6.0, 1.35, 'trx_id < min(活跃) 或 = 自己 → 可见；trx_id ∈ 活跃列表 → 不可见（沿链找更旧版本）', fs=9.5)
    txt(ax, 6.0, 0.9, 'RC：每次 SELECT 新 ReadView；RR：事务内第一次 SELECT 生成后复用', fs=9.5)
    return fig

# ---------------------------------------------------------------- 10 redis cache
def fig_redis_cache():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), dpi=150)
    titles = ['缓存穿透', '缓存击穿', '缓存雪崩']
    notes = ['查询不存在 key，\n每次打 DB', '热点 key 过期瞬间，\n大量请求打 DB', '大量 key 同时过期，\nDB 被打爆']
    for ax, t, note in zip(axes, titles, notes):
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
        txt(ax, 5, 5.5, t, fs=12, weight='bold')
        box(ax, 0.7, 3.6, 3.2, 1.1, YELLOW, rounded=True)
        txt(ax, 2.3, 4.15, '请求', fs=10.5)
        box(ax, 5.4, 3.6, 3.9, 1.1, ORANGE, rounded=True)
        txt(ax, 7.35, 4.15, '缓存 Redis', fs=10.5)
        box(ax, 5.4, 1.2, 3.9, 1.1, GREEN, rounded=True)
        txt(ax, 7.35, 1.75, '数据库 MySQL', fs=10.5)
        arrow(ax, (3.9, 4.15), (5.4, 4.15), color='#555555', lw=1.6, shrinkB=4)
        txt(ax, 4.55, 4.35, '查缓存', fs=8.5, color='#555555')
        if t == '缓存穿透':
            arrow(ax, (7.35, 3.6), (7.35, 2.3), color='#c0392b', lw=1.8, shrinkA=4, shrinkB=4)
            txt(ax, 8.6, 3.0, '缓存无 key，\n直接打 DB', fs=8.5, color='#c0392b', ha='left')
        else:
            txt(ax, 8.6, 3.0, '多请求同时\n穿透', fs=8.5, color='#c0392b', ha='left')
            for dx in [-0.45, 0.0, 0.45]:
                arrow(ax, (2.3 + dx, 3.6), (7.35, 2.3), color='#c0392b', lw=1.2,
                      ms=11, shrinkA=4, shrinkB=4)
        txt(ax, 5, 0.35, note, fs=9, color='#555555')
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------- 11 bloom
def fig_bloom():
    fig, ax = new_ax((10.5, 4.9), (0, 11), (0, 5.2))
    title(ax, '布隆过滤器：k 个哈希函数映射到位数组', y=4.9)
    items = [('a', [2, 6, 9]), ('b', [3, 6, 10]), ('c', [1, 4, 8])]
    colors = ['#c0392b', '#2e75b6', '#2e7d32']
    nbits = 12
    for i in range(nbits):
        x = 1.0 + i * 0.72
        box(ax, x, 2.0, 0.6, 0.6, '#ffffff', ec='#888888', lw=1)
    txt(ax, 1.0 + nbits * 0.72 / 2, 1.52, '位数组（m 位）', fs=9, color='#555555')
    setmap = {}
    for (n, idxs), c in zip(items, colors):
        for i in idxs:
            setmap[i] = c
    for i, c in setmap.items():
        box(ax, 1.0 + i * 0.72, 2.0, 0.6, 0.6, c, ec=c, lw=1.2)
        txt(ax, 1.0 + i * 0.72 + 0.3, 2.3, '1', fs=9, color='white', weight='bold')
    # items and hash arrows (above bit array)
    for k, (n, idxs), c in zip(range(3), items, colors):
        y = 4.25 - k * 0.42
        box(ax, 0.3, y - 0.18, 0.9, 0.36, c, ec=c)
        txt(ax, 0.75, y, n, fs=11, weight='bold', color='white')
        txt(ax, 1.7, y, 'h1 h2 h3', fs=9, color='#555555')
        for i in idxs:
            ax.plot([2.1, 1.0 + i * 0.72 + 0.3], [y - 0.15, 2.3], color=c,
                    lw=1.0, alpha=0.7, zorder=1)
    # query example row (below bit array, clear of everything)
    box(ax, 0.9, 0.72, 0.9, 0.36, '#888888')
    txt(ax, 1.35, 0.9, '查询 x', fs=9, weight='bold', color='white')
    txt(ax, 2.1, 0.9, '→ 3 位都为 1 → 可能存在（有 1 位为 0 → 一定不存在）', fs=9, ha='left')
    txt(ax, 5.9, 0.2, '特点：不能删除、有误判率（Counting BF 可删）；RedisBloom 提供 BF.ADD / BF.EXISTS',
        fs=8.5, color='#555555')
    return fig

# ---------------------------------------------------------------- 12 CAP
def fig_cap():
    fig, ax = new_ax((8.5, 5.2), (0, 10), (0, 5.8))
    title(ax, 'CAP 理论：分区时 C 与 A 只能二选一', y=5.5)
    C = (5.0, 3.9); A = (1.3, 0.9); P = (8.7, 0.9)
    tri = Polygon([C, A, P], closed=True, fill=False, ec='#333333', lw=2, zorder=2)
    ax.add_patch(tri)
    txt(ax, C[0], 4.35, 'C 一致性', fs=12, weight='bold')
    txt(ax, C[0], C[1] - 0.45, 'Consistency', fs=8.5, color='#666666')
    txt(ax, A[0] - 0.1, 0.45, 'A 可用性', fs=12, weight='bold')
    txt(ax, A[0], -0.1, 'Availability', fs=8.5, color='#666666')
    txt(ax, P[0] + 0.15, 0.45, 'P 分区容错', fs=12, weight='bold')
    txt(ax, P[0] + 0.2, -0.1, 'Partition tolerance', fs=8.5, color='#666666')
    txt(ax, 3.15, 2.55, 'CA（单机/无分区）', fs=9, color='#555555', rotation=-53)
    txt(ax, 6.85, 2.55, 'CP：ZooKeeper/etcd\n（分区时拒写）', fs=9, color='#555555', rotation=-53)
    txt(ax, 4.7, 2.5, 'AP：Eureka / Redis\n（分区时旧值应答）', fs=9, color='#555555')
    box(ax, 1.3, 4.6, 7.4, 0.6, '#fdf6e3', ec='#cccccc')
    txt(ax, 5.0, 4.9, '核心：网络分区不可避免 → 必须选 P，再在 C 与 A 之间权衡', fs=10, weight='bold')
    return fig

# ---------------------------------------------------------------- 13 token vs leaky bucket
def fig_token_bucket():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=150)
    # token bucket
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    txt(ax, 5, 5.5, '令牌桶（允许突发）', fs=12, weight='bold')
    box(ax, 2.0, 1.0, 6.0, 3.2, '#ffffff', ec='#333333', lw=1.6)
    txt(ax, 5, 3.95, '桶容量 capacity', fs=8.5, color='#555555')
    tok = [('T', '#c0392b'), ('T', '#c0392b'), ('T', '#c0392b'),
           ('T', '#c0392b'), ('T', '#e0e0e0')]
    for i, (s, c) in enumerate(tok):
        col = i % 3; rowi = i // 3
        cx = 2.9 + col * 1.4; cy = 3.25 - rowi * 1.05
        cc = Circle((cx, cy), 0.36, fc=c, ec='#922b21' if c != '#e0e0e0' else '#999999',
                    zorder=3)
        ax.add_patch(cc)
        txt(ax, cx, cy, s, fs=8, color='white' if c != '#e0e0e0' else '#666666',
            weight='bold', zorder=4)
    txt(ax, 7.6, 2.2, '令牌', fs=9, color='#555555')
    arrow(ax, (1.0, 4.1), (2.0, 3.3), color='#2e75b6', lw=2)
    txt(ax, 1.55, 4.55, '按速率 r 放入', fs=9, color='#2e75b6')
    arrow(ax, (5.0, 1.0), (5.0, 0.25), color='#2e75b6', lw=2)
    txt(ax, 5.0, 0.15, '请求取令牌（有则放行）', fs=9, color='#2e75b6', va='top')
    # leaky bucket
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    txt(ax, 5, 5.5, '漏桶（输出平滑）', fs=12, weight='bold')
    box(ax, 2.0, 1.0, 6.0, 3.2, '#ffffff', ec='#333333', lw=1.6)
    txt(ax, 5, 3.95, '队列（请求堆积）', fs=8.5, color='#555555')
    for i in range(3):
        box(ax, 2.7 + i * 1.4, 1.6, 1.1, 0.8, '#dae8fc', ec='#666666', zorder=3)
        txt(ax, 3.25 + i * 1.4, 2.0, f'R{i+1}', fs=9.5, zorder=4)
    arrow(ax, (1.0, 4.1), (2.0, 3.1), color='#2e75b6', lw=2)
    txt(ax, 1.55, 4.55, '请求入桶', fs=9, color='#2e75b6')
    arrow(ax, (5.0, 1.0), (5.0, 0.25), color='#c0392b', lw=2)
    txt(ax, 5.0, 0.15, '恒定速率流出（不突发）', fs=9, color='#c0392b', va='top')
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------- 14 consistent hash
def fig_chash():
    fig, ax = new_ax((9.0, 5.4), (0, 10), (0, 6.0))
    title(ax, '一致性哈希环：key 顺时针找最近节点', y=5.7)
    cx, cy, R = 5.0, 3.2, 1.9
    ring = Circle((cx, cy), R, fill=False, ec='#333333', lw=2.2, zorder=2)
    ax.add_patch(ring)
    spec = [('Node A', 30, 'center'), ('Node B', 150, 'right'), ('Node C', 270, 'center')]
    for name, deg, ha in spec:
        a = math.radians(deg)
        x, y = cx + R * math.cos(a), cy + R * math.sin(a)
        r = Rectangle((x - 0.24, y - 0.24), 0.48, 0.48, fc='#2e75b6', ec='#1f4e79',
                      lw=1.4, zorder=4)
        ax.add_patch(r)
        txt(ax, x, y, name[-1], fs=10, weight='bold', color='white', zorder=5)
        if deg == 270:
            txt(ax, x, y - 0.42, name, fs=10, color='#1f4e79', ha='center')
        else:
            lx = cx + (R + 0.8) * math.cos(a)
            ly = cy + (R + 0.8) * math.sin(a)
            txt(ax, lx, ly, name, fs=10, color='#1f4e79', ha=ha)
    keys = [('k1', 60), ('k2', 110), ('k3', 200), ('k4', 300), ('k5', 340)]
    for name, deg in keys:
        a = math.radians(deg)
        x, y = cx + (R - 0.35) * math.cos(a), cy + (R - 0.35) * math.sin(a)
        c = Circle((x, y), 0.14, fc='#c0392b', ec='#922b21', zorder=4)
        ax.add_patch(c)
        kx = cx + (R - 0.95) * math.cos(a)
        ky = cy + (R - 0.95) * math.sin(a)
        txt(ax, kx, ky, name, fs=8.5, color='#922b21')
    txt(ax, 5.0, 4.7, '0 ~ 2³²-1', fs=9, color='#888888')
    txt(ax, 5.0, 0.25, '节点增减只影响环上相邻区间的 key（约 1/N 迁移）；虚拟节点解决数据倾斜', fs=9,
        color='#555555')
    return fig

# ---------------------------------------------------------------- 15 git branch
def fig_git():
    fig, ax = new_ax((10.0, 5.4), (0, 11), (0, 5.6))
    title(ax, 'Git 分支：feature 从 main 分出，merge 合并回 main', y=5.35)
    main_pts = [(1.0, 0.9), (2.7, 0.9), (4.4, 0.9), (6.1, 0.9), (9.6, 0.9)]
    feat_pts = [(2.7, 3.3), (4.4, 3.3), (6.1, 2.1)]
    for x, y in main_pts:
        c = Circle((x, y), 0.28, fc='#d5e8d4', ec='#4a7c59', lw=1.6, zorder=4)
        ax.add_patch(c)
    for x, y in feat_pts:
        c = Circle((x, y), 0.28, fc='#dae8fc', ec='#2e75b6', lw=1.6, zorder=4)
        ax.add_patch(c)
    c = Circle((8.2, 0.9), 0.28, fc='#ffe6cc', ec='#c55a11', lw=1.8, zorder=5)
    ax.add_patch(c)
    for (x1, y1), (x2, y2) in [(main_pts[0], main_pts[1]), (main_pts[1], main_pts[2]),
                              (main_pts[2], main_pts[3]), (main_pts[3], (8.2, 0.9)),
                              ((8.2, 0.9), main_pts[4])]:
        arrow(ax, (x1 + 0.28, y1), (x2 - 0.28, y2), color='#4a7c59', lw=1.8, ms=12)
    for (x1, y1), (x2, y2) in [(main_pts[1], feat_pts[0]), (feat_pts[0], feat_pts[1]),
                               (feat_pts[1], feat_pts[2]), (feat_pts[2], (8.2, 0.9))]:
        arrow(ax, (x1 + 0.28, y1), (x2 - 0.28, y2), color='#2e75b6', lw=1.8, ms=12)
    txt(ax, 1.0, 0.45, 'c0', fs=9, ha='center')
    txt(ax, 2.7, 0.45, 'c1', fs=9, ha='center')
    txt(ax, 4.4, 0.45, 'c2', fs=9, ha='center')
    txt(ax, 9.6, 0.45, 'main', fs=10, weight='bold', color='#4a7c59')
    txt(ax, 2.7, 3.75, 'c3 (feature)', fs=9, color='#2e75b6', ha='center')
    txt(ax, 6.1, 2.55, 'c4 (feature)', fs=9, color='#2e75b6', ha='center')
    txt(ax, 4.4, 3.75, 'feature 分支', fs=9.5, color='#2e75b6', ha='center')
    txt(ax, 8.2, 1.4, 'merge 提交', fs=9, color='#c55a11', ha='center')
    txt(ax, 5.5, 4.9, 'git merge --no-ff 保留分支历史；fast-forward 则直接前移 main 指针', fs=9.5,
        color='#555555')
    return fig

# ---------------------------------------------------------------- 16 LRU
def fig_lru():
    fig, ax = new_ax((10.5, 4.9), (0, 11), (0, 5.3))
    title(ax, 'LRU Cache：哈希表 O(1) 定位 + 双向链表记录访问顺序', y=5.05)
    box(ax, 0.4, 3.2, 3.0, 1.4, YELLOW, ec='#333333')
    txt(ax, 1.9, 4.35, '哈希表 dict', fs=11, weight='bold')
    rows = [('key 1', 'node1'), ('key 2', 'node2'), ('key 3', 'node3')]
    for i, (k, v) in enumerate(rows):
        y = 4.1 - i * 0.38
        txt(ax, 1.15, y, k, fs=9, ha='center')
        txt(ax, 2.65, y, '→ ' + v, fs=9, ha='center')
    box(ax, 4.6, 3.4, 6.0, 1.2, BLUE, ec='#333333')
    txt(ax, 7.6, 4.42, '双向链表（最近使用 → 最久未使用）', fs=10, weight='bold')
    nodes = [('1', 'node1'), ('2', 'node2'), ('3', 'node3')]
    xs = [5.3, 6.8, 8.3]
    for x, (k, name) in zip(xs, nodes):
        c = Circle((x, 3.75), 0.32, fc='#ffffff', ec='#333333', lw=1.6, zorder=4)
        ax.add_patch(c)
        txt(ax, x, 3.75, k, fs=11, weight='bold', zorder=5)
        txt(ax, x, 3.12, name, fs=8.5, color='#555555')
    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow(ax, (x1 + 0.32, 3.82), (x2 - 0.32, 3.82), color='#2e75b6', lw=1.4, ms=10)
        arrow(ax, (x2 - 0.32, 3.68), (x1 + 0.32, 3.68), color='#c0392b', lw=1.4, ms=10)
    txt(ax, 5.7, 4.2, 'head（最久未使用）', fs=8.5, color='#922b21')
    txt(ax, 8.9, 4.2, 'tail（最近使用）', fs=8.5, color='#2e75b6')
    # head eviction annotation (text + arrow separate)
    txt(ax, 1.5, 1.6, '容量满 → 淘汰 head', fs=9.5, color='#c0392b')
    arrow(ax, (2.7, 1.8), (5.3, 3.35), color='#c0392b', lw=1.6)
    # tail annotation
    txt(ax, 9.3, 2.55, 'get/put 命中 → 移到 tail', fs=9.5, color='#2e75b6')
    arrow(ax, (9.3, 3.3), (8.62, 3.75), color='#2e75b6', lw=1.6)
    txt(ax, 5.5, 0.7, '容量满 → popitem(last=False) 弹队首；move_to_end 实现“访问即置顶”', fs=9.5,
        color='#555555')
    return fig

ALL = {
    'python_gil': fig_gil, 'os_process_states': fig_process_states,
    'os_paging': fig_paging, 'net_tcp_handshake': fig_tcp_handshake,
    'net_tcp_fourwave': fig_tcp_fourwave, 'net_congestion': fig_congestion,
    'net_http2': fig_http2, 'db_bplus': fig_bplus, 'db_mvcc': fig_mvcc,
    'redis_cache': fig_redis_cache, 'redis_bloom': fig_bloom,
    'dist_cap': fig_cap, 'dist_token_bucket': fig_token_bucket,
    'dist_consistent_hash': fig_chash, 'git_branch': fig_git, 'hand_lru': fig_lru,
}

def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(ALL)
    bad = []
    for name in targets:
        if name not in ALL:
            print(f'!! unknown figure {name}')
            continue
        fig = ALL[name]()
        path = os.path.join(OUT, name + '.png')
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        issues = report(fig, name)
        if issues:
            bad.append(name)
        plt.close(fig)
        print(f'    saved {path}')
    print()
    if bad:
        print('FIGURES WITH OVERLAPS:', ', '.join(bad))
        sys.exit(1)
    print('ALL OK - no overlaps detected')

if __name__ == '__main__':
    main()