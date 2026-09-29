# -*- coding: utf-8 -*-
"""Apply overlap fixes to 6 LLM notebook cells (source only; outputs cleared).
Runs: python tools/patch_llm_cells.py
"""
import json, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = '04-LLM大模型/教学/'

PATCHES = [
    # (path, cell_idx, old, new)
    (BASE + '02-多头注意力.ipynb', 11,
     "ax.text(1.5, -1.2, 'W_q 分块 0（head 0 的输入子空间）', ha='center', fontsize=10, color='darkred')",
     "ax.text(1.5, -1.2, 'W_q 分块 0（head 0）', ha='center', fontsize=10, color='darkred')"),

    (BASE + '04-前馈网络与激活函数.ipynb', 15,
     "fig, ax = plt.subplots(figsize=(9, 3.4))",
     "fig, ax = plt.subplots(figsize=(11, 3.4))"),

    (BASE + '04-前馈网络与激活函数.ipynb', 15,
     """ax.annotate('', xy=(5.4, 1.2), xytext=(5.2, 1.2), arrowprops=dict(arrowstyle='->', lw=2))
ax.add_patch(plt.Rectangle((6.2, 0.5), 2.2, 1.4, facecolor='#C44E52', alpha=0.8))
ax.text(7.3, 1.2, 'down(·)\\n→ d_model', ha='center', va='center', fontsize=10, color='white')
ax.set_xlim(-0.5, 9); ax.set_ylim(0, 2.2); ax.axis('off')""",
     """ax.annotate('', xy=(5.4, 1.2), xytext=(5.2, 1.2), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(9.0, 1.2), xytext=(8.6, 1.2), arrowprops=dict(arrowstyle='->', lw=2))
ax.add_patch(plt.Rectangle((9.0, 0.5), 2.2, 1.4, facecolor='#C44E52', alpha=0.8))
ax.text(10.1, 1.2, 'down(·)\\n→ d_model', ha='center', va='center', fontsize=10, color='white')
ax.set_xlim(-0.5, 12); ax.set_ylim(0, 2.2); ax.axis('off')"""),

    (BASE + '06-编码器与解码器.ipynb', 13,
     "box(ax, 6.2, 2.4, 3.2, 0.8, 'enc_out  [B,T,8]（来自 Encoder）', fc='#E6F4EA', ec='#38761D')\narrow(ax, 7.8, 3.2, 4.4, 5.2)   # K/V 来源",
     "box(ax, 0.3, 2.2, 3.2, 0.8, 'enc_out  [B,T,8]（来自 Encoder）', fc='#E6F4EA', ec='#38761D')\narrow(ax, 1.9, 3.0, 2.7, 5.0)   # K/V 来源"),

    (BASE + '08-注意力变种MQA与GQA.ipynb', 15,
     """for ax, (title, kv_of_q, tag) in zip(axes, schemes):
    for qi in range(8):
        ax.add_patch(plt.Rectangle((qi * 0.9, 2.2), 0.8, 0.9, fc='#4C72B0', ec='k'))
        ax.text(qi * 0.9 + 0.4, 2.65, f'Q{qi}', ha='center', va='center', fontsize=8, color='white')
        ki = kv_of_q[qi]
        ax.add_patch(plt.Rectangle((ki * 0.9, 0.6), 0.8, 0.9, fc='#55A868', ec='k'))
        ax.text(ki * 0.9 + 0.4, 1.05, f'KV{ki}', ha='center', va='center', fontsize=8, color='white')
        ax.annotate('', xy=(qi * 0.9 + 0.4, 1.5), xytext=(qi * 0.9 + 0.4, 2.2),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.0))""",
     """for ax, (title, kv_of_q, tag) in zip(axes, schemes):
    for qi in range(8):
        ax.add_patch(plt.Rectangle((qi * 0.9, 2.2), 0.8, 0.9, fc='#4C72B0', ec='k'))
        ax.text(qi * 0.9 + 0.4, 2.65, f'Q{qi}', ha='center', va='center', fontsize=8, color='white')
    for ki in sorted(set(kv_of_q)):
        ax.add_patch(plt.Rectangle((ki * 0.9, 0.6), 0.8, 0.9, fc='#55A868', ec='k'))
        ax.text(ki * 0.9 + 0.4, 1.05, f'KV{ki}', ha='center', va='center', fontsize=8, color='white')
    for qi in range(8):
        ki = kv_of_q[qi]
        ax.annotate('', xy=(ki * 0.9 + 0.4, 1.5), xytext=(qi * 0.9 + 0.4, 2.2),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=1.0))"""),

    (BASE + '15-RAG与Agent.ipynb', 3,
     """for i, d in enumerate(docs):
    ax.annotate(f'D{i}', xy=xy[i], fontsize=9, color='#4C72B0')""",
     """offs = [(0.06, 0.05), (-0.06, 0.05), (-0.06, -0.05), (0.06, 0.05), (0.06, -0.05)]
for i, d in enumerate(docs):
    ax.annotate(f'D{i}', xy=xy[i], xytext=(xy[i][0] + offs[i][0], xy[i][1] + offs[i][1]),
                fontsize=9, color='#4C72B0')"""),

    (BASE + '22-深度量化GPTQ与AWQ.ipynb', 13,
     """axes[1].set_title('显存：INT4 ≈ 1/13 of fp64', fontsize=12)
for b, m in zip(bars, mem_vals):
    axes[1].text(b.get_x() + b.get_width() / 2, m * 1.02, f'{m}', ha='center', fontsize=9)""",
     """axes[1].set_title('显存：INT4 ≈ 1/13 of fp64', fontsize=12)
axes[1].set_ylim(0, max(mem_vals) * 1.25)
for b, m in zip(bars, mem_vals):
    axes[1].text(b.get_x() + b.get_width() / 2, m * 1.06, f'{m}', ha='center', va='bottom', fontsize=9)"""),
]

ok = True
for path, idx, old, new in PATCHES:
    data = json.load(open(path, encoding='utf-8'))
    src = ''.join(data['cells'][idx].get('source', []))
    if old not in src:
        print(f'OLD NOT FOUND in {path} cell {idx}:')
        print('  ', old[:80].replace('\n', '\\n'))
        ok = False
        continue
    new_src = src.replace(old, new, 1)
    data['cells'][idx]['source'] = [new_src]
    data['cells'][idx]['outputs'] = []
    data['cells'][idx]['execution_count'] = None
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f'patched {path} cell {idx}')

print('ALL OK' if ok else 'SOME FAILED')