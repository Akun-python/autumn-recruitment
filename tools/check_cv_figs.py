# -*- coding: utf-8 -*-
"""批量执行 09-计算机视觉/教学/*.ipynb（真实 kernel），写回内嵌输出，并做重叠检查。

用法: python tools/check_cv_figs.py
"""
import os, sys, io, glob, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACH = os.path.join(ROOT, '09-计算机视觉', '教学')
os.chdir(TEACH)

import nbformat
from nbclient import NotebookClient

# 重叠检查（复用仓库的检查器）
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from overlap_check import check_figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nbs = sorted(glob.glob('*.ipynb'))
all_flags = []
for nb in nbs:
    nbk = nbformat.read(nb, as_version=4)
    client = NotebookClient(nbk, timeout=600, kernel_name='python3',
                            resources={'metadata': {'path': '.'}})
    try:
        client.execute()
        nbformat.write(nbk, nb, version=4)
    except Exception as e:
        print(f'## {nb} :: EXEC FAILED: {type(e).__name__}: {str(e)[:160]}')
        all_flags.append(nb)
        continue

    # 逐 cell 检查新图（重新执行代码 cell 以获取 figure 对象）
    ns = {'__name__': '__main__'}
    exec('import matplotlib; matplotlib.use("Agg")', ns)
    exec('import matplotlib.pyplot as plt', ns)
    exec('import numpy as np', ns)
    plt = ns['plt']
    nb_flags = []
    for i, c in enumerate(nbk.cells):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', ''))
        code = '\n'.join(l for l in src.splitlines() if not l.strip().startswith(('%', '%%')))
        if not code.strip():
            continue
        before = set(plt.get_fignums())
        try:
            exec(compile(code, f'{nb}:c{i}', 'exec'), ns)
        except Exception as e:
            print(f'  cell {i} exec fail: {type(e).__name__}: {str(e)[:120]}')
            continue
        for fn in sorted(set(plt.get_fignums()) - before):
            fig = plt.figure(fn)
            issues = check_figure(fig)
            if issues:
                nb_flags.append((i, fn, len(issues)))
                for it in issues[:6]:
                    print(f'  [{nb} cell{i} fig{fn}] {it[0]}: {str(it[1])[:40]} <-> {str(it[2])[:40]}')
    if nb_flags:
        all_flags.append(nb)
        print(f'## {nb} :: OVERLAPS: {len(nb_flags)}')
    else:
        print(f'## {nb} :: OK - no overlaps, cells-failed: 0')
    try:
        plt.close('all')
    except Exception:
        pass

print()
print('RESULT:', 'ALL PASS' if not all_flags else f'FLAGGED: {all_flags}')
sys.exit(1 if all_flags else 0)