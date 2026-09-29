# -*- coding: utf-8 -*-
"""只检查指定 notebook：nbclient 执行 + 重叠检查 + cell 失败统计。"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACH = os.path.join(ROOT, '08-优化算法', '教学')
os.chdir(TEACH)

import nbformat
from nbclient import NotebookClient
sys.path.insert(0, os.path.join(ROOT, 'tools'))
from overlap_check import check_figure
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

targets = sys.argv[1:] or ['*.ipynb']
import glob
nbs = []
for t in targets:
    nbs += sorted(glob.glob(t))
nbs = sorted(set(nbs))

any_fail = False
for nb in nbs:
    nbk = nbformat.read(nb, as_version=4)
    client = NotebookClient(nbk, timeout=600, kernel_name='python3',
                            resources={'metadata': {'path': '.'}})
    try:
        client.execute()
        nbformat.write(nbk, nb, version=4)
    except Exception as e:
        print(f'## {nb} :: EXEC FAILED: {type(e).__name__}: {str(e)[:200]}')
        any_fail = True
        continue

    ns = {'__name__': '__main__'}
    exec('import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np', ns)
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
            print(f'  [cell{i}] exec fail: {type(e).__name__}: {str(e)[:120]}')
            any_fail = True
            continue
        for fn in sorted(set(plt.get_fignums()) - before):
            issues = check_figure(plt.figure(fn))
            if issues:
                nb_flags.append((i, fn, len(issues)))
                for it in issues[:6]:
                    print(f'  [{nb} cell{i} fig{fn}] {it[0]}: {str(it[1])[:40]} <-> {str(it[2])[:40]}')
    if nb_flags:
        any_fail = True
        print(f'## {nb} :: OVERLAPS: {len(nb_flags)}')
    else:
        print(f'## {nb} :: OK - no overlaps')
    plt.close('all')

print('RESULT:', 'ALL PASS' if not any_fail else 'FLAGGED')
sys.exit(1 if any_fail else 0)