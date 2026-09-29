# -*- coding: utf-8 -*-
"""只做图重叠检查（不执行 kernel），用于快速迭代新图排版。

用法: python tools/check_overlap_only.py [notebook文件名...]  （默认检查 08-优化算法/教学/ 全部）
"""
import os
import sys
import io
import glob
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACH = os.path.join(ROOT, '08-优化算法', '教学')
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from overlap_check import check_figure

targets = sys.argv[1:] or sorted(glob.glob(os.path.join(TEACH, '*.ipynb')))
any_flag = False
for path in targets:
    fname = os.path.basename(path)
    if not fname.endswith('.ipynb'):
        fname += '.ipynb'
    nb = json.load(open(os.path.join(TEACH, fname), encoding='utf-8'))
    ns = {'__name__': '__main__'}
    exec('import matplotlib; matplotlib.use("Agg")', ns)
    exec('import matplotlib.pyplot as plt', ns)
    exec('import numpy as np', ns)
    plt_ = ns['plt']
    flags = 0
    for i, c in enumerate(nb['cells']):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', ''))
        code = '\n'.join(l for l in src.splitlines() if not l.strip().startswith(('%', '%%')))
        if not code.strip():
            continue
        before = set(plt_.get_fignums())
        try:
            exec(compile(code, f'{fname}:c{i}', 'exec'), ns)
        except Exception as e:
            print(f'  EXEC FAIL {fname} cell{i}: {type(e).__name__}: {str(e)[:120]}')
            any_flag = True
            continue
        for fn in sorted(set(plt_.get_fignums()) - before):
            fig = plt_.figure(fn)
            issues = check_figure(fig)
            if issues:
                flags += len(issues)
                for it in issues[:8]:
                    print(f'  [{fname} cell{i} fig{fn}] {it[0]}: {str(it[1])[:45]} <-> {str(it[2])[:45]}')
    try:
        plt_.close('all')
    except Exception:
        pass
    print(('FLAG ' if flags else 'OK   ') + fname + (f' ({flags} issues)' if flags else ''))
    if flags:
        any_flag = True
print('RESULT:', 'ALL PASS' if not any_flag else 'FLAGGED')
sys.exit(1 if any_flag else 0)