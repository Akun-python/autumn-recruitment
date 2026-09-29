# -*- coding: utf-8 -*-
"""Execute each 04-LLM大模型/教学 notebook cell-by-cell, snapshot the NEW
figures each cell creates, run the overlap checker, then close everything so
each notebook is isolated. Results go to tools/llm_stdout.txt.
Run: python tools/check_llm_figs.py
"""
import json, glob, os, sys, io

TOOLS = os.path.abspath(os.path.dirname(__file__))
TEACH = os.path.abspath(os.path.join(TOOLS, os.pardir, '04-LLM大模型', '教学'))

def sanitize(src):
    out = []
    for ln in src.splitlines():
        s = ln.strip()
        if s.startswith('%') or s.startswith('%%'):
            continue
        out.append(ln)
    return '\n'.join(out)

def main():
    nbs = sorted(glob.glob(os.path.join(TEACH, '*.ipynb')))
    for nb in nbs:
        data = json.load(open(nb, encoding='utf-8'))
        ns = {'__name__': '__main__'}
        exec('import matplotlib; matplotlib.use("Agg")', ns)
        exec('import matplotlib.pyplot as plt', ns)
        plt = ns['plt']
        exec('import numpy as np', ns)
        baseline = set(plt.get_fignums())
        tag = os.path.basename(nb)
        cell_fails = []
        for i, c in enumerate(data['cells']):
            if c.get('cell_type') != 'code':
                continue
            src = ''.join(c.get('source', ''))
            code = sanitize(src)
            if not code.strip():
                continue
            before = set(plt.get_fignums())
            try:
                exec(compile(code, f'{tag}:cell{i}', 'exec'), ns)
            except Exception as e:
                cell_fails.append((i, f'{type(e).__name__}: {e}'))
                continue
            after = set(plt.get_fignums())
            newnums = sorted(after - before)
            for fn in newnums:
                fig = plt.figure(fn)
                try:
                    from overlap_check import report
                    report(fig, f'{tag}:cell{i}')
                except Exception as e:
                    print(f'[{tag}:cell{i}] CHECK FAIL {type(e).__name__}: {e}')
        # close figures created in this notebook only
        try:
            for fn in set(plt.get_fignums()):
                if fn not in baseline:
                    plt.close(fn)
        except Exception:
            pass
        fails = ', '.join(f'c{i}({e})' for i, e in cell_fails[:6])
        print(f'## {tag} :: cells-failed: {len(cell_fails)} {fails}')
        sys.stdout.flush()

if __name__ == '__main__':
    main()